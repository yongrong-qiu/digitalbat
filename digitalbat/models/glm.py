import numpy as np
import torch
from torch import nn


class Bat_GLM(nn.Module):
    """
    Session-wise GLM for neural encoding.
    Linear model: y = activation(W @ x + b)

    Equivalent to GLM_PyTorch but with session-wise heads.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.

    output_dim_dict : dict
        Dictionary mapping session names to number of neurons.
        e.g., {'Dataset_3_14543_240419': 289, 'Dataset_1_32622_231009': 95}

    activation : str
        Activation function: 'exp', 'softplus', 'sigmoid', or 'linear'.
        Use 'exp' for Poisson regression with PoissonNLLLoss(log_input=False).
    """
    def __init__(
        self,
        input_dim: int,
        output_dim_dict: dict,
        activation: str = 'exp',
    ):
        super().__init__()

        self.readouts = nn.ModuleDict({
            session_name: nn.Linear(input_dim, unit_num)
            for session_name, unit_num in output_dim_dict.items()
        })

        if activation == 'exp':
            self.final_activation = torch.exp
        elif activation == 'softplus':
            self.final_activation = nn.functional.softplus
        elif activation == 'sigmoid':
            self.final_activation = torch.sigmoid
        elif activation == 'linear':
            self.final_activation = lambda x: x
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, session_name: str, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, input_dim) or (batch_size, T, input_dim)

        Returns
        -------
        torch.Tensor
            Shape (batch_size, num_neurons) or (batch_size, T, num_neurons)
        """
        return self.final_activation(self.readouts[session_name](x))


def create_cosine_bumps_torch(x, centers, width):
    """Raised cosine bumps in PyTorch.

    x: (...,) tensor of positions
    centers: (K,) tensor of bump centers
    width: scalar bump width

    returns: (..., K) tensor
    """
    diff = x.unsqueeze(-1) - centers
    return (torch.cos(2 * np.pi * diff / width) * 0.5 + 0.5) * (diff.abs() < width / 2).float()


def create_circular_cosine_bumps_torch(theta, centers, width):
    """Raised cosine bumps for circular (periodic) variables.

    theta: (...,) tensor of angles in radians
    centers: (K,) tensor of bump centers
    width: scalar bump width

    returns: (..., K) tensor
    """
    diff = theta.unsqueeze(-1) - centers
    # Wrap difference to [-pi, pi] for circular periodicity
    diff = diff - 2 * np.pi * torch.round(diff / (2 * np.pi))
    return (torch.cos(2 * np.pi * diff / width) * 0.5 + 0.5) * (diff.abs() < width / 2).float()


def cosine_basis_params(n_bases_per_dim=8, pos_range=(-3.0, 3.0), width_to_spacing_ratio=4):
    """Return ``(centers (K,), width)`` for the 3D tensor-product cosine basis used by
    :class:`GLM_3DCosineBasis`.  Exposed so the identical basis can be built outside the
    model (e.g. as a fixed place representation) without instantiating the GLM/readout."""
    centers = torch.linspace(pos_range[0], pos_range[1], n_bases_per_dim)
    spacing = (pos_range[1] - pos_range[0]) / (n_bases_per_dim - 1)
    return centers, width_to_spacing_ratio * spacing


def spatial_cosine_basis(xyz, centers, width):
    """3D tensor-product raised-cosine basis from ``(..., >=3)`` z-scored positions ->
    ``(..., len(centers) ** 3)``.  Shared by :meth:`GLM_3DCosineBasis._spatial_basis` and
    usable standalone (pair with :func:`cosine_basis_params`)."""
    centers = centers.to(xyz.device)
    b_x = create_cosine_bumps_torch(xyz[..., 0], centers, width)
    b_y = create_cosine_bumps_torch(xyz[..., 1], centers, width)
    b_z = create_cosine_bumps_torch(xyz[..., 2], centers, width)
    bases = (
        b_x.unsqueeze(-1).unsqueeze(-1)
        * b_y.unsqueeze(-2).unsqueeze(-1)
        * b_z.unsqueeze(-2).unsqueeze(-2)
    )
    return bases.flatten(-3)


def peraxis_cosine_basis_params(n_per_axis, pos_range=(-3.0, 3.0),
                                width_to_spacing_ratio=4):
    """Per-axis analogue of :func:`cosine_basis_params`.

    ``n_per_axis`` is an int (symmetric ``n`` per axis, identical to
    :func:`cosine_basis_params`) or a 3-tuple ``(nx, ny, nz)``.  Returns
    ``(centers_list, widths)`` where ``centers_list`` is a list of 3 1-D tensors
    (one per axis) and ``widths`` a list of 3 floats.  Used for the anisotropic
    place basis (e.g. fewer z bases when the room's z-extent is shorter)."""
    if isinstance(n_per_axis, (int, np.integer)):
        n_per_axis = (int(n_per_axis),) * 3
    centers, widths = [], []
    for n in n_per_axis:
        c = torch.linspace(pos_range[0], pos_range[1], int(n))
        spacing = (pos_range[1] - pos_range[0]) / (int(n) - 1)
        centers.append(c)
        widths.append(width_to_spacing_ratio * spacing)
    return centers, widths


def peraxis_spatial_cosine_basis(xyz, centers_list, widths):
    """3D tensor-product raised-cosine basis with a SEPARATE centre grid + width per
    axis (anisotropic).  ``centers_list`` = 3 tensors, ``widths`` = 3 floats.  Maps
    ``(..., >=3)`` z-scored positions -> ``(..., nx*ny*nz)``.  Reduces to
    :func:`spatial_cosine_basis` when all three axes share centres/width."""
    b = [create_cosine_bumps_torch(xyz[..., d], centers_list[d].to(xyz.device), widths[d])
         for d in range(3)]
    bases = (
        b[0].unsqueeze(-1).unsqueeze(-1)
        * b[1].unsqueeze(-2).unsqueeze(-1)
        * b[2].unsqueeze(-2).unsqueeze(-2)
    )
    return bases.flatten(-3)


def _install_place_basis(module, n_bases_per_dim, n_bases_per_axis, pos_range,
                         width_to_spacing_ratio):
    """Register the (possibly anisotropic) 3D place-basis buffers on ``module`` and
    return ``n_spatial``.

    Symmetric (``n_bases_per_dim`` per axis, registering ``module.centers`` /
    ``module.width`` exactly as before) UNLESS ``n_bases_per_axis`` (int or 3-tuple
    ``(nx, ny, nz)``) is given, in which case per-axis centre grids are registered
    as ``module.centers_0/1/2`` with ``module.place_widths``.  Pairs with
    :func:`_eval_place_basis`."""
    if n_bases_per_axis is not None:
        centers_list, widths = peraxis_cosine_basis_params(
            n_bases_per_axis, pos_range, width_to_spacing_ratio)
        for d, c in enumerate(centers_list):
            module.register_buffer(f'centers_{d}', c)
        module.place_widths = [float(w) for w in widths]
        module._peraxis = True
        return int(np.prod([len(c) for c in centers_list]))
    centers = torch.linspace(pos_range[0], pos_range[1], n_bases_per_dim)
    spacing = (pos_range[1] - pos_range[0]) / (n_bases_per_dim - 1)
    module.register_buffer('centers', centers)
    module.width = width_to_spacing_ratio * spacing
    module._peraxis = False
    return n_bases_per_dim ** 3


def _eval_place_basis(module, xyz):
    """Evaluate the 3D place basis installed by :func:`_install_place_basis`."""
    if getattr(module, '_peraxis', False):
        cl = [getattr(module, f'centers_{d}') for d in range(3)]
        return peraxis_spatial_cosine_basis(xyz, cl, module.place_widths)
    return spatial_cosine_basis(xyz, module.centers, module.width)


class GLM_3DCosineBasis(nn.Module):
    """3D tensor-product cosine basis expansion + session-wise linear GLM.

    Uses 3D raised cosine bumps over (x, y, z) z-scored position.
    When `lag_seconds` contains multiple values, the design matrix at each
    time step is the concatenation of spatial bases evaluated at the
    corresponding time-lagged positions (zero-padded at sequence edges).

    Parameters
    ----------
    lag_seconds : list of float
        Time offsets (in seconds) at which to evaluate the spatial basis.
        Default ``[0.0]`` gives the position-only GLM.
    bin_width : float
        Temporal bin width in seconds, used to convert lag_seconds to bins.
    """
    def __init__(
        self,
        output_dim_dict,
        n_bases_per_dim=8,
        pos_range=(-3.0, 3.0),
        width_to_spacing_ratio=4,
        activation='exp',
        lag_seconds=None,
        bin_width=0.1,
        n_bases_per_axis=None,
    ):
        super().__init__()
        if lag_seconds is None:
            lag_seconds = [0.0]
        self.lag_bins = [int(round(s / bin_width)) for s in lag_seconds]
        n_lags = len(self.lag_bins)

        # Symmetric n_bases_per_dim per axis unless n_bases_per_axis=(nx,ny,nz) is
        # given (anisotropic place basis, e.g. fewer z bases for a shorter room).
        n_spatial = _install_place_basis(self, n_bases_per_dim, n_bases_per_axis,
                                         pos_range, width_to_spacing_ratio)
        self.n_bases_per_dim = n_bases_per_dim

        self.glm = Bat_GLM(
            input_dim=n_spatial * n_lags,
            output_dim_dict=output_dim_dict,
            activation=activation,
        )
        if n_lags > 1:
            print(f"GLM: {n_lags} lags x {n_spatial} spatial = "
                  f"{n_spatial * n_lags} features, "
                  f"lag_bins={self.lag_bins}")

    def _spatial_basis(self, xyz):
        """Compute 3D tensor-product cosine basis from (N, T, >=3) positions."""
        return _eval_place_basis(self, xyz)

    def _make_design(self, x, mask=None):
        # x: (N, T, behavior_dim) — first 3 dims = x, y, z (z-scored by dataloader)
        N, T, _ = x.shape
        spatial = self._spatial_basis(x)  # (N, T, K^3)
        K3 = spatial.shape[-1]

        # Zero out spatial basis at invalid (padded) timesteps BEFORE shifting,
        # so that lags reaching into the padded zone get zeros rather than
        # basis values computed from garbage padding data.
        if mask is not None:
            spatial = spatial * mask.unsqueeze(-1)

        lagged = []
        for lag in self.lag_bins:
            if lag == 0:
                lagged.append(spatial)
            elif lag > 0:
                pad = spatial.new_zeros(N, lag, K3)
                lagged.append(torch.cat([spatial[:, lag:, :], pad], dim=1))
            else:
                pad = spatial.new_zeros(N, -lag, K3)
                lagged.append(torch.cat([pad, spatial[:, :T + lag, :]], dim=1))

        return torch.cat(lagged, dim=-1)  # (N, T, K^3 * n_lags)

    def forward(self, session_name, x, mask=None, **kwargs):
        design = self._make_design(x, mask)
        return self.glm(session_name, design)


class GLM_3DCosineBasis_Kinematics(nn.Module):
    """GLM with 3D spatial cosine basis + circular velocity-direction bumps + speed + IMU.

    Design matrix per time lag:
    - 3D raised cosine spatial basis from position (K^3 features)
    - Circular cosine bumps for velocity direction angles (n_vel_bumps * 3 features)
    - Speed (1 feature)
    - IMU acceleration + gyroscope (6 features)

    Velocity direction is converted to three rotation angles (one per axis)
    using atan2, then expanded with periodic cosine bumps.

    Precondition: the behaviour tensor must use the fixed column layout
    ``[pos(0:3), speed(3), vel_unit(4:7), IMU(7:13)]`` (behaviour_dim >= 13). This
    requires the dataloader to be built with ``use_speed_head_dir=True`` and
    ``use_IMU_flag=True`` (both default False); ``_make_design`` asserts on it.
    """
    def __init__(
        self,
        output_dim_dict,
        behavior_stats,
        n_bases_per_dim=8,
        pos_range=(-3.0, 3.0),
        width_to_spacing_ratio=4,
        activation='exp',
        lag_seconds=None,
        bin_width=0.1,
        n_vel_bumps=8,
        vel_width_ratio=4,
    ):
        super().__init__()
        if lag_seconds is None:
            lag_seconds = [0.0]
        self.lag_bins = [int(round(s / bin_width)) for s in lag_seconds]
        n_lags = len(self.lag_bins)

        # Position spatial basis
        centers = torch.linspace(pos_range[0], pos_range[1], n_bases_per_dim)
        spacing = (pos_range[1] - pos_range[0]) / (n_bases_per_dim - 1)
        self.register_buffer('centers', centers)
        self.width = width_to_spacing_ratio * spacing
        self.n_bases_per_dim = n_bases_per_dim

        # Circular velocity bumps: evenly spaced around [-pi, pi)
        self.n_vel_bumps = n_vel_bumps
        vel_centers = torch.linspace(-np.pi, np.pi, n_vel_bumps + 1)[:-1]
        vel_spacing = 2 * np.pi / n_vel_bumps
        self.register_buffer('vel_centers', vel_centers)
        self.vel_width = vel_width_ratio * vel_spacing

        # Store behavior z-score stats per session for un-z-scoring vel_unit
        self._session_to_idx = {}
        for i, (sname, (mean, std)) in enumerate(behavior_stats.items()):
            self._session_to_idx[sname] = i
            self.register_buffer(f'beh_mean_{i}',
                                 torch.tensor(mean.squeeze(), dtype=torch.float32))
            self.register_buffer(f'beh_std_{i}',
                                 torch.tensor(std.squeeze(), dtype=torch.float32))

        n_spatial = n_bases_per_dim ** 3
        n_vel = n_vel_bumps * 3
        n_extra = 1 + 6  # speed + IMU (3 acc + 3 gyr)
        n_features_per_lag = n_spatial + n_vel + n_extra

        self.glm = Bat_GLM(
            input_dim=n_features_per_lag * n_lags,
            output_dim_dict=output_dim_dict,
            activation=activation,
        )
        print(f'GLM Kinematics: {n_lags} lags x '
              f'({n_spatial} spatial + {n_vel} vel + {n_extra} speed+IMU) = '
              f'{n_features_per_lag * n_lags} features')

    def _spatial_basis(self, xyz):
        centers = self.centers.to(xyz.device)
        b_x = create_cosine_bumps_torch(xyz[..., 0], centers, self.width)
        b_y = create_cosine_bumps_torch(xyz[..., 1], centers, self.width)
        b_z = create_cosine_bumps_torch(xyz[..., 2], centers, self.width)
        bases = (
            b_x.unsqueeze(-1).unsqueeze(-1)
            * b_y.unsqueeze(-2).unsqueeze(-1)
            * b_z.unsqueeze(-2).unsqueeze(-2)
        )
        return bases.flatten(-3)

    def _velocity_bumps(self, x, session_name):
        """Convert z-scored vel_unit back to raw, compute 3 rotation angles, expand with circular bumps."""
        idx = self._session_to_idx[session_name]
        mean = getattr(self, f'beh_mean_{idx}')
        std = getattr(self, f'beh_std_{idx}')
        # Un-z-score vel_unit (dims 4-6)
        vel_raw = x[..., 4:7] * std[4:7] + mean[4:7]
        # Rotation angles around each axis
        angle_z = torch.atan2(vel_raw[..., 1], vel_raw[..., 0])  # atan2(vy, vx)
        angle_x = torch.atan2(vel_raw[..., 2], vel_raw[..., 1])  # atan2(vz, vy)
        angle_y = torch.atan2(vel_raw[..., 0], vel_raw[..., 2])  # atan2(vx, vz)
        centers = self.vel_centers.to(x.device)
        bumps = torch.cat([
            create_circular_cosine_bumps_torch(angle_x, centers, self.vel_width),
            create_circular_cosine_bumps_torch(angle_y, centers, self.vel_width),
            create_circular_cosine_bumps_torch(angle_z, centers, self.vel_width),
        ], dim=-1)
        return bumps

    def _make_design(self, x, mask=None, session_name=None):
        N, T, D = x.shape
        # Fixed behaviour-column layout: 0:3 position, 3 speed, 4:7 velocity unit
        # vector, 7:13 IMU (3 accel + 3 gyro). This holds only when the loader is
        # built with use_speed_head_dir=True and use_IMU_flag=True (both default to
        # False in bat_data_loader, and no_3d_location shifts every column). Guard
        # against silent column misalignment.
        assert D >= 13, (
            f'GLM_3DCosineBasis_Kinematics expects behaviour_dim >= 13 with layout '
            f'[pos(0:3), speed(3), vel_unit(4:7), IMU(7:13)]; got behaviour_dim={D}. '
            f'Build the dataloader with use_speed_head_dir=True and use_IMU_flag=True.')
        spatial = self._spatial_basis(x)                  # (N, T, K^3)
        vel_bumps = self._velocity_bumps(x, session_name) # (N, T, n_vel_bumps*3)
        speed = x[..., 3:4]                               # (N, T, 1)
        imu = x[..., 7:13]                                # (N, T, 6)
        features = torch.cat([spatial, vel_bumps, speed, imu], dim=-1)

        if mask is not None:
            features = features * mask.unsqueeze(-1)

        F = features.shape[-1]
        lagged = []
        for lag in self.lag_bins:
            if lag == 0:
                lagged.append(features)
            elif lag > 0:
                pad = features.new_zeros(N, lag, F)
                lagged.append(torch.cat([features[:, lag:, :], pad], dim=1))
            else:
                pad = features.new_zeros(N, -lag, F)
                lagged.append(torch.cat([pad, features[:, :T + lag, :]], dim=1))

        return torch.cat(lagged, dim=-1)

    def forward(self, session_name, x, mask=None, **kwargs):
        design = self._make_design(x, mask, session_name=session_name)
        return self.glm(session_name, design)


def make_log_raised_cosine_basis(n_basis, n_lags, bin_width, c=None,
                                 include_zero_lag=False):
    """Log-spaced raised-cosine temporal basis (Pillow et al. 2008, Fig. S1).

    ``b_j(t) = 0.5*cos(clip((log(t+c) - center_j) * pi/(2*db), -pi, pi)) + 0.5``,
    with log-spaced centers and spacing ``db`` (half-support ``2*db`` matches the
    paper's ``[phi-pi, phi+pi]`` with ``pi/2`` spacing).

    Parameters
    ----------
    include_zero_lag : bool
        If False (default), sample at strictly-past bin lags ``1..n_lags`` — use
        for spike-history filters, which must be causal (a neuron's current bin
        can't predict itself). If True, also include lag 0 (the current bin),
        giving ``n_lags + 1`` rows (lags ``0..n_lags``) — use for coupling
        filters, whose simultaneous-bin term from *other* neurons is legitimate
        (the self-diagonal is masked out, so there is no leakage).

    Returns ``(B [n_rows, n_basis] float32 tensor, lag_seconds, log_centers)``,
    where ``n_rows`` is ``n_lags`` (history) or ``n_lags + 1`` (coupling).
    """
    if c is None:
        c = bin_width
    start = 0 if include_zero_lag else 1
    lags = (np.arange(start, n_lags + 1) * bin_width).astype(np.float64)
    nlin = lambda t: np.log(t + c)
    centers = np.linspace(nlin(lags[0]), nlin(lags[-1]), n_basis)
    db = centers[1] - centers[0] if n_basis > 1 else 1.0
    arg = (nlin(lags)[:, None] - centers[None, :]) * np.pi / (2 * db)
    B = (np.cos(np.clip(arg, -np.pi, np.pi)) + 1.0) / 2.0
    return torch.tensor(B, dtype=torch.float32), lags, centers


def make_symmetric_raised_cosine_basis(n_basis, n_lags, bin_width, c=None):
    """Acausal (forward + backward) log-raised-cosine temporal basis, lag 0 EXCLUDED.

    Builds the one-sided log-spaced raised-cosine basis over magnitudes
    ``1..n_lags`` bins (identical to :func:`make_log_raised_cosine_basis`), then
    mirrors it symmetrically about lag 0 into a block-structured matrix that spans
    both past and future lags while always skipping the current bin (lag 0). The
    two blocks (past, future) carry *separate* basis columns, so a filter built on
    this basis can weight the past and the future independently even though the
    bump shapes mirror each other.

    Masking lag 0 keeps the self-prediction problem non-trivial: a neuron's
    current bin is never used to predict itself.

    Returns ``(B [2*n_lags, 2*n_basis] float32 tensor, lag_seconds [2*n_lags],
    log_centers [n_basis])``. Rows are ordered ``[-n_lags..-1, +1..+n_lags]`` bins;
    columns ``0..n_basis-1`` are the past block, ``n_basis..2*n_basis-1`` the future.
    """
    one_sided, mags, centers = make_log_raised_cosine_basis(
        n_basis, n_lags, bin_width, c=c, include_zero_lag=False)  # (n_lags, n_basis)
    B1 = one_sided.numpy()
    N = n_lags
    B = np.zeros((2 * N, 2 * n_basis), dtype=np.float32)
    B[:N, :n_basis] = B1[::-1]    # past rows (lags -N..-1): magnitude N..1
    B[N:, n_basis:] = B1          # future rows (lags +1..+N): magnitude 1..N
    lag_seconds = np.concatenate([-mags[::-1], mags])
    return torch.tensor(B, dtype=torch.float32), lag_seconds, centers


def coupling_group_penalty(W):
    """Group-lasso over off-diagonal coupling filters: ``mean_{i!=j} ||l_{j,i,:}||_2``.

    Shrinks whole connections toward zero (Pillow's sparse coupling penalty, used
    to eliminate unnecessary coupling filters). ``W`` has shape ``(U, U, n_basis)``.
    """
    U = W.shape[0]
    off = ~torch.eye(U, dtype=torch.bool, device=W.device)
    return torch.sqrt((W ** 2).sum(-1) + 1e-12)[off].mean()


_OTHER_CLUSTER_KEY = 'other'


class GLM_Coupled(nn.Module):
    """Position GLM (3D cosine place basis) + optional Pillow-style temporal filters.

    Conditional intensity for neuron ``j`` (position-only stimulus, no lags/kinematics)::

        eta_j(t) = [W_spatial . phi(x_t) + mu_j]
                   + sum_b h_{j,b}  * (B_h * y_j)(t)             # spike history (own past)
                   + sum_{i!=j},b l_{j,i,b} * (B_l * y_i)(t)     # coupling (other neurons' past)
                   + sum_b a_{j,b}  * (B_a * y_j)(t)             # acausal self (own past+future)
        rate_j(t) = activation( eta_j(t) )

    Spike-history uses STRICTLY PAST bins (lags ``1..n_lags``) — a neuron's
    current bin never predicts itself. Coupling additionally includes lag 0 (the
    simultaneous bin) when ``coup_basis`` has ``n_lags + 1`` rows: using *other*
    neurons' current-bin spikes is legitimate because the diagonal (``i == j``)
    is masked out, so there is no self-leakage.

    The acausal self-filter (``acausal_basis``) is like the spike-history filter
    but looks BOTH backward and forward in time (lags ``-n_lags..-1`` and
    ``+1..+n_lags``), with lag 0 always excluded so the neuron's current bin is
    never used to predict itself. Past and future bumps carry independent weights
    (see :func:`make_symmetric_raised_cosine_basis`).

    Joint (position, trajectory) conditioning
    -----------------------------------------
    When ``cluster_keys`` is given, the place-field readout ``W_spatial`` is made
    **trajectory-cluster specific**: each session has one ``nn.Linear`` per flight
    cluster (plus an ``'other'`` catch-all). At forward time each flight is routed
    to its cluster's readout via ``cluster_idx``. Spike-history / coupling terms
    remain session-level (shared across clusters).

    Parameters
    ----------
    output_dim_dict : dict
        ``{session_name: n_neurons}``.
    hist_basis, coup_basis : torch.Tensor or None
        ``(n_lags, n_basis)`` temporal bases (see ``make_log_raised_cosine_basis``).
    acausal_basis : torch.Tensor or None
        ``(2*n_lags, n_basis_total)`` symmetric past+future self-filter basis with
        lag 0 excluded (see ``make_symmetric_raised_cosine_basis``).
    n_lags : int
        Causal window length in bins (must match the basis row count).
    cluster_keys : dict or None
        ``{session_name: [cluster_id, ...]}`` of major clusters per session. When
        provided, enables joint conditioning. An ``'other'`` readout is always added.
    """
    def __init__(self, output_dim_dict, n_bases_per_dim=16, pos_range=(-3.0, 3.0),
                 width_to_spacing_ratio=4, activation='exp', bin_width=0.1,
                 hist_basis=None, coup_basis=None, acausal_basis=None,
                 n_lags=None, cluster_keys=None, n_bases_per_axis=None,
                 lag_seconds=None):
        super().__init__()
        # Symmetric n_bases_per_dim per axis unless n_bases_per_axis=(nx,ny,nz) is
        # given (anisotropic place basis, e.g. fewer z bases for a shorter room).
        self.n_spatial = _install_place_basis(self, n_bases_per_dim, n_bases_per_axis,
                                             pos_range, width_to_spacing_ratio)
        self.n_bases_per_dim = n_bases_per_dim
        self.bin_width = bin_width
        # Optional SPATIAL time-lags: the design at each t concatenates the place
        # basis evaluated at time-lagged positions (zero-padded at edges). Default
        # [0.0] = position-only (unchanged). With cluster_keys this yields a
        # per-cluster "GLM + lags" (trajectory), cheap under a coarse place basis.
        if lag_seconds is None:
            lag_seconds = [0.0]
        self.spatial_lag_bins = [int(round(s / bin_width)) for s in lag_seconds]
        readout_in = self.n_spatial * len(self.spatial_lag_bins)

        if activation == 'exp':
            self.final_activation = torch.exp
        elif activation == 'softplus':
            self.final_activation = nn.functional.softplus
        elif activation == 'sigmoid':
            self.final_activation = torch.sigmoid
        elif activation == 'linear':
            self.final_activation = lambda x: x
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # ---- Spatial place-field readout: shared per session, or per (session, cluster)
        self.joint = cluster_keys is not None
        if not self.joint:
            self.readouts = nn.ModuleDict({
                s: nn.Linear(readout_in, u) for s, u in output_dim_dict.items()})
            self.cluster_keys = None
        else:
            self.cluster_keys = {}
            readout_dict = {}
            for s, u in output_dim_dict.items():
                keys = [str(k) for k in cluster_keys[s]]
                if _OTHER_CLUSTER_KEY not in keys:
                    keys.append(_OTHER_CLUSTER_KEY)
                self.cluster_keys[s] = keys
                readout_dict[s] = nn.ModuleDict({
                    k: nn.Linear(readout_in, u) for k in keys})
            self.readouts = nn.ModuleDict(readout_dict)

        # ---- Temporal filters
        self.use_hist = hist_basis is not None
        self.use_coup = coup_basis is not None
        self.n_lags = n_lags
        if self.use_hist:
            self.register_buffer('hist_basis', hist_basis)
            self.hist_weights = nn.ParameterDict({
                s: nn.Parameter(torch.zeros(u, hist_basis.shape[1]))
                for s, u in output_dim_dict.items()})
        if self.use_coup:
            self.register_buffer('coup_basis', coup_basis)
            self.coup_weights = nn.ParameterDict({
                s: nn.Parameter(torch.zeros(u, u, coup_basis.shape[1]))
                for s, u in output_dim_dict.items()})
        self.use_acausal = acausal_basis is not None
        if self.use_acausal:
            self.register_buffer('acausal_basis', acausal_basis)
            self.acausal_weights = nn.ParameterDict({
                s: nn.Parameter(torch.zeros(u, acausal_basis.shape[1]))
                for s, u in output_dim_dict.items()})

    def _spatial_basis(self, xyz):
        return _eval_place_basis(self, xyz)  # (N, T, K^3) or (N, T, nx*ny*nz)

    def _make_design(self, x, mask=None):
        spatial = self._spatial_basis(x)
        if mask is not None:
            spatial = spatial * mask.unsqueeze(-1)   # zero padded bins before shifting
        if len(self.spatial_lag_bins) == 1 and self.spatial_lag_bins[0] == 0:
            return spatial                            # position-only (no lags)
        N, T, K = spatial.shape
        lagged = []
        for lag in self.spatial_lag_bins:
            if lag == 0:
                lagged.append(spatial)
            elif lag > 0:
                pad = spatial.new_zeros(N, lag, K)
                lagged.append(torch.cat([spatial[:, lag:, :], pad], dim=1))
            else:
                pad = spatial.new_zeros(N, -lag, K)
                lagged.append(torch.cat([pad, spatial[:, :T + lag, :]], dim=1))
        return torch.cat(lagged, dim=-1)              # (N, T, K * n_lags)

    def _lagged_stack(self, spike, n_lags_total):
        # spike (N,T,U) -> S (N,T,U,n_lags_total) with S[...,l] = spike[t-l]
        # (lag 0 = current bin; zeros for t<l).
        N, T, U = spike.shape
        out = spike.new_zeros(N, T, U, n_lags_total)
        out[..., 0] = spike
        for l in range(1, min(n_lags_total, T)):
            out[:, l:, :, l] = spike[:, :T - l, :]
        return out

    def _acausal_stack(self, spike):
        # spike (N,T,U) -> A (N,T,U,2*n_lags), rows ordered [-n_lags..-1, +1..+n_lags].
        # Past block: A[...,k] = spike[t-(n_lags-k)]; future block: A[...,n_lags+k] =
        # spike[t+(k+1)]. Lag 0 is never included. Out-of-range bins stay zero.
        N, T, U = spike.shape
        L = self.n_lags
        A = spike.new_zeros(N, T, U, 2 * L)
        for k in range(L):
            m = L - k                       # past shift in {L..1}
            if m < T:
                A[:, m:, :, k] = spike[:, :T - m, :]
        for k in range(L):
            m = k + 1                       # future shift in {1..L}
            if m < T:
                A[:, :T - m, :, L + k] = spike[:, m:, :]
        return A

    def _spatial_drive(self, session_name, design, cluster_idx):
        if not self.joint:
            return self.readouts[session_name](design)
        ro = self.readouts[session_name]
        keys = self.cluster_keys[session_name]
        N, T, _ = design.shape
        units = ro[keys[0]].out_features
        eta = design.new_zeros(N, T, units)
        if cluster_idx is None:
            cluster_idx = torch.full((N,), -1, dtype=torch.long, device=design.device)
        groups = {}
        for n, c in enumerate(cluster_idx.tolist()):
            k = str(int(c)) if str(int(c)) in ro else _OTHER_CLUSTER_KEY
            groups.setdefault(k, []).append(n)
        for k, idx in groups.items():
            sel = torch.as_tensor(idx, device=design.device)
            eta[sel] = ro[k](design[sel])
        return eta

    def _apply_temporal(self, eta, session_name, spike, mask):
        """Add the Pillow-style spike-history / coupling / acausal-self filter
        contributions to the (spatial) linear predictor ``eta``. No-op when none
        of the temporal filters are enabled. Shared by :class:`GLM_Coupled` and
        :class:`GLM_Flight` (which differ only in the spatial drive)."""
        if not (self.use_hist or self.use_coup or self.use_acausal):
            return eta
        assert spike is not None, 'history/coupling/acausal GLM requires the spike train'
        sp = spike.float()
        if mask is not None:
            sp = sp * mask.unsqueeze(-1)  # no leakage from padded bins
        if self.use_hist or self.use_coup:
            L_h = self.hist_basis.shape[0] if self.use_hist else 0
            L_c = self.coup_basis.shape[0] if self.use_coup else 0
            S = self._lagged_stack(sp, max(L_h + 1, L_c))  # (N,T,U,*); S[...,0]=current
            if self.use_hist:
                # strictly past: lags 1..L_h
                hf = torch.einsum('ntul,lb->ntub', S[..., 1:1 + L_h], self.hist_basis)
                eta = eta + (hf * self.hist_weights[session_name]).sum(-1)
            if self.use_coup:
                # lags 0..L_c-1 (includes the simultaneous bin; diagonal masked below)
                cf = torch.einsum('ntul,lb->ntub', S[..., :L_c], self.coup_basis)
                N, T, U, nc = cf.shape
                W = self.coup_weights[session_name]
                W = W * (1.0 - torch.eye(U, device=W.device)).unsqueeze(-1)  # drop i==j
                eta = eta + cf.reshape(N, T, U * nc) @ W.reshape(U, U * nc).t()
        if self.use_acausal:
            # own past+future, lag 0 excluded by construction of the basis/stack
            A = self._acausal_stack(sp)  # (N,T,U,2*n_lags)
            af = torch.einsum('ntul,lb->ntub', A, self.acausal_basis)
            eta = eta + (af * self.acausal_weights[session_name]).sum(-1)
        return eta

    def forward(self, session_name, x, mask=None, spike=None,
                cluster_idx=None, **kwargs):
        design = self._make_design(x, mask)
        eta = self._spatial_drive(session_name, design, cluster_idx)
        eta = self._apply_temporal(eta, session_name, spike, mask)
        return self.final_activation(eta)


class GLM_Flight(GLM_Coupled):
    """Trajectory GLM with a 1-D "distance-along-path" basis for clustered flights.

    Like :class:`GLM_Coupled` with ``cluster_keys`` (joint / trajectory conditioning),
    but the per-cluster place-field readout is replaced by a 1-D basis arranged ALONG
    the flight path: each timestep's normalized cumulative arc length ``s in [0, 1]``
    (distance travelled / total path length, computed per flight in metres after
    un-z-scoring) is expanded with ``n_flight_bases`` raised-cosine bumps and routed to
    that cluster's readout. Unclustered ('other') flights fall back to the standard
    anisotropic 3-D place basis (they share no common geometry, so an along-path
    coordinate is meaningless for them).

    Spike-history / coupling / acausal temporal filters (if enabled) are identical to
    :class:`GLM_Coupled` and shared across clusters.

    Parameters
    ----------
    behavior_stats : dict
        ``{session_name: (mean, std)}`` position z-score stats (>=3 dims); used to
        un-z-score xyz to metres before computing arc length.
    cluster_keys : dict
        ``{session_name: [cluster_id, ...]}`` of major clusters per session (as in
        :class:`GLM_Coupled`); an ``'other'`` readout is always added.
    n_flight_bases : int
        Number of 1-D raised-cosine bases along the normalized path (clustered flights).
    n_bases_per_axis : int or (nx, ny, nz)
        Place-basis resolution for the 'other' readout when ``other_mode='spatial'``
        (anisotropic; e.g. fewer z).
    other_mode : {'spatial', 'flight'}
        How unclustered ('other') flights are encoded. ``'spatial'`` (default): a
        3-D place basis (``n_bases_per_axis``) — 'other' flights share no geometry,
        so this is a genuine place field. ``'flight'``: 'other' flights get their OWN
        set of 1-D arc-length bases (``n_other_flight_bases``, default
        ``n_flight_bases``) with their own readout — an along-path model for the
        pooled 'other' set (captures average along-flight rate modulation with far
        fewer parameters than a 3-D field).
    n_other_flight_bases : int or None
        Number of 1-D bases for the 'other' readout when ``other_mode='flight'``
        (defaults to ``n_flight_bases``).
    """
    def __init__(self, output_dim_dict, behavior_stats, cluster_keys,
                 n_flight_bases=8, s_range=(0.0, 1.0), flight_width_ratio=2.0,
                 n_bases_per_dim=16, n_bases_per_axis=None, pos_range=(-3.0, 3.0),
                 width_to_spacing_ratio=4, activation='exp', bin_width=0.1,
                 hist_basis=None, coup_basis=None, acausal_basis=None, n_lags=None,
                 other_mode='spatial', n_other_flight_bases=None):
        assert cluster_keys is not None, 'GLM_Flight is per-cluster (joint) by design'
        assert other_mode in ('spatial', 'flight'), other_mode
        super().__init__(output_dim_dict, n_bases_per_dim=n_bases_per_dim,
                         pos_range=pos_range, width_to_spacing_ratio=width_to_spacing_ratio,
                         activation=activation, bin_width=bin_width,
                         hist_basis=hist_basis, coup_basis=coup_basis,
                         acausal_basis=acausal_basis, n_lags=n_lags,
                         cluster_keys=cluster_keys, n_bases_per_axis=n_bases_per_axis)
        self.n_place = self.n_spatial            # 'other' 3-D place basis size
        self.n_flight_bases = n_flight_bases
        self.other_mode = other_mode
        # 1-D arc-length basis over normalized path fraction s in [0, 1] (clustered).
        s_centers = torch.linspace(s_range[0], s_range[1], n_flight_bases)
        spacing = (s_range[1] - s_range[0]) / (n_flight_bases - 1)
        self.register_buffer('s_centers', s_centers)
        self.s_width = float(flight_width_ratio * spacing)
        # 'other' readout input size + (for other_mode='flight') its own 1-D basis.
        if other_mode == 'flight':
            n_ob = int(n_other_flight_bases or n_flight_bases)
            oc = torch.linspace(s_range[0], s_range[1], n_ob)
            self.register_buffer('s_other_centers', oc)
            self.s_other_width = float(flight_width_ratio * (s_range[1] - s_range[0]) / (n_ob - 1))
            self.n_other_flight_bases = n_ob
            other_in = n_ob
        else:
            other_in = self.n_place
        # Rebuild per-(session, cluster) readouts: clustered -> 1-D (n_flight_bases);
        # 'other' -> 3-D place (n_place) or its own 1-D basis (other_in).
        readout_dict = {}
        for s, u in output_dim_dict.items():
            keys = self.cluster_keys[s]                       # includes 'other'
            readout_dict[s] = nn.ModuleDict({
                k: nn.Linear(other_in if k == _OTHER_CLUSTER_KEY else n_flight_bases, u)
                for k in keys})
        self.readouts = nn.ModuleDict(readout_dict)
        # Per-session z-score stats -> un-z-score xyz to metres for arc length.
        self._session_to_idx = {}
        for i, (s, (mean, std)) in enumerate(behavior_stats.items()):
            self._session_to_idx[s] = i
            m = torch.as_tensor(np.asarray(mean).reshape(-1)[:3], dtype=torch.float32)
            sd = torch.as_tensor(np.asarray(std).reshape(-1)[:3], dtype=torch.float32)
            self.register_buffer(f'beh_mean_{i}', m)
            self.register_buffer(f'beh_std_{i}', sd)

    def _arclength(self, x, mask, session_name):
        """Normalized cumulative distance-along-path s in [0, 1], per flight."""
        idx = self._session_to_idx[session_name]
        mean = getattr(self, f'beh_mean_{idx}'); std = getattr(self, f'beh_std_{idx}')
        xyz = x[..., :3] * std + mean                          # (N,T,3) metres
        step = torch.linalg.norm(xyz[:, 1:] - xyz[:, :-1], dim=-1)   # (N,T-1)
        if mask is not None:
            step = step * (mask[:, 1:] * mask[:, :-1])          # ignore padded steps
        s = torch.zeros_like(xyz[..., 0])                       # (N,T)
        s[:, 1:] = torch.cumsum(step, dim=1)
        total = s.amax(dim=1, keepdim=True).clamp_min(1e-6)     # per-flight path length
        return (s / total).clamp(0.0, 1.0)

    def _flight_spatial_drive(self, session_name, x, mask, cluster_idx):
        """Route each flight to its cluster's 1-D readout; 'other' -> 3-D place."""
        ro = self.readouts[session_name]
        keys = self.cluster_keys[session_name]
        N, T, _ = x.shape
        units = ro[keys[0]].out_features
        eta = x.new_zeros(N, T, units)
        if cluster_idx is None:
            cluster_idx = torch.full((N,), -1, dtype=torch.long, device=x.device)
        groups = {}
        for n, c in enumerate(cluster_idx.tolist()):
            k = str(int(c)) if str(int(c)) in ro else _OTHER_CLUSTER_KEY
            groups.setdefault(k, []).append(n)
        for k, idx in groups.items():
            sel = torch.as_tensor(idx, device=x.device)
            xs = x[sel]
            ms = mask[sel] if mask is not None else None
            if k == _OTHER_CLUSTER_KEY and self.other_mode == 'spatial':
                design = self._spatial_basis(xs)                # 3-D place (anisotropic)
            elif k == _OTHER_CLUSTER_KEY:                       # other_mode == 'flight'
                s = self._arclength(xs, ms, session_name)       # 'other's own 1-D basis
                design = create_cosine_bumps_torch(s, self.s_other_centers.to(x.device), self.s_other_width)
            else:
                s = self._arclength(xs, ms, session_name)       # (n, T) clustered 1-D
                design = create_cosine_bumps_torch(s, self.s_centers.to(x.device), self.s_width)
            if ms is not None:
                design = design * ms.unsqueeze(-1)
            eta[sel] = ro[k](design)
        return eta

    def forward(self, session_name, x, mask=None, spike=None, cluster_idx=None, **kwargs):
        eta = self._flight_spatial_drive(session_name, x, mask, cluster_idx)
        eta = self._apply_temporal(eta, session_name, spike, mask)
        return self.final_activation(eta)
