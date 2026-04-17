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
    ):
        super().__init__()
        if lag_seconds is None:
            lag_seconds = [0.0]
        self.lag_bins = [int(round(s / bin_width)) for s in lag_seconds]
        n_lags = len(self.lag_bins)

        centers = torch.linspace(pos_range[0], pos_range[1], n_bases_per_dim)
        spacing = (pos_range[1] - pos_range[0]) / (n_bases_per_dim - 1)
        self.register_buffer('centers', centers)
        self.width = width_to_spacing_ratio * spacing
        self.n_bases_per_dim = n_bases_per_dim

        n_spatial = n_bases_per_dim ** 3
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
        centers = self.centers.to(xyz.device)
        b_x = create_cosine_bumps_torch(xyz[..., 0], centers, self.width)
        b_y = create_cosine_bumps_torch(xyz[..., 1], centers, self.width)
        b_z = create_cosine_bumps_torch(xyz[..., 2], centers, self.width)
        bases = (
            b_x.unsqueeze(-1).unsqueeze(-1)
            * b_y.unsqueeze(-2).unsqueeze(-1)
            * b_z.unsqueeze(-2).unsqueeze(-2)
        )
        return bases.flatten(-3)  # (N, T, K^3)

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
        N, T, _ = x.shape
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
