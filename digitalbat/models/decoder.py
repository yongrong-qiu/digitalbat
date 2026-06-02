"""Position decoders that map population spiking activity to allocentric position.

Two decoders, sharing a common dataloader interface (the namedtuple `batch.spike`
+ `batch.behavior` produced by `bat_data_loader`):

  - `Bat_Transformer_Decoder`: an "inverted" `Bat_Transformer`. Per-session
    `nn.Linear(num_neurons, latent_dim)` readin gives each neuron a learned
    column embedding (the inverse of the per-session `readouts` in the encoder
    model). A RoPE transformer encoder runs on the resulting latent sequence,
    and a shared linear head maps to z-scored 3D position.

  - `NaiveBayes_Decoder`: Zhang et al. (1998) population Bayesian decoder.
    Per-session smoothed Poisson rate maps from training data; argmax of the
    log-posterior at each test timepoint.

Conventions:
  - Spikes (input) are the raw `batch.spike` counts per bin (shape (B, T, K)
    where K = neurons in that session). The Naive Bayes decoder REQUIRES these
    to be raw counts.
  - Positions (target) are the FIRST 3 channels of `batch.behavior`, which are
    z-scored (x, y, z). The transformer decoder predicts these in z-scored
    space; the notebook un-z-scores when reporting error in meters.
  - `batch.mask` (B, T) marks valid timesteps; both decoders ignore padded
    bins for fitting/loss and for evaluation.
"""

import copy
from typing import Callable

import numpy as np
import torch
import torch.optim as optim
import tqdm
from torch import nn

from ..utility.utils import round_robin_session_batches

from .layers import TransformerEncoderRoPE


class Bat_Transformer_Decoder(nn.Module):
    """Transformer decoder mapping spikes -> z-scored 3D position.

    Inverse of `Bat_Transformer`: instead of session-specific readouts that
    project a shared latent to per-session neuron counts, this model has
    session-specific readins that project per-session neuron counts to a
    shared latent. The readin's weight matrix has shape (latent_dim, K_session),
    so column j is the learned embedding of neuron j in that session.

    Parameters
    ----------
    input_dim_dict : dict
        Mapping {session_name: num_neurons}. Same shape conventions as
        `Bat_Transformer.output_dim_dict` but used for input projections here.
    output_dim : int
        Number of position dimensions to predict. Default 3 (x, y, z).
    latent_dim, num_heads, ff_dim, num_layers, dropout_prob :
        Transformer hyperparameters; mirror `Bat_Transformer`.
    final_activation : nn.Module
        Applied to the final position head. Default `nn.Identity()` since
        positions are real-valued (z-scored).
    input_transform : {'identity', 'sqrt'}
        Optional variance-stabilizing transform applied to the spike input
        before the readin. `'sqrt'` uses `sqrt(x)` to tame the long tail
        of raw Poisson counts.
    """

    def __init__(
        self,
        input_dim_dict: dict,
        output_dim: int = 3,
        latent_dim: int = 32,
        num_heads: int = 4,
        ff_dim: int = 32,
        num_layers: int = 1,
        dropout_prob: float = 0.3,
        readin_dropout: float = 0.1,
        use_low_rank: bool = False,
        low_rank_r: int = 4,
        final_activation: nn.Module = nn.Identity(),
        input_transform: str = 'identity',
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        if input_transform not in ('identity', 'sqrt'):
            raise ValueError(f"input_transform must be 'identity' or 'sqrt', got {input_transform!r}")
        self.input_transform = input_transform

        # Per-session neuron embeddings: rows of W are latent positions of each neuron.
        self.readins = nn.ModuleDict({
            session_name: nn.Linear(unit_num, latent_dim)
            for session_name, unit_num in input_dim_dict.items()
        })
        self.readin_dropout = nn.Dropout(readin_dropout)

        self.layers = nn.ModuleList([
            TransformerEncoderRoPE(
                hidden_dim=latent_dim,
                num_head=num_heads,
                ff_dim=ff_dim,
                dropout=dropout_prob,
                use_low_rank=use_low_rank,
                low_rank_r=low_rank_r,
            )
            for _ in range(num_layers)
        ])

        self.head = nn.Linear(latent_dim, output_dim)
        self.final_activation = final_activation

    def forward(
        self,
        session_name: str,
        spikes: torch.Tensor,
        mask: torch.Tensor,
        is_causal: bool = False,
        is_bigbird: bool = False,
        attention_window_size: int = 3,
        num_global: int = 2,
        num_random: int = 1,
        return_attn_weights: bool = False,
    ) -> torch.Tensor:
        """spikes: (B, T, K_session). mask: (B, T) float, 1.0 valid."""
        mask = mask.to(spikes.device)
        attn_mask = torch.einsum("bi,bj->bij", mask.float(), mask.float()).bool()
        attn_mask = attn_mask[:, None, :, :]

        if self.input_transform == 'sqrt':
            spikes = torch.sqrt(torch.clamp(spikes, min=0.0))

        x = self.readin_dropout(self.readins[session_name](spikes))

        if return_attn_weights:
            attn_weights_list = []
            for layer in self.layers:
                x, aw = layer(
                    x, x, x,
                    attn_mask=attn_mask,
                    is_causal=is_causal,
                    return_attn_weights=True,
                    is_bigbird=is_bigbird,
                    attention_window_size=attention_window_size,
                    num_global=num_global,
                    num_random=num_random,
                )
                attn_weights_list.append(aw)
        else:
            for layer in self.layers:
                x = layer(
                    x, x, x,
                    attn_mask=attn_mask,
                    is_causal=is_causal,
                    return_attn_weights=False,
                    is_bigbird=is_bigbird,
                    attention_window_size=attention_window_size,
                    num_global=num_global,
                    num_random=num_random,
                )

        out = self.final_activation(self.head(x))
        if return_attn_weights:
            return out, attn_weights_list
        return out


def bat_decoder_trainer(
    model: Bat_Transformer_Decoder,
    dataloaders: dict,
    device,
    pos_dims: int = 3,
    is_causal: bool = False,
    is_bigbird: bool = False,
    attention_window_size: int = 3,
    num_global: int = 2,
    num_random: int = 1,
    n_epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    loss_fn: nn.Module | None = None,
    valid_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    valid_stat: str = "median",
    early_stopping_patience: int | None = None,
    save_model: bool = False,
    model_name: str = "best_decoder_weights.pth",
    trial=None,
):
    """Train `Bat_Transformer_Decoder` to predict z-scored position from spikes.

    The convention `batch.behavior[:, :, :pos_dims]` are the z-scored target
    positions (set by `bat_data_loader`'s `behavior_mean`/`behavior_std`).
    Loss and validation are masked: padded timesteps are dropped before the
    loss is computed.

    `valid_fn` should return a per-sample-or-per-dim score where LARGER is
    better (so MSE callers should negate). Defaults to negative-MSE per dim.

    `trial`, when given, is an Optuna trial: the per-epoch `val_score` is
    reported for pruning and `optuna.TrialPruned` is raised if the pruner
    flags the trial (no extra dependency unless `trial` is passed).
    """
    model = model.to(device)
    if loss_fn is None:
        loss_fn = nn.MSELoss()
    loss_fn = loss_fn.to(device)
    if valid_fn is None:
        def valid_fn(y_true, y_pred):
            # negative MSE per output dim → larger = better
            return -np.nanmean((y_true - y_pred) ** 2, axis=0)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_score = -np.inf
    best_weights = None
    val_scores = []
    epochs_without_improvement = 0

    for epoch in tqdm.tqdm(range(n_epochs), desc="Decoder training"):
        model.train()
        # Round-robin per-session batches: see `round_robin_session_batches`.
        for session_name, batch in round_robin_session_batches(dataloaders["train"]):
            X = batch.spike.to(device)
            y = batch.behavior[:, :, :pos_dims].to(device)
            m = batch.mask.to(device)
            y_pred = model(
                session_name, X, m,
                is_causal=is_causal,
                is_bigbird=is_bigbird,
                attention_window_size=attention_window_size,
                num_global=num_global,
                num_random=num_random,
            )
            loss = loss_fn(y_pred[m == 1.0], y[m == 1.0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation is pooled across sessions and flights: every valid
        # timestep contributes one row to a single (N_total, pos_dims) array,
        # `valid_fn` is called once, and `valid_stat` aggregates the resulting
        # per-pos-dim score vector. Matches `train_perceiver`'s pooled R².
        model.eval()
        all_yt, all_yp = [], []
        with torch.no_grad():
            for session_name, session_data in dataloaders["valid"].items():
                for batch in session_data:
                    X = batch.spike.to(device)
                    y = batch.behavior[:, :, :pos_dims].to(device)
                    m = batch.mask.to(device)
                    y_pred = model(
                        session_name, X, m,
                        is_causal=is_causal,
                        is_bigbird=is_bigbird,
                        attention_window_size=attention_window_size,
                        num_global=num_global,
                        num_random=num_random,
                    )
                    all_yt.append(y[m == 1.0].cpu().numpy())
                    all_yp.append(y_pred[m == 1.0].cpu().numpy())

        if not all_yt:
            val_score = -np.inf
        else:
            yt_np = np.vstack(all_yt)
            yp_np = np.vstack(all_yp)
            score_arr = valid_fn(yt_np, yp_np)
            if valid_stat == "mean":
                val_score = float(np.nanmean(score_arr))
            elif valid_stat == "median":
                val_score = float(np.nanmedian(score_arr))
            else:
                raise ValueError(f"unknown valid_stat={valid_stat}")
        val_scores.append(val_score)

        if trial is not None:
            import optuna
            trial.report(val_score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if val_score > best_score:
            best_score = val_score
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1} (no improvement for {early_stopping_patience} epochs)")
                break

    if save_model:
        torch.save(model.state_dict(), model_name)
    if best_weights is not None:
        model.load_state_dict(best_weights)
    return best_score, best_weights, val_scores


# =============================================================================
# Naive Bayes population decoder (Zhang et al., 1998)
# =============================================================================


def _smooth_map(data: np.ndarray, valid_mask: np.ndarray, sigma: float) -> np.ndarray:
    """Normalized Gaussian smoothing that respects an occupancy/validity mask.

    Mirrors `spatial_info.smooth_map`: smooths data and the binary weight
    mask separately and then divides, so out-of-support bins contribute zero
    weight rather than biasing edges toward zero.
    """
    from scipy.ndimage import gaussian_filter
    data_masked = np.where(valid_mask, data, 0.0)
    weights = valid_mask.astype(float)
    smooth_data = gaussian_filter(data_masked, sigma=sigma)
    smooth_weights = gaussian_filter(weights, sigma=sigma)
    out = np.full_like(data, np.nan, dtype=float)
    nz = smooth_weights > 1e-10
    out[nz] = smooth_data[nz] / smooth_weights[nz]
    return out


class NaiveBayes_Decoder:
    """Per-session N-D Naive Bayes population decoder (Zhang et al. 1998).

    Model: spikes given position are independent Poisson with rate
    `f_i(p) * dt`, so

        log P(p | n) = log P(p) + Σ_i [n_i log(τ f_i(p)) - τ f_i(p)] + const

    Decoded position is the argmax of this log-posterior over an N-D grid.
    Default is 2D (x, y); pass `n_bins=(nx, ny, nz)` for 3D.

    Fit with `fit(spikes, positions)`; predict with `predict(spikes)`.
    Fit and predict use independent data — there is no internal CV here.
    """

    def __init__(
        self,
        n_bins: tuple = (30, 25),
        lims: tuple | None = None,
        sigma: float = 2.0,
        dt: float = 0.1,
        min_occupancy: int = 5,
        rate_floor: float = 1e-4,
        use_uniform_prior: bool = False,
    ):
        """
        Parameters
        ----------
        n_bins : tuple of int
            Grid resolution per spatial dimension. Length determines `ndim`
            (2 or 3 typical). Example: (30, 25) for 2D, (30, 25, 15) for 3D.
        lims : sequence of (lo, hi) or None
            Per-axis spatial bounds. If None, set from training-data extent
            at fit time. If provided, length must match `n_bins`.
        sigma : float
            Gaussian kernel SD in BINS (not meters) for smoothing
            rate/occupancy maps. Applied isotropically across all axes.
        dt : float
            Bin width in seconds (must match `data_config['bin_width']`).
        min_occupancy : int
            Minimum samples per bin to be considered "in support" of the
            rate map.
        rate_floor : float
            Floor (Hz) on rates before taking logs.
        use_uniform_prior : bool
            If True, use flat prior over occupied bins. If False, use the
            smoothed empirical occupancy as P(p).
        """
        self.n_bins = tuple(n_bins)
        self.ndim = len(self.n_bins)
        if lims is not None and len(lims) != self.ndim:
            raise ValueError(f"len(lims)={len(lims)} != ndim={self.ndim}")
        self.lims = None if lims is None else [tuple(L) for L in lims]
        self.sigma = sigma
        self.dt = dt
        self.min_occupancy = min_occupancy
        self.rate_floor = rate_floor
        self.use_uniform_prior = use_uniform_prior

        self.edges = None       # list of (n_bins_d + 1,) per axis
        self.centers = None     # list of (n_bins_d,) per axis
        self.rate_maps = None   # (n_units, *n_bins)
        self.log_rate_maps = None
        self.tau_rate_sum = None  # (*n_bins)
        self.log_prior = None     # (*n_bins)
        self.support_mask = None  # (*n_bins) bool

    def fit(
        self,
        spikes: np.ndarray,
        positions: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> "NaiveBayes_Decoder":
        """Build smoothed rate maps and prior from training data.

        Parameters
        ----------
        spikes : (N_t, n_units) raw spike counts.
        positions : (N_t, ndim) positions (meters).
        mask : (N_t,) bool/float, 1.0 = valid. None = all valid.
        """
        positions = np.asarray(positions)
        if positions.ndim != 2 or positions.shape[1] != self.ndim:
            raise ValueError(
                f"positions must be (N_t, {self.ndim}); got {positions.shape}")

        if mask is None:
            keep = np.ones(positions.shape[0], dtype=bool)
        else:
            keep = np.asarray(mask).astype(bool)
        spikes = spikes[keep]
        positions = positions[keep]

        if self.lims is None:
            self.lims = [
                (float(np.nanmin(positions[:, d])),
                 float(np.nanmax(positions[:, d])))
                for d in range(self.ndim)
            ]
        self.edges = [
            np.linspace(lo, hi, n + 1)
            for (lo, hi), n in zip(self.lims, self.n_bins)
        ]
        self.centers = [0.5 * (e[:-1] + e[1:]) for e in self.edges]

        n_units = spikes.shape[1]
        occ, _ = np.histogramdd(positions, bins=self.edges)
        valid = occ >= self.min_occupancy
        if valid.sum() == 0:
            raise ValueError("No spatial bins meet min_occupancy threshold.")

        if self.sigma > 0:
            sm_occ = _smooth_map(occ, valid, self.sigma)
            support = np.isfinite(sm_occ) & (sm_occ > 1e-10)
        else:
            sm_occ = np.where(valid, occ, np.nan)
            support = valid

        self.support_mask = support

        log_prior = np.full(self.n_bins, -np.inf)
        if self.use_uniform_prior:
            log_prior[support] = -np.log(support.sum())
        else:
            occ_sum = float(np.nansum(sm_occ[support]))
            log_prior[support] = np.log(sm_occ[support] / occ_sum)
        self.log_prior = log_prior

        # Per-unit smoothed rate maps. Out-of-support bins → rate_floor so
        # log(τ·f) is finite (they get suppressed by support_mask at decode).
        rate_maps = np.full((n_units,) + self.n_bins, self.rate_floor)
        for u in range(n_units):
            spk, _ = np.histogramdd(
                positions, bins=self.edges, weights=spikes[:, u])
            if self.sigma > 0:
                sm_spk = _smooth_map(spk, valid, self.sigma)
                rm = np.full(self.n_bins, self.rate_floor)
                rm[support] = np.maximum(
                    sm_spk[support] / (sm_occ[support] * self.dt),
                    self.rate_floor,
                )
            else:
                rm = np.full(self.n_bins, self.rate_floor)
                rm[valid] = np.maximum(
                    spk[valid] / (occ[valid] * self.dt), self.rate_floor)
            rate_maps[u] = rm

        self.rate_maps = rate_maps
        self.log_rate_maps = np.log(self.dt * rate_maps)
        self.tau_rate_sum = (self.dt * rate_maps).sum(axis=0)
        return self

    def predict(
        self,
        spikes: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decode position at each test timepoint.

        Returns
        -------
        positions_dec : (N_test_valid, ndim) decoded positions (meters).
        valid_mask : (N_test,) bool — which input rows produced predictions.
        """
        if self.rate_maps is None:
            raise RuntimeError("Call fit() before predict().")
        if mask is None:
            keep = np.ones(spikes.shape[0], dtype=bool)
        else:
            keep = np.asarray(mask).astype(bool)
        spikes_valid = spikes[keep]

        # Σ_i n_i log(τ f_i)  −  τ Σ_i f_i  +  log P(p), restricted to support.
        # Flatten spatial axes for a single matmul.
        n_units = self.log_rate_maps.shape[0]
        flat_log_rates = self.log_rate_maps.reshape(n_units, -1)
        log_post_flat = spikes_valid @ flat_log_rates
        log_post_flat -= self.tau_rate_sum.reshape(1, -1)
        log_post_flat += self.log_prior.reshape(1, -1)

        oob = ~self.support_mask.reshape(-1)
        if oob.any():
            log_post_flat[:, oob] = -np.inf

        flat_idx = np.argmax(log_post_flat, axis=1)
        idx_per_axis = np.unravel_index(flat_idx, self.n_bins)
        positions_dec = np.column_stack(
            [self.centers[d][idx_per_axis[d]] for d in range(self.ndim)]
        )
        return positions_dec, keep


class NaiveBayes_Continuity_Decoder(NaiveBayes_Decoder):
    """Naive Bayes with continuity prior — Zhang et al. 1998 'Bayesian 2-step'.

    On top of the standard Poisson rate maps, fits a per-bin average speed map
    `U(x)` from consecutive-timestep position differences in the training data.
    At decode time, uses a greedy forward pass: for each timestep within a
    flight,

        x̂_t = argmax_x [ log P(x | n_t) + log N(x_{t-1} | x, σ_t(x)) ]

    where `σ_t(x) = clip(c · U(x), σ_min, σ_max)` (Zhang Eq. 44 with d=1).
    Including the σ-dependent normalization of the Gaussian (i.e. the
    `-D log σ_t(x)` term) matters here because σ varies across candidate x.
    The previous-step position `x_{t-1}` is the previously decoded point, not
    the ground-truth — this is the paper's online MAP forward pass, not Viterbi.
    The prior is reset at flight boundaries so trajectories don't bleed into
    each other.

    Parameters
    ----------
    c_speed : float
        Multiplier on `U(x)` to set the kernel SD. Larger c → weaker prior.
    sigma_min, sigma_max : float
        Floor and ceiling on σ_t in METERS. Without them, low-speed bins
        produce arbitrarily tight priors that lock decoding to the last
        position; high-speed bins produce arbitrarily loose ones. Defaults
        match the spirit of the paper's 0.2-0.6 m clip but are wider.
    Other args : passed to `NaiveBayes_Decoder`.
    """

    def __init__(
        self,
        c_speed: float = 1.0,
        sigma_min: float = 0.05,
        sigma_max: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.c_speed = c_speed
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.speed_map = None
        self._sigma_per_bin = None
        self._grid_positions = None

    def fit(
        self,
        spikes: np.ndarray,
        positions: np.ndarray,
        flight_id: np.ndarray,
        t_idx: np.ndarray,
        dt: float,
        mask: np.ndarray | None = None,
    ) -> "NaiveBayes_Continuity_Decoder":
        """Fit rate maps + occupancy prior + per-bin speed map.

        `flight_id` and `t_idx` align row-for-row with `positions`/`spikes` and
        let us identify consecutive timesteps within a flight (so we can
        compute speed without leaking across flight boundaries).
        """
        if mask is None:
            keep = np.ones(positions.shape[0], dtype=bool)
        else:
            keep = np.asarray(mask).astype(bool)

        # Standard rate-map / log-prior fit (sets self.edges, self.centers, etc.)
        super().fit(spikes[keep], positions[keep])

        fid = np.asarray(flight_id)[keep].astype(int)
        tid = np.asarray(t_idx)[keep].astype(int)
        pos = positions[keep]

        order = np.lexsort((tid, fid))
        fid_s = fid[order]
        tid_s = tid[order]
        pos_s = pos[order]

        same_flight = fid_s[:-1] == fid_s[1:]
        consecutive = (tid_s[1:] - tid_s[:-1]) == 1
        valid_pairs = same_flight & consecutive
        if not valid_pairs.any():
            raise ValueError("No consecutive timestep pairs found within flights.")

        speeds = np.linalg.norm(pos_s[1:] - pos_s[:-1], axis=1) / dt
        speeds = speeds[valid_pairs]
        # Use x_t (the earlier point of each consecutive pair) for U(x_t),
        # matching Zhang Eq. 44 where σ_t depends on the candidate x_t.
        pos_at_speed = pos_s[:-1][valid_pairs]

        speed_count, _ = np.histogramdd(pos_at_speed, bins=self.edges)
        speed_sum, _ = np.histogramdd(
            pos_at_speed, bins=self.edges, weights=speeds)
        valid_bins = speed_count > 0
        raw_speed = np.where(
            valid_bins, speed_sum / np.maximum(speed_count, 1), 0.0)

        if self.sigma > 0:
            smoothed = _smooth_map(raw_speed, valid_bins, self.sigma)
        else:
            smoothed = np.where(valid_bins, raw_speed, np.nan)

        global_mean_speed = float(np.mean(speeds))
        self.speed_map = np.where(
            np.isfinite(smoothed), smoothed, global_mean_speed)
        self._sigma_per_bin = np.clip(
            self.c_speed * self.speed_map, self.sigma_min, self.sigma_max)

        # Pre-compute grid positions in row-major order to align with the
        # flattened log-rate matrix used in `predict`.
        grids = np.meshgrid(*self.centers, indexing='ij')
        self._grid_positions = np.stack([g.ravel() for g in grids], axis=1)

        return self

    def predict(
        self,
        spikes: np.ndarray,
        flight_id: np.ndarray,
        t_idx: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Greedy forward decode with continuity prior. Sequential per flight."""
        if self._sigma_per_bin is None:
            raise RuntimeError("Call fit() before predict().")
        if mask is None:
            keep = np.ones(spikes.shape[0], dtype=bool)
        else:
            keep = np.asarray(mask).astype(bool)

        spk = spikes[keep]
        fid = np.asarray(flight_id)[keep].astype(int)
        tid = np.asarray(t_idx)[keep].astype(int)
        n_t = spk.shape[0]
        D = self.ndim

        # 1-step log-posterior at every test timepoint (vectorized).
        n_units = self.log_rate_maps.shape[0]
        flat_log_rates = self.log_rate_maps.reshape(n_units, -1)
        log_post = spk @ flat_log_rates
        log_post -= self.tau_rate_sum.reshape(1, -1)
        log_post += self.log_prior.reshape(1, -1)
        oob = ~self.support_mask.reshape(-1)
        if oob.any():
            log_post[:, oob] = -np.inf

        sigma_flat = self._sigma_per_bin.reshape(-1)
        log_sigma = np.log(sigma_flat)
        inv2sig2 = 0.5 / (sigma_flat ** 2)
        grid = self._grid_positions  # (M, D)

        # Sort timesteps by (flight, t_idx) for sequential MAP decoding.
        order = np.lexsort((tid, fid))
        fid_s = fid[order]
        log_post_s = log_post[order]

        decoded_s = np.zeros((n_t, D))
        prev_fid = -1
        prev_pos = None
        for i in range(n_t):
            if fid_s[i] != prev_fid:
                lp = log_post_s[i]
            else:
                d2 = ((grid - prev_pos[None, :]) ** 2).sum(axis=1)
                lp = log_post_s[i] - d2 * inv2sig2 - D * log_sigma
            idx = int(np.argmax(lp))
            decoded_s[i] = grid[idx]
            prev_fid = fid_s[i]
            prev_pos = decoded_s[i]

        inv_order = np.argsort(order)
        return decoded_s[inv_order], keep


# ---------------------------------------------------------------------------
# Correlation-aware decoder: MAP inversion of a fitted GLM_Coupled encoder
# ---------------------------------------------------------------------------
#
# The Naive-Bayes decoders above treat neurons as conditionally independent
# given position. `GLM_MAP_Decoder` instead inverts a *fixed, pre-trained*
# `GLM_Coupled` (the "GLM + hist + coup" encoder from model_comparison.ipynb),
# whose spike-history and cross-neuron coupling filters model the population's
# noise correlations explicitly. Given a whole flight's spikes, it finds the
# trajectory X = [x(1..T)] that maximizes the GLM's Poisson log-likelihood plus
# a trajectory-smoothness prior (a MAP / Laplace estimate), by gradient descent
# on the 3T-dimensional trajectory. Because each neuron's predicted rate at t
# depends on *other* neurons' spikes (the coupling term, including the lag-0
# simultaneous bin), the decoder uses co-firing structure that NB ignores.
#
# The GLM_Coupled encoder is treated as a black box accessed through its public
# pieces: `_spatial_basis`, `_lagged_stack`, `readouts[session]`, the temporal
# bases (`hist_basis`/`coup_basis`) and their weights. It returns exp(eta) =
# expected spike COUNT per bin (trained with PoissonNLLLoss(log_input=False)),
# so the Poisson likelihood uses Delta = 1 (the bin width is already folded into
# the readout bias). Positions are decoded in the GLM's z-scored coordinates and
# converted back to meters at the end.


def glm_history_term(glm, session_name, spike):
    """Spike-history + coupling contribution to eta (the X-independent part).

    Mirrors the temporal-filter branch of ``GLM_Coupled.forward`` but excludes
    the spatial drive and the readout bias (the bias is the per-neuron baseline
    ``b_i`` and is supplied by :func:`glm_place_field_fn` instead, so it is not
    double-counted).

    Parameters
    ----------
    glm : GLM_Coupled
        A fitted encoder with ``use_hist`` and/or ``use_coup`` enabled.
    session_name : str
        Session key into the per-session filter weights.
    spike : torch.Tensor
        ``(B, T, N)`` observed spike counts (``B`` flights, zero-padded). A
        single flight is passed as ``(1, T, N)``.

    Returns
    -------
    torch.Tensor
        ``(B, T, N)`` constant-in-X contribution to the linear predictor.
    """
    sp = spike.float()
    L_h = glm.hist_basis.shape[0] if glm.use_hist else 0
    L_c = glm.coup_basis.shape[0] if glm.use_coup else 0
    S = glm._lagged_stack(sp, max(L_h + 1, L_c))  # (B,T,N,*); S[...,0]=current
    B, T, U = sp.shape
    term = sp.new_zeros(B, T, U)
    if glm.use_hist:
        # strictly past: lags 1..L_h
        hf = torch.einsum('ntul,lb->ntub', S[..., 1:1 + L_h], glm.hist_basis)
        term = term + (hf * glm.hist_weights[session_name]).sum(-1)
    if glm.use_coup:
        # lags 0..L_c-1 (includes the simultaneous bin; diagonal masked below)
        cf = torch.einsum('ntul,lb->ntub', S[..., :L_c], glm.coup_basis)
        _, _, _, nc = cf.shape
        W = glm.coup_weights[session_name]
        W = W * (1.0 - torch.eye(U, device=W.device)).unsqueeze(-1)  # drop i==j
        term = term + cf.reshape(B, T, U * nc) @ W.reshape(U, U * nc).t()
    return term  # (B, T, N)


def glm_place_field_fn(glm, session_name):
    """Return a differentiable, shape-generic callable ``X_z -> f``.

    ``f`` is the spatial drive ``readouts[session](phi(x))`` *including* the
    readout bias (the per-neuron baseline ``b_i``). ``X_z`` must be in the GLM's
    z-scored position coordinates (``pos_range``). The callable preserves leading
    dims, so it works for one flight ``(T,3) -> (T,N)``, a padded batch
    ``(B,T,3) -> (B,T,N)``, and per-element under ``torch.func.vmap``.
    """
    readout = glm.readouts[session_name]

    def place_field(X_z):
        return readout(glm._spatial_basis(X_z))

    return place_field


def _make_neg_log_posterior(y, history_term, place_field_fn, sigma_a):
    """Build the negated log-posterior closure for one flight.

    Objective (negated for minimization)::

        eta(t)    = history_term(t) + f(x(t))            # (T, N)
        log_lik   = sum_t,i [ y_i(t) eta_i(t) - exp(eta_i(t)) ]   # Delta = 1
        d2(t)     = x(t+1) - 2 x(t) + x(t-1)             # acceleration
        log_prior = -||d2||^2 / (2 sigma_a^2)
    """
    history_term = history_term.detach()  # constant in X; no graph to encoder params
    T = y.shape[0]

    def neg_log_posterior(X):
        eta = history_term + place_field_fn(X)
        log_lik = (y * eta - torch.exp(eta)).sum()
        if T >= 3:
            d2 = X[2:] - 2.0 * X[1:-1] + X[:-2]
            log_prior = -(d2 ** 2).sum() / (2.0 * sigma_a ** 2)
        else:
            log_prior = X.new_zeros(())
        return -(log_lik + log_prior)

    return neg_log_posterior


def _lbfgs_basin(neg_log_posterior, X_init, lbfgs_iters=25):
    """Cheap L-BFGS descent to a local basin. Returns ``(X (T,3), nlp)``.

    ``lbfgs_iters`` is the total L-BFGS iteration budget (a single ``step`` with
    ``max_iter``). From a warm start this is plenty; the Newton polish finishes.
    """
    X = X_init.clone().detach().requires_grad_(True)
    opt = torch.optim.LBFGS([X], lr=0.1, max_iter=lbfgs_iters,
                            line_search_fn='strong_wolfe',
                            tolerance_grad=1e-6, tolerance_change=1e-9)

    def closure():
        opt.zero_grad()
        loss = neg_log_posterior(X)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        nlp = float(neg_log_posterior(X).item())
    return X.detach(), nlp


def _newton_polish(neg_log_posterior, X_init, newton_iters=3, damping=1e-6):
    """Newton polish on the full ``3T x 3T`` Hessian + Laplace covariances.

    Returns ``(X_hat (T,3), marginal_cov (T,3,3), nlp)``. The marginal
    covariances are the diagonal ``3x3`` blocks of the inverted Hessian.
    """
    T = X_init.shape[0]
    X = X_init.clone().detach().requires_grad_(True)
    eye = torch.eye(T * 3, device=X.device, dtype=X.dtype)
    with torch.no_grad():
        nlp_cur = float(neg_log_posterior(X).item())
    for _ in range(newton_iters):
        loss = neg_log_posterior(X)
        g = torch.autograd.grad(loss, X, create_graph=False)[0].reshape(-1)
        H = torch.autograd.functional.hessian(
            neg_log_posterior, X, vectorize=True).reshape(T * 3, T * 3)
        H = H + damping * eye
        try:
            delta = torch.linalg.solve(H, g).reshape(T, 3)
        except RuntimeError:
            break
        # Backtracking line search: monotone decrease guards against overshoot.
        alpha = 1.0
        accepted = False
        for _bt in range(8):
            with torch.no_grad():
                Xn = X - alpha * delta
                nlpn = float(neg_log_posterior(Xn).item())
            if nlpn < nlp_cur:
                with torch.no_grad():
                    X.copy_(Xn)
                nlp_cur = nlpn
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break

    with torch.no_grad():
        nlp = float(neg_log_posterior(X).item())
    H = torch.autograd.functional.hessian(
        neg_log_posterior, X, vectorize=True).reshape(T * 3, T * 3) + damping * eye
    try:
        Sigma = torch.linalg.inv(H).reshape(T, 3, T, 3)
        marginal_cov = torch.stack([Sigma[t, :, t, :] for t in range(T)])
    except RuntimeError:
        marginal_cov = torch.full((T, 3, 3), float('nan'),
                                  device=X.device, dtype=X.dtype)
    return X.detach(), marginal_cov, nlp


def decode_flight(y, history_term, place_field_fn, sigma_a, X_init,
                  lbfgs_iters=20, newton_iters=3, damping=1e-6):
    """MAP-decode one flight from a single warm start (L-BFGS then Newton).

    Returns ``(X_hat (T,3), marginal_cov (T,3,3), neg_log_post)``. For ``T < 3``
    the acceleration prior is undefined, so the warm start is returned with NaN
    covariance.
    """
    T = y.shape[0]
    nlp_fn = _make_neg_log_posterior(y, history_term, place_field_fn, sigma_a)
    if T < 3:
        X_hat = X_init.detach().clone()
        cov = torch.full((T, 3, 3), float('nan'),
                         device=X_init.device, dtype=X_init.dtype)
        return X_hat, cov, float(nlp_fn(X_hat).item())
    X_basin, _ = _lbfgs_basin(nlp_fn, X_init, lbfgs_iters=lbfgs_iters)
    return _newton_polish(nlp_fn, X_basin, newton_iters=newton_iters,
                          damping=damping)


def decode_with_restarts(y, history_term, place_field_fn, sigma_a, X_inits,
                         lbfgs_iters=20, newton_iters=3):
    """Multi-restart MAP decode: cheap L-BFGS on all inits, polish the winner.

    Each warm start in ``X_inits`` is taken to its basin with L-BFGS only; the
    init with the lowest negative log posterior is then Newton-polished once
    (the expensive full-Hessian step). This keeps the Hessian count independent
    of the number of restarts. Returns
    ``(X_hat, marginal_cov, neg_log_post, spread)`` where ``spread`` is the mean
    per-timestep L2 disagreement (z-scored units) between the restart basins — a
    cheap ambiguity diagnostic.
    """
    T = y.shape[0]
    nlp_fn = _make_neg_log_posterior(y, history_term, place_field_fn, sigma_a)

    if T < 3:
        X_hat = X_inits[0].detach().clone()
        cov = torch.full((T, 3, 3), float('nan'),
                         device=X_hat.device, dtype=X_hat.dtype)
        return X_hat, cov, float(nlp_fn(X_hat).item()), 0.0

    basins = [_lbfgs_basin(nlp_fn, Xi, lbfgs_iters=lbfgs_iters)
              for Xi in X_inits]
    best_idx = min(range(len(basins)), key=lambda i: basins[i][1])
    X_hat, marginal_cov, nlp = _newton_polish(
        nlp_fn, basins[best_idx][0], newton_iters=newton_iters)

    if len(basins) > 1:
        stacked = torch.stack([b[0] for b in basins])            # (R, T, 3)
        spread = float((stacked - X_hat.unsqueeze(0))
                       .norm(dim=-1).mean().item())
    else:
        spread = 0.0
    return X_hat, marginal_cov, nlp, spread


def _second_difference_matrix(T, device, dtype):
    """``(T-2, T)`` discrete 2nd-difference operator ``D2`` (stencil [1,-2,1])."""
    D2 = torch.zeros(max(T - 2, 0), T, device=device, dtype=dtype)
    for r in range(T - 2):
        D2[r, r] = 1.0
        D2[r, r + 1] = -2.0
        D2[r, r + 2] = 1.0
    return D2


def _prior_hessian(mask, sigma_a):
    """Constant acceleration-prior Hessian, ``(B, 3T, 3T)``.

    The prior ``(1/2 sigma_a^2) * sum ||D2 x||^2`` is quadratic, so its Hessian
    is ``(1/sigma_a^2) (D2^T diag(pm) D2) ⊗ I_3`` — independent of X. ``pm`` keeps
    only interior windows whose three timesteps are all valid.
    """
    B, T = mask.shape
    dev, dtype = mask.device, mask.dtype
    D2 = _second_difference_matrix(T, dev, dtype)               # (T-2, T)
    pm = mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]             # (B, T-2)
    # Hs[b] = (1/sigma_a^2) D2^T diag(pm[b]) D2  -> (B, T, T)
    Hs = torch.einsum('rt,br,rs->bts', D2, pm, D2) / (sigma_a ** 2)
    Hp = torch.zeros(B, 3 * T, 3 * T, device=dev, dtype=dtype)
    idx = torch.arange(T, device=dev)
    for d in range(3):
        ii = idx * 3 + d
        Hp[:, ii.unsqueeze(1), ii.unsqueeze(0)] = Hs
    return Hp


def _analytic_hessian_and_grad(glm, session_name, X, history_term, y, mask,
                               Hp, eye, pad_diag, damping):
    """Analytic negative-log-posterior gradient + Hessian at ``X``.

    Exploits the GLM's structure so no dense reverse-mode Hessian is built:

    * The likelihood Hessian is **block-diagonal in time** — one ``3x3`` block per
      timestep — because ``x(t)`` only enters that bin's rates.
    * The place field's position derivatives ``grad_x f`` (``3``) and
      ``grad^2_x f`` (``3x3``) come from **forward-mode** ``jacfwd`` of the 3->K^3
      spatial basis (cheap: input is 3-D), not reverse-mode through K^3 outputs.
    * The prior Hessian ``Hp`` is constant (quadratic prior), passed in.

    Returns ``(g (B,3T), H (B,3T,3T))`` for the *negative* log posterior.
    """
    B, T, _ = X.shape
    readout = glm.readouts[session_name]
    W = readout.weight                                          # (N, K^3)

    basis_pt = lambda x3: glm._spatial_basis(x3)               # (3,) -> (K^3,)
    jac_fn = torch.func.jacfwd(basis_pt)                       # (K^3, 3)
    hes_fn = torch.func.jacfwd(torch.func.jacfwd(basis_pt))    # (K^3, 3, 3)
    Xf = X.reshape(B * T, 3)
    J = torch.func.vmap(jac_fn)(Xf).reshape(B, T, -1, 3)       # (B,T,K^3,3)
    Hbz = torch.func.vmap(hes_fn)(Xf).reshape(B, T, -1, 3, 3)  # (B,T,K^3,3,3)

    eta = history_term + readout(glm._spatial_basis(X))        # (B,T,N)
    lam = torch.exp(eta)
    gf = torch.einsum('nk,btkd->btnd', W, J)                   # (B,T,N,3)
    Hf = torch.einsum('nk,btkde->btnde', W, Hbz)              # (B,T,N,3,3)

    m = mask.unsqueeze(-1)                                     # (B,T,1)
    resid = (lam - y) * m                                      # (B,T,N), masked
    # Per-timestep likelihood Hessian block (of the negative log-lik):
    #   (lam_i - y_i) Hf_i  +  lam_i gf_i gf_i^T
    Hblk = (torch.einsum('btn,btnde->btde', resid, Hf)
            + torch.einsum('btn,btnd,btne->btde', lam * mask.unsqueeze(-1),
                           gf, gf))                            # (B,T,3,3)
    # Likelihood gradient (negative log-lik): sum_i (lam_i - y_i) gf_i
    g_lik = torch.einsum('btn,btnd->btd', resid, gf).reshape(B, 3 * T)

    # Assemble block-diagonal likelihood Hessian into (B,3T,3T).
    H = Hp.clone()
    for t in range(T):
        H[:, 3 * t:3 * t + 3, 3 * t:3 * t + 3] += Hblk[:, t]
    H = H + damping * eye + torch.diag_embed(pad_diag)
    # Prior gradient = Hp @ X (quadratic, zero linear term).
    g = g_lik + torch.bmm(Hp, X.reshape(B, 3 * T, 1)).squeeze(-1)
    return g, H


def decode_session_batch(glm, session_name, y, history_term, mask, sigma_a,
                         X_inits, lbfgs_iters=20, newton_iters=3, damping=1e-6):
    """Batched MAP decode of all flights in a session (shared neuron set).

    All tensors are zero-padded to a common ``Tmax`` with a validity ``mask``.
    Phase 1 runs ONE L-BFGS over the whole ``(restart x flight)`` batch on the
    masked summed objective — flights are independent in the objective; only the
    L-BFGS line-search step is shared, which the warm starts and the Newton polish
    absorb. Phase 2 Newton-polishes the per-flight winners using the **analytic
    block-diagonal likelihood Hessian** (see :func:`_analytic_hessian_and_grad`)
    plus a single batched solve, so the Hessian never materializes the dense
    reverse-mode autograd graph through the K^3 design — the only step that does
    not batch cheaply otherwise.

    Parameters
    ----------
    glm : GLM_Coupled
        The fixed encoder (provides the readout weight + spatial basis).
    session_name : str
        Session key for the readout.
    y, history_term : torch.Tensor
        ``(B, Tmax, N)`` spike counts and precomputed history+coupling drive.
    mask : torch.Tensor
        ``(B, Tmax)`` 1.0 on valid timesteps, 0.0 on padding.
    sigma_a : float
        Acceleration-noise scale (z-scored units).
    X_inits : list of torch.Tensor
        Each ``(B, Tmax, 3)`` z-scored warm start (one per restart).

    Returns
    -------
    X_hat : (B, Tmax, 3) ; marginal_cov : (B, Tmax, 3, 3) ; nlp : (B,) ;
    spread : (B,)  — all on ``y.device``. Padded entries are inert (X stays 0).
    """
    history_term = history_term.detach()
    B, Tmax, N = y.shape
    R = len(X_inits)
    dev, dtype = y.device, y.dtype
    place_field_fn = glm_place_field_fn(glm, session_name)

    def batched_nlp(X, yy, hh, mm):
        """Per-flight negative log posterior (masked). X: (M,Tmax,3) -> (M,)."""
        eta = hh + place_field_fn(X)
        ll = ((yy * eta - torch.exp(eta)).sum(-1) * mm).sum(-1)
        d2 = X[:, 2:] - 2.0 * X[:, 1:-1] + X[:, :-2]
        pm = mm[:, 2:] * mm[:, 1:-1] * mm[:, :-2]
        lp = -((d2 ** 2).sum(-1) * pm).sum(-1) / (2.0 * sigma_a ** 2)
        return -(ll + lp)

    # ---- Phase 1: batched L-BFGS over R*B flights from the restart inits.
    y_b = y.repeat(R, 1, 1)
    hist_b = history_term.repeat(R, 1, 1)
    mask_b = mask.repeat(R, 1)
    Xbig = torch.cat(list(X_inits), dim=0).clone().detach()      # (R*B,Tmax,3)
    Xbig = Xbig * mask_b.unsqueeze(-1)   # zero padded positions (kept finite)
    Xbig.requires_grad_(True)

    opt = torch.optim.LBFGS([Xbig], lr=0.1, max_iter=lbfgs_iters,
                            line_search_fn='strong_wolfe',
                            tolerance_grad=1e-6, tolerance_change=1e-9)

    def closure():
        opt.zero_grad()
        loss = batched_nlp(Xbig, y_b, hist_b, mask_b).sum()
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        nlp_all = batched_nlp(Xbig, y_b, hist_b, mask_b).reshape(R, B)
        Xbasins = Xbig.detach().reshape(R, B, Tmax, 3)
    best_r = nlp_all.argmin(dim=0)                               # (B,)
    bidx = torch.arange(B, device=dev)
    X = Xbasins[best_r, bidx].clone()                           # (B,Tmax,3)

    if R > 1:
        diff = (Xbasins - X.unsqueeze(0)).norm(dim=-1)          # (R,B,Tmax)
        spread = ((diff * mask.unsqueeze(0)).sum(-1)
                  / mask.sum(-1).clamp(min=1)).mean(0)          # (B,)
    else:
        spread = torch.zeros(B, device=dev, dtype=dtype)

    # ---- Phase 2: Newton polish with the analytic block-diagonal Hessian.
    eye = torch.eye(Tmax * 3, device=dev, dtype=dtype)
    # Pin padded timestep blocks (mask=0) to a unit diagonal so the batched solve
    # stays well-conditioned and padded updates are exactly zero.
    pad_diag = (1.0 - mask).repeat_interleave(3, dim=1)         # (B, 3Tmax)
    Hp = _prior_hessian(mask, sigma_a)                          # constant (B,3T,3T)

    @torch.no_grad()
    def grad_hess(Xc):
        return _analytic_hessian_and_grad(
            glm, session_name, Xc, history_term, y, mask, Hp, eye, pad_diag,
            damping)

    # Newton with a per-flight backtracking line search: guarantees a monotone
    # decrease of each flight's objective, so an overshooting full step (e.g.
    # near a sharp place-field ridge) can never blow the trajectory up.
    with torch.no_grad():
        nlp_cur = batched_nlp(X, y, history_term, mask)         # (B,)
    for _ in range(newton_iters):
        g, H = grad_hess(X)
        try:
            delta = torch.linalg.solve(H, g.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            break
        delta = delta.reshape(B, Tmax, 3)
        alpha = torch.ones(B, device=dev, dtype=dtype)
        X_best, nlp_best = X.clone(), nlp_cur.clone()
        for _bt in range(8):
            with torch.no_grad():
                Xn = X - alpha[:, None, None] * delta
                nlpn = batched_nlp(Xn, y, history_term, mask)
            better = nlpn < nlp_best
            X_best = torch.where(better[:, None, None], Xn, X_best)
            nlp_best = torch.where(better, nlpn, nlp_best)
            if (nlp_best < nlp_cur).all():
                break
            alpha = torch.where(nlpn < nlp_cur, alpha, alpha * 0.5)
        X, nlp_cur = X_best, nlp_best

    # ---- Laplace marginal covariances + final per-flight nlp.
    with torch.no_grad():
        nlp = batched_nlp(X, y, history_term, mask)             # (B,)
        _, H = grad_hess(X)
    try:
        Sigma = torch.linalg.inv(H).reshape(B, Tmax, 3, Tmax, 3)
        marginal_cov = torch.stack(
            [Sigma[:, t, :, t, :] for t in range(Tmax)], dim=1)  # (B,Tmax,3,3)
    except RuntimeError:
        marginal_cov = torch.full((B, Tmax, 3, 3), float('nan'),
                                  device=dev, dtype=dtype)
    return X.detach(), marginal_cov, nlp, spread


class GLM_MAP_Decoder:
    """Correlation-aware position decoder: MAP inversion of a ``GLM_Coupled``.

    The encoder is fixed (loaded from a checkpoint); ``fit`` only estimates the
    acceleration-prior scale ``sigma_a`` from training trajectories. ``predict``
    MAP-decodes each flight from a warm start (typically the NB+continuity
    decoder's output) with a few restarts and returns positions in meters.

    Parameters
    ----------
    glm : GLM_Coupled
        Fitted encoder, already on ``device`` and in eval mode. Must include the
        ``session_name`` readout and the temporal filters.
    session_name : str
        Session key for the per-session readout / filter weights.
    behavior_mean, behavior_std : array-like
        Behavior z-score stats; the first three entries are the x/y/z position
        mean/std used to map between meters and the GLM's z-scored coordinates.
    n_restarts : int
        Number of warm starts per flight (the first is the bare warm start; the
        rest are jittered).
    restart_jitter : float
        SD of Gaussian jitter added to the warm start (z-scored units).
    """

    def __init__(self, glm, session_name, behavior_mean, behavior_std,
                 n_restarts=4, restart_jitter=0.05, device='cpu',
                 lbfgs_iters=25, newton_iters=3):
        self.glm = glm
        self.session_name = session_name
        self.device = device
        self.n_restarts = n_restarts
        self.restart_jitter = restart_jitter
        self.lbfgs_iters = lbfgs_iters
        self.newton_iters = newton_iters
        # The encoder is fixed: freeze its parameters so the MAP optimizer only
        # differentiates w.r.t. the trajectory X (and so the precomputed
        # history term carries no stale graph back to encoder params).
        for p in glm.parameters():
            p.requires_grad_(False)
        bm = np.asarray(behavior_mean).squeeze()
        bs = np.asarray(behavior_std).squeeze()
        self.pos_mean = torch.as_tensor(
            bm[:3], dtype=torch.float32, device=device)
        self.pos_std = torch.as_tensor(
            bs[:3], dtype=torch.float32, device=device)
        self.place_field_fn = glm_place_field_fn(glm, session_name)
        self.sigma_a = None

    def _to_z(self, pos_m):
        """meters (..., 3) -> z-scored (torch, on device)."""
        p = torch.as_tensor(pos_m, dtype=torch.float32, device=self.device)
        return (p - self.pos_mean) / self.pos_std

    def _to_meters(self, X_z):
        return (X_z * self.pos_std + self.pos_mean)

    def fit(self, spikes, positions, flight_id, t_idx, dt=None):
        """Estimate ``sigma_a`` from training-flight 2nd-differences (z-scored).

        ``dt`` is accepted for interface symmetry with the NB decoders but is
        unused — the discrete acceleration scale absorbs the bin width.
        """
        fid = np.asarray(flight_id).astype(int)
        tid = np.asarray(t_idx).astype(int)
        pos_z = ((np.asarray(positions, dtype=np.float64)
                  - self.pos_mean.cpu().numpy())
                 / self.pos_std.cpu().numpy())
        sq_sum = 0.0
        n_interior = 0
        for f in np.unique(fid):
            sel = np.where(fid == f)[0]
            sel = sel[np.argsort(tid[sel])]
            if sel.size < 3:
                continue
            p = pos_z[sel]
            d2 = p[2:] - 2.0 * p[1:-1] + p[:-2]          # (T-2, 3)
            sq_sum += float((d2 ** 2).sum())
            n_interior += d2.shape[0]
        if n_interior == 0:
            raise ValueError("No flight with >= 3 timesteps to fit sigma_a.")
        # sigma_a^2 = (1 / (3 * n_interior)) * sum ||d2||^2  (3 spatial dims)
        self.sigma_a = float(np.sqrt(sq_sum / (3.0 * n_interior)))
        return self

    def predict(self, spikes, flight_id, t_idx, warm_start,
                batch_by_session=True, batch_size=32):
        """MAP-decode every flight; return positions in meters (row-aligned).

        Parameters
        ----------
        spikes : np.ndarray
            ``(Nrows, N)`` spike counts.
        flight_id, t_idx : np.ndarray
            ``(Nrows,)`` flight id and within-flight bin index (row-aligned).
        warm_start : np.ndarray
            ``(Nrows, 3)`` warm-start positions in meters (e.g. NB+continuity).
        batch_by_session : bool
            If True (default), decode flights in padded batches of ``batch_size``
            (one L-BFGS + one vmap-Hessian Newton pass per chunk) instead of one
            flight at a time — much faster on GPU, where the per-flight loop
            leaves the device underutilized. The two paths give the same
            objective; the batched L-BFGS shares a line-search step across the
            chunk (see :func:`decode_session_batch`).
        batch_size : int
            Flights per padded chunk. Lower it if the vmap-Hessian intermediate
            (~``chunk x 3Tmax x T x N``) runs out of memory on a big session.

        Returns
        -------
        positions_m : np.ndarray
            ``(Nrows, 3)`` decoded positions in meters.
        info : dict
            ``marginal_cov`` (dict flight_id -> (T,3,3) np.ndarray, z-scored),
            ``spread`` (dict flight_id -> float), ``sigma_a``.
        """
        if self.sigma_a is None:
            raise RuntimeError("Call fit() before predict().")
        fid = np.asarray(flight_id).astype(int)
        tid = np.asarray(t_idx).astype(int)
        decoded_m = np.zeros((spikes.shape[0], 3), dtype=np.float64)
        info = {'marginal_cov': {}, 'spread': {}, 'sigma_a': self.sigma_a}
        # Per-flight row order (sorted by t_idx), reused by both paths.
        order = {int(f): np.where(fid == f)[0][np.argsort(tid[fid == f])]
                 for f in np.unique(fid)}

        if not batch_by_session:
            for f, rows in order.items():
                y = torch.as_tensor(spikes[rows], dtype=torch.float32,
                                    device=self.device)          # (T, N)
                history_term = glm_history_term(
                    self.glm, self.session_name, y.unsqueeze(0))[0]  # (T, N)
                X0 = self._to_z(warm_start[rows])                # (T, 3) z-scored
                X_inits = [X0] + [
                    X0 + self.restart_jitter * torch.randn_like(X0)
                    for _ in range(self.n_restarts - 1)]
                X_hat, cov, _, spread = decode_with_restarts(
                    y, history_term, self.place_field_fn, self.sigma_a, X_inits,
                    lbfgs_iters=self.lbfgs_iters, newton_iters=self.newton_iters)
                decoded_m[rows] = self._to_meters(X_hat).cpu().numpy()
                info['marginal_cov'][int(f)] = cov.cpu().numpy()
                info['spread'][int(f)] = spread
            return decoded_m, info

        # ---- Batched-by-session path: pad flights into chunks.
        flights = list(order.keys())
        for c0 in range(0, len(flights), batch_size):
            chunk = flights[c0:c0 + batch_size]
            self._decode_chunk(chunk, order, spikes, warm_start,
                               decoded_m, info)
        return decoded_m, info

    def _decode_chunk(self, chunk, order, spikes, warm_start, decoded_m, info):
        """Decode one padded batch of flights and scatter into ``decoded_m``."""
        lengths = [len(order[f]) for f in chunk]
        Tmax = max(lengths)
        B, N = len(chunk), spikes.shape[1]
        dev = self.device
        y = torch.zeros(B, Tmax, N, device=dev)
        mask = torch.zeros(B, Tmax, device=dev)
        X0 = torch.zeros(B, Tmax, 3, device=dev)
        for bi, f in enumerate(chunk):
            rows, T = order[f], lengths[bi]
            y[bi, :T] = torch.as_tensor(spikes[rows], dtype=torch.float32,
                                        device=dev)
            mask[bi, :T] = 1.0
            X0[bi, :T] = self._to_z(warm_start[rows])
        history_term = glm_history_term(
            self.glm, self.session_name, y)                 # (B, Tmax, N)
        X_inits = [X0] + [
            X0 + self.restart_jitter * torch.randn_like(X0) * mask.unsqueeze(-1)
            for _ in range(self.n_restarts - 1)]

        X_hat, cov, _, spread = decode_session_batch(
            self.glm, self.session_name, y, history_term, mask, self.sigma_a,
            X_inits, lbfgs_iters=self.lbfgs_iters, newton_iters=self.newton_iters)
        X_hat_m = self._to_meters(X_hat).cpu().numpy()
        cov_np, spread_np = cov.cpu().numpy(), spread.cpu().numpy()
        for bi, f in enumerate(chunk):
            rows, T = order[f], lengths[bi]
            if T < 3:
                # No acceleration prior; mirror the per-flight fallback by
                # keeping the warm start with undefined covariance.
                decoded_m[rows] = warm_start[rows]
                info['marginal_cov'][f] = np.full((T, 3, 3), np.nan)
            else:
                decoded_m[rows] = X_hat_m[bi, :T]
                info['marginal_cov'][f] = cov_np[bi, :T]
            info['spread'][f] = float(spread_np[bi])
