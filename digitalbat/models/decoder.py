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
):
    """Train `Bat_Transformer_Decoder` to predict z-scored position from spikes.

    The convention `batch.behavior[:, :, :pos_dims]` are the z-scored target
    positions (set by `bat_data_loader`'s `behavior_mean`/`behavior_std`).
    Loss and validation are masked: padded timesteps are dropped before the
    loss is computed.

    `valid_fn` should return a per-sample-or-per-dim score where LARGER is
    better (so MSE callers should negate). Defaults to negative-MSE per dim.
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
