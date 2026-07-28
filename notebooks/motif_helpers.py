"""Dynamical-motif segmentation of Twin latent trajectories.

Two interchangeable segmenters over batches of variable-length sequences:

1. `MaskedGaussianHMM` / `MaskedDiagonalGaussianHMM` -- dynamax HMMs with a
   sticky (diagonal-boosted Dirichlet) prior on the transition rows, patched to
   handle padded, variable-length trials *exactly*. Gives soft posteriors, a
   proper held-out marginal likelihood for model selection, and vmapped
   (GPU-ready) batching over trials.

2. `potts_kmeans` -- the hard-assignment limit: Lloyd's algorithm with the
   assignment step replaced by a Viterbi/DP sweep over the label sequence,

       min_{mu, z}  sum_i sum_t ||x_it - mu_{z_it}||^2
                    + lam * sum_t 1[z_it != z_i,t+1]

   with an optional minimum dwell time encoded directly in the DP state.

The two are the same objective up to the emission model: for a spherical HMM
with shared covariance sigma^2 I and a transition matrix with `p_stay` on the
diagonal and `p_switch` off it, Viterbi minimises the Potts objective with

       lam = 2 * sigma^2 * log(p_stay / p_switch)          [see `lambda_from_hmm`]

Padding convention: sequences are stacked into `(N, T_max, D)` with a companion
`(N, T_max)` float mask (1 = valid). Padded emission rows must be finite (zeros
are used by `stack_sequences`); their likelihood contribution is forced to zero.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Sequence

import numpy as np

_DYNAMAX = os.path.expanduser('~/Documents/code/linderman/dynamax')
if os.path.isdir(_DYNAMAX) and _DYNAMAX not in sys.path:
    sys.path.insert(0, _DYNAMAX)

import jax
import jax.numpy as jnp
import jax.random as jr

from dynamax.hidden_markov_model import DiagonalGaussianHMM, GaussianHMM
from dynamax.hidden_markov_model.inference import hmm_two_filter_smoother


# --------------------------------------------------------------------------
# padding / stacking
# --------------------------------------------------------------------------
def stack_sequences(seqs: Sequence[np.ndarray]):
    """Stack ragged `(T_i, D)` sequences into `(N, T_max, D)` + `(N, T_max)` mask."""
    n = len(seqs)
    t_max = max(len(s) for s in seqs)
    d = seqs[0].shape[1]
    out = np.zeros((n, t_max, d), dtype=np.float64)
    mask = np.zeros((n, t_max), dtype=np.float64)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
        mask[i, :len(s)] = 1.0
    return out, mask


def unstack(arr: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    """Inverse of `stack_sequences` for a `(N, T_max, ...)` array."""
    lengths = mask.sum(1).astype(int)
    return [np.asarray(arr[i, :lengths[i]]) for i in range(len(lengths))]


# --------------------------------------------------------------------------
# masked (variable-length) HMMs
# --------------------------------------------------------------------------
class _MaskedEStep:
    """Exact E-step for zero-padded batches.

    dynamax vmaps `e_step` over `(emissions, inputs)`, and every downstream
    sufficient statistic is *linear* in the posterior marginals. So it suffices
    to (a) zero the per-state emission log-likelihoods at padded timesteps --
    which makes both the filtered/smoothed marginals at valid steps and
    `marginal_loglik` exactly those of the unpadded sequence -- and (b) zero the
    posterior marginals and transition pair-marginals over the padded tail
    before collecting statistics. The mask rides along in `inputs[:, 0]`.
    """

    @property
    def inputs_shape(self):
        """One input channel: the validity mask."""
        return (1,)

    def e_step(self, params, emissions, inputs=None):
        """Masked forward-backward + sufficient statistics."""
        num_timesteps = emissions.shape[0]
        mask = jnp.ones(num_timesteps) if inputs is None else inputs[:, 0]

        initial_probs = self.initial_component._compute_initial_probs(params.initial, None)
        trans_mat = self.transition_component._compute_transition_matrices(params.transitions, None)
        lls = self.emission_component._compute_conditional_logliks(params.emissions, emissions, None)
        # Padded steps become uninformative (likelihood 1 under every state).
        lls = jnp.where(mask[:, None] > 0, lls, 0.0)

        # Force the time-varying code path so `trans_probs` stays resolved per
        # timestep -- otherwise dynamax hands back the sum and we cannot mask it.
        trans_mat_t = jnp.broadcast_to(trans_mat, (num_timesteps - 1,) + trans_mat.shape)
        post = hmm_two_filter_smoother(initial_probs, trans_mat_t, lls)

        pair_mask = mask[:-1] * mask[1:]
        transition_stats = jnp.einsum('tij,t->ij', post.trans_probs, pair_mask)

        masked_post = post._replace(smoothed_probs=post.smoothed_probs * mask[:, None])
        initial_stats = self.initial_component.collect_suff_stats(params.initial, masked_post, None)
        emission_stats = self.emission_component.collect_suff_stats(
            params.emissions, masked_post, emissions, None)
        return (initial_stats, transition_stats, emission_stats), post.marginal_loglik

    # -- masked convenience wrappers -------------------------------------
    def _masked_args(self, params, emissions, mask):
        """Inference arguments with padded steps made uninformative."""
        lls = self.emission_component._compute_conditional_logliks(params.emissions, emissions, None)
        lls = jnp.where(mask[:, None] > 0, lls, 0.0)
        return (self.initial_component._compute_initial_probs(params.initial, None),
                self.transition_component._compute_transition_matrices(params.transitions, None),
                lls)

    def batch_marginal_log_prob(self, params, emissions, mask):
        """Per-sequence marginal log likelihood, exact under padding."""
        from dynamax.hidden_markov_model.inference import hmm_filter
        fn = lambda y, m: hmm_filter(*self._masked_args(params, y, m)).marginal_loglik
        return np.asarray(jax.vmap(fn)(jnp.asarray(emissions), jnp.asarray(mask)))

    def batch_most_likely_states(self, params, emissions, mask):
        """Viterbi path per sequence; padded steps are meaningless (masked out)."""
        from dynamax.hidden_markov_model.inference import hmm_posterior_mode
        fn = lambda y, m: hmm_posterior_mode(*self._masked_args(params, y, m))
        return np.asarray(jax.vmap(fn)(jnp.asarray(emissions), jnp.asarray(mask)))

    def batch_smoothed_probs(self, params, emissions, mask):
        """Posterior state marginals per sequence, `(N, T, K)`."""
        from dynamax.hidden_markov_model.inference import hmm_smoother
        fn = lambda y, m: hmm_smoother(*self._masked_args(params, y, m)).smoothed_probs
        return np.asarray(jax.vmap(fn)(jnp.asarray(emissions), jnp.asarray(mask)))


class MaskedGaussianHMM(_MaskedEStep, GaussianHMM):
    """Full-covariance Gaussian HMM over zero-padded, variable-length trials."""


class MaskedDiagonalGaussianHMM(_MaskedEStep, DiagonalGaussianHMM):
    """Diagonal-covariance Gaussian HMM over zero-padded, variable-length trials."""


def fit_sticky_hmm(emissions, mask, num_states, *, stickiness=100.0, concentration=1.1,
                   diagonal=False, num_iters=60, num_restarts=3, seed=0, verbose=False,
                   emission_prior_scale=1e-1, emission_prior_extra_df=1.0):
    """Fit a sticky Gaussian HMM by EM, keeping the best of `num_restarts`.

    `stickiness` is the extra Dirichlet pseudo-count added to the diagonal of
    every transition row (dynamax's `transition_matrix_stickiness`); it is the
    knob that buys temporally contiguous motifs. Returns `(hmm, params, lps)`
    where `lps` is the log joint over EM iterations of the winning restart.
    """
    emissions = jnp.asarray(emissions)
    mask = jnp.asarray(mask)
    inputs = mask[..., None]
    dim = emissions.shape[-1]
    kw = dict(transition_matrix_stickiness=stickiness,
              transition_matrix_concentration=concentration)
    if diagonal:
        # DiagonalGaussianHMM parameterises its emission prior differently.
        hmm = MaskedDiagonalGaussianHMM(num_states, dim, **kw)
    else:
        hmm = MaskedGaussianHMM(num_states, dim, emission_prior_scale=emission_prior_scale,
                                emission_prior_extra_df=emission_prior_extra_df, **kw)

    # k-means init on *valid* points only (dynamax's built-in would include pads).
    from sklearn.cluster import KMeans
    valid = np.asarray(emissions)[np.asarray(mask) > 0]
    best = None
    for r in range(num_restarts):
        km = KMeans(num_states, n_init=5, random_state=seed + r).fit(valid)
        means = jnp.asarray(km.cluster_centers_)
        params, props = hmm.initialize(jr.PRNGKey(seed + r), method='prior',
                                       emission_means=means)
        params, lps = hmm.fit_em(params, props, emissions, inputs=inputs,
                                 num_iters=num_iters, verbose=verbose)
        if best is None or lps[-1] > best[1][-1]:
            best = (params, lps)
    return hmm, best[0], np.asarray(best[1])


def lambda_from_hmm(params, *, num_states=None):
    """Potts switching penalty `lam` implied by a fitted HMM's transition matrix.

    Uses the mean self-transition vs. mean off-diagonal probability and the mean
    emission variance, per `lam = 2 sigma^2 log(p_stay / p_switch)`. This is the
    exact correspondence only for a shared spherical covariance, so treat it as
    the scale on which the two knobs live rather than an identity.
    """
    A = np.asarray(params.transitions.transition_matrix)
    k = A.shape[0] if num_states is None else num_states
    p_stay = np.mean(np.diag(A))
    p_switch = (1.0 - p_stay) / (k - 1)
    covs = np.asarray(params.emissions.covs)
    sigma2 = np.mean(np.diagonal(covs, axis1=-2, axis2=-1)) if covs.ndim == 3 else np.mean(covs)
    return float(2 * sigma2 * np.log(p_stay / max(p_switch, 1e-12)))


# --------------------------------------------------------------------------
# Potts-regularised k-means (hard assignment, explicit lambda + min dwell)
# --------------------------------------------------------------------------
def potts_viterbi(cost, mask, lam, min_dwell=1):
    """Exact DP over label sequences for the Potts objective.

    `cost` is `(N, T, K)` of per-timestep assignment costs (squared distance to
    each centroid), `mask` is `(N, T)`. Minimises
    `sum_t cost[t, z_t] + lam * #switches` subject to every run lasting at least
    `min_dwell` steps. The dwell constraint is carried in the DP state, so this
    is still exact -- `O(T K min_dwell)` with the min-over-all-states trick.
    Each sequence terminates at its own last valid step, and the final run is
    allowed to be truncated by that boundary (only interior runs are forced to
    reach `min_dwell`).
    """
    cost = np.asarray(cost, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    n, t_max, k = cost.shape
    m = max(int(min_dwell), 1)
    big = np.float64(1e18)
    lengths = mask.sum(1).astype(int)
    rows = np.arange(n)
    states = np.arange(k)[None, :]

    # V[n, k, c]: best cost of a path ending at t in state k, having dwelt c+1
    # steps there (bucket c == m-1 saturates and means "free to leave").
    V = np.full((n, k, m), big)
    V[:, :, 0] = cost[:, 0, :]
    # back-pointers: previous (state, dwell bucket) for each (t, seq, state, bucket)
    bp_state = np.zeros((t_max, n, k, m), dtype=np.int16)
    bp_dwell = np.zeros((t_max, n, k, m), dtype=np.int8)
    # snapshot of V at each sequence's own terminal step
    V_end = np.where(lengths[:, None, None] == 1, V, big)

    for t in range(1, t_max):
        # --- switching in: legal only out of the saturated bucket -----------
        leavable = V[:, :, m - 1]                                    # (n, k)
        order = np.argsort(leavable, axis=1, kind='stable')[:, :2] if k > 1 \
            else np.zeros((n, 1), dtype=int)
        best_j = order[:, 0]
        best_v = leavable[rows, best_j]
        if k > 1:
            second_j, second_v = order[:, 1], leavable[rows, order[:, 1]]
        else:                       # single state: switching is impossible
            second_j, second_v = best_j, np.full(n, big)
        is_self = states == best_j[:, None]
        switch_v = np.where(is_self, second_v[:, None], best_v[:, None]) + lam
        switch_j = np.where(is_self, second_j[:, None], best_j[:, None])

        newV = np.full((n, k, m), big)
        newV[:, :, 0] = switch_v
        bp_state[t, :, :, 0] = switch_j
        bp_dwell[t, :, :, 0] = m - 1

        # --- staying: bucket c from c-1, with c == m-1 absorbing ------------
        for c in range(1, m):
            if c < m - 1:
                newV[:, :, c] = V[:, :, c - 1]
                bp_dwell[t, :, :, c] = c - 1
            else:
                prev = V[:, :, m - 2:m]
                j = np.argmin(prev, axis=2)
                newV[:, :, c] = np.take_along_axis(prev, j[:, :, None], 2)[:, :, 0]
                bp_dwell[t, :, :, c] = (m - 2 + j).astype(np.int8)
            bp_state[t, :, :, c] = states
        if m == 1:
            # no dwell constraint: staying lands back in the single bucket
            stay = V[:, :, 0]
            took_switch = switch_v < stay
            newV[:, :, 0] = np.where(took_switch, switch_v, stay)
            bp_state[t, :, :, 0] = np.where(took_switch, switch_j, states)
            bp_dwell[t, :, :, 0] = 0

        newV += cost[:, t, :, None]
        V = newV
        ends = lengths == t + 1
        if ends.any():
            V_end[ends] = V[ends]

    # --- backtrace, each sequence from its own last valid step --------------
    z = np.full((n, t_max), -1, dtype=np.int16)
    total = np.zeros(n)
    for i in range(n):
        ti = int(lengths[i]) - 1
        if ti < 0:
            continue
        ks, cs = np.unravel_index(int(np.argmin(V_end[i])), (k, m))
        total[i] = V_end[i, ks, cs]
        for t in range(ti, -1, -1):
            z[i, t] = ks
            if t == 0:
                break
            ks, cs = int(bp_state[t, i, ks, cs]), int(bp_dwell[t, i, ks, cs])
    return z, total


def potts_kmeans(emissions, mask, num_states, lam, *, min_dwell=1, num_iters=30,
                 num_restarts=3, seed=0, tol=1e-7):
    """Lloyd's algorithm with a Viterbi/DP assignment step (Potts k-means).

    Returns `(labels, centroids, objective)`; `labels` is `(N, T_max)` with -1
    at padded steps.
    """
    X = np.asarray(emissions, dtype=np.float64)
    mask = np.asarray(mask)
    valid = mask > 0
    flat = X[valid]
    best = None
    for r in range(num_restarts):
        rng = np.random.default_rng(seed + r)
        mu = flat[rng.choice(len(flat), num_states, replace=False)].copy()
        prev = np.inf
        for _ in range(num_iters):
            # (N, T, K) squared distances
            cost = ((X[:, :, None, :] - mu[None, None, :, :]) ** 2).sum(-1)
            z, per_seq = potts_viterbi(cost, mask, lam, min_dwell=min_dwell)
            obj = per_seq.sum()
            lab = z[valid]
            for kk in range(num_states):
                sel = lab == kk
                if sel.any():
                    mu[kk] = flat[sel].mean(0)
            if prev - obj < tol * max(abs(prev), 1.0):
                break
            prev = obj
        if best is None or obj < best[2]:
            best = (z, mu.copy(), obj)
    return best


# --------------------------------------------------------------------------
# motif-sequence descriptors
# --------------------------------------------------------------------------
def run_length_encode(path: np.ndarray) -> list[tuple[int, int]]:
    """`[(state, duration), ...]` for one Viterbi path (padding already trimmed)."""
    path = np.asarray(path)
    if len(path) == 0:
        return []
    change = np.flatnonzero(np.diff(path)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(path)]
    return [(int(path[s]), int(e - s)) for s, e in zip(starts, ends)]


def motif_string(path: np.ndarray) -> tuple[int, ...]:
    """Order of motifs visited, durations collapsed."""
    return tuple(s for s, _ in run_length_encode(path))


def dwell_times(paths: Sequence[np.ndarray], num_states: int) -> dict[int, np.ndarray]:
    """Pooled run durations per motif."""
    out: dict[int, list[int]] = {k: [] for k in range(num_states)}
    for p in paths:
        for s, d in run_length_encode(p):
            out[s].append(d)
    return {k: np.asarray(v) for k, v in out.items()}


def normalized_edit_distance(a: Sequence[int], b: Sequence[int]) -> float:
    """Levenshtein distance between two motif orders, / max(len)."""
    a, b = list(a), list(b)
    if not a and not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(len(a), len(b))


def transition_counts(paths: Sequence[np.ndarray], num_states: int,
                      *, exclude_self=True) -> np.ndarray:
    """Empirical motif-to-motif transition counts from Viterbi paths."""
    C = np.zeros((num_states, num_states))
    for p in paths:
        p = np.asarray(p)
        for a, b in zip(p[:-1], p[1:]):
            if exclude_self and a == b:
                continue
            C[a, b] += 1
    return C


def grouped_sequence_distance(motifs: Sequence[Sequence[int]], groups: np.ndarray,
                              *, n_perm: int = 1000, seed: int = 0, blocks=None):
    """Within- vs between-group motif-sequence similarity with a permutation null.

    `motifs[i]` is the motif order of flight `i` (see `motif_string`), `groups[i]`
    its label (e.g. route or trajectory cluster). Returns a dict with the mean
    normalised edit distance for within- and between-group pairs, their
    difference, and a two-sided p-value from shuffling group labels. If `blocks`
    is given (e.g. session), only pairs inside a block are used and the shuffle
    is done within block -- which keeps session identity from manufacturing the
    effect.
    """
    rng = np.random.default_rng(seed)
    n = len(motifs)
    groups = np.asarray(groups)
    blocks = np.zeros(n, int) if blocks is None else np.asarray(blocks)

    iu, ju = np.triu_indices(n, 1)
    keep = blocks[iu] == blocks[ju]
    iu, ju = iu[keep], ju[keep]
    d = np.array([normalized_edit_distance(motifs[a], motifs[b]) for a, b in zip(iu, ju)])

    def stat(g):
        same = g[iu] == g[ju]
        if same.sum() == 0 or (~same).sum() == 0:
            return np.nan, np.nan, np.nan
        return d[same].mean(), d[~same].mean(), d[~same].mean() - d[same].mean()

    within, between, obs = stat(groups)
    null = np.empty(n_perm)
    for p in range(n_perm):
        g = groups.copy()
        for b in np.unique(blocks):                 # shuffle within block
            m = blocks == b
            g[m] = rng.permutation(g[m])
        null[p] = stat(g)[2]
    pval = (1 + np.sum(np.abs(null) >= abs(obs))) / (n_perm + 1)
    return dict(within=within, between=between, diff=obs, p=pval,
                null_mean=float(np.nanmean(null)), n_pairs=len(d))


def state_profile(paths: Sequence[np.ndarray], values: Sequence[np.ndarray],
                  num_states: int, *, agg=np.mean):
    """Per-motif aggregate of a per-timestep quantity (e.g. position, speed).

    `values[i]` is `(T_i, ...)` aligned with `paths[i]`. Returns `(K, ...)`.
    """
    flat_z = np.concatenate([np.asarray(p) for p in paths])
    flat_v = np.concatenate([np.asarray(v) for v in values])
    return np.stack([agg(flat_v[flat_z == k], axis=0) if (flat_z == k).any()
                     else np.full(flat_v.shape[1:], np.nan) for k in range(num_states)])


def match_states(ref: np.ndarray, other: np.ndarray, num_states: int):
    """Hungarian relabelling of `other` onto `ref` (both flat label vectors)."""
    from scipy.optimize import linear_sum_assignment
    C = np.zeros((num_states, num_states))
    np.add.at(C, (ref.astype(int), other.astype(int)), 1)
    r, c = linear_sum_assignment(-C)
    lut = np.zeros(num_states, dtype=int)
    lut[c] = r
    return lut, C[r, c].sum() / max(len(ref), 1)
