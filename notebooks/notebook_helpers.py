"""Shared constants and helpers for preprocess-data-v0j-share.ipynb.

Reward/perch target locations, distance thresholds, spatial-bin widths, and
target-proximity helpers used across the non-flight Mantel, rate-map,
place-field, and Jacobian-magnification analyses.

Also hosts the shared **Mantel / partial-Mantel** tests (Spearman-based,
NaN-robust, plus directed-predictor variants) and the **MRM** (multiple
regression on distance matrices) helpers — Freedman-Lane permutation inference
and the commonality variance partition — so the across-environment notebooks
can import them instead of redefining them inline.
"""
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import rankdata
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram


# ===========================================================================
# Mantel / partial-Mantel tests (Spearman rho on matrix upper triangles)
# ===========================================================================
def _perm_pvalue(r_obs, perm_rs):
    """One-sided upper-tail permutation p, robust to non-finite values.

    A degenerate (constant / over-NaN) target or control makes the observed
    Spearman / partial-Spearman undefined; without a guard ``perm_rs >= nan``
    is all-False and yields the spuriously small floor p = 1/(n_perms+1). We
    return NaN when ``r_obs`` is undefined so such sessions don't masquerade as
    significant, and drop any non-finite null draws from the tail count."""
    if not np.isfinite(r_obs):
        return np.nan
    finite = np.isfinite(perm_rs)
    n = int(finite.sum())
    if n == 0:
        return np.nan
    return (np.sum(perm_rs[finite] >= r_obs) + 1) / (n + 1)


def spearman(x, y):
    """Spearman rho = Pearson on ranks, dropping pairs where either value is NaN
    (constant rate-map / FR rows yield NaN correlations)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan
    rx, ry = rankdata(x[m]), rankdata(y[m])
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return np.corrcoef(rx, ry)[0, 1]


def partial_r(a, b, c):
    """Partial Spearman of a and b given c (Pearson partial-correlation formula
    applied to Spearman rhos = partial Pearson on ranks)."""
    r_ab, r_ac, r_bc = spearman(a, b), spearman(a, c), spearman(b, c)
    denom = np.sqrt(max(1 - r_ac ** 2, 0) * max(1 - r_bc ** 2, 0))
    return (r_ab - r_ac * r_bc) / denom if denom > 0 else np.nan


def mantel(a_sym, b_vec, rng, n_perms):
    """Mantel: permute labels of square matrix `a_sym`, Spearman-correlate its
    upper triangle with `b_vec`. Returns (r_obs, p, perm_rs)."""
    N = a_sym.shape[0]
    iu, ju = np.triu_indices(N, k=1)
    r_obs = spearman(a_sym[iu, ju], b_vec)
    perm_rs = np.empty(n_perms)
    for pi in range(n_perms):
        perm = rng.permutation(N)
        perm_rs[pi] = spearman(a_sym[np.ix_(perm, perm)][iu, ju], b_vec)
    p = _perm_pvalue(r_obs, perm_rs)
    return r_obs, p, perm_rs


def partial_mantel(a_sym, b_vec, c_vec, rng, n_perms):
    """Partial Mantel: permute `a_sym`, partial-correlate its upper triangle with
    `b_vec` given `c_vec`. Returns (r_obs, p, perm_rs)."""
    N = a_sym.shape[0]
    iu, ju = np.triu_indices(N, k=1)
    r_obs = partial_r(a_sym[iu, ju], b_vec, c_vec)
    perm_rs = np.empty(n_perms)
    for pi in range(n_perms):
        perm = rng.permutation(N)
        perm_rs[pi] = partial_r(a_sym[np.ix_(perm, perm)][iu, ju], b_vec, c_vec)
    p = _perm_pvalue(r_obs, perm_rs)
    return r_obs, p, perm_rs


def dir_both(M):
    """Stack both off-diagonal directions of a (sub)matrix: [M[i,j]; M[j,i]]."""
    N = M.shape[0]
    iu, ju = np.triu_indices(N, k=1)
    return np.concatenate([M[iu, ju], M[ju, iu]])


def mantel_directed_pred(target_sym, K_dir, rng, n_perms):
    """Mantel: symmetric target (permuted) vs a DIRECTED predictor matrix K_dir
    (both off-diagonal directions stacked; reduces to `mantel` when K is symmetric)."""
    pred = dir_both(K_dir)

    def tvec(M):
        t = M[np.triu_indices(M.shape[0], k=1)]
        return np.concatenate([t, t])

    r_obs = spearman(tvec(target_sym), pred)
    N = target_sym.shape[0]
    perm_rs = np.empty(n_perms)
    for pi in range(n_perms):
        perm = rng.permutation(N)
        perm_rs[pi] = spearman(tvec(target_sym[np.ix_(perm, perm)]), pred)
    p = _perm_pvalue(r_obs, perm_rs)
    return r_obs, p, perm_rs


def partial_mantel_directed(K_dir, b_sym_vec, c_sym_vec, rng, n_perms):
    """Partial Mantel: DIRECTED predictor K_dir (permuted), target b given control c
    (both symmetric upper-triangle vectors, duplicated to match K_dir's directions)."""
    b2 = np.concatenate([b_sym_vec, b_sym_vec])
    c2 = np.concatenate([c_sym_vec, c_sym_vec])
    r_obs = partial_r(dir_both(K_dir), b2, c2)
    N = K_dir.shape[0]
    perm_rs = np.empty(n_perms)
    for pi in range(n_perms):
        perm = rng.permutation(N)
        perm_rs[pi] = partial_r(dir_both(K_dir[np.ix_(perm, perm)]), b2, c2)
    p = _perm_pvalue(r_obs, perm_rs)
    return r_obs, p, perm_rs


def partial_mantel_dir_control(pred_sym, b_sym_vec, K_dir_control, rng, n_perms):
    """Partial Mantel: symmetric predictor (permuted), target b, controlling for a
    DIRECTED matrix K_dir_control (both directions). pred/target duplicated."""
    b2 = np.concatenate([b_sym_vec, b_sym_vec])
    c = dir_both(K_dir_control)

    def pvec(M):
        t = M[np.triu_indices(M.shape[0], k=1)]
        return np.concatenate([t, t])

    r_obs = partial_r(pvec(pred_sym), b2, c)
    N = pred_sym.shape[0]
    perm_rs = np.empty(n_perms)
    for pi in range(n_perms):
        perm = rng.permutation(N)
        perm_rs[pi] = partial_r(pvec(pred_sym[np.ix_(perm, perm)]), b2, c)
    p = _perm_pvalue(r_obs, perm_rs)
    return r_obs, p, perm_rs


# ===========================================================================
# MRM (multiple regression on distance matrices): Freedman-Lane + partition
# ===========================================================================
def zrank(v):
    """Rank-transform then standardize (mean 0, sd 1) — makes the MRM the
    multiple-regression generalization of Spearman and puts every column on a
    common scale for variance partitioning."""
    r = rankdata(np.asarray(v, dtype=float)).astype(float)
    r -= r.mean()
    s = r.std()
    return r / s if s > 0 else r


def ols_r2(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def ols_tstat(y, X, j, XtX_inv=None):
    """t-statistic for coefficient j of OLS y ~ X."""
    n, p = X.shape
    if XtX_inv is None:
        XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    sigma2 = float(resid @ resid) / (n - p)
    se = np.sqrt(sigma2 * np.diag(XtX_inv))
    return (beta[j] / se[j]) if se[j] > 0 else 0.0


def freedman_lane(y, X_full, j, rng, n_perm, n_units=None):
    """Freedman-Lane test for column j of X_full. Reduced model = all OTHER
    columns; permute the reduced-model residuals, add back the reduced fit, refit
    the full model, recompute t for column j. Returns (t_obs, p_two_sided, perm_t).

    The rows of y / X_full are the N(N-1)/2 upper-triangle entries (np.triu_indices
    order) of per-session N x N matrices. When n_units=N is given each permutation
    is a single Mantel-style neuron relabeling applied jointly to the rows and
    columns of the reduced-model residual matrix (preserving the matrix dependence
    structure). n_units=None falls back to independent pair shuffling."""
    n, p = X_full.shape
    Z = X_full[:, [k for k in range(p) if k != j]]
    beta_z, *_ = np.linalg.lstsq(Z, y, rcond=None)
    yhat_z = Z @ beta_z
    e_z = y - yhat_z
    XtX_inv = np.linalg.pinv(X_full.T @ X_full)
    t_obs = ols_tstat(y, X_full, j, XtX_inv)
    H = XtX_inv @ X_full.T
    dof = n - p
    djj = XtX_inv[j, j]
    if n_units is not None:
        E_z = squareform(e_z)
        iu, ju = np.triu_indices(n_units, k=1)
    perm_t = np.empty(n_perm)
    for i in range(n_perm):
        if n_units is not None:
            perm = rng.permutation(n_units)
            estar = E_z[np.ix_(perm, perm)][iu, ju]
        else:
            estar = e_z[rng.permutation(n)]
        ystar = yhat_z + estar
        beta = H @ ystar
        resid = ystar - X_full @ beta
        sigma2 = float(resid @ resid) / dof
        se_j = np.sqrt(sigma2 * djj)
        perm_t[i] = (beta[j] / se_j) if se_j > 0 else 0.0
    p_val = (np.sum(np.abs(perm_t) >= abs(t_obs)) + 1) / (n_perm + 1)
    return t_obs, p_val, perm_t


def ols_r2_cols(Y, X):
    """R^2 of every column of Y (n, m) against a shared design X (n, p) — the
    vectorized counterpart of :func:`ols_r2`, used to score whole permutation
    blocks with a single lstsq."""
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    ss_res = ((Y - X @ beta) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    return np.where(ss_tot > 0, 1 - ss_res / np.where(ss_tot > 0, ss_tot, 1.0), np.nan)


def freedman_lane_r2(y, X_full, j, rng, n_perm, n_units):
    """Freedman-Lane test for column j of X_full with the SEMIPARTIAL R^2 as the
    statistic: unique R^2 = R^2(full) - R^2(reduced), reduced = all other columns.

    Identical null to :func:`freedman_lane` — the reduced-model residuals are
    reshaped into an n_units x n_units matrix and relabeled jointly across rows and
    columns, so the exchangeable unit is the neuron and the pair-level dependence is
    preserved — but it scores unique R^2 rather than the coefficient t, so the P
    attaches directly to the quantity MRM figures plot. Requires every upper-triangle
    pair to be finite (rows of y / X_full in np.triu_indices order).

    Returns (r2_obs, p_one_sided, perm_r2). One-sided because unique R^2 >= 0.
    """
    Z = X_full[:, [k for k in range(X_full.shape[1]) if k != j]]
    beta_z, *_ = np.linalg.lstsq(Z, y, rcond=None)
    yhat_z = Z @ beta_z
    E_z = squareform(y - yhat_z)
    iu, ju = np.triu_indices(n_units, k=1)
    ystar = yhat_z[:, None] + np.stack(
        [E_z[np.ix_(p, p)][iu, ju] for p in (rng.permutation(n_units) for _ in range(n_perm))], 1)
    r2_obs = ols_r2(y, X_full) - ols_r2(y, Z)
    perm_r2 = ols_r2_cols(ystar, X_full) - ols_r2_cols(ystar, Z)
    p_val = (np.sum(perm_r2 >= r2_obs) + 1) / (n_perm + 1)
    return r2_obs, float(p_val), perm_r2


def build_nuisance(probe, depth, lateral, fr, iu, ju):
    """Nuisance columns (no intercept): same-probe indicator, within-probe
    (mean-centered) negative distance, mean-FR similarity (-|FR diff|). Distance
    is undefined across probes -> 0 for cross-probe pairs, with the indicator
    absorbing the offset. Zero-variance columns are dropped. Returns (cols, names)."""
    same_probe = (probe[iu] == probe[ju]).astype(float)
    negd = -squareform(pdist(np.column_stack([depth, lateral])))[iu, ju]
    dist_within = np.where(same_probe > 0, negd, 0.0)
    if (same_probe > 0).any():
        dist_within = np.where(same_probe > 0,
                               dist_within - dist_within[same_probe > 0].mean(), 0.0)
    fr_sim = -np.abs(fr[iu] - fr[ju])
    cols, names = [], []
    for nm, col in [('same probe', same_probe), ('within-probe dist', dist_within),
                    ('FR sim', fr_sim)]:
        if np.nanstd(col) > 0:
            cols.append(zrank(col))
            names.append(nm)
    return cols, names


def run_mrm_session(target_corr, interest_mats, probe, depth, lateral, fr,
                    rng, n_perm, min_pairs=30):
    """One MRM for a (session, target). interest_mats: dict name -> (N,N) matrix.
    Returns (records, partition) or (None, None) if degenerate / too few pairs."""
    N = target_corr.shape[0]
    if N < 4:
        return None, None
    iu, ju = np.triu_indices(N, k=1)
    if len(iu) < min_pairs:
        return None, None
    y = zrank(target_corr[iu, ju])
    interest = {nm: zrank(M[iu, ju]) for nm, M in interest_mats.items()}
    nuis_cols, _ = build_nuisance(probe, depth, lateral, fr, iu, ju)
    X_nuis = np.column_stack([np.ones(len(iu))] + nuis_cols)
    names = list(interest)
    X_full = np.column_stack([X_nuis] + [interest[k] for k in names])
    n_nuis = X_nuis.shape[1]

    r2_nuis = ols_r2(y, X_nuis)
    r2_full = ols_r2(y, X_full)
    total = r2_full - r2_nuis
    unique = {}
    for idx, k in enumerate(names):
        keep_cols = list(range(n_nuis)) + [n_nuis + m for m in range(len(names)) if m != idx]
        unique[k] = r2_full - ols_r2(y, X_full[:, keep_cols])
    shared = total - sum(unique.values())

    records = []
    for idx, k in enumerate(names):
        t_obs, p_val, perms = freedman_lane(y, X_full, n_nuis + idx, rng, n_perm, n_units=N)
        records.append(dict(predictor=k, t=t_obs, p=p_val,
                            unique_r2=unique[k], n_pairs=len(iu), n_neurons=N,
                            perm_t=perms))
    partition = dict(r2_nuis=r2_nuis, r2_full=r2_full, total=total, shared=shared,
                     n_pairs=len(iu), n_neurons=N,
                     **{f'unique_{k}': v for k, v in unique.items()})
    return records, partition


# Spatial bin width (meters) for 2D rate maps and binned analyses.
loc_bin_width_rm = 0.3

# Time-warp bin count for place-field analysis.
N_BINS_PF = 50

# Per-dataset reward/perch locations. Dataset 2 uses a different setup with
# both targets on the -x side; Datasets 1 and 3 share the original layout.
_FLIGHT_ROOM = dict(
    rew_locs=np.array([
        [2.70, -0.86, 1.57],
        [2.50,  0.92, 1.57],
    ]),
    perch_locs=np.array([
        [-2.55, 1.57, 1.82],
        [-1.98, 2.08, 1.74],
        [2.46, 1.61, 1.91],
        [1.89, -2.03, 1.94],
        [1.26, -1.92, 1.89],
        [2.37, -1.82, 2.02]
    ]),
)

# Physical bounds of the shared indoor flight room (Datasets 1 & 3), taken from the
# per-session Extracted_Behavior `r_lim` (identical across these sessions). The
# analysis frame `grid_lim=3.0` (see `plot_*` helpers) is a slightly padded version.
INDOOR_ROOM_RLIM = np.array([[-2.9, 2.9],    # x (m)
                             [-2.6, 2.6],    # y (m)
                             [0.0, 2.3]])    # z (m, floor -> ceiling)
INDOOR_ROOM_DIMS = tuple(np.round(np.diff(INDOOR_ROOM_RLIM, axis=1).ravel(), 3))  # (5.8, 5.2, 2.3) m (x, y, z)

DATASET_TARGETS = {
    'Dataset_1': _FLIGHT_ROOM,
    'Dataset_2': dict(
        # Raw ciholas frame (x no longer negated in the loader).
        rew_locs=np.array([[-3.00, 0.70, 1.90]]),
        perch_locs=np.array([
            [-3.20, 1.80, 2.20],
            [4.81, -2.25, 2.27],
            [5.42, -2.15, 1.63]
        ]),
    ),
    'Dataset_3': _FLIGHT_ROOM,
}

# Krishna bat-11719 has two rooms with distinct layouts (measured from the flight
# paths), so its targets are resolved per room rather than per dataset prefix:
# the field station (_fs sessions) and the indoor lab (solo / social sessions).
KRISHNA_FS_TARGETS = dict(
    rew_locs=np.array([[4.33, -0.12, 2.00]]),
    perch_locs=np.array([
        [-3.64, -1.67, 2.80],
        [-3.31, 0.75, 1.71]]),
)
KRISHNA_LAB_TARGETS = dict(
    rew_locs=np.array([
        [2.78, -0.94, 1.61],
        [2.75,  0.90, 1.66],
    ]),
    perch_locs=np.array([
        [-2.66, -1.15, 2.15],
        [-2.45,  2.28, 2.21],
        [-2.78,  1.88, 1.90],
        [ 2.10,  2.34, 2.21],
        [ 2.65,  2.05, 2.10],
        [-2.74, 1.24, 2.01]
    ]),
)

# Bat 14753 has its own layout, measured from its flights: the reward feeder is
# relocated in each room, and the outdoor field station gains a third perch.
# Perches otherwise match the 11719 rooms (the indoor-lab perches are identical;
# the field station reuses 11719's two and adds one).
KRISHNA_14753_FS_TARGETS = dict(
    rew_locs=np.array([
        [5.00,  0.64, 1.30],
        [4.43, -1.38, 1.10],
    ]),
    perch_locs=np.vstack([KRISHNA_FS_TARGETS['perch_locs'],
                          [-3.25, -0.93, 2.81]]),
)
KRISHNA_14753_LAB_TARGETS = dict(
    rew_locs=np.array([
        [2.20,  0.00, 1.30],
        [2.48, -0.21, 2.14],
    ]),
    perch_locs=KRISHNA_LAB_TARGETS['perch_locs'],
)

DIST_THRESH = 0.6  # meters

# Flat D1/3 reference constants for plots that still use the original markers.
REW_LOCS = DATASET_TARGETS['Dataset_1']['rew_locs']
PERCH_LOCS = DATASET_TARGETS['Dataset_1']['perch_locs']


def get_target_locs(session_name):
    """Return (rew_locs, perch_locs) for the dataset the session belongs to.

    Krishna sessions are resolved by room (field station for ``_fs`` sessions,
    indoor lab otherwise) since the two rooms have different reward/perch layouts,
    and by bat (14753 has its own per-room layout; other bats use the 11719 rooms).
    """
    prefix = '_'.join(session_name.split('_')[:2])
    if prefix == 'Dataset_Krishna':
        bat = session_name.split('_')[2]
        is_fs = '_fs' in session_name
        if bat == '14753':
            t = KRISHNA_14753_FS_TARGETS if is_fs else KRISHNA_14753_LAB_TARGETS
        else:
            t = KRISHNA_FS_TARGETS if is_fs else KRISHNA_LAB_TARGETS
        return t['rew_locs'], t['perch_locs']
    if prefix not in DATASET_TARGETS:
        raise KeyError(f'No target locations for dataset prefix {prefix!r}')
    t = DATASET_TARGETS[prefix]
    return t['rew_locs'], t['perch_locs']


def compute_target_proximity(positions, session_name=None,
                             rew_locs=None, perch_locs=None,
                             dist_thresh=DIST_THRESH):
    """For (N, 3) positions, return distance-to-nearest and at-location masks
    for reward and perch.

    If session_name is given, the dataset-specific locations are used;
    otherwise explicit rew_locs/perch_locs are used (falling back to the
    D1/3 layout).
    """
    if session_name is not None:
        rew_locs, perch_locs = get_target_locs(session_name)
    if rew_locs is None:
        rew_locs = REW_LOCS
    if perch_locs is None:
        perch_locs = PERCH_LOCS
    pos = np.asarray(positions)
    dist_rew = np.min(np.linalg.norm(pos[:, None, :] - rew_locs[None, :, :], axis=-1), axis=1)
    dist_perch = np.min(np.linalg.norm(pos[:, None, :] - perch_locs[None, :, :], axis=-1), axis=1)
    return dict(
        dist_rew=dist_rew,
        dist_perch=dist_perch,
        at_rew=dist_rew < dist_thresh,
        at_perch=dist_perch < dist_thresh,
    )


def plot_spatial_field(df, value_col='err_3d', value_label='3-D error (m)',
                       pos_cols=('x_true', 'y_true', 'z_true'), agg='median',
                       cmap='magma', grid_lim=3.0, grid_bin=0.3, min_count=5,
                       n_dist_bins=12, shared_room_prefixes=('Dataset_1', 'Dataset_3'),
                       indoor_filter=None, suptitle=None):
    """Three-panel spatial summary of a per-timepoint quantity (decoding error,
    Jacobian magnification, ...) over test timepoints:

      (1) x-y heat map (`agg` per spatial bin) over the **shared indoor flight room**
          (Datasets 1 & 3 — the only sessions sharing a common ``[-grid_lim, grid_lim]``
          frame and reward/perch layout); reward (*) and perch (^) markers overlaid.
      (2) value vs distance-to-nearest-reward — equal-count binned median + IQR.
      (3) value vs distance-to-nearest-perch  — equal-count binned median + IQR.

    ``indoor_filter(session) -> bool`` selects the sessions for ALL panels and defaults to
    the **shared flight room (Datasets 1 & 3)** — the same sessions as the x-y heat map — so
    the distance curves are directly comparable across notebooks. (The outdoor field station
    is ``Dataset_2`` and the Krishna ``_fs`` sessions, per the ``session_room`` mapping:
    ``_fs`` -> krishna_fs, ``Dataset_2`` -> dataset_2, else lab. Pass a custom
    ``indoor_filter`` — e.g. the full ``lab`` room — to pool a broader set.)

    ``df`` needs columns: ``session``, the three ``pos_cols``, and ``value_col``.
    Returns ``(fig, df_indoor_with_distances)``.
    """
    from scipy.stats import binned_statistic_2d
    px, py, _pz = pos_cols
    if indoor_filter is None:
        indoor_filter = lambda s: s.startswith(tuple(shared_room_prefixes))

    df = df[df['session'].map(indoor_filter)].copy()
    df['dist_rew'] = np.nan
    df['dist_perch'] = np.nan
    for sess, idx in df.groupby('session').groups.items():
        prox = compute_target_proximity(df.loc[idx, list(pos_cols)].to_numpy(),
                                        session_name=sess)
        df.loc[idx, 'dist_rew'] = prox['dist_rew']
        df.loc[idx, 'dist_perch'] = prox['dist_perch']

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # (1) x-y heat map over the shared flight room (Datasets 1 & 3)
    room = df[df['session'].str.startswith(shared_room_prefixes)]
    edges = np.arange(-grid_lim, grid_lim + grid_bin, grid_bin)
    if len(room):
        stat, _, _, _ = binned_statistic_2d(room[px], room[py], room[value_col],
                                            statistic=agg, bins=[edges, edges])
        cnt, _, _, _ = binned_statistic_2d(room[px], room[py], room[value_col],
                                           statistic='count', bins=[edges, edges])
        stat[cnt < min_count] = np.nan
        vmin, vmax = np.nanpercentile(stat, [2.5, 97.5])
        pcm = axes[0].pcolormesh(edges, edges, stat.T, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(pcm, ax=axes[0], label=f'{agg} {value_label}')
        axes[0].scatter(REW_LOCS[:, 0], REW_LOCS[:, 1], marker='*', s=170, c='cyan',
                        edgecolor='k', linewidth=0.6, label='reward', zorder=5)
        axes[0].scatter(PERCH_LOCS[:, 0], PERCH_LOCS[:, 1], marker='^', s=70, c='lime',
                        edgecolor='k', linewidth=0.6, label='perch', zorder=5)
        axes[0].legend(fontsize=7, loc='upper right')
    axes[0].set(xlabel='x (m)', ylabel='y (m)', xlim=(-grid_lim, grid_lim),
                ylim=(-grid_lim, grid_lim), title='x-y map (Datasets 1 & 3)')
    axes[0].set_aspect('equal', 'box')

    # (2, 3) value vs distance, equal-count binned median + IQR (indoor sessions)
    for ax, dcol, dlbl in [(axes[1], 'dist_rew', 'nearest reward'),
                           (axes[2], 'dist_perch', 'nearest perch')]:
        d = df[[dcol, value_col]].dropna()
        x, y = d[dcol].to_numpy(), d[value_col].to_numpy()
        q = np.unique(np.quantile(x, np.linspace(0, 1, n_dist_bins + 1)))
        bi = np.clip(np.digitize(x, q) - 1, 0, len(q) - 2)
        cen, med, lo, hi = [], [], [], []
        for k in range(len(q) - 1):
            sel = bi == k
            if sel.sum() < min_count:
                continue
            cen.append(np.median(x[sel])); med.append(np.median(y[sel]))
            lo.append(np.percentile(y[sel], 25)); hi.append(np.percentile(y[sel], 75))
        ax.fill_between(cen, lo, hi, color='tab:blue', alpha=0.2, label='IQR')
        ax.plot(cen, med, 'o-', color='tab:blue', label='binned median')
        ax.set(xlabel=f'distance to {dlbl} (m)', ylabel=value_label,
               title=f'{value_label} vs distance to {dlbl}')
        ax.legend(fontsize=7)

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig, df


def plot_value_vs_occupancy(panels, occ_source, occ_pos_cols=('x_true', 'y_true'),
                            grid_lim=3.0, grid_bin=0.3, min_count=5, agg='median',
                            shared_room_prefixes=('Dataset_1', 'Dataset_3'), suptitle=None):
    """Per spatial bin (shared indoor room, `[-grid_lim, grid_lim]` x-y grid), relate a
    quantity (decoding error, Jacobian magnification, ...) to OCCUPANCY = test-sample
    count per bin. Mirrors the occupancy section of preprocess-data-v0j-share.ipynb.

    panels: list of ``(df, value_col, value_label, pos_cols)``. Each df is binned into
    the grid, `agg`-aggregated per bin, and scattered against the per-bin occupancy
    (computed once from `occ_source`'s `occ_pos_cols`), with a binned trend (mean ± s.e.m.)
    and Pearson r over bins with >= `min_count` samples. Returns the figure.
    """
    from scipy.stats import binned_statistic_2d
    edges = np.arange(-grid_lim, grid_lim + grid_bin, grid_bin)

    op = occ_source[occ_source['session'].str.startswith(shared_room_prefixes)]
    occ, _, _ = np.histogram2d(op[occ_pos_cols[0]], op[occ_pos_cols[1]], bins=[edges, edges])

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 4.8), squeeze=False)
    for ax, (df, vcol, vlabel, pcols) in zip(axes[0], panels):
        sub = df[df['session'].str.startswith(shared_room_prefixes)]
        vstat, _, _, _ = binned_statistic_2d(sub[pcols[0]], sub[pcols[1]], sub[vcol],
                                             statistic=agg, bins=[edges, edges])
        vcnt, _, _, _ = binned_statistic_2d(sub[pcols[0]], sub[pcols[1]], sub[vcol],
                                            statistic='count', bins=[edges, edges])
        valid = (occ >= min_count) & (vcnt >= min_count) & np.isfinite(vstat)
        x, y = occ[valid], vstat[valid]
        ax.scatter(x, y, s=14, alpha=0.4, color='0.5', rasterized=True)
        if len(x) > 2:
            be = np.linspace(x.min(), x.max(), 11)
            bi = np.clip(np.digitize(x, be) - 1, 0, len(be) - 2)
            tx, ty, te = [], [], []
            for k in range(len(be) - 1):
                sel = bi == k
                if sel.sum() < min_count:
                    continue
                tx.append(np.median(x[sel])); ty.append(y[sel].mean())
                te.append(y[sel].std() / np.sqrt(sel.sum()))
            ax.errorbar(tx, ty, yerr=te, fmt='o-', color='firebrick', capsize=3, zorder=5)
            r = np.corrcoef(x, y)[0, 1]
            ax.annotate(f'r = {r:.3f}  (n={len(x)} bins)', xy=(0.04, 0.96),
                        xycoords='axes fraction', va='top', fontsize=11)
        ax.set(xlabel='occupancy (test samples / bin)', ylabel=f'{agg} {vlabel}',
               title=f'{vlabel} vs occupancy')
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def _add_flight_speed(df, pos_cols, dt, group_cols=('session', 'flight_id'),
                      time_col='t_idx'):
    """Return a copy of `df` with a 'speed' column (m/s) = |d(position)/dt| per flight.

    Speed is computed via ``np.gradient`` over the timesteps of each flight, ordered by
    `time_col`. Valid flight bins are contiguous, so `time_col` increments uniformly and
    `dt` is the spacing between consecutive rows of one flight (pass ``bin_width * stride``
    when the rows were subsampled). Flights with < 2 timesteps get NaN speed.
    """
    out = df.sort_values([*group_cols, time_col]).copy()
    pos = out[list(pos_cols)].to_numpy(float)
    spd = np.full(len(out), np.nan)
    for rows in out.groupby(list(group_cols), sort=False).indices.values():
        rows = np.sort(rows)
        if len(rows) < 2:
            continue
        spd[rows] = np.linalg.norm(np.gradient(pos[rows], dt, axis=0), axis=1)
    out['speed'] = spd
    return out


def plot_value_vs_speed(panels, shared_room_prefixes=('Dataset_1', 'Dataset_3'),
                        min_count=5, n_bins=16, suptitle=None):
    """Relate a quantity (decode error, Jacobian magnification, ...) to flight SPEED.

    Mirrors the 'magnification vs speed' panel of preprocess-data-v0j-share.ipynb: per
    flight, speed = |d(position)/dt| (`_add_flight_speed`); per panel a per-sample scatter
    on a log speed axis plus a geometric-binned trend (mean +/- s.e.m.), annotated with
    the Pearson r on log(speed) (matches the log x-axis) and the scale-free Spearman rho.

    panels: list of ``(df, value_col, value_label, pos_cols, time_col, dt)``. Restricted
    to `shared_room_prefixes` so curves are comparable across notebooks. Returns
    ``(fig, {value_col: speed-augmented DataFrame})``.
    """
    from scipy.stats import spearmanr
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 4.8), squeeze=False)
    augmented = {}
    for ax, (df, vcol, vlabel, pcols, tcol, dt) in zip(axes[0], panels):
        sub = df[df['session'].str.startswith(shared_room_prefixes)]
        sub = _add_flight_speed(sub, pcols, dt, time_col=tcol).dropna(subset=['speed', vcol])
        augmented[vcol] = sub
        x, y = sub['speed'].to_numpy(float), sub[vcol].to_numpy(float)
        ax.scatter(x, y, s=3, alpha=0.2, color='0.5', rasterized=True)

        lo = max(np.quantile(x, 0.01), 1e-3)
        hi = np.quantile(x, 0.99)
        edges = np.geomspace(lo, hi, n_bins)
        dig = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
        tx, ty, te = [], [], []
        for k in range(len(edges) - 1):
            sel = dig == k
            if sel.sum() < min_count:
                continue
            tx.append(np.sqrt(edges[k] * edges[k + 1]))
            ty.append(y[sel].mean())
            te.append(y[sel].std() / np.sqrt(sel.sum()))
        ax.errorbar(tx, ty, yerr=te, fmt='o-', color='firebrick', capsize=3, zorder=5)

        r_log = np.corrcoef(np.log(np.maximum(x, 1e-6)), y)[0, 1]
        rho, _ = spearmanr(x, y)
        ax.annotate(f'r(log) = {r_log:.3f}\n$\\rho$ = {rho:.3f}  (n={len(x)})',
                    xy=(0.04, 0.96), xycoords='axes fraction', va='top', fontsize=11)
        ax.set_xscale('log')
        ax.set(xlabel='speed (m/s)', ylabel=vlabel, title=f'{vlabel} vs speed')
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig, augmented


# ===========================================================================
# Similarity-matrix plotting: color maps, Mantel boxplots, clustered heatmaps
# ===========================================================================
SUBREGION_COLORS = {'CA1': 'tab:red', 'CA3': 'tab:purple'}
PROBE_COLORS = ['tab:blue', 'tab:orange', 'tab:green']
PROBE_CMAP = ListedColormap(PROBE_COLORS)
SUBREGION_CMAP = ListedColormap([SUBREGION_COLORS['CA1'], SUBREGION_COLORS['CA3']])
SUBREGION_CMAP.set_bad('white')


def plot_mantel_boxplots(mantel_df, comparison_col='comparison', ylabel='Mantel $\\rho$',
                         ncols=4, axes=None, ax=None, palette=None):
    """Plot Mantel test results: grey boxplot of merged null + colored dots by bat.

    Three drawing modes:
      * default (``axes`` and ``ax`` both ``None``): build and return a new
        multi-panel figure, one panel per comparison.
      * ``axes=[...]`` (one Axes per comparison): draw the multi-panel style onto
        a caller-supplied list of axes (shared canvas); returns the parent figure.
      * ``ax=<Axes>``: grouped single-axes mode -- every comparison is one x-tick
        (grey null boxplots + bat-coloured Mantel-r dots), with a session-level
        Wilcoxon-vs-0 star above each. Use this to fold the panel onto one cell of
        a merged gridspec. Returns the parent figure.
    """
    from scipy.stats import wilcoxon
    comparisons = list(mantel_df[comparison_col].unique())
    n = len(comparisons)

    if ax is not None:                                   # grouped single-axes mode
        null_long = pd.concat(
            [pd.DataFrame({comparison_col: c,
                           'null': np.concatenate(mantel_df[mantel_df[comparison_col] == c]['perm_rs'].values)})
             for c in comparisons], ignore_index=True)
        sns.boxplot(data=null_long, x=comparison_col, y='null', order=comparisons, color='0.85',
                    width=0.5, whis=(0.5, 99.5), showfliers=False, ax=ax)
        bats_present = [b for b in (palette or {}) if b in set(mantel_df['bat'])] or None
        sns.stripplot(data=mantel_df, x=comparison_col, y='r', order=comparisons, hue='bat',
                      palette=palette, hue_order=bats_present, size=6, jitter=0.12,
                      edgecolor='k', linewidth=0.4, zorder=5, ax=ax, legend=False)
        ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
        ymax = mantel_df['r'].max()
        for idx, comp in enumerate(comparisons):
            rs = mantel_df.loc[mantel_df[comparison_col] == comp, 'r'].dropna().values
            if len(rs) >= 3 and np.any(rs != 0):
                p = wilcoxon(rs, alternative='greater').pvalue
                star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                if star:
                    ax.text(idx, ymax * 1.05, star, ha='center', va='bottom', fontsize=13)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('')
        ax.set_xticks(range(n))
        ax.set_xticklabels(comparisons, rotation=25, ha='right')
        sns.despine(ax=ax)
        return ax.figure

    if axes is not None:                                 # caller-supplied panel list
        fig = np.ravel(axes)[0].figure
        axgrid = np.ravel(axes)
    else:
        nrows = math.ceil(n / ncols)
        ncols_actual = min(n, ncols)
        fig, axgrid = plt.subplots(nrows, ncols_actual, figsize=(4 * ncols_actual, 4 * nrows),
                                   squeeze=False)
        axgrid = axgrid.ravel()
    for idx, comp in enumerate(comparisons):
        ax_i = axgrid[idx]
        sub = mantel_df[mantel_df[comparison_col] == comp].reset_index(drop=True)
        all_nulls = np.concatenate(sub['perm_rs'].values)
        sns.boxplot(y=all_nulls, color='0.85', width=0.4, whis=(0.5, 99.5),
                    fliersize=0, ax=ax_i)
        sns.stripplot(data=sub, y='r', hue='bat', ax=ax_i, size=8, jitter=0.1,
                      zorder=5, edgecolor='k', linewidth=0.5)
        for _, row in sub.iterrows():
            star = '***' if row['p'] < 0.001 else '**' if row['p'] < 0.01 else '*' if row['p'] < 0.05 else ''
            if star:
                ax_i.text(0.25, row['r'], star, ha='left', va='center', fontsize=8)
        ax_i.set_ylabel(ylabel)
        ax_i.set_title(comp, fontsize=11)
        ax_i.axhline(0, color='k', linewidth=0.5, linestyle='--')
        ax_i.set_xticks([])
        ax_i.legend(title='Bat', fontsize=7, title_fontsize=7)
    for idx in range(n, len(axgrid)):                    # hide unused axes
        axgrid[idx].set_visible(False)
    if axes is None:
        plt.tight_layout()
    return fig


def cluster_within_sessions(clu_df, clustering_matrix):
    """Per-session hierarchical clustering. Returns global reordering index."""
    global_order = []
    for session_name, grp in clu_df.groupby('session_name'):
        idx = np.where(clu_df['session_name'] == session_name)[0]
        N = len(grp)
        block = clustering_matrix[np.ix_(idx, idx)]
        block_clean = np.nan_to_num(block, nan=0.0)
        dist = 1 - np.abs(block_clean)
        np.fill_diagonal(dist, 0)
        if N >= 2:
            dist_condensed = squareform(dist, checks=False)
            Z = linkage(dist_condensed, method='average')
            sess_order = dendrogram(Z, no_plot=True)['leaves']
        else:
            sess_order = list(range(N))
        global_order.extend(idx[sess_order])
    return np.array(global_order)


def plot_similarity_heatmaps(matrices, clu_df, global_order, suptitle='',
                             figsize_per_panel=(6.5, 6), vmin=-1, vmax=1,
                             boundary_col='session_name',
                             axis_label='Neuron (clustered within session)'):
    """
    Plot side-by-side similarity matrix heatmaps with subregion/probe/FR/depth
    gutter strips. `boundary_col` (or None) selects which df column drives the
    internal block-boundary lines.

    Parameters
    ----------
    matrices : list of (matrix, title) tuples. Matrices are (N_total, N_total),
               already reordered by global_order.
    """
    n_panels = len(matrices)
    N_total = len(global_order)
    probe_ids_reordered = clu_df['probe_id'].values[global_order].astype(int)
    mean_fr_reordered = clu_df['mean_fr'].values[global_order]
    depth_reordered = clu_df['neuron_pos_depth'].values[global_order]
    subregion_arr = clu_df['subregion'].values[global_order]
    sub_codes = np.ma.masked_invalid(np.where(subregion_arr == 'CA1', 0.0,
                                              np.where(subregion_arr == 'CA3', 1.0, np.nan)))

    fig, axes = plt.subplots(1, n_panels,
                             figsize=(figsize_per_panel[0] * n_panels, figsize_per_panel[1]))
    if n_panels == 1:
        axes = [axes]

    for ax, (mat, title) in zip(axes, matrices):
        im = ax.imshow(mat, aspect='auto', cmap='RdBu_r',
                       vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_xlabel(axis_label)
        ax.set_ylabel(axis_label)

        # Block boundary lines wherever `boundary_col` value changes in the reordered df
        if boundary_col is not None:
            bvals = clu_df[boundary_col].values[global_order]
            for i in range(1, len(bvals)):
                if bvals[i] != bvals[i - 1]:
                    ax.axhline(i - 0.5, color='k', linewidth=0.5)
                    ax.axvline(i - 0.5, color='k', linewidth=0.5)

        divider = make_axes_locatable(ax)
        # Subregion strip (innermost)
        ax_top_s = divider.append_axes('top', size='2%', pad=0.02)
        ax_top_s.imshow(sub_codes[None, :], aspect='auto', cmap=SUBREGION_CMAP,
                        vmin=0, vmax=1, interpolation='nearest')
        ax_top_s.set_xlim(-0.5, N_total - 0.5)
        ax_top_s.axis('off')

        ax_right_s = divider.append_axes('right', size='2%', pad=0.02)
        ax_right_s.imshow(sub_codes[:, None], aspect='auto', cmap=SUBREGION_CMAP,
                          vmin=0, vmax=1, interpolation='nearest')
        ax_right_s.set_ylim(N_total - 0.5, -0.5)
        ax_right_s.axis('off')

        # Probe strip
        ax_top_p = divider.append_axes('top', size='2%', pad=0.02)
        ax_top_p.imshow(probe_ids_reordered[None, :], aspect='auto',
                        cmap=PROBE_CMAP, vmin=0, vmax=2, interpolation='nearest')
        ax_top_p.set_xlim(-0.5, N_total - 0.5)
        ax_top_p.axis('off')

        ax_right_p = divider.append_axes('right', size='2%', pad=0.02)
        ax_right_p.imshow(probe_ids_reordered[:, None], aspect='auto',
                          cmap=PROBE_CMAP, vmin=0, vmax=2, interpolation='nearest')
        ax_right_p.set_ylim(N_total - 0.5, -0.5)
        ax_right_p.axis('off')

        # Mean FR strip
        ax_top_fr = divider.append_axes('top', size='2%', pad=0.02)
        ax_top_fr.imshow(mean_fr_reordered[None, :], aspect='auto',
                         cmap='viridis', interpolation='nearest')
        ax_top_fr.set_xlim(-0.5, N_total - 0.5)
        ax_top_fr.axis('off')

        ax_right_fr = divider.append_axes('right', size='2%', pad=0.02)
        ax_right_fr.imshow(mean_fr_reordered[:, None], aspect='auto',
                           cmap='viridis', interpolation='nearest')
        ax_right_fr.set_ylim(N_total - 0.5, -0.5)
        ax_right_fr.axis('off')

        # Depth strip (outermost)
        ax_top_d = divider.append_axes('top', size='2%', pad=0.02)
        ax_top_d.imshow(depth_reordered[None, :], aspect='auto',
                        cmap='magma', interpolation='nearest')
        ax_top_d.set_xlim(-0.5, N_total - 0.5)
        ax_top_d.axis('off')
        ax_top_d.set_title(title)

        ax_right_d = divider.append_axes('right', size='2%', pad=0.02)
        ax_right_d.imshow(depth_reordered[:, None], aspect='auto',
                          cmap='magma', interpolation='nearest')
        ax_right_d.set_ylim(N_total - 0.5, -0.5)
        ax_right_d.axis('off')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.08)

    # Legend on last axis
    last_ax = axes[-1]
    for p_id, color in enumerate(PROBE_COLORS):
        last_ax.plot([], [], 's', color=color, label=f'Probe {p_id+1}', markersize=6)
    last_ax.plot([], [], 's', color='gold', label='Mean FR', markersize=6)
    last_ax.plot([], [], 's', color='darkmagenta', label='Depth', markersize=6)
    for sub_label in ('CA1', 'CA3'):
        last_ax.plot([], [], 's', color=SUBREGION_COLORS[sub_label],
                     label=sub_label, markersize=6)
    last_ax.legend(loc='upper right', fontsize=8, framealpha=0.8)

    if suptitle:
        fig.suptitle(suptitle, fontsize=15)
    plt.tight_layout()
    return fig


# ===========================================================================
# Dual-field example cells: raw spikes vs empirical rate vs predicted rate
# ---------------------------------------------------------------------------
# Picks neurons whose place field sits in two *physically different* locations
# on two *different flight clusters*, then draws, per cluster, three views of
# the flight path (3D + x-y projection):
#   (1) raw spikes   -- red dots on a black flight path,
#   (2) empirical    -- path coloured by observed firing rate (counts / dt),
#   (3) predicted    -- path coloured by the model's predicted rate,
# with panels (2) and (3) sharing one colour scale for a fair comparison.
# All inputs come from the **test** dataloaders only.
# ===========================================================================
def build_flight_records(X, y, pred, mask, ftype, behavior_mean, behavior_std):
    """Split padded ``(N, T, *)`` test batches into per-flight records of valid
    timesteps, with denormalised 3D position.

    Returns a list of dicts ``{'xyz': (t, 3), 'obs': (t, n), 'pred': (t, n),
    'cluster': int}`` where ``xyz`` columns are physical x, y, z (meters),
    ``obs`` are observed binned spike counts and ``pred`` the model's predicted
    counts. Only timesteps with ``mask == 1.0`` are kept.
    """
    bm = np.asarray(behavior_mean)[0, 0, :3]
    bs = np.asarray(behavior_std)[0, 0, :3]
    flights = []
    for i in range(X.shape[0]):
        m = mask[i] == 1.0
        if m.sum() < 2:
            continue
        xyz = X[i][m][:, :3] * bs + bm
        flights.append(dict(xyz=xyz, obs=y[i][m], pred=pred[i][m],
                            cluster=int(ftype[i])))
    return flights


def select_dual_field_cells(flights, n_neurons, dt, r_lim,
                            loc_bin_width=0.3, min_flights=4, min_spikes=8,
                            min_peak_rate=2.0, min_field_sep=1.0, n_top=2):
    """Rank neurons by how spatially separated their two strongest place fields
    are across two *different* flight clusters.

    For each neuron and each flight cluster with at least ``min_flights``
    flights, a smoothed 2D (x, y) rate map is built and its peak rate / location
    recorded. A neuron is a dual-field candidate when two clusters each drive a
    peak of at least ``min_peak_rate`` spikes/s whose peak locations lie at least
    ``min_field_sep`` meters apart. Returns up to ``n_top`` candidates sorted by
    peak separation (descending), each a dict with the neuron index, the two
    cluster ids, per-cluster peak rate / location and the separation distance.
    """
    from collections import defaultdict
    from digitalbat.utility.utils import generate_2d_rate_map

    by_cluster = defaultdict(list)
    for f in flights:
        by_cluster[f['cluster']].append(f)
    by_cluster = {c: fl for c, fl in by_cluster.items() if len(fl) >= min_flights}

    loc_xx_lim, loc_yy_lim = r_lim[0], r_lim[1]
    candidates = []
    for neuron in range(n_neurons):
        peaks = {}
        for c, fl in by_cluster.items():
            xx = np.concatenate([f['xyz'][:, 0] for f in fl])
            yy = np.concatenate([f['xyz'][:, 1] for f in fl])
            sc = np.concatenate([f['obs'][:, neuron] for f in fl])
            if sc.sum() < min_spikes:
                continue
            x_edges, y_edges, _, smap = generate_2d_rate_map(
                loc_xx=xx, loc_yy=yy, unit_spike_count=sc,
                loc_xx_lim=loc_xx_lim, loc_yy_lim=loc_yy_lim,
                loc_bin_width=loc_bin_width, dt=dt)
            if not np.isfinite(smap).any():
                continue
            peak_rate = np.nanmax(smap)
            if peak_rate < min_peak_rate:
                continue
            ix, iy = np.unravel_index(np.nanargmax(smap), smap.shape)
            loc = np.array([0.5 * (x_edges[ix] + x_edges[ix + 1]),
                            0.5 * (y_edges[iy] + y_edges[iy + 1])])
            peaks[c] = dict(rate=float(peak_rate), loc=loc, n_flights=len(fl))
        if len(peaks) < 2:
            continue
        cl = list(peaks)
        best = None
        for a in range(len(cl)):
            for b in range(a + 1, len(cl)):
                sep = float(np.linalg.norm(peaks[cl[a]]['loc'] - peaks[cl[b]]['loc']))
                if best is None or sep > best['sep']:
                    best = dict(sep=sep, clusters=(cl[a], cl[b]),
                                peak_rate=min(peaks[cl[a]]['rate'], peaks[cl[b]]['rate']))
        if best['sep'] >= min_field_sep:
            candidates.append(dict(neuron=neuron, sep=best['sep'],
                                   clusters=best['clusters'], peak_rate=best['peak_rate'],
                                   peaks={c: peaks[c] for c in best['clusters']}))
    candidates.sort(key=lambda d: (d['sep'], d['peak_rate']), reverse=True)
    return candidates[:n_top]


def add_3d_scalebar(ax, length=1.0, corner=(0, 0, 0), color='k', lw=2.5,
                    fontsize=15, gap=0.12, unit='m', y_nudge=(0.0, 0.0, 0.0), x_nudge=(0.0, 0.0, 0.0), hide_frame=True,
                    origin_shift=(0.0, 0.0, 0.0), alpha=1.0, z_nudge=(0.0, 0.0, 0.0), label_weight='normal'):
    """Replace a 3-D axis's frame/ticks/labels with an x/y/z scale-bar triad:
    three orthogonal bars of physical ``length`` from a bounding-box corner
    (0/1 per axis = low/high limit), each labelled x/y/z, plus one
    '<length> <unit>' note. Reads the current x/y/z limits, so call after the
    limits and box_aspect are set."""
    import matplotlib.patheffects as pe
    lims = (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())
    spans = np.array([l[1] - l[0] for l in lims], float)
    o = np.array([lims[i][corner[i]] for i in range(3)], float)
    o = o + np.array([(-1 if corner[i] else 1) * 0.05 * spans[i] for i in range(3)])
    o = o + np.array(origin_shift, float) * spans          # shift the whole bar (e.g. down)
    stroke = [pe.withStroke(linewidth=3, foreground='white')]
    for i, lab in enumerate(('x', 'y', 'z')):
        step = np.zeros(3); step[i] = (-1 if corner[i] else 1) * length
        end = o + step
        ax.plot(*zip(o, end), color=color, lw=lw, zorder=10, solid_capstyle='round', clip_on=False, alpha=alpha)
        lp = end + step / length * gap * spans[i]
        if lab == 'y':
            lp = lp + np.array(y_nudge, float) * spans   # nudge y label up/right
        elif lab == 'z':
            lp = lp + np.array(z_nudge, float) * spans   # nudge z label up (clear of the bar)
        elif lab == 'x':
            lp = lp + np.array(x_nudge, float) * spans   # nudge x label toward the bar
        ax.text(*lp, lab, color=color, fontsize=fontsize, ha='center', va='center',
                fontweight=label_weight, zorder=11, path_effects=stroke, clip_on=False, alpha=alpha)
    ax.text(*(o - np.array([0, 0, gap * spans[2]])), f'{length:g} {unit}', color=color,
            fontsize=fontsize - 1, ha='center', va='top', zorder=11, path_effects=stroke, clip_on=False, alpha=alpha)
    if hide_frame:
        ax.set_axis_off()
    else:                                        # keep gridlines; drop spines, ticks, labels
        ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
        ax.tick_params(length=0)                 # no tick marks
        for _a3 in (ax.xaxis, ax.yaxis, ax.zaxis):
            _a3.set_ticklabels([])               # no tick labels (keep ticks -> keep grid)
            if hasattr(_a3, 'line'):
                _a3.line.set_visible(False)      # no axis line (spine)
            _a3.pane.set_edgecolor('none')       # no pane border
            _a3._axinfo['tick']['inward_factor'] = 0   # no tick marks (3-D)
            _a3._axinfo['tick']['outward_factor'] = 0


def plot_dual_field_cell(flights, neuron, clusters, dt, r_lim,
                         title='', peaks=None, rate_cmap='viridis',
                         spike_color='red', axes3d=None, col_title_fs=13, row_label_fs=12,
                         row_labels=None, startend='markers'):
    """Draw the spikes / empirical-rate / predicted-rate triptych for one neuron
    on two flight clusters, in 3D and in x-y projection.

    Produces TWO figures -- one 3D, one x-y projection -- each with
    ``len(clusters)`` rows by 3 columns (raw spikes, empirical rate, predicted
    rate). The empirical and predicted panels share a single colour scale
    (``norm``) so they are directly comparable; raw-spike panels show a black
    flight path with red spike dots (dot area scales with the binned spike
    count). Returns ``(fig_3d, fig_2d)``.
    """
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D

    clusters = list(clusters)
    data = {c: [f for f in flights if f['cluster'] == c] for c in clusters}

    # Shared colour scale across empirical & predicted, both clusters.
    all_rates = []
    for c in clusters:
        for f in data[c]:
            all_rates.append(f['obs'][:, neuron] / dt)
            all_rates.append(f['pred'][:, neuron] / dt)
    all_rates = np.concatenate(all_rates) if all_rates else np.array([0.0])
    vmax = float(np.nanpercentile(all_rates, 99))
    norm = plt.Normalize(0.0, max(vmax, 1e-6))

    xlim, ylim, zlim = r_lim[0], r_lim[1], r_lim[2]
    col_titles = ['Empirical spikes', 'Empirical rate', 'Predicted rate (test set)']

    def _seg_vals(v):
        return 0.5 * (v[:-1] + v[1:])

    START_C, END_C = '#00b050', 'deeppink'   # green start, magenta end (shared by markers/arrows)

    def _draw(ax, fl, mode, three_d, annotate_se=False):
        starts = np.array([f['xyz'][0] for f in fl])
        ends = np.array([f['xyz'][-1] for f in fl])
        if startend == 'markers':
            # start/end markers first so they sit behind the trajectory lines
            for pts, mk, fc in ((starts, '^', '#ff7f0e'), (ends, 's', END_C)):
                if not len(pts):
                    continue
                if three_d:
                    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], marker=mk, s=26, facecolors=fc,
                               edgecolors='k', linewidths=0.6, depthshade=False, zorder=0)
                else:
                    ax.scatter(pts[:, 0], pts[:, 1], marker=mk, s=26, facecolors=fc,
                               edgecolors='k', linewidths=0.6, zorder=0)
        for f in fl:
            xyz = f['xyz']
            if mode == 'spikes':
                if three_d:
                    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color='k', lw=0.5, alpha=0.5)
                else:
                    ax.plot(xyz[:, 0], xyz[:, 1], color='k', lw=0.5, alpha=0.5)
            else:
                vals = (f['obs'][:, neuron] if mode == 'emp' else f['pred'][:, neuron]) / dt
                if three_d:
                    pts = xyz[:, :3].reshape(-1, 1, 3)
                    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                    lc = Line3DCollection(segs, cmap=rate_cmap, norm=norm)
                else:
                    pts = xyz[:, :2].reshape(-1, 1, 2)
                    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                    lc = LineCollection(segs, cmap=rate_cmap, norm=norm)
                lc.set_array(_seg_vals(vals))
                lc.set_linewidth(1.6)
                ax.add_collection(lc)
        if mode == 'spikes':
            xs, ys, zs, ss = [], [], [], []
            for f in fl:
                hit = f['obs'][:, neuron] > 0
                if not hit.any():
                    continue
                xs.append(f['xyz'][hit, 0]); ys.append(f['xyz'][hit, 1])
                zs.append(f['xyz'][hit, 2]); ss.append(f['obs'][hit, neuron])
            if xs:
                xs = np.concatenate(xs); ys = np.concatenate(ys)
                zs = np.concatenate(zs); ss = 12 * np.concatenate(ss)
                if three_d:
                    ax.scatter(xs, ys, zs, s=ss, c=spike_color, alpha=0.75,
                               edgecolors='none', depthshade=False)
                else:
                    ax.scatter(xs, ys, s=ss, c=spike_color, alpha=0.75, edgecolors='none')
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        if three_d:
            ax.set_zlim(zlim)
            ax.set_box_aspect([max(lim[1] - lim[0], 1e-3) for lim in (xlim, ylim, zlim)])  # 1 m equal on x/y/z
            ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
            ax.view_init(elev=22, azim=-68)
        else:
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel('x (m)')
        if startend == 'arrows' and annotate_se:
            _draw_startend_arrows(ax, starts, ends, three_d)

    def _draw_startend_arrows(ax, starts, ends, three_d):
        """Black arrow + text label at the (mean) start / end of the flights,
        instead of a per-flight marker. Start is nudged one way and end the other
        so their labels don't collide; when the two mean points coincide (e.g. a
        loop that returns to its origin) a single 'start/end' label is drawn."""
        import matplotlib.patheffects as pe
        _stroke = [pe.withStroke(linewidth=2.5, foreground='white')]
        span = np.array([xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]], float)
        if not len(starts) or not len(ends):
            return
        cs = np.asarray(starts, float).mean(0)
        ce = np.asarray(ends, float).mean(0)
        if np.linalg.norm(cs - ce) < 0.6:                       # coincide (~ one target radius)
            items = [((cs + ce) / 2, 'start/end', 0)]
        else:
            items = [(cs, 'start', -1), (ce, 'end', +1)]
        for c, label, sgn in items:
            if three_d:
                off = np.array([sgn * 0.18 * span[0], 0.0, 0.22 * span[2]])
                base = c + off
                ax.quiver(base[0], base[1], base[2], -off[0], -off[1], -off[2], color='k',
                          lw=1.6, arrow_length_ratio=0.4, zorder=6, clip_on=False)
                ax.text(base[0], base[1], base[2], label, color='k', fontsize=row_label_fs,
                        ha='center', va='bottom', fontweight='bold', zorder=7, path_effects=_stroke,
                        clip_on=False)
            else:
                base = (c[0] + sgn * 0.28 * span[0], c[1] + 0.34 * span[1])
                ax.annotate(label, xy=(c[0], c[1]), xytext=base, color='k', ha='center',
                            va='bottom', fontsize=row_label_fs, fontweight='bold',
                            path_effects=_stroke,
                            arrowprops=dict(arrowstyle='-|>', color='k', lw=1.6))

    def _populate(axes, three_d):
        for ci, c in enumerate(clusters):
            row_label = row_labels[ci] if row_labels is not None else f'cluster {c}'
            for col, mode in enumerate(('spikes', 'emp', 'pred')):
                ax = axes[ci, col]
                _draw(ax, data[c], mode, three_d, annotate_se=(col == 0))
                if ci == 0:
                    ax.set_title(col_titles[col], fontsize=col_title_fs)
                if col == 0:
                    if three_d:
                        ax.text2D(-0.10, 0.5, row_label, transform=ax.transAxes,
                                  rotation=90, va='center', ha='center', fontsize=row_label_fs)
                    else:
                        ax.set_ylabel(row_label, fontsize=row_label_fs)
        if startend == 'markers':
            legend_handles = [
                Line2D([], [], marker='^', color='w', markerfacecolor='#ff7f0e',
                       markeredgecolor='k', markersize=7, label='start'),
                Line2D([], [], marker='s', color='w', markerfacecolor=END_C,
                       markeredgecolor='k', markersize=7, label='end'),
            ]
            axes[0, 0].legend(handles=legend_handles, loc='upper right', fontsize=8,
                              framealpha=0.85)

    sm = ScalarMappable(norm=norm, cmap=rate_cmap)
    sm.set_array([])

    # Shared-canvas mode: draw only the 3-D triptych onto caller-provided axes
    # (a (len(clusters), 3) grid of 3-D axes); return the mappable for a colourbar.
    if axes3d is not None:
        _populate(np.asarray(axes3d, dtype=object).reshape(len(clusters), 3), True)
        return sm

    figs = []
    for three_d in (True, False):
        fig, axes = plt.subplots(
            len(clusters), 3, figsize=(11, 3.6 * len(clusters)), squeeze=False,
            subplot_kw=dict(projection='3d') if three_d else None)
        _populate(axes, three_d)
        fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.015, pad=0.02,
                     label='firing rate (spikes/s)')
        view = '3D' if three_d else 'x-y'
        fig.suptitle(f'{title}  [{view}]' if title else view, fontsize=14)
        figs.append(fig)
    return tuple(figs)


# --- Joint encode/decode latent conventions (shared by Fig 2 & ED Fig 5) ---
def masked_latent(x, T, kind):
    """Pre-read-out latent of the MASKED (predicted) modality for one joint-model pass.

    `x` is the full transformer output ``(B, 2T, D)``; the joint model lays tokens
    out as ``cat([neural(0:T), behaviour(T:2T)])``. In an encode pass the neural tokens
    are masked and predicted from behaviour, so the encode latent is the neural half
    ``x[:, :T]`` (what the spike read-out acts on); in a decode pass the behaviour
    tokens are masked and predicted from neural, so the decode latent is the behaviour
    half ``x[:, T:]`` (what the behaviour read-out acts on). Each pass's latent is the
    representation of the modality being reconstructed.
    """
    if kind == 'encode':
        return x[:, :T]
    if kind == 'decode':
        return x[:, T:]
    raise ValueError(f"kind must be 'encode' or 'decode', got {kind!r}")


def fit_shared_latent_pca(latent_encode, latent_decode, n_components=3, random_state=None):
    """Fit ONE PCA on the row-concatenation of the encode and decode latents.

    Returns the fitted sklearn ``PCA`` so both branches share a single principal-
    component basis: call ``.transform`` on either branch to project into the shared
    space. Fig 2 and ED Fig 5 use this on their own (different) data subsets, so the
    concatenation-PCA functionality is identical even though the inputs differ.
    """
    from sklearn.decomposition import PCA
    X = np.vstack([latent_encode, latent_decode])
    return PCA(n_components=n_components, random_state=random_state).fit(X)


# --- Interactive Plotly HTML with a live matplotlib-style (elev, azim) readout ---
# Plotly's 3-D scenes rotate on click-drag, but their orientation is a camera
# `eye` vector, not the (elev, azim) that matplotlib's Axes3D.view_init wants.
# This overlay converts every scene's camera to (elev, azim) and updates it live
# as you drag, so you can dial in a view (e.g. the Fig-2 "figure-eight") in the
# browser and read off the exact numbers to paste into view_init.
_VIEW_READOUT_JS = r"""
var gd = document.getElementById('{plot_id}');
var box = document.createElement('div');
box.style.cssText = 'position:fixed;top:8px;right:8px;z-index:9999;'
  + 'background:rgba(255,255,255,0.92);padding:6px 10px;font:12px/1.45 monospace;'
  + 'color:#111;border:1px solid #666;border-radius:4px;white-space:pre;pointer-events:none;';
document.body.appendChild(box);
function _ea(eye) {
  var r = Math.hypot(eye.x, eye.y, eye.z);
  return {e: Math.asin(eye.z / r) * 180 / Math.PI,
          a: Math.atan2(eye.y, eye.x) * 180 / Math.PI};
}
var _last = null;
function _render() {
  var L = gd._fullLayout || gd.layout || {};
  var lines = [];
  Object.keys(L).forEach(function(k) {
    if (!/^scene[0-9]*$/.test(k)) return;
    var cam = L[k] && L[k].camera;
    if (!cam || !cam.eye) return;
    var v = _ea(cam.eye);
    lines.push((k === _last ? '▶ ' : '  ') + k
               + ': elev=' + v.e.toFixed(1) + ' azim=' + v.a.toFixed(1));
  });
  box.textContent = 'view_init (elev, azim)\n' + (lines.join('\n') || '(drag a 3-D panel)');
}
gd.on('plotly_relayout', function(ed) {
  Object.keys(ed).forEach(function(k) {
    var m = k.match(/^(scene[0-9]*)\./);
    if (m) _last = m[1];
  });
  _render();
});
_render();
"""


def write_html_with_view_readout(fig, out_path, **kwargs):
    """`fig.write_html` plus a drag-live matplotlib (elev, azim) overlay.

    Drop-in for ``fig.write_html(out_path)`` on any Plotly 3-D figure. The saved
    HTML shows a small corner box reporting each scene's current view as the
    (elev, azim) you would pass to ``Axes3D.view_init`` — updated on every
    click-drag — so an orientation found in the browser transfers exactly to a
    matplotlib figure. Extra ``kwargs`` pass through to ``fig.write_html``.
    """
    kwargs.setdefault('post_script', _VIEW_READOUT_JS)
    fig.write_html(out_path, **kwargs)
