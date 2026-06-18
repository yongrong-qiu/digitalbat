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
                        xycoords='axes fraction', va='top', fontsize=9)
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
                    xy=(0.04, 0.96), xycoords='axes fraction', va='top', fontsize=9)
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


def plot_mantel_boxplots(mantel_df, comparison_col='comparison', ylabel='Mantel $\\rho$', ncols=4):
    """Plot Mantel test results: grey boxplot of merged null + colored dots by bat."""
    comparisons = mantel_df[comparison_col].unique()
    n = len(comparisons)
    nrows = math.ceil(n / ncols)
    ncols_actual = min(n, ncols)
    fig, axes = plt.subplots(nrows, ncols_actual, figsize=(4 * ncols_actual, 4 * nrows),
                             squeeze=False)
    for idx, comp in enumerate(comparisons):
        ax = axes[idx // ncols_actual, idx % ncols_actual]
        sub = mantel_df[mantel_df[comparison_col] == comp].reset_index(drop=True)
        all_nulls = np.concatenate(sub['perm_rs'].values)
        sns.boxplot(y=all_nulls, color='0.85', width=0.4, whis=(0.5, 99.5),
                    fliersize=0, ax=ax)
        sns.stripplot(data=sub, y='r', hue='bat', ax=ax, size=8, jitter=0.1,
                      zorder=5, edgecolor='k', linewidth=0.5)
        for _, row in sub.iterrows():
            star = '***' if row['p'] < 0.001 else '**' if row['p'] < 0.01 else '*' if row['p'] < 0.05 else ''
            if star:
                ax.text(0.25, row['r'], star, ha='left', va='center', fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(comp, fontsize=9)
        ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
        ax.set_xticks([])
        ax.legend(title='Bat', fontsize=7, title_fontsize=7)
    # Hide unused axes
    for idx in range(n, nrows * ncols_actual):
        axes[idx // ncols_actual, idx % ncols_actual].set_visible(False)
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
        fig.suptitle(suptitle, fontsize=13)
    plt.tight_layout()
    return fig
