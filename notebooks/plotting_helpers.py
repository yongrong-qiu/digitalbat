"""Unified publication-figure styling and plotting helpers for figures.ipynb.

This module centralises the *look* of every panel in the paper so the scattered
analysis notebooks can be consolidated into one figure notebook with a single,
consistent style:

  * one seaborn theme, large legible fonts, constant panel height (``PANEL_H``);
  * one canonical bat colour map (``digitalbat.utility.utils.bat_palette``) and a
    canonical bat / model ordering used for both the x-axis and the hue;
  * paired session-level comparisons drawn as points *connected* by lines (never
    floating lines and points), coloured by bat;
  * 95% CIs (bootstrap or from a supplied null) wherever a summary is shown;
  * significance annotated with 3-tier stars on brackets that **do not** stretch
    the y-axis (drawn in a blended data/axes transform), hiding n.s.;
  * a p-value table printer and a Nature-style markdown legend helper.

Statistics default to *session-level* nonparametric tests (Wilcoxon signed-rank
for paired, Mann-Whitney U for unpaired) per project convention; figures show
per-neuron / per-flight distributions but tests run on per-session aggregates
unless a caller overrides ``unit``.
"""
from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from digitalbat.utility.utils import bat_palette, parse_bat, stars_for_p

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
PANEL_H = 3.0            # standard panel height (inches)
POINT_SIZE = 7.0         # session-level marker size
NULL_CI = (2.5, 97.5)    # percentiles for a 95% CI


def set_pub_style():
    """Apply the unified publication style (call once at notebook top)."""
    sns.set_theme(context='talk', style='ticks')
    mpl.rcParams.update({
        'figure.dpi': 110,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'pdf.fonttype': 42,           # editable text in Illustrator
        'ps.fonttype': 42,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.titlesize': 16,
        'axes.labelsize': 15,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 12,
        'legend.title_fontsize': 12,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.2,
        'lines.linewidth': 2.0,
    })


# ---------------------------------------------------------------------------
# Canonical orderings and palettes
# ---------------------------------------------------------------------------
def bat_order():
    """Canonical bat id ordering (Dataset 1-3 bats, Krishna excluded here)."""
    from digitalbat.datasets.defaults import DEFAULT_SESSION_NAMES
    seen = []
    for s in DEFAULT_SESSION_NAMES:
        b = s.split('_')[2]
        if b not in seen:
            seen.append(b)
    return seen


def bat_pal():
    """`{bat_id: rgb}` restricted to the default (non-Krishna) bats."""
    full = bat_palette()
    return {b: full[b] for b in bat_order()}


def bat_of(session_name):
    """Bat id for a session (``parse_bat`` wrapper)."""
    return parse_bat(session_name)


# Consensus / display names and orderings for the model-comparison figures.
# Fill in / extend once the source-notebook model keys are known; kept here so
# every figure orders and colours models identically.
MODEL_DISPLAY = {}       # raw key -> pretty label
MODEL_ORDER = []         # pretty labels, left-to-right


def order_categorical(series, order):
    """Return `series` as an ordered Categorical using the subset of `order`
    actually present (keeps seaborn x/hue ordering consistent across panels)."""
    present = [o for o in order if o in set(series)]
    return pd.Categorical(series, categories=present, ordered=True)


# ---------------------------------------------------------------------------
# Significance brackets that DO NOT stretch the y-axis
# ---------------------------------------------------------------------------
def sig_bracket(ax, x1, x2, p, *, level=0, color='k', linewidth=1.2,
                fontsize=15, base=0.98, step=0.07, show_ns=False):
    """Draw a 3-tier-star significance bracket between categorical positions
    ``x1`` and ``x2`` **without** changing ``ax`` limits.

    The bracket is drawn in a blended transform: x in data coordinates, y in
    axes-fraction coordinates, so it floats above the data near the top of the
    panel regardless of the data range. ``level`` stacks multiple brackets
    (0 = lowest). Hidden entirely when p >= .05 unless ``show_ns``.
    """
    stars = stars_for_p(p)
    if not stars:
        if not show_ns:
            return
        stars = 'n.s.'
    trans = ax.get_xaxis_transform()  # x data, y axes-fraction
    y = base - level * step
    h = 0.015
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], transform=trans,
            lw=linewidth, color=color, clip_on=False)
    ax.text((x1 + x2) / 2, y + h, stars, transform=trans, ha='center',
            va='bottom', fontsize=fontsize, fontweight='bold', color=color,
            clip_on=False)


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------
def bootstrap_ci(values, *, statistic=np.median, n_boot=10000, ci=95, seed=0):
    """Bootstrap CI of a statistic over a 1-D sample. Returns (lo, point, hi)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = statistic(rng.choice(values, size=(n_boot, values.size), replace=True), axis=1)
    lo, hi = np.percentile(boot, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return lo, statistic(values), hi


def null_ci(perm_values, ci=95):
    """Two-sided CI band of a null / permutation distribution. Returns (lo, hi)."""
    perm_values = np.asarray(perm_values, dtype=float)
    perm_values = perm_values[np.isfinite(perm_values)]
    return tuple(np.percentile(perm_values, [(100 - ci) / 2, 100 - (100 - ci) / 2]))


# ---------------------------------------------------------------------------
# Session-level statistical tests
# ---------------------------------------------------------------------------
def paired_session_test(df, value_col, cond_col, cond_a, cond_b,
                        session_col='session', agg='median'):
    """Wilcoxon signed-rank on per-session aggregates of two paired conditions.

    Aggregates ``value_col`` per session within each condition, pairs sessions
    present in both, and returns a dict with n, the two condition medians, the
    signed-rank statistic and two-sided p. Uses raw values (no FDE clipping).
    """
    piv = (df.groupby([session_col, cond_col])[value_col].agg(agg)
             .unstack(cond_col))
    if cond_a not in piv or cond_b not in piv:
        return dict(n=0, p=np.nan, stat=np.nan, med_a=np.nan, med_b=np.nan)
    pair = piv[[cond_a, cond_b]].dropna()
    a, b = pair[cond_a].to_numpy(), pair[cond_b].to_numpy()
    if len(a) < 1 or np.all(a - b == 0):
        stat, p = np.nan, np.nan
    else:
        res = stats.wilcoxon(a, b, alternative='two-sided')
        stat, p = float(res.statistic), float(res.pvalue)
    return dict(n=len(a), p=p, stat=stat,
                med_a=float(np.median(a)), med_b=float(np.median(b)))


def onesample_session_test(df, value_col, session_col='session', agg='median',
                           popmean=0.0):
    """Wilcoxon signed-rank of per-session aggregates vs ``popmean`` (e.g. 0)."""
    vals = df.groupby(session_col)[value_col].agg(agg).to_numpy()
    vals = vals[np.isfinite(vals)]
    d = vals - popmean
    if len(d) < 1 or np.all(d == 0):
        return dict(n=len(vals), p=np.nan, stat=np.nan, med=float(np.median(vals)) if len(vals) else np.nan)
    res = stats.wilcoxon(d, alternative='two-sided')
    return dict(n=len(vals), p=float(res.pvalue), stat=float(res.statistic),
                med=float(np.median(vals)))


def unpaired_session_test(df, value_col, group_col, group_a, group_b,
                          session_col='session', agg='median'):
    """Mann-Whitney U on per-session aggregates of two independent groups."""
    per = df.groupby([session_col, group_col])[value_col].agg(agg).reset_index()
    a = per.loc[per[group_col] == group_a, value_col].to_numpy()
    b = per.loc[per[group_col] == group_b, value_col].to_numpy()
    if len(a) < 1 or len(b) < 1:
        return dict(n_a=len(a), n_b=len(b), p=np.nan, stat=np.nan)
    res = stats.mannwhitneyu(a, b, alternative='two-sided')
    return dict(n_a=len(a), n_b=len(b), p=float(res.pvalue), stat=float(res.statistic),
                med_a=float(np.median(a)), med_b=float(np.median(b)))


# ---------------------------------------------------------------------------
# Paired session-level comparison plot (points connected by lines, by bat)
# ---------------------------------------------------------------------------
def paired_lines(ax, df, value_col, cond_col, order, *, session_col='session',
                 agg='median', palette=None, jitter=0.0, alpha=0.9,
                 line_kw=None, point_kw=None):
    """Draw a session-level paired plot: one connected line per session across the
    ordered conditions, points at each condition, coloured by bat.

    ``df`` is aggregated to one value per (session, condition). Returns the
    per-session wide DataFrame (index=session, columns=conditions) for stats.
    """
    if palette is None:
        palette = bat_pal()
    piv = (df.groupby([session_col, cond_col])[value_col].agg(agg)
             .unstack(cond_col)
             .reindex(columns=order))
    xs = np.arange(len(order))
    line_kw = dict(lw=1.1, alpha=0.55, zorder=1) | (line_kw or {})
    point_kw = dict(s=45, zorder=3, edgecolor='k', linewidth=0.5) | (point_kw or {})
    rng = np.random.default_rng(0)
    for sess, row in piv.iterrows():
        c = palette.get(bat_of(sess), '0.5')
        dx = rng.uniform(-jitter, jitter) if jitter else 0.0
        y = row.to_numpy(dtype=float)
        ax.plot(xs + dx, y, color=c, **line_kw)
        ax.scatter(xs + dx, y, color=c, **point_kw)
    ax.set_xticks(xs)
    ax.set_xticklabels(order)
    sns.despine(ax=ax)
    return piv


def bat_legend(ax, palette=None, title='Bat', loc='best', **kw):
    """Add a bat colour legend (one handle per bat in canonical order)."""
    if palette is None:
        palette = bat_pal()
    handles = [plt.Line2D([], [], marker='o', linestyle='', color=c,
                          markeredgecolor='k', markeredgewidth=0.5, label=b)
               for b, c in palette.items()]
    ax.legend(handles=handles, title=title, loc=loc, frameon=False, **kw)


# ---------------------------------------------------------------------------
# P-value tables and Nature-style legends
# ---------------------------------------------------------------------------
def fmt_p(p):
    """Nature-style exact p: 'P < 0.001' or 'P = 0.0xx' (3 decimals)."""
    if p is None or not np.isfinite(p):
        return 'P = n/a'
    return 'P < 0.001' if p < 0.001 else f'P = {p:.3f}'


def p_table(records, sort_col=None):
    """Assemble a p-value table from a list of dicts and print it in full.

    Adds a ``stars`` column and a Nature-style ``P`` string column, prints
    without truncation, and returns the DataFrame.
    """
    tab = pd.DataFrame(records)
    if 'p' in tab:
        tab['stars'] = tab['p'].map(stars_for_p)
        tab['P'] = tab['p'].map(fmt_p)
    if sort_col and sort_col in tab:
        tab = tab.sort_values(sort_col).reset_index(drop=True)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None,
                           'display.width', 200, 'display.float_format', lambda v: f'{v:.4g}'):
        print(tab.to_string(index=False))
    return tab


def legend(md):
    """Render a Nature-style figure legend from a markdown string.

    Use in a markdown cell normally; this helper just dedents/wraps for the rare
    case of emitting a legend from a code cell via ``IPython.display.Markdown``.
    """
    from IPython.display import Markdown
    return Markdown(textwrap.dedent(md).strip())


# ---------------------------------------------------------------------------
# Model-comparison figure (session medians + per-unit) with stars vs a reference
# ---------------------------------------------------------------------------
def _star_row(ax, order, pvals, frac=0.96):
    """One 3-tier star per category (vs the reference) near the top of `ax`."""
    lo, hi = ax.get_ylim()
    y = lo + frac * (hi - lo)
    for x, m in enumerate(order):
        s = stars_for_p(pvals.get(m, np.nan))
        if s:
            ax.text(x, y, s, ha='center', va='center', fontsize=14,
                    fontweight='bold', color='k')


def compare_models_fig(unit_df, value_col, order, ref, *, labels=None,
                       ylabel='', unit_ylabel=None, unit_name='neuron',
                       session_col='session', model_col='model', agg='median',
                       clip=None, palette=None, zero_line=False,
                       higher_is_better=True, figsize=None):
    """Two-panel model comparison in the unified style.

    Left: per-session aggregate (`agg`) as a grey boxplot with per-session points
    coloured by bat; 3-tier stars mark session-level Wilcoxon signed-rank vs
    `ref`. Right: the per-unit distribution (boxplot). `clip=(lo,hi)` bounds the
    *axes* only (e.g. FDE to [-1,1]); raw values drive the stats.

    Returns (fig, p_table) where p_table has one row per model vs `ref`.
    """
    if palette is None:
        palette = bat_pal()
    labels = labels or {m: m for m in order}
    order = [m for m in order if m in set(unit_df[model_col])]
    sess = (unit_df.groupby([session_col, 'bat', model_col], observed=True)[value_col]
                   .agg(agg).reset_index())
    wide = sess.pivot_table(index=[session_col, 'bat'], columns=model_col,
                            values=value_col, observed=False)
    recs, pvals = [], {}
    for m in order:
        if m == ref or m not in wide or ref not in wide:
            continue
        pair = wide[[m, ref]].dropna()
        d = (pair[ref] - pair[m]) if higher_is_better else (pair[m] - pair[ref])
        if len(pair) >= 3:
            res = stats.wilcoxon(pair[m], pair[ref], alternative='two-sided')
            pvals[m] = float(res.pvalue)
            recs.append(dict(model=m, vs=ref, n=len(pair), median_diff=float(d.median()),
                             p=pvals[m]))

    figsize = figsize or (max(8, 1.4 * len(order)) + 2, PANEL_H + 1.5)
    fig, (axs, axu) = plt.subplots(1, 2, figsize=figsize)
    bats_present = [b for b in palette if b in set(sess['bat'])]
    for ax, data, is_sess in ((axs, sess, True), (axu, unit_df, False)):
        sns.boxplot(data=data, x=model_col, y=value_col, order=order, color='0.85',
                    width=0.6, linewidth=1.0, showfliers=False, ax=ax)
        if is_sess:
            sns.stripplot(data=data, x=model_col, y=value_col, order=order, hue='bat',
                          palette=palette, hue_order=bats_present, jitter=0.18, size=6,
                          alpha=0.9, edgecolor='k', linewidth=0.3, ax=ax, legend=False)
        if zero_line:
            ax.axhline(0, color='0.5', ls='--', lw=0.9, zorder=0)
        if clip:
            ax.set_ylim(*clip)
        ax.set_xlabel('')
        ax.set_ylabel(ylabel if is_sess else (unit_ylabel or ylabel))
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([labels[m] for m in order], rotation=35, ha='right')
        _star_row(ax, order, pvals)
        sns.despine(ax=ax)
    axs.set_title('session ' + agg)
    axu.set_title(f'per {unit_name}')
    fig.tight_layout()
    ptab = pd.DataFrame(recs)
    return fig, ptab


# ---------------------------------------------------------------------------
# Latent embeddings (Fig 2 / ED Fig 5): extraction, route labels, projection
# ---------------------------------------------------------------------------
INDOOR_PREFIXES = ('Dataset_1', 'Dataset_3')   # shared indoor lab room


def is_indoor(session):
    """True for the shared indoor lab room (Datasets 1 & 3)."""
    return session.startswith(INDOOR_PREFIXES)


# Route = (origin, destination) endpoint classes: 0 neither, 1 reward, 2 perch.
ROUTE_COLORS = ['0.7', 'firebrick', 'darkorange', 'forestgreen', 'steelblue']
ROUTE_LABELS = ['other', 'rew→rew', 'rew→perch', 'perch→rew', 'perch→perch']


def endpoint_class(pt, rew_locs, perch_locs, thresh):
    """Classify a single 3D point: 1 near reward, 2 near perch, else 0."""
    if np.linalg.norm(pt[None, :] - rew_locs, axis=1).min() < thresh:
        return 1
    if np.linalg.norm(pt[None, :] - perch_locs, axis=1).min() < thresh:
        return 2
    return 0


def route_id(origin, dest):
    """(origin, dest) endpoint classes -> route id 0..4 (0 = any 'neither')."""
    return 0 if (origin == 0 or dest == 0) else 1 + (origin - 1) * 2 + (dest - 1)


def zscore_by_session(values, session):
    """Per-session z-score (so x-coordinate colour is comparable across rooms)."""
    out = np.zeros_like(values, dtype=float)
    for s in np.unique(session):
        m = session == s
        sd = values[m].std()
        out[m] = (values[m] - values[m].mean()) / (sd if sd > 1e-12 else 1.0)
    return out


def pca_rotation(X, n=3, seed=0):
    """Fit a PCA(n) on X and return (transform_fn, fitted_pca)."""
    from sklearn.decomposition import PCA
    p = PCA(n_components=n, random_state=seed).fit(X)
    return p.transform, p


def lobe_projection(X, group, n_pca=3, seed=0):
    """Project `X` (N, d) to 2-D so the two `group` lobes separate horizontally.

    PCA to `n_pca`, then rotate so the axis of maximal separation between the two
    most populated non-zero `group` classes (e.g. reward- vs perch-origin routes)
    is the horizontal axis; the vertical axis is the top residual PC. Returns
    (emb2d (N, 2), transform_fn) so the SAME rotation can be applied to another
    array (e.g. a different colouring of the same points)."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_pca, random_state=seed).fit(X)
    P = pca.transform(X)
    grp = np.asarray(group)
    classes = [c for c in np.unique(grp) if c != 0]
    if len(classes) >= 2:
        counts = {c: int((grp == c).sum()) for c in classes}
        a, b = sorted(counts, key=counts.get, reverse=True)[:2]
        d = P[grp == a].mean(0) - P[grp == b].mean(0)
    else:
        d = np.array([1.0] + [0.0] * (n_pca - 1))
    d = d / (np.linalg.norm(d) + 1e-12)
    resid = P - np.outer(P @ d, d)
    v = PCA(n_components=1, random_state=seed).fit(resid).components_[0]
    v = v / (np.linalg.norm(v) + 1e-12)

    def _tf(Y):
        Q = pca.transform(Y)
        return np.column_stack([Q @ d, Q @ v])
    return np.column_stack([P @ d, P @ v]), _tf


def plot_embedding(ax, emb2d, color, *, categorical=False, cmap='viridis',
                   cat_colors=None, s=4, alpha=0.5, vlim=None, rasterized=True):
    """Scatter a 2D embedding coloured by a per-point variable.

    Continuous: `color` is numeric, coloured with `cmap` (optional (vmin,vmax) `vlim`).
    Categorical: `color` is integer class ids, coloured with `cat_colors` (list).
    Returns the mappable (for a shared colourbar) or None (categorical).
    """
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal', 'datalim')
    for sp in ax.spines.values():
        sp.set_visible(False)
    if categorical:
        cc = cat_colors or ROUTE_COLORS
        for k in np.unique(color):
            m = color == k
            ax.scatter(emb2d[m, 0], emb2d[m, 1], s=s, alpha=alpha,
                       color=cc[int(k) % len(cc)], rasterized=rasterized, linewidths=0)
        return None
    vmin, vmax = vlim if vlim is not None else np.percentile(color, [2, 98])
    sc = ax.scatter(emb2d[:, 0], emb2d[:, 1], c=color, cmap=cmap, s=s, alpha=alpha,
                    vmin=vmin, vmax=vmax, rasterized=rasterized, linewidths=0)
    return sc


# ---------------------------------------------------------------------------
# Figure saving
# ---------------------------------------------------------------------------
FIG_DIR = None  # set by notebook; e.g. Path('out/figures')


def save_fig(fig, name, *, subdir='figures'):
    """Save `fig` as both PDF and PNG under ``notebooks/out/<subdir>/``."""
    from pathlib import Path
    base = FIG_DIR or Path('out') / subdir
    base.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(base / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    return base / f'{name}.pdf'
