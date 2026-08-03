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
        'axes.titlesize': 18,
        'axes.labelsize': 17,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 14,
        'legend.title_fontsize': 14,
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
# Canonical categorical palettes
# ---------------------------------------------------------------------------
# Six independent category *kinds* each get their own colour family so a colour
# never carries two meanings within a figure.  Listed in order of importance:
#   1. routes          - ROUTE_COLORS / ROUTE_LABELS (hand-picked, defined below)
#   2. bats            - bat_pal()  (tab10; from digitalbat.utility.utils)
#   3. neuron clusters - readout-weight clusters            (Set2, pastel)
#   4. environments    - recording room / indoor-outdoor    (named ENV_PALETTE)
#   5. sessions        - individual recording sessions      (husl ramp)
#   6. flight clusters - trajectory clusters                (tab20b)
# Builders take the *ordered* category values and return a ``{value: rgb}`` dict
# keyed by the values themselves (dtype preserved) so they drop straight into
# seaborn via ``hue=<col>, palette=<dict>`` - see ``cat_boxplot``.
CLUSTER_CMAP = 'Set2'           # neuron (readout) clusters
SESSION_CMAP = 'husl'           # recording sessions
FLIGHT_CLUSTER_CMAP = 'tab20b'  # flight / trajectory clusters

# Environments: canonical names -> colour. Superset of the indoor/outdoor split
# and the 3-config room map (lab / dataset_2 / krishna_fs) used across notebooks;
# the two namings share colours because they are the same rooms (lab = indoor,
# dataset_2 = outdoor).
#
# Chosen 2026-07-30 to clear every other categorical palette in the paper. The old
# set could not: its indoor blue #3b6ea5 sat CIELAB dE 6.6 from bat-blue #1f77b4,
# its krishna green 10.6 from route rew->rew, and its outdoor orange 20.7 from the
# negative-motif orange - so ED Fig 4, which draws motif colour and environment on
# one canvas, had no non-clashing env hue at all. The replacement was picked by a
# grid search in LCh over what is left once the motif pair, all six bat colours and
# the five route colours are excluded: cyan-teal / olive / rose, min dE 31.2 / 37.1
# / 33.2 to the nearest taken colour (nearest are bat-blue, bat-green, bat-brown),
# >= 62 apart from each other, and >= 71 apart under a Vienot deuteranope
# simulation. Olive for the outdoor room is also a small semantic win.
ENV_PALETTE = {
    'indoor': '#1f9fb5', 'outdoor': '#7d8a33',
    'lab': '#1f9fb5', 'dataset_2': '#7d8a33', 'krishna_fs': '#d98aa8',
}


def _ordered_palette(values, source, *, cycle=False):
    """``{value: rgb}`` for the distinct ``values`` (first-seen order) drawn from
    a seaborn colour source. ``cycle`` wraps a fixed qualitative palette when
    there are more categories than colours (else the source is sampled to N)."""
    keys = list(dict.fromkeys(values))
    n = len(keys)
    if cycle:
        base = sns.color_palette(source)
        cols = [base[i % len(base)] for i in range(n)]
    else:
        cols = sns.color_palette(source, n_colors=max(n, 3))
    return dict(zip(keys, cols))


def route_palette():
    """``{route_id: rgb}`` for route ids 0..4 (labels in ROUTE_LABELS)."""
    return {i: mpl.colors.to_rgb(c) for i, c in enumerate(ROUTE_COLORS)}


def cluster_palette(order):
    """``{cluster_id: rgb}`` for neuron (readout) clusters in the given order."""
    return _ordered_palette(order, CLUSTER_CMAP)


def env_palette(order=None):
    """``{env: rgb}`` for environments; the full canonical map when ``order`` is
    None, else restricted/ordered to ``order`` (unknown envs -> grey)."""
    src = ENV_PALETTE.keys() if order is None else order
    return {e: mpl.colors.to_rgb(ENV_PALETTE.get(e, '0.5')) for e in src}


def session_palette(sessions):
    """``{session: rgb}`` for recording sessions in the given order (husl ramp)."""
    return _ordered_palette(sessions, SESSION_CMAP)


def flight_cluster_palette(order):
    """``{flight_cluster_id: rgb}`` for trajectory clusters in the given order."""
    return _ordered_palette(order, FLIGHT_CLUSTER_CMAP, cycle=True)


def cat_boxplot(ax, data, x, y, order, palette, *, showfliers=False, **kw):
    """Categorical boxplot coloured by a ``{level: rgb}`` dict, robust to
    seaborn >= 0.13.

    Assigns ``hue=x`` internally (with ``legend=False``) so a dict palette keyed
    by the native level values is honoured - avoiding the "palette without hue"
    deprecation, which silently stringifies the levels and breaks int/other-keyed
    palette dicts. ``palette`` may also be any seaborn colour name.
    """
    return sns.boxplot(data=data, x=x, y=y, order=order, hue=x, hue_order=order,
                       palette=palette, legend=False, showfliers=showfliers,
                       ax=ax, **kw)


def tukey_trim(data, y, by, *, k=1.5):
    """Return ``data`` with rows whose ``y`` lies outside the Tukey whisker range
    ``[Q1 - k*IQR, Q3 + k*IQR]``, with quartiles computed **within each group**
    defined by the ``by`` column(s).

    This is the violin analogue of a boxplot's ``showfliers=False``: the surviving
    range is exactly what the box whiskers spanned, so a violin drawn on the result
    stops at the whiskers instead of stretching over the fliers. Display-only -
    pass raw (untrimmed) values to any statistics.
    """
    by = [by] if isinstance(by, str) else list(by)
    grp = data.groupby(by, observed=True)[y]
    q1 = grp.transform(lambda s: s.quantile(0.25))
    q3 = grp.transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    return data[(data[y] >= q1 - k * iqr) & (data[y] <= q3 + k * iqr)]


def cat_violin(ax, data, x, y, order, palette, *, inner='box', cut=0,
               density_norm='width', trim_outliers=False, **kw):
    """Categorical violinplot coloured by a ``{level: rgb}`` dict - the violin
    analogue of :func:`cat_boxplot`, for neuron-/flight-level distributions where
    the shape (not just the quartiles) is informative.

    Uses the same internal ``hue=x`` trick so a dict palette keyed by the native
    level values survives seaborn >= 0.13. ``cut=0`` bounds the kernel density to
    the observed data range (important for bounded metrics such as correlation /
    FDE); ``density_norm='width'`` gives every violin the same max width, so
    categories with different n stay visually comparable (as boxes were).
    ``trim_outliers=True`` drops points beyond each level's Tukey whiskers
    (:func:`tukey_trim`) before the KDE - the violin equivalent of
    ``showfliers=False`` (display-only; does not touch any stats).
    """
    if trim_outliers:
        data = tukey_trim(data, y, x)
    return sns.violinplot(data=data, x=x, y=y, order=order, hue=x, hue_order=order,
                          palette=palette, legend=False, inner=inner, cut=cut,
                          density_norm=density_norm, ax=ax, **kw)


# ---------------------------------------------------------------------------
# Significance brackets that DO NOT stretch the y-axis
# ---------------------------------------------------------------------------
def _marker_for_p(p, dagger=False):
    """3-tier stars via stars_for_p, plus an optional '†' for a marginal trend
    (0.05 <= p < 0.10) when ``dagger=True``; '' otherwise."""
    s = stars_for_p(p)
    if s or not dagger or p is None or not np.isfinite(p):
        return s
    return '†' if 0.05 <= p < 0.10 else ''


def sig_bracket(ax, x1, x2, p, *, level=0, color='k', linewidth=1.2,
                fontsize=17, base=0.98, step=0.07, show_ns=False, star_gap=0.0,
                dagger=False, bar_drop=0.0):
    """Draw a 3-tier-star significance bracket between categorical positions
    ``x1`` and ``x2`` **without** changing ``ax`` limits.

    The bracket is drawn in a blended transform: x in data coordinates, y in
    axes-fraction coordinates, so it floats above the data near the top of the
    panel regardless of the data range. ``level`` stacks multiple brackets
    (0 = lowest). Hidden entirely when p >= .05 unless ``show_ns``. ``star_gap``
    nudges the star label vertically off the bar (negative pulls it closer).
    """
    stars = _marker_for_p(p, dagger)
    if not stars:
        if not show_ns:
            return
        stars = 'n.s.'
    trans = ax.get_xaxis_transform()  # x data, y axes-fraction
    y = base - level * step
    h = 0.015
    yb = y - bar_drop                  # bar (and its verticals) can dip without moving the marker
    ax.plot([x1, x1, x2, x2], [yb, yb + h, yb + h, yb], transform=trans,
            lw=linewidth, color=color, clip_on=False)
    ax.text((x1 + x2) / 2, y + h + star_gap, stars, transform=trans, ha='center',
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


def paired_lineplot(ax, data, x, y, order, *, session_col='session', bat_col='bat',
                    palette=None, agg='mean', line_lw=0.8, line_alpha=0.35,
                    marker_size=3.0, mean_color='k', mean_lw=2.6, ci=95, n_boot=2000,
                    seed=0, set_xticklabels=True):
    """Paired session-level line plot: one faint thin line per session (coloured by
    bat) connecting the ordered ``x`` categories, overlaid with a thick black
    point-plot of the across-session **mean ± 95% CI**.

    This is the paired analogue of a combined box + per-bat swarm for session-level
    comparisons (few, ``<= 17``, sessions): the connecting lines expose the
    within-session pairing across categories that a swarm hides, while the black
    point-plot summarises the group mean and its bootstrap CI. ``data`` is long-form
    with (at least) ``session_col``, ``bat_col``, ``x`` and ``y``; it is aggregated
    (``agg``) to one value per (session, category) for the lines, and the same
    per-session values drive the mean/CI. Returns the per-session wide DataFrame
    (index=[session, bat], columns=categories) for downstream stats.
    """
    if palette is None:
        palette = bat_pal()
    xs = np.arange(len(order))
    wide = data.pivot_table(index=[session_col, bat_col], columns=x, values=y,
                            aggfunc=agg, observed=True).reindex(columns=order)
    for (sess, bat), row in wide.iterrows():
        ax.plot(xs, row.to_numpy(dtype=float), '-o', color=palette.get(bat, '0.5'),
                lw=line_lw, alpha=line_alpha, ms=marker_size, mew=0, zorder=2)
    # across-session mean +/- 95% CI, thick and black, drawn on top. It runs on the SESSION
    # values just plotted as lines (`wide` melted back to long), not on `data`: handing the raw
    # rows to pointplot made the overlay a mean over ROWS -- pooled across sessions and weighted
    # by each session's row count, and computed with `mean` where the lines used `agg` -- so the
    # black marker could sit outside the cloud of lines it was supposed to summarise (and a
    # single non-finite row could carry it to infinity). This is what the docstring above always
    # claimed; it was the code that disagreed.
    sess_long = (wide.reset_index()
                 .melt(id_vars=[session_col, bat_col], value_vars=list(order),
                       var_name=x, value_name=y).dropna(subset=[y]))
    sns.pointplot(data=sess_long, x=x, y=y, order=order, color=mean_color,
                  estimator='mean', errorbar=('ci', ci), n_boot=n_boot, seed=seed,
                  linewidth=mean_lw, markersize=mean_lw * 2.4, capsize=0.14,
                  err_kws={'linewidth': mean_lw * 0.8}, zorder=10, ax=ax)
    ax.set_xticks(xs)
    if set_xticklabels:
        ax.set_xticklabels(order)
    sns.despine(ax=ax)
    return wide


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


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values for one family of tests.

    ``pvals`` is a sequence or a ``{key: p}`` mapping; the return has the same
    shape with every raw p replaced by its adjusted value, so comparing the
    result to 0.05 controls the family-wise error rate at 0.05. Adjusted values
    are capped at 1.0 and made monotone in the sorted order (the running max),
    which is what makes the step-down procedure valid. Non-finite entries pass
    through as NaN and do not count towards the family size.
    """
    is_map = isinstance(pvals, dict)
    items = list(pvals.items()) if is_map else list(enumerate(pvals))
    real = sorted([(k, float(p)) for k, p in items
                   if p is not None and np.isfinite(p)], key=lambda kv: kv[1])
    m, run, adj = len(real), 0.0, {}
    for i, (k, p) in enumerate(real):
        run = max(run, (m - i) * p)
        adj[k] = min(run, 1.0)
    out = {k: adj.get(k, np.nan) for k, _ in items}
    return out if is_map else [out[k] for k, _ in items]


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
            ax.text(x, y, s, ha='center', va='center', fontsize=16,
                    fontweight='bold', color='k')


def _star_brackets(ax, order, pvals, ref, *, show_ns=False, base=1.02, step=0.09,
                   star_gap=0.0, dagger=False, dagger_drop=0.0):
    """Stacked significance brackets from `ref` to every other category.

    One bracket per non-reference model (a line connecting the two x-positions
    with 3-tier stars above), nested so the narrowest span sits lowest and the
    widest on top. Drawn in the axes-fraction transform used by ``sig_bracket``;
    with ``base`` >= 1 they float just above the panel frame (clear of the box
    whiskers). Only the brackets that actually render are stacked, so hiding
    ``n.s.`` (``show_ns=False``, the default) leaves no gaps."""
    xref = order.index(ref)
    ordered = sorted((abs(x - xref), x, m) for x, m in enumerate(order) if m != ref)
    shown = [(x, m) for _, x, m in ordered
             if show_ns or _marker_for_p(pvals.get(m, np.nan), dagger)]
    for lvl, (x, m) in enumerate(shown):   # narrowest first -> lowest bracket
        _p = pvals.get(m, np.nan)
        _bd = dagger_drop if (dagger_drop and _marker_for_p(_p, dagger) == '†') else 0.0
        sig_bracket(ax, xref, x, _p, level=0, base=base + lvl * step,
                    show_ns=show_ns, star_gap=star_gap, dagger=dagger, bar_drop=_bd)


def pooled_ptable(unit_df, value_col, order, ref, id_cols, *, model_col='model',
                  higher_is_better=True):
    """Pooled per-unit paired Wilcoxon signed-rank of every model vs `ref`.

    Pairs rows on `id_cols` (e.g. ['session','flight_id'] or ['session','unit']) so
    each physical unit contributes one matched (model, ref) observation, then runs a
    two-sided signed-rank across all pooled units. Returns a DataFrame (one row per
    model) with the same columns as compare_models_fig's session table — model, vs,
    n, median_diff, p — so it can be printed with `p_table` or turned into a
    `{model: p}` dict for compare_models_fig's `unit_pvals`. NB: pooling every unit
    inflates n and makes these tests anticonservative relative to the session-level
    test; use it to star the per-unit panel, not as the primary inference.
    """
    wide = unit_df.pivot_table(index=id_cols, columns=model_col, values=value_col,
                               observed=True)
    recs = []
    for m in order:
        if m == ref or m not in wide or ref not in wide:
            continue
        pair = wide[[m, ref]].dropna()
        if len(pair) < 3:
            continue
        d = (pair[ref] - pair[m]) if higher_is_better else (pair[m] - pair[ref])
        recs.append(dict(model=m, vs=ref, n=int(len(pair)), median_diff=float(d.median()),
                         p=float(stats.wilcoxon(pair[m], pair[ref],
                                                alternative='two-sided').pvalue)))
    return pd.DataFrame(recs)


def compare_models_fig(unit_df, value_col, order, ref, *, labels=None,
                       ylabel='', unit_ylabel=None, unit_name='neuron',
                       session_col='session', model_col='model', agg='median',
                       clip=None, sess_ylim=None, palette=None, zero_line=False,
                       higher_is_better=True, figsize=None, axes=None,
                       star_style='row', star_kw=None, unit_pvals=None, unit_id_cols=None,
                       show_titles=True, title_y=1.0, trim_unit_outliers=True, show_unit=True,
                       show_xticklabels=True, show_sess_ylabel=True, show_unit_ylabel=True):
    """Two-panel model comparison in the unified style.

    Left: per-session aggregate (`agg`) as a grey boxplot with per-session points
    swarmed and coloured by bat; 3-tier stars mark session-level Wilcoxon signed-rank
    vs `ref`. Right (optional): the per-unit (neuron/flight) distribution as a violin
    (points beyond each model's Tukey whiskers dropped from the KDE unless
    `trim_unit_outliers=False`; display-only, stats use raw values). The unit panel is
    drawn only when a unit axis is supplied — pass `axes=(ax_sess, None)` (or a single
    Axes, or `show_unit=False` when creating the figure) for a session-only panel. Pass `unit_pvals`
    (`{model: p}`), or `unit_id_cols` (e.g. ['session','flight_id']) to compute one
    via `pooled_ptable`, to star the right panel with a *different* test appropriate
    to the pooled per-unit distribution shown there (per-flight / per-neuron
    Wilcoxon); when both are omitted the right panel repeats the session-level stars.
    `clip=(lo,hi)` bounds the
    *axes* only (e.g. FDE to [-1,1]); raw values drive the stats. `sess_ylim=(lo,hi)`
    overrides the left (session) panel's ylim (applied before the stars are placed,
    so they stay on-axis); the right panel keeps `clip`. Pass `axes=(ax_sess, ax_unit)`
    to draw onto an existing shared canvas instead of a new figure.

    `star_style='row'` (default) prints one star per category above it. `'bracket'`
    instead draws a connecting line from `ref` to each other category with the
    3-tier stars above (stacked, narrowest span lowest) — useful when `ref` is the
    baseline being compared against every alternative; `star_kw` is forwarded to
    the bracket drawer (e.g. `show_ns`, `base`, `step`).

    For small-multiple grids, `show_titles` / `show_xticklabels` /
    `show_sess_ylabel` / `show_unit_ylabel` suppress the corresponding decoration
    on interior panels, and `title_y` lifts the panel titles (axes-fraction) to
    clear tall significance brackets.

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

    if unit_pvals is None and unit_id_cols is not None:   # per-unit stars for the right panel
        _upt = pooled_ptable(unit_df, value_col, order, ref, unit_id_cols,
                             model_col=model_col, higher_is_better=higher_is_better)
        unit_pvals = dict(zip(_upt['model'], _upt['p']))

    if axes is not None:                              # draw onto a shared canvas; when the
        axs, axu = axes if isinstance(axes, (tuple, list)) else (axes, None)  # unit axis is
        fig = axs.figure                              # None (or omitted) only the session panel is drawn
    else:
        figsize = figsize or (max(8, 1.4 * len(order)) + 2, PANEL_H + 1.5)
        if show_unit:
            fig, (axs, axu) = plt.subplots(1, 2, figsize=figsize)
        else:
            fig, axs = plt.subplots(1, 1, figsize=figsize); axu = None
    _panels = [(axs, sess, True)] + ([(axu, unit_df, False)] if axu is not None else [])
    for ax, data, is_sess in _panels:
        if is_sess:                                   # session aggregates (few, <=17): faint per-bat lines + black mean +/- 95% CI
            paired_lineplot(ax, data, model_col, value_col, order, session_col=session_col,
                            bat_col='bat', palette=palette, set_xticklabels=False)
        else:                                         # per-unit (neuron/flight) distribution: violin for shape
            _vd = tukey_trim(data, value_col, model_col) if trim_unit_outliers else data
            sns.violinplot(data=_vd, x=model_col, y=value_col, order=order, color='0.85',
                           width=0.85, linewidth=1.0, inner='box', cut=0,
                           density_norm='width', ax=ax)
        if zero_line:
            ax.axhline(0, color='0.5', ls='--', lw=0.9, zorder=0)
        if is_sess and sess_ylim is not None:
            ax.set_ylim(*sess_ylim)
        elif clip:
            ax.set_ylim(*clip)
        ax.set_xlabel('')
        show_yl = show_sess_ylabel if is_sess else show_unit_ylabel
        ax.set_ylabel((ylabel if is_sess else (unit_ylabel or ylabel)) if show_yl else '')
        ax.set_xticks(range(len(order)))
        if show_xticklabels:
            ax.set_xticklabels([labels[m] for m in order], rotation=35, ha='right')
        else:
            ax.set_xticklabels([])
        _pv = pvals if (is_sess or unit_pvals is None) else unit_pvals
        if star_style == 'bracket' and ref in order:
            _star_brackets(ax, order, _pv, ref, **(star_kw or {}))
        else:
            _star_row(ax, order, _pv)
        sns.despine(ax=ax)
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(   # 0.0 -> 0, drop trailing zeros
            lambda _v, _p: ('0' if f'{_v:g}' in ('0', '-0') else f'{_v:g}')))
    if show_titles:                          # session-median panel title dropped by request
        if axu is not None:
            axu.set_title(f'per {unit_name}', y=title_y)
    if axes is None:
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
# Routes: three deep matte colours from the built-in ColorBrewer 'Dark2' palette
# (teal-green[0], magenta[3], gold[5]) plus a hand-picked deep slate blue. Dark2's
# ochre[6] read too close to its gold, and its orange[1]/purple[2]/green[4] collide
# with the tab10 bat hues, so those are dropped. All four sit clear of every bat
# colour (min CIELAB ΔE ≈ 23, vs bat-blue) so routes and bats never read alike where
# they co-appear (Fig 2 RSA side-bars). 'other' stays grey. Used only in Fig 2.
ROUTE_COLORS = ['0.7'] + [sns.color_palette('Dark2')[i] for i in (0, 3, 5)] + ['#3b4a8c']
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


def plot_trajectories_3d(ax, trajs, values, *, categorical=False, cmap='viridis',
                         cat_colors=None, vlim=None, lw=1.2, alpha=0.8, rasterized=True,
                         show_endpoints=False, start_color='tab:orange', end_color='deeppink'):
    """Draw per-flight trajectories as connected 3-D polylines on a 3-D `ax`.

    `trajs` is a list of (T_i, 3) arrays (one flight each). `values` supplies the
    colour: if `categorical`, a per-flight class id (whole line one colour, via
    `cat_colors`); otherwise a per-flight (T_i,) array coloured per segment with
    `cmap` (optional (vmin, vmax) `vlim`). With `show_endpoints`, each flight's first
    (`start_color`) and last (`end_color`) point is marked; the markers are drawn
    *before* the lines so they sit behind them. Returns a mappable (continuous) or None.
    """
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    trajs = [np.asarray(t, float) for t in trajs]
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    if show_endpoints:
        good = [t for t in trajs if len(t) >= 1]
        if good:
            starts = np.array([t[0] for t in good]); ends = np.array([t[-1] for t in good])
            ax.scatter(starts[:, 0], starts[:, 1], starts[:, 2], s=16, marker='^', color=start_color,
                       depthshade=False, edgecolors='none', zorder=0, rasterized=rasterized)
            ax.scatter(ends[:, 0], ends[:, 1], ends[:, 2], s=16, color=end_color,
                       depthshade=False, edgecolors='none', zorder=0, rasterized=rasterized)
    if categorical:
        cc = cat_colors or ROUTE_COLORS
        for t, v in zip(trajs, values):
            if len(t) < 2:
                continue
            ax.plot(t[:, 0], t[:, 1], t[:, 2], '-', lw=lw, alpha=alpha,
                    color=cc[int(v) % len(cc)], rasterized=rasterized)
        sm = None
    else:
        allc = np.concatenate([np.asarray(v, float) for v in values])
        vmin, vmax = vlim if vlim is not None else np.nanpercentile(allc, [2, 98])
        norm = plt.Normalize(vmin, vmax)
        for t, v in zip(trajs, values):
            if len(t) < 2:
                continue
            pts = t.reshape(-1, 1, 3)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            v = np.asarray(v, float)
            lc = Line3DCollection(segs, cmap=cmap, norm=norm, linewidths=lw, alpha=alpha)
            lc.set_array((v[:-1] + v[1:]) / 2)
            lc.set_rasterized(rasterized)
            ax.add_collection3d(lc)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    pts = np.vstack(trajs)
    ax.set_xlim(pts[:, 0].min(), pts[:, 0].max())
    ax.set_ylim(pts[:, 1].min(), pts[:, 1].max())
    ax.set_zlim(pts[:, 2].min(), pts[:, 2].max())
    return sm


# ---------------------------------------------------------------------------
# Figure saving
# ---------------------------------------------------------------------------
FIG_DIR = None  # set by notebook; e.g. Path('out/figures')


def _cap_first(s):
    """Capitalise the first character iff it is a lowercase ASCII letter."""
    c = s[:1]
    return c.upper() + s[1:] if (c.isascii() and c.isalpha() and c.islower()) else s


def _cap_legend(leg):
    if leg is None:
        return
    title = leg.get_title()
    if title is not None:
        title.set_text(_cap_first(title.get_text()))
    for txt in leg.get_texts():
        txt.set_text(_cap_first(txt.get_text()))


def capitalize_labels(fig):
    """Upper-case the first letter of every title, axis label, legend entry and
    categorical (non-numeric) tick label in `fig`. Anything starting with a digit,
    symbol, math (``$``) or non-ASCII/Greek character is left untouched. Idempotent."""
    for ax in fig.axes:
        # Update text in place so a custom title position (e.g. set_title(y=...))
        # survives; re-issuing set_title/set_xlabel/set_ylabel would reset it.
        ax.title.set_text(_cap_first(ax.get_title()))
        ax.xaxis.label.set_text(_cap_first(ax.get_xlabel()))
        ax.yaxis.label.set_text(_cap_first(ax.get_ylabel()))
        if hasattr(ax, 'get_zlabel'):
            ax.set_zlabel(_cap_first(ax.get_zlabel()))
        for axis, setter in ((ax.xaxis, ax.set_xticklabels), (ax.yaxis, ax.set_yticklabels)):
            labs = [t.get_text() for t in axis.get_ticklabels()]
            if any(l[:1].isascii() and l[:1].isalpha() and l[:1].islower() for l in labs):
                try:
                    setter([_cap_first(l) for l in labs])
                except Exception:
                    pass
        _cap_legend(ax.get_legend())
    for leg in getattr(fig, 'legends', []):
        _cap_legend(leg)
    if getattr(fig, '_suptitle', None) is not None:
        fig._suptitle.set_text(_cap_first(fig._suptitle.get_text()))
    return fig


def save_fig(fig, name, *, subdir='figures', capitalize=True, exts=('pdf', 'png')):
    """Save `fig` under ``notebooks/out/<subdir>/<ext>/`` for each of `exts`.

    With ``capitalize`` (default), first letters of titles / axis labels / legend
    text / categorical tick labels are upper-cased via :func:`capitalize_labels`.

    Add ``'svg'`` to `exts` for figures that have to survive a round trip through
    Inkscape. Any artist drawn with alpha < 1 and ``rasterized=True`` becomes an RGBA
    image in the PDF, which matplotlib writes as an image plus a separate soft mask
    (``/SMask``); Inkscape misrenders those. SVG carries the same raster with a plain
    opacity attribute instead, so it imports cleanly.
    """
    from pathlib import Path
    if capitalize:
        capitalize_labels(fig)
    base = FIG_DIR or Path('out') / subdir
    for ext in exts:
        d = base / ext
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    return base / 'pdf' / f'{name}.pdf'
