"""Per-session trajectory comparison plots shared across decoder notebooks.

Three plotters take a uniform `polylines_per_decoder` shape:

    {decoder_key: {(session, flight_id): (truth, pred)}}

`truth`/`pred` are `(T, 2)` arrays for the 2D grid and `(T, 3)` for the 3D
grid + zoom. `decoder_specs` is an ordered list of `(decoder_key, label, color)`
tuples controlling column order and per-decoder line color.

`polylines_from_per_timestep_df` is the standard way to build the inner dict
when each decoder writes one prediction per timestep with `x_*`/`y_*`/`z_*`
columns. PV-Perceiver-style snippet decoders should produce the same dict
shape themselves (see `pp_decoder.reconstruct_pv_per_flight`).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def polylines_from_per_timestep_df(df, *,
                                   pred_cols=('x_pred', 'y_pred', 'z_pred'),
                                   truth_cols=('x_true', 'y_true', 'z_true'),
                                   session_col='session',
                                   flight_id_col='flight_id',
                                   sort_by='t_abs',
                                   min_len=2):
    """Build `{(session, flight_id): (truth, pred)}` from per-timestep predictions.

    Pass length-2 `pred_cols`/`truth_cols` for 2D plots, length-3 for 3D.
    """
    out = {}
    for (sess, fid), g in df.groupby([session_col, flight_id_col], sort=False):
        if len(g) < min_len:
            continue
        g = g.sort_values(sort_by)
        truth = g[list(truth_cols)].to_numpy(np.float64)
        pred = g[list(pred_cols)].to_numpy(np.float64)
        out[(sess, int(fid))] = (truth, pred)
    return out


def _legend_handles(decoder_specs):
    return [
        Line2D([0], [0], color='k', linewidth=1.5, label='truth'),
        *[Line2D([0], [0], color=c, linewidth=1.5, label=l)
          for _, l, c in decoder_specs],
    ]


def _draw_polyline(ax, points, *, dim, **kwargs):
    if dim == 2:
        ax.plot(points[:, 0], points[:, 1], **kwargs)
    else:
        ax.plot(points[:, 0], points[:, 1], points[:, 2], **kwargs)


def _iter_session_polylines(polylines_per_decoder, decoder_key, session):
    return (
        (truth, pred)
        for (s_, _fid), (truth, pred) in polylines_per_decoder.get(decoder_key, {}).items()
        if s_ == session
    )


def plot_trajectory_grid(session_list, decoder_specs, polylines_per_decoder,
                         *, dim=2,
                         figsize_per_panel=None,
                         session_label_fn=None,
                         suptitle=None,
                         truth_kwargs=None,
                         pred_kwargs=None):
    """Grid plot of test trajectories: rows = sessions, columns = decoders.

    Parameters
    ----------
    session_list : ordered iterable of session names; one row per session.
    decoder_specs : ordered list of `(decoder_key, label, color)` tuples;
        one column per entry.
    polylines_per_decoder : `{decoder_key: {(session, flight_id): (truth, pred)}}`.
    dim : 2 (top-down) or 3 (3D axes).
    figsize_per_panel : `(w, h)` tuple. Defaults to (2.4, 2.2) for 2D, (2.6, 2.4) for 3D.
    session_label_fn : callable session_name -> ylabel string. Default: identity.
    suptitle : figure-level title.
    truth_kwargs / pred_kwargs : optional `dict`s of extra `ax.plot` kwargs.
    """
    if dim not in (2, 3):
        raise ValueError(f'dim must be 2 or 3, got {dim}')
    if figsize_per_panel is None:
        figsize_per_panel = (2.4, 2.2) if dim == 2 else (2.6, 2.4)

    truth_kw = {'color': 'k', 'alpha': 0.5, 'linewidth': 0.8, **(truth_kwargs or {})}
    pred_kw = {'alpha': 0.6, 'linewidth': 0.8, **(pred_kwargs or {})}

    n_dec, n_sess = len(decoder_specs), len(session_list)
    fig_kwargs = dict(figsize=(n_dec * figsize_per_panel[0],
                                n_sess * figsize_per_panel[1]))

    if dim == 2:
        fig, axes = plt.subplots(n_sess, n_dec, squeeze=False, **fig_kwargs)
    else:
        fig = plt.figure(**fig_kwargs)
        axes = None  # 3D axes are added on demand below

    for i, sess in enumerate(session_list):
        for j, (dk, label, color) in enumerate(decoder_specs):
            if dim == 2:
                ax = axes[i, j]
            else:
                ax = fig.add_subplot(n_sess, n_dec, i * n_dec + j + 1, projection='3d')
            plotted = False
            for truth, pred in _iter_session_polylines(polylines_per_decoder, dk, sess):
                _draw_polyline(ax, truth, dim=dim, **truth_kw)
                _draw_polyline(ax, pred, dim=dim, color=color, **pred_kw)
                plotted = True
            ax.tick_params(labelsize=5 if dim == 3 else 6)
            if not plotted:
                ax.set_visible(False)
                continue
            if dim == 2:
                ax.set_aspect('equal')
            if i == 0:
                ax.set_title(label, fontsize=8)
            if j == 0:
                lbl = session_label_fn(sess) if session_label_fn else sess
                if dim == 2:
                    ax.set_ylabel(lbl, fontsize=7)
                else:
                    ax.text2D(-0.2, 0.5, lbl, rotation=90, transform=ax.transAxes,
                              ha='right', va='center', fontsize=7)

    fig.legend(handles=_legend_handles(decoder_specs), loc='upper right',
               bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8)
    if suptitle:
        fig.suptitle(suptitle, y=1.0)
    rect_left = 0.02 if dim == 3 else 0
    fig.tight_layout(rect=[rect_left, 0, 0.97, 0.99])
    return fig


def plot_trajectory_zoom_3d(zoom_session, decoder_specs, polylines_per_decoder,
                            *, figsize_per_panel=(4.5, 4.5),
                            suptitle=None,
                            truth_kwargs=None,
                            pred_kwargs=None):
    """Single-session 3D zoom: 1 row × len(decoder_specs) columns."""
    truth_kw = {'color': 'k', 'alpha': 0.5, 'linewidth': 0.9, **(truth_kwargs or {})}
    pred_kw = {'alpha': 0.7, 'linewidth': 0.9, **(pred_kwargs or {})}
    n_dec = len(decoder_specs)
    fig = plt.figure(figsize=(n_dec * figsize_per_panel[0], figsize_per_panel[1]))
    for k, (dk, label, color) in enumerate(decoder_specs):
        ax = fig.add_subplot(1, n_dec, k + 1, projection='3d')
        plotted = False
        for truth, pred in _iter_session_polylines(polylines_per_decoder, dk, zoom_session):
            _draw_polyline(ax, truth, dim=3, **truth_kw)
            _draw_polyline(ax, pred, dim=3, color=color, **pred_kw)
            plotted = True
        ax.set_title(label, fontsize=11)
        ax.set_xlabel('x (m)', fontsize=9)
        ax.set_ylabel('y (m)', fontsize=9)
        ax.set_zlabel('z (m)', fontsize=9)
        if not plotted:
            ax.text2D(0.5, 0.5, '(no data)', transform=ax.transAxes,
                      ha='center', va='center', fontsize=9)
    fig.legend(handles=_legend_handles(decoder_specs), loc='upper right',
               frameon=False, fontsize=10)
    if suptitle:
        fig.suptitle(suptitle, y=1.02, fontsize=12)
    fig.tight_layout()
    return fig
