"""Trajectory geometry utilities shared across decoding notebooks.

Two trajectory comparisons that come up in every PV-Perceiver / NB / s2s
analysis:

  - **Mean squared jerk** (`compute_mean_squared_jerk`) — third time-derivative
    of position; a smoothness statistic on a (T, d) polyline.
  - **Discrete Fréchet distance** (`discrete_frechet`) — Eiter & Mannila 1994;
    "leash" distance between two polylines under any monotone reparameterisation.
    Penalises shape mismatch but not phase mismatch — complements per-timestep
    Euclidean error.

`longest_contiguous_run` is the boolean-mask helper used by MSJ to locate the
single longest valid stretch within a per-flight validity mask.
"""
from __future__ import annotations

import numpy as np
from numba import njit


def compute_mean_squared_jerk(positions, dt):
    """Mean squared jerk (third derivative magnitude) of a (T, d) trajectory.

    Returns NaN for trajectories shorter than 4 samples (`np.gradient` needs
    at least that many to produce three nested derivatives).
    """
    if positions.shape[0] < 4:
        return np.nan
    vel = np.gradient(positions, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    jerk = np.gradient(acc, dt, axis=0)
    return float(np.mean(jerk ** 2))


def longest_contiguous_run(mask):
    """Indices of the longest contiguous True segment in a 1D boolean array."""
    valid_indices = np.where(mask)[0]
    if len(valid_indices) == 0:
        return np.array([], dtype=int)
    breaks = np.where(np.diff(valid_indices) != 1)[0]
    segments = np.split(valid_indices, breaks + 1)
    return max(segments, key=len)


@njit(cache=True)
def _frechet_inner(D):
    n, m = D.shape
    ca = np.empty((n, m))
    ca[0, 0] = D[0, 0]
    for i in range(1, n):
        ca[i, 0] = ca[i - 1, 0] if ca[i - 1, 0] > D[i, 0] else D[i, 0]
    for j in range(1, m):
        ca[0, j] = ca[0, j - 1] if ca[0, j - 1] > D[0, j] else D[0, j]
    for i in range(1, n):
        for j in range(1, m):
            a = ca[i - 1, j]
            b = ca[i - 1, j - 1]
            c = ca[i, j - 1]
            mn = a if a < b else b
            if c < mn:
                mn = c
            ca[i, j] = mn if mn > D[i, j] else D[i, j]
    return ca[n - 1, m - 1]


def discrete_frechet(P, Q):
    """Discrete Fréchet distance between two polylines (Eiter & Mannila 1994).

    Same units as the inputs. The two polylines may have different lengths.
    `_frechet_inner` is njit-compiled — the first call pays a one-shot
    compile cost; repeated calls are fast.
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    n, m = len(P), len(Q)
    if n == 0 or m == 0:
        return np.nan
    if n == 1 and m == 1:
        return float(np.linalg.norm(P[0] - Q[0]))
    D = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=-1)
    return float(_frechet_inner(D))


__all__ = [
    'compute_mean_squared_jerk',
    'discrete_frechet',
    'longest_contiguous_run',
]
