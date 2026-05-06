"""Shared constants and helpers for preprocess-data-v0j-share.ipynb.

Reward/perch target locations, distance thresholds, spatial-bin widths, and
target-proximity helpers used across the non-flight Mantel, rate-map,
place-field, and Jacobian-magnification analyses.
"""
import numpy as np


# Spatial bin width (meters) for 2D rate maps and binned analyses.
loc_bin_width_rm = 0.3

# Time-warp bin count for place-field analysis.
N_BINS_PF = 50

# Per-dataset reward/perch locations. Dataset 2 uses a different setup with
# both targets on the -x side; Datasets 1 and 3 share the original layout.
_FLIGHT_ROOM = dict(
    rew_locs=np.array([
        [2.80, -0.86, 1.57],
        [2.75,  0.96, 1.57],
    ]),
    perch_locs=np.array([
        [-2.55, 1.57, 1.82],
        [-1.98, 2.08, 1.74],
    ]),
)

DATASET_TARGETS = {
    'Dataset_1': _FLIGHT_ROOM,
    'Dataset_2': dict(
        rew_locs=np.array([[-3.00, 0.70, 1.90]]),
        perch_locs=np.array([[-3.20, 1.80, 2.20]]),
    ),
    'Dataset_3': _FLIGHT_ROOM,
}

DIST_THRESH = 0.6  # meters

# Flat D1/3 reference constants for plots that still use the original markers.
REW_LOCS = DATASET_TARGETS['Dataset_1']['rew_locs']
PERCH_LOCS = DATASET_TARGETS['Dataset_1']['perch_locs']


def get_target_locs(session_name):
    """Return (rew_locs, perch_locs) for the dataset the session belongs to."""
    prefix = '_'.join(session_name.split('_')[:2])
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
