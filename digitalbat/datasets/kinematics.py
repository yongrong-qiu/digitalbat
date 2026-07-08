"""Kinematic-augmented behaviour for the joint encode/decode model.

The standard 3-D-position behaviour vector can be optionally augmented with
kinematics: speed, **heading** (3 velocity-direction angles, circular -> encoded
as sin/cos so MSE never sees a wrap-around discontinuity), angular velocity
(gyro) and linear acceleration. Layout of the augmented 16-D vector::

    [ x, y, z | speed | sinθx, sinθy, sinθz, cosθx, cosθy, cosθz | ax, ay, az | gx, gy, gz ]
      pos(3)   spd(1)            heading sin/cos (6)               accel(3)     gyro(3)

Position / speed / accel / gyro stay in the loader's z-scored units; the heading
sin/cos channels are left raw in [-1, 1] (already unit-scale). Decoded heading is
recovered as ``atan2(sin, cos)``.

Use :func:`make_joint_dataloaders` to load once and get position-only (dim-3) and
kinematic (dim-16) dataloaders sharing identical flights/splits, so an Optuna
``use_kinematics`` flag can switch between them without reloading.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import bat_data_loader, BehaviorSpikeDataset

__all__ = ['heading_sincos', 'augment_behavior', 'make_joint_dataloaders', 'KIN_BEHAVIOR_DIM']

KIN_BEHAVIOR_DIM = 16
# index layout of the raw (use_speed_head_dir + use_IMU) behaviour vector
_POS, _SPEED, _VELU, _ACC, _GYR = slice(0, 3), slice(3, 4), slice(4, 7), slice(7, 10), slice(10, 13)


def heading_sincos(vel_unit):
    """Raw velocity-unit-vector ``(..., 3)`` -> ``(..., 6)`` sin/cos of the three
    heading angles (atan2 per coordinate plane, matching ``glm.py``). Circular-safe:
    the (sin, cos) pair has no wrap-around discontinuity, so MSE is a proper loss."""
    vx, vy, vz = vel_unit[..., 0], vel_unit[..., 1], vel_unit[..., 2]
    ang = np.stack([np.arctan2(vz, vy), np.arctan2(vx, vz), np.arctan2(vy, vx)], axis=-1)
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=-1)


def augment_behavior(behavior, mean, std):
    """``(..., 13)`` z-scored behaviour -> ``(..., 16)`` kinematic-augmented vector.

    ``mean`` / ``std`` are the per-channel z-score stats (``meta_info_dict``) used to
    un-z-score the velocity unit vector before computing heading angles.
    """
    mean = np.asarray(mean).reshape(-1)
    std = np.asarray(std).reshape(-1)
    raw_velu = behavior[..., _VELU] * std[_VELU] + mean[_VELU]      # un-z-score for the angles
    head = heading_sincos(raw_velu)                                # (..., 6)
    return np.concatenate([behavior[..., _POS], behavior[..., _SPEED], head,
                           behavior[..., _ACC], behavior[..., _GYR]], axis=-1)


def _rebuild(dl, new_behavior):
    """A DataLoader identical to ``dl`` (sampler, batch size, spike/mask/index tensors)
    but with ``behavior`` replaced by ``new_behavior`` (numpy ``(N, T, C)``)."""
    ds = dl.dataset
    samples = (new_behavior.astype(np.float32), ds.spike.numpy(), ds.mask.numpy(),
               ds.flight_path_index.numpy(), ds.flight_type_index.numpy())
    new_ds = BehaviorSpikeDataset(samples=samples, behavior_mean=None, behavior_std=None)
    return DataLoader(new_ds, sampler=dl.sampler, batch_size=dl.batch_size)


def make_joint_dataloaders(data_config):
    """Load once with kinematics on; return ``(dl_pos, dl_kin, unit_num_dict, meta)``.

    ``dl_pos`` is the dim-3 position-only loader and ``dl_kin`` the dim-16 augmented
    loader, sharing identical flights/splits (same load). Both are
    ``{split: {session: DataLoader}}``.
    """
    dc = dict(data_config)
    dc.update(use_speed_head_dir=True, use_IMU_flag=True)
    dl, bdim, units, meta = bat_data_loader(**dc)
    if bdim != 13:
        raise ValueError(f'expected raw kinematic behavior_dim=13, got {bdim}')
    dl_pos, dl_kin = {}, {}
    for split, sess in dl.items():
        dl_pos[split], dl_kin[split] = {}, {}
        for s, loader in sess.items():
            beh = loader.dataset.behavior.numpy()                  # (N, T, 13) z-scored
            dl_pos[split][s] = _rebuild(loader, beh[..., _POS])
            dl_kin[split][s] = _rebuild(loader, augment_behavior(
                beh, meta['behavior_mean'][s], meta['behavior_std'][s]))
    return dl_pos, dl_kin, units, meta
