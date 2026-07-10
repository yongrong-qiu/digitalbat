"""ED Fig 2 "decode-cost" generation pipeline.

Faithful extraction of the per-flight Fréchet + reconstruction-error decode-cost
pipeline from ``notebooks/decode-position-comparison.ipynb`` so that BOTH that
notebook and ``figures.ipynb`` can import the leaf functions instead of defining
them inline.

It produces per-flight Fréchet + reconstruction-error tables for 4 decoders
(``glm_map``, ``glm_map_coup``, ``glm_map_joint``, ``transformer``) across up to
6 modes (``real``, ``warp_only``, ``trial_shuffled``, ``time_shuffled``,
``both_shuffled``, ``synthetic``, ``synthetic_traj``).

The leaf functions are ported VERBATIM from the notebook cells (only
notebook-global references turned into explicit parameters; print/plot/legacy
lines dropped). Numerical fidelity is verified downstream against a canonical
pickle, so the internal logic / variable names / operation order are preserved.

Sections
--------
- Surrogate dataloaders
- Clustering
- Per-mode driver
- GLM-inversion
- Transformer
- Aggregation / export
- Orchestrator
"""

import copy as _copy
import os
from collections import Counter
from functools import partial as _partial

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from scipy.interpolate import interp1d
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# Reused package leaf functions.  The package `make_nb` / `make_nb_continuity`
# take `dt` explicitly (the notebook versions closed over
# `data_config['bin_width']`) but are otherwise identical, so we reuse them and
# pass `dt=data_config['bin_width']` at every call site.
from digitalbat.decoding.nb_pipeline import (
    gather_session_arrays, make_nb, make_nb_continuity,
)
from digitalbat.decoding import discrete_frechet
# `frechet_distance` lives in utility.utils (not exported from decoding.geometry).
from digitalbat.utility.utils import frechet_distance, set_random_state
from digitalbat.models.glm import (
    GLM_Coupled, make_log_raised_cosine_basis, _OTHER_CLUSTER_KEY,
)
from digitalbat.models.decoder import (
    GLM_MAP_Decoder, Bat_Transformer_Decoder, bat_decoder_trainer,
)
from digitalbat.models.joint import JointMaskedTransformer
from digitalbat.training.joint_trainer import train_joint, scheme_weights
from digitalbat.datasets.synthetic import (
    make_synthetic_train_dataloaders, make_synthetic_train_dataloaders_traj,
)
from digitalbat.datasets.data import bat_data_loader


# ---------------------------------------------------------------------------
# Module-level constants (fixed in the notebook; kept as constants here).
# ---------------------------------------------------------------------------
DS_CLUS = 7              # downsample length (matches FlightClusters ds_clus=7)
CLUSTER_ALPHA = 0.8      # single-linkage threshold
MIN_CLUSTER_FLIGHTS = 3  # train clusters with < this many flights -> UNCLUSTERED
CLUSTER_TAG = f'trainonlyclus_n{DS_CLUS}_a{CLUSTER_ALPHA:.2f}_m{MIN_CLUSTER_FLIGHTS}'
MAX_ALPHA = 1.5          # ceiling when bumping α to escape a single-cluster fit
UNCLUSTERED = -1

# Joint (pos × traj) clustering — matches model_comparison.ipynb.
CLUSTER_DS_JOINT = 7
CLUSTER_ALPHA_JOINT = 0.8
CLUSTER_MAX_ALPHA_JOINT = 1.5

# Place-GLM synthetic-baseline training config.
GLM_SYNTH_EPOCHS = 300
GLM_SYNTH_LR = 1e-2
GLM_SYNTH_WD = 1.0
GLM_SYNTH_PATIENCE = 20

# GLM basis config.
HIST_N_BASIS, COUP_N_BASIS, N_LAGS, BIN_WIDTH = 6, 4, 10, 0.10

# Each row is (warp, permute, time_shuffle, name).
SHUFFLE_CONFIGS = [
    (False, False, False, 'real'),
    (True,  False, False, 'warp_only'),
    (True,  True,  False, 'trial_shuffled'),
    (False, False, True,  'time_shuffled'),
    (True,  True,  True,  'both_shuffled'),
]

# Transformer decoder training constants (from transformer-build.py).
MAX_EPOCHS = 1000
DATA_TAG = '_acc50_preln_17_sess'

# Modes whose *training* data depends on per-flight cluster IDs — they need
# fresh checkpoints whenever the clustering changes.
CLUSTER_DEPENDENT_MODES = {'warp_only', 'trial_shuffled',
                           'both_shuffled', 'synthetic_traj'}

# Model order for the 4-decoder ED Fig 2 export.
MODEL_ORDER_4G = [
    'naive_bayes',
    'naive_bayes_lag',
    'nb_continuity',
    'naive_bayes_cluster',
    'nb_continuity_cluster',
    'svm',
    'svm_cluster',
    'glm_map',
    'glm_map_coup',
    'glm_map_joint',
    'transformer',
    'joint',
]
MODEL_LABEL_4G = {
    'naive_bayes':            'NB (place)',
    'naive_bayes_cluster':    'NB (place, trajectory)',
    'naive_bayes_lag':        'NB (place) + lag',
    'nb_continuity':          'NB (place) + continuity',
    'glm_map':                'GLM-inversion (correlation-aware)',
    'glm_map_coup':           'GLM-inversion (coup, no hist)',
    'glm_map_joint':          'GLM-inversion (joint pos × traj)',
    'nb_continuity_cluster':  'NB (place, trajectory) + continuity',
    'svm':                    'pairwise SVM (place)',
    'svm_cluster':            'pairwise SVM (place, trajectory)',
    'transformer':            'transformer',
    'joint':                  'Hippo-NEDS',
}

# Joint (Hippo-NEDS) decoder defaults — the consensus masked encoder/decoder used
# as a *decoder* (all behaviour masked → predict 3-D position).  Overridable via
# generate_ed2_decode_costs(joint_params=..., joint_train=...) so the notebook can
# pass its parsed consensus hyperparameters.  Trained fresh per mode on that mode's
# surrogate spikes (real valid/test by reference), in the cost-pipeline session
# universe (existing joint checkpoints live in the 17-session universe and do not
# transfer key-for-key).
JOINT_DECODER_PARAMS = dict(num_layers=4, latent_dim=64, num_heads=4, ff_dim=64,
                            dropout=0.2)
JOINT_DECODER_INPUT_TRANSFORM = 'identity'
# Consensus training config (lam/pos_weight/scheme weights parsed from the
# production checkpoint joint_v2_cross1000_..._lam0.68_pw0.36_we2.8_wd2.2_wa0.31).
# `scheme_weights` requires w_encode / w_decode / w_aux.
JOINT_DECODER_TRAIN = dict(n_epochs=1000, lr=3e-3, weight_decay=1e-2, lam=0.68,
                           pos_weight=0.36, w_encode=2.8, w_decode=2.2, w_aux=0.31,
                           fde_weight=0.5, err_ref=0.3, fde_ref=0.1)
DATA_ORDER_4G = ['real', 'trial_shuffled', 'time_shuffled',
                 'both_shuffled', 'synthetic', 'synthetic_traj']
DATA_LABEL_4G = {
    'real':            'real',
    'trial_shuffled':  'trial-shuffled',
    'time_shuffled':   'time-shuffled',
    'both_shuffled':   'both-shuffled',
    'synthetic':       'synthetic (place)',
    'synthetic_traj':  'synthetic (place, traj)',
}

# Fréchet prediction-column map (verbatim from frechet-compute / s4g-data-prep).
TRUTH_3D = ['x_true', 'y_true', 'z_true']
DECODER_PRED_COLS = [
    ('naive_bayes',           ('x_pred_2d', 'y_pred_2d'), ('x_pred_3d', 'y_pred_3d', 'z_pred_3d')),
    ('naive_bayes_lag',       ('x_pred_2d', 'y_pred_2d'), ('x_pred_3d', 'y_pred_3d', 'z_pred_3d')),
    ('nb_continuity',         ('x_pred_2d', 'y_pred_2d'), ('x_pred_3d', 'y_pred_3d', 'z_pred_3d')),
    ('glm_map',               ('x_pred_2d', 'y_pred_2d'), ('x_pred_3d', 'y_pred_3d', 'z_pred_3d')),
    ('glm_map_coup',          ('x_pred_2d', 'y_pred_2d'), ('x_pred_3d', 'y_pred_3d', 'z_pred_3d')),
    ('glm_map_joint',         ('x_pred_2d', 'y_pred_2d'), ('x_pred_3d', 'y_pred_3d', 'z_pred_3d')),
    ('svm',                   ('x_pred_2d', 'y_pred_2d'), ('x_pred_3d', 'y_pred_3d', 'z_pred_3d')),
    ('svm_cluster',           ('x_pred_2d', 'y_pred_2d'), ('x_pred_3d', 'y_pred_3d', 'z_pred_3d')),
    ('naive_bayes_cluster',   ('x_pred_2d', 'y_pred_2d'), ('x_pred_3d', 'y_pred_3d', 'z_pred_3d')),
    ('nb_continuity_cluster', ('x_pred_2d', 'y_pred_2d'), ('x_pred_3d', 'y_pred_3d', 'z_pred_3d')),
    ('transformer',           ('x_pred', 'y_pred'),       ('x_pred', 'y_pred', 'z_pred')),
    ('joint',                 ('x_pred', 'y_pred'),       ('x_pred', 'y_pred', 'z_pred')),
    ('cebra',                 ('x_pred', 'y_pred'),       ('x_pred', 'y_pred', 'z_pred')),
    ('pv_perceiver_rel',      ('x_pred', 'y_pred'),       ('x_pred', 'y_pred', 'z_pred')),
    ('pv_perceiver_abs',      ('x_pred', 'y_pred'),       ('x_pred', 'y_pred', 'z_pred')),
]
_CLUSTER_PRED_COLS_4G = [
    ('naive_bayes_cluster',     ('x_pred_2d', 'y_pred_2d'),
     ('x_pred_3d', 'y_pred_3d', 'z_pred_3d')),
    ('nb_continuity_cluster',   ('x_pred_2d', 'y_pred_2d'),
     ('x_pred_3d', 'y_pred_3d', 'z_pred_3d')),
]


def parse_bat(session_name):
    """`Dataset_3_14543_240419` -> '14543'."""
    return session_name.split('_')[2]


# ===========================================================================
# ## Surrogate dataloaders
# ===========================================================================
def _phantom_warp_flight(spk_d, beh_d, L_d, L_new):
    """Warp one flight's binned spikes + behavior from L_d to L_new bins.

    Phantom-spike rebinning per neuron: each integer count `c` at bin
    `b` becomes `c` evenly-spaced phantoms at sub-bin positions
    `b + (i + 0.5) / c`, those continuous positions are scaled by
    `L_new / L_d`, then rehistogrammed into [0, L_new) integer bins.
    Conserves per-flight per-neuron spike totals exactly.

    Behavior is linearly interpolated at the new bin centers
    (z-scored space; linear interpolation preserves z-score
    semantics).
    """
    K = spk_d.shape[1]
    new_spk = np.zeros((L_new, K), dtype=spk_d.dtype)
    scale = L_new / L_d
    for k in range(K):
        bins_with_spikes = np.where(spk_d[:, k] > 0)[0]
        if len(bins_with_spikes) == 0:
            continue
        phantoms = np.concatenate([
            b + (np.arange(int(spk_d[b, k])) + 0.5) / int(spk_d[b, k])
            for b in bins_with_spikes
        ])
        warped = phantoms * scale
        idx = np.clip(warped.astype(int), 0, L_new - 1)
        np.add.at(new_spk[:, k], idx, 1)

    old_centers = np.arange(L_d) + 0.5
    new_centers_in_old = (np.arange(L_new) + 0.5) / scale
    new_beh = np.empty((L_new, beh_d.shape[1]), dtype=beh_d.dtype)
    for c_dim in range(beh_d.shape[1]):
        new_beh[:, c_dim] = np.interp(
            new_centers_in_old, old_centers, beh_d[:L_d, c_dim],
        )
    return new_spk, new_beh


def _shuffled_dataset(ds, warp, permute, time_shuffle, rng):
    """Build a dataset variant by composing up to three independent
    transformations:
      warp          — phantom-spike-rebin every flight in a cluster to
                      the cluster's mean valid length (touches spikes
                      AND behavior).
      permute       — per-neuron permutation of flights within a
                      cluster (only meaningful after warp, since it
                      relies on all members of a cluster being the
                      same length).
      time_shuffle  — per-flight, identical-across-neurons permutation
                      of valid timesteps (applies to spikes AND
                      behavior, so position-tuning is preserved).
    """
    spk = ds.spike.numpy().copy()
    beh = ds.behavior.numpy().copy()
    msk = ds.mask.numpy().astype(bool)
    fti = ds.flight_type_index.numpy().astype(int)
    N, T, K = spk.shape
    lengths = msk.sum(axis=1).astype(int)

    if warp:
        for cluster in np.unique(fti):
            grp = np.where(fti == cluster)[0]
            if len(grp) < 2:
                continue
            L_avg = int(round(lengths[grp].mean()))
            if L_avg < 2 or L_avg > T:
                continue
            for d in grp:
                L_d = int(lengths[d])
                if L_d < 2 or L_d == L_avg:
                    continue
                new_spk_d, new_beh_d = _phantom_warp_flight(
                    spk[d, :L_d], beh[d, :L_d], L_d, L_avg)
                spk[d, :L_avg] = new_spk_d
                spk[d, L_avg:] = 0
                beh[d, :L_avg] = new_beh_d
                beh[d, L_avg:] = 0
                lengths[d] = L_avg

    if permute:
        donor_spk = spk.copy()
        for cluster in np.unique(fti):
            grp = np.where(fti == cluster)[0]
            if len(grp) < 2:
                continue
            for k in range(K):
                perm = rng.permutation(grp)
                for d_idx, d in enumerate(grp):
                    dp = int(perm[d_idx])
                    if dp == d:
                        continue
                    L = int(lengths[d])
                    if L != int(lengths[dp]):
                        continue  # safety: only swap same-length flights
                    spk[d, :L, k] = donor_spk[dp, :L, k]

    if time_shuffle:
        for d in range(N):
            L = int(lengths[d])
            if L < 2:
                continue
            perm = rng.permutation(L)
            spk[d, :L] = spk[d, perm]
            beh[d, :L] = beh[d, perm]

    # Mask reflects current per-flight valid lengths (changes only
    # under warp).
    new_msk = np.zeros_like(msk)
    for d in range(N):
        new_msk[d, :int(lengths[d])] = True

    new_ds = _copy.copy(ds)
    new_ds.spike = torch.from_numpy(spk).float()
    new_ds.behavior = torch.from_numpy(beh).float()
    new_ds.mask = torch.from_numpy(new_msk.astype(np.float32))
    return new_ds


def build_shuffled_dataloaders(dataloaders, warp, permute, time_shuffle, seed):
    rng = np.random.default_rng(seed)
    out = {'train': {}, 'valid': {}, 'test': {}}
    for split in ('train', 'valid', 'test'):
        for sn, loader in dataloaders[split].items():
            new_ds = _shuffled_dataset(
                loader.dataset, warp, permute, time_shuffle, rng)
            out[split][sn] = DataLoader(
                new_ds, sampler=loader.sampler, batch_size=loader.batch_size,
            )
    return out


def build_mode_dataloaders(dataloaders, meta_info_dict, data_config, *, seed,
                           run_shuffles=True, run_synthetic=True,
                           synth_sampling='empirical'):
    """Build `mode_dataloaders` / `modes`, porting the orchestration tail of
    shuffled-build.py.  Keeps all 5 SHUFFLE_CONFIGS (real, warp_only,
    trial_shuffled, time_shuffled, both_shuffled) + synthetic + synthetic_traj.
    The NB-calibration `synthetic_test_loaders` block is intentionally dropped.
    """
    mode_dataloaders = {'real': dataloaders}
    if run_shuffles:
        for warp, permute, time, name in SHUFFLE_CONFIGS:
            if name == 'real':
                continue
            # Same seed for every config so configs that share a step reuse the
            # same underlying permutations as far as the RNG state allows.
            mode_dataloaders[name] = build_shuffled_dataloaders(
                dataloaders, warp, permute, time, seed=seed)

        # Asymmetric test-set override for the trial-shuffle pipeline: training
        # stays trial-shuffled but valid + test are swapped back to real.
        for _override in ('warp_only', 'trial_shuffled'):
            if _override in mode_dataloaders:
                mode_dataloaders[_override]['valid'] = dataloaders['valid']
                mode_dataloaders[_override]['test'] = dataloaders['test']

    if run_synthetic:
        # Position-only simulator (train resampled, valid/test by reference).
        syn_dataloaders, simulators = make_synthetic_train_dataloaders(
            dataloaders, meta_info_dict, dt=data_config['bin_width'],
            mode=synth_sampling, seed=seed)
        assert syn_dataloaders['valid'] is dataloaders['valid']
        assert syn_dataloaders['test'] is dataloaders['test']
        mode_dataloaders['synthetic'] = syn_dataloaders

        # Trajectory-specific simulator (per session × flight-cluster).
        syn_traj_dataloaders, simulators_traj = make_synthetic_train_dataloaders_traj(
            dataloaders, meta_info_dict, dt=data_config['bin_width'],
            mode=synth_sampling, seed=seed)
        assert syn_traj_dataloaders['valid'] is dataloaders['valid']
        assert syn_traj_dataloaders['test'] is dataloaders['test']
        mode_dataloaders['synthetic_traj'] = syn_traj_dataloaders

    modes = list(mode_dataloaders.keys())
    return mode_dataloaders, modes


# ===========================================================================
# ## Clustering
# ===========================================================================
def _downsample_3d(pos_xyz, mask, ds_clus=DS_CLUS):
    """(N, T, 3) + (N, T) mask → (N, ds_clus, 3).  NaN row if L_valid < 2."""
    N = pos_xyz.shape[0]
    out = np.full((N, ds_clus, 3), np.nan, dtype=np.float64)
    for i in range(N):
        L = int(mask[i].sum())
        if L < 2:
            continue
        traj = pos_xyz[i, mask[i]]                 # (L, 3)
        new_idx = np.linspace(0, L - 1, ds_clus)
        for d in range(3):
            out[i, :, d] = interp1d(np.arange(L), traj[:, d], kind='linear')(new_idx)
    return out


def _session_split_trajs(dataloaders, sn, meta_info_dict):
    bm = np.asarray(meta_info_dict['behavior_mean'][sn]).reshape(-1)
    bs = np.asarray(meta_info_dict['behavior_std'][sn]).reshape(-1)
    out = {}
    for split in ('train', 'valid', 'test'):
        ds = dataloaders[split][sn].dataset
        pos = (ds.behavior.numpy() * bs + bm)[..., :3]
        msk = ds.mask.numpy().astype(bool)
        out[split] = _downsample_3d(pos, msk)
    return out


def refit_train_clusters_inplace(dataloaders, meta_info_dict):
    """Mutate each dataset.flight_type_index to use train-only clusters
    + nearest-neighbour reassignment for valid/test."""
    info = {}
    for sn in dataloaders['train'].keys():
        base_alpha = CLUSTER_ALPHA
        traj = _session_split_trajs(dataloaders, sn, meta_info_dict)

        train_traj = traj['train']
        keep = np.where(~np.isnan(train_traj[:, 0, 0]))[0]
        n = len(keep)
        if n < 2:
            continue
        X = train_traj[keep]
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = D[j, i] = frechet_distance(X[i], X[j])

        # Single-linkage at α; bump α up to MAX_ALPHA if everything
        # collapses into one cluster (matches FlightClusters.compute).
        used_alpha = base_alpha
        while True:
            Z = linkage(squareform(D, checks=False), method='single')
            train_labels = fcluster(Z, t=used_alpha, criterion='distance').astype(int)
            if len(np.unique(train_labels)) > 1 or used_alpha >= MAX_ALPHA:
                break
            used_alpha += 0.1

        # Pool small train clusters into UNCLUSTERED; only "major" clusters
        # (>= MIN_CLUSTER_FLIGHTS members) survive and can receive reassignments.
        counts = Counter(train_labels.tolist())
        majset = {int(c) for c, k in counts.items() if k >= MIN_CLUSTER_FLIGHTS}

        new_fti = {}
        train_full = np.full(train_traj.shape[0], UNCLUSTERED, dtype=np.int64)
        train_full[keep] = [lab if lab in majset else UNCLUSTERED for lab in train_labels]
        new_fti['train'] = train_full
        for split in ('valid', 'test'):
            t = traj[split]
            out = np.full(t.shape[0], UNCLUSTERED, dtype=np.int64)
            for i in range(t.shape[0]):
                if np.isnan(t[i, 0, 0]):
                    continue
                dists = np.array([frechet_distance(t[i], X[j]) for j in range(n)])
                j_min = int(np.argmin(dists))
                lab = int(train_labels[j_min])
                if dists[j_min] <= used_alpha and lab in majset:
                    out[i] = lab
            new_fti[split] = out

        for split, fti in new_fti.items():
            ds = dataloaders[split][sn].dataset
            ds.flight_type_index = torch.from_numpy(fti.astype(np.float32))

        info[sn] = dict(
            alpha=used_alpha,
            n_train=n,
            n_clusters=len(majset),
            unclust_valid=int((new_fti['valid'] == UNCLUSTERED).sum()),
            n_valid=int(new_fti['valid'].size),
            unclust_test=int((new_fti['test'] == UNCLUSTERED).sum()),
            n_test=int(new_fti['test'].size),
        )
    return info


def _downsample_traj_joint(pos_xyz, mask, ds=CLUSTER_DS_JOINT):
    N = pos_xyz.shape[0]
    out = np.full((N, ds, 3), np.nan)
    for i in range(N):
        L = int(mask[i].sum())
        if L < 2:
            continue
        traj = pos_xyz[i, mask[i]]
        new_idx = np.linspace(0, L - 1, ds)
        for d in range(3):
            out[i, :, d] = interp1d(np.arange(L), traj[:, d])(new_idx)
    return out


def compute_joint_flight_clusters(dataloaders, meta_info_dict, *,
                                  alpha=CLUSTER_ALPHA_JOINT, ds=CLUSTER_DS_JOINT,
                                  min_flights=MIN_CLUSTER_FLIGHTS):
    """Reproduce model_comparison.ipynb's compute_flight_clusters.

    Returns ({session: {flight_path_index: cluster_id}}, {session: [major ids]}).
    Cluster ids are scipy fcluster labels on TRAIN flights; valid/test flights
    take their nearest train flight's label iff Frechet <= alpha else
    UNCLUSTERED; train clusters with < min_flights pooled into 'other'.
    """
    flight_cluster, cluster_keys = {}, {}
    for sn in dataloaders['train'].keys():
        bm = np.asarray(meta_info_dict['behavior_mean'][sn]).reshape(-1)
        bs = np.asarray(meta_info_dict['behavior_std'][sn]).reshape(-1)
        trajs, fpis = {}, {}
        for split in ('train', 'valid', 'test'):
            dset = dataloaders[split][sn].dataset
            pos = (dset.behavior.numpy() * bs + bm)[..., :3]
            trajs[split] = _downsample_traj_joint(pos, dset.mask.numpy().astype(bool), ds=ds)
            fpis[split] = dset.flight_path_index.numpy().astype(int)
        Xtr = trajs['train']
        keep = np.where(~np.isnan(Xtr[:, 0, 0]))[0]
        X, n = Xtr[keep], len(keep)
        mapping = {}
        if n < 2:
            for split in ('train', 'valid', 'test'):
                for f in fpis[split]:
                    mapping[int(f)] = UNCLUSTERED
            flight_cluster[sn], cluster_keys[sn] = mapping, []
            continue
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = D[j, i] = frechet_distance(X[i], X[j])
        used_alpha = alpha
        while True:
            Z = linkage(squareform(D, checks=False), method='single')
            labels = fcluster(Z, t=used_alpha, criterion='distance').astype(int)
            if len(np.unique(labels)) > 1 or used_alpha >= CLUSTER_MAX_ALPHA_JOINT:
                break
            used_alpha += 0.1
        counts = Counter(labels)
        major = sorted(int(c) for c, k in counts.items() if k >= min_flights)
        majset = set(major)
        for p, gi in enumerate(keep):
            lab = int(labels[p])
            mapping[int(fpis['train'][gi])] = lab if lab in majset else UNCLUSTERED
        for gi in range(Xtr.shape[0]):
            mapping.setdefault(int(fpis['train'][gi]), UNCLUSTERED)
        for split in ('valid', 'test'):
            t = trajs[split]
            for i in range(t.shape[0]):
                f = int(fpis[split][i])
                if np.isnan(t[i, 0, 0]):
                    mapping[f] = UNCLUSTERED
                    continue
                dists = np.array([frechet_distance(t[i], X[j]) for j in range(n)])
                jm = int(np.argmin(dists))
                lab = int(labels[jm])
                mapping[f] = lab if (dists[jm] <= used_alpha and lab in majset) else UNCLUSTERED
        flight_cluster[sn], cluster_keys[sn] = mapping, major
    return flight_cluster, cluster_keys


# ===========================================================================
# ## Per-mode driver
# ===========================================================================
def _gather_with_cluster(loader, bm, bs):
    """gather_session_arrays + per-timestep flight_type_index."""
    spk, beh, msk, fpi, fti = [], [], [], [], []
    for batch in loader:
        spk.append(batch.spike.numpy())
        beh.append(batch.behavior.numpy())
        msk.append(batch.mask.numpy().astype(bool))
        fpi.append(batch.flight_path_index.numpy())
        fti.append(batch.flight_type_index.numpy())
    spk = np.concatenate(spk, axis=0)
    beh = np.concatenate(beh, axis=0)
    msk = np.concatenate(msk, axis=0)
    fpi = np.concatenate(fpi, axis=0)
    fti = np.concatenate(fti, axis=0)
    bm_a = np.asarray(bm).squeeze()
    bs_a = np.asarray(bs).squeeze()
    pos = beh[..., :3] * bs_a[None, None, :3] + bm_a[None, None, :3]
    B, T = msk.shape
    fpi_t = np.broadcast_to(fpi[:, None], (B, T))
    fti_t = np.broadcast_to(fti[:, None], (B, T))
    t_idx = np.broadcast_to(np.arange(T)[None, :], (B, T))
    return {
        'spikes': spk[msk],
        'pos_x': pos[..., 0][msk],
        'pos_y': pos[..., 1][msk],
        'pos_z': pos[..., 2][msk],
        'flight_id': fpi_t[msk].astype(int),
        'flight_type': fti_t[msk].astype(int),
        't_idx': t_idx[msk].astype(int),
    }


def _records_from_routed(arr, pred_2d, pred_3d, session_name, bat):
    if pred_2d is None or pred_3d is None:
        return []
    p2d_true = np.column_stack([arr['pos_x'], arr['pos_y']])
    p3d_true = np.column_stack([arr['pos_x'], arr['pos_y'], arr['pos_z']])
    valid = ~np.isnan(pred_2d[:, 0])
    out = []
    for i in np.where(valid)[0]:
        err_2d = float(np.sqrt(((p2d_true[i] - pred_2d[i]) ** 2).sum()))
        err_3d = float(np.sqrt(((p3d_true[i] - pred_3d[i]) ** 2).sum()))
        out.append({
            'bat': bat,
            'session': session_name,
            'flight_id': int(arr['flight_id'][i]),
            't_idx': int(arr['t_idx'][i]),
            'flight_type': int(arr['flight_type'][i]),
            'x_true': float(p3d_true[i, 0]),
            'y_true': float(p3d_true[i, 1]),
            'z_true': float(p3d_true[i, 2]),
            'x_pred_2d': float(pred_2d[i, 0]),
            'y_pred_2d': float(pred_2d[i, 1]),
            'x_pred_3d': float(pred_3d[i, 0]),
            'y_pred_3d': float(pred_3d[i, 1]),
            'z_pred_3d': float(pred_3d[i, 2]),
            'err_2d': err_2d,
            'err_3d': err_3d,
        })
    return out


def run_per_mode_per_session(fit_eval_fn, decoder_name, *, modes, dataloaders,
                             mode_dataloaders, meta_info_dict, needs_valid=False):
    """Shared driver: iterates (session, mode), accumulates dfs by mode.

    `fit_eval_fn(train_loader, [valid_loader,] test_loader, session_name, ..., eval_train_loader=...)`
    must return either (records, decoders) or (records, decoders, extras);
    here we keep only records (and `extras` if returned, for nb-lag).
    """
    records_by_mode = {m: {'test': [], 'train': []} for m in modes}
    decoders_by_mode = {m: {} for m in modes}
    extras_by_mode = {m: {} for m in modes}
    for session_name in dataloaders['test'].keys():
        for mode in modes:
            train_loader = mode_dataloaders[mode]['train'][session_name]
            test_loader = mode_dataloaders[mode]['test'][session_name]
            # For modes where training data is a surrogate but test is
            # real (synthetic, synthetic_traj, trial_shuffled, warp_only),
            # evaluate the 'train' split on real-train spikes so the
            # train-side numbers reflect the decoder's performance on
            # REAL data — matching the train+test = real baseline.
            eval_train_loader = (
                dataloaders['train'][session_name]
                if mode in ('synthetic', 'synthetic_traj',
                            'trial_shuffled', 'warp_only')
                else None
            )
            args = [train_loader]
            if needs_valid:
                args.append(mode_dataloaders[mode]['valid'][session_name])
            args.append(test_loader)
            out = fit_eval_fn(
                *args, session_name, meta_info_dict,
                eval_train_loader=eval_train_loader,
            )
            if out[0] is None:
                continue
            rec, dec = out[0], out[1]
            records_by_mode[mode]['test'].extend(rec['test'])
            records_by_mode[mode]['train'].extend(rec['train'])
            decoders_by_mode[mode][session_name] = dec
            if len(out) >= 3:
                extras_by_mode[mode][session_name] = out[2]

    dfs = {m: pd.DataFrame(r['test']).assign(decoder=decoder_name)
           for m, r in records_by_mode.items()}
    train_dfs = {m: pd.DataFrame(r['train']).assign(decoder=decoder_name)
                 for m, r in records_by_mode.items()}
    return dfs, train_dfs, decoders_by_mode, extras_by_mode


# ===========================================================================
# ## GLM-inversion
# ===========================================================================
def make_glm_bases(bin_width):
    """Build the GLM history / coupling bases + `glm_common` dict.

    Bases + config MUST match model_comparison.ipynb's setup/build-models cells
    so the checkpoints load. Returns (glm_hist_basis, glm_coup_basis, glm_common).
    """
    glm_hist_basis, _, _ = make_log_raised_cosine_basis(HIST_N_BASIS, N_LAGS, BIN_WIDTH)
    glm_coup_basis, _, _ = make_log_raised_cosine_basis(
        COUP_N_BASIS, N_LAGS, BIN_WIDTH, include_zero_lag=True)
    glm_common = dict(n_bases_per_dim=16, pos_range=(-3.0, 3.0),
                      width_to_spacing_ratio=4, activation='exp',
                      bin_width=bin_width)
    return glm_hist_basis, glm_coup_basis, glm_common


def make_glm_encoders(unit_num_dict, glm_common, glm_hist_basis, glm_coup_basis,
                      coup_only_basis, N_LAGS, ckpt_map, device, dataloaders,
                      meta_info_dict):
    """Load glm_hist_coup + glm_coup + glm_joint and build ablation dicts +
    joint clustering.  Returns everything the GLM fit-eval closures need.

    `ckpt_map`: {'glm_map': path, 'glm_coup': path, 'glm_joint': path}.
    `coup_only_basis`: the coupling basis for glm_coup (== glm_coup_basis).
    """
    # --- glm_hist_coup --------------------------------------------------------
    GLM_MAP_CKPT = ckpt_map['glm_map']
    assert os.path.exists(GLM_MAP_CKPT), f'Missing GLM checkpoint {GLM_MAP_CKPT}.'
    glm_hist_coup = GLM_Coupled(
        output_dim_dict=unit_num_dict, **glm_common,
        hist_basis=glm_hist_basis, coup_basis=glm_coup_basis, n_lags=N_LAGS)
    _gmiss, _gunexp = glm_hist_coup.load_state_dict(
        torch.load(GLM_MAP_CKPT, map_location=device), strict=False)
    assert not _gmiss, f'GLM_MAP checkpoint missing required keys: {_gmiss[:5]}'
    glm_hist_coup = glm_hist_coup.to(device).eval()
    assert set(glm_hist_coup.readouts.keys()) >= set(dataloaders['test'].keys()), \
        'GLM checkpoint is missing readouts for some decode sessions.'

    def _ablate_encoder(zero_hist=False, zero_coup='none'):
        enc = _copy.deepcopy(glm_hist_coup)
        if zero_hist:
            enc.hist_basis.zero_()
        if zero_coup == 'all':
            enc.coup_basis.zero_()
        elif zero_coup == 'lagged':
            enc.coup_basis[1:].zero_()
        return enc.to(device).eval()

    glm_encoders_by_mode = {
        'trial_shuffled': _ablate_encoder(zero_coup='all'),
        'time_shuffled':  _ablate_encoder(zero_hist=True, zero_coup='lagged'),
        'both_shuffled':  _ablate_encoder(zero_hist=True, zero_coup='all'),
    }

    # --- glm_coup -------------------------------------------------------------
    GLM_COUP_CKPT = ckpt_map['glm_coup']
    assert os.path.exists(GLM_COUP_CKPT), f'Missing GLM checkpoint {GLM_COUP_CKPT}.'
    glm_coup = GLM_Coupled(
        output_dim_dict=unit_num_dict, **glm_common,
        coup_basis=coup_only_basis, n_lags=N_LAGS)            # NO hist_basis
    _cmiss, _cunexp = glm_coup.load_state_dict(
        torch.load(GLM_COUP_CKPT, map_location=device), strict=False)
    assert not _cmiss, f'GLM_COUP checkpoint missing required keys: {_cmiss[:5]}'
    glm_coup = glm_coup.to(device).eval()

    def _ablate_coup(zero_coup='none'):
        enc = _copy.deepcopy(glm_coup)
        if zero_coup == 'all':
            enc.coup_basis.zero_()
        elif zero_coup == 'lagged':
            enc.coup_basis[1:].zero_()
        return enc.to(device).eval()

    glm_coup_encoders_by_mode = {
        'trial_shuffled': _ablate_coup(zero_coup='all'),
        'time_shuffled':  _ablate_coup(zero_coup='lagged'),
        'both_shuffled':  _ablate_coup(zero_coup='all'),
    }

    # --- glm_joint ------------------------------------------------------------
    flight_cluster_joint, cluster_keys_joint = compute_joint_flight_clusters(
        dataloaders, meta_info_dict)
    GLM_JOINT_CKPT = ckpt_map['glm_joint']
    assert os.path.exists(GLM_JOINT_CKPT), f'Missing GLM checkpoint {GLM_JOINT_CKPT}.'
    glm_joint = GLM_Coupled(
        output_dim_dict=unit_num_dict, **glm_common,
        cluster_keys=cluster_keys_joint)                     # NO temporal filters
    _jmiss, _junexp = glm_joint.load_state_dict(
        torch.load(GLM_JOINT_CKPT, map_location=device), strict=False)
    assert not _jmiss, (f'GLM_JOINT checkpoint missing required keys: {_jmiss[:5]} — '
                        'reproduced clustering disagrees, or a needed session/cluster is absent')
    glm_joint = glm_joint.to(device).eval()

    return dict(
        glm_hist_coup=glm_hist_coup, glm_coup=glm_coup, glm_joint=glm_joint,
        glm_encoders_by_mode=glm_encoders_by_mode,
        glm_coup_encoders_by_mode=glm_coup_encoders_by_mode,
        flight_cluster_joint=flight_cluster_joint,
        cluster_keys_joint=cluster_keys_joint,
    )


# --- Synthetic place-GLM baseline (position GLM-inv) -------------------------
def _place_glm_ckpt(session, mode, cluster, synth_dir, synth_sampling):
    """Disk key; folds in the synth sampling + (for traj) the cluster tag so
    stale fits re-train when the surrogate spikes or clustering change."""
    sub = f'{mode}_{synth_sampling}'
    if mode == 'synthetic_traj':
        sub += f'_{CLUSTER_TAG}'
    safe = str(cluster).replace('/', '_')
    return f'{synth_dir}/{session}__{sub}__c{safe}.pth'


def _collect_by_cluster(loader):
    """{cluster_int -> [(behavior, spike, mask), ...]} of per-batch flight
    subsets (sequence form), for place-GLM minibatch training."""
    buckets = {}
    for b in loader:
        fti = b.flight_type_index.numpy().astype(int)
        for c in np.unique(fti):
            sel = np.where(fti == c)[0]
            buckets.setdefault(int(c), []).append(
                (b.behavior[sel], b.spike[sel], b.mask[sel]))
    return buckets


def _train_place_glm_inplace(glm, session, batches, device):
    """Poisson-NLL fit of a place-only GLM over sequence minibatches (the GLM
    uses only the first 3 z-scored behavior dims; place-only needs no spike)."""
    opt = torch.optim.AdamW(glm.parameters(), lr=GLM_SYNTH_LR,
                            weight_decay=GLM_SYNTH_WD)
    loss_fn = nn.PoissonNLLLoss(log_input=False).to(device)
    best, best_w, no_imp = float('inf'), None, 0
    for _ep in range(GLM_SYNTH_EPOCHS):
        glm.train()
        tot, ntot = 0.0, 0
        for X, Y, M in batches:
            X, Y, M = X.to(device), Y.to(device), M.to(device)
            v = M == 1.0
            if v.sum() == 0:
                continue
            out = glm(session, X, M)
            loss = loss_fn(out[v], Y[v].float())
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * int(v.sum()); ntot += int(v.sum())
        epoch_loss = tot / max(ntot, 1)
        if epoch_loss < best - 1e-5:
            best, best_w, no_imp = epoch_loss, _copy.deepcopy(glm.state_dict()), 0
        else:
            no_imp += 1
            if no_imp >= GLM_SYNTH_PATIENCE:
                break
    if best_w is not None:
        glm.load_state_dict(best_w)
    return glm.eval()


def _place_glm(session, mode, cluster, n_units, batches, *, glm_common, device,
               synth_dir, synth_sampling):
    """Load a cached place-only GLM for (session, mode, cluster) or train+save."""
    path = _place_glm_ckpt(session, mode, cluster, synth_dir, synth_sampling)
    glm = GLM_Coupled(output_dim_dict={session: n_units}, **glm_common).to(device)
    if os.path.exists(path):
        state = torch.load(path, map_location=device)
        if set(state.keys()) == set(glm.state_dict().keys()):
            glm.load_state_dict(state)
            return glm.eval()
    _train_place_glm_inplace(glm, session, batches, device)
    torch.save(glm.state_dict(), path)
    return glm


def _route_predict_glm_map(arr, decoders, warm):
    """Route each flight (by flight_type) to its cluster's GLM_MAP_Decoder;
    fall back to 'other'. Returns (N, 3) with NaN where no decoder applies."""
    N = arr['spikes'].shape[0]
    pred = np.full((N, 3), np.nan)
    fti = arr['flight_type']
    for c in np.unique(fti):
        m = fti == c
        dec = decoders.get(int(c)) or decoders.get('other')
        if dec is None:
            continue
        p, _ = dec.predict(
            arr['spikes'][m], flight_id=arr['flight_id'][m],
            t_idx=arr['t_idx'][m], warm_start=warm[m])
        pred[m] = p
    return pred


def _fit_eval_glm_map_synth(train_loader, test_loader, eval_train_loader,
                            session_name, mode, bm, bs, r_lim, bat, *,
                            glm_common, device, dt, nb_best, nb_cont_best,
                            synth_dir, synth_sampling):
    """Train a single place-only GLM on the synthetic train spikes and invert it
    on the real eval splits -- the synthetic baseline for the POSITION GLM-inv
    decoders (one per-session place field, no cluster routing, for both
    synthetic and synthetic_traj)."""
    per_cluster = False   # position GLM-inv: single place field, no routing
    train = gather_session_arrays(train_loader, bm, bs)
    if train['spikes'].shape[0] < 100:
        return None, None
    n_units = train['spikes'].shape[1]
    train_pos_3d = np.column_stack([train['pos_x'], train['pos_y'], train['pos_z']])

    # Build the train buckets (which clusters get their own place field).
    buckets = _collect_by_cluster(train_loader)
    counts = {c: sum(x[0].shape[0] for x in v) for c, v in buckets.items()}
    train_sets = {}
    if per_cluster:
        # `UNCLUST` in the notebook is a typo alias for UNCLUSTERED; this branch
        # is dead (per_cluster=False) but kept working by referencing UNCLUSTERED.
        major = [c for c in buckets
                 if c != UNCLUSTERED and counts[c] >= MIN_CLUSTER_FLIGHTS]
        for c in major:
            train_sets[c] = buckets[c]
        other = [b for c, v in buckets.items() if c not in major for b in v]
        if other:
            train_sets['other'] = other
    else:
        train_sets['other'] = [b for v in buckets.values() for b in v]

    # Shared NB+continuity warm-start (same as the non-synthetic path).
    nb_c_3d = make_nb_continuity(
        3, r_lim, nb_cont_best[3]['c_speed'], nb_best[3], dt
    ).fit(train['spikes'], train_pos_3d, flight_id=train['flight_id'],
          t_idx=train['t_idx'], dt=dt)

    # Train/load each place GLM and wrap in a GLM_MAP_Decoder (shared sigma_a).
    decoders = {}
    for key, batches in train_sets.items():
        if sum(x[0].shape[0] for x in batches) < MIN_CLUSTER_FLIGHTS:
            continue
        glm_c = _place_glm(session_name, mode, key, n_units, batches,
                           glm_common=glm_common, device=device,
                           synth_dir=synth_dir, synth_sampling=synth_sampling)
        decoders[key] = GLM_MAP_Decoder(
            glm_c, session_name, bm, bs, device=device
        ).fit(train['spikes'], train_pos_3d, flight_id=train['flight_id'],
              t_idx=train['t_idx'], dt=dt)
    if not decoders:
        return None, None

    records = {'test': [], 'train': []}
    for split, loader in [('test', test_loader), ('train', eval_train_loader)]:
        arr = _gather_with_cluster(loader, bm, bs)
        if arr['spikes'].shape[0] == 0:
            continue
        warm, _ = nb_c_3d.predict(
            arr['spikes'], flight_id=arr['flight_id'], t_idx=arr['t_idx'])
        pred3d = _route_predict_glm_map(arr, decoders, warm)
        records[split] = _records_from_routed(
            arr, pred3d[:, :2], pred3d, session_name, bat)
    return records, {'3d': decoders}


def _fit_eval_glm_map(train_loader, test_loader, session_name, meta_info_dict,
                      eval_train_loader=None, *, full_encoder, enc_by_mode,
                      skip_synth=False, loader_to_mode, device, dt,
                      nb_best, nb_cont_best, glm_common, synth_dir,
                      synth_sampling):
    """MAP-decode by inverting the (per-mode ablated) GLM+hist+coup encoder.

    Fits sigma_a + an NB+continuity warm-start on `train_loader`, then decodes
    `test_loader` and `eval_train_loader` (defaults to train).  3D only (the GLM
    place fields are 3D); the 2D prediction is the (x, y) of the 3D estimate.

    The encoder is fixed (real-trained, never retrained); for the shuffle modes
    we invert the matching ablated copy from `enc_by_mode` (coupling and/or
    spike-history bases zeroed), selected via the train-loader identity.
    """
    bm = meta_info_dict['behavior_mean'][session_name]
    bs = meta_info_dict['behavior_std'][session_name]
    r_lim = meta_info_dict['r_lim'][session_name]
    bat = parse_bat(session_name)
    # Synthetic modes: train a place-only GLM on the surrogate spikes (one per
    # session, or one per cluster for synthetic_traj) and invert it on real test.
    _mode = loader_to_mode.get(id(train_loader), 'real')
    if _mode in ('synthetic', 'synthetic_traj'):
        # `skip_synth` decoders (coup-only) share glm_map's single place-GLM
        # synthetic baseline exactly, so we skip them to avoid a duplicate row.
        if skip_synth:
            return None, None
        return _fit_eval_glm_map_synth(
            train_loader, test_loader, eval_train_loader or train_loader,
            session_name, _mode, bm, bs, r_lim, bat,
            glm_common=glm_common, device=device, dt=dt,
            nb_best=nb_best, nb_cont_best=nb_cont_best,
            synth_dir=synth_dir, synth_sampling=synth_sampling)
    train = gather_session_arrays(train_loader, bm, bs)
    test = gather_session_arrays(test_loader, bm, bs)
    if train['spikes'].shape[0] < 100 or test['spikes'].shape[0] < 10:
        return None, None
    eval_train = (gather_session_arrays(eval_train_loader, bm, bs)
                  if eval_train_loader is not None else train)
    train_pos_3d = np.column_stack([train['pos_x'], train['pos_y'], train['pos_z']])
    test_pos_3d = np.column_stack([test['pos_x'], test['pos_y'], test['pos_z']])
    et_pos_3d = np.column_stack(
        [eval_train['pos_x'], eval_train['pos_y'], eval_train['pos_z']])

    # Pick the encoder ablation matching this mode (full encoder for real /
    # warp_only / synthetic*); inferred from the train-loader identity.
    mode = loader_to_mode.get(id(train_loader), 'real')
    encoder = enc_by_mode.get(mode, full_encoder)

    # Warm start: NB+continuity 3D MAP (same hyperparameters as the
    # `nb_continuity` decoder, so the comparison shares an initialization).
    nb_c_3d = make_nb_continuity(
        3, r_lim, nb_cont_best[3]['c_speed'], nb_best[3], dt
    ).fit(train['spikes'], train_pos_3d,
          flight_id=train['flight_id'], t_idx=train['t_idx'],
          dt=dt)

    dec = GLM_MAP_Decoder(
        encoder, session_name, bm, bs, device=device
    ).fit(train['spikes'], train_pos_3d,
          flight_id=train['flight_id'], t_idx=train['t_idx'],
          dt=dt)

    records = {'test': [], 'train': []}
    for split, arr, p3d in [('test', test, test_pos_3d),
                            ('train', eval_train, et_pos_3d)]:
        warm, _ = nb_c_3d.predict(
            arr['spikes'], flight_id=arr['flight_id'], t_idx=arr['t_idx'])
        pos_dec_3d, _ = dec.predict(
            arr['spikes'], flight_id=arr['flight_id'], t_idx=arr['t_idx'],
            warm_start=warm)
        err_3d = np.sqrt(((p3d - pos_dec_3d) ** 2).sum(axis=1))
        err_2d = np.sqrt(((p3d[:, :2] - pos_dec_3d[:, :2]) ** 2).sum(axis=1))
        for i in range(len(err_3d)):
            records[split].append({
                'bat': bat,
                'session': session_name,
                'flight_id': int(arr['flight_id'][i]),
                't_idx': int(arr['t_idx'][i]),
                'x_true': float(p3d[i, 0]),
                'y_true': float(p3d[i, 1]),
                'z_true': float(p3d[i, 2]),
                'x_pred_2d': float(pos_dec_3d[i, 0]),
                'y_pred_2d': float(pos_dec_3d[i, 1]),
                'x_pred_3d': float(pos_dec_3d[i, 0]),
                'y_pred_3d': float(pos_dec_3d[i, 1]),
                'z_pred_3d': float(pos_dec_3d[i, 2]),
                'err_2d': float(err_2d[i]),
                'err_3d': float(err_3d[i]),
            })
    return records, {'3d': dec}


# --- Joint (pos × traj) GLM inversion ----------------------------------------
def _route_predict_glm_map_joint(arr, decoders, warm, fcmap):
    """Route each flight (flight_path_index -> joint cluster id -> readout key)
    to its cluster's GLM_MAP_Decoder; fall back to the 'other' readout.  Returns
    (N, 3) with NaN where no decoder applies."""
    N = arr['spikes'].shape[0]
    pred = np.full((N, 3), np.nan)
    cid = np.array([fcmap.get(int(f), UNCLUSTERED) for f in arr['flight_id']])
    keys = np.array([str(c) if str(c) in decoders else _OTHER_CLUSTER_KEY
                     for c in cid])
    for k in np.unique(keys):
        m = keys == k
        dec = decoders.get(k)
        if dec is None:
            continue
        p, _ = dec.predict(
            arr['spikes'][m], flight_id=arr['flight_id'][m],
            t_idx=arr['t_idx'][m], warm_start=warm[m])
        pred[m] = p
    return pred


def _cluster_place_encoder(session_name, cluster_key, n_units, *, glm_joint,
                           glm_common, device):
    """Wrap one cluster's joint-encoder place-field readout in a non-joint place
    GLM so GLM_MAP_Decoder (which expects a single Linear readout) can invert
    it.  Shares the joint encoder's spatial basis via glm_common."""
    enc = GLM_Coupled(output_dim_dict={session_name: n_units}, **glm_common).to(device)
    enc.readouts[session_name].load_state_dict(
        glm_joint.readouts[session_name][cluster_key].state_dict())
    return enc.eval()


def _collect_by_joint_cluster(loader, fcmap):
    """Like `_collect_by_cluster` but buckets flights by their alpha=0.8 JOINT
    cluster (flight_path_index -> fcmap)."""
    buckets = {}
    for b in loader:
        fpi = b.flight_path_index.numpy().astype(int)
        cids = np.array([fcmap.get(int(f), UNCLUSTERED) for f in fpi])
        for c in np.unique(cids):
            sel = np.where(cids == c)[0]
            buckets.setdefault(int(c), []).append(
                (b.behavior[sel], b.spike[sel], b.mask[sel]))
    return buckets


def _fit_eval_glm_map_joint_synth(train_loader, test_loader, eval_train_loader,
                                  session_name, mode, bm, bs, r_lim, bat, *,
                                  glm_common, device, dt, nb_best, nb_cont_best,
                                  flight_cluster_joint, synth_dir, synth_sampling):
    """Joint synthetic baseline: per-cluster place GLMs keyed by the alpha=0.8
    joint clustering, trained on the surrogate spikes and inverted on the real
    eval splits."""
    train = gather_session_arrays(train_loader, bm, bs)
    if train['spikes'].shape[0] < 100:
        return None, None
    n_units = train['spikes'].shape[1]
    train_pos_3d = np.column_stack([train['pos_x'], train['pos_y'], train['pos_z']])
    fcmap = flight_cluster_joint.get(session_name, {})

    # Per-(joint)cluster train buckets; small clusters pooled into 'other'.
    buckets = _collect_by_joint_cluster(train_loader, fcmap)
    counts = {c: sum(x[0].shape[0] for x in v) for c, v in buckets.items()}
    major = [c for c in buckets
             if c != UNCLUSTERED and counts[c] >= MIN_CLUSTER_FLIGHTS]
    train_sets = {str(c): buckets[c] for c in major}
    other = [b for c, v in buckets.items() if c not in major for b in v]
    if other:
        train_sets[_OTHER_CLUSTER_KEY] = other

    nb_c_3d = make_nb_continuity(
        3, r_lim, nb_cont_best[3]['c_speed'], nb_best[3], dt
    ).fit(train['spikes'], train_pos_3d, flight_id=train['flight_id'],
          t_idx=train['t_idx'], dt=dt)

    # Cache key folds in the joint clustering so it never collides with the
    # position single-place baseline and stales when alpha/ds change.
    cache_mode = f'{mode}_jclus_a{CLUSTER_ALPHA_JOINT:.2f}_ds{CLUSTER_DS_JOINT}'
    decoders = {}
    for key, batches in train_sets.items():
        if sum(x[0].shape[0] for x in batches) < MIN_CLUSTER_FLIGHTS:
            continue
        glm_c = _place_glm(session_name, cache_mode, key, n_units, batches,
                           glm_common=glm_common, device=device,
                           synth_dir=synth_dir, synth_sampling=synth_sampling)
        decoders[key] = GLM_MAP_Decoder(
            glm_c, session_name, bm, bs, device=device
        ).fit(train['spikes'], train_pos_3d, flight_id=train['flight_id'],
              t_idx=train['t_idx'], dt=dt)
    if not decoders:
        return None, None

    records = {'test': [], 'train': []}
    for split, loader in [('test', test_loader), ('train', eval_train_loader)]:
        arr = _gather_with_cluster(loader, bm, bs)
        if arr['spikes'].shape[0] == 0:
            continue
        warm, _ = nb_c_3d.predict(
            arr['spikes'], flight_id=arr['flight_id'], t_idx=arr['t_idx'])
        pred3d = _route_predict_glm_map_joint(arr, decoders, warm, fcmap)
        records[split] = _records_from_routed(
            arr, pred3d[:, :2], pred3d, session_name, bat)
    return records, {'3d': decoders}


def _fit_eval_glm_map_joint(train_loader, test_loader, session_name,
                            meta_info_dict, eval_train_loader=None, *,
                            glm_joint, glm_common, device, dt, nb_best,
                            nb_cont_best, flight_cluster_joint, loader_to_mode,
                            synth_dir, synth_sampling):
    bm = meta_info_dict['behavior_mean'][session_name]
    bs = meta_info_dict['behavior_std'][session_name]
    r_lim = meta_info_dict['r_lim'][session_name]
    bat = parse_bat(session_name)
    _mode = loader_to_mode.get(id(train_loader), 'real')
    if _mode in ('synthetic', 'synthetic_traj'):
        return _fit_eval_glm_map_joint_synth(
            train_loader, test_loader, eval_train_loader or train_loader,
            session_name, _mode, bm, bs, r_lim, bat,
            glm_common=glm_common, device=device, dt=dt,
            nb_best=nb_best, nb_cont_best=nb_cont_best,
            flight_cluster_joint=flight_cluster_joint,
            synth_dir=synth_dir, synth_sampling=synth_sampling)
    train = gather_session_arrays(train_loader, bm, bs)
    if train['spikes'].shape[0] < 100:
        return None, None
    n_units = train['spikes'].shape[1]
    train_pos_3d = np.column_stack([train['pos_x'], train['pos_y'], train['pos_z']])
    fcmap = flight_cluster_joint.get(session_name, {})

    # Shared NB+continuity warm-start (same hyperparameters as glm_map).
    nb_c_3d = make_nb_continuity(
        3, r_lim, nb_cont_best[3]['c_speed'], nb_best[3], dt
    ).fit(train['spikes'], train_pos_3d,
          flight_id=train['flight_id'], t_idx=train['t_idx'],
          dt=dt)

    # One GLM_MAP_Decoder per cluster readout (incl. 'other'); shared sigma_a.
    decoders = {}
    for k in glm_joint.readouts[session_name].keys():
        enc_k = _cluster_place_encoder(session_name, k, n_units,
                                       glm_joint=glm_joint, glm_common=glm_common,
                                       device=device)
        decoders[k] = GLM_MAP_Decoder(
            enc_k, session_name, bm, bs, device=device
        ).fit(train['spikes'], train_pos_3d,
              flight_id=train['flight_id'], t_idx=train['t_idx'],
              dt=dt)

    records = {'test': [], 'train': []}
    for split, loader in [('test', test_loader),
                          ('train', eval_train_loader or train_loader)]:
        arr = _gather_with_cluster(loader, bm, bs)
        if arr['spikes'].shape[0] == 0:
            continue
        warm, _ = nb_c_3d.predict(
            arr['spikes'], flight_id=arr['flight_id'], t_idx=arr['t_idx'])
        pred3d = _route_predict_glm_map_joint(arr, decoders, warm, fcmap)
        records[split] = _records_from_routed(
            arr, pred3d[:, :2], pred3d, session_name, bat)
    return records, {'3d': decoders}


def make_glm_map_fit_eval(*, full_encoder, enc_by_mode, loader_to_mode, device,
                          dt, nb_best, nb_cont_best, glm_common, synth_dir,
                          synth_sampling, skip_synth=False):
    """Factory returning a `_fit_eval_glm_map` closure with all globals bound.

    Set `skip_synth=True` for the coup-only variant (shares glm_map's synthetic
    place baseline).  The returned fn matches the `run_per_mode_per_session`
    fit_eval signature.
    """
    return _partial(
        _fit_eval_glm_map, full_encoder=full_encoder, enc_by_mode=enc_by_mode,
        skip_synth=skip_synth, loader_to_mode=loader_to_mode, device=device,
        dt=dt, nb_best=nb_best, nb_cont_best=nb_cont_best, glm_common=glm_common,
        synth_dir=synth_dir, synth_sampling=synth_sampling)


def make_glm_map_joint_fit_eval(*, glm_joint, glm_common, device, dt, nb_best,
                                nb_cont_best, flight_cluster_joint,
                                loader_to_mode, synth_dir, synth_sampling):
    """Factory returning a `_fit_eval_glm_map_joint` closure with globals bound."""
    return _partial(
        _fit_eval_glm_map_joint, glm_joint=glm_joint, glm_common=glm_common,
        device=device, dt=dt, nb_best=nb_best, nb_cont_best=nb_cont_best,
        flight_cluster_joint=flight_cluster_joint, loader_to_mode=loader_to_mode,
        synth_dir=synth_dir, synth_sampling=synth_sampling)


# ===========================================================================
# ## Transformer
# ===========================================================================
def make_decoder_model(params, unit_num_dict):
    """Instantiate a Bat_Transformer_Decoder from an Optuna best_params dict."""
    fixed_config = dict(
        input_dim_dict=unit_num_dict,
        output_dim=3,
        num_heads=4,
        final_activation=nn.Identity(),
    )
    return Bat_Transformer_Decoder(
        latent_dim=params['latent_dim'],
        ff_dim=params['ff_dim'],
        num_layers=params['num_layers'],
        dropout_prob=params['dropout_prob'],
        readin_dropout=params['readin_dropout'],
        input_transform=params['input_transform'],
        **fixed_config,
    )


def _make_decoder_loss(name):
    """Searchable training loss."""
    return {'mse': nn.MSELoss(),
            'huber': nn.SmoothL1Loss(),
            'l1': nn.L1Loss()}[name]


def _neg_euclid_z(y_true, y_pred):
    """Per-timestep negative Euclidean error in z-scored space (larger=better)."""
    return -np.sqrt(((y_true - y_pred) ** 2).sum(axis=1))


def _params_to_substr(params):
    return '_'.join(
        f'{k}_{v:.4g}' if isinstance(v, float) else f'{k}_{v}'
        for k, v in params.items()
    )


def _try_load_state_dict(model, path, device):
    """Load `path` into `model` only if the state_dict's keys exactly match the
    model's.  Returns True on success, False on schema mismatch."""
    state = torch.load(path, map_location=device)
    expected = set(model.state_dict().keys())
    saved = set(state.keys())
    if expected == saved:
        model.load_state_dict(state)
        return True
    return False


def build_transformer_decoders(best_params, unit_num_dict, modes,
                               mode_dataloaders, device, seed, out_dir='./out',
                               synth_sampling='empirical'):
    """Load (or train-if-missing) one Bat_Transformer_Decoder per mode.

    Ports the checkpoint-path construction from transformer-build/train.py.
    Returns `decoder_models` (dict {mode -> model on device}).
    """
    SHUFFLE_HP_OVERRIDES = {}
    SHUFFLE_WD_OVERRIDES = {}

    decoder_models, decoder_paths, decoder_params = {}, {}, {}
    for mode in modes:
        mode_params = dict(best_params)
        if mode in SHUFFLE_HP_OVERRIDES:
            mode_params.update(SHUFFLE_HP_OVERRIDES[mode])
        if mode in SHUFFLE_WD_OVERRIDES:
            mode_params['weight_decay'] = SHUFFLE_WD_OVERRIDES[mode]
        decoder_params[mode] = mode_params
        mode_substr = _params_to_substr(mode_params)
        suffix = '' if mode == 'real' else f'_{mode}'
        if mode in CLUSTER_DEPENDENT_MODES:
            suffix += f'_{CLUSTER_TAG}'
        if mode in ('synthetic', 'synthetic_traj') and synth_sampling == 'empirical':
            suffix += '_emp'
        decoder_paths[mode] = (
            f'{out_dir}/bat_decoder/bat_decoder_seed_{seed}_{mode_substr}{suffix}{DATA_TAG}.pth'
        )

        set_random_state(seed)
        decoder_models[mode] = make_decoder_model(mode_params, unit_num_dict)

        loaded = (os.path.exists(decoder_paths[mode])
                  and _try_load_state_dict(decoder_models[mode],
                                           decoder_paths[mode], device))
        if loaded:
            decoder_models[mode] = decoder_models[mode].to(device)
        else:
            bat_decoder_trainer(
                model=decoder_models[mode],
                dataloaders=mode_dataloaders[mode],
                device=device,
                pos_dims=3,
                n_epochs=MAX_EPOCHS,
                lr=mode_params['lr'],
                weight_decay=mode_params['weight_decay'],
                loss_fn=_make_decoder_loss(mode_params['loss_name']),
                valid_fn=_neg_euclid_z,
                valid_stat='median',
                early_stopping_patience=50,
                save_model=True,
                model_name=decoder_paths[mode],
            )
    return decoder_models


def _eval_decoder_records(model, dataloaders, meta_info_dict, device):
    """Run the trained decoder on test+train; return {split: list[record]}."""
    records = {'test': [], 'train': []}
    model.eval()
    with torch.no_grad():
        for split in ('test', 'train'):
            for session_name, session_data in dataloaders[split].items():
                bm = meta_info_dict['behavior_mean'][session_name].squeeze()
                bs = meta_info_dict['behavior_std'][session_name].squeeze()
                bat = parse_bat(session_name)
                for batch in session_data:
                    X = batch.spike.to(device)
                    m = batch.mask.to(device)
                    y_true_z = batch.behavior[:, :, :3].numpy()
                    y_pred_z = model(session_name, X, m).cpu().numpy()
                    mask_np = batch.mask.numpy().astype(bool)
                    B, T = mask_np.shape
                    y_true = y_true_z * bs[None, None, :3] + bm[None, None, :3]
                    y_pred = y_pred_z * bs[None, None, :3] + bm[None, None, :3]
                    fpi_per_t = np.broadcast_to(
                        batch.flight_path_index.numpy()[:, None], (B, T))
                    t_idx_per = np.broadcast_to(np.arange(T)[None, :], (B, T))
                    y_true_flat = y_true[mask_np]
                    y_pred_flat = y_pred[mask_np]
                    fpi_flat = fpi_per_t[mask_np].astype(int)
                    t_flat = t_idx_per[mask_np].astype(int)
                    err_2d = np.sqrt(((y_true_flat[:, :2] - y_pred_flat[:, :2]) ** 2).sum(axis=1))
                    err_3d = np.sqrt(((y_true_flat - y_pred_flat) ** 2).sum(axis=1))
                    for i in range(len(err_2d)):
                        records[split].append({
                            'bat': bat,
                            'session': session_name,
                            'flight_id': int(fpi_flat[i]),
                            't_idx': int(t_flat[i]),
                            'x_true': float(y_true_flat[i, 0]),
                            'y_true': float(y_true_flat[i, 1]),
                            'z_true': float(y_true_flat[i, 2]),
                            'x_pred': float(y_pred_flat[i, 0]),
                            'y_pred': float(y_pred_flat[i, 1]),
                            'z_pred': float(y_pred_flat[i, 2]),
                            'err_2d': float(err_2d[i]),
                            'err_3d': float(err_3d[i]),
                        })
    return records


def eval_transformer_by_mode(decoder_models, mode_dataloaders, modes,
                             meta_info_dict, device):
    """Evaluate each mode's decoder on its own mode's data.  Returns
    transformer_dfs (dict {mode -> per-flight-timestep test DataFrame})."""
    transformer_dfs = {}
    for mode in modes:
        recs = _eval_decoder_records(
            decoder_models[mode], mode_dataloaders[mode], meta_info_dict, device)
        transformer_dfs[mode] = pd.DataFrame(recs['test']).assign(decoder='transformer')
    return transformer_dfs


# ===========================================================================
# ## Joint (Hippo-NEDS) decoder — masked encoder/decoder used as a decoder
# ===========================================================================
def _joint_session_room(session_name):
    """3-config room map (matches figures.ipynb `joint_session_room` /
    joint-encode-decode.ipynb `session_room`)."""
    if '_fs' in session_name:
        return 'krishna_fs'
    if session_name.startswith('Dataset_2_'):
        return 'dataset_2'
    return 'lab'


def build_joint_decoders(modes, mode_dataloaders, unit_num_dict, meta_info_dict,
                         device, seed, out_dir='./out', *, joint_params=None,
                         joint_train=None, input_transform=None,
                         synth_sampling='empirical'):
    """Load (or train-if-missing) one Hippo-NEDS joint model per mode.

    Each model is trained on that mode's surrogate TRAIN spikes (real valid/test
    by reference — same contract as the transformer decoder) and used purely as a
    3-D position decoder (all behaviour masked at eval).  Returns
    `joint_models` (dict {mode -> model on device}).
    """
    jp = dict(JOINT_DECODER_PARAMS if joint_params is None else joint_params)
    jt = dict(JOINT_DECODER_TRAIN if joint_train is None else joint_train)
    itf = JOINT_DECODER_INPUT_TRANSFORM if input_transform is None else input_transform
    session_to_room = {s: _joint_session_room(s) for s in unit_num_dict}

    joint_models = {}
    for mode in modes:
        suffix = '' if mode == 'real' else f'_{mode}'
        if mode in CLUSTER_DEPENDENT_MODES:
            suffix += f'_{CLUSTER_TAG}'
        if mode in ('synthetic', 'synthetic_traj') and synth_sampling == 'empirical':
            suffix += '_emp'
        ckpt = (f'{out_dir}/joint/ed2_joint_decoder_seed_{seed}'
                f'_L{jp["num_layers"]}_d{jp["latent_dim"]}_ff{jp["ff_dim"]}{suffix}.pth')
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)

        set_random_state(seed)
        jm = JointMaskedTransformer(
            unit_num_dict, behavior_dim=3, latent_dim=jp['latent_dim'],
            num_heads=jp['num_heads'], ff_dim=jp['ff_dim'], num_layers=jp['num_layers'],
            dropout=jp['dropout'], input_transform=itf,
            session_to_room=session_to_room).to(device)
        if os.path.exists(ckpt) and _try_load_state_dict(jm, ckpt, device):
            joint_models[mode] = jm.to(device).eval()
            continue
        jm, _best, _ = train_joint(
            jm, mode_dataloaders[mode], device, meta_info_dict, seed=seed,
            n_epochs=jt['n_epochs'], lr=jt['lr'], weight_decay=jt['weight_decay'],
            lam=jt['lam'], pos_weight=jt['pos_weight'], scheme_p=scheme_weights(jt),
            select='joint', fde_weight=jt['fde_weight'], err_ref=jt['err_ref'],
            fde_ref=jt['fde_ref'])
        torch.save(jm.state_dict(), ckpt)
        joint_models[mode] = jm.eval()
    return joint_models


@torch.no_grad()
def _eval_joint_records(model, dataloaders, meta_info_dict, device):
    """Decode 3-D position (all behaviour masked) on the real TEST split; return
    per-timestep records in the shared decode schema (mirrors
    `_eval_decoder_records` so `build_cost_tables` treats it identically)."""
    records = []
    model.eval()
    for session_name, session_data in dataloaders['test'].items():
        bm = np.asarray(meta_info_dict['behavior_mean'][session_name]).reshape(-1)[:3]
        bs = np.asarray(meta_info_dict['behavior_std'][session_name]).reshape(-1)[:3]
        bat = parse_bat(session_name)
        for batch in session_data:
            m = batch.mask.to(device)
            beh_target = batch.mask.clone().to(device)     # decode: mask all behaviour
            neural_target = torch.zeros_like(beh_target)
            _, beh_pred, _ = model(session_name, batch.spike.to(device),
                                   batch.behavior.to(device), m, neural_target, beh_target)
            y_pred = beh_pred[:, :, :3].cpu().numpy() * bs + bm
            y_true = batch.behavior[:, :, :3].numpy() * bs + bm
            mask_np = batch.mask.numpy().astype(bool)
            B, T = mask_np.shape
            fpi_per_t = np.broadcast_to(batch.flight_path_index.numpy()[:, None], (B, T))
            t_idx_per = np.broadcast_to(np.arange(T)[None, :], (B, T))
            yt, yp = y_true[mask_np], y_pred[mask_np]
            fpi_flat = fpi_per_t[mask_np].astype(int)
            t_flat = t_idx_per[mask_np].astype(int)
            err_2d = np.sqrt(((yt[:, :2] - yp[:, :2]) ** 2).sum(axis=1))
            err_3d = np.sqrt(((yt - yp) ** 2).sum(axis=1))
            for i in range(len(err_2d)):
                records.append({
                    'bat': bat, 'session': session_name, 'flight_id': int(fpi_flat[i]),
                    't_idx': int(t_flat[i]),
                    'x_true': float(yt[i, 0]), 'y_true': float(yt[i, 1]), 'z_true': float(yt[i, 2]),
                    'x_pred': float(yp[i, 0]), 'y_pred': float(yp[i, 1]), 'z_pred': float(yp[i, 2]),
                    'err_2d': float(err_2d[i]), 'err_3d': float(err_3d[i]),
                })
    return records


def eval_joint_by_mode(joint_models, mode_dataloaders, modes, meta_info_dict, device):
    """Decode the real TEST flights with each mode's joint model.  Returns
    joint_dfs (dict {mode -> per-flight-timestep test DataFrame})."""
    joint_dfs = {}
    for mode in modes:
        recs = _eval_joint_records(
            joint_models[mode], mode_dataloaders[mode], meta_info_dict, device)
        joint_dfs[mode] = pd.DataFrame(recs).assign(decoder='joint')
    return joint_dfs


# ===========================================================================
# ## Aggregation / export
# ===========================================================================
def frechet_per_flight_4g(decoder_dfs):
    """Like compute_frechet_per_flight, but leaves the `decoder` column as
    plain str — so cluster variants survive.

    `decoder_dfs`: list of (decoder_name, df, pred_2d_cols, pred_3d_cols).
    """
    records = []
    for decoder, df, pred_2d_cols, pred_3d_cols in decoder_dfs:
        for (bat, session, flight_id), grp in df.groupby(
                ['bat', 'session', 'flight_id'], sort=False):
            grp = grp.sort_values('t_idx')
            truth = grp[TRUTH_3D].to_numpy(dtype=np.float64)
            pred2d = grp[list(pred_2d_cols)].to_numpy(dtype=np.float64)
            pred3d = grp[list(pred_3d_cols)].to_numpy(dtype=np.float64)
            records.append({
                'bat': bat,
                'session': session,
                'flight_id': int(flight_id),
                'decoder': decoder,
                'n_t': len(grp),
                'frechet_2d': discrete_frechet(truth[:, :2], pred2d),
                'frechet_3d': discrete_frechet(truth, pred3d),
            })
    return pd.DataFrame(records)


def build_cost_tables(per_model_dfs_by_mode, modes, seed, *, model_order,
                      frechet_by_mode_4g=None, out_path=None, model_label=None,
                      data_label=None, data_order=None, figs4g_med=None):
    """Build the ED Fig 2 export dict (frechet_by_mode + errflight_by_mode).

    Ports ed2-export.py.  `per_model_dfs_by_mode`: {mode -> {model -> df}}.
    `frechet_by_mode_4g`: {mode -> per-flight Fréchet df} (== FRECHET_4G); if
    None it is recomputed here from the per-model prediction tables.
    """
    if frechet_by_mode_4g is None:
        frechet_by_mode_4g = {}
        for _m in modes:
            coords = []
            for d, p2, p3 in DECODER_PRED_COLS:
                df = per_model_dfs_by_mode[_m].get(d)
                if df is not None and not df.empty and 'x_true' in df.columns:
                    coords.append((d, df, p2, p3))
            for d, p2, p3 in _CLUSTER_PRED_COLS_4G:
                df = per_model_dfs_by_mode[_m].get(d)
                if df is not None and not df.empty and 'x_true' in df.columns:
                    coords.append((d, df, p2, p3))
            frechet_by_mode_4g[_m] = frechet_per_flight_4g(coords)

    # Per-flight Fréchet for every model across every mode (decoder -> model).
    _fr_by_mode = pd.concat(
        [frechet_by_mode_4g[_m].rename(columns={'decoder': 'model'}).assign(data=_m)
         for _m in modes
         if frechet_by_mode_4g.get(_m) is not None and not frechet_by_mode_4g[_m].empty],
        ignore_index=True)

    # Per-flight median reconstruction error for every model across every mode.
    _err_pieces_ed2 = []
    for _m in modes:
        for _d, _df in per_model_dfs_by_mode[_m].items():
            if _df.empty or 'err_2d' not in _df.columns:
                continue
            g = (_df.groupby(['bat', 'session', 'flight_id'], observed=True)[['err_2d', 'err_3d']]
                 .median().reset_index())
            g['model'] = _d
            g['data'] = _m
            _err_pieces_ed2.append(g)
    _err_by_mode = pd.concat(_err_pieces_ed2, ignore_index=True)

    ed2_export = dict(
        frechet_by_mode=_fr_by_mode[_fr_by_mode['model'].isin(model_order)].reset_index(drop=True),
        errflight_by_mode=_err_by_mode[_err_by_mode['model'].isin(model_order)].reset_index(drop=True),
        figs4g_med=(figs4g_med.copy() if figs4g_med is not None else None),
        model_label=(model_label if model_label is not None else MODEL_LABEL_4G),
        data_label=(data_label if data_label is not None else DATA_LABEL_4G),
        model_order=model_order,
        data_order=(data_order if data_order is not None else DATA_ORDER_4G),
        modes=list(modes), seed=seed)
    if out_path is not None:
        pd.to_pickle(ed2_export, out_path)
    return ed2_export


# ===========================================================================
# ## Orchestrator
# ===========================================================================
def _load_optuna_best_params(out_dir, seed):
    """Load the cached Optuna best_params for the NB / NB+continuity / transformer
    decoders (trace of the notebook's cache paths).

    Returns (nb_best, nb_cont_best, transformer_best_params).
    """
    import pickle

    nb_best = {}
    for ndim in (2, 3):
        # TODO(verify): notebook path ./out/nb_decoder/nb_decoder_{ndim}d_seed_{seed}_optuna.pkl
        nb_study_path = f'{out_dir}/nb_decoder/nb_decoder_{ndim}d_seed_{seed}_optuna.pkl'
        with open(nb_study_path, 'rb') as f:
            nb_best[ndim] = pickle.load(f).best_params

    nb_cont_best = {}
    for ndim in (2, 3):
        # TODO(verify): notebook path ./out/nb_decoder/nb_cont_decoder_{ndim}d_seed_{seed}_optuna.pkl
        nb_cont_study_path = f'{out_dir}/nb_decoder/nb_cont_decoder_{ndim}d_seed_{seed}_optuna.pkl'
        with open(nb_cont_study_path, 'rb') as f:
            nb_cont_best[ndim] = pickle.load(f).best_params

    # TODO(verify): transformer study path uses DATA_TAG='_acc50_preln_17_sess'
    #   ./out/bat_decoder/bat_decoder_seed_{seed}_optuna_metersearch{DATA_TAG}.pkl
    transformer_study_path = (
        f'{out_dir}/bat_decoder/bat_decoder_seed_{seed}_optuna_metersearch{DATA_TAG}.pkl')
    with open(transformer_study_path, 'rb') as f:
        transformer_best_params = pickle.load(f).best_params

    return nb_best, nb_cont_best, transformer_best_params


def generate_ed2_decode_costs(data_config, *, device, seed, out_dir='./out',
                              out_path=None, run_shuffles=True, run_synthetic=True,
                              synth_sampling='empirical', run_joint=True,
                              joint_params=None, joint_train=None,
                              joint_input_transform=None):
    """Top-level ED Fig 2 decode-cost generator (what figures.ipynb calls).

    Self-contained given cached artifacts under `out_dir`.  Runs the decoders
    (glm_map, glm_map_coup, glm_map_joint, transformer, and — when `run_joint` —
    the Hippo-NEDS joint decoder) across the modes and returns the ED Fig 2 export
    dict (also pickled if `out_path` given).  `joint_params` / `joint_train` /
    `joint_input_transform` override the joint decoder's consensus config.
    """
    dt = data_config['bin_width']

    # 1. Data.
    dataloaders, behavior_dim, unit_num_dict, meta_info_dict = bat_data_loader(
        **data_config)

    # 2. Train-only flight clustering (in place).
    refit_train_clusters_inplace(dataloaders, meta_info_dict)

    # 3. Mode dataloaders.
    mode_dataloaders, modes = build_mode_dataloaders(
        dataloaders, meta_info_dict, data_config, seed=seed,
        run_shuffles=run_shuffles, run_synthetic=run_synthetic,
        synth_sampling=synth_sampling)

    # 4. Optuna best params.
    nb_best, nb_cont_best, transformer_best_params = _load_optuna_best_params(
        out_dir, seed)

    # 5. GLM bases + encoders + ablations + joint clustering + loader->mode map.
    glm_hist_basis, glm_coup_basis, glm_common = make_glm_bases(dt)
    glm = make_glm_encoders(
        unit_num_dict, glm_common, glm_hist_basis, glm_coup_basis,
        coup_only_basis=glm_coup_basis, N_LAGS=N_LAGS,
        ckpt_map={
            'glm_map':   f'{out_dir}/bat_glm/bat_glm_hist_coup_position_seed_1_optuna_acc50_default17.pth',
            'glm_coup':  f'{out_dir}/bat_glm/bat_glm_coup_position_seed_1_optuna_acc50_default17.pth',
            'glm_joint': f'{out_dir}/bat_glm/bat_glm_joint_seed_1_optuna_acc50_default17.pth',
        },
        device=device, dataloaders=dataloaders, meta_info_dict=meta_info_dict)

    # `_loader_to_mode` maps each per-mode TRAIN DataLoader id -> its mode, so
    # the fit-eval closures can pick the matching ablated encoder / synth path.
    loader_to_mode = {
        id(mode_dataloaders[m]['train'][sn]): m
        for m in modes for sn in dataloaders['test'].keys()
    }
    synth_dir = f'{out_dir}/bat_glm/synth_place'
    os.makedirs(synth_dir, exist_ok=True)

    # 6. Per-mode transformer decoders (load or train-if-missing).
    decoder_models = build_transformer_decoders(
        transformer_best_params, unit_num_dict, modes, mode_dataloaders,
        device, seed, out_dir=out_dir, synth_sampling=synth_sampling)

    # 7. Run the 4 drivers.
    glm_map_fit_eval = make_glm_map_fit_eval(
        full_encoder=glm['glm_hist_coup'], enc_by_mode=glm['glm_encoders_by_mode'],
        loader_to_mode=loader_to_mode, device=device, dt=dt,
        nb_best=nb_best, nb_cont_best=nb_cont_best, glm_common=glm_common,
        synth_dir=synth_dir, synth_sampling=synth_sampling)
    glm_map_coup_fit_eval = make_glm_map_fit_eval(
        full_encoder=glm['glm_coup'], enc_by_mode=glm['glm_coup_encoders_by_mode'],
        loader_to_mode=loader_to_mode, device=device, dt=dt,
        nb_best=nb_best, nb_cont_best=nb_cont_best, glm_common=glm_common,
        synth_dir=synth_dir, synth_sampling=synth_sampling, skip_synth=True)
    glm_map_joint_fit_eval = make_glm_map_joint_fit_eval(
        glm_joint=glm['glm_joint'], glm_common=glm_common, device=device, dt=dt,
        nb_best=nb_best, nb_cont_best=nb_cont_best,
        flight_cluster_joint=glm['flight_cluster_joint'],
        loader_to_mode=loader_to_mode, synth_dir=synth_dir,
        synth_sampling=synth_sampling)

    glm_map_dfs, _, _, _ = run_per_mode_per_session(
        glm_map_fit_eval, 'glm_map', modes=modes, dataloaders=dataloaders,
        mode_dataloaders=mode_dataloaders, meta_info_dict=meta_info_dict)
    glm_map_coup_dfs, _, _, _ = run_per_mode_per_session(
        glm_map_coup_fit_eval, 'glm_map_coup', modes=modes, dataloaders=dataloaders,
        mode_dataloaders=mode_dataloaders, meta_info_dict=meta_info_dict)
    glm_map_joint_dfs, _, _, _ = run_per_mode_per_session(
        glm_map_joint_fit_eval, 'glm_map_joint', modes=modes, dataloaders=dataloaders,
        mode_dataloaders=mode_dataloaders, meta_info_dict=meta_info_dict)
    transformer_dfs = eval_transformer_by_mode(
        decoder_models, mode_dataloaders, modes, meta_info_dict, device)

    # 7b. Per-mode Hippo-NEDS joint decoder (train-if-missing, decode real test).
    joint_dfs = {}
    if run_joint:
        joint_models = build_joint_decoders(
            modes, mode_dataloaders, unit_num_dict, meta_info_dict, device, seed,
            out_dir=out_dir, joint_params=joint_params, joint_train=joint_train,
            input_transform=joint_input_transform, synth_sampling=synth_sampling)
        joint_dfs = eval_joint_by_mode(
            joint_models, mode_dataloaders, modes, meta_info_dict, device)

    # 8. Assemble PER_MODEL_DFS_4G + FRECHET_4G.
    PER_MODEL_DFS_4G = {}
    for _m in modes:
        PER_MODEL_DFS_4G[_m] = {
            'glm_map':        glm_map_dfs.get(_m, pd.DataFrame()),
            'glm_map_coup':   glm_map_coup_dfs.get(_m, pd.DataFrame()),
            'glm_map_joint':  glm_map_joint_dfs.get(_m, pd.DataFrame()),
            'transformer':    transformer_dfs.get(_m, pd.DataFrame()),
            'joint':          joint_dfs.get(_m, pd.DataFrame()),
        }

    FRECHET_4G = {}
    for _m in modes:
        coords = []
        for d, p2, p3 in DECODER_PRED_COLS:
            df = PER_MODEL_DFS_4G[_m].get(d)
            if df is not None and not df.empty and 'x_true' in df.columns:
                coords.append((d, df, p2, p3))
        for d, p2, p3 in _CLUSTER_PRED_COLS_4G:
            df = PER_MODEL_DFS_4G[_m].get(d)
            if df is not None and not df.empty and 'x_true' in df.columns:
                coords.append((d, df, p2, p3))
        FRECHET_4G[_m] = frechet_per_flight_4g(coords)

    # 9. Build + return the export dict.
    _model_order = ['glm_map', 'glm_map_coup', 'glm_map_joint', 'transformer']
    if run_joint:
        _model_order.append('joint')
    return build_cost_tables(
        PER_MODEL_DFS_4G, modes, seed, model_order=_model_order,
        frechet_by_mode_4g=FRECHET_4G, out_path=out_path)
