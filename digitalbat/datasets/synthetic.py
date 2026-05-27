"""Build a parallel dataloader set with Poisson-resampled training spikes.

Fits a smoothed 3D rate map per session on the real training spikes and
positions, then replaces only `dataloaders['train']` spikes with Poisson
samples from that rate map. `valid` and `test` are passed through by
reference so they are guaranteed identical to the real ones.

Used to test how much downstream models rely on long-range temporal
dependencies in the spike trains beyond per-neuron place tuning.
"""
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PlaceCellSimulator3D


def make_synthetic_train_dataloaders(
    dataloaders,
    meta_info_dict,
    dt,
    bin_size=0.10,
    smoothing_sigma=1.0,
    poisson=True,
    seed=0,
):
    """Return a parallel dataloader dict with synthetic Poisson training spikes.

    Parameters
    ----------
    dataloaders : dict
        Output of ``bat_data_loader``: ``{'train', 'valid', 'test'} -> {session: DataLoader}``.
    meta_info_dict : dict
        Output of ``bat_data_loader``. Used to un-z-score ``batch.behavior`` back
        to physical (X, Y, Z) for rate-map fitting/sampling.
    dt : float
        Bin width in seconds (matches ``data_config['bin_width']``).
    bin_size : float
        Spatial bin size in meters.
    smoothing_sigma : float | None
        Gaussian smoothing sigma in bins; None disables smoothing.
    poisson : bool
        If True, sample ``np.random.poisson(rate*dt)``. If False, return the
        deterministic mean ``rate*dt`` (useful as a no-Poisson-noise baseline).
    seed : int
        Seed for reproducibility.

    Returns
    -------
    syn_dataloaders : dict
        Same shape as ``dataloaders``. ``valid``/``test`` are the same objects.
    simulators : dict[str, PlaceCellSimulator3D]
        Per-session fitted simulators (``.rate_map``, ``.edges``, ``.shape``).
    """
    rng = np.random.default_rng(seed)
    syn = {'train': {}, 'valid': dataloaders['valid'], 'test': dataloaders['test']}
    simulators = {}

    for session, train_loader in dataloaders['train'].items():
        ds = train_loader.dataset

        beh_z = ds.behavior.numpy()    # (N, T, behavior_dim), z-scored
        spk = ds.spike.numpy()         # (N, T, K), raw counts
        mask = ds.mask.numpy()         # (N, T)

        bm = np.asarray(meta_info_dict['behavior_mean'][session]).reshape(-1)
        bs = np.asarray(meta_info_dict['behavior_std'][session]).reshape(-1)
        pos_xyz = (beh_z * bs + bm)[..., :3]

        beh_list, spk_list = [], []
        for i in range(pos_xyz.shape[0]):
            valid = mask[i] == 1.0
            beh_list.append(pos_xyz[i, valid].T)   # (3, T_valid)
            spk_list.append(spk[i, valid].T)       # (K, T_valid)

        sim = PlaceCellSimulator3D(bin_size=bin_size, smoothing_sigma=smoothing_sigma)
        sim.fit(beh_list, spk_list, dt=dt)
        simulators[session] = sim

        np.random.seed(int(rng.integers(0, 2**31 - 1)))
        syn_list = sim.simulate(beh_list, dt=dt, poisson=poisson)

        syn_spk = spk.copy()           # preserve zero padding outside mask
        for i, syn_i in enumerate(syn_list):
            valid = mask[i] == 1.0
            syn_spk[i, valid] = syn_i.T.astype(syn_spk.dtype)

        syn_ds = copy.copy(ds)
        syn_ds.spike = torch.from_numpy(syn_spk).float()

        syn['train'][session] = DataLoader(
            syn_ds,
            sampler=train_loader.sampler,
            batch_size=train_loader.batch_size,
        )

    return syn, simulators


def make_synthetic_train_dataloaders_traj(
    dataloaders,
    meta_info_dict,
    dt,
    bin_size=0.10,
    smoothing_sigma=1.0,
    poisson=True,
    min_cluster_size=3,
    seed=0,
):
    """Trajectory-specific variant of ``make_synthetic_train_dataloaders``.

    Instead of fitting one 3D rate map per session on all training
    flights, fit a separate ``PlaceCellSimulator3D`` for each flight
    cluster with more than ``min_cluster_size`` training trajectories
    (analogous to direction-specific place fields). All remaining
    clusters are pooled into a single ``'other'`` simulator. At
    simulation time each training trajectory is routed to its
    cluster-specific simulator (or to ``'other'``).
    Valid/test loaders are passed through by reference.
    """
    rng = np.random.default_rng(seed)
    syn = {'train': {}, 'valid': dataloaders['valid'], 'test': dataloaders['test']}
    simulators = {}

    for session, train_loader in dataloaders['train'].items():
        ds = train_loader.dataset

        beh_z = ds.behavior.numpy()
        spk = ds.spike.numpy()
        mask = ds.mask.numpy()
        fti = ds.flight_type_index.numpy().astype(int)

        bm = np.asarray(meta_info_dict['behavior_mean'][session]).reshape(-1)
        bs = np.asarray(meta_info_dict['behavior_std'][session]).reshape(-1)
        pos_xyz = (beh_z * bs + bm)[..., :3]

        beh_list, spk_list = [], []
        for i in range(pos_xyz.shape[0]):
            valid = mask[i] == 1.0
            beh_list.append(pos_xyz[i, valid].T)
            spk_list.append(spk[i, valid].T)

        unique_clusters, counts = np.unique(fti, return_counts=True)
        major_clusters = {
            int(c) for c, ct in zip(unique_clusters, counts) if ct > min_cluster_size
        }

        cluster_to_sim = {}
        for c in major_clusters:
            inds = np.where(fti == c)[0]
            sim = PlaceCellSimulator3D(bin_size=bin_size, smoothing_sigma=smoothing_sigma)
            sim.fit(
                [beh_list[i] for i in inds],
                [spk_list[i] for i in inds],
                dt=dt,
            )
            cluster_to_sim[c] = sim

        other_inds = [i for i in range(len(fti)) if int(fti[i]) not in major_clusters]
        if other_inds:
            sim = PlaceCellSimulator3D(bin_size=bin_size, smoothing_sigma=smoothing_sigma)
            sim.fit(
                [beh_list[i] for i in other_inds],
                [spk_list[i] for i in other_inds],
                dt=dt,
            )
            cluster_to_sim['other'] = sim

        simulators[session] = cluster_to_sim

        np.random.seed(int(rng.integers(0, 2**31 - 1)))

        syn_spk = spk.copy()  # preserve zero padding outside mask
        for i in range(len(fti)):
            c = int(fti[i])
            sim = cluster_to_sim.get(c) if c in major_clusters else cluster_to_sim.get('other')
            if sim is None:
                continue
            syn_i = sim.simulate([beh_list[i]], dt=dt, poisson=poisson)[0]
            valid = mask[i] == 1.0
            syn_spk[i, valid] = syn_i.T.astype(syn_spk.dtype)

        syn_ds = copy.copy(ds)
        syn_ds.spike = torch.from_numpy(syn_spk).float()

        syn['train'][session] = DataLoader(
            syn_ds,
            sampler=train_loader.sampler,
            batch_size=train_loader.batch_size,
        )

    return syn, simulators
