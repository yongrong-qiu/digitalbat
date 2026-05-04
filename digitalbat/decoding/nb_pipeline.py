"""Naive Bayes decoder pipelines reused across notebooks.

Three decoders share infrastructure with `notebooks/decode-position-comparison.ipynb`:

  - vanilla NB (Zhang et al. 1998, Bayesian 1-step) via `NaiveBayes_Decoder`
  - NB + continuity prior via `NaiveBayes_Continuity_Decoder`
  - NB with per-neuron lag selected by held-out Poisson information

The high-level entry point `fit_all_nb_decoders` returns a long-format
per-timestep DataFrame combining all three decoders, ready for joining with
other models' outputs (e.g. seq-to-seq transformer, PV-Perceiver) in
comparison plots.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from digitalbat.models.decoder import (
    NaiveBayes_Continuity_Decoder,
    NaiveBayes_Decoder,
)


LAG_BINS = np.arange(-5, 6)
MIN_VAL_SPIKES = 5
NULL_RATE_FLOOR_HZ = 1e-2
ZERO_LAG_IDX = int(np.where(LAG_BINS == 0)[0][0])
NB_CONT_SIGMA_CLIP = (0.02, 5.0)


def gather_session_arrays(loader, behavior_mean, behavior_std):
    """Concatenate spikes / positions / flight_id / t_idx over all batches.

    Returns dict keyed on valid timesteps only, with raw (un-z-scored)
    positions and raw spike counts. `flight_id` is per-timestep
    (broadcast from `batch.flight_path_index`); `t_idx` is the within-flight
    bin index.
    """
    spk_list, beh_list, mask_list, fpi_list = [], [], [], []
    for batch in loader:
        spk_list.append(batch.spike.numpy())
        beh_list.append(batch.behavior.numpy())
        mask_list.append(batch.mask.numpy().astype(bool))
        fpi_list.append(batch.flight_path_index.numpy())
    spikes = np.concatenate(spk_list, axis=0)
    behavior = np.concatenate(beh_list, axis=0)
    mask = np.concatenate(mask_list, axis=0)
    fpi = np.concatenate(fpi_list, axis=0)

    bm = np.asarray(behavior_mean).squeeze()
    bs = np.asarray(behavior_std).squeeze()
    pos = behavior[..., :3] * bs[None, None, :3] + bm[None, None, :3]

    B, T, _ = behavior.shape
    fpi_per_t = np.broadcast_to(fpi[:, None], (B, T))
    t_idx_per = np.broadcast_to(np.arange(T)[None, :], (B, T))

    return {
        'spikes': spikes[mask],
        'pos_x': pos[..., 0][mask],
        'pos_y': pos[..., 1][mask],
        'pos_z': pos[..., 2][mask],
        'flight_id': fpi_per_t[mask].astype(int),
        't_idx': t_idx_per[mask].astype(int),
    }


def make_nb(ndim, r_lim, params, dt):
    """Instantiate a vanilla NB decoder from per-session r_lim + best_params."""
    if ndim == 3:
        n_bins = (params['n_bins_x'], params['n_bins_y'], params['n_bins_z'])
    else:
        n_bins = (params['n_bins_x'], params['n_bins_y'])
    return NaiveBayes_Decoder(
        n_bins=n_bins,
        lims=[tuple(r_lim[d]) for d in range(ndim)],
        sigma=params['sigma'],
        dt=dt,
        min_occupancy=params['min_occupancy'],
        use_uniform_prior=params['use_uniform_prior'],
    )


def make_nb_continuity(ndim, r_lim, c_speed, nb_hp, dt,
                       sigma_clip=NB_CONT_SIGMA_CLIP):
    """NB+continuity decoder reusing vanilla-NB hyperparameters."""
    if ndim == 3:
        n_bins = (nb_hp['n_bins_x'], nb_hp['n_bins_y'], nb_hp['n_bins_z'])
    else:
        n_bins = (nb_hp['n_bins_x'], nb_hp['n_bins_y'])
    return NaiveBayes_Continuity_Decoder(
        c_speed=c_speed,
        sigma_min=sigma_clip[0],
        sigma_max=sigma_clip[1],
        n_bins=n_bins,
        lims=[tuple(r_lim[d]) for d in range(ndim)],
        sigma=nb_hp['sigma'],
        dt=dt,
        min_occupancy=nb_hp['min_occupancy'],
        use_uniform_prior=nb_hp['use_uniform_prior'],
    )


def shift_spikes_per_neuron(spikes, flight_id, lags):
    """For each neuron i with lag τᵢ, return spike count at index t-τᵢ.

    Within-flight only: invalid where (t-τᵢ) is out of bounds or in a
    different flight (flight_id mismatch). Returns (shifted, valid_mask).
    """
    spikes = np.asarray(spikes)
    flight_id = np.asarray(flight_id)
    lags = np.asarray(lags, dtype=int)
    N, K = spikes.shape
    t = np.arange(N)
    s = t[:, None] - lags[None, :]
    in_bounds = (s >= 0) & (s < N)
    s_safe = np.clip(s, 0, N - 1)
    same_flight = flight_id[s_safe] == flight_id[:, None]
    valid = in_bounds & same_flight
    shifted = spikes[s_safe, np.arange(K)[None, :]]
    shifted = np.where(valid, shifted, 0)
    return shifted, valid


def position_to_flat_bin(pos, edges, n_bins):
    """Map (N, ndim) positions to (N,) flat bin indices (row-major)."""
    n_bins = tuple(n_bins)
    idx_per_axis = [
        np.clip(np.digitize(pos[:, d], edges[d]) - 1, 0, n_bins[d] - 1)
        for d in range(len(n_bins))
    ]
    return np.ravel_multi_index(idx_per_axis, n_bins)


def per_neuron_val_poisson_info(nb_at_lag, val_pos_flat, val_spikes_lagged,
                                val_valid_t,
                                min_val_spikes=MIN_VAL_SPIKES,
                                null_rate_floor=NULL_RATE_FLOOR_HZ):
    """Held-out Poisson info (bits/sec) per neuron at a given lag τ."""
    K = nb_at_lag.rate_maps.shape[0]
    if val_valid_t.sum() < 1:
        return np.full(K, np.nan)
    n_v = val_spikes_lagged[val_valid_t].T
    pos_idx = val_pos_flat[val_valid_t]
    log_rates_dt = nb_at_lag.log_rate_maps.reshape(K, -1)[:, pos_idx]
    rates_dt = (nb_at_lag.dt * nb_at_lag.rate_maps).reshape(K, -1)[:, pos_idx]
    T_dt = max(n_v.shape[1] * nb_at_lag.dt, 1e-12)
    spike_sum = n_v.sum(axis=1)
    mean_rate = np.maximum(spike_sum / T_dt, null_rate_floor)
    LL_model = (n_v * log_rates_dt).sum(axis=1) - rates_dt.sum(axis=1)
    LL_null = spike_sum * np.log(mean_rate * nb_at_lag.dt) - mean_rate * T_dt
    info_bits_sec = (LL_model - LL_null) / np.log(2) / T_dt
    info_bits_sec[spike_sum < min_val_spikes] = np.nan
    return info_bits_sec


def fit_lagsweep_and_select(ndim, r_lim, params, train_arr, valid_arr, dt):
    """Sweep τ ∈ LAG_BINS, fit NB at each τ, score on val, pick τᵢ*.

    Returns (chosen_lags, val_info_per_lag, rate_maps_per_lag, shared_nb).
    `shared_nb` is the τ=0 fit; its edges/centers/log_prior/support_mask
    are reused by `assemble_lag_nb`.
    """
    cols = ['pos_x', 'pos_y'] + (['pos_z'] if ndim == 3 else [])
    train_pos = np.column_stack([train_arr[c] for c in cols])
    valid_pos = np.column_stack([valid_arr[c] for c in cols])
    K = train_arr['spikes'].shape[1]

    shared_nb = make_nb(ndim, r_lim, params, dt).fit(train_arr['spikes'], train_pos)
    val_pos_flat = position_to_flat_bin(valid_pos, shared_nb.edges, shared_nb.n_bins)

    n_lags = len(LAG_BINS)
    n_bins = shared_nb.n_bins
    rate_floor = shared_nb.rate_floor
    rate_maps_per_lag = np.full((n_lags, K) + n_bins, rate_floor)
    val_info_per_lag = np.full((n_lags, K), np.nan)

    for j, tau in enumerate(LAG_BINS):
        train_shifted, train_valid = shift_spikes_per_neuron(
            train_arr['spikes'], train_arr['flight_id'],
            np.full(K, tau, dtype=int),
        )
        train_valid_t = train_valid[:, 0]
        if train_valid_t.sum() < 100:
            continue
        nb_tau = make_nb(ndim, r_lim, params, dt).fit(
            train_shifted[train_valid_t], train_pos[train_valid_t],
        )
        rate_maps_per_lag[j] = nb_tau.rate_maps
        val_shifted, val_valid = shift_spikes_per_neuron(
            valid_arr['spikes'], valid_arr['flight_id'],
            np.full(K, tau, dtype=int),
        )
        val_valid_t = val_valid[:, 0]
        if val_valid_t.sum() < 5:
            continue
        val_info_per_lag[j] = per_neuron_val_poisson_info(
            nb_tau, val_pos_flat, val_shifted, val_valid_t,
        )

    filled = np.where(np.isnan(val_info_per_lag), -np.inf, val_info_per_lag)
    all_nan = np.all(np.isnan(val_info_per_lag), axis=0)
    filled[ZERO_LAG_IDX, all_nan] = 0.0
    chosen_lags = LAG_BINS[filled.argmax(axis=0)]
    return chosen_lags, val_info_per_lag, rate_maps_per_lag, shared_nb


def assemble_lag_nb(shared_nb, rate_maps_per_lag, chosen_lags):
    """Build a fitted NB whose neuron-i rate map comes from τᵢ*."""
    K = shared_nb.rate_maps.shape[0]
    new_maps = np.empty_like(shared_nb.rate_maps)
    lag_to_idx = {int(t): j for j, t in enumerate(LAG_BINS)}
    for i in range(K):
        j = lag_to_idx[int(chosen_lags[i])]
        new_maps[i] = rate_maps_per_lag[j, i]
    shared_nb.rate_maps = new_maps
    shared_nb.log_rate_maps = np.log(shared_nb.dt * new_maps)
    shared_nb.tau_rate_sum = (shared_nb.dt * new_maps).sum(axis=0)
    return shared_nb


def predict_lagged(nb, spikes, flight_id, chosen_lags):
    """Per-(t, i) masked Poisson decoding; returns (T, ndim)."""
    shifted, valid_mask = shift_spikes_per_neuron(spikes, flight_id, chosen_lags)
    K = nb.rate_maps.shape[0]
    flat_log = nb.log_rate_maps.reshape(K, -1)
    flat_rates_dt = (nb.dt * nb.rate_maps).reshape(K, -1)
    log_term = shifted @ flat_log
    penalty = valid_mask.astype(float) @ flat_rates_dt
    log_post = log_term - penalty + nb.log_prior.reshape(1, -1)
    oob = ~nb.support_mask.reshape(-1)
    if oob.any():
        log_post[:, oob] = -np.inf
    flat_idx = np.argmax(log_post, axis=1)
    idx_per_axis = np.unravel_index(flat_idx, nb.n_bins)
    return np.column_stack(
        [nb.centers[d][idx_per_axis[d]] for d in range(nb.ndim)]
    )


def _per_timestep_records(session_name, arr, p2d, p3d, pos_dec_2d, pos_dec_3d,
                          decoder_name, dt):
    err_2d = np.sqrt(((p2d - pos_dec_2d) ** 2).sum(axis=1))
    err_3d = np.sqrt(((p3d - pos_dec_3d) ** 2).sum(axis=1))
    out = []
    for i in range(len(err_2d)):
        out.append({
            'decoder': decoder_name,
            'session': session_name,
            'flight_id': int(arr['flight_id'][i]),
            't_idx': int(arr['t_idx'][i]),
            't_abs': float(arr['t_idx'][i]) * dt,
            'x_true': float(p3d[i, 0]),
            'y_true': float(p3d[i, 1]),
            'z_true': float(p3d[i, 2]),
            'x_pred_2d': float(pos_dec_2d[i, 0]),
            'y_pred_2d': float(pos_dec_2d[i, 1]),
            'x_pred': float(pos_dec_3d[i, 0]),
            'y_pred': float(pos_dec_3d[i, 1]),
            'z_pred': float(pos_dec_3d[i, 2]),
            'err_2d': float(err_2d[i]),
            'err_3d': float(err_3d[i]),
        })
    return out


def session_arrays_from_pp_flights(flights, session_name):
    """Build NB-ready session arrays from a list of pseudo-population flights.

    Each flight dict must have keys 'pv' (T, n_pseudo), 'xyz' (T, 3),
    'session', and 'flight_id' — the format produced by `extract_flights`
    in `notebooks/pv_decode.ipynb`. Returns the same dict layout as
    `gather_session_arrays`, with `pv` standing in for spikes.

    `t_idx` is set to `arange(T_flight)` per flight, which is correct as long
    as valid bins inside each flight's batch row are contiguous from index 0
    (true under `split_each_trajectory_flag=True` with end-padding).
    """
    sess = [f for f in flights if f['session'] == session_name]
    if not sess:
        return None
    spk = np.concatenate([f['pv'] for f in sess], axis=0).astype(np.float64)
    xyz = np.concatenate([f['xyz'] for f in sess], axis=0)
    fid = np.concatenate(
        [np.full(f['pv'].shape[0], f['flight_id'], dtype=int) for f in sess]
    )
    tid = np.concatenate(
        [np.arange(f['pv'].shape[0], dtype=int) for f in sess]
    )
    return {
        'spikes': spk,
        'pos_x': xyz[:, 0],
        'pos_y': xyz[:, 1],
        'pos_z': xyz[:, 2],
        'flight_id': fid,
        't_idx': tid,
    }


def _fit_predict_one_session(session_name, train, valid, eval_, r_lim, *,
                             nb_best, nb_cont_best, dt):
    """Fit NB / NB-cont / NB-lag on (train, valid) and predict on eval_.

    Each of `train`, `valid`, `eval_` is a dict matching `gather_session_arrays`
    output. Returns a list of per-timestep record dicts (extends one decoder
    after another).
    """
    train_p2d = np.column_stack([train['pos_x'], train['pos_y']])
    train_p3d = np.column_stack([train['pos_x'], train['pos_y'], train['pos_z']])
    eval_p2d = np.column_stack([eval_['pos_x'], eval_['pos_y']])
    eval_p3d = np.column_stack([eval_['pos_x'], eval_['pos_y'], eval_['pos_z']])
    out = []

    # Vanilla NB
    nb_2d = make_nb(2, r_lim, nb_best[2], dt).fit(train['spikes'], train_p2d)
    nb_3d = make_nb(3, r_lim, nb_best[3], dt).fit(train['spikes'], train_p3d)
    pos_dec_2d, _ = nb_2d.predict(eval_['spikes'])
    pos_dec_3d, _ = nb_3d.predict(eval_['spikes'])
    out.extend(_per_timestep_records(
        session_name, eval_, eval_p2d, eval_p3d,
        pos_dec_2d, pos_dec_3d, 'naive_bayes', dt,
    ))

    # NB + continuity
    nb_c_2d = make_nb_continuity(
        2, r_lim, nb_cont_best[2]['c_speed'], nb_best[2], dt,
    ).fit(
        train['spikes'], train_p2d,
        flight_id=train['flight_id'], t_idx=train['t_idx'], dt=dt,
    )
    nb_c_3d = make_nb_continuity(
        3, r_lim, nb_cont_best[3]['c_speed'], nb_best[3], dt,
    ).fit(
        train['spikes'], train_p3d,
        flight_id=train['flight_id'], t_idx=train['t_idx'], dt=dt,
    )
    pos_dec_2d, _ = nb_c_2d.predict(
        eval_['spikes'], flight_id=eval_['flight_id'], t_idx=eval_['t_idx'])
    pos_dec_3d, _ = nb_c_3d.predict(
        eval_['spikes'], flight_id=eval_['flight_id'], t_idx=eval_['t_idx'])
    out.extend(_per_timestep_records(
        session_name, eval_, eval_p2d, eval_p3d,
        pos_dec_2d, pos_dec_3d, 'nb_continuity', dt,
    ))

    # NB with per-neuron lag
    lags_2d, _, rmaps_2d, shared_2d = fit_lagsweep_and_select(
        2, r_lim, nb_best[2], train, valid, dt,
    )
    lags_3d, _, rmaps_3d, shared_3d = fit_lagsweep_and_select(
        3, r_lim, nb_best[3], train, valid, dt,
    )
    nb_lag_2d = assemble_lag_nb(shared_2d, rmaps_2d, lags_2d)
    nb_lag_3d = assemble_lag_nb(shared_3d, rmaps_3d, lags_3d)
    pos_dec_2d = predict_lagged(nb_lag_2d, eval_['spikes'], eval_['flight_id'], lags_2d)
    pos_dec_3d = predict_lagged(nb_lag_3d, eval_['spikes'], eval_['flight_id'], lags_3d)
    out.extend(_per_timestep_records(
        session_name, eval_, eval_p2d, eval_p3d,
        pos_dec_2d, pos_dec_3d, 'naive_bayes_lag', dt,
    ))

    return out


def fit_all_nb_decoders(dataloaders, meta_info_dict, *,
                        nb_best, nb_cont_best, dt, split='test',
                        min_train=100, min_eval=10, verbose=False):
    """Fit NB / NB-cont / NB-lag per session from raw spikes via dataloaders.

    Parameters
    ----------
    dataloaders : dict {'train': {session: loader}, 'valid': ..., 'test': ...}
    meta_info_dict : output of `bat_data_loader`; must include
        'behavior_mean', 'behavior_std', 'r_lim'.
    nb_best : dict {2: <best_params>, 3: <best_params>} from optuna.
    nb_cont_best : dict {2: {'c_speed': float, ...}, 3: ...} from optuna.
    dt : bin width in seconds.
    split : 'test' (default), 'valid', or 'train' — which split to predict on.

    Returns
    -------
    pred_df : long-format DataFrame; see `fit_all_nb_decoders_from_pp` for columns.
    """
    records = []
    for session_name in dataloaders[split].keys():
        bm = meta_info_dict['behavior_mean'][session_name]
        bs = meta_info_dict['behavior_std'][session_name]
        r_lim = meta_info_dict['r_lim'][session_name]

        train = gather_session_arrays(dataloaders['train'][session_name], bm, bs)
        valid = gather_session_arrays(dataloaders['valid'][session_name], bm, bs)
        eval_ = gather_session_arrays(dataloaders[split][session_name], bm, bs)

        if (train['spikes'].shape[0] < min_train
                or eval_['spikes'].shape[0] < min_eval
                or valid['spikes'].shape[0] < min_eval):
            if verbose:
                print(f'  {session_name}: too few samples, skipping')
            continue

        records.extend(_fit_predict_one_session(
            session_name, train, valid, eval_, r_lim,
            nb_best=nb_best, nb_cont_best=nb_cont_best, dt=dt,
        ))
        if verbose:
            print(f'  {session_name}: done '
                  f'(K={train["spikes"].shape[1]}, T_test={eval_["spikes"].shape[0]})')

    return pd.DataFrame(records)


def _per_session_pp_arrays(flights, sessions):
    """Map session → session_arrays dict (or skip session if missing)."""
    out = {}
    for s in sessions:
        arr = session_arrays_from_pp_flights(flights, s)
        if arr is not None:
            out[s] = arr
    return out


def nb_pp_objective(trial, ndim, train_arrays, valid_arrays, r_lim_dict, dt):
    """Optuna objective for NB on pseudo-population (search space mirrors
    `notebooks/decode-position-comparison.ipynb`).

    Returns negative mean-of-per-session-median 3D error (higher = better).
    """
    sigma = trial.suggest_float('sigma', 0.5, 4.0)
    n_bins_x = trial.suggest_categorical('n_bins_x', [15, 20, 25, 30, 40, 50])
    n_bins_y = trial.suggest_categorical('n_bins_y', [15, 20, 25, 30, 40, 50])
    min_occ = trial.suggest_int('min_occupancy', 1, 10)
    use_uniform = trial.suggest_categorical('use_uniform_prior', [True, False])
    if ndim == 3:
        n_bins_z = trial.suggest_categorical('n_bins_z', [8, 10, 15, 20])
        n_bins = (n_bins_x, n_bins_y, n_bins_z)
    else:
        n_bins = (n_bins_x, n_bins_y)

    cols = ['pos_x', 'pos_y'] + (['pos_z'] if ndim == 3 else [])
    per_med = []
    for s, train in train_arrays.items():
        valid = valid_arrays.get(s)
        if valid is None:
            continue
        train_pos = np.column_stack([train[c] for c in cols])
        valid_pos = np.column_stack([valid[c] for c in cols])
        try:
            r_lim = r_lim_dict[s]
            nb = NaiveBayes_Decoder(
                n_bins=n_bins,
                lims=[tuple(r_lim[d]) for d in range(ndim)],
                sigma=sigma, dt=dt,
                min_occupancy=min_occ,
                use_uniform_prior=use_uniform,
            ).fit(train['spikes'], train_pos)
            pos_dec, _ = nb.predict(valid['spikes'])
            err = np.sqrt(((valid_pos - pos_dec) ** 2).sum(axis=1))
            per_med.append(np.median(err))
        except Exception:
            return -float('inf')
    if not per_med:
        return -float('inf')
    return -float(np.mean(per_med))


def nb_cont_pp_objective(trial, ndim, train_arrays, valid_arrays,
                         r_lim_dict, dt, nb_hp):
    """Optuna objective for NB+continuity prior on pseudo-population."""
    c_speed = trial.suggest_float('c_speed', 0.05, 5.0, log=True)
    cols = ['pos_x', 'pos_y'] + (['pos_z'] if ndim == 3 else [])
    per_med = []
    for s, train in train_arrays.items():
        valid = valid_arrays.get(s)
        if valid is None:
            continue
        train_pos = np.column_stack([train[c] for c in cols])
        valid_pos = np.column_stack([valid[c] for c in cols])
        try:
            r_lim = r_lim_dict[s]
            nb = make_nb_continuity(ndim, r_lim, c_speed, nb_hp, dt).fit(
                train['spikes'], train_pos,
                flight_id=train['flight_id'], t_idx=train['t_idx'], dt=dt,
            )
            pos_dec, _ = nb.predict(
                valid['spikes'],
                flight_id=valid['flight_id'], t_idx=valid['t_idx'],
            )
            err = np.sqrt(((valid_pos - pos_dec) ** 2).sum(axis=1))
            per_med.append(np.median(err))
        except Exception:
            return -float('inf')
    if not per_med:
        return -float('inf')
    return -float(np.mean(per_med))


def run_nb_pp_optuna(flights_train, flights_valid, meta_info_dict, dt, *,
                    n_trials=20, seed=1, output_dir='./out', verbose=False):
    """Run NB + NB-continuity Optuna searches (2D and 3D) on the pseudo-population.

    Studies cache to:
      - `{output_dir}/nb_decoder_pp_{ndim}d_seed_{seed}_optuna.pkl`
      - `{output_dir}/nb_cont_decoder_pp_{ndim}d_seed_{seed}_optuna.pkl`

    Search spaces match `notebooks/decode-position-comparison.ipynb`.

    Returns
    -------
    nb_best : dict {2: best_params, 3: best_params}
    nb_cont_best : dict {2: {'c_speed': float}, 3: ...}
    """
    import os
    import pickle
    import optuna

    sessions = sorted({f['session'] for f in flights_train}
                      & {f['session'] for f in flights_valid})
    train_arrays = _per_session_pp_arrays(flights_train, sessions)
    valid_arrays = _per_session_pp_arrays(flights_valid, sessions)
    common = sorted(set(train_arrays) & set(valid_arrays))
    train_arrays = {s: train_arrays[s] for s in common}
    valid_arrays = {s: valid_arrays[s] for s in common}
    r_lim_dict = meta_info_dict['r_lim']
    print(f'NB pseudo-pop search: {len(common)} sessions')

    nb_best, nb_cont_best = {}, {}
    for ndim in (2, 3):
        nb_path = os.path.join(output_dir, f'nb_decoder_pp_{ndim}d_seed_{seed}_optuna.pkl')
        if os.path.exists(nb_path):
            with open(nb_path, 'rb') as f:
                study = pickle.load(f)
            print(f'NB-pp-{ndim}D: loaded study with {len(study.trials)} trials')
        else:
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=seed),
            )
        remaining = max(0, n_trials - len(study.trials))
        if remaining > 0:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study.optimize(
                lambda t, n=ndim: nb_pp_objective(
                    t, n, train_arrays, valid_arrays, r_lim_dict, dt,
                ),
                n_trials=remaining, show_progress_bar=verbose,
            )
            with open(nb_path, 'wb') as f:
                pickle.dump(study, f)
        nb_best[ndim] = study.best_params
        print(f'NB-pp-{ndim}D best valid err: {-study.best_value:.4f} m')
        print(f'  best_params: {study.best_params}')

        cont_path = os.path.join(output_dir, f'nb_cont_decoder_pp_{ndim}d_seed_{seed}_optuna.pkl')
        if os.path.exists(cont_path):
            with open(cont_path, 'rb') as f:
                cont_study = pickle.load(f)
            print(f'NB-cont-pp-{ndim}D: loaded study with {len(cont_study.trials)} trials')
        else:
            cont_study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=seed),
            )
        remaining = max(0, n_trials - len(cont_study.trials))
        if remaining > 0:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            cont_study.optimize(
                lambda t, n=ndim, hp=nb_best[ndim]: nb_cont_pp_objective(
                    t, n, train_arrays, valid_arrays, r_lim_dict, dt, hp,
                ),
                n_trials=remaining, show_progress_bar=verbose,
            )
            with open(cont_path, 'wb') as f:
                pickle.dump(cont_study, f)
        nb_cont_best[ndim] = cont_study.best_params
        print(f'NB-cont-pp-{ndim}D best c_speed: {cont_study.best_params["c_speed"]:.4g}'
              f', valid err: {-cont_study.best_value:.4f} m')

    return nb_best, nb_cont_best


def fit_all_nb_decoders_from_pp(flights_train, flights_valid, flights_eval,
                                meta_info_dict, *,
                                nb_best, nb_cont_best, dt,
                                min_train=100, min_eval=10, verbose=False):
    """Fit NB / NB-cont / NB-lag per session using bat_former pseudo-population.

    Drop-in replacement for `fit_all_nb_decoders` when the input features
    are bat_former encoder outputs (3369-dim pseudo-population) rather than
    raw spike counts. NB is still per-session because position bins live in
    the session's room — only the input features are pooled-multi-session.

    Parameters mirror `fit_all_nb_decoders`, with `flights_*` lists in the
    format produced by `extract_flights` in `notebooks/pv_decode.ipynb`
    (each entry has 'pv', 'xyz', 'session', 'flight_id', 'flight_type').

    Returns
    -------
    pred_df : long-format DataFrame with columns
        decoder, session, flight_id, t_idx, t_abs,
        x_true, y_true, z_true,
        x_pred_2d, y_pred_2d, x_pred, y_pred, z_pred,
        err_2d, err_3d.
        decoder ∈ {'naive_bayes', 'nb_continuity', 'naive_bayes_lag'}.
    """
    sessions = sorted({f['session'] for f in flights_eval})
    records = []
    for session_name in sessions:
        train = session_arrays_from_pp_flights(flights_train, session_name)
        valid = session_arrays_from_pp_flights(flights_valid, session_name)
        eval_ = session_arrays_from_pp_flights(flights_eval, session_name)
        if train is None or valid is None or eval_ is None:
            continue
        if (train['spikes'].shape[0] < min_train
                or eval_['spikes'].shape[0] < min_eval
                or valid['spikes'].shape[0] < min_eval):
            if verbose:
                print(f'  {session_name}: too few samples, skipping')
            continue
        r_lim = meta_info_dict['r_lim'][session_name]
        records.extend(_fit_predict_one_session(
            session_name, train, valid, eval_, r_lim,
            nb_best=nb_best, nb_cont_best=nb_cont_best, dt=dt,
        ))
        if verbose:
            print(f'  {session_name}: done '
                  f'(K_pp={train["spikes"].shape[1]}, T_eval={eval_["spikes"].shape[0]})')

    return pd.DataFrame(records)
