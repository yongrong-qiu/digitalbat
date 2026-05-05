"""PV-Perceiver training/eval pipeline shared across notebooks.

The Perceiver IO decoder in `digitalbat.models.perceiver` is wrapped here with
the plumbing that originally lived inline in `notebooks/pv_decode.ipynb`:

  - flight extraction (per-session real spikes OR pseudo-population from a
    trained encoder),
  - the per-flight `PVTrajectoryDataset` that samples a t_sample bin and
    builds relative- or absolute-time queries around it,
  - `suggest_hyperparams` for an Optuna search,
  - `train_perceiver` / `evaluate_perceiver`,
  - a one-line `trial_callback` for Optuna progress.

Shared by `notebooks/pv_decode.ipynb` (pseudo-pop input, single global model)
and `notebooks/decode-position-comparison.ipynb` (real spike input, per-session
models — same hyperparameters, one PerceiverIOTrajectoryDecoder per session).
"""
from __future__ import annotations

import copy
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import r2_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
import tqdm

from digitalbat.models.perceiver import (
    PerceiverIOConfig, PerceiverIOTrajectoryDecoder,
)
from digitalbat.utility.utils import set_random_state


# ---------------------------------------------------------------------------
# Flight extraction
# ---------------------------------------------------------------------------

def _unzscore_xyz(behavior_z, behavior_mean, behavior_std):
    """`behavior_z` (B, T, behavior_dim) → real-units (x, y, z) (B, T, 3)."""
    bm = np.asarray(behavior_mean).squeeze()
    bs = np.asarray(behavior_std).squeeze()
    return behavior_z[..., :3] * bs[None, None, :3] + bm[None, None, :3]


def extract_real_flights(dataloaders, meta_info_dict, tier):
    """Per-flight real spike counts + 3D position, kept per-session.

    Yields one dict per flight in `dataloaders[tier]`, with `pv` set to that
    flight's raw spike-count matrix `(T_flight, K_session)`. Each session has
    its own neuron count `K_session`, so the resulting flights *cannot* be
    mixed across sessions in a single PerceiverIOTrajectoryDecoder; downstream
    code groups by `session` before training.

    Mirrors the bookkeeping fields produced by `extract_pseudo_pop_flights` so
    plotting / metrics helpers can consume either source interchangeably.

    Returns
    -------
    list[dict] : keys 'pv', 'xyz', 'session', 'flight_id', 'flight_type'.
    """
    bm_all = meta_info_dict['behavior_mean']
    bs_all = meta_info_dict['behavior_std']

    flights = []
    for sn, loader in dataloaders[tier].items():
        bm, bs = bm_all[sn], bs_all[sn]
        for batch in loader:
            M = batch.mask.numpy()
            X = batch.behavior.numpy()
            spk = batch.spike.numpy()
            fpi = batch.flight_path_index.numpy()
            fti = batch.flight_type_index.numpy()
            xyz_full = _unzscore_xyz(X, bm, bs)

            for b in range(M.shape[0]):
                valid = M[b] == 1.0
                if not valid.any():
                    continue
                flights.append({
                    'pv': spk[b, valid].astype(np.float32, copy=False),
                    'xyz': xyz_full[b, valid].astype(np.float32, copy=False),
                    'session': sn,
                    'flight_id': int(fpi[b]),
                    'flight_type': int(fti[b]),
                })
    return flights


def build_super_ensemble(encoder, latents, session_order, device):
    """Map encoder latents to a concatenated pseudo-population.

    Used by `extract_pseudo_pop_flights`: each session's readout transforms the
    same latent vector into a per-session predicted-rate vector, and we
    concatenate across sessions to build a single fixed-dim pseudo-pop.
    """
    final_act = encoder.final_activation
    dtype_ref = next(encoder.parameters()).dtype
    lat_t = torch.from_numpy(latents).to(device=device, dtype=dtype_ref)
    parts = []
    with torch.no_grad():
        for sn in session_order:
            p = encoder.readouts[sn](lat_t)
            p = final_act(p).detach().cpu().numpy()
            parts.append(p)
    return np.concatenate(parts, axis=1)


def extract_pseudo_pop_flights(dataloaders, encoder, meta_info_dict, tier,
                               device):
    """Per-flight pseudo-population (encoder-predicted rates) + 3D positions.

    Run the encoder on each batch, capture the readout-input latents via a
    forward pre-hook, then push those latents through *every* session's readout
    and concatenate to form the pseudo-population. The result is a single
    flat input space (`sum(unit_num_dict.values())`) used by the global
    PV-Perceiver in `notebooks/pv_decode.ipynb`.
    """
    bm_all = meta_info_dict['behavior_mean']
    bs_all = meta_info_dict['behavior_std']
    session_order = list(dataloaders[tier].keys())

    flight_data_raw = []
    for sn in session_order:
        bm, bs = bm_all[sn], bs_all[sn]
        captured = []
        hook = encoder.readouts[sn].register_forward_pre_hook(
            lambda mod, inp: captured.append(inp[0].detach().cpu())
        )
        with torch.no_grad():
            for batch in dataloaders[tier][sn]:
                _ = encoder(sn, batch.behavior.to(device), batch.mask.to(device))
                lat = captured[-1]
                M = batch.mask.numpy()
                X = batch.behavior.numpy()
                fpi = batch.flight_path_index.numpy()
                fti = batch.flight_type_index.numpy()
                xyz_full = _unzscore_xyz(X, bm, bs)

                for b in range(M.shape[0]):
                    valid = M[b] == 1.0
                    if not valid.any():
                        continue
                    flight_data_raw.append({
                        'latent': lat[b, valid].numpy(),
                        'xyz': xyz_full[b, valid],
                        'session': sn,
                        'flight_id': int(fpi[b]),
                        'flight_type': int(fti[b]),
                    })
        hook.remove()

    all_latents = np.concatenate([f['latent'] for f in flight_data_raw], axis=0)
    all_preds = build_super_ensemble(encoder, all_latents, session_order, device)

    flights = []
    offset = 0
    for fd in flight_data_raw:
        n = fd['latent'].shape[0]
        flights.append({
            'pv': all_preds[offset:offset + n],
            'xyz': fd['xyz'],
            'session': fd['session'],
            'flight_id': fd['flight_id'],
            'flight_type': fd['flight_type'],
        })
        offset += n
    return flights


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PVTrajectoryDataset(Dataset):
    """Yields (population_vector, position_targets, validity_mask, t_sample, flight_idx).

    For each flight, samples `samples_per_flight` random t_sample bins at init.
    Call `resample()` between epochs to draw a fresh set (training only).

    The query grid is built from `time_mode`:
      - 'relative': times in [-T_max, +T_max] around t_sample
      - 'absolute': times in [0, T_max] from flight onset
    """

    def __init__(self, flights, time_mode, T_max, dt, samples_per_flight=10,
                 rng=None):
        self.flights = flights
        self.time_mode = time_mode
        self.T_max = T_max
        self.dt = dt
        self.samples_per_flight = samples_per_flight
        self.rng = rng or np.random.default_rng(42)

        if time_mode == 'relative':
            self.query_times = np.arange(-T_max, T_max + dt / 2, dt)
        else:
            self.query_times = np.arange(0, T_max + dt / 2, dt)
        self.O = len(self.query_times)
        self.query_offsets = np.round(self.query_times / dt).astype(np.int64)

        self._build_examples()

    def _build_examples(self):
        self.examples = []
        for fi, flight in enumerate(self.flights):
            T_flight = flight['xyz'].shape[0]
            t_samples = self.rng.choice(T_flight,
                                        size=self.samples_per_flight,
                                        replace=True)
            for ts in t_samples:
                self.examples.append((fi, int(ts)))

    def resample(self):
        self._build_examples()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        fi, t_sample = self.examples[idx]
        flight = self.flights[fi]
        trajectory = flight['xyz']
        T_flight = trajectory.shape[0]

        if self.time_mode == 'relative':
            flight_indices = t_sample + self.query_offsets
        else:
            flight_indices = self.query_offsets

        valid = (flight_indices >= 0) & (flight_indices < T_flight)
        clamped = np.clip(flight_indices, 0, T_flight - 1)
        position_targets = trajectory[clamped].astype(np.float32)
        position_targets[~valid] = 0.0
        validity_mask = valid.astype(np.float32)

        pv = flight['pv'][t_sample].astype(np.float32, copy=False)

        return (
            torch.from_numpy(pv),
            torch.from_numpy(position_targets),
            torch.from_numpy(validity_mask),
            t_sample,
            fi,
        )


# ---------------------------------------------------------------------------
# Hyperparameter search + training
# ---------------------------------------------------------------------------

def suggest_perceiver_hyperparams(trial, time_mode, n_neurons, T_max, dt, *,
                                  max_epochs=80, warmup_steps=100):
    """Build a `PerceiverIOConfig` from an Optuna trial.

    Defaults (`max_epochs=80`, `warmup_steps=100`) are tuned for the search;
    final retrain calls override with longer schedules (e.g. 500 / 500).

    `d_head` is searched instead of `n_heads` so the categorical space stays
    static across trials (Optuna requires a fixed value space). `n_heads` is
    derived as `D // d_head`; trials where `d_head > D` are pruned.
    """
    C = trial.suggest_categorical('C', [8, 16, 32, 64, 128])
    L = trial.suggest_categorical('L', [8, 16, 32, 64])
    D = trial.suggest_categorical('D', [64, 128, 256, 512])
    d_head = trial.suggest_categorical('d_head', [16, 32, 64, 128])
    if d_head > D:
        raise optuna.TrialPruned()
    n_heads = D // d_head
    n_process_layers = trial.suggest_int('n_process_layers', 1, 6)
    D_ff_multiplier = trial.suggest_categorical('D_ff_multiplier', [1, 2, 4])
    n_fourier_bands = trial.suggest_categorical('n_fourier_bands', [8, 16, 32])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 0.1, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    lambda_valid = trial.suggest_float('lambda_valid', 0.01, 1.0)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

    return PerceiverIOConfig(
        n_neurons=n_neurons, C=C, L=L, D=D, n_heads=n_heads,
        n_process_layers=n_process_layers, D_ff_multiplier=D_ff_multiplier,
        time_mode=time_mode, T_max=T_max, dt=dt,
        n_fourier_bands=n_fourier_bands, learning_rate=learning_rate,
        weight_decay=weight_decay, dropout_rate=dropout_rate,
        lambda_valid=lambda_valid, batch_size=batch_size,
        max_epochs=max_epochs, warmup_steps=warmup_steps,
    )


def fixed_perceiver_config(best_params, time_mode, n_neurons, T_max, dt, *,
                           max_epochs=500, warmup_steps=500):
    """Reproduce a config from a stored Optuna `best_params` dict.

    Wraps `optuna.trial.FixedTrial` so `suggest_perceiver_hyperparams` runs
    with the previously chosen values — keeps the search and retrain code
    paths in lock-step.
    """
    return suggest_perceiver_hyperparams(
        optuna.trial.FixedTrial(best_params), time_mode, n_neurons, T_max, dt,
        max_epochs=max_epochs, warmup_steps=warmup_steps,
    )


def trial_callback(study, trial):
    """Compact one-line per-trial summary (use as `study.optimize(callbacks=[…])`)."""
    if trial.state == optuna.trial.TrialState.COMPLETE:
        print(f'  trial {trial.number}: R²={trial.value:.4f}  '
              f'best={study.best_value:.4f}  params={trial.params}')
    elif trial.state == optuna.trial.TrialState.PRUNED:
        last = trial.intermediate_values
        last_r2 = list(last.values())[-1] if last else float('nan')
        print(f'  trial {trial.number}: PRUNED at epoch {len(last)} '
              f'(last R²={last_r2:.4f})')
    elif trial.state == optuna.trial.TrialState.FAIL:
        print(f'  trial {trial.number}: FAIL')


def train_perceiver(config, flights_train, flights_valid, device, *,
                    trial=None, verbose=True,
                    samples_per_flight_train=10, samples_per_flight_val=10,
                    random_seed=1, num_workers=2, val_batch_size=256):
    """Train a `PerceiverIOTrajectoryDecoder`. Returns (best_val_r2, weights, history).

    `trial` enables Optuna pruning; pass the active trial through. With
    `num_workers > 0` the workers are forked per-epoch (no `persistent_workers`),
    so `train_ds.resample()` between epochs takes effect for every iteration.

    Early stops on validation R² (across all valid (B, O) targets) with
    `config.patience` epochs of no improvement.
    """
    set_random_state(random_seed)

    train_ds = PVTrajectoryDataset(
        flights_train, config.time_mode, config.T_max, config.dt,
        samples_per_flight=samples_per_flight_train,
        rng=np.random.default_rng(random_seed),
    )
    val_ds = PVTrajectoryDataset(
        flights_valid, config.time_mode, config.T_max, config.dt,
        samples_per_flight=samples_per_flight_val,
        rng=np.random.default_rng(random_seed + 1),
    )

    pin = (device == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True, drop_last=False,
                              num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)

    model = PerceiverIOTrajectoryDecoder(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = config.max_epochs * len(train_loader)

    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / max(config.warmup_steps, 1)
        progress = (step - config.warmup_steps) / max(total_steps - config.warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_r2 = -np.inf
    best_weights = None
    patience_counter = 0
    history = {'train_loss': [], 'val_r2': [], 'val_pos_loss': [], 'val_valid_auc': []}

    iterator = (tqdm.tqdm(range(config.max_epochs), desc='Training')
                if verbose else range(config.max_epochs))
    for epoch in iterator:
        train_ds.resample()

        model.train()
        epoch_loss, n_batches = 0.0, 0
        for pv, pos_tgt, vmask, _, _ in train_loader:
            pv, pos_tgt, vmask = pv.to(device), pos_tgt.to(device), vmask.to(device)
            pos_pred, valid_logit, _, _ = model(pv)
            loss, _, _ = model.compute_loss(pos_pred, valid_logit, pos_tgt, vmask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
        history['train_loss'].append(epoch_loss / max(n_batches, 1))

        model.eval()
        all_pos_true, all_pos_pred = [], []
        all_valid_true, all_valid_logit = [], []
        with torch.no_grad():
            for pv, pos_tgt, vmask, _, _ in val_loader:
                pv, pos_tgt, vmask = pv.to(device), pos_tgt.to(device), vmask.to(device)
                pos_pred, valid_logit, _, _ = model(pv)
                mask_bool = vmask.bool()
                all_pos_true.append(pos_tgt[mask_bool].cpu().numpy())
                all_pos_pred.append(pos_pred[mask_bool].cpu().numpy())
                all_valid_true.append(vmask.cpu().numpy().ravel())
                all_valid_logit.append(valid_logit.cpu().numpy().ravel())

        pos_true_np = np.concatenate(all_pos_true)
        pos_pred_np = np.concatenate(all_pos_pred)
        val_r2 = float(r2_score(pos_true_np, pos_pred_np))
        val_pos_loss = float(np.mean((pos_true_np - pos_pred_np) ** 2))
        vt = np.concatenate(all_valid_true)
        vl = np.concatenate(all_valid_logit)
        val_auc = float(roc_auc_score(vt, vl)) if len(np.unique(vt)) > 1 else 0.5

        history['val_r2'].append(val_r2)
        history['val_pos_loss'].append(val_pos_loss)
        history['val_valid_auc'].append(val_auc)

        if trial is not None:
            trial.report(val_r2, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch + 1}')
                break

        if verbose and hasattr(iterator, 'set_postfix'):
            iterator.set_postfix(r2=f'{val_r2:.4f}',
                                 best=f'{best_val_r2:.4f}',
                                 auc=f'{val_auc:.3f}')

    return best_val_r2, best_weights, history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_perceiver(model, flights, config, device, *,
                       samples_per_flight=10, batch_size=256, num_workers=0,
                       rng_seed=0):
    """Evaluate on `flights`. Returns a long-format DataFrame, one row per query.

    Each (flight, t_sample) snippet contributes O records (= len(query_times)).
    Records carry `time_mode`, `query_time`, `t_abs` (absolute flight time of
    the queried bin), `delta_t` (offset from the PV-sample bin), the predicted
    and ground-truth (x, y, z), and the validity logit/target. `err_3d` is set
    on rows where `valid_true == 1.0`.
    """
    model.eval()
    ds = PVTrajectoryDataset(
        flights, config.time_mode, config.T_max, config.dt,
        samples_per_flight=samples_per_flight,
        rng=np.random.default_rng(rng_seed),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)

    records = []
    with torch.no_grad():
        for pv, pos_tgt, vmask, t_sample_batch, fi_batch in loader:
            pv = pv.to(device)
            pos_pred, valid_logit, _, _ = model(pv)
            pos_pred = pos_pred.cpu().numpy()
            valid_logit = valid_logit.cpu().numpy()
            pos_tgt = pos_tgt.numpy()
            vmask = vmask.numpy()
            t_sample_np = t_sample_batch.numpy()
            fi_np = fi_batch.numpy()

            B, O, _ = pos_pred.shape
            query_times = ds.query_times

            for b in range(B):
                t_s = int(t_sample_np[b])
                flight = flights[int(fi_np[b])]
                for i in range(O):
                    qt = float(query_times[i])
                    if config.time_mode == 'relative':
                        t_abs = (t_s + round(qt / config.dt)) * config.dt
                        delta_t = qt
                    else:
                        t_abs = qt
                        delta_t = qt - t_s * config.dt

                    records.append({
                        'time_mode': config.time_mode,
                        'query_time': qt,
                        't_abs': t_abs,
                        'delta_t': delta_t,
                        't_sample': t_s,
                        'flight_idx': int(fi_np[b]),
                        'session': flight['session'],
                        'flight_id': int(flight['flight_id']),
                        'valid_true': vmask[b, i],
                        'valid_logit': valid_logit[b, i],
                        'x_true': pos_tgt[b, i, 0] if vmask[b, i] else np.nan,
                        'y_true': pos_tgt[b, i, 1] if vmask[b, i] else np.nan,
                        'z_true': pos_tgt[b, i, 2] if vmask[b, i] else np.nan,
                        'x_pred': pos_pred[b, i, 0],
                        'y_pred': pos_pred[b, i, 1],
                        'z_pred': pos_pred[b, i, 2],
                    })

    df = pd.DataFrame(records)
    valid = df['valid_true'] == 1.0
    df.loc[valid, 'err_3d'] = np.sqrt(
        (df.loc[valid, 'x_true'] - df.loc[valid, 'x_pred']) ** 2 +
        (df.loc[valid, 'y_true'] - df.loc[valid, 'y_pred']) ** 2 +
        (df.loc[valid, 'z_true'] - df.loc[valid, 'z_pred']) ** 2
    )
    return df


def perceiver_per_timestep_df(snippets_df, dt):
    """Mean-pool PV-Perceiver snippet predictions to one row per (session, flight_id, t_idx).

    The PV-Perceiver decoder returns predictions over a query window centered
    on (or starting from) a randomly-sampled `t_sample` bin. With multiple
    `samples_per_flight`, several queries hit the same absolute flight bin —
    here we mean-pool them so the result has the same per-timestep schema as
    `nb_df` / `transformer_df` (one row per `(session, flight_id, t_idx)`),
    enabling them to be joined or concatenated for cross-decoder comparison.

    Parameters
    ----------
    snippets_df : long-format DataFrame from `evaluate_perceiver`. Only rows
        with `valid_true == 1.0` are kept.
    dt : seconds per bin (`data_config['bin_width']`).

    Returns
    -------
    DataFrame with columns: session, flight_id, t_idx, x_pred, y_pred, z_pred,
    x_true, y_true, z_true, err_2d, err_3d.
    """
    s = snippets_df[snippets_df['valid_true'] == 1.0].copy()
    s['t_idx'] = (s['t_abs'] / dt).round().astype(int)
    agg = (s.groupby(['session', 'flight_id', 't_idx'], observed=True)
            .agg(x_pred=('x_pred', 'mean'),
                 y_pred=('y_pred', 'mean'),
                 z_pred=('z_pred', 'mean'),
                 x_true=('x_true', 'mean'),
                 y_true=('y_true', 'mean'),
                 z_true=('z_true', 'mean'))
            .reset_index())
    agg['err_2d'] = np.sqrt(
        (agg['x_true'] - agg['x_pred']) ** 2
        + (agg['y_true'] - agg['y_pred']) ** 2
    )
    agg['err_3d'] = np.sqrt(
        (agg['x_true'] - agg['x_pred']) ** 2
        + (agg['y_true'] - agg['y_pred']) ** 2
        + (agg['z_true'] - agg['z_pred']) ** 2
    )
    return agg


__all__ = [
    'PVTrajectoryDataset',
    'build_super_ensemble',
    'evaluate_perceiver',
    'extract_pseudo_pop_flights',
    'extract_real_flights',
    'fixed_perceiver_config',
    'perceiver_per_timestep_df',
    'suggest_perceiver_hyperparams',
    'train_perceiver',
    'trial_callback',
]
