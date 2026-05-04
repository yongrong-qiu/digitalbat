"""Pseudo-population helpers: feed bat_former encoder predictions into per-session
decoders that originally consumed raw spike counts.

The PV-Perceiver decoders in `notebooks/pv_decode.ipynb` read from a 3369-dim
"pseudo-population" — the bat_former encoder's latent run through every
session's readout and concatenated. To make a fair comparison with NB and
seq-to-seq decoders, those models also need to see the pseudo-population
instead of K-dim real spike counts.

This module provides:

  - `PseudoPopLoader` : wraps an existing `Bat_data_loader` per-session loader
    and yields batches whose `.spike` field is replaced with the matching
    pseudo-population tensor (per flight, picked from a precomputed lookup).

  - `make_pp_dataloaders` : applies `PseudoPopLoader` across a `dataloaders`
    dict (train/valid/test × session).

  - `train_s2s_pp` / `eval_s2s_pp` : train + evaluate a `Bat_Transformer_Decoder`
    whose `input_dim_dict` is `{session: n_pseudo}` (per-session readins, same
    3369-dim input across all sessions).
"""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from digitalbat.models.decoder import Bat_Transformer_Decoder, bat_decoder_trainer


def build_pp_lookup(flights):
    """Per-flight pseudo-pop lookup keyed on (session, flight_id).

    Each entry is the (T_flight, n_pseudo) array from `extract_flights`. The
    notebook's `flights_train` / `flights_valid` / `flights_test` lists are
    accepted directly.
    """
    return {(f['session'], int(f['flight_id'])): f['pv'] for f in flights}


class PseudoPopLoader:
    """Iterable wrapper that swaps `.spike` for a pseudo-population tensor.

    Each call to `iter(self)` triggers a fresh pass of the wrapped loader,
    so this is safe to use in `bat_decoder_trainer`'s per-epoch loop. Lengths
    of valid timesteps must match between the dataloader's mask and the stored
    pseudo-pop array — otherwise that flight is silently skipped (the batch
    row stays zero, which is masked out downstream).
    """

    def __init__(self, orig_loader, session_name: str, pp_lookup, n_pseudo: int):
        self.orig = orig_loader
        self.session_name = session_name
        self.pp_lookup = pp_lookup
        self.n_pseudo = n_pseudo

    def __iter__(self):
        for batch in self.orig:
            B, T_max = batch.mask.shape
            X_pp = torch.zeros(B, T_max, self.n_pseudo, dtype=torch.float32)
            for b in range(B):
                fid = int(batch.flight_path_index[b])
                pp = self.pp_lookup.get((self.session_name, fid))
                if pp is None:
                    continue
                valid_idx = torch.where(batch.mask[b] == 1.0)[0]
                if len(valid_idx) != pp.shape[0]:
                    continue  # mask/pp length mismatch — drop silently
                X_pp[b, valid_idx, :] = torch.from_numpy(pp).float()
            yield batch._replace(spike=X_pp)


def make_pp_dataloaders(dataloaders, flights_by_split, n_pseudo):
    """Wrap `dataloaders[split][session]` so `.spike` becomes the pseudo-pop tensor.

    Parameters
    ----------
    dataloaders : dict {'train': {session: loader}, 'valid': ..., 'test': ...}
    flights_by_split : dict {'train': flights_list, 'valid': ..., 'test': ...}
    n_pseudo : int — pseudo-population dimensionality (e.g. 3369).
    """
    out = {}
    for split, per_session in dataloaders.items():
        flights = flights_by_split.get(split, [])
        pp_lookup = build_pp_lookup(flights)
        out[split] = {
            sn: PseudoPopLoader(loader, sn, pp_lookup, n_pseudo)
            for sn, loader in per_session.items()
        }
    return out


def s2s_pp_objective(trial, pp_dataloaders, unit_num_dict, n_pseudo, device, *,
                     seed, max_epochs=1000, patience=30):
    """Optuna objective for the seq-to-seq transformer on pseudo-population.

    Search space and trainer settings mirror
    `notebooks/decode-position-comparison.ipynb`. Returns the trainer's
    best_score (negative mean MSE per dim) — higher is better.
    """
    import torch.nn as nn
    from digitalbat.utility.utils import set_random_state
    from digitalbat.models.decoder import bat_decoder_trainer

    params = dict(
        latent_dim=trial.suggest_categorical('latent_dim', [32, 64, 128]),
        ff_dim=trial.suggest_categorical('ff_dim', [32, 64, 128]),
        num_layers=trial.suggest_int('num_layers', 1, 3),
        dropout_prob=trial.suggest_float('dropout_prob', 0.0, 0.5),
        readin_dropout=trial.suggest_float('readin_dropout', 0.0, 0.3),
        input_transform=trial.suggest_categorical('input_transform', ['identity', 'sqrt']),
    )
    set_random_state(seed)
    model = make_s2s_pp(unit_num_dict, n_pseudo, params=params,
                        final_activation=nn.Identity())
    best_score, _, _ = bat_decoder_trainer(
        model=model, dataloaders=pp_dataloaders, device=device, pos_dims=3,
        n_epochs=max_epochs, lr=1e-3, weight_decay=0.01,
        loss_fn=nn.MSELoss(), valid_stat='mean',
        early_stopping_patience=patience, save_model=False,
    )
    return best_score


def run_s2s_pp_optuna(pp_dataloaders, unit_num_dict, n_pseudo, device, *,
                     seed=1, n_trials=20, max_epochs=1000, patience=30,
                     study_path='./out/bat_decoder_pp_seed_1_optuna.pkl',
                     verbose=False):
    """Run (or resume) the s2s pseudo-pop Optuna study, return best_params."""
    import os
    import pickle
    import optuna

    if os.path.exists(study_path):
        with open(study_path, 'rb') as f:
            study = pickle.load(f)
        print(f's2s pp: loaded study with {len(study.trials)} trials')
    else:
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=seed),
        )

    remaining = max(0, n_trials - len(study.trials))
    if remaining > 0:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(
            lambda t: s2s_pp_objective(
                t, pp_dataloaders, unit_num_dict, n_pseudo, device,
                seed=seed, max_epochs=max_epochs, patience=patience,
            ),
            n_trials=remaining, show_progress_bar=verbose,
        )
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)

    print(f's2s pp best score (mean -MSE per dim): {study.best_value:.4f}')
    print('Best params:')
    for k, v in study.best_params.items():
        print(f'  {k}: {v}')
    return study.best_params


def make_s2s_pp(unit_num_dict, n_pseudo, *, params, output_dim=3,
                final_activation=None):
    """Construct a `Bat_Transformer_Decoder` whose readins all see `n_pseudo` features.

    Per-session readins are kept (one Linear per session, all 3369→latent_dim) —
    same architecture as the spike-input s2s, just with input_dim replaced by
    `n_pseudo` for every session. Architectural hyperparameters come from
    `params` (typically the decode-position-comparison Optuna `best_params`).
    """
    if final_activation is None:
        final_activation = torch.nn.Identity()
    return Bat_Transformer_Decoder(
        input_dim_dict={sn: n_pseudo for sn in unit_num_dict},
        output_dim=output_dim,
        num_heads=4,
        final_activation=final_activation,
        latent_dim=params['latent_dim'],
        ff_dim=params['ff_dim'],
        num_layers=params['num_layers'],
        dropout_prob=params['dropout_prob'],
        readin_dropout=params['readin_dropout'],
        input_transform=params['input_transform'],
    )


def train_s2s_pp(model, pp_dataloaders, device, *,
                 lr, weight_decay, n_epochs, early_stopping_patience=20,
                 valid_fn=None, save_path=None):
    """Thin wrapper around `bat_decoder_trainer` for clarity at call sites."""
    return bat_decoder_trainer(
        model, pp_dataloaders, device,
        lr=lr, weight_decay=weight_decay, n_epochs=n_epochs,
        early_stopping_patience=early_stopping_patience,
        valid_fn=valid_fn,
        save_model=(save_path is not None),
        model_name=save_path or 'best_decoder_weights.pth',
    )


def eval_s2s_pp(model, pp_dataloaders, meta_info_dict, *,
                device, dt, split='test'):
    """Predict per-timestep position with `model` and return a long-format DataFrame.

    Mirrors the s2s-eval cell in `notebooks/pv_decode.ipynb`: un-z-scores both
    truth and prediction per session, then emits one record per valid timestep.
    """
    model.eval()
    records = []
    with torch.no_grad():
        for session_name, session_data in pp_dataloaders[split].items():
            bm = meta_info_dict['behavior_mean'][session_name].squeeze()
            bs = meta_info_dict['behavior_std'][session_name].squeeze()
            for batch in session_data:
                X = batch.spike.to(device)
                m = batch.mask.to(device)
                y_true_z = batch.behavior[:, :, :3].numpy()
                y_pred_z = model(session_name, X, m).cpu().numpy()
                mask_np = batch.mask.numpy().astype(bool)
                B, T = mask_np.shape
                y_true = y_true_z * bs[None, None, :3] + bm[None, None, :3]
                y_pred = y_pred_z * bs[None, None, :3] + bm[None, None, :3]
                fpi = batch.flight_path_index.numpy()

                for b in range(B):
                    for t in range(T):
                        if not mask_np[b, t]:
                            continue
                        err = float(np.sqrt(((y_true[b, t] - y_pred[b, t]) ** 2).sum()))
                        records.append({
                            't_abs': t * dt,
                            'session': session_name,
                            'flight_id': int(fpi[b]),
                            'err_3d': err,
                            'x_true': float(y_true[b, t, 0]),
                            'y_true': float(y_true[b, t, 1]),
                            'z_true': float(y_true[b, t, 2]),
                            'x_pred': float(y_pred[b, t, 0]),
                            'y_pred': float(y_pred[b, t, 1]),
                            'z_pred': float(y_pred[b, t, 2]),
                        })
    return pd.DataFrame(records)
