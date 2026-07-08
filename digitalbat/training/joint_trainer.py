"""Hyperparameter search + training for the JointMaskedTransformer consensus model.

Extracted from ``notebooks/joint-encode-decode.ipynb`` (§6) so any notebook can
train the position-only consensus model on demand (train-if-missing). The masking
schemes, :func:`sample_mask`, and :func:`joint_loss` live in
:mod:`digitalbat.models.joint`; the round-robin batcher in
:mod:`digitalbat.utility.utils`. This module adds the Pareto-selecting training
loop (:func:`train_joint`), the validation metrics, and the Optuna orchestrator
:func:`ensure_joint_model` that reproduces the position-only (``use_kinematics=False``)
consensus checkpoint.

Notebook globals ``meta_info_dict`` and ``SEED`` are threaded through as the
``meta`` and ``seed`` arguments here.
"""
import os
import copy
import random
import pickle

import numpy as np
import torch
from torch import nn
import tqdm

from ..models.joint import JointMaskedTransformer, SCHEMES, sample_mask, joint_loss
from ..utility.utils import (round_robin_session_batches, poisson_fde_population,
                             set_random_state)

__all__ = ['train_joint', 'scheme_weights', 'ensure_joint_model']

# Defaults matching joint-encode-decode.ipynb §6.
INPUT_TRANSFORM = 'identity'
LAYER_CHOICES = [1, 2, 4]
START_LAYERS = 4
ERR_REF, FDE_REF = 0.3, 0.1
TRIAL_EPOCHS = 300
FINAL_EPOCHS = 1000
N_TRIALS = 200


@torch.no_grad()
def _decode_median_err_m(model, loaders, meta, device):
    """Mean over sessions of median 3-D decode error (metres) on a split."""
    model.eval()
    per_session = []
    for s, sd in loaders.items():
        std = np.asarray(meta['behavior_std'][s]).squeeze()[:3]
        errs = []
        for b in sd:
            m = b.mask.to(device)
            bt = b.mask.clone().to(device)            # decode: all behaviour masked
            nt = torch.zeros_like(bt)
            _, beh_pred, _ = model(s, b.spike.to(device), b.behavior.to(device), m, nt, bt)
            yp = beh_pred[:, :, :3].cpu().numpy()
            yt = b.behavior[:, :, :3].numpy()
            mk = b.mask.numpy().astype(bool)
            resid = (yt - yp)[mk] * std[None, :]
            if resid.size:
                errs.append(np.sqrt((resid ** 2).sum(axis=1)))
        if errs:
            per_session.append(float(np.median(np.concatenate(errs))))
    return float(np.mean(per_session)) if per_session else np.inf


@torch.no_grad()
def _encode_median_fde(model, loaders, device):
    """Median FDE across neurons, encode task (spikes <- behaviour)."""
    model.eval()
    fdes = []
    for s, sd in loaders.items():
        yt_l, yp_l = [], []
        for b in sd:
            m = b.mask.to(device)
            nt = b.mask.clone().to(device)            # encode: all neural masked
            bt = torch.zeros_like(nt)
            rate, _, _ = model(s, b.spike.to(device), b.behavior.to(device), m, nt, bt)
            mk = b.mask.numpy().astype(bool)
            yt_l.append(b.spike.numpy()[mk]); yp_l.append(rate.cpu().numpy()[mk])
        if yt_l:
            fdes.append(poisson_fde_population(np.vstack(yt_l), np.vstack(yp_l)))
    return float(np.nanmedian(np.hstack(fdes))) if fdes else np.nan


def _cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _pareto_update(cand, err, fde, model):
    """Maintain a Pareto front over (err down, fde up). Store a CPU checkpoint only
    when the new (err, fde) is non-dominated; drop members it dominates. Returns
    (cand, improved)."""
    dominated = any((o['err'] <= err and o['fde'] >= fde) and
                    (o['err'] < err or o['fde'] > fde) for o in cand)
    if dominated:
        return cand, False
    cand = [o for o in cand if not ((err <= o['err'] and fde >= o['fde']) and
                                     (err < o['err'] or fde > o['fde']))]
    cand.append(dict(err=err, fde=fde, w=_cpu_state(model)))
    return cand, True


def train_joint(model, dataloaders, device, meta, *, seed=1, n_epochs=300, lr=3e-3,
                weight_decay=1e-2, lam=1.0, pos_weight=1.0, schemes=SCHEMES, scheme_p=None,
                select='joint', fde_weight=0.5, err_ref=1.0, fde_ref=0.01,
                early_stopping_patience=40, verbose=True):
    """select='joint': pick the Pareto-front checkpoint maximising
    fde_weight*(FDE/fde_ref) - (1-fde_weight)*(err/err_ref). Dividing by reference
    magnitudes (err_ref m, fde_ref) puts the two on a common raw scale: with
    err_ref=1.0 and fde_ref=0.01 a 0.01 FDE gain counts as a 1 m error gain, i.e.
    FDE is effectively up-weighted ~100x (its raw scale is ~100x smaller). 'decode':
    legacy min-decode-error. Returns (model, best, hist) with best a dict of the
    selected epoch's err / fde / joint score."""
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    pois_fn = nn.PoissonNLLLoss(log_input=False, reduction='none').to(device)
    hist = {'val_err_m': [], 'val_fde': []}
    rng = random.Random(seed)
    best_err, best_w, no_improve = np.inf, None, 0          # select='decode'
    cand = []                                               # select='joint' Pareto front
    for ep in tqdm.tqdm(range(n_epochs), desc='joint'):
        model.train()
        for s, b in round_robin_session_batches(dataloaders['train']):
            scheme = rng.choices(schemes, weights=scheme_p)[0]
            m = b.mask.to(device)
            nt, bt = sample_mask(b.mask, scheme)
            nt, bt = nt.to(device), bt.to(device)
            rate, beh, _ = model(s, b.spike.to(device), b.behavior.to(device), m, nt, bt)
            loss = joint_loss(rate, beh, b.spike.to(device), b.behavior.to(device),
                              m, nt, bt, lam, pois_fn, pos_weight=pos_weight)
            opt.zero_grad(); loss.backward(); opt.step()
        err = _decode_median_err_m(model, dataloaders['valid'], meta, device)
        fde = _encode_median_fde(model, dataloaders['valid'], device)
        hist['val_err_m'].append(err); hist['val_fde'].append(fde)
        if select == 'decode':
            improved = err < best_err
            if improved:
                best_err, best_w = err, copy.deepcopy(model.state_dict())
        else:
            cand, improved = _pareto_update(cand, err, fde, model)
        no_improve = 0 if improved else no_improve + 1
        if early_stopping_patience and no_improve >= early_stopping_patience:
            if verbose:
                print(f'early stop @ epoch {ep + 1}')
            break
    if select == 'joint':
        errs = np.array([o['err'] for o in cand], float)
        fdes = np.array([o['fde'] for o in cand], float)
        joint = fde_weight * (fdes / fde_ref) - (1.0 - fde_weight) * (errs / err_ref)
        bi = int(np.nanargmax(joint))
        model.load_state_dict(cand[bi]['w'])
        best = dict(err=float(errs[bi]), fde=float(fdes[bi]), joint=float(joint[bi]))
    else:
        if best_w is not None:
            model.load_state_dict(best_w)
        best = dict(err=float(best_err), fde=np.nan, joint=np.nan)
    return model, best, hist


def scheme_weights(params):
    """Per-task sampling weights in SCHEMES order (encode/decode up-weighted vs w_aux)."""
    wmap = {'encode': params['w_encode'], 'decode': params['w_decode']}
    return [wmap.get(s, params['w_aux']) for s in SCHEMES]


def _build_joint(params, unit_num_dict, session_to_room, input_transform, num_heads, seed):
    set_random_state(seed)
    return JointMaskedTransformer(
        unit_num_dict, behavior_dim=3, latent_dim=params['latent_dim'], num_heads=num_heads,
        ff_dim=params['ff_dim'], num_layers=params['num_layers'], dropout=params['dropout'],
        input_transform=input_transform, session_to_room=session_to_room)


def _ckpt_path(out_dir, p, n_sessions, input_transform):
    """Position-only (kin0) consensus checkpoint name (matches joint-encode-decode §6.3)."""
    return (f"{out_dir}/joint_v2_kin0_L{int(p['num_layers'])}_d{p['latent_dim']}"
            f"_ff{p['ff_dim']}_lam{p['lam']:.2g}_pw{p['pos_weight']:.2f}_{input_transform}"
            f"_we{p['w_encode']:.1f}_wd{p['w_decode']:.1f}_wa{p['w_aux']:.2f}_n{n_sessions}.pth")


def ensure_joint_model(dataloaders, unit_num_dict, session_to_room, meta, device, *,
                       seed=1, out_dir='./out/joint', n_trials=N_TRIALS,
                       input_transform=INPUT_TRANSFORM, num_heads=4,
                       err_ref=ERR_REF, fde_ref=FDE_REF, trial_epochs=TRIAL_EPOCHS,
                       final_epochs=FINAL_EPOCHS, layer_choices=None, start_layers=START_LAYERS,
                       study_path=None):
    """Load-or-(search + train + save) the position-only consensus JointMaskedTransformer.

    Runs an Optuna search of `n_trials` over the §6.3 v2 space with
    ``use_kinematics`` fixed False and ``fde_weight=0.5``, caches the study, then
    trains the best config for `final_epochs` and saves it. The search is a no-op if
    the cached study already has >= `n_trials` trials, and training is skipped if the
    derived checkpoint already exists. Returns ``(model, best_params, ckpt_path)``.
    """
    import optuna
    layer_choices = list(layer_choices) if layer_choices is not None else list(LAYER_CHOICES)
    os.makedirs(out_dir, exist_ok=True)
    study_path = study_path or f'{out_dir}/joint_optuna_v2_kin0.pkl'

    def objective(trial):
        params = dict(
            num_layers=trial.suggest_categorical('num_layers', layer_choices),
            latent_dim=trial.suggest_categorical('latent_dim', [32, 64, 128]),
            ff_dim=trial.suggest_categorical('ff_dim', [32, 64, 128]),
            dropout=trial.suggest_float('dropout', 0.0, 0.4),
            lr=trial.suggest_float('lr', 1e-4, 3e-2, log=True),
            weight_decay=trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True),
            w_encode=trial.suggest_float('w_encode', 1.0, 12.0, log=True),
            w_decode=trial.suggest_float('w_decode', 1.0, 12.0, log=True),
            w_aux=trial.suggest_float('w_aux', 0.1, 1.0, log=True),
            lam=trial.suggest_float('lam', 1e-2, 1e1, log=True),
            pos_weight=trial.suggest_float('pos_weight', 0.0, 1.0))
        model = _build_joint(params, unit_num_dict, session_to_room, input_transform,
                             num_heads, seed)
        _, best, _ = train_joint(model, dataloaders, device, meta, seed=seed,
                                 n_epochs=trial_epochs, lr=params['lr'],
                                 weight_decay=params['weight_decay'], lam=params['lam'],
                                 pos_weight=params['pos_weight'],
                                 scheme_p=scheme_weights(params), select='joint',
                                 fde_weight=0.5, err_ref=err_ref, fde_ref=fde_ref, verbose=False)
        trial.set_user_attr('err_m', best['err']); trial.set_user_attr('fde', best['fde'])
        return best['joint']

    start = dict(num_layers=start_layers, latent_dim=64, ff_dim=64, dropout=0.2, lr=3e-3,
                 weight_decay=1e-2, w_encode=1.0, w_decode=1.0, w_aux=1.0, lam=1.0, pos_weight=1.0)
    if os.path.exists(study_path):
        with open(study_path, 'rb') as f:
            study = pickle.load(f)
        print(f'[joint] loaded study with {len(study.trials)} trials')
    else:
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=seed))
        study.enqueue_trial(start)
    remaining = max(0, n_trials - len(study.trials))
    if remaining > 0:
        print(f'[joint] running {remaining} Optuna trials (each trains a full model)...')
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=remaining, show_progress_bar=True)
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)

    best_params = dict(start); best_params.update(study.best_params)
    print(f'[joint] best joint score {study.best_value:.4f}; params {best_params}')
    ckpt_path = _ckpt_path(out_dir, best_params, len(unit_num_dict), input_transform)
    model = _build_joint(best_params, unit_num_dict, session_to_room, input_transform,
                         num_heads, seed).to(device)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f'[joint] loaded final checkpoint {ckpt_path}')
    else:
        print(f'[joint] training final consensus model ({final_epochs} epochs)...')
        model, _b, _ = train_joint(model, dataloaders, device, meta, seed=seed,
                                   n_epochs=final_epochs, lr=best_params['lr'],
                                   weight_decay=best_params['weight_decay'],
                                   lam=best_params['lam'], pos_weight=best_params['pos_weight'],
                                   scheme_p=scheme_weights(best_params), select='joint',
                                   fde_weight=0.5, err_ref=err_ref, fde_ref=fde_ref)
        torch.save(model.state_dict(), ckpt_path)
        print(f'[joint] saved {ckpt_path}')
    return model.eval(), best_params, ckpt_path
