"""Foundation-transfer with the JOINT (NEDS) model, single-stage, no HP search.

Ports the old Bat_Transformer transfer sweep (foundation_transfer.ipynb) to the
updated JointMaskedTransformer used paper-wide. For each leave-one-out fold we train
a `foundation` on the source sessions, then per held-out target a `scratch`
baseline, a `frozen` (core + behaviour read-in/read-out frozen; only the target's
neural read-in/read-out trained) adapter, and a `fine-tune` (core+behaviour at
0.1x LR, target neural at full LR) adapter. Every model is trained with a single
fixed-HP `train_joint` over the 6 standard masking schemes (no Optuna, no
base+infillft two-stage). Both encode (per-neuron FDE/corr) and decode (per-flight
3-D Fréchet + recon error) are scored on the target's test split.

  .pth checkpoints -> /media/alowet/ssd2/twin/transfer_joint/{group}/
  result pkls       -> notebooks/out/transfer_joint/{group}/results_{group}_{enc,dec}.pkl

Usage (run from repo root or notebooks/):
  python tmp/train_joint_transfer.py --mode dry --fold Dataset_1_32622_231009
  python tmp/train_joint_transfer.py --mode both        # full run (both CV schemes)
"""
import os
import sys
import copy
import random
import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn

# Run relative to notebooks/ so ./out/... paths match the notebooks.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(os.path.join(REPO, 'notebooks'))

from digitalbat.datasets.defaults import DEFAULT_SESSION_NAMES, make_data_config
from digitalbat.datasets.kinematics import make_joint_dataloaders
from digitalbat.models.joint import (JointMaskedTransformer, JointEncoderAdapter,
                                      SCHEMES, sample_mask, within_double_batch,
                                      forecast_input_mask, joint_loss, encode_eval_records)
from digitalbat.training.joint_trainer import (train_joint, scheme_weights,
                                               _decode_median_err_m, _encode_median_fde,
                                               _pareto_update)
from digitalbat.utility.utils import (round_robin_session_batches, parse_bat,
                                      set_random_state)
from digitalbat.decoding.geometry import discrete_frechet

# ---------------------------------------------------------------------------
# Fixed configuration (matches figures.ipynb's consensus joint model; no search)
# ---------------------------------------------------------------------------
SEED = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_TRANSFORM = 'identity'
JOINT_PARAMS = dict(num_layers=4, latent_dim=64, num_heads=4, ff_dim=64, dropout=0.2)
LAM, POS_W = 0.61, 0.84
SCHEME_WEIGHTS_PARAMS = dict(w_encode=1.9, w_decode=2.8, w_aux=0.67)
SCHEME_P = scheme_weights(SCHEME_WEIGHTS_PARAMS)          # per-scheme sampling weights
LR, WD = 3e-3, 1e-2                                        # train_joint / START_PARAMS defaults
CORE_FRAC = 0.1                                            # fine-tune core LR = CORE_FRAC * LR
# Selection (Pareto joint score) — matches joint-encode-decode.ipynb §6.
FDE_WEIGHT, ERR_REF, FDE_REF = 0.5, 0.3, 0.1
# Epoch budgets (early stopping cuts these short). Env overrides ease smoke tests.
_E = lambda k, d: int(os.environ.get(k, d))
FOUND_EPOCHS, FOUND_PATIENCE = _E('TJT_FOUND_EPOCHS', 500), _E('TJT_FOUND_PATIENCE', 40)
SCRATCH_EPOCHS, SCRATCH_PATIENCE = _E('TJT_SCRATCH_EPOCHS', 500), _E('TJT_SCRATCH_PATIENCE', 40)
FT_EPOCHS, FT_PATIENCE = _E('TJT_FT_EPOCHS', 500), _E('TJT_FT_PATIENCE', 40)

CKPT_ROOT = os.environ.get('TJT_CKPT_ROOT', '/media/alowet/ssd2/twin/transfer_joint')
PKL_ROOT = os.environ.get('TJT_PKL_ROOT', os.path.join(REPO, 'notebooks', 'out', 'transfer_joint'))


def joint_session_room(s):
    """3-config room map (matches figures.ipynb / joint-encode-decode.ipynb)."""
    if '_fs' in s:
        return 'krishna_fs'
    if s.startswith('Dataset_2_'):
        return 'dataset_2'
    return 'lab'


# ---------------------------------------------------------------------------
# Data (loaded once, 17-session joint universe)
# ---------------------------------------------------------------------------
print(f'device={DEVICE} | loading joint dataloaders ...', flush=True)
DATA_CONFIG = make_data_config(session_names=list(DEFAULT_SESSION_NAMES))
DL, _DL_KIN, UNIT, META = make_joint_dataloaders(DATA_CONFIG)
SESSIONS = list(UNIT)
ROOM = {s: joint_session_room(s) for s in SESSIONS}
BATS = sorted({parse_bat(s) for s in SESSIONS})
print(f'{len(SESSIONS)} sessions | {sum(UNIT.values())} neurons | {len(BATS)} bats', flush=True)


# ---------------------------------------------------------------------------
# Model builders / masked training loop for the freeze + param-group modes
# ---------------------------------------------------------------------------
def build_model(unit_sub, room_sub):
    set_random_state(SEED)
    return JointMaskedTransformer(unit_sub, behavior_dim=3, input_transform=INPUT_TRANSFORM,
                                  session_to_room=room_sub, **JOINT_PARAMS).to(DEVICE)


def _select_best(cand):
    errs = np.array([o['err'] for o in cand], float)
    fdes = np.array([o['fde'] for o in cand], float)
    joint = FDE_WEIGHT * (fdes / FDE_REF) - (1.0 - FDE_WEIGHT) * (errs / ERR_REF)
    return int(np.nanargmax(joint)), errs, fdes


def _train_masked(model, train_loaders, valid_loaders, optimizer, *, n_epochs, patience):
    """train_joint's inner loop with an externally-built optimizer (frozen / fine-tune).
    Selects the Pareto-front checkpoint by the same joint score as the main model."""
    pois_fn = nn.PoissonNLLLoss(log_input=False, reduction='none').to(DEVICE)
    rng = random.Random(SEED)
    cand, no_improve = [], 0
    for ep in range(n_epochs):
        model.train()
        for s, b in round_robin_session_batches(train_loaders):
            scheme = rng.choices(SCHEMES, weights=SCHEME_P)[0]
            spike, behavior, m = b.spike.to(DEVICE), b.behavior.to(DEVICE), b.mask.to(DEVICE)
            if scheme == 'within':
                spike = torch.cat([spike, spike], 0)
                behavior = torch.cat([behavior, behavior], 0)
                m = torch.cat([m, m], 0)
                nt, bt = within_double_batch(b.mask); n_in, b_in = nt, bt
            elif scheme in ('forecast', 'forecast_stim'):
                nt, bt = sample_mask(b.mask, scheme)
                n_in, b_in = nt, forecast_input_mask(b.mask, scheme)
            else:
                nt, bt = sample_mask(b.mask, scheme); n_in, b_in = nt, bt
            n_in, b_in, nt, bt = (t.to(DEVICE) for t in (n_in, b_in, nt, bt))
            rate, beh, _ = model(s, spike, behavior, m, n_in, b_in)
            loss = joint_loss(rate, beh, spike, behavior, m, nt, bt, LAM, pois_fn, pos_weight=POS_W)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        err = _decode_median_err_m(model, valid_loaders, META, DEVICE)
        fde = _encode_median_fde(model, valid_loaders, DEVICE)
        cand, improved = _pareto_update(cand, err, fde, model)
        no_improve = 0 if improved else no_improve + 1
        if patience and no_improve >= patience:
            break
    bi, errs, fdes = _select_best(cand)
    model.load_state_dict(cand[bi]['w'])
    return model, dict(err=float(errs[bi]), fde=float(fdes[bi]))


# ---------------------------------------------------------------------------
# The four training modes (train-if-missing)
# ---------------------------------------------------------------------------
def _save(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def train_full(sessions, path, *, n_epochs, patience, tag):
    """foundation (many sessions) or scratch (one session): train_joint, all params."""
    if os.path.exists(path):
        return
    unit_sub = {s: UNIT[s] for s in sessions}
    room_sub = {s: ROOM[s] for s in sessions}
    model = build_model(unit_sub, room_sub)
    tl = {s: DL['train'][s] for s in sessions}
    vl = {s: DL['valid'][s] for s in sessions}
    model, best, _ = train_joint(model, {'train': tl, 'valid': vl}, DEVICE, META,
                                 seed=SEED, n_epochs=n_epochs, lr=LR, weight_decay=WD,
                                 lam=LAM, pos_weight=POS_W, scheme_p=SCHEME_P,
                                 select='joint', fde_weight=FDE_WEIGHT, err_ref=ERR_REF,
                                 fde_ref=FDE_REF, early_stopping_patience=patience, verbose=False)
    _save(model, path)
    print(f'  [{tag}] {os.path.basename(path)}  err={best["err"]:.3f} fde={best["fde"]:.3f}', flush=True)


def train_adapt(target, sources, foundation_path, mode, path):
    """frozen / fine-tune: warm-start from foundation, train the target's
    foundation-absent weights (`target_new`: its neural read-in/read-out, plus — for
    cross-environment folds where the target's room is unseen — that room's behaviour
    read-in/read-out). frozen: only those; fine-tune: those at LR, the shared core +
    seen-room behaviour at 0.1x LR. Other sessions' neural layers stay frozen."""
    if os.path.exists(path):
        return
    all_s = list(sources) + [target]
    model = build_model({s: UNIT[s] for s in all_s}, {s: ROOM[s] for s in all_s})
    state = torch.load(foundation_path, map_location=DEVICE)
    tgt_pref = (f'spike_readin.{target}.', f'spike_readout.{target}.')
    filtered = {k: v for k, v in state.items() if not k.startswith(tgt_pref)}
    msg = model.load_state_dict(filtered, strict=False)
    assert not msg.unexpected_keys, msg.unexpected_keys
    # Missing keys = weights the foundation never had (random init -> must be trained):
    # always the target's neural heads; for a novel room, also that room's behaviour heads.
    src_rooms = {ROOM[s] for s in sources}
    tgt_room = ROOM[target]
    allowed = tgt_pref if tgt_room in src_rooms else tgt_pref + (
        f'beh_readin.{tgt_room}.', f'beh_readout.{tgt_room}.')
    assert all(k.startswith(allowed) for k in msg.missing_keys), msg.missing_keys

    missing = set(msg.missing_keys)
    src_neural = tuple(f'spike_readin.{s}.' for s in sources) + tuple(f'spike_readout.{s}.' for s in sources)
    target_new, other_neural, core_beh = [], [], []
    for n, p in model.named_parameters():
        if n in missing:
            target_new.append(p)              # target-specific, foundation-absent
        elif n.startswith(src_neural):
            other_neural.append(p)            # source sessions' neural heads
        else:
            core_beh.append(p)                # shared transformer core + seen-room behaviour

    for p in other_neural:
        p.requires_grad = False
    for p in target_new:
        p.requires_grad = True
    if mode == 'frozen':
        for p in core_beh:
            p.requires_grad = False
        opt = torch.optim.AdamW(target_new, lr=LR, weight_decay=WD)
    else:  # fine-tune
        for p in core_beh:
            p.requires_grad = True
        opt = torch.optim.AdamW([{'params': core_beh, 'lr': LR * CORE_FRAC, 'weight_decay': WD},
                                 {'params': target_new, 'lr': LR, 'weight_decay': WD}])
    tl = {target: DL['train'][target]}
    vl = {target: DL['valid'][target]}
    model, best = _train_masked(model, tl, vl, opt, n_epochs=FT_EPOCHS, patience=FT_PATIENCE)
    _save(model, path)
    print(f'  [{mode}] {os.path.basename(path)}  err={best["err"]:.3f} fde={best["fde"]:.3f}', flush=True)


# ---------------------------------------------------------------------------
# Evaluation (encode per-neuron FDE/corr; decode per-flight Fréchet + recon)
# ---------------------------------------------------------------------------
def _load_for_eval(path, sessions):
    model = build_model({s: UNIT[s] for s in sessions}, {s: ROOM[s] for s in sessions})
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model.eval()


@torch.no_grad()
def encode_eval(model, target, sessions):
    adapter = JointEncoderAdapter(model, {s: UNIT[s] for s in sessions}).to(DEVICE).eval()
    mini = {'test': {target: DL['test'][target]}}
    return encode_eval_records(adapter, mini, DEVICE)      # session, bat, unit, corr, fde


@torch.no_grad()
def decode_eval(model, target):
    bm = np.asarray(META['behavior_mean'][target]).reshape(-1)[:3]
    bs = np.asarray(META['behavior_std'][target]).reshape(-1)[:3]
    bat = parse_bat(target)
    recs = []
    for b in DL['test'][target]:
        m = b.mask.to(DEVICE)
        bt = b.mask.clone().to(DEVICE); nt = torch.zeros_like(bt)   # decode: all behaviour masked
        _, beh_pred, _ = model(target, b.spike.to(DEVICE), b.behavior.to(DEVICE), m, nt, bt)
        yp = beh_pred[:, :, :3].cpu().numpy() * bs + bm
        yt = b.behavior[:, :, :3].numpy() * bs + bm
        mn = b.mask.numpy().astype(bool); B, T = mn.shape
        fpi = np.broadcast_to(b.flight_path_index.numpy()[:, None], (B, T))
        tix = np.broadcast_to(np.arange(T)[None, :], (B, T))
        ytf, ypf = yt[mn], yp[mn]
        for i in range(len(ytf)):
            recs.append(dict(flight_id=int(fpi[mn][i]), t_idx=int(tix[mn][i]),
                             x_true=ytf[i, 0], y_true=ytf[i, 1], z_true=ytf[i, 2],
                             x_pred=ypf[i, 0], y_pred=ypf[i, 1], z_pred=ypf[i, 2],
                             err_3d=float(np.linalg.norm(ytf[i] - ypf[i]))))
    df = pd.DataFrame(recs)
    rows = []
    for fid, g in df.groupby('flight_id'):
        g = g.sort_values('t_idx')
        rows.append(dict(session=target, bat=bat, flight_id=int(fid),
                         frechet_3d=discrete_frechet(g[['x_true', 'y_true', 'z_true']].to_numpy(float),
                                                     g[['x_pred', 'y_pred', 'z_pred']].to_numpy(float)),
                         err_3d=float(g['err_3d'].median())))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def foundation_path(group, heldout):
    return f'{CKPT_ROOT}/{group}/foundation_{heldout}.pth'


def scratch_path(target):
    return f'{CKPT_ROOT}/scratch/scratch_{target}.pth'          # shared across CV schemes


def adapt_path(group, heldout, target, mode):
    return f'{CKPT_ROOT}/{group}/{mode}_{heldout}__{target}.pth'


# ---------------------------------------------------------------------------
# Fold definitions + driver
# ---------------------------------------------------------------------------
def folds_for(group):
    """Return (heldout, targets, sources) triples. `sources` is what the fold's
    foundation is trained on; `targets` are the held-out sessions adapted to it."""
    if group == 'session':
        return [(s, [s], [x for x in SESSIONS if x != s]) for s in SESSIONS]
    if group == 'bat':
        out = []
        for bat in BATS:
            tgt = [s for s in SESSIONS if parse_bat(s) == bat]
            out.append((bat, tgt, [s for s in SESSIONS if s not in tgt]))
        return out
    if group == 'env':
        # Leave-one-environment-out: foundation trained purely on the *other*
        # environment, then adapted to each held-out session. indoor = datasets 1 & 3
        # ('lab'); outdoor = dataset 2 ('dataset_2'). Cross-environment transfer.
        indoor = [s for s in SESSIONS if ROOM[s] == 'lab']
        outdoor = [s for s in SESSIONS if ROOM[s] == 'dataset_2']
        return [('outdoor', outdoor, indoor), ('indoor', indoor, outdoor)]
    raise ValueError(group)


def eval_one(mode, path, target, sessions, group, heldout):
    model = _load_for_eval(path, sessions)
    enc = encode_eval(model, target, sessions).assign(mode=mode, group=group, heldout=heldout)
    dec = decode_eval(model, target).assign(mode=mode, group=group, heldout=heldout)
    del model
    torch.cuda.empty_cache()
    return enc, dec


def run_group(group, folds, save=True):
    enc_parts, dec_parts = [], []
    for fi, (heldout, targets, sources) in enumerate(folds):
        fpath = foundation_path(group, heldout)
        print(f'[{group}] fold {fi + 1}/{len(folds)} heldout={heldout} '
              f'({len(sources)} src, {len(targets)} tgt)', flush=True)
        train_full(sources, fpath, n_epochs=FOUND_EPOCHS, patience=FOUND_PATIENCE, tag='foundation')
        for t in targets:
            train_full([t], scratch_path(t), n_epochs=SCRATCH_EPOCHS,
                       patience=SCRATCH_PATIENCE, tag='scratch')
            frz, fnt = adapt_path(group, heldout, t, 'frozen'), adapt_path(group, heldout, t, 'finetune')
            train_adapt(t, sources, fpath, 'frozen', frz)
            train_adapt(t, sources, fpath, 'finetune', fnt)
            for mode, path, sess in (('scratch', scratch_path(t), [t]),
                                     ('frozen', frz, sources + [t]),
                                     ('fine-tune', fnt, sources + [t])):
                enc, dec = eval_one(mode, path, t, sess, group, heldout)
                enc_parts.append(enc); dec_parts.append(dec)
    enc_df = pd.concat(enc_parts, ignore_index=True)
    dec_df = pd.concat(dec_parts, ignore_index=True)
    if save:
        out = os.path.join(PKL_ROOT, group)
        os.makedirs(out, exist_ok=True)
        enc_df.to_pickle(os.path.join(out, f'results_{group}_encode.pkl'))
        dec_df.to_pickle(os.path.join(out, f'results_{group}_decode.pkl'))
        print(f'[{group}] saved {len(enc_df)} encode rows, {len(dec_df)} decode rows', flush=True)
    print(enc_df.groupby('mode')[['fde', 'corr']].median(), flush=True)
    print(dec_df.groupby('mode')[['frechet_3d', 'err_3d']].median(), flush=True)
    return enc_df, dec_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['dry', 'session', 'bat', 'env', 'both', 'all'], default='both')
    ap.add_argument('--fold', default=None, help='session name for --mode dry')
    args = ap.parse_args()

    if args.mode == 'dry':
        target = args.fold or SESSIONS[7]
        print(f'DRY RUN: single session fold, heldout={target}', flush=True)
        run_group('session', [(target, [target], [x for x in SESSIONS if x != target])], save=False)
        return
    groups = {'both': ['session', 'bat'], 'all': ['session', 'bat', 'env']}.get(args.mode, [args.mode])
    for g in groups:
        run_group(g, folds_for(g), save=True)


if __name__ == '__main__':
    main()
