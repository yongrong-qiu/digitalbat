"""ED Fig 5 context-window sweep with the JOINT model (replaces the old
Bat_Transformer / Bat_Transformer_Decoder). For each window size W, load tiled
(window_size=W) joint dataloaders, train one joint model single-stage (6 schemes,
fixed HP, no Optuna), and score encode (per-neuron FDE/corr) + decode (per-flight
Fréchet, per-point recon) on the test split. The 'full' context point (W=None) is
NOT trained here — the notebook appends it from the main joint model's already-computed
encode_metrics_df / joint_frechet / joint_decode_test.

  .pth  -> /media/alowet/ssd2/twin/transfer_joint/context/joint_ctx{W}.pth
  pkls  -> notebooks/out/transfer_joint/context/{ctx_encode,ctx_decode_fr,ctx_decode_rc}.pkl

Usage:  python tmp/train_joint_context.py [--windows 1,3,5,7,9,11]
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(os.path.join(REPO, 'notebooks'))

from digitalbat.datasets.defaults import DEFAULT_SESSION_NAMES, make_data_config
from digitalbat.datasets.kinematics import make_joint_dataloaders
from digitalbat.models.joint import (JointMaskedTransformer, JointEncoderAdapter, encode_eval_records,
                                      SCHEMES, sample_mask, within_double_batch,
                                      forecast_input_mask, joint_loss)
from digitalbat.training.joint_trainer import (scheme_weights, _decode_median_err_m,
                                               _encode_median_fde, _pareto_update)
from digitalbat.utility.utils import round_robin_session_batches, parse_bat, set_random_state
from digitalbat.decoding.geometry import discrete_frechet
import random

# Fixed config — identical to tmp/train_joint_transfer.py (consensus joint model).
SEED = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_TRANSFORM = 'identity'
JOINT_PARAMS = dict(num_layers=4, latent_dim=64, num_heads=4, ff_dim=64, dropout=0.2)
LAM, POS_W = 0.61, 0.84
SCHEME_P = scheme_weights(dict(w_encode=1.9, w_decode=2.8, w_aux=0.67))
LR, WD = 3e-3, 1e-2
FDE_WEIGHT, ERR_REF, FDE_REF = 0.5, 0.3, 0.1
EPOCHS = int(os.environ.get('TJC_EPOCHS', 500))
PATIENCE = int(os.environ.get('TJC_PATIENCE', 40))

CKPT_ROOT = '/media/alowet/ssd2/twin/transfer_joint/context'
PKL_ROOT = os.path.join(REPO, 'notebooks', 'out', 'transfer_joint', 'context')
CTX_WINDOWS = [1, 3, 5, 7, 9, 11]


def joint_session_room(s):
    if '_fs' in s:
        return 'krishna_fs'
    if s.startswith('Dataset_2_'):
        return 'dataset_2'
    return 'lab'


def build_model(unit_sub, room_sub):
    set_random_state(SEED)
    return JointMaskedTransformer(unit_sub, behavior_dim=3, input_transform=INPUT_TRANSFORM,
                                  session_to_room=room_sub, **JOINT_PARAMS).to(DEVICE)


@torch.no_grad()
def decode_eval(model, loaders, meta):
    """Per-flight 3-D Fréchet + per-point recon error on every session's test split."""
    model.eval()
    fr_rows, rc_rows = [], []
    for sn, sd in loaders['test'].items():
        bm = np.asarray(meta['behavior_mean'][sn]).reshape(-1)[:3]
        bs = np.asarray(meta['behavior_std'][sn]).reshape(-1)[:3]
        bat = parse_bat(sn); recs = []
        for b in sd:
            m = b.mask.to(DEVICE)
            bt = b.mask.clone().to(DEVICE); nt = torch.zeros_like(bt)
            _, beh_pred, _ = model(sn, b.spike.to(DEVICE), b.behavior.to(DEVICE), m, nt, bt)
            yp = beh_pred[:, :, :3].cpu().numpy() * bs + bm
            yt = b.behavior[:, :, :3].numpy() * bs + bm
            mn = b.mask.numpy().astype(bool); B, T = mn.shape
            fpi = np.broadcast_to(b.flight_path_index.numpy()[:, None], (B, T))
            tix = np.broadcast_to(np.arange(T)[None, :], (B, T))
            ytf, ypf = yt[mn], yp[mn]
            for i in range(len(ytf)):
                rec = dict(session=sn, bat=bat, flight_id=int(fpi[mn][i]), t_idx=int(tix[mn][i]),
                           x_true=ytf[i, 0], y_true=ytf[i, 1], z_true=ytf[i, 2],
                           x_pred=ypf[i, 0], y_pred=ypf[i, 1], z_pred=ypf[i, 2],
                           err_3d=float(np.linalg.norm(ytf[i] - ypf[i])))
                recs.append(rec)
                rc_rows.append(dict(session=sn, bat=bat, flight_id=rec['flight_id'], err_3d=rec['err_3d']))
        df = pd.DataFrame(recs)
        for fid, g in df.groupby('flight_id'):
            g = g.sort_values('t_idx')
            fr_rows.append(dict(session=sn, bat=bat, flight_id=int(fid),
                                frechet_3d=discrete_frechet(g[['x_true', 'y_true', 'z_true']].to_numpy(float),
                                                            g[['x_pred', 'y_pred', 'z_pred']].to_numpy(float))))
    return pd.DataFrame(fr_rows), pd.DataFrame(rc_rows)


def _train_joint_guarded(model, loaders, meta, tag=''):
    """train_joint's loop + Pareto selection, but skips degenerate (grad-less) batches.
    Small windows (W=1) make forecast/within/cross produce empty targets -> joint_loss
    returns a grad-less zero whose backward() errors; skip those (correctness-preserving)."""
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    pois_fn = nn.PoissonNLLLoss(log_input=False, reduction='none').to(DEVICE)
    rng = random.Random(SEED)
    cand, no_improve = [], 0
    for ep in range(EPOCHS):
        model.train()
        for s, b in round_robin_session_batches(loaders['train']):
            scheme = rng.choices(SCHEMES, weights=SCHEME_P)[0]
            spike, behavior, m = b.spike.to(DEVICE), b.behavior.to(DEVICE), b.mask.to(DEVICE)
            if scheme == 'within':
                spike = torch.cat([spike, spike], 0); behavior = torch.cat([behavior, behavior], 0)
                m = torch.cat([m, m], 0); nt, bt = within_double_batch(b.mask); n_in, b_in = nt, bt
            elif scheme in ('forecast', 'forecast_stim'):
                nt, bt = sample_mask(b.mask, scheme); n_in, b_in = nt, forecast_input_mask(b.mask, scheme)
            else:
                nt, bt = sample_mask(b.mask, scheme); n_in, b_in = nt, bt
            n_in, b_in, nt, bt = (t.to(DEVICE) for t in (n_in, b_in, nt, bt))
            rate, beh, _ = model(s, spike, behavior, m, n_in, b_in)
            loss = joint_loss(rate, beh, spike, behavior, m, nt, bt, LAM, pois_fn, pos_weight=POS_W)
            if not loss.requires_grad:                 # empty targets this batch -> nothing to learn
                continue
            opt.zero_grad(); loss.backward(); opt.step()
        err = _decode_median_err_m(model, loaders['valid'], meta, DEVICE)
        fde = _encode_median_fde(model, loaders['valid'], DEVICE)
        cand, improved = _pareto_update(cand, err, fde, model)
        no_improve = 0 if improved else no_improve + 1
        if ep % 20 == 0 or no_improve >= PATIENCE:
            print(f'    {tag} ep{ep}: err={err:.3f} fde={fde:.3f} (no_improve={no_improve})', flush=True)
        if PATIENCE and no_improve >= PATIENCE:
            break
    errs = np.array([o['err'] for o in cand], float); fdes = np.array([o['fde'] for o in cand], float)
    joint = FDE_WEIGHT * (fdes / FDE_REF) - (1.0 - FDE_WEIGHT) * (errs / ERR_REF)
    bi = int(np.nanargmax(joint))
    model.load_state_dict(cand[bi]['w'])
    return model, dict(err=float(errs[bi]), fde=float(fdes[bi]))


def train_or_load(W, loaders, unit, meta, sessions, room):
    path = f'{CKPT_ROOT}/joint_ctx{W}.pth'
    model = build_model({s: unit[s] for s in sessions}, {s: room[s] for s in sessions})
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        return model.eval()
    model, best = _train_joint_guarded(model, loaders, meta, tag=f'ctx{W}')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f'  ctx{W}: trained  err={best["err"]:.3f} fde={best["fde"]:.3f}', flush=True)
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', default=','.join(map(str, CTX_WINDOWS)))
    args = ap.parse_args()
    windows = [int(w) for w in args.windows.split(',')]

    enc_parts, fr_parts, rc_parts = [], [], []
    for W in windows:
        print(f'[ctx] window_size={W}: loading tiled joint dataloaders ...', flush=True)
        dc = make_data_config(session_names=list(DEFAULT_SESSION_NAMES))
        dc.update(window_size=W, stride_train=W, stride_test=W)
        dl, _dk, unit, meta = make_joint_dataloaders(dc)
        sessions = list(unit)
        room = {s: joint_session_room(s) for s in sessions}
        model = train_or_load(W, dl, unit, meta, sessions, room)

        adapter = JointEncoderAdapter(model, unit).to(DEVICE).eval()
        enc = encode_eval_records(adapter, dl, DEVICE).assign(context=W, context_x=W)
        fr, rc = decode_eval(model, dl, meta)
        enc_parts.append(enc)
        fr_parts.append(fr.assign(context_x=W))
        rc_parts.append(rc.assign(context_x=W))
        print(f'  ctx{W}: encode median fde={enc["fde"].median():.3f} corr={enc["corr"].median():.3f} '
              f'| decode median frechet={fr["frechet_3d"].median():.3f} recon={rc["err_3d"].median():.3f}',
              flush=True)
        del model, adapter
        torch.cuda.empty_cache()

    os.makedirs(PKL_ROOT, exist_ok=True)
    pd.concat(enc_parts, ignore_index=True).to_pickle(f'{PKL_ROOT}/ctx_encode.pkl')
    pd.concat(fr_parts, ignore_index=True).to_pickle(f'{PKL_ROOT}/ctx_decode_fr.pkl')
    pd.concat(rc_parts, ignore_index=True).to_pickle(f'{PKL_ROOT}/ctx_decode_rc.pkl')
    print(f'[ctx] saved encode/decode pkls to {PKL_ROOT}', flush=True)


if __name__ == '__main__':
    main()
