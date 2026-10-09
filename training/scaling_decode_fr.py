"""Decode-arm Fréchet distance for the session-universe scaling Twins (ED Fig 3d).

The scaling checkpoints (from tmp/train_joint_scaling.py) were trained with the
production encode+decode+cross recipe, so they have a functional decode arm. Here we
score each subset model on the decode task (behaviour masked, spikes seen) -> per-flight
3-D Fréchet distance, mirroring tmp/train_joint_context.py's `decode_eval`. Subsets are
reproduced deterministically (same seed/order as run_joint_scaling), so each checkpoint
is matched to the exact sessions it was trained on.

Reuses the already-loaded dataloaders / build_joint / subset logic from
tmp/train_joint_scaling.py (imported as a module; its main() is guarded).

Output: notebooks/out/transfer/scaling_joint_session/results_scaling_joint_session_decode_fr.pkl
        (per-flight rows: scaling_group, n_sessions, sample_id, session, bat, flight_id, frechet_3d)
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import train_joint_scaling as S          # loads joint dataloaders, build_joint, subset logic

from digitalbat.utility.utils import parse_bat
from digitalbat.decoding.geometry import discrete_frechet

DEVICE = S.DEVICE
DL = S.joint_dataloaders
META = S.joint_meta


@torch.no_grad()
def decode_eval(model, sessions):
    """Per-flight 3-D Fréchet on the test split of every session in `sessions`."""
    model.eval()
    fr_rows = []
    for sn in sessions:
        bm = np.asarray(META['behavior_mean'][sn]).reshape(-1)[:3]
        bs = np.asarray(META['behavior_std'][sn]).reshape(-1)[:3]
        bat = parse_bat(sn); recs = []
        for b in DL['test'][sn]:
            m = b.mask.to(DEVICE)
            bt = b.mask.clone().to(DEVICE); nt = torch.zeros_like(bt)   # decode: behaviour masked
            _, beh_pred, _ = model(sn, b.spike.to(DEVICE), b.behavior.to(DEVICE), m, nt, bt)
            yp = beh_pred[:, :, :3].cpu().numpy() * bs + bm
            yt = b.behavior[:, :, :3].numpy() * bs + bm
            mn = b.mask.numpy().astype(bool); B, T = mn.shape
            fpi = np.broadcast_to(b.flight_path_index.numpy()[:, None], (B, T))
            tix = np.broadcast_to(np.arange(T)[None, :], (B, T))
            ytf, ypf = yt[mn], yp[mn]
            for i in range(len(ytf)):
                recs.append(dict(flight_id=int(fpi[mn][i]), t_idx=int(tix[mn][i]),
                                 x_true=ytf[i, 0], y_true=ytf[i, 1], z_true=ytf[i, 2],
                                 x_pred=ypf[i, 0], y_pred=ypf[i, 1], z_pred=ypf[i, 2]))
        df = pd.DataFrame(recs)
        for fid, g in df.groupby('flight_id'):
            g = g.sort_values('t_idx')
            fr_rows.append(dict(session=sn, bat=bat, flight_id=int(fid),
                                frechet_3d=discrete_frechet(
                                    g[['x_true', 'y_true', 'z_true']].to_numpy(float),
                                    g[['x_pred', 'y_pred', 'z_pred']].to_numpy(float))))
    return fr_rows


def eval_ckpt(path, sessions):
    unit_sub = {s: S.joint_unit_num_dict[s] for s in sessions}
    model = S.build_joint(unit_sub).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    rows = decode_eval(model, sessions)
    del model; torch.cuda.empty_cache()
    return rows


def main():
    universe = list(S.JOINT_SESSIONS); N_full = len(universe)
    group = 'session'
    rows = []

    # N = 1 (single-session Twins)
    print(f'[decode-fr] N=1 ({len(universe)} sessions)', flush=True)
    for t in universe:
        rows += [{**r, 'scaling_group': group, 'n_sessions': 1, 'sample_id': 0}
                 for r in eval_ckpt(S.n1_ckpt(t), [t])]

    # N in {2,4,8}: reproduce the exact stratified subsets (same rng/order as training)
    rng = np.random.default_rng(S.SEED)
    for N in (2, 4, 8):
        subs = S.stratified_subsamples(universe, N, 10, rng)
        print(f'[decode-fr] N={N} ({len(subs)} subsets)', flush=True)
        for sid, sub in enumerate(subs):
            rows += [{**r, 'scaling_group': group, 'n_sessions': N, 'sample_id': sid}
                     for r in eval_ckpt(S.joint_scaling_ckpt(group, N, sid), sub)]

    # N = N_full (production checkpoint)
    print(f'[decode-fr] N_full={N_full} (production ckpt)', flush=True)
    rows += [{**r, 'scaling_group': group, 'n_sessions': N_full, 'sample_id': 0}
             for r in eval_ckpt(S.JOINT_PROD_CKPT, universe)]

    out = os.path.join(S.joint_scaling_dir(group),
                       'results_scaling_joint_session_decode_fr.pkl')
    pd.DataFrame(rows).to_pickle(out)
    print(f'[decode-fr] wrote {len(rows)} flight-rows -> {out}', flush=True)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
