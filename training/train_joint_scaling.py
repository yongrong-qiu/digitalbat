"""Background driver for the joint-Twin scaling analysis in foundation_transfer.ipynb.

Mirrors the notebook's `joint_scaling_setup` + `joint_scaling_run` cells (same
package calls, same production encode+decode+cross recipe). Train-if-missing and
resumable: per-subset checkpoints under out/transfer/scaling_joint_*, and the
long-form per-neuron results pickles the notebook's plot cell reads.

Run from the repo root or anywhere; it chdirs into notebooks/ so ./out resolves
like the notebook. Heavy: ~78 Twin foundations at up to 1000 epochs.
"""
import os
import sys
import pickle as _pkl

import numpy as np
import pandas as pd
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(os.path.join(REPO, 'notebooks'))

sys.path.insert(0, os.path.join(REPO, 'notebooks'))
import machine_paths as _mp                      # notebooks/ is cwd after chdir above
_P = _mp.setup(verbose=False)
from digitalbat.datasets.defaults import DEFAULT_SESSION_NAMES, make_data_config
from digitalbat.datasets.kinematics import make_joint_dataloaders
from digitalbat.models.joint import JointMaskedTransformer, JointEncoderAdapter
from digitalbat.training.joint_trainer import train_joint
from digitalbat.utility.utils import set_random_state, poisson_fde_population, corr

SEED = 1
DEVICE = 'cuda'
OUT_ROOT = os.path.abspath('./out/transfer')

JOINT_SESSIONS = list(DEFAULT_SESSION_NAMES)


def _dataset_of(s): return 'D' + s.split('_')[1]
JOINT_SESSIONS_D13 = [s for s in JOINT_SESSIONS if _dataset_of(s) != 'D2']


def session_room(s):
    if '_fs' in s:                 return 'krishna_fs'
    if s.startswith('Dataset_2_'): return 'dataset_2'
    return 'lab'
SESSION_TO_ROOM = {s: session_room(s) for s in JOINT_SESSIONS}

_jdc = make_data_config(session_names=JOINT_SESSIONS, parent_folder=_P.data_root)
joint_dataloaders, _kin, joint_unit_num_dict, joint_meta = make_joint_dataloaders(_jdc)
JOINT_BEHAVIOR_DIM = 3
print(f'joint Twin scaling: {len(joint_unit_num_dict)} sessions, '
      f'{sum(joint_unit_num_dict.values())} neurons; D1+D3 = {len(JOINT_SESSIONS_D13)}', flush=True)

CROSS_SCHEMES = ['encode', 'decode', 'cross']
CROSS_PARAMS = max(
    (t for t in _pkl.load(open('./out/joint/joint_optuna_cross_1000.pkl', 'rb')).trials
     if t.state.name == 'COMPLETE'), key=lambda t: t.value).params
JOINT_ARCH = dict(latent_dim=64, num_heads=4, ff_dim=64, num_layers=4,
                  input_transform='identity', dropout=CROSS_PARAMS['dropout'])
JOINT_ERR_REF, JOINT_FDE_REF = 0.3, 0.1
JOINT_SCALING_EPOCHS = 1000
JOINT_PROD_CKPT = ('./out/joint/joint_v2_cross1000_kin0_L4_d64_ff64'
                   '_lam0.68_pw0.36_identity_we2.8_wd2.2_wa0.31_n17.pth')
SCALING_JOINT_ROOT = OUT_ROOT


def build_joint(unit_sub, seed=SEED):
    set_random_state(seed)
    return JointMaskedTransformer(
        dict(unit_sub), behavior_dim=JOINT_BEHAVIOR_DIM,
        session_to_room={s: SESSION_TO_ROOM[s] for s in unit_sub}, **JOINT_ARCH)


def _filter_joint_dl(sessions):
    keep = list(sessions)
    return {sp: {s: joint_dataloaders[sp][s] for s in keep}
            for sp in ('train', 'valid', 'test')}


def train_joint_subset(sessions, save_path, seed=SEED, n_epochs=JOINT_SCALING_EPOCHS):
    if os.path.exists(save_path):
        print(f'  [skip] {os.path.basename(save_path)}', flush=True)
        return save_path
    unit_sub = {s: joint_unit_num_dict[s] for s in sessions}
    model = build_joint(unit_sub, seed=seed).to(DEVICE)
    model, best, _ = train_joint(
        model, _filter_joint_dl(sessions), DEVICE, joint_meta, seed=seed,
        n_epochs=n_epochs, lr=CROSS_PARAMS['lr'], weight_decay=CROSS_PARAMS['weight_decay'],
        lam=CROSS_PARAMS['lam'], pos_weight=CROSS_PARAMS['pos_weight'],
        schemes=CROSS_SCHEMES,
        scheme_p=[CROSS_PARAMS['w_encode'], CROSS_PARAMS['w_decode'], CROSS_PARAMS['w_aux']],
        select='joint', fde_weight=0.5, err_ref=JOINT_ERR_REF, fde_ref=JOINT_FDE_REF,
        early_stopping_patience=40, verbose=False)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f'  [joint] {os.path.basename(save_path)}  '
          f'val_fde={best["fde"]:.4f} err={best["err"]:.3f}m', flush=True)
    del model; torch.cuda.empty_cache()
    return save_path


@torch.no_grad()
def eval_joint_encode(state_path, sessions, eval_sessions):
    unit_sub = {s: joint_unit_num_dict[s] for s in sessions}
    model = build_joint(unit_sub).to(DEVICE)
    model.load_state_dict(torch.load(state_path, map_location=DEVICE))
    adapter = JointEncoderAdapter(model, unit_sub).to(DEVICE).eval()
    rows = []
    for t in eval_sessions:
        yt, yp = [], []
        for b in joint_dataloaders['test'][t]:
            X = b.behavior.to(DEVICE); mask = b.mask.to(DEVICE)
            pred = adapter(t, X, mask)
            yt.append(b.spike.to(DEVICE)[mask == 1.0].cpu().numpy())
            yp.append(pred[mask == 1.0].detach().cpu().numpy())
        yt = np.vstack(yt); yp = np.vstack(yp)
        fde = poisson_fde_population(yt, yp); pc = corr(yt, yp, axis=0)
        rows += [{'target_session': t, 'neuron_idx': i, 'fde': float(f), 'corr': float(c)}
                 for i, (f, c) in enumerate(zip(fde, pc))]
    del model, adapter; torch.cuda.empty_cache()
    return rows


def stratified_subsamples(universe, N, n_samples, rng):
    universe = list(universe); N_full = len(universe)
    assert N <= N_full
    anchors = list(universe); rng.shuffle(anchors)
    samples = []
    for i in range(n_samples):
        anchor = anchors[i % N_full]
        rest = list(rng.choice([s for s in universe if s != anchor], size=N - 1, replace=False))
        samples.append(sorted([anchor] + rest))
    missing = sorted(set(universe) - set().union(*samples))
    for m in missing:
        best = None
        for si, sub in enumerate(samples):
            if m in sub: continue
            for ji, s in enumerate(sub):
                others = sum(1 for oi, o in enumerate(samples) if oi != si and s in o)
                if others >= 1 and (best is None or others > best[2]):
                    best = (si, ji, others)
        assert best is not None, f'cannot insert {m} without dropping coverage'
        si, ji, _ = best
        samples[si][ji] = m; samples[si] = sorted(samples[si])
    assert set().union(*samples) == set(universe)
    return samples


def joint_scaling_dir(group):
    d = os.path.join(SCALING_JOINT_ROOT, f'scaling_joint_{group}'); os.makedirs(d, exist_ok=True)
    return d


def joint_scaling_ckpt(group, N, sid):
    return os.path.join(joint_scaling_dir(group), f'joint_scaling_{group}_N{N}_s{sid}.pth')


def n1_ckpt(session):
    d = os.path.join(SCALING_JOINT_ROOT, 'scaling_joint_N1'); os.makedirs(d, exist_ok=True)
    return os.path.join(d, f'joint_scaling_N1_{session}.pth')


def run_joint_scaling(group, universe, sizes=(2, 4, 8), n_samples=10, seed=SEED):
    universe = list(universe); N_full = len(universe)
    rng = np.random.default_rng(seed)
    rows = []
    print(f'[{group}] N=1 single-session Twins ({len(universe)})', flush=True)
    for t in universe:
        p = n1_ckpt(t); train_joint_subset([t], p, seed=seed)
        rows += [{**r, 'scaling_group': group, 'n_sessions': 1, 'sample_id': 0}
                 for r in eval_joint_encode(p, [t], [t])]
    for N in sizes:
        print(f'[{group}] N={N} stratified subsets (x{n_samples})', flush=True)
        for sid, sub in enumerate(stratified_subsamples(universe, N, n_samples, rng)):
            p = joint_scaling_ckpt(group, N, sid); train_joint_subset(sub, p, seed=seed)
            rows += [{**r, 'scaling_group': group, 'n_sessions': N, 'sample_id': sid}
                     for r in eval_joint_encode(p, sub, sub)]
    print(f'[{group}] N_full={N_full}', flush=True)
    if set(universe) == set(JOINT_SESSIONS):
        full_path = JOINT_PROD_CKPT
    else:
        full_path = joint_scaling_ckpt(group, N_full, 0); train_joint_subset(universe, full_path, seed=seed)
    rows += [{**r, 'scaling_group': group, 'n_sessions': N_full, 'sample_id': 0}
             for r in eval_joint_encode(full_path, universe, universe)]
    return pd.DataFrame(rows)


if __name__ == '__main__':
    for g, univ in [('session', JOINT_SESSIONS), ('session_d13', JOINT_SESSIONS_D13)]:
        df = run_joint_scaling(g, univ)
        out = os.path.join(joint_scaling_dir(g), f'results_scaling_joint_{g}.pkl')
        df.to_pickle(out)
        print(f'[{g}] wrote {len(df)} neuron-rows -> {out}', flush=True)
    print('ALL DONE', flush=True)
