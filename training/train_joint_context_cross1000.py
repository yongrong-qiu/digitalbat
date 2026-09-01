"""ED Fig 3a-c context-window sweep, retrained under the PRODUCTION recipe.

Supersedes `train_joint_context.py`, which trained each windowed joint model with
hard-coded (untuned) hyperparameters at a 500-epoch cap over all six masking schemes,
while the sweep's reference point (`context = full`) was the production
`joint_v2_cross1000_..._n17.pth` — Optuna-tuned and trained at 1000 epochs. The
full-context advantage in ED Fig 3a-c therefore bundled context length with HP tuning
and a 2x epoch budget. Here every point on the x-axis gets the same treatment as
production:

  * masking schemes  : encode / decode / cross only (the `cross_1000` set)
  * architecture     : L4 / d64 / ff64 / 4 heads, identity read-in (fixed, as in §6.10)
  * hyperparameters  : per-window Optuna search (TPE, `N_TRIALS` trials at
                       `TRIAL_EPOCHS` epochs, first trial enqueued at the production
                       params), objective = validation joint score
                       0.5*(FDE/FDE_REF) - 0.5*(err/ERR_REF)

`lam` — the behaviour-MSE vs spike-NLL weight — is **pinned** at the production value
rather than searched (production's §6.10 study did search it). Searching it per window
lets each x-point pick its own encode/decode balance, so the sweep's three metric rows
would drift along that frontier for reasons unrelated to context length; pinning it
keeps one loss balance across the whole x-axis. `pos_weight` is pinned too, but that is
cosmetic: it splits behaviour MSE into position vs auxiliary kinematic channels, and at
`behavior_dim = 3` (the production `kin0` config) there are no auxiliary channels, so the
term never fires (see `joint_loss`). Pinning it just frees a dead TPE dimension.
  * final model      : refit at `FINAL_EPOCHS` = 1000 epochs, early stopping
                       patience 40, Pareto-front checkpoint selection on validation

`window = full` (window_size=None, i.e. whole flights) is trained here too, so the
reference point is protocol-matched rather than inherited from the production run.
Its metrics are tagged `context_x = 13` / `context = 'full'`, the same convention the
notebook used when it appended the production model's numbers.

Outputs are tagged by `VARIANT`, so nothing from the old run — or from the earlier
`VARIANT='searchlam'` pass, kept as a robustness point — is overwritten:
  ckpt        -> /media/alowet/ssd2/twin/transfer_joint/context_cross1000_{VARIANT}/joint_ctx{W}.pth
  study       -> notebooks/out/transfer_joint/context/optuna/ctx{W}_cross1000_{VARIANT}.pkl
  per-window  -> notebooks/out/transfer_joint/context/per_window/ctx{W}_{VARIANT}_{encode,fr,rc}.pkl
  combined    -> notebooks/out/transfer_joint/context/ctx_{encode,decode_fr,decode_rc}_cross1000_{VARIANT}.pkl
  best params -> notebooks/out/transfer_joint/context/ctx_best_params_cross1000_{VARIANT}.json

Resumable at every level: completed Optuna trials are reloaded from the pickled study,
a finished window is skipped if its per-window pkls exist, and a finished final model is
reloaded from its checkpoint.

Usage:  python training/train_joint_context_cross1000.py
        python training/train_joint_context_cross1000.py --windows full,11,9,7,5,3,1
        python training/train_joint_context_cross1000.py --trials 50 --trial-epochs 300
"""
import os
import sys
import json
import time
import random
import argparse
import pickle as _pkl

import numpy as np
import pandas as pd
import torch
from torch import nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(os.path.join(REPO, 'notebooks'))

import optuna

sys.path.insert(0, os.path.join(REPO, 'notebooks'))
import machine_paths as _mp                      # notebooks/ is cwd after chdir above
_P = _mp.setup(verbose=False)
from digitalbat.datasets.defaults import DEFAULT_SESSION_NAMES, make_data_config
from digitalbat.datasets.kinematics import make_joint_dataloaders
from digitalbat.models.joint import (JointMaskedTransformer, JointEncoderAdapter,
                                     encode_eval_records, sample_mask, joint_loss)
from digitalbat.training.joint_trainer import (_decode_median_err_m, _encode_median_fde,
                                               _pareto_update)
from digitalbat.utility.utils import round_robin_session_batches, parse_bat, set_random_state
from digitalbat.decoding.geometry import discrete_frechet

SEED = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_TRANSFORM = 'identity'
ARCH = dict(num_layers=4, latent_dim=64, num_heads=4, ff_dim=64)   # §6.10 LONG_ARCH
SCHEMES = ['encode', 'decode', 'cross']                            # production 'cross_1000' set
ERR_REF, FDE_REF, FDE_WEIGHT = 0.3, 0.1, 0.5                       # joint-encode-decode.ipynb §6
PATIENCE = 40

N_TRIALS = int(os.environ.get('TJC_TRIALS', 50))
TRIAL_EPOCHS = int(os.environ.get('TJC_TRIAL_EPOCHS', 300))        # epoch cap per Optuna trial
FINAL_EPOCHS = int(os.environ.get('TJC_FINAL_EPOCHS', 1000))       # final refit (= production)

# Held at the production values for every window; the rest of the §6.10 space is searched.
PINNED = ('lam', 'pos_weight')
VARIANT = os.environ.get('TJC_VARIANT', 'pinlam')

CKPT_ROOT = os.environ.get(
    'TJC_CKPT_ROOT', f'/media/alowet/ssd2/twin/transfer_joint/context_cross1000_{VARIANT}')
PKL_ROOT = os.environ.get('TJC_PKL_ROOT',
                          os.path.join(REPO, 'notebooks', 'out', 'transfer_joint', 'context'))
PROD_STUDY = './out/joint/joint_optuna_cross_1000.pkl'
PROD_CKPT = ('./out/joint/joint_v2_cross1000_kin0_L4_d64_ff64'
             '_lam0.68_pw0.36_identity_we2.8_wd2.2_wa0.31_n17.pth')
CTX_WINDOWS = [1, 3, 5, 7, 9, 11, 'full']      # ascending; 'full' = window_size None
FULL_CONTEXT_X = 13                            # x-position the notebook uses for 'full'


def prod_params():
    """Best trial of the production cross_1000 study — enqueued as trial 0 (§6.10 START)."""
    st = _pkl.load(open(PROD_STUDY, 'rb'))
    best = max((t for t in st.trials if t.state.name == 'COMPLETE'), key=lambda t: t.value)
    return {k: best.params[k] for k in ('dropout', 'lr', 'weight_decay', 'w_encode',
                                        'w_decode', 'w_aux', 'lam', 'pos_weight')}


def joint_session_room(s):
    if '_fs' in s:
        return 'krishna_fs'
    if s.startswith('Dataset_2_'):
        return 'dataset_2'
    return 'lab'


def wtag(W):
    return 'full' if W == 'full' else str(W)


def context_x(W):
    return FULL_CONTEXT_X if W == 'full' else W


def load_window(W):
    """Tiled joint dataloaders for window size W ('full' -> whole flights, window_size=None)."""
    dc = make_data_config(session_names=list(DEFAULT_SESSION_NAMES), parent_folder=_P.data_root)
    if W != 'full':
        dc.update(window_size=W, stride_train=W, stride_test=W)
    dl, _kin, unit, meta = make_joint_dataloaders(dc)
    return dl, unit, meta


def build_model(unit, params, seed):
    set_random_state(seed)
    return JointMaskedTransformer(unit, behavior_dim=3, input_transform=INPUT_TRANSFORM,
                                  session_to_room={s: joint_session_room(s) for s in unit},
                                  dropout=params['dropout'], **ARCH).to(DEVICE)


def train_one(model, loaders, meta, params, *, n_epochs, seed, tag='', verbose_every=0):
    """`joint_trainer.train_joint` restricted to encode/decode/cross, plus a guard that
    skips grad-less batches (a `cross` draw can mask nothing when a batch is tiny, which
    makes `joint_loss` return a constant zero whose backward() errors). Identical
    otherwise: AdamW, per-epoch validation, Pareto-front checkpointing, early stopping."""
    opt = torch.optim.AdamW(model.parameters(), lr=params['lr'],
                            weight_decay=params['weight_decay'])
    pois_fn = nn.PoissonNLLLoss(log_input=False, reduction='none').to(DEVICE)
    scheme_p = [params['w_encode'], params['w_decode'], params['w_aux']]   # SCHEMES order
    rng = random.Random(seed)
    cand, no_improve, stop_ep = [], 0, n_epochs
    for ep in range(n_epochs):
        model.train()
        for s, b in round_robin_session_batches(loaders['train']):
            scheme = rng.choices(SCHEMES, weights=scheme_p)[0]
            spike, behavior, m = b.spike.to(DEVICE), b.behavior.to(DEVICE), b.mask.to(DEVICE)
            nt, bt = sample_mask(b.mask, scheme)
            nt, bt = nt.to(DEVICE), bt.to(DEVICE)
            rate, beh, _ = model(s, spike, behavior, m, nt, bt)
            loss = joint_loss(rate, beh, spike, behavior, m, nt, bt, params['lam'],
                              pois_fn, pos_weight=params['pos_weight'])
            if not loss.requires_grad:                  # nothing masked this batch
                continue
            opt.zero_grad(); loss.backward(); opt.step()
        err = _decode_median_err_m(model, loaders['valid'], meta, DEVICE)
        fde = _encode_median_fde(model, loaders['valid'], DEVICE)
        cand, improved = _pareto_update(cand, err, fde, model)
        no_improve = 0 if improved else no_improve + 1
        if verbose_every and (ep % verbose_every == 0):
            print(f'    {tag} ep{ep}: err={err:.3f} fde={fde:.4f} (no_improve={no_improve})',
                  flush=True)
        if PATIENCE and no_improve >= PATIENCE:
            stop_ep = ep + 1
            break
    if not cand:
        return model, dict(err=np.inf, fde=np.nan, joint=-np.inf, epochs=stop_ep)
    errs = np.array([o['err'] for o in cand], float)
    fdes = np.array([o['fde'] for o in cand], float)
    joint = FDE_WEIGHT * (fdes / FDE_REF) - (1.0 - FDE_WEIGHT) * (errs / ERR_REF)
    bi = int(np.nanargmax(joint))
    model.load_state_dict(cand[bi]['w'])
    return model, dict(err=float(errs[bi]), fde=float(fdes[bi]), joint=float(joint[bi]),
                       epochs=stop_ep)


def search_window(W, loaders, unit, meta, n_trials):
    """Per-window Optuna search (TPE, §6.10 space minus `PINNED`). Returns the full params
    dict: the best trial's searched values merged with the pinned production values."""
    os.makedirs(os.path.join(PKL_ROOT, 'optuna'), exist_ok=True)
    study_path = os.path.join(PKL_ROOT, 'optuna', f'ctx{wtag(W)}_cross1000_{VARIANT}.pkl')

    pinned = {k: v for k, v in prod_params().items() if k in PINNED}

    def objective(trial):
        params = dict(
            pinned,
            dropout=trial.suggest_float('dropout', 0.0, 0.4),
            lr=trial.suggest_float('lr', 1e-4, 3e-2, log=True),
            weight_decay=trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True),
            w_encode=trial.suggest_float('w_encode', 1.0, 12.0, log=True),
            w_decode=trial.suggest_float('w_decode', 1.0, 12.0, log=True),
            w_aux=trial.suggest_float('w_aux', 0.1, 3.0, log=True))
        t0 = time.time()
        model = build_model(unit, params, seed=trial.number)       # §6.10: seed = trial index
        model, best = train_one(model, loaders, meta, params,
                                n_epochs=TRIAL_EPOCHS, seed=trial.number)
        trial.set_user_attr('err_m', best['err']); trial.set_user_attr('fde', best['fde'])
        trial.set_user_attr('epochs', best['epochs'])
        print(f'  [ctx{wtag(W)}] trial {trial.number}: joint={best["joint"]:.4f} '
              f'err={best["err"]:.3f} fde={best["fde"]:.4f} '
              f'({best["epochs"]} ep, {time.time() - t0:.0f}s)', flush=True)
        del model; torch.cuda.empty_cache()
        return best['joint'] if np.isfinite(best['joint']) else -1e9

    def save_cb(study, trial):
        with open(study_path, 'wb') as f:
            _pkl.dump(study, f)

    study = None
    if os.path.exists(study_path):
        with open(study_path, 'rb') as f:
            study = _pkl.load(f)
        print(f'  [ctx{wtag(W)}] resuming study with {len(study.trials)} trials', flush=True)
    if study is None:
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
        # start from production's searched values (the pinned ones are not trial params)
        study.enqueue_trial({k: v for k, v in prod_params().items() if k not in PINNED})
    # count only trials that actually ran: `enqueue_trial` leaves a WAITING trial in
    # `study.trials`, so counting all of them would silently run n_trials - 1 (as the
    # production §6.10 study did: 19 COMPLETE for LONG_TRIALS = 20).
    ran = [t for t in study.trials if t.state.name in ('COMPLETE', 'PRUNED', 'FAIL')]
    todo = n_trials - len(ran)
    if todo > 0:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=todo, callbacks=[save_cb])
        with open(study_path, 'wb') as f:
            _pkl.dump(study, f)
    sel = max((t for t in study.trials if t.state.name == 'COMPLETE'), key=lambda t: t.value)
    print(f'  [ctx{wtag(W)}] best trial #{sel.number}: valid joint={sel.value:.4f} '
          f'searched={ {k: round(v, 4) for k, v in sel.params.items()} } '
          f'pinned={ {k: round(v, 4) for k, v in pinned.items()} }', flush=True)
    return dict(pinned, **sel.params)


@torch.no_grad()
def decode_eval(model, loaders, meta):
    """Per-flight 3-D Fréchet + per-point recon error on every session's test split.

    A windowed loader tiles one flight into several sequences that all carry the *same*
    ``flight_path_index``, while ``t_idx`` only indexes position *within* a window. Ordering
    a flight's rows by ``t_idx`` alone therefore interleaves its tiles (all the t=0 bins,
    then all the t=1 bins, …) instead of concatenating them — which is what the previous
    context driver did, corrupting the per-flight path for every W > 1. Rows are ordered
    here by ``(seq_idx, t_idx)``, where ``seq_idx`` is the sequence's position in loader
    order: the test sampler is a ``SubsetSequentialSampler`` and consecutive tiles of a
    flight join continuously (verified: median tile→tile step 0.293 m vs 0.288 m within a
    tile, against 1.83 m under a shuffled tile order), so loader order is time order.

    Stitching the tiles back together also makes the metric comparable across window sizes:
    every W scores the same whole flights, rather than W-length fragments whose Fréchet
    distance shrinks mechanically with segment length.
    """
    model.eval()
    fr_rows, rc_parts = [], []
    for sn, sd in loaders['test'].items():
        bm = np.asarray(meta['behavior_mean'][sn]).reshape(-1)[:3]
        bs = np.asarray(meta['behavior_std'][sn]).reshape(-1)[:3]
        bat = parse_bat(sn)
        parts, seq0 = [], 0
        for b in sd:
            m = b.mask.to(DEVICE)
            bt = b.mask.clone().to(DEVICE); nt = torch.zeros_like(bt)   # decode: behaviour hidden
            _, beh_pred, _ = model(sn, b.spike.to(DEVICE), b.behavior.to(DEVICE), m, nt, bt)
            yp = beh_pred[:, :, :3].cpu().numpy() * bs + bm
            yt = b.behavior[:, :, :3].numpy() * bs + bm
            mn = b.mask.numpy().astype(bool); B, T = mn.shape
            seq = np.broadcast_to((seq0 + np.arange(B))[:, None], (B, T))
            fpi = np.broadcast_to(b.flight_path_index.numpy()[:, None], (B, T))
            tix = np.broadcast_to(np.arange(T)[None, :], (B, T))
            ytf, ypf = yt[mn], yp[mn]
            parts.append(pd.DataFrame(dict(
                flight_id=fpi[mn].astype(int), seq_idx=seq[mn], t_idx=tix[mn],
                x_true=ytf[:, 0], y_true=ytf[:, 1], z_true=ytf[:, 2],
                x_pred=ypf[:, 0], y_pred=ypf[:, 1], z_pred=ypf[:, 2],
                err_3d=np.linalg.norm(ytf - ypf, axis=1))))
            seq0 += B
        df = (pd.concat(parts, ignore_index=True)
                .sort_values(['flight_id', 'seq_idx', 't_idx'], kind='stable'))
        rc_parts.append(df[['flight_id', 'err_3d']].assign(session=sn, bat=bat))
        for fid, g in df.groupby('flight_id', sort=False):
            fr_rows.append(dict(session=sn, bat=bat, flight_id=int(fid),
                                frechet_3d=discrete_frechet(
                                    g[['x_true', 'y_true', 'z_true']].to_numpy(float),
                                    g[['x_pred', 'y_pred', 'z_pred']].to_numpy(float))))
    rc = pd.concat(rc_parts, ignore_index=True)
    return pd.DataFrame(fr_rows), rc[['session', 'bat', 'flight_id', 'err_3d']]


def prod_reference():
    """Test-set encode/decode tables for the production `cross1000` model, evaluated on the
    full-context (untiled) test split — its native setting. Computed once and cached, so each
    window can be reported against it as it finishes."""
    path = os.path.join(PKL_ROOT, 'prod_reference.pkl')
    if os.path.exists(path):
        return _pkl.load(open(path, 'rb'))
    print('[ref] evaluating the production cross1000 model for comparison …', flush=True)
    loaders, unit, meta = load_window('full')
    model = build_model(unit, prod_params(), seed=SEED)
    model.load_state_dict(torch.load(PROD_CKPT, map_location=DEVICE))
    model.eval()
    adapter = JointEncoderAdapter(model, unit).to(DEVICE).eval()
    enc = encode_eval_records(adapter, loaders, DEVICE)
    fr, rc = decode_eval(model, loaders, meta)
    ref = dict(encode=enc, fr=fr, rc=rc)
    _pkl.dump(ref, open(path, 'wb'))
    print(f'[ref] production: encode median fde={enc["fde"].median():.4f} '
          f'corr={enc["corr"].median():.4f} | decode median frechet={fr["frechet_3d"].median():.3f} '
          f'recon={rc["err_3d"].median():.3f}', flush=True)
    del model, adapter, loaders
    torch.cuda.empty_cache()
    return ref


def compare_to_prod(W, out, ref):
    """Print this window against the production model: medians, deltas, and a session-level
    paired Wilcoxon (the same unit of analysis ED Fig 3 stars use)."""
    from scipy.stats import wilcoxon
    rows = [('Poisson FDE', 'encode', 'fde', +1), ('Pearson r', 'encode', 'corr', +1),
            ('Fréchet (m)', 'fr', 'frechet_3d', -1), ('recon err (m)', 'rc', 'err_3d', -1)]
    print(f'  --- ctx{wtag(W)} vs production cross1000 (+ = window better) ---', flush=True)
    for label, key, col, sign in rows:
        w, r = out[key], ref[key]
        mw, mr = w[col].median(), r[col].median()
        sw = w.groupby('session')[col].median()
        sr = r.groupby('session')[col].median()
        common = sw.index.intersection(sr.index)
        try:
            p = wilcoxon(sw[common], sr[common]).pvalue
        except ValueError:                            # all-zero differences
            p = np.nan
        print(f'    {label:<14} window={mw:+.4f}  prod={mr:+.4f}  '
              f'delta={sign * (mw - mr):+.4f}  ({sign * (mw - mr) / abs(mr) * 100:+.1f}%)  '
              f'session-paired p={p:.2g}  n={len(common)}', flush=True)


def window_paths(W):
    d = os.path.join(PKL_ROOT, 'per_window'); os.makedirs(d, exist_ok=True)
    t = wtag(W)
    return {k: os.path.join(d, f'ctx{t}_{VARIANT}_{k}.pkl') for k in ('encode', 'fr', 'rc')}


def run_window(W, n_trials):
    """Search -> final 1000-epoch refit -> test-set encode/decode tables for one window."""
    paths = window_paths(W)
    if all(os.path.exists(p) for p in paths.values()):
        print(f'[ctx{wtag(W)}] results present — skipping', flush=True)
        out = {k: pd.read_pickle(p) for k, p in paths.items()}
        compare_to_prod(W, out, prod_reference())
        return out, None

    print(f'[ctx{wtag(W)}] loading joint dataloaders '
          f'(window_size={None if W == "full" else W}) …', flush=True)
    loaders, unit, meta = load_window(W)

    params = search_window(W, loaders, unit, meta, n_trials)

    ckpt = os.path.join(CKPT_ROOT, f'joint_ctx{wtag(W)}.pth')
    pjson = ckpt.replace('.pth', '.params.json')
    os.makedirs(CKPT_ROOT, exist_ok=True)
    model = build_model(unit, params, seed=SEED)
    # Reuse a final model only if it was trained with the params the search now selects —
    # a resumed search can pick a new best, and a state dict from the old best would load
    # without error (same shapes) while silently belonging to a different configuration.
    reusable = (os.path.exists(ckpt) and os.path.exists(pjson)
                and json.load(open(pjson)) == params)
    if reusable:
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        print(f'[ctx{wtag(W)}] loaded final model {ckpt}', flush=True)
        best = None
    else:
        t0 = time.time()
        model, best = train_one(model, loaders, meta, params, n_epochs=FINAL_EPOCHS,
                                seed=SEED, tag=f'ctx{wtag(W)}', verbose_every=25)
        torch.save(model.state_dict(), ckpt)
        json.dump(params, open(pjson, 'w'), indent=2, default=float)
        print(f'[ctx{wtag(W)}] final refit: err={best["err"]:.3f} fde={best["fde"]:.4f} '
              f'({best["epochs"]} ep, {time.time() - t0:.0f}s) -> {ckpt}', flush=True)
    model.eval()

    adapter = JointEncoderAdapter(model, unit).to(DEVICE).eval()
    enc = encode_eval_records(adapter, loaders, DEVICE).assign(context=wtag(W),
                                                              context_x=context_x(W))
    fr, rc = decode_eval(model, loaders, meta)
    fr = fr.assign(context=wtag(W), context_x=context_x(W))
    rc = rc.assign(context=wtag(W), context_x=context_x(W))
    out = dict(encode=enc, fr=fr, rc=rc)
    for k, p in paths.items():
        out[k].to_pickle(p)
    print(f'[ctx{wtag(W)}] encode median fde={enc["fde"].median():.4f} '
          f'corr={enc["corr"].median():.4f} | decode median frechet={fr["frechet_3d"].median():.3f} '
          f'recon={rc["err_3d"].median():.3f}', flush=True)

    del model, adapter, loaders
    torch.cuda.empty_cache()
    compare_to_prod(W, out, prod_reference())
    return out, dict(params=params, best=best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', default=','.join(map(str, CTX_WINDOWS)))
    ap.add_argument('--trials', type=int, default=N_TRIALS)
    args = ap.parse_args()
    windows = [w if w == 'full' else int(w) for w in args.windows.split(',')]

    parts, meta_rows = {k: [] for k in ('encode', 'fr', 'rc')}, {}
    for W in windows:
        out, info = run_window(W, args.trials)
        for k in parts:
            parts[k].append(out[k])
        if info is not None:
            meta_rows[wtag(W)] = info

    names = dict(encode=f'ctx_encode_cross1000_{VARIANT}.pkl',
                 fr=f'ctx_decode_fr_cross1000_{VARIANT}.pkl',
                 rc=f'ctx_decode_rc_cross1000_{VARIANT}.pkl')
    for k, fname in names.items():
        pd.concat(parts[k], ignore_index=True).to_pickle(os.path.join(PKL_ROOT, fname))
    if meta_rows:
        jpath = os.path.join(PKL_ROOT, f'ctx_best_params_cross1000_{VARIANT}.json')
        old = json.load(open(jpath)) if os.path.exists(jpath) else {}
        old.update(meta_rows)
        json.dump(old, open(jpath, 'w'), indent=2, default=float)
    print(f'[ctx] wrote combined tables to {PKL_ROOT}', flush=True)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
