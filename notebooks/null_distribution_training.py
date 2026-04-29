"""Training scaffolds extracted from preprocess-data-v0j-share.ipynb.

These blocks were silenced (fully commented) in the notebook because
they are expensive one-shot jobs whose outputs are cached to disk; the
notebook re-loads those checkpoints rather than retraining.  Moved
here so the notebook stays uncluttered.

Sections:
  1. Transformer weight_decay tuning (saves no checkpoint by itself;
     pick best wd, then run section 2's training in the notebook).
  2. Final transformer training at best_wd_transformer
     (the trainer call removed from notebook cell 12 / id=04d6dc69).
  3. Null-distribution helper: DataSample namedtuple and
     circshift_dataloaders (shared by sections 4 and 6).
  4. Transformer null distribution: train on circularly shifted spikes,
     evaluate on real test set, compute per-neuron p-values.
  5. Final per-variant GLM training at best_wds (the trainer call
     removed from notebook cell 98 / id=cb74549a).
  6. GLM null distribution: same circular-shift framework as section 4
     but for each GLM variant.

Usage
-----
This file is not meant to be run top-to-bottom.  Each section assumes
the notebook has already been executed up to and including the cell
that defines its prerequisites (model_config, dataloaders, trainer_config,
glm_variant_info, etc.).  The intended workflow is:

  1. Run the notebook through cell 16 (transformer load).
  2. To regenerate transformer null distribution: paste section 4 into
     a new cell and run.  Outputs cache to ./out/null/transformer_*.pth.
  3. To regenerate GLM null distributions: run sections 5+6 likewise.

All sections write to ./out/ subdirectories so re-running is cheap if
the checkpoints already exist.
"""

# =============================================================================
# Section 1 — Transformer weight_decay tuning (notebook cell 11 / 4fbea27f)
# =============================================================================
# Sweeps weight_decay candidates on the validation set, warm-starting each
# from the previous best.  Run once; record the chosen value in cell 12's
# `best_wd_transformer = ...`.
from copy import deepcopy

weight_decay_candidates = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

best_wd_transformer = None
best_val_score_transformer = -np.inf
best_weights_transformer = None
wd_results_transformer = []

for wd in weight_decay_candidates:
    print(f'Testing weight_decay={wd}')

    # Create model, warm-start from previous best if available
    bat_former_tune = Bat_Transformer(**model_config)
    if best_weights_transformer is not None:
        bat_former_tune.load_state_dict(deepcopy(best_weights_transformer))

    # Train with this weight_decay
    tune_config = dict(
        model=bat_former_tune,
        dataloaders=dataloaders,
        device=device,
        is_causal=False,
        is_bigbird=False,
        attention_window_size=9,
        num_global=2,
        n_epochs=1000,
        lr=0.01,
        weight_decay=wd,
        loss_fn=nn.PoissonNLLLoss(log_input=False),
        valid_fn=poisson_fde_population,
        valid_stat='median',
        early_stopping_patience=100,
        save_model=False,
    )
    val_score, weights, val_scores = bat_former_trainer(**tune_config)
    wd_results_transformer.append({'weight_decay': wd, 'val_score': val_score})
    print(f'  Final val score (median FDE, filtered): {val_score:.4f}')

    if val_score > best_val_score_transformer:
        best_val_score_transformer = val_score
        best_wd_transformer = wd
        best_weights_transformer = deepcopy(weights)

print(f'\nBest weight_decay: {best_wd_transformer} with val score: {best_val_score_transformer:.4f}')

fig, ax = plt.subplots(figsize=(6, 4))
wds = [r['weight_decay'] for r in wd_results_transformer]
scores = [r['val_score'] for r in wd_results_transformer]
ax.semilogx([w if w > 0 else 1e-6 for w in wds], scores, 'o-')
ax.axvline(best_wd_transformer if best_wd_transformer > 0 else 1e-6,
           color='r', linestyle='--', label=f'Best: {best_wd_transformer}')
ax.set_xlabel('Weight Decay')
ax.set_ylabel('Validation FDE (median, filtered)')
ax.set_title('Transformer Weight Decay Tuning')
ax.legend()
plt.show()


# =============================================================================
# Section 2 — Final transformer training (cell 12 / 04d6dc69 commented tail)
# =============================================================================
# Trains the final transformer at the chosen weight_decay and saves it to
# trainer_config['model_name'].  trainer_config is set up in cell 12.
best_acc, best_weights, acc_vals = bat_former_trainer(**trainer_config)
print(f'Final model trained with weight_decay={best_wd_transformer}, '
      f'best val score: {best_acc:.4f}')

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
axes.plot(acc_vals)
axes.set_title(f"Best: {best_acc:.4f}")
axes.set_xlabel("Epoch")
axes.set_ylabel("FDE (median, filtered)")
plt.show()


# =============================================================================
# Section 3 — Null-distribution helper (cell 15 / be666694)
# =============================================================================
# Used by sections 4 and 6.  Defines DataSample (mirrors the dataloader
# batch namedtuple) and circshift_dataloaders, which rolls each flight's
# spike tensor along the time axis by a random offset within the valid
# (non-padded) timesteps.  The 'test' tier is left untouched.
from collections import namedtuple

DataSample = namedtuple('DataSample', ['behavior', 'spike', 'mask',
                                        'flight_path_index', 'flight_type_index'])


def circshift_dataloaders(dataloaders, rng, tiers=('train', 'valid')):
    """Return a shallow copy of *dataloaders* with circularly shifted spikes.

    For every flight in the specified tiers, the spike tensor is rolled along
    the time axis by a random offset (within valid timesteps), breaking the
    temporal alignment with behavior while preserving spike autocorrelation.
    The 'test' tier (and any tier not listed) is left untouched.
    """
    shifted = {}
    for tier in dataloaders:
        if tier not in tiers:
            shifted[tier] = dataloaders[tier]
            continue
        shifted[tier] = {}
        for session_name, loader in dataloaders[tier].items():
            new_batches = []
            for batch in loader:
                spike_shifted = batch.spike.clone()
                for i in range(batch.spike.shape[0]):
                    valid = batch.mask[i] == 1.0
                    n_valid = valid.sum().item()
                    if n_valid > 1:
                        shift = int(rng.integers(1, n_valid))
                        spike_shifted[i, valid] = torch.roll(
                            batch.spike[i, valid], shift, dims=0)
                new_batches.append(DataSample(
                    behavior=batch.behavior,
                    spike=spike_shifted,
                    mask=batch.mask,
                    flight_path_index=batch.flight_path_index,
                    flight_type_index=batch.flight_type_index,
                ))
            shifted[tier][session_name] = new_batches
    return shifted


# =============================================================================
# Section 4 — Transformer null distribution (cell 18 / eefb8eaf)
# =============================================================================
# Trains a fresh transformer on circularly shifted spikes for each of
# n_null_shifts independent rotations, evaluates on the real test set, and
# computes per-neuron p-values (one-sided: fraction of null scores >= real).
# Each null model is cached to ./out/null/transformer_null_*.pth so re-running
# only retrains missing shifts.  Inherits metrics, flight_filters, model_config,
# best_wd_transformer, transformer_results, dataloaders, device, seed from
# the notebook.
import os

n_null_shifts = 100
null_dir = './out/null'
os.makedirs(null_dir, exist_ok=True)

transformer_null = {}   # key -> session -> (n_shifts, n_neurons)
for metric_name, valid_fn in metrics.items():
    for flight_filter in flight_filters:
        key = f"{metric_name}_{flight_filter or 'all'}"
        transformer_null[key] = {s: [] for s in dataloaders['test'].keys()}

for shift_i in range(n_null_shifts):
    print(f"\rNull shift {shift_i + 1}/{n_null_shifts}", end="")
    null_rng = np.random.default_rng(shift_i + seed)  # deterministic RNG for reproducibility
    null_path = os.path.join(null_dir, f'transformer_null_{shift_i:03d}_seed_{seed}_wd_{best_wd_transformer}.pth')

    null_model = Bat_Transformer(**model_config)

    if os.path.exists(null_path):
        null_model.load_state_dict(torch.load(null_path, map_location=device))
        null_model = null_model.to(device)
    else:
        shifted_dl = circshift_dataloaders(dataloaders, null_rng)
        bat_former_trainer(
            model=null_model, dataloaders=shifted_dl, device=device,
            n_epochs=1000, lr=0.01, weight_decay=best_wd_transformer,
            loss_fn=nn.PoissonNLLLoss(log_input=False),
            valid_fn=poisson_fde_population, valid_stat='median',
            early_stopping_patience=100, save_model=True, model_name=null_path,
        )

    null_model.eval()
    for metric_name, valid_fn in metrics.items():
        for flight_filter in flight_filters:
            key = f"{metric_name}_{flight_filter or 'all'}"
            scores = evaluate_model(
                null_model, dataloaders, valid_fn, device,
                flight_filter=flight_filter, meta_info_dict=meta_info_dict,
            )
            for s in scores:
                transformer_null[key][s].append(scores[s])

    del null_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("\nDone.")

# Stack and compute p-values
transformer_pvals = {}
for key in transformer_null:
    transformer_null[key] = {
        s: np.stack(v) for s, v in transformer_null[key].items() if v
    }
    transformer_pvals[key] = {}
    for s in transformer_null[key]:
        real = transformer_results[key][s]
        null = transformer_null[key][s]  # (n_shifts, n_neurons)
        transformer_pvals[key][s] = (null >= real[np.newaxis, :]).mean(axis=0)

    all_pvals = np.hstack(list(transformer_pvals[key].values()))
    print(f"Transformer {key}: {(all_pvals < 0.05).mean():.1%} significant (p < 0.05)")


# =============================================================================
# Section 5 — Final per-variant GLM training (cell 98 / cb74549a commented tail)
# =============================================================================
# Trains each GLM variant (GLM, GLM + lags, GLM + lags + kin) at its best wd
# from `best_wds` (set in the notebook).  Saves to `model_path` from
# `glm_model_names` (also set in the notebook).
for name, (cfg, cls, dls) in glm_variant_info.items():
    wd = best_wds[name]
    model_path = glm_model_names[name]
    print(f"\n{'='*20} Training {name} (wd={wd}) {'='*20}")

    best_acc, best_weights, acc_vals = bat_former_trainer(
        model=glm_models[name], dataloaders=dls, device=device,
        n_epochs=1000, lr=0.01, weight_decay=wd,
        loss_fn=nn.PoissonNLLLoss(log_input=False),
        valid_fn=poisson_fde_population, valid_stat='median',
        early_stopping_patience=20, save_model=True, model_name=model_path,
    )
    print(f'  Best val score: {best_acc:.4f}')

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(acc_vals)
    ax.set(title=f"{name}: {best_acc:.4f}", xlabel="Epoch", ylabel="FDE (median)")
    plt.show()


# =============================================================================
# Section 6 — GLM null distribution (cell 101 / 4192d7a2)
# =============================================================================
# Same circular-shift framework as section 4 but per GLM variant. Caches
# to ./out/null/{variant_tag}_null_*.pth.
all_glm_null = {}
all_glm_pvals = {}
for name, (cfg, cls, dls) in glm_variant_info.items():
    wd = best_wds[name]
    tag = name.lower().replace(' + ', '_').replace(' ', '_')
    mi = meta_info_dict_kin if 'kin' in name else meta_info_dict
    print(f"\n{'='*20} Null distribution: {name} (wd={wd}) {'='*20}")

    glm_null = {}
    for metric_name, valid_fn in metrics.items():
        for flight_filter in flight_filters:
            key = f"{metric_name}_{flight_filter or 'all'}"
            glm_null[key] = {s: [] for s in dls['test'].keys()}

    for shift_i in range(n_null_shifts):
        print(f"\r  Null shift {shift_i + 1}/{n_null_shifts}", end="")
        null_rng_glm = np.random.default_rng(shift_i + seed)
        null_path = os.path.join(null_dir, f'{tag}_null_{shift_i:03d}_seed_{seed}_wd_{wd}.pth')

        null_glm = cls(**cfg)

        if os.path.exists(null_path):
            null_glm.load_state_dict(torch.load(null_path, map_location=device))
            null_glm = null_glm.to(device)
        else:
            shifted_dl = circshift_dataloaders(dls, null_rng_glm)
            bat_former_trainer(
                model=null_glm, dataloaders=shifted_dl, device=device,
                n_epochs=1000, lr=0.01, weight_decay=wd,
                loss_fn=nn.PoissonNLLLoss(log_input=False),
                valid_fn=poisson_fde_population, valid_stat='median',
                early_stopping_patience=20, save_model=True, model_name=null_path,
            )

        null_glm.eval()
        for metric_name, valid_fn in metrics.items():
            for flight_filter in flight_filters:
                key = f"{metric_name}_{flight_filter or 'all'}"
                scores = evaluate_model(
                    null_glm, dls, valid_fn, device,
                    flight_filter=flight_filter, meta_info_dict=mi,
                )
                for s in scores:
                    glm_null[key][s].append(scores[s])

        del null_glm
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n  Done.")

    # Stack and compute p-values
    glm_pvals_cur = {}
    for key in glm_null:
        glm_null[key] = {
            s: np.stack(v) for s, v in glm_null[key].items() if v
        }
        glm_pvals_cur[key] = {}
        for s in glm_null[key]:
            real = all_glm_results[name][key][s]
            null = glm_null[key][s]
            glm_pvals_cur[key][s] = (null >= real[np.newaxis, :]).mean(axis=0)

        all_pvals = np.hstack(list(glm_pvals_cur[key].values()))
        print(f"  {name} {key}: {(all_pvals < 0.05).mean():.1%} significant (p < 0.05)")

    all_glm_null[name] = glm_null
    all_glm_pvals[name] = glm_pvals_cur

glm_pvals = all_glm_pvals['GLM']
glm_lag_pvals = all_glm_pvals['GLM + lags']
glm_kin_pvals = all_glm_pvals['GLM + lags + kin']
