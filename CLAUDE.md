# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Editing Rules

**Do not modify the `digitalbat/` package without explicit permission.** Only modify:
- Notebooks (`notebooks/`)
- Temporary files (`tmp/`)

**Plotting**: Prefer seaborn over matplotlib for all visualizations.

**Data analysis**: In general, prefer pandas DataFrames over collections of arrays/dicts. Don't duplicate DataFrames unnecessarily. Make use of Pandas's groupby, agg, pivot, melt, and similar operations to avoid loops over data structures where possible.

**Working with Jupyter Notebooks**: I often will edit notebook alongside Claude. To prevent conflicts, it is therefore critical to refresh your memory after every prompt.
1. Read Before Writing: After every prompt, execute a read command on the relevant cell(s) of the notebook before performing a write operation, reducing the chance of outdated content causing an overwrite.
2. Explicit Cell Targeting: Always get the specific cell IDs first, and then use the cell_id parameter to target the exact position, preventing the tool from arbitrarily replacing cells.

**Notebook changes go in a worktree**: Every request that changes a notebook must be done in a git worktree, not in the primary working directory.
1. **One worktree per conversation**: On the first notebook change of a conversation, enter a worktree based on the current `HEAD` of the checked-out branch (`worktree.baseRef=head` is already configured). Reuse that same worktree for every subsequent notebook change in the conversation — do not create a new one per request.
2. **Assume HEAD has diverged on every write**: I edit notebooks in the primary directory while you work. Before writing back to the primary copy, re-read the current primary-directory state of the notebook, then apply *only* the cells you changed in the worktree — never copy the whole worktree file over, and never `git checkout`/`git restore` the primary copy. If a cell you changed has also changed in the primary copy, stop and ask rather than clobbering it.
3. **Target cells explicitly**: Carry the same cell-ID discipline into the worktree — diff by cell ID, not by file, so the merge back is a small set of targeted cell edits.

**Reverting changes**: Be extremeley cautious when using `git checkout` or `git restore` to revert notebook changes — this discards ALL uncommitted edits, including the user's own work. To undo only your last edit, re-read the current file state and apply a targeted fix. If you need a previous version of specific lines, use `git show HEAD:path` to read it without overwriting the working copy.

**For running Python code**: Write scripts to `tmp/*.py` files and execute them with `python tmp/script.py`. Do not use `python << 'EOF'` heredoc statements or `python -c "..."` inline commands (especially multi-line ones), as these are flagged for potential obfuscation and require manual acceptance. Newlines followed by `#` inside a quoted argument can hide arguments from path validation.

## Project Overview

Digital twin model for bat hippocampal neurons. Predicts neural spiking activity from behavioral variables (3D location, trajectory, IMU) during free flight, and decodes behavior from neural activity.

## Usage

No formal package installation. Import directly with the digitalbat directory in PYTHONPATH:

```python
from digitalbat.models.former import Bat_Transformer
from digitalbat.models.mlp import Bat_MLP, Bat_MLP_Behavior
from digitalbat.datasets.data import Data_Prep_for_Raw
from digitalbat.training.trainer import bat_former_trainer, bat_mlp_trainer
```

## Architecture

### Dual-Branch Encoding Model (Bat_Transformer)

The main model in `models/former.py` has two parallel encoding branches that are concatenated before session-specific readout heads:

1. **Place Branch**: 3D position → RBF spatial basis (Gaussian place fields) → MLP embedding
2. **Trajectory Branch**: Full behavioral sequence → Linear embedding → Transformer encoder with RoPE

Either branch can be disabled via `use_place_as_input` / `use_trajectory_as_input` flags.

### Session-Wise Heads

All models use `nn.ModuleDict` for session-specific layers to handle multiple recording sessions with different neuron counts:
- `Bat_Transformer.readouts[session_name]` - output projection per session
- `Bat_Transformer.spatial_basis[session_name]` - RBF centers/sigma per session
- `Bat_MLP.readouts[session_name]` - output projection per session

The `output_dim_dict` parameter (e.g., `{'Dataset_3_14543_240419': 289, 'Dataset_1_32622_231009': 95}`) configures these at initialization.

### Data Pipeline Flow

`Data_Prep_for_Raw` must be called in order:
1. `get_all_behavior()` - loads and resamples behavior data, sets `_behavior_ready=True`
2. `get_all_spike()` - bins spikes, requires behavior ready, sets `_spike_ready=True`
3. `get_flight_clustering()` - clusters flight trajectories, requires spike ready
4. `split_flight_clusters_for_training()` - creates train/valid/test splits

### Transformer Attention Options

`TransformerEncoderRoPE` in `models/layers.py` supports:
- Standard full attention (default)
- Causal attention (`is_causal=True`)
- BigBird sparse attention (`is_bigbird=True`) with sliding window + global + random tokens

### Validation Metrics

Trainers accept custom `valid_fn(y_true, y_pred) -> array` returning per-neuron scores. Common choices:
- `corr(y_true, y_pred, axis=0)` - Pearson correlation
- `poisson_fde_population(y_true, y_pred)` - Poisson fraction deviance explained

## Typical Workflow

### High-Level Data Loading

Use `bat_data_loader` for end-to-end data preparation:

```python
from digitalbat.datasets.data import bat_data_loader

data_config = dict(
    use_speed_head_dir=False,
    use_IMU_flag=False,
    use_click_flag=False,
    use_LFP_flag=False,
    only_flight_flag=True,
    spike_count_flag=True,
    bin_width=0.10,  # seconds
    split_flights_for_train='random',  # or 'manual'
    split_each_trajectory_flag=True,
    window_size=None,
    session_names=['Dataset_3_14543_240419', ...],
)

dataloaders, meta_info_dict = bat_data_loader(data_config)
```

Returns:
- `dataloaders['train'|'valid'|'test'][session_name]` - PyTorch DataLoaders
- `meta_info_dict` - contains `unit_num_dict`, `behavior_dim`, `spatial_centers_sigma`, `flight_type_test_ood`, `flight_type_test_indomain`

### Batch Structure

When `split_each_trajectory_flag=True`, batches are namedtuples with fields:
- `batch.behavior` - shape `(N, T, behavior_dim)`
- `batch.spike` - shape `(N, T, num_neurons)`
- `batch.mask` - shape `(N, T)`, 1.0 for valid timesteps
- `batch.flight_path_index` - trajectory IDs
- `batch.flight_type_index` - cluster type IDs

### Model Forward Pass

```python
y_pred = model(
    session_name,
    X_batch,           # behavior tensor
    mask_batch,        # validity mask
    is_causal=False,
    is_bigbird=False,
)
# Apply mask for loss: loss_fn(y_pred[mask == 1.0], y_true[mask == 1.0])
```

### Evaluation Pattern

```python
model.eval()
for session_name, session_data in dataloaders['test'].items():
    y_val_list, y_pred_list = [], []
    for batch in session_data:
        y_pred = model(session_name, batch.behavior.to(device), batch.mask.to(device))
        y_val_list.append(batch.spike[batch.mask == 1.0].numpy())
        y_pred_list.append(y_pred[batch.mask == 1.0].detach().cpu().numpy())
    scores = valid_fn(np.vstack(y_val_list), np.vstack(y_pred_list))
```
