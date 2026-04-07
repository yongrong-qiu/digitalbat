# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
