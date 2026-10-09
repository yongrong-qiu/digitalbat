# digitalbat

**A hippocampal digital Twin reveals latent structure in freely flying bats.**

`digitalbat` models CA1/CA3 single-unit activity recorded with Neuropixels in freely
flying Egyptian fruit bats. One masked encoder–decoder transformer, the **Twin**, does two
things with a single set of weights and one latent space shared across 17 sessions, 6 bats
and 2 environments:

- it predicts spiking from 3-D flight behavior (*encoding*);
- it recovers the flight path from spiking (*decoding*).

This repository holds the model package, the training drivers and the notebooks behind
every figure in the paper.

> **If you are here for the paper figures, go straight to
> [`notebooks/figures.ipynb`](notebooks/figures.ipynb)** and the
> [figure map](#figure-map) below. The tag **`paper-figures-v1`** marks the version used
> for the paper.

---

## Contents

- [Repository layout](#repository-layout)
- [Getting started](#getting-started)
- [The figure notebooks](#the-figure-notebooks)
  - [How `figures.ipynb` is organised](#how-figuresipynb-is-organised)
  - [Figure map](#figure-map)
  - [Where outputs go](#where-outputs-go)
  - [Caches, checkpoints and train-if-missing](#caches-checkpoints-and-train-if-missing)
  - [Inputs that are *not* regenerated automatically](#inputs-that-are-not-regenerated-automatically)
- [Other notebooks](#other-notebooks)
- [The `digitalbat` package](#the-digitalbat-package)
- [Training drivers (`training/`)](#training-drivers-training)

---

## Repository layout

```
digitalbat/
├── digitalbat/                 # Python package: data, models, training, decoding, utilities
│   ├── datasets/               #   loading, binning, flight clustering, default session list
│   ├── models/                 #   Twin (joint.py), Bat_Transformer, GLMs, MLPs, decoders
│   ├── training/               #   trainers incl. train-if-missing for the Twin
│   ├── decoding/               #   Naive-Bayes / Perceiver / decode-cost pipelines, Fréchet
│   └── utility/                #   metrics (Poisson FDE, corr), flight clusters, palettes
├── notebooks/
│   ├── figures.ipynb           # ★ every figure in the paper, and Table S1
│   ├── plotting_helpers.py     #   publication style, palettes, stats brackets, save_fig()
│   ├── notebook_helpers.py     #   reward/perch locations, Mantel and MRM tests
│   ├── machine_paths.py        #   one-toggle data/code path profiles (edit this first)
│   ├── motif_helpers.py        #   latent-motif segmentation (latent_motifs.ipynb)
│   ├── null_distribution_training.py
│   ├── *.ipynb                 #   analysis notebooks (see "Other notebooks")
│   └── out/                    #   caches, checkpoints and exported figures (git-ignored*)
├── training/                   # standalone, resumable training drivers (ED Fig. 3, transfer)
├── pyproject.toml              # package metadata, dependencies and optional extras
├── requirements.txt            # exact versions used for the paper
├── CLAUDE.md                   # architecture notes and API cheat-sheet
└── LICENSE
```

\* `notebooks/out/` is in `.gitignore`. The only files under it in git are the
unified-CEBRA checkpoint (`out/cebra/…pt`) and the Twin Optuna study
(`out/joint/joint_optuna_schemewt.pkl`).

---

## Getting started

### 1. Environment

Dependencies are declared in [`pyproject.toml`](pyproject.toml). Install PyTorch first
if you need a particular CUDA build (see [pytorch.org](https://pytorch.org/get-started/locally/)),
then install the repo in editable mode from the repo root:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126   # optional: pick your CUDA build
pip install -e ".[figures]"
pip install --no-deps cebra==0.6.1     # needed by figures.ipynb; see note below
```

`cebra` is installed separately because its metadata pins `numpy<2` on Windows, which
would downgrade NumPy. It runs fine on NumPy 2.4, and the `figures` extra already
installs its other dependencies.

| Extra | Adds | Needed for |
|---|---|---|
| *(none)* | `torch`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`, `h5py`, `numba`, `transformers` | the `digitalbat` package and the `training/` drivers |
| `figures` | `optuna`, `POT` (imported as `ot`), `statsmodels`, `joblib`, `jupyterlab`, `ipykernel`, plus `cebra` (installed separately, above) | `figures.ipynb` |
| `interpret` | `shap` | `digitalbat.utility.interpreter` |
| `motifs` | `dynamax`, `jax` | `latent_motifs.ipynb` |
| `notebooks` | `plotly`, `umap-learn`, `ipympl`, `ripser`, `persim`, `rastermap`, `diptest` | the exploratory notebooks |
| `all` | all of the above | |

Python ≥ 3.10 is required. To reproduce the paper's environment exactly (Python 3.13,
PyTorch 2.7.1 + CUDA 12.6, NumPy 2.4, SciPy 1.18, pandas 2.2, scikit-learn 1.6,
Optuna 4.8), use the pinned [`requirements.txt`](requirements.txt), which covers the
core package and `figures.ipynb`:

```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install --no-deps cebra==0.6.1
```

Installing the package is optional for the notebooks: they put the repo root on
`sys.path` themselves (via `machine_paths.setup()`).

A CUDA GPU is used if one is available (`device = 'cuda' if torch.cuda.is_available()`).
Re-running the notebook from cached checkpoints works on CPU. Retraining models from
scratch is only practical on a GPU; the paper's models were trained on a single NVIDIA
GeForce RTX 5090.

### 2. Point the code at the data

The recordings were previously reported by Forli et al. and are available from the
corresponding author on reasonable request. Everything that touches disk goes through
[`notebooks/machine_paths.py`](notebooks/machine_paths.py). It defines one `Paths`
profile per machine:

| Field | Meaning |
|---|---|
| `data_root` | Folder holding the `Dataset_*` session folders (use forward slashes, keep the trailing `/`). `UnitMatch/` is expected inside it. |
| `code_root` | Folder that contains this repo checkout (fallback only; the notebook's own checkout is preferred). |
| `cebra_dir` | Local CEBRA checkout to put on `sys.path`, or `None` to use the pip-installed `cebra`. |

`MACHINE = 'auto'` picks `win_batlab` on Windows and `ubuntu` everywhere else. To run
somewhere new, add a profile to `PROFILES` and set `MACHINE` to its name. The first cell
of `figures.ipynb` prints which profile it picked and whether `data_root` was found.

The 17 sessions used throughout the paper are `DEFAULT_SESSION_NAMES` in
[`digitalbat/datasets/defaults.py`](digitalbat/datasets/defaults.py). They come from
Datasets 1–3: 13 sessions in the indoor flight room (4 bats) and 4 in the outdoor field
station (`Dataset_2_*`, 2 bats). `KRISHNA_SESSION_NAMES` are not part of the paper.

### 3. Run

```bash
git checkout paper-figures-v1   # optional: the exact version used for the paper
cd notebooks                    # the notebooks assume cwd = notebooks/ (all paths are ./out/...)
jupyter lab figures.ipynb
```

Run **Shared setup** (cells 1–8) first. After that, each figure section can be run on
its own, with the dependencies [listed below](#figure-map).

---

## The figure notebooks

[`notebooks/figures.ipynb`](notebooks/figures.ipynb) produces every figure in the paper
(Figs. 1–4 and Extended Data Figs. 1–10) and Supplementary Table S1, with each figure's
rendered output saved in the notebook. Every analysis uses one Twin checkpoint and the same
17 sessions:

```
out/joint/joint_v2_cross1000_kin0_L4_d64_ff64_lam0.68_pw0.36_identity_we2.8_wd2.2_wa0.31_n17.pth
```

That checkpoint is a 4-layer, 64-d latent `JointMaskedTransformer` trained for 1,000
epochs on the encode, decode and cross-modal infilling masking schemes. Its training
recipe is in `joint-encode-decode.ipynb` §6.10. `figures.ipynb` loads the data **once**,
builds the model **once**, and every panel reuses those objects.

### How `figures.ipynb` is organised

Each figure is one section, and every section follows the same pattern:

1. a markdown **header** with the figure's title and a one-line summary of each panel;
2. one or more **data cells** that compute (or load from cache) everything the figure needs;
3. one **canvas cell** that lays out the panels and calls `ph.save_fig(...)`;
4. a markdown **caption** with the figure legend exactly as it appears in the paper.

Sections follow the data dependencies rather than figure order. Extended Data Fig. 4
defines the flight motifs that Figs. 2 and 3 use, so it comes before them, and the
Fig. 3 data cells also feed Extended Data Figs. 8 and 9. Extended Data Figs. 6–7 draw in
their own 7-pt style: their first cell saves the notebook's plotting style and their last
cell restores it, so run that block from top to bottom. Cell indices below refer to
`paper-figures-v1`. Cell IDs stay fixed across edits, so search for the ID if the indices
have moved. Schematic panels are drawn outside the notebooks and are listed as such.

### Figure map

#### Shared setup — run first

| Cells | Cell ID | What it does |
|---|---|---|
| 1 | `placeholder` | Imports, `machine_paths` setup, `ph.set_pub_style()`, `FIG_DIR = out/figures`, global seed = 1, `OUR_MODEL = 'Twin'` |
| 3 | `215a54ff` | Loads the 17 sessions **once** with `make_joint_dataloaders` (position-only, `behavior_dim = 3`) |
| 4 | `descstats` | Dataset descriptive statistics (neuron / session / bat / environment / flight counts reported in Methods) |
| 5 | `de9a3bd3` | Builds the Twin from `JOINT_CKPT`. If the checkpoint is missing it runs a 200-trial Optuna search and trains the winner (`training.joint_trainer.ensure_joint_model`). |
| 6 | `63e8be8c` | `JointEncoderAdapter` and the per-neuron encode-metric helper |
| 8 | `4c999546` | **Load-or-train utilities** for every baseline model (GLMs, Naïve Bayes, …). `FORCE_RETRAIN = False` |

#### Main figures

| Figure | Panels | `figures.ipynb` cells (canvas in bold) | Saved as |
|---|---|---|---|
| **Fig. 1** — A functional digital Twin achieves superior prediction of hippocampal activity and behavior | **B** decoded flight · **C** decoding accuracy: Naïve Bayes, Unified CEBRA, Twin · **D** example unit, observed vs. predicted rate maps · **E–F** encoding fit: place GLM, trajectory-specific GLM, Twin · **H** transfer to held-out sessions. *Schematics:* **A**, **G** | 10 `21dbef05` decode arm · 11 `b64769bb` (B) · 12 `e9128f87` Naïve Bayes · 13 `cebra_grid_search` · 14 `6d778b26` Unified CEBRA · 15 `9fb97b5c` (C) · 16 `607274c7` (D) · 17 `a91b7f03` (E–F) · **18 `fig1_merged`** · 19 caption | `fig1_full` |
| **Fig. 2** — The Twin latent representation warps physical space to align geometry and dynamics across animals | **F–G** held-out flight segments in Physical XYZ / Unified CEBRA / Twin decode / Twin encode, colored by x and by motif · **H** segment RSA (Hausdorff) · **I** across − within-motif distance · **J** leave-one-route-out motif decoding · **L–M** RSA on DMD temporal and spatial profiles · **N** across − within-motif DMD contrast. *Schematics:* **A–E**, **K** | 38 `daa02161` latent extraction · 39 `5fcfd162` per-flight RSA records · 44 `fig2m_motifs` motif segments · 45 `fig2m_rsa` (H–J) · 46 `fig2_dmd_data` (L–N) · **47 `fd1df114`** · 48 caption | `fig2_embeddings` (`.pdf/.png/.svg`) |
| **Fig. 3** — Latent representations expose spatial generalization at the single-neuron and population level | **A–C** example route and unit on four paths · **D** place-cell sequences in normalized flight progress · **E–G** PV correlation, 3-D distance and goal-heading matrices · **H–I** PV-vs.-lag curves and peak-order shuffle tests · **J** the example unit in the Twin latent · **K–M** rate-map stability across paths (Wasserstein distance) | 72 `07756d6f` raw-spike cache and pair search · 73 `fig3_ratemap_sim` (K–M data) · 74 `23e9fd4f` analysis core · 75 `9e5b5918` curated pairs · **76 `fig3_multipath`** · 77 `fig3-panelb-standalone` (B on its own) · 78 caption | `fig3_multipath_unit<N>`, `fig3_panelB` |
| **Fig. 4** — Neuronal readout weights encapsulate known and novel biological structure | **A** readout PC1 vs. firing rate · **B** residual-readout clustering · **C** cross-validated spatial information · **D–E** neural→position attribution · **F** sleep-matched similarity matrices · **G–I** MRM $R^2$ vs. non-flight rest and sleep · **J** CA1 vs. CA3 classification · **K** MRM within CA1 and CA3 | 87 `5b32c185` readouts and subregions · 88 `fig4_cluster` · 89 `fig4_attribution` · 90 `0715f686` rate maps, non-flight, coupling · 91 `059c3ab5` sleep matching (UnitMatch) · 92 `fig4_stats` · 93 `7ab4c277` · 94 `fig4_mrm_r2` · **95 `fig4_merged`** · 96 caption | `fig4_full` |

#### Extended Data figures

| Figure | Panels | Cells (canvas in bold) | Saved as |
|---|---|---|---|
| **ED Fig. 1** — Model setup and full model ladders | **D** 12-model decoder ladder · **E–H** every encoding model (FDE and r). *Schematics:* **A–C** | `figures.ipynb` 21 `973ed220` encoders · 22 `lstm_rnn_search` (LSTM-RNN) · 23 `cebra_multi_search` (multi-session CEBRA) · 24 `02cfa61a` decoder ladder · **25 `edfig1_merged`** · 26 caption. Needs cell 17. | `edfig1_full` |
| **ED Fig. 2** — Surrogate-trained controls isolate the structure exploited by the digital Twin | **A–C** decode cost: trial shuffle, synthetic place, synthetic trajectory · **D–E** encode cost | `figures.ipynb` 28 `edfig2_costdata` · 29 `edfig2_encode` · **30 `edfig2_merged`** · 31 caption | `edfig2_full` |
| **ED Fig. 3** — Scaling context window and training data improves digital Twin performance | **A** context window · **B** number of training sessions; rows Fréchet / FDE / r | `figures.ipynb` 33 `edfig3_ensure` (runs the [`training/`](#training-drivers-training) drivers if results are missing — long) · 34 `12eac58c` · 35 `7ad46d86` · **36 `edfig3_merged`** · 37 caption | `edfig3_full` |
| **ED Fig. 4** — Flight in both rooms is organized along a principal axis, which defines motifs | **A–D** per-flight axes, fitted room axis, motif segmentation · **E** motifs in the Twin latent | `figures.ipynb` **41 `edfig4_motif_axis`** · 42 caption. Defines the motifs used by Figs. 2–3. | `edfig4_motif_axis` |
| **ED Fig. 5** — The digital Twin aligns encode and decode latents and readouts | **A** shared PCA of encode/decode latents · **B–C** principal angles · **D** behavior-readout energy in the neural readout | `figures.ipynb` **50 `edfig5_subspaces`** · 51 caption | `edfig5_subspaces` |
| *(shared, ED Figs. 6–7)* | per-timepoint encode/decode latents, positions, kinematics, spike counts and landmark labels on the test flights | `figures.ipynb` 52 header · 53 `edfig67_setup` (imports; saves the plotting style) · 54 `edfig67_style` (7-pt style, palettes) · 55 `edfig67_consts` · 56 `edfig67_latents` · 57 `edfig67_pca` · 58 `edfig67_landmarks` | — |
| **ED Fig. 6** — RSA of landmark positions and neural latent decoding of kinematic variables | **A–D** landmark scatter with linear-SVM boundaries and landmark RSA, physical vs. latent · **F** kinematic decoding. *Schematic:* **E** | `figures.ipynb` 60 `edfig6_kin_data` · **61 `edfig6_kin_plot`** (F) · 62 `edfig6_landmark_fn` · 63 `edfig6_landmark_data` · **64 `edfig6_landmark_plot`** (A–D) · 65 caption | `edfig6_kinematics`, `edfig6_landmarks` |
| **ED Fig. 7** — Dynamic mode decomposition (DMD) on exemplary flights | **A–H** raw and reconstructed latents, eigenvalues and DMD power spectra for two example flights | `figures.ipynb` 67 `edfig7_dmd_fn` · **68 `edfig7_dmd_plot`** · 69 caption · 70 `edfig67_restore` (restores the plotting style) | `edfig7_dmd` |
| **ED Fig. 8** — Generalization of place-cell sequences is not inherited from flight heading | **A** azimuth · **B** goal heading | `figures.ipynb` **80 `edfig8_azimuth`** · 81 caption. Needs cells 72–76. | `edfig8_azimuth` |
| **ED Fig. 9** — Additional examples of similar single-neuron tuning along geometrically different paths | two further route examples, laid out as Fig. 3A–J | `figures.ipynb` **83 `65a76f1f`** · 84 `edfig9-panelb-standalone` (B on its own) · 85 caption. Needs cells 72–76. | `edfig9_route_example_<session>`, `edfig9_panelB_<session>` |
| **ED Fig. 10** — Residual readout cluster comparisons | **A–B** orthogonality · **C** cluster composition · **D–G** per-cluster FR, fraction CA1, FDE, r · **H–I** within vs. across cluster | `figures.ipynb` **98 `e82f0a32`** · 99 caption. Needs the Fig. 4 data cells (87–94). | `edfig10_full` |

#### Supplementary Table S1 — Auxiliary masking-task sweep

`figures.ipynb` cells 100 `table_s1_md` and 101 `table_s1_code`. The cell tabulates
`out/joint/joint_scheme_sweep.pkl` and writes `out/figures/table_s1_scheme_sweep.csv` and
`out/figures/table_s1_scheme_sweep.tex`. The `.tex` file is a complete LaTeX table with the
paper's caption and layout.

### Where outputs go

`ph.save_fig(fig, name)` (in [`plotting_helpers.py`](notebooks/plotting_helpers.py))
writes each figure at 300 dpi to

```
notebooks/out/figures/pdf/<name>.pdf
notebooks/out/figures/png/<name>.png
notebooks/out/figures/svg/<name>.svg     # Fig. 2 only
```

By default it also capitalises the first letter of titles, axis labels and legends;
`capitalize=False` turns this off. Panel letters are added at layout time: the canvases
are unlettered, and the caption cell names the panels.

### Caches, checkpoints and train-if-missing

`figures.ipynb` is meant to run end-to-end from an empty `out/`. Every model loads its
checkpoint if one exists and **trains it otherwise**, then saves it. Re-runs are
therefore fast, and a fresh clone can still reproduce everything given enough GPU time.

| Location (under `notebooks/out/`) | Contents |
|---|---|
| `joint/` | Twin checkpoint, Twin Optuna study, Table S1 sweep |
| `bat_glm/` | GLM family (place, trajectory-specific, + history / coupling) |
| `cebra/` | Unified and multi-session CEBRA models and Optuna studies |
| `decode_costs/` | ED Fig. 2 per-flight surrogate decode costs |
| `fig3_cache/` | Fig. 3 raw-spike, rate-map-similarity and bin-free caches |
| `fig4_cache/` | Fig. 4 attribution, SVM and MRM caches |
| `transfer/`, `transfer_joint/` | ED Fig. 3 and Fig. 1H scaling, context-window and transfer results |
| `figures/` | Exported figures and Table S1 (see above) |

Useful switches:

- `FORCE_RETRAIN = True` (cell 8) ignores cached baseline checkpoints.
- `FIG4_FORCE = True` recomputes the Fig. 4 caches.
- `LSTM_N_TRIALS` and `MS_N_TRIALS` (cells 22–23): raise these to extend an Optuna study instead of reloading it.
- The Fig. 3 and Fig. 4 caches are keyed on a hash of the checkpoint and parameters, so they rebuild automatically when an input changes.

### Inputs that are *not* regenerated automatically

| Needed by | Input | Where it comes from |
|---|---|---|
| Everything | `Dataset_*` session folders | `data_root` in `machine_paths.py` |
| Fig. 4F, H (sleep) | `UnitMatch/` tables and `Dataset_0_*` sleep sessions | `data_root` |
| Table S1 | `out/joint/joint_scheme_sweep.pkl` | `joint-encode-decode.ipynb` §6.8 |
| Fig. 1H | `out/transfer_joint/session/results_session_encode.pkl` | `python training/train_joint_transfer.py` |
| Schematic panels (Fig. 1A, G; Fig. 2A–E, K; ED Fig. 1A–C; ED Fig. 6E) | Drawings | Made outside this repo |

---

## Other notebooks

These notebooks develop the individual analyses. `figures.ipynb` contains the versions
used for the paper; the notebooks below are the reference for model selection,
ablations and exploration.

| Notebook | What it is | Related figures |
|---|---|---|
| `joint-encode-decode.ipynb` | Develops the Twin (joint masked encoder–decoder): masking schemes, trainer, Optuna searches, final model (§6.10), auxiliary-task sweep (§6.8) | Twin checkpoint, Table S1 |
| `joint-figures.ipynb` | Population latent analyses: trajectory and landmark RSA, kinematic decoding, DMD | ED Figs. 6–7 (the copies in `figures.ipynb` are the reference) |
| `joint-figures-single-unit.ipynb` | Single-unit place-field analyses in physical vs. latent space | Fig. 3 (related) |
| `model_comparison.ipynb` | Encoder ladder: GLM variants (lags, kinematics, history, coupling), MLP, `Bat_Transformer`, Twin | Fig. 1E–F, ED Fig. 1E–H |
| `decode-position-comparison.ipynb` | Decoder ladder: Naïve Bayes (± continuity, ± lag), transformer decoder, Twin decode arm, CEBRA, PV-Perceiver; surrogate decode costs | Fig. 1C, ED Fig. 1D, ED Fig. 2A–C |
| `cebra-comparison.ipynb` | CEBRA embeddings and decoding vs. the transformer decoders | Fig. 1C, Fig. 2 (CEBRA column) |
| `internals.ipynb` | Twin in encode mode: rate maps, depth profiles and CA1/CA3, attribution (Jacobian), Mantel and MRM tests | Fig. 4, ED Fig. 10 |
| `preprocess-data-v0j-share.ipynb` | Single-layer `Bat_Transformer` encoder analyses (dataloader, rate maps, held-out spatial information, metrics); `internals.ipynb` is the Twin counterpart | — |
| `investigate_similar_trajectories.ipynb` | Finds flight-cluster pairs that share segments or endpoints but diverge in path | Fig. 3, ED Figs. 8–9 |
| `latent_motifs.ipynb` | Potts / HMM / rSLDS segmentation of Twin latent trajectories into motifs | Fig. 2, ED Fig. 4 (related) |
| `foundation_transfer.ipynb` | Scaling (number of training sessions) and transfer with the Twin | Fig. 1H, ED Fig. 3B |
| `perturbations.ipynb` | Context splicing, temporal receptive field, train-time context length | ED Fig. 3A (related) |
| `pv_decode.ipynb` | Perceiver-IO decoder of whole trajectories from one population vector | — |
| `neurons_across_envs.ipynb` | Same units across environments (Krishna sessions): readout similarity vs. rate-map correlation | — |
| `population_raster.ipynb` | Population raster and position traces | Schematic material |
| `grosmark.ipynb`, `grosmark_smoke.ipynb` | Rat CA1 linear-track comparison dataset (Grosmark & Buzsáki 2016, CRCNS hc-11) | — |
| `synthetic-rate-maps.ipynb` | Sanity check of the synthetic place-cell spike generator | ED Fig. 2 (related) |
| `sanity-check-behavior.ipynb`, `check_acceleration_artifacts.ipynb`, `cluster_ttl_times.ipynb` | Behavior QC: value ranges, acceleration artifacts, TTL-aligned video lookup | — |

---

## The `digitalbat` package

| Module | Contents |
|---|---|
| `datasets/data.py` | `Data_Prep_for_Raw` (load → bin → cluster flights → split), `bat_data_loader`, `PlaceCellSimulator3D` |
| `datasets/defaults.py` | `DEFAULT_SESSION_NAMES` (the paper's 17 sessions), `make_data_config()` |
| `datasets/kinematics.py` | Kinematic-augmented behavior and `make_joint_dataloaders` (used by `figures.ipynb`) |
| `datasets/synthetic.py` | Resampled training spikes for the surrogate controls (ED Fig. 2) |
| `models/joint.py` | **`JointMaskedTransformer`** (the Twin), masking `SCHEMES`, `JointEncoderAdapter`, `encode_eval_records` |
| `models/former.py`, `former_defaults.py` | Dual-branch `Bat_Transformer` encoder (place RBF branch + RoPE trajectory branch) and its defaults |
| `models/layers.py` | `TransformerEncoderRoPE` (full / causal / BigBird attention), RBF basis, losses |
| `models/glm.py`, `mlp.py` | GLM family (cosine bases, spike-history and coupling filters) and MLP baselines |
| `models/decoder.py`, `perceiver.py` | Transformer position decoder; Perceiver-IO trajectory decoder |
| `training/trainer.py`, `joint_trainer.py` | Trainers; `ensure_joint_model` (Optuna search + train-if-missing for the Twin) |
| `decoding/` | Naïve-Bayes pipelines, PV-Perceiver pipeline, ED Fig. 2 `cost_pipeline`, Fréchet / jerk geometry, trajectory plots |
| `utility/utils.py` | Metrics (`corr`, `poisson_fde_population`), `FlightClusters`, `frechet_distance`, `bat_palette`, seeding |
| `utility/interpreter.py`, `TrajectoryRF.py` | Model-interpretation and trajectory receptive-field helpers |

[`CLAUDE.md`](CLAUDE.md) has an API cheat-sheet: the data-config keys, the batch
namedtuple fields, the forward-pass signature and the evaluation loop.

---

## Training drivers (`training/`)

These are standalone, resumable scripts for jobs too heavy to run inside a notebook. They
chdir into `notebooks/`, so results land in `notebooks/out/`. They skip checkpoints that
already exist, and `figures.ipynb` calls the ED Fig. 3 drivers automatically if their
result tables are missing.

| Script | Produces |
|---|---|
| `train_joint_context_cross1000.py` | ED Fig. 3A: context-window sweep, every window tuned and trained under the same protocol |
| `train_joint_scaling.py` | ED Fig. 3B: number-of-sessions scaling (encoding metrics) |
| `scaling_decode_fr.py` | ED Fig. 3B: number-of-sessions scaling (decoding Fréchet distance) |
| `train_joint_transfer.py` | Fig. 1G–H: leave-one-session-out transfer (`--mode env` for the environment variant) |
| `train_joint_context.py` | Context-window sweep at fixed hyperparameters (500 epochs); not used for the paper figures |

Run any of them from the repo root, for example:

```bash
python training/train_joint_scaling.py
```

---

## License

See [LICENSE](LICENSE).
