# digitalbat

**A transformer-based functional digital twin of the bat hippocampus.**

`digitalbat` models CA1/CA3 single-unit activity recorded with Neuropixels from freely
flying Egyptian fruit bats. One masked encoder–decoder transformer (the **Twin**)
predicts spiking from 3-D flight behaviour (*encoding*), and recovers the flight path
from spiking (*decoding*). It does both with a single set of weights and one latent space shared across
17 sessions, 6 bats and 2 environments. This repository contains the model package,
the training drivers and the analysis notebooks behind every figure in the paper.

> **If you are here for the paper figures, go straight to
> [`notebooks/figures.ipynb`](notebooks/figures.ipynb)** and the
> [figure map](#the-figure-notebook-notebooksfiguresipynb) below. The figure version in
> the paper is tagged **`paper-figures-v1`**.

---

## Contents

- [Repository layout](#repository-layout)
- [Getting started](#getting-started)
- [The figure notebook (`notebooks/figures.ipynb`)](#the-figure-notebook-notebooksfiguresipynb)
  - [How the notebook is organised](#how-the-notebook-is-organised)
  - [Figure map](#figure-map)
  - [Where outputs go](#where-outputs-go)
  - [Caches, checkpoints and train-if-missing](#caches-checkpoints-and-train-if-missing)
  - [Inputs that are *not* regenerated automatically](#inputs-that-are-not-regenerated-automatically)
  - [Versions, figure numbering and history](#versions-figure-numbering-and-history)
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
│   ├── figures.ipynb           # ★ every main and Extended Data figure in the paper
│   ├── plotting_helpers.py     #   publication style, palettes, stats brackets, save_fig()
│   ├── notebook_helpers.py     #   reward/perch locations, Mantel and MRM tests
│   ├── machine_paths.py        #   one-toggle data/code path profiles (edit this first)
│   ├── motif_helpers.py        #   latent-motif segmentation (latent_motifs.ipynb)
│   ├── null_distribution_training.py
│   ├── *.ipynb                 #   source / exploratory analyses (see "Other notebooks")
│   └── out/                    #   caches, checkpoints and exported figures (git-ignored*)
├── training/                   # standalone, resumable training drivers (ED Fig 3, transfer)
├── CLAUDE.md                   # architecture notes and API cheat-sheet
└── LICENSE
```

\* `notebooks/out/` is in `.gitignore`. The only files under it in git are the
unified-CEBRA checkpoint (`out/cebra/…pt`) and the Twin Optuna study
(`out/joint/joint_optuna_schemewt.pkl`).

---

## Getting started

### 1. Environment

There is no `setup.py` or requirements file, and the package is not installed. The
notebooks put the repo root on `sys.path` themselves (via `machine_paths.setup()`).
These packages are needed:

| Purpose | Packages |
|---|---|
| Core | `python ≥ 3.10`, `torch`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`, `joblib` |
| `figures.ipynb` extras | `optuna`, `cebra`, `POT` (imported as `ot`), `statsmodels` |
| Package extras (some modules) | `numba`, `shap`, `transformers` |
| `latent_motifs.ipynb` only | `dynamax`, `jax` |

A CUDA GPU is used if one is available (`device = 'cuda' if torch.cuda.is_available()`).
Re-running the notebook from cached checkpoints works on CPU. Retraining models from
scratch is only practical on a GPU.

### 2. Point the code at the data

Everything that touches disk goes through
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
Datasets 1–3: 13 indoor-room sessions and 4 outdoor-enclosure (`Dataset_2_*`) sessions.
The Krishna sessions (`KRISHNA_SESSION_NAMES`) are excluded from the paper analyses.

### 3. Run

```bash
git checkout paper-figures-v1   # optional: the exact figure version used in the paper
cd notebooks                    # the notebooks assume cwd = notebooks/ (all paths are ./out/...)
jupyter lab figures.ipynb
```

Run **Shared setup** (cells 1–8) first. After that, each figure section can be run on
its own, with the dependencies [listed below](#figure-map).

---

## The figure notebook (`notebooks/figures.ipynb`)

`figures.ipynb` is the single source for every figure and Extended Data (ED) figure in
the paper. It loads the data **once**, builds the **one consensus Twin model** once, and
every panel reuses those objects:

```
out/joint/joint_v2_cross1000_kin0_L4_d64_ff64_lam0.68_pw0.36_identity_we2.8_wd2.2_wa0.31_n17.pth
```

That checkpoint is a 4-layer, 64-d latent `JointMaskedTransformer`, trained for 1000
epochs on the encode + decode + cross masking schemes. Its training recipe is in
`joint-encode-decode.ipynb` §6.10.

Most figure code was ported here from the exploratory notebooks (see
[Other notebooks](#other-notebooks)). Those notebooks are kept for provenance, but
**`figures.ipynb` is authoritative**: if the two disagree, the version in
`figures.ipynb` is the one in the paper.

### How the notebook is organised

Each figure is one section, and every section follows the same pattern:

1. a markdown **header** listing the panels;
2. one or more **data cells** that compute (or load from cache) everything the figure needs;
3. one **canvas cell** (cell ID usually `*_merged`) that lays out all the panels and calls `ph.save_fig(...)`;
4. a markdown **caption** with the paper's figure legend, including the exact statistics.

Sections are **not in figure-number order**. The order follows the data dependencies:
ED Fig 4 defines the motifs that Fig 2 uses, so it comes before Fig 2. The table below
lists sections in notebook order. Cell indices refer to `paper-figures-v1`. Cell IDs
are stable across edits, so search for the ID if the indices have moved.

### Figure map

#### Shared setup — run first

| Cells | Cell ID | What it does |
|---|---|---|
| 1 | `placeholder` | Imports, `machine_paths` setup, `ph.set_pub_style()`, `FIG_DIR = out/figures`, global seed = 1, `OUR_MODEL = 'Twin'` |
| 3 | `215a54ff` | Loads the 17 default sessions **once** with `make_joint_dataloaders` (position-only, `behavior_dim = 3`) |
| 4 | `descstats` | Dataset descriptive statistics (neuron / session / bat / environment / flight counts for Methods) |
| 5 | `de9a3bd3` | Instantiates the consensus Twin (`JOINT_CKPT`). If the checkpoint is missing it runs a 200-trial Optuna search and trains the winner (`training.joint_trainer.ensure_joint_model`). |
| 6 | `63e8be8c` | `JointEncoderAdapter` and the per-neuron encode-metric helper |
| 8 | `4c999546` | **Load-or-train utilities** for every baseline model (GLMs, NB, …). `FORCE_RETRAIN = False` |

#### Main and Extended Data figures

| Figure | Panels / topic | Cells (IDs) | Saved as | Notes |
|---|---|---|---|---|
| **Fig 1** — the Twin outperforms standard approaches | **b** example decoded flight · **c** decoding (3-D Fréchet): NB (place) vs unified CEBRA vs Twin · **d** dual-field example unit · **e–f** encoding FDE / Pearson r: GLM (place), GLM (trajectory-specific place), Twin · **i** leave-one-session-out transfer | 10 `21dbef05` decode arm · 11 `b64769bb` (1b) · 12 `e9128f87` NB · 13 `cebra_grid_search` · 14 `6d778b26` unified CEBRA · 15 `9fb97b5c` (1c) · 16 `607274c7` (1d) · 17 `a91b7f03` (1e/f) · **18 `fig1_merged`** · 19 caption | `fig1_full` (also `cebra_hparam_optuna`, a CEBRA HP-search diagnostic) | **a**, **g–h** are external schematics |
| **ED Fig 1** — full model ladders | **d** 9-decoder ladder · **e–h** every encoding model (FDE and r) | 21 `973ed220` encoders · 22 `lstm_rnn_search` (Tampuu 2019 LSTM) · 23 `cebra_multi_search` · 24 `02cfa61a` decoder ladder · **25 `edfig1_merged`** · 26 caption | `edfig1_full` | **a–c** external schematics. Needs cell 17. |
| **ED Fig 2** — surrogate-trained controls | **a–c** decode cost (trial-shuffle, synthetic place, synthetic trajectory) · **d–e** encode cost for the ladder retrained on synthetic spikes | 28 `edfig2_costdata` · 29 `edfig2_encode` · **30 `edfig2_merged`** · 31 caption | `edfig2_full` | Decode costs self-generate via `digitalbat.decoding.cost_pipeline` |
| **ED Fig 3** — scaling context window and training data | context-window sweep and #-sessions scaling × {Fréchet, FDE, r} | 33 `edfig3_ensure` · 34 `12eac58c` · 35 `7ad46d86` · **36 `edfig5_merged`** · 37 caption | `edfig3_full` | Cell 33 runs the [`training/`](#training-drivers-training) drivers if their result pickles are missing (**heavy**). The canvas cell ID still says `edfig5` from an earlier numbering. |
| *(shared)* latent extraction | per-timepoint encode/decode Twin latents and the Fig 2c RSA data | 38 `daa02161` · 39 `5fcfd162` | — | Used by Fig 2, ED Fig 4, ED Fig 5 and Fig 4 |
| **ED Fig 4** — principal flight axis defines motifs | per-room flight axis, orientation tensors, two-motif segmentation | **41 `edfig4_motif_axis`** · 42 caption | `edfig4_motif_axis` | **Defines the motifs used by Fig 2 and Fig 3**, so run it before them |
| **Fig 2** — Twin latents warp physical space | Physical XYZ / unified CEBRA / Twin decode / Twin encode, by motif: 3-D trajectories, motif-segment RSA, route decoding, per-motif DMD | 44 `fig2m_motifs` · 45 `fig2m_rsa` · 46 `fig2_dmd_data` · **47 `fd1df114`** · 48 caption | `fig2_embeddings` (`.pdf/.png/.svg`) | |
| **ED Fig 5** — encode/decode latent and readout subspace geometry | shared-PCA of encode vs decode latents, principal angles, participation ratio, readout subspace energy | **50 `edfig5_fig2conv`** · 51 caption | `edfig5_subspaces` | |
| **Fig 3** — same route, different paths | four geometric paths of one reward→reward route; physical vs latent rate maps; place-cell sequences on route progress; PV and peak-order tests; rate-map stability (normalised Wasserstein distance) | 53 `07756d6f` raw-spike cache and pair search · 54 `fig3_ratemap_sim` · 55 `23e9fd4f` analysis core · 56 `9e5b5918` curated pairs · **57 `fig3_multipath`** · 58 `fig3-panelb-standalone` · 59 caption | `fig3_multipath_unit<N>`, `fig3_panelB` | Loads **raw** spike times with `Data_Prep_for_Raw` (separate from the binned loaders) |
| **ED Fig 6** — azimuth control | the Fig 3 sequence on heading (azimuth) axes; circular rank correlation | **61 `edfig5_azimuth`** · 62 caption | `edfig6_azimuth` | Needs cells 53–56 |
| **ED Fig 9** — more same-route/different-path examples | Fig 3 rows for the other curated sessions | **64 `65a76f1f`** · 65 `edfig9-panelb-standalone` · 66 caption | `edfig9_route_example_<session>`, `edfig9_panelB_<session>` | Needs cells 53–56 |
| **Fig 4** — readout weights capture biological structure | **a** readout PC1 vs FR · **b** residual-readout clustering · **c** CV spatial information per cluster · **d–e** neural→position attribution (Jacobian) · **f** sleep-matched example matrices · **g–i** MRM R² vs non-flight rest / sleep · **j** CA1 vs CA3 SVM · **k** MRM within CA1 vs CA3 | 68 `5b32c185` readouts and subregion · 69 `fig4_cluster` · 70 `fig4_attribution` · 71 `0715f686` rate maps, non-flight, coupling · 72 `059c3ab5` sleep matching (UnitMatch) · 73 `fig4_stats` · 74 `7ab4c277` · 75 `fig4_mrm_r2` · **76 `fig4_merged`** · 77 caption | `fig4_full` (also `fig4_attr_index_ca1_ca3`) | Needs `UnitMatch/` and `Dataset_0` sleep sessions under `data_root` |
| **ED Fig 10** — readout-cluster comparisons | orthogonality vs shuffle, cluster composition, per-cluster FR / FDE / %CA1 / r, within vs across cluster | **79 `e82f0a32`** · 80 caption | `edfig10_full` | Needs the Fig 4 data cells (68–75) |
| **ED Table 1** — auxiliary masking-task sweep | 16 combinations of {within, cross, forecast, forecast_stim} on top of encode + decode | 81 header · **82 `ed_table1_code`** | `out/figures/ed_table1_scheme_sweep.csv` | Reads `out/joint/joint_scheme_sweep.pkl` (see below) |

### Where outputs go

`ph.save_fig(fig, name)` (in [`plotting_helpers.py`](notebooks/plotting_helpers.py))
writes each figure at 300 dpi to

```
notebooks/out/figures/pdf/<name>.pdf
notebooks/out/figures/png/<name>.png
notebooks/out/figures/svg/<name>.svg     # Fig 2 only (exts=('pdf','png','svg'))
```

By default it also capitalises the first letter of titles, axis labels and legends;
`capitalize=False` turns this off. The legend text for each figure is the markdown
caption cell that follows its canvas cell.

### Caches, checkpoints and train-if-missing

The notebook is meant to run end-to-end from an empty `out/`. Every model loads its
checkpoint if one exists and **trains it otherwise**, then saves it. Re-runs are
therefore fast, and a fresh clone can still reproduce everything given enough GPU time.

| Location (under `notebooks/out/`) | Contents |
|---|---|
| `joint/` | Consensus Twin checkpoint, Twin Optuna study, ED Table 1 sweep |
| `bat_glm/` | GLM family (place, trajectory-specific, + history / coupling) |
| `cebra/` | Unified and multi-session CEBRA models and Optuna studies |
| `decode_costs/` | ED Fig 2 per-flight surrogate decode costs |
| `fig3_cache/` | Fig 3 raw-spike, rate-map-similarity and bin-free caches |
| `transfer/`, `transfer_joint/` | ED Fig 3 scaling and context-window results |
| `figures/` | Exported figures (see above) |

Useful switches:

- `FORCE_RETRAIN = True` (cell 8) ignores cached baseline checkpoints.
- `LSTM_N_TRIALS` and `MS_N_TRIALS` (cells 22–23): raise these to extend an Optuna study instead of reloading it.
- Several caches (Fig 3, Fig 4) are keyed on a hash of the checkpoint and parameters, so they rebuild automatically when an input changes.

### Inputs that are *not* regenerated automatically

| Needed by | Input | Where it comes from |
|---|---|---|
| Everything | `Dataset_*` session folders | `data_root` in `machine_paths.py` |
| Fig 4 f, h (sleep) | `UnitMatch/` tables and `Dataset_0_*` sleep sessions | `data_root` |
| ED Table 1 | `out/joint/joint_scheme_sweep.pkl` | `joint-encode-decode.ipynb` §6.8 |
| Fig 1 **a, g–h**; ED Fig 1 **a–c** | Schematics | Drawn separately, not in this repo |

### Versions, figure numbering and history

| Tag | What it marks |
|---|---|
| **`paper-figures-v1`** | The version of `figures.ipynb` used for the figures currently in the paper |
| `fig5-swr-supp-v1` | Archive of the removed SWR figure (old Fig 5 / ED Fig 7); see below |

- **The ED figure numbering has gaps on purpose.** ED Figs 7 and 8 are not produced by
  `figures.ipynb`. The SWR figures (old Fig 5 and old ED Fig 7) were removed in
  `194022b`, and what used to be ED Figs 7 and 8 were renumbered to **9 and 10** in
  `587b7fa`. Some cell IDs (`edfig5_merged`, `edfig5_azimuth`, …) still carry their
  pre-rename numbers. Match figures by **section header and saved filename**, not by
  cell ID.
- **Archived SWR analysis.** The full removed Fig 5 / ED Fig 7 block (SWR timing,
  endpoint and SWR-rate LMEs, SWR-coloured latent trajectories) is preserved at
  `fig5-swr-supp-v1` (commit `1aa0b94`):
  ```bash
  git show fig5-swr-supp-v1:notebooks/figures.ipynb > figures_fig5_archive.ipynb
  ```
  The live SWR analyses continue in `swr.ipynb` and `explore_swr.ipynb`.

---

## Other notebooks

These notebooks are where each analysis was developed. Most of their figure-relevant
code has been ported into `figures.ipynb`, but they remain the reference for model
selection, ablations and exploration.

| Notebook | What it is | Feeds |
|---|---|---|
| `joint-encode-decode.ipynb` | Develops the joint masked encoder–decoder (the Twin): masking schemes, trainer, Optuna searches, consensus model (§6.10), auxiliary-task sweep (§6.8) | Consensus checkpoint, ED Table 1 |
| `model_comparison.ipynb` | Encoder ladder: GLM variants (lags, kinematics, history, coupling), MLP, `Bat_Transformer`, Twin | Fig 1e–f, ED Fig 1e–h |
| `decode-position-comparison.ipynb` | Decoder ladder: Naive Bayes (± continuity, ± lag), transformer decoder, Twin decode arm, CEBRA, PV-Perceiver; surrogate decode costs | Fig 1c, ED Fig 1d, ED Fig 2a–c |
| `cebra-comparison.ipynb` | CEBRA embeddings and decoding vs the transformer decoders | Fig 1c, Fig 2 (CEBRA column) |
| `internals.ipynb` | Twin in encode mode: rate maps, depth profiles and CA1/CA3, attribution (Jacobian), Mantel and MRM tests | Fig 4, ED Fig 10 |
| `preprocess-data-v0j-share.ipynb` | Original `Bat_Transformer` analysis notebook (dataloader, rate maps, held-out spatial info, metrics); `internals.ipynb` is its Twin clone | Historical |
| `investigate_similar_trajectories.ipynb` | Finds flight-cluster pairs that share segments or endpoints but diverge in path | Fig 3, ED Fig 6, ED Fig 9 |
| `latent_motifs.ipynb` | Potts / HMM / rSLDS segmentation of Twin latent trajectories into motifs | Motif definition (ED Fig 4, Fig 2) |
| `foundation_transfer.ipynb` | Scaling (# training sessions) and foundation-model transfer with the Twin | Fig 1i, ED Fig 3 |
| `perturbations.ipynb` | Context splicing, temporal receptive field, train-time context length | ED Fig 3 (context), exploration |
| `pv_decode.ipynb` | Perceiver-IO decoder of whole trajectories from one population vector | Exploration |
| `neurons_across_envs.ipynb` | Same units across environments (Krishna sessions): readout similarity vs rate-map correlation | Exploration |
| `swr.ipynb`, `explore_swr.ipynb` | Sharp-wave-ripple timing around flights; re-extracted SWR detector (v2) | Removed Fig 5 / SWR work |
| `joint-figures.ipynb`, `joint-figures-single-unit.ipynb` | Earlier joint-model figure notebooks (population latent; single-unit place fields) | Superseded by `figures.ipynb` |
| `population_raster.ipynb` | Population raster and position traces for schematics | Schematic material |
| `grosmark.ipynb`, `grosmark_smoke.ipynb` | Rat CA1 linear-track comparison dataset (Grosmark & Buzsáki 2016, CRCNS hc-11) | Exploration |
| `synthetic-rate-maps.ipynb` | Sanity check of the synthetic place-cell spike generator | ED Fig 2 controls |
| `sanity-check-behavior.ipynb`, `check_acceleration_artifacts.ipynb`, `cluster_ttl_times.ipynb` | Behaviour QC: value ranges, acceleration artifacts, TTL-aligned video lookup | QC |

---

## The `digitalbat` package

| Module | Contents |
|---|---|
| `datasets/data.py` | `Data_Prep_for_Raw` (load → bin → cluster flights → split), `bat_data_loader`, `PlaceCellSimulator3D` |
| `datasets/defaults.py` | `DEFAULT_SESSION_NAMES` (the paper's 17 sessions), `KRISHNA_SESSION_NAMES`, `make_data_config()` |
| `datasets/kinematics.py` | Kinematic-augmented behaviour and `make_joint_dataloaders` (used by `figures.ipynb`) |
| `datasets/synthetic.py` | Poisson-resampled training spikes for the surrogate controls (ED Fig 2) |
| `models/joint.py` | **`JointMaskedTransformer`** (the Twin), masking `SCHEMES`, `JointEncoderAdapter`, `encode_eval_records` |
| `models/former.py`, `former_defaults.py` | Original dual-branch `Bat_Transformer` encoder (place RBF branch + RoPE trajectory branch) and its defaults |
| `models/layers.py` | `TransformerEncoderRoPE` (full / causal / BigBird attention), RBF basis, losses |
| `models/glm.py`, `mlp.py` | GLM family (cosine bases, spike-history and coupling filters) and MLP baselines |
| `models/decoder.py`, `perceiver.py` | Transformer position decoder; Perceiver-IO trajectory decoder |
| `training/trainer.py`, `joint_trainer.py` | Trainers; `ensure_joint_model` (Optuna search + train-if-missing for the Twin) |
| `decoding/` | Naive-Bayes pipelines, PV-Perceiver pipeline, ED Fig 2 `cost_pipeline`, Fréchet / jerk geometry, trajectory plots |
| `utility/utils.py` | Metrics (`corr`, `poisson_fde_population`), `FlightClusters`, `frechet_distance`, `bat_palette`, seeding |
| `utility/interpreter.py`, `TrajectoryRF.py` | Model-interpretation and trajectory receptive-field helpers |

[`CLAUDE.md`](CLAUDE.md) has an API cheat-sheet: the data-config keys, the batch
namedtuple fields, the forward-pass signature and the evaluation loop.

---

## Training drivers (`training/`)

These are standalone, resumable scripts for the jobs that are too heavy to run inside a
notebook. They chdir into `notebooks/`, so results land in `notebooks/out/`. They skip
checkpoints that already exist, and ED Fig 3 calls them automatically if its result
pickles are missing.

| Script | Produces |
|---|---|
| `train_joint_context_cross1000.py` | ED Fig 3 context-window sweep under the production recipe (supersedes `train_joint_context.py`) |
| `train_joint_scaling.py` | ED Fig 3 #-sessions scaling (encode metrics) |
| `scaling_decode_fr.py` | ED Fig 3 scaling (decode Fréchet) |
| `train_joint_transfer.py` | Leave-one-out foundation transfer (`--mode env` for ED Fig 3 environment transfer) |
| `train_joint_context.py` | Earlier context sweep (fixed HPs, 500 epochs); kept for reference |

Run any of them from the repo root, for example:

```bash
python training/train_joint_scaling.py
```

---

## License

See [LICENSE](LICENSE).
