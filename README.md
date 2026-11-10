# A hippocampal digital Twin reveals latent structure in freely flying bats

Code for **A hippocampal digital Twin reveals latent structure in freely flying bats**
(Lowet\*, Qiu\*, Forli, Kasuba, Lin, Tolias†, Yartsev†).

We trained a multimodal transformer, the digital **Twin**, jointly on hippocampal activity and
3-D flight in freely flying Egyptian fruit bats (3,080 neurons, 17 sessions, 6 bats, indoor
and outdoor). One model predicts neural activity from behavior and decodes behavior from
neural activity. It outperforms standard approaches and transfers to held-out sessions.
Interrogating the Twin then generated hypotheses that we confirmed in the recorded data:

- **Latent space:** neurons fire at corresponding stages of spatially distinct paths that share
  an origin and destination, preserving their firing order even as their place fields move.
- **Readout weights:** functional relationships among neurons learned during flight generalize
  to waking rest and sleep, and differ between CA1 and CA3.

Methods, statistics and figure legends are in the paper.

## System requirements

- **Operating system:** tested on Windows 11 Pro (build 26200). The code is pure Python, with no
  OS-specific calls, and the analyses were developed on Ubuntu Linux.
- **Software:** Python 3.13 with the exact versions in [`requirements.txt`](requirements.txt), which
  are the versions tested. Key packages: PyTorch 2.7.1 (CUDA 12.6 and CPU builds), NumPy 2.4.6,
  SciPy 1.18.0, pandas 2.2.3, scikit-learn 1.6.1, CEBRA 0.6.1, Optuna 4.8.0, POT 0.9.7,
  matplotlib 3.11.1, seaborn 0.13.2 and JupyterLab 4.6.3.
- **Hardware:** no non-standard hardware is required. A CUDA GPU is optional: the demo retrains its
  models by default only when one is present (tested with an NVIDIA RTX 3090), and otherwise runs
  CPU-only on the paper's models. The demo peaks at about 8 GB of RAM, so 16 GB is enough.

## Installation

```bash
git clone https://github.com/yongrong-qiu/digitalbat.git
cd digitalbat
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu   # NVIDIA GPU: .../whl/cu126
pip install -r requirements.txt
pip install --no-deps cebra==0.6.1
```

`cebra` is installed without its dependency pins because on Windows they require `numpy<2`; it
runs fine on NumPy 2.4. On Windows, clone to a short path such as `C:\src\digitalbat` (or enable
long paths), because some checkpoint names under `notebooks/out/` are long. A fresh install took
3 min on our test computer (CPU build of PyTorch, fast network connection). The CUDA build adds a
download of about 2.5 GB.

## Demo

[`notebooks/demo.ipynb`](notebooks/demo.ipynb) uses one public session,
`Dataset_1_32622_231009`. It runs the paper's code for Fig. 1B–F, Fig. 3 and Extended Data Fig. 8
on that session. `MODELS` in the notebook chooses where the models (the Twin, two Poisson GLMs, a
naïve-Bayes decoder and CEBRA) come from:

- **`'paper'`** (default without a GPU): the paper's models, trained on all 17 sessions and
  committed under `notebooks/out/`. Nothing is trained, and **Fig. 3A–L reproduce the paper
  exactly**.
- **`'retrain'`** (default with a CUDA GPU): retrains the Twin and the Fig. 1 baselines on that
  session. Please note that this produces different and underpowered Twin (and unified CEBRA) models compared to the full dataset. We include it here just to illustrate how training works. All published numbers run through `notebooks/figures.ipynb`.
- **`'cached'`:** the single-session models that `'retrain'` produced on our GPU, committed in
  `notebooks/out/demo_weights/`.

The notebook prints which models a run uses. Everything after loading or training the models is
computed in every mode.

1. **Data.** Download the session from figshare
   ([doi:10.6084/m9.figshare.34805709](https://doi.org/10.6084/m9.figshare.34805709)). The code
   reads only these five files (81 MB of the download). The rest is included for completeness:

   ```
   <DATA_DIR>/
     Dataset_1_32622_231009/Extracted_Behavior_231009.mat
     Dataset_1_32622_231009/IMU_data.mat
     Dataset_1_32622_231009/merges_Dataset_1_32622_231009.mat
     To_transfer/out_Gunits_probe1_Dataset_1_32622_231009_with_metrics.mat
     To_transfer/out_Gunits_probe2_Dataset_1_32622_231009_with_metrics.mat
   ```

   Put them in `data/` at the repo root (git ignores it), or anywhere else and set the
   environment variable `DIGITALBAT_DEMO_DATA` to that folder.
2. **Run** from `notebooks/`: open `demo.ipynb` in JupyterLab and run all cells, or run it
   headless with `jupyter nbconvert --to notebook --execute demo.ipynb --output demo_run.ipynb`.

**Expected output.** The committed notebook keeps the rendered output of a CPU-only `'paper'` run,
and figures are saved to `notebooks/out/demo/figures/{pdf,png}/`.

- **Exact in every mode:** Fig. 3A–I and Extended Data Fig. 8 depend only on the recorded data, so
  they match the paper exactly. That means 47 place cells, and a place-cell sequence preserved
  across paths (Spearman ρ = 0.909 between Path 1 training and held-out flights, 0.534 for Path 2,
  0.543 for Path 3; all *P* = 5.00 × 10⁻⁴).
- **Exact with `'paper'`:** Fig. 3J–L, every statistic included (for example panel K's
  *P* = 4.4 × 10⁻⁹).
  - **J–K:** use the paper's Twin and the PCA basis the figure was drawn in
    (`notebooks/out/fig3_cache/latent_pca3.npz`, fit to all 17 sessions' test-flight latents).
  - **L:** pools 15 sessions. The notebook recomputes this session's values, checks them against
    the paper's committed results (they agree to 10⁻¹⁴), and takes the other sessions from those
    results.
- **Panel M with `'paper'`:** built the same way as L, on the matching 15-D basis
  (`latent_pca15.npz`). This session's values agree with the committed results to 10⁻⁵, the
  rounding of projecting onto a saved basis. Like `figures.ipynb`, it matches the published
  figure only to within shuffle variation (see [Reproducing the figures](#reproducing-the-figures)).
- **Fig. 1B–F with `'paper'`:** this session's values from the paper's models. The paper's Fig. 1
  pools 17 sessions.
- **Single-session versions** (`'retrain'`, `'cached'`): Fig. 1B–F and Fig. 3J–M come from models
  trained on this one session. On the 13 test flights, the Twin still encodes and decodes best:
  - **Encoding:** median per-neuron Poisson FDE 0.09, against 0.07 and 0.04 for the two GLMs.
  - **Decoding:** median per-flight Fréchet distance 0.50 m, against 1.39 m for CEBRA and 2.53 m
    for naïve Bayes.
  - **Tests:** per-unit Wilcoxon *P* ≤ 0.004 for every comparison.

  GPU retraining is not bit-for-bit deterministic, so these numbers vary slightly between runs.
  With a Twin trained on only one session, however, this session's latent rate maps are no more
  stable across paths than its physical ones. Panel K gives Wilcoxon *P* = 0.3–0.6 across our runs,
  against 4.4 × 10⁻⁹ with the 17-session Twin.

**Expected run time** on our test computer (Intel Core i9-13900K, NVIDIA RTX 3090):

- **CPU-only, `'paper'` or `'cached'`:** 4 min with all 24 cores, and 5 min with 8 logical
  processors.
- **`'retrain'` on the GPU:** 7 min.

**What the public session cannot reproduce.** From this repository and the public session alone:

- **Exact:** Fig. 3 (A–I in every mode; J–L with `'paper'`) and Extended Data Fig. 8. Panel M
  matches with `'paper'` to within shuffle variation.
- **This session's values only:** Fig. 1B–F.
- **Not reproducible:** Fig. 1H, Figs. 2 and 4, Extended Data Figs. 1–7, 9 and 10, and Table S1.
  These aggregate over all 17 sessions, or draw examples from sessions that are not public.

Their trained models and results are committed under `notebooks/out/` (see below). The last cell of
`demo.ipynb` gives the reason for each figure.

## Instructions for use

To run the software on your own recordings, store each session in the demo's layout under one
folder:

| File | Contents (MATLAB `.mat`) |
|---|---|
| `<session>/Extracted_Behavior_<date>.mat` | `r` (T × 3 position, m), `t` (time stamps, s), `Fs` (Hz), `bflying` (T × 1 flight flag), `v_abs` (speed), `v` (velocity), `r_lim` (arena bounds) |
| `<session>/IMU_data.mat` | `NP_imu` struct with `acc`, `gyr` and `t` |
| `<session>/merges_<session>.mat` | `merges`: unit IDs to merge, per probe (may be empty) |
| `To_transfer/out_Gunits_probe<k>_<session>_with_metrics.mat` | `out_Gunits` struct with `cluster_id`, `spikeTimes_usec`, `pType`, `spikePos_um` |

Session names follow `Dataset_<dataset>_<bat>_<date>`. The dataset prefix selects the arena's
reward and perch locations (`DATASET_TARGETS` in
[`notebooks/notebook_helpers.py`](notebooks/notebook_helpers.py)), so a new arena needs an entry
there. Then, in `demo.ipynb`, point `DATA_DIR` at the folder, list the sessions in `SESSIONS` (one
or more), set `MODELS = 'retrain'` (the paper's models only cover the paper's sessions) and run all
cells. Every model is trained on the listed sessions. With several sessions,
the Twin shares one transformer across them, with a spike read-in and read-out per session. For
Fig. 3, set `MP4_SESSION`, `MP4_PATHS` and `MP4_UNITS` in cell `fig3_multipath` to one of the
same-route path pairs that the Fig. 3 data cell lists in `df_altpath`.

## Reproducing the figures

Every figure (Figs. 1–4, Extended Data Figs. 1–10) and Supplementary Table S1 is produced by
[`notebooks/figures.ipynb`](notebooks/figures.ipynb), which keeps its rendered outputs. The
tag **`paper-figures-v1`** marks the version used for the paper.

1. **Data.** The recordings were reported by Forli et al. and are available from the
   corresponding author on reasonable request (one session is public; see [Demo](#demo)). Set
   `data_root` in [`notebooks/machine_paths.py`](notebooks/machine_paths.py) to the folder that
   holds the `Dataset_*` sessions (and `UnitMatch/` for the sleep analyses).
2. **Environment.** See [Installation](#installation). A GPU is needed only to retrain models.
3. **Run** from `notebooks/`: the *Shared setup* cells first, then any figure section.
   Every checkpoint, Optuna study and cache the notebook reads is committed under
   `notebooks/out/` (about 230 MB): these are the paper's models, trained on all 17 sessions,
   so a fresh clone reproduces every figure without retraining or recomputing. Anything missing
   from `notebooks/out/` is trained or recomputed. Figures are written to
   `notebooks/out/figures/{pdf,png}/`.

Fig. 3M is the one panel that reproduces only to within shuffle variation. Its z-scores compare
each neuron's rate-map stability with 100 random circular shifts of its spikes. The published
panel comes from an earlier run whose shuffles cannot be regenerated, so the notebook draws new
ones. The notebook's per-route median z-scores are −0.95, −0.23 and −0.27, against the published
−0.99, −0.19 and −0.32, with the same session counts and significance.

Each section of the notebook has a header, its data cells, one canvas cell that draws and
saves the figure, and the figure legend from the paper.

| Figure | Canvas cell ID | Saved as |
|---|---|---|
| Fig. 1 | `fig1_merged` | `fig1_full` |
| Fig. 2 | `fd1df114` | `fig2_embeddings` |
| Fig. 3 | `fig3_multipath` | `fig3_multipath_unit<N>` |
| Fig. 4 | `fig4_merged` | `fig4_full` |
| Extended Data Fig. 1 | `edfig1_merged` | `edfig1_full` |
| Extended Data Fig. 2 | `edfig2_merged` | `edfig2_full` |
| Extended Data Fig. 3 | `edfig3_merged` | `edfig3_full` |
| Extended Data Fig. 4 | `edfig4_motif_axis` | `edfig4_motif_axis` |
| Extended Data Fig. 5 | `edfig5_subspaces` | `edfig5_subspaces` |
| Extended Data Fig. 6 | `edfig6_landmark_plot`, `edfig6_kin_plot` | `edfig6_landmarks`, `edfig6_kinematics` |
| Extended Data Fig. 7 | `edfig7_dmd_plot` | `edfig7_dmd` |
| Extended Data Fig. 8 | `edfig8_azimuth` | `edfig8_azimuth` |
| Extended Data Fig. 9 | `65a76f1f` | `edfig9_route_example_<session>` |
| Extended Data Fig. 10 | `e82f0a32` | `edfig10_full` |
| Table S1 | `table_s1_code` | `table_s1_scheme_sweep.{csv,tex}` |

Sections follow data dependencies rather than figure order (Extended Data Fig. 4 defines the
flight motifs used by Figs. 2–3). Schematic panels were drawn separately. The Fig. 1H
transfer results come from `python training/train_joint_transfer.py`, and the Table S1 sweep
is read from `notebooks/out/joint/joint_scheme_sweep.pkl`. The Extended Data Fig. 3 drivers in
`training/` run automatically if their results are missing.

## Repository

| Path | Contents |
|---|---|
| `digitalbat/models/joint.py` | The Twin (`JointMaskedTransformer`) and its masking schemes |
| `digitalbat/datasets/` | Data loading, binning, flight clustering, the paper's 17 sessions (`defaults.py`) |
| `digitalbat/models/`, `digitalbat/decoding/` | Baseline encoders and decoders (GLMs, MLP, Naïve Bayes) |
| `digitalbat/training/` | Trainers, including train-if-missing for the Twin |
| `notebooks/figures.ipynb` | All paper figures, from the 17-session models in `notebooks/out/` |
| `notebooks/demo.ipynb` | Demo on one public session: Fig. 1B–F, Fig. 3 and Extended Data Fig. 8, with the paper's models or retrained ones |
| `training/` | Long-running jobs for Fig. 1H and Extended Data Fig. 3 |
| `pyproject.toml`, `requirements.txt` | Dependencies and optional extras; exact versions used for the paper |

## License

See [LICENSE](LICENSE).
