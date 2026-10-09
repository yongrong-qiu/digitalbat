# digitalbat

Code for **[A hippocampal digital Twin reveals latent structure in freely flying bats](PAPER_URL)**
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

## Reproducing the figures

Every figure (Figs. 1–4, Extended Data Figs. 1–10) and Supplementary Table S1 is produced by
[`notebooks/figures.ipynb`](notebooks/figures.ipynb), which keeps its rendered outputs. The
tag **`paper-figures-v1`** marks the version used for the paper.

1. **Data.** The recordings were reported by Forli et al. and are available from the
   corresponding author on reasonable request. Set `data_root` in
   [`notebooks/machine_paths.py`](notebooks/machine_paths.py) to the folder that holds the
   `Dataset_*` sessions (and `UnitMatch/` for the sleep analyses).
2. **Environment.** Python with `torch`, `numpy`, `scipy`, `pandas`, `scikit-learn`,
   `matplotlib`, `seaborn`, `optuna`, `cebra`, `POT` and `statsmodels`. The package is used
   from the repo directly, without installation. A GPU is needed only to retrain models.
3. **Run** from `notebooks/`: the *Shared setup* cells first, then any figure section.
   Models load from `notebooks/out/` when cached and are trained otherwise. Figures are
   written to `notebooks/out/figures/{pdf,png}/`.

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
| `notebooks/figures.ipynb` | All paper figures |
| `training/` | Long-running jobs for Fig. 1H and Extended Data Fig. 3 |

## License

See [LICENSE](LICENSE).
