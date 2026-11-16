# XSymNet

CNN-based classification of powder diffraction patterns into crystallographic
subgroups (from an ISODISTORT subgroup tree), coupled with TOPAS for synthetic
training-data generation and follow-up Rietveld/Pawley refinement.

The pipeline has three stages:

1. **Data preparation (Windows + TOPAS required)** — P1 Pawley pre-processing and
   threaded generation of synthetic training patterns with TOPAS.
2. **Training (Linux GPU machine)** — train the CNN + multi-head attention
   classifier on the generated tensors.
3. **Prediction & refinement** — classify experimental patterns (top-k subgroups
   with logits), then verify the top candidates with real TOPAS Rietveld
   refinement.

---

## Repository layout

```
XsymNet/
├── pixi.toml                  # environment + task definitions (pixi)
├── README.md                  # this file
├── docs/                      # detailed documentation
│   ├── README_CONFIG.md             # config-file system reference
│   ├── README_CONFIG_update_1.md    # config-file update (raw strings etc.)
│   └── README_interactive_plotter.md
├── data_prep/                 # STAGE 1 — TOPAS scripts (Windows + TOPAS license)
│   ├── write_TOPAS_funcs.py         # library: generates all TOPAS input-file sections
│   ├── write_TOPAS_P1.py            # P1 Pawley pre-processing (instrument calibration)
│   ├── write_TOPAS_ML_step_threaded.py  # threaded synthetic training-data generation
│   ├── write_TOPAS_ML_multiphase.py # multiphase refinement inputs
│   ├── write_TOPAS_rietveld.py      # Rietveld refinement of predicted subgroups
│   ├── write_TOPAS_TREE_REFINE.py   # refinement across the subgroup tree
│   ├── write_TOPAS_experimental_pawley.py
│   ├── write_TOPAS_OrderParameter.py
│   ├── write_rietveld_results.py    # harvest refinement results
│   ├── run_pdf_calc.py              # PDF calculation (needs `pdf` pixi env, diffpy)
│   ├── symmetry_checker.py / mismatch_checker.py / crystal_transform.py
│   ├── instrument_parameters/       # instrument/peakshape/macros/pre-xdd library
│   └── local.inc, topasxsymnet.inc  # TOPAS include files
├── xsymnet/                   # core ML package
│   ├── xsymnet_classes.py           # models: CNN_MultiHeadAttention, GradCAM,
│   │                                #   IntegratedGradients, AttentionRollout, Dataset
│   ├── xsymnet_classes_d.py         # model variant (optuna_d)
│   ├── xsymnet_funcs.py             # data loading, scaling, split, top-k/stability metrics
│   └── xgboost_class.py             # GPU XGBoost classifier wrapper
├── training/                  # STAGE 2/3 — training & prediction scripts
│   ├── xsymnet_bragg_seed_input.py  # MAIN entry: config-file + seed driven
│   ├── xsymnet_bragg_seed.py        # seeded, hardcoded inputs (legacy)
│   ├── xsymnet_bragg.py             # original hardcoded script (legacy)
│   ├── xsymnet_optuna.py / xsymnet_optuna_d.py   # Optuna hyperparameter search
│   ├── xgboost_run.py / xsymnet_xgboost.py       # XGBoost baseline
│   └── metrics_on_top10.py          # aggregate top-10 metrics across seeds
├── io_utils/                  # IO / data-file helpers
│   ├── config_reader.py             # parser for training config (.ini/.txt) files
│   ├── utility_functions.py         # tensor conversion, DOF counting, etc.
│   └── convert_* / tof_to_d.py / STANDARDIZE_XY.py / truncate_experimental_data.py /
│       get_peak_count.py
├── plotting/                  # all PLOT_*/PAPER_* figure scripts, interactive_plotter.py,
│                              # tensor viewers (EXAMINE_TENSOR.py, view_tensor.py, ...)
└── diagnostics/               # TEST_* / PROTO_* scratch tests
```

Material data folders (`<material_name>/` with `ml_data/`, `experimental_*`,
`topas_strs_<material_name>/`, ...) stay at the **repository root** and are
git-ignored. All scripts resolve data paths relative to the current working
directory — **always run from the repo root** (pixi tasks do this automatically).

> Note: the IO package is named `io_utils`, not `io`, because a top-level `io/`
> package would shadow Python's standard-library `io` module and break
> torch/pandas.

---

## Environment setup with pixi

### 1. Install pixi

Windows (PowerShell):
```powershell
iwr -useb https://pixi.sh/install.ps1 | iex
```
Linux:
```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

### 2. Create the environment

From the repo root:
```bash
pixi install
```
This solves one lockfile (`pixi.lock`, commit it) covering both platforms:

- **win-64**: Python 3.11, numpy/pandas/scipy/scikit-learn/matplotlib, PyTorch
  (used by the data-prep scripts to pack tensors), tk.
- **linux-64**: everything above plus `pytorch-cuda 12.4`, `cupy`, `dask` and
  `dask-cuda` for GPU training and the multi-GPU XGBoost baseline.

Optional PDF environment (for `data_prep/run_pdf_calc.py`, needs diffpy):
```bash
pixi install -e pdf
```

### 3. Run tasks

pixi tasks run with the repo root as the working directory; extra arguments are
appended to the command:

```bash
pixi run prep-p1                 # python -m data_prep.write_TOPAS_P1
pixi run gen-data                # python -m data_prep.write_TOPAS_ML_step_threaded
pixi run train configs/config_wo3_1043K.txt 48   # training with config + seed
pixi run metrics                 # aggregate top-10 metrics across seeds
pixi run rietveld                # write Rietveld inputs for predicted subgroups
pixi run plotter                 # interactive file plotter
```

See all tasks: `pixi task list`. Equivalent manual form (no pixi task):
`python -m training.xsymnet_bragg_seed_input <CONFIG_FILE> <SEED>` — the `-m`
form is required so the package imports resolve; do not run the scripts by path.

> If your Windows prep machine has no GPU / does not need CUDA, comment out the
> `pytorch-cuda` line in `pixi.toml` to shrink the environment.

---

## Requirements per stage

| Stage | Machine | Requirements |
|---|---|---|
| P1 pre-proc + training-data generation | **Windows only** | **Licensed TOPAS** (e.g. `C:\TOPAS6`/`C:\TOPAS7` with `tc.exe`), pixi env |
| Training + prediction | Linux (multi-)GPU recommended | pixi env, NVIDIA driver for CUDA 12.4 |
| Plotting / utilities | either | pixi env |

TOPAS itself is **not** installed by pixi — it is commercial software you must
install and license separately. The prep scripts only *write* TOPAS input files
and call `tc.exe` (TOPAS command line) through a generated `.bat` file.

---

## Workflow

### Stage 0 — Material folder & subgroup tree

1. Create `<material_name>/` at the repo root, with an experimental-data folder
   (e.g. `expt_data/` — the name is passed to the scripts).
2. Generate the subgroup tree in **ISODISTORT**:
   - TOPAS output files → `<material_name>/topas_strs_<material_name>/`
   - CIFs → `<material_name>/subgroup_cifs_<material_name>/`

### Stage 1 — P1 pre-processing (`data_prep/write_TOPAS_P1.py`)

User inputs at the top of `main()`:

- `material_name` — must match the folder created above.
- `use_strain_modes` — whether strain modes were enabled in ISODISTORT.
- `instrument_file` / `peakshape_file` — files inside
  `data_prep/instrument_parameters/` (the instrument library).
- `use_subgroups` — list of subgroup indices to run as P1 Pawley; check the last
  index of the file list in `topas_strs_<material_name>/` (the tree root is
  typically the P1 structure).
- `use_single_experimental_data` — `True`: refine one file
  (`experimental_data_file`); `False`: batch-refine a whole folder
  (`experimental_data_path` glob).
- `use_custom_macros` / `use_pre_xdd` — pick files from the instrument library
  (macros appended to the input; pre-xdd lines inserted before the XDD section).
- `path_to_topas_directory` — TOPAS install used in the generated `.bat`.

Everything below the "END OF USER INPUT" marker is for power users.

Run `pixi run prep-p1` → generates `.inp` files and a `.bat` in
`<material_name>/batchfiles_topas/`. Double-click the `.bat` (Windows) to run
the Pawley refinements in TOPAS. **Inspect the fits**: lattice parameters, fit
quality, etc. Tweak manually and rerun if needed. Then:

- keep the fitted XY data (it becomes the experimental pattern for prediction), and
- update the instrument files in `data_prep/instrument_parameters/` with the
  fitted instrument parameters & background.

Edge case: if every subgroup member needs a Pawley refinement (e.g. custom
lattice-parameter work), set `use_subgroups` to the full index list, regenerate,
and run the batch for all members before continuing.

### Stage 2 — Training-data generation (`data_prep/write_TOPAS_ML_step_threaded.py`)

User inputs at the top of `main()`:

- `material_name` — same as before. `phase_name` — label for this dataset/problem.
- `instrument_file` — the **updated** instrument file from Stage 1.
- `use_strain_modes` — `False` switches to `custom_latparam_bounds` (direct
  a,b,c,α,β,γ perturbation); symmetry constraints are handled automatically.
  Bounds for strain modes: tiny changes make a BIG difference in lattice
  parameters (default `[-0.002, 0.002]`).
- `beq_dict` — random beq bounds per atom label; check the `.str` files for the
  exact atom labels.
- `optimize_scale = False` and `flag_create_randomstates = True` are deprecated
  (leave as-is).
- `number_of_random_states` — ~250–500 patterns per subgroup for ML training.

Run `pixi run gen-data`. Per subgroup, the threaded workers write
`rs_xxxx.inp` files into `<material_name>/ml_data_generation/<phase_name>/temp*`,
run them through TOPAS terminal mode (`tc.exe`) via a generated batch file,
collect the output `.xy` files, and save the Y-values of all random states as
one tensor: `<material_name>/ml_data/<phase_name>/topas<sg>_<material>_<phase>_rs.pt`.
The first `rs_0001.inp` of each subgroup is copied to
`ml_data_generation/<phase_name>/RS_SGT_INP/` for manual checking; temp folders
are deleted at the end. `generate_mode_lines` (in `write_TOPAS_funcs.py`) parses
the ISODISTORT `.str` files and can be reused standalone for batch fitting.

**Transfer to the GPU machine**: the `<material_name>/ml_data/<phase_name>/*.pt`
tensors and the fitted experimental `.xy` from Stage 1. Make sure the
experimental and training data share the same x-axis start/end points.

### Stage 3 — Training & prediction (`training/xsymnet_bragg_seed_input.py`)

Config-driven; see `docs/README_CONFIG_update_1.md` for the full format.

Layout on the GPU machine (repo root):

```
<material_name>/
├── ml_data/<phase_name>/*.pt             # training tensors from Stage 2
└── experimental_xy/ (or as configured)   # fitted experimental pattern
```

Training config (`.ini`/`.txt`, git-ignored) sets material/phase, epochs, batch
size, `data_start_index` (normally 0) / `data_end_index`, `percent_for_testing`,
`scale` (max value of training data — keep the default), scaling method, model
save path (`SEED` placeholder is replaced by the seed), experimental files glob,
CAM / integrated-gradients / attention-rollout switches, warmstart/pretrained.

- Model architecture parameters (`channel_1 = 16`, `channel_2 = 64`, kernels,
  strides, `num_heads`, `linear_1`) are set in `xsymnet_bragg_seed_input.py`.
- Mini-batch size = `training_batchsize` / number of GPUs.
- Run one seed: `pixi run train configs/config_wo3_1043K.txt 48`
  (or loop over seeds in a shell script, e.g. `run_seed.sh`).

Each run produces, sharing a core name from the config:

- saved model `.pt` (best validation loss, early stopping enabled),
- `zz_<modelname>.txt` — per-seed log: top-5 predicted subgroups with logits,
  DOF counts, validation metrics (top-k accuracy, F1), model structure, config copy,
- optional CAM / integrated-gradients / attention-rollout PNGs overlaid on the
  experimental pattern.

Then aggregate across seeds:
`pixi run metrics` (`training/metrics_on_top10.py`) →
`prediction_stability_top10_*.csv` + validation metrics.

### Stage 4 — Verification refinement

Take the stable predicted subgroups from the prediction-stability file and feed
them into `selected_subgroups` in `data_prep/write_TOPAS_rietveld.py`; run
`pixi run rietveld` to generate the `.inp` + `.bat`, then run TOPAS to confirm
the structure by real Rietveld refinement. (`write_TOPAS_ML_multiphase.py` and
`write_TOPAS_TREE_REFINE.py` cover multiphase and whole-tree variants.)

### Plotting & utilities

- `plotting/` — all `PLOT_*` manuscript-figure scripts, `PAPER_PLOT_RIETVELD.py`,
  `plot_logit_evolution.py`, tensor viewers, and the GUI `interactive_plotter.py`
  (`pixi run plotter`, see `docs/README_interactive_plotter.md`).
- `io_utils/` — config parser, tensor/xy conversion, standardization helpers.
- `diagnostics/` — scratch tests.

---

## License

XSymNet © 2026 Dayton Kizzire, Yuanpeng Zhang, Maksim Eremenko and Matthew
Tucker is licensed under the GNU General Public License v3.0 (GPL-3.0).
See the `LICENSE` file for the full legal text, or
https://www.gnu.org/licenses/gpl-3.0.html

You may use, study, modify, and redistribute this software (including
modified or derivative versions), provided that any distributed copies
or derivative works are also licensed under the GPL-3.0 and that you
make the corresponding source code available.

## Notes

- Run everything from the **repo root** (`pixi run ...` handles this).
- `pixi.lock` should be committed; `.pixi/` is git-ignored.
- The legacy hardcoded training scripts (`xsymnet_bragg.py`,
  `xsymnet_bragg_seed.py`) and the Optuna/XGBoost variants share the same
  package imports and run the same way (`python -m training.<name>`).
