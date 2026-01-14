### Gaussian Process optimiser

This package implements **Bayesian optimisation with a Gaussian Process surrogate** to tune
`graph.*` and `detector.*` hyperparameters, while **reusing the existing runner pipeline**
(`runners/runner_lib.py`) so that dataset/model construction and artifacts match your current runs.

### How it works

- **Trial execution**: each trial creates a small config file derived from `--base-config` plus parameter overrides, then calls `runners.runner_lib.run_one_config()` to run the full pipeline and write artifacts.
- **Objective**: a scalar metric is read from the trial’s `metrics/metrics.json` using `--metric-path` (e.g. `metrics_adv.roc_auc`). For metrics where smaller is better (e.g. `fpr_at_tpr95`), use `--minimize`.
- **Optimisation**: after an initial random design, the optimiser fits a GP and proposes new points via **Expected Improvement**.

### Metric path options (`--metric-path`)

The optimiser reads the objective from each trial’s `runs/.../metrics/metrics.json`. That file has this structure:

- **`threshold`**: calibrated detector threshold (scalar)
- **`metrics_adv`**: detection metrics on **clean vs adversarial**
- **`metrics_ood`**: detection metrics on **clean vs OOD** (only present if OOD is enabled)

Valid `--metric-path` values are dotted paths into those objects. The most useful **scalar** metrics are:

- **Adversarial detection (recommended)**
  - `metrics_adv.roc_auc`
  - `metrics_adv.pr_auc`
  - `metrics_adv.fpr_at_tpr95` (use with `--minimize`)
  - `metrics_adv.accuracy`
  - `metrics_adv.precision`
  - `metrics_adv.recall`
  - `metrics_adv.f1`

- **OOD detection (only if `metrics_ood` exists)**
  - `metrics_ood.roc_auc`
  - `metrics_ood.pr_auc`
  - `metrics_ood.fpr_at_tpr95` (use with `--minimize`)
  - `metrics_ood.accuracy`
  - `metrics_ood.precision`
  - `metrics_ood.recall`
  - `metrics_ood.f1`

There are also **array-valued** entries (useful for plotting, not as an optimisation objective):
- `metrics_adv.fpr`, `metrics_adv.tpr`, `metrics_adv.thresholds_roc`
- `metrics_adv.pr_precision`, `metrics_adv.pr_recall`, `metrics_adv.pr_thresholds`
- `metrics_adv.confusion_matrix`
and similarly under `metrics_ood.*` when present.

## STEP-BY-STEP :)))

1) open a bash shell
2) activate your virtual environment
```bash
source .venv/bin/activate
```
3) Execute the Gaussian Optimiser
```bash
python -m optimiser.cli \
  --base-config YOUR_PATH_TO_A_CONFIG_FILE \
  --dataset-name THE_ASSOCIATED_DATASET_NAME_AS_DEFINED_IN_THE_DATA_REGISTRY \
  --model-name PRETTY_SELF_EXPLANITORY \
  --space optimiser/spaces/constraints.yaml \
  --metric-path FOR_WHAT_METRIC_DO_WE_OPTIMISE \
  --study-dir THIS_IS_WHERE_OUTPUT_IS_WRITEN_TO \
  --n-trials SELF_EXPLANATORY \
  --n-initial THIS_IS_THE_INIT_OF_GP_WHERE_WE_DO_RANDOM_SEARCH \
  --device cpu
```

### All optimiser CLI options

All flags below are supported by `python -m optimiser.cli`:

- **Required**
  - `--base-config <path>`: base YAML/JSON config to optimise
  - `--dataset-name <name>`: dataset name passed to `src.api.run_pipeline` (e.g. `IMAGE`, `TABULAR`, `VECTOR`)
  - `--model-name <name>`: model name passed to `src.api.run_pipeline` (e.g. `CNN`, `MLP`)
  - `--space <path>`: search space spec file (`.yaml/.yml/.json`)

- **Objective**
  - `--metric-path <dotted.path>`: metric path inside `metrics/metrics.json` (default: `metrics_adv.roc_auc`)
  - `--minimize`: minimize the metric instead of maximizing

- **Study/output**
  - `--study-dir <path>`: output directory (default: `optimiser_outputs/study`)

- **Bayesian optimisation (GP + EI)**
  - `--n-trials <int>`: total number of trials (default: `30`)
  - `--n-initial <int>`: random warm-up trials before GP (default: `8`)
  - `--n-candidates <int>`: random candidate pool size per BO step (default: `256`)
  - `--xi <float>`: EI exploration parameter (default: `0.01`)
  - `--seed <int>`: RNG seed (default: `42`)

- **Gaussian Process settings**
  - `--gp-noise <float>`: observation noise level (default: `1e-6`)
  - `--gp-restarts <int>`: GP hyperparameter optimizer restarts (default: `3`)

- **Runner/pipeline knobs**
  - `--device {cpu,cuda,auto}`: override device (default: use config)
  - `--enable-latex`: enable LaTeX rendering for plots (default: off)
  - `--export-features {npy,npy+csv,npy+parquet}`: feature export format (default: `npy`)
  - `--verbose`: also stream per-run logs to stdout
  - `--no-filter-clean-to-correct`: disable filtering clean eval set to model-correct points (default: filtering is enabled)
  - `--max-points-for-scoring <int>`: cap on (masked) points used for scoring (default: `400`)

- **Dataset overrides (fixed for all trials; useful for VECTOR variants)**
  - `--data-dataset-type <str>`: sets `data.dataset_type` (e.g. `torus_one_hole`, `torus_two_holes`, `nested_spheres`, `Blobs`)
  - `--data-n-points <int>`: sets `data.n_points`

### Example

Tune common topology settings on IMAGE and maximise adversarial ROC-AUC:

```bash
python -m optimiser.cli \
  --base-config config/IMAGE/base.yaml \
  --dataset-name IMAGE \
  --model-name CNN \
  --space optimiser/spaces/topology_basic.yaml \
  --metric-path metrics_adv.roc_auc \
  --study-dir optimiser_outputs/IMAGE_gp_rocauc \
  --n-trials 30 \
  --n-initial 8
```

Minimise `fpr_at_tpr95` instead:

```bash
python -m optimiser.cli \
  --base-config config/IMAGE/base.yaml \
  --dataset-name IMAGE \
  --model-name CNN \
  --space optimiser/spaces/topology_basic.yaml \
  --metric-path metrics_adv.fpr_at_tpr95 \
  --minimize \
  --study-dir optimiser_outputs/IMAGE_gp_fpr95
```

### Outputs

Under `--study-dir` you’ll get:

- **`history.jsonl` / `history.json`**: optimiser trial records and the current best trial.
- **`runs/`**: per-trial runner outputs (same structure as `outputs/` from existing runners).
- **`configs/`**: the per-trial materialized configs (JSON) that were executed.

### Visualising a finished optimisation

Once you have a `history.jsonl`, generate paper-friendly parameter-space plots with:

```bash
python -m optimiser.plot_history \
  --history <study-dir>/history.jsonl \
  --outdir <study-dir>/figs \
  --space optimiser/spaces/topology_basic.yaml
```

This writes:
- best-so-far curve, scatter matrix, PCA embedding, parallel coordinates
- **top-vs-rest marginals**, **binned mean±CI**, **rank correlation**
- **interaction slices** split by `graph.topo_preprocess`
- **partial dependence** (RF surrogate) and a **top-K table** (CSV + PNG)

