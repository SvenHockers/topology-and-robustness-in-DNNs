### Gaussian Process optimiser

This package implements **Bayesian optimisation with a Gaussian Process surrogate** to tune
`graph.*` and `detector.*` hyperparameters, while **reusing the existing runner pipeline**
(`runners/runner_lib.py`) so dataset/model construction and artifacts match your existing runs.

### How it works

- **Trial execution**: each trial creates a small config file derived from `--base-config` plus parameter overrides, then calls `runners.runner_lib.run_one_config()` to run the full pipeline and write artifacts.
- **Objective**: a scalar metric is read from the trial’s `metrics/metrics.json` using `--metric-path` (e.g. `metrics_adv.roc_auc`). For metrics where smaller is better (e.g. `fpr_at_tpr95`), use `--minimize`.
- **Optimisation**: after an initial random design, the optimiser fits a GP and proposes new points via **Expected Improvement**.

### Two CLIs

- **Single-config optimisation**: `python -m optimiser.cli` (one base config → one study dir)
- **Batch optimisation over many configs**: `python -m optimiser.cli_batch` (recursively runs one study per YAML)

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

## Setup (Windows + venv)

1) Create/activate your venv (examples):

- **PowerShell**

```bash
.\venv\Scripts\Activate.ps1
```

- **cmd.exe**

```bash
.\venv\Scripts\activate.bat
```

2) Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Single-config optimisation (`optimiser.cli`)

### Quick start

```bash
python -m optimiser.cli \
  --base-config config/TABULAR/base.yaml \
  --dataset-name TABULAR \
  --model-name MLP \
  --space optimiser/spaces/constrains.yaml \
  --metric-path metrics_adv.roc_auc \
  --study-dir optimiser_outputs/TABULAR_gp \
  --n-trials 30 \
  --n-initial 8
```

### CLI options

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

## Batch optimisation over `config/final/*` (`optimiser.cli_batch`)

This is the tool to run a whole *subdirectory* of your `config/final/` experiments in one command.

### Quick start (runs nested YAMLs by default)

Example: run all experiments under `config/final/tabular/` (including nested ones like `baseline/`):

```bash
python -m optimiser.cli_batch \
  --config-dir config/final/tabular \
  --dataset-name TABULAR \
  --model-name MLP \
  --space optimiser/spaces/constrains.yaml \
  --metric-path metrics_adv.roc_auc \
  --output-root optimiser_outputs/final \
  --n-trials 30 \
  --n-initial 8
```

### What it does

- **Discovery**: recursively finds `*.yaml` / `*.yml` under `--config-dir` (nested by default).
- **Execution**: runs **one independent optimisation study per YAML**.
- **Organisation**: writes outputs **per YAML**, mirroring the folder structure under `--output-root`.
- **Batch summary**: writes a batch-level `summary.json` and `summary.csv`.
- **Resume/overwrite**:
  - default: if a study directory already exists, it is **resumed** (continues appending trials)
  - `--overwrite`: deletes the per-config study directory before running it

### Output layout

For each discovered YAML `config/final/<subdir>/<path>/<name>.yaml`, the tool writes:

- `<output-root>/<subdir>/<path>/<name>/`
  - `history.jsonl`, `history.json`
  - `configs/` (materialized per-trial configs)
  - `runs/` (runner artifacts per trial)
  - `figs/` (if `--make-plots` is enabled)

And a batch aggregate summary:
- `<output-root>/<subdir>/_aggregate/summary.json`
- `<output-root>/<subdir>/_aggregate/summary.csv`

### CLI options (batch)

- **Required**
  - `--config-dir <dir>`: directory containing configs to optimise (recursively scanned)
  - `--dataset-name <name>`: dataset name passed to `src.api.run_pipeline` (e.g. `TABULAR`, `IMAGE`, `VECTOR`)
  - `--model-name <name>`: model name passed to `src.api.run_pipeline` (e.g. `MLP`, `CNN`)
  - `--space <path>`: search space spec (`.yaml/.yml/.json`) used for *all* YAMLs

- **Objective**
  - `--metric-path <dotted.path>`: metric path inside `metrics/metrics.json` (default: `metrics_adv.roc_auc`)
  - `--minimize`: minimize metric (default: maximize)

- **Batch/output**
  - `--output-root <path>`: batch output root (default: `optimiser_outputs/final`)
  - `--overwrite`: delete existing per-config study dirs before running them

- **Discovery controls**
  - `--extensions yaml,yml`: which config extensions to include (default: `yaml,yml`)
  - `--ignore <glob>`: ignore patterns (repeatable)
  - `--ignore-baseline`: skip any config under a `baseline/` folder

- **Bayesian optimisation (GP + EI)**
  - `--n-trials <int>` (default: `30`)
  - `--n-initial <int>` (default: `8`)
  - `--n-candidates <int>` (default: `256`)
  - `--xi <float>` (default: `0.01`)
  - `--seed <int>` (default: `42`)

- **Gaussian Process settings**
  - `--gp-noise <float>` (default: `1e-6`)
  - `--gp-restarts <int>` (default: `3`)

- **Runner/pipeline knobs**
  - `--device {cpu,cuda,auto}`: override device (default: use config)
  - `--enable-latex`: enable LaTeX rendering for plots (default: off)
  - `--export-features {npy,npy+csv,npy+parquet}` (default: `npy`)
  - `--verbose`: stream per-run logs to stdout
  - `--no-filter-clean-to-correct`: disable filtering clean eval set to model-correct points (default: filtering enabled)
  - `--max-points-for-scoring <int>` (default: `400`)

- **Dataset overrides (fixed for the entire batch; useful for VECTOR variants)**
  - `--data-dataset-type <str>`: sets `data.dataset_type`
  - `--data-n-points <int>`: sets `data.n_points`

- **Post-processing**
  - `--make-plots`: auto-run `optimiser.plot_history` for each study

- **Fixed-override behavior**
  - `--force-fixed-overrides`: force the batch fixed graph/detector overrides even if a YAML explicitly sets those fields

#### Note on `baseline/` configs

If your `baseline/` YAMLs intentionally set different values (e.g. `graph.use_topology: false` or `detector.topo_feature_keys: [...]`),
the batch CLI will **not overwrite explicit YAML fields by default**. The “fixed overrides” are only applied to fields that are *missing*
from the YAML. Use `--force-fixed-overrides` only if you explicitly want to clobber YAML settings.

Also note: this protection only applies to the **batch fixed overrides**. Parameters specified in your `--space` file (e.g. `graph.k`,
`graph.topo_preprocess`) still vary per trial as usual.

### Visualising a finished optimisation

Once you have a `history.jsonl`, generate paper-friendly parameter-space plots with:

```bash
python -m optimiser.plot_history \
  --history <study-dir>/history.jsonl \
  --outdir <study-dir>/figs \
  --space optimiser/spaces/constrains.yaml
```

This writes:
- best-so-far curve, scatter matrix, PCA embedding, parallel coordinates
- **top-vs-rest marginals**, **binned mean±CI**, **rank correlation**
- **interaction slices** split by `graph.topo_preprocess`
- **partial dependence** (RF surrogate) and a **top-K table** (CSV + PNG)

