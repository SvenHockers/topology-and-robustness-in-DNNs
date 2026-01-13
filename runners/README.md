### Batch runners (`runners/`)

This folder contains scripts to **run the repository pipeline across many configuration files** under `config/`, while writing a consistent artifact layout under `outputs/` and reporting **adversarial/OOD success counts**.

The implementation is shared in `runners/runner_lib.py`; the `run_*.py` files are thin CLI wrappers.

---

### Scripts

- **`runners/run_all.py`**
  - Discovers **all immediate subdirectories** in `--config-root` (default `config/`)
  - Invokes `runners/run_<subdir>.py` via subprocess when present (supports subdir names containing `-`)
  - Writes aggregate success-count outputs to:
    - `outputs/_aggregate/aggregate_success_counts.csv`
    - `outputs/_aggregate/aggregate_success_counts.json`

- **Per-subdir runners**
  - **`runners/run_IMAGE.py`** → runs `config/IMAGE/**` using `dataset_name="IMAGE"`, `model_name="CNN"`
  - **`runners/run_TABULAR.py`** → runs `config/TABULAR/**` using `dataset_name="TABULAR"`, `model_name="MLP"`
  - **`runners/run_VECTOR.py`** → runs `config/VECTOR/**` using `dataset_name="VECTOR"`, `model_name="MLP"`

---

### Quickstart (Windows + venv)

Use your **venv Python** so dependencies (NumPy, PyTorch, etc.) are available.

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python runners\run_all.py --max-workers 1
```

You can also run without activation:

```powershell
.\.venv\Scripts\python.exe runners\run_all.py --max-workers 1
```

---

### CLI options (all scripts)

All runners accept the same flags:

- **`--config-root`**: config directory root (default: `config/`)
- **`--output-root`**: outputs directory root (default: `outputs/`)
- **`--extensions`**: comma-separated config extensions (default: `yaml,yml,json`)
- **`--ignore`**: repeatable glob patterns to exclude files (example: `--ignore "*template*" --ignore "*README*"`)
- **`--device`**: override device for all runs (`cpu` | `cuda` | `auto`). If omitted, uses the config value.
- **`--dry-run`**: discover configs and print planned output paths, but do not execute the pipeline
- **`--max-workers`**: parallelism per subdir runner (default: `1`)
- **`--export-features`**: `npy` | `npy+csv` | `npy+parquet` (default: `npy`)
- **`--verbose`**: also stream per-run logs to stdout (full logs always go to `outputs/.../logs/run.log`)

Examples:

```powershell
python runners\run_all.py --ignore "*template*" --max-workers 4
python runners\run_IMAGE.py --dry-run
python runners\run_TABULAR.py --export-features npy+csv
python runners\run_all.py --device cuda
```

---

### What the runner executes

For each discovered config file, the runner:

- Loads the config (`.yaml/.yml` via `ExperimentConfig.from_yaml` with `base:` inheritance; `.json` via `ExperimentConfig.from_dict`)
- Calls the repo pipeline (see `src/api.py`) via the runner wrapper:
  - `runners.runner_lib.run_pipeline_from_config(config_path=..., output_dir=...)`
- Persists artifacts under the output directory for that config

Notes:

- The pipeline itself is `src.api.run_pipeline()`; it does not take an output directory, so the runner writes artifacts.
- The runner sets `eval_only_successful_attacks=True` so the pipeline computes an adversarial success signal.
- OOD runs when `cfg.ood.enabled: true` in your YAML (unless overridden in code later).

---

### Output layout (per config file)

The output folder **mirrors `config/`**:

- Config: `config/<subdir>/.../my_config.yaml`
- Output: `outputs/<subdir>/.../my_config/` (filename stem, no extension)

Each run folder contains:

- **`images/`**: plots/overlays saved from pipeline figures when possible
- **`raw/`**: raw artifacts
  - **`raw/eval_adv.npz`**: adversarial eval labels + raw scores
  - **`raw/eval_ood.npz`**: OOD eval labels + raw scores (if OOD ran)
  - **`raw/records.jsonl`**: per-sample records (see below)
  - **`raw/features/`**: **raw feature vectors** (score dict arrays) saved unmodified
- **`metrics/`**
  - **`metrics/metrics.json`**: compact metrics summary
  - **`metrics/success_counts.json`** and **`metrics/success_counts.csv`**: adversarial/OOD success totals
- **`logs/`**
  - **`logs/run.log`**: full log
  - **`logs/summary.log`**: one-line summary like `adversarial_success=12/50, ood_success=33/40`
  - **`logs/error.txt`**: written only on failure
- **`metadata.json`**: run metadata (timestamp, git commit if available, config path, status, duration, feature file shapes/dtypes, success counts, etc.)

---

### Feature vectors (`raw/features/`)

The runner persists **each array in the pipeline “scores dicts”** (clean/adv/ood; val/test) as individual files:

- Default: `.npy` (NumPy binary)
- Optional: also export `.csv` or `.parquet` (best-effort; parquet requires `pyarrow`)

These are saved **unmodified** and tracked in `metadata.json` with `shape`, `dtype`, and `count`.

---

### Adversarial / OOD success counts

Per-run, the runner writes:

- **`metrics/success_counts.json`**
- **`metrics/success_counts.csv`**
- **`logs/summary.log`** (human-readable line)

Primary derivation (preferred):

- **Adversarial success** comes from the pipeline’s `attack_test.meta` (`adv_mask` / `success_rate`), which reflects the pipeline’s “successful attack” definition.
- **OOD success** is derived from `eval_ood` outputs using the detector threshold: count of OOD-labelled points with detector score ≥ threshold.

Robust fallback (“shim”):

- The runner also writes **`raw/records.jsonl`** with fields:
  - `sample_id`
  - `is_adversarial`, `adversarial_success`
  - `is_ood`, `ood_success`
- If a run fails mid-way or future pipelines change output formats, the shim can still attempt to compute counts from:
  - `metrics/success_counts.json` (if already present)
  - `raw/records.jsonl`
  - `predictions.jsonl` (if you add one later)

If counts can’t be derived, they are recorded as `null` with an explanatory `notes` string (the run does **not** fail due to missing counts).

---

### Aggregate outputs

After `run_all.py` finishes, you’ll get:

- **`outputs/_aggregate/aggregate_success_counts.csv`**: one row per run (plus status/error info)
- **`outputs/_aggregate/aggregate_success_counts.json`**: totals + the same per-run rows

---

### Notes / tips

- Start with `--dry-run` to verify discovery and output paths.
- Keep `--max-workers 1` initially (models/attacks can be compute-heavy); increase once stable.
- If a subdir has no dedicated `run_<subdir>.py`, `run_all.py` falls back to a generic runner:
  - `IMAGE` → `CNN`
  - everything else → `MLP`

