## Topology & robustness in deep neural networks

This repository contains:

- **Core library**: `src/` (datasets, models, topology-based detectors, evaluation, plotting)
- **Batch runners**: `runners/` (execute many configs under `config/` and write artifacts under `outputs/`)
- **Bayesian optimisation**: `optimiser/` (Gaussian-process optimiser that reuses the runner pipeline)
- **Configs**: `config/` (YAML configs used by runners/optimiser)
> Note: exploratory notebooks were removed as non-production artifacts.

## Installation

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Production CLI entrypoints (kept stable)

These entrypoints are preserved as stable interfaces:

- **Single-config optimisation**: `python -m optimizers.cli ...`
- **Batch optimisation**: `python -m optimisers.cli_batch ...`

They are thin compatibility shims that delegate to the implementation in `optimiser/` (see “Compatibility notes”).

### `optimizers.cli` (single-config GP optimisation)

```bash
python -m optimizers.cli \
  --base-config config/final/tabular/base_e_0.1.yaml \
  --dataset-name TABULAR \
  --model-name MLP \
  --space optimiser/spaces/constrains.yaml \
  --metric-path metrics_adv.roc_auc \
  --study-dir optimiser_outputs/study \
  --n-trials 30
```

### `optimisers.cli_batch` (batch GP optimisation over a config directory)

```bash
python -m optimisers.cli_batch \
  --config-dir config/final/tabular \
  --dataset-name TABULAR \
  --model-name MLP \
  --space optimiser/spaces/constrains.yaml \
  --metric-path metrics_adv.roc_auc \
  --output-root optimiser_outputs/final \
  --n-trials 30
```

## Batch runners (execute configs → write artifacts)

For runner details and output layout, see `runners/README.md`.

Quickstart:

```bash
python runners/run_all.py --dry-run
python runners/run_all.py --max-workers 1
```

## Configuration

- **Experiment configs**: `config/` (YAML; supports `base:` inheritance via `src.utils.ExperimentConfig.from_yaml`)
- **Model selection rationale**: `model_config.md`

## Tests (smoke)

Lightweight smoke tests validate that the required CLI entrypoints resolve and respond to `--help`:

```bash
python -m unittest discover -s tests
```

Or via the task runner:

```bash
make test
```

## Compatibility notes (import paths)

The optimiser implementation in this repo is named `optimiser/` (British spelling, singular). For deployment environments that expect:

- `optimizers.cli`
- `optimisers.cli_batch`

this repo provides compatibility packages:

- `optimizers/cli.py` → delegates to `optimiser/cli.py`
- `optimisers/cli_batch.py` → delegates to `optimiser/cli_batch.py`

No runtime logic is duplicated; behaviour is preserved by importing and calling the original `main()` functions.

