## Topology & robustness in deep neural networks

This repository contains:

- **Core library**: `src/` (datasets, models, topology-based detectors, evaluation, plotting)
- **Runner library (internal)**: `optimisers/runner_lib.py` (executes pipeline + writes artifacts)
- **Bayesian optimisation**: `optimisers/` (Gaussian-process optimiser that reuses the runner library)
- **Configs**: `config/` (YAML configs used by the optimiser)
> Note: exploratory notebooks were removed as non-production artifacts.

## Installation

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## CLI entrypoint

Single command entrypoint:

- `python -m optimisers ...`

It supports:

- Single-config optimisation (default): `python -m optimisers --base-config ...`
- Batch optimisation: `python -m optimisers batch --config-dir ...`
- Plotting: `python -m optimisers plot-history --history ...`

### Single-config GP optimisation

```bash
python -m optimisers \
  --base-config config/final/tabular/base_e_0.1.yaml \
  --dataset-name TABULAR \
  --model-name MLP \
  --space optimisers/spaces/constrains.yaml \
  --metric-path metrics_adv.roc_auc \
  --study-dir optimiser_outputs/study \
  --n-trials 30
```

### Batch GP optimisation over a config directory

```bash
python -m optimisers batch \
  --config-dir config/final/tabular \
  --dataset-name TABULAR \
  --model-name MLP \
  --space optimisers/spaces/constrains.yaml \
  --metric-path metrics_adv.roc_auc \
  --output-root optimiser_outputs/final \
  --n-trials 30
```

## Runner library (internal)

`optimisers/runner_lib.py` is the internal execution engine used by the optimiser to materialize per-trial configs,
run `src.api.run_pipeline()`, and write artifacts (metrics, raw features, logs).

## Configuration

- **Experiment configs**: `config/` (YAML; supports `base:` inheritance via `src.utils.ExperimentConfig.from_yaml`)

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

The canonical implementation in this repo is `optimisers/`.

