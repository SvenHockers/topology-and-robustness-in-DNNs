## Topology & robustness in deep neural networks

This repository contains:

- **Core library**: `src/` (datasets, models, topology-based detectors, evaluation, plotting)
- **Runner library (internal)**: `optimisers/runner_lib.py` (executes pipeline + writes artifacts)
- **Bayesian optimisation CLI**: `optimisers/` (Gaussian-process optimiser that reuses the runner library)
- **Experiment configs**: `config/` (YAML used by the optimiser; supports `base:` inheritance via `src.utils.ExperimentConfig.from_yaml`)

> Note: exploratory notebooks were removed as non-production artifacts.

## Quickstart (recommended)

Create and activate a virtual environment, then use the Makefile targets:

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
make help
```

## Makefile usage

The Makefile is the easiest way to run the main experiments. It wraps `python -m optimisers batch ...` with sensible defaults.

### Discover available targets

```bash
make help
```

### Install dependencies

```bash
make install
```

By default this runs `python3 -m pip install -r requirements.txt`. If your env uses a different interpreter, override `PYTHON`:

```bash
make install PYTHON=.venv/bin/python
```

### Run smoke tests

```bash
make test
```

### Show optimiser CLI help

```bash
make cli-help
```

### Run full batch sweeps (all configs in a dataset directory)

These targets run **all YAML configs** under `config/final/<dataset>/` and write results to `OUT` (default: `out`).

```bash
make run-tabular
make run-mnist
make run-synthetic-shapes
make run-blobs
make run-nested-spheres
make run-torus-one-hole
make run-torus-two-holes
```

Run everything sequentially:

```bash
make run-all-final
```

### Run topology-only sweeps (PH features only; baseline scores disabled)

Each dataset now has a `config/final/<dataset>/topology_only/` directory, containing YAMLs that inherit from the corresponding `base*.yaml` and set:

- `graph.use_topology: true`
- `graph.use_baseline_scores: false`

This isolates performance coming **purely from persistent-homology summary features** (no `degree`, `laplacian`, tangent scores, etc.).

```bash
make topology-only-tabular
make topology-only-mnist
make topology-only-synthetic-shapes
make topology-only-blobs
make topology-only-nested-spheres
make topology-only-torus-one-hole
make topology-only-torus-two-holes
```

#### Common overrides (batch runs)

The batch targets accept a few standard overrides:

- **`OUT`**: output directory root (default `out`)
- **`SPACE`**: optimiser search space YAML (default `optimisers/spaces/constrains.yaml`)
- **`RUN_TRIALS`**: number of optimisation trials per config (default `15`)
- **`RUN_INITIAL`**: number of random initial trials (default `5`)
- **`RUN_SEED`**: random seed (default `30`)

Example:

```bash
make run-tabular OUT=out/tabular RUN_TRIALS=30 RUN_INITIAL=10 RUN_SEED=1
```

### Run OOD-only sweep for synthetic_shapes

This target runs **only** the OOD configs for `synthetic_shapes` by ignoring the baseline directory and any `base*` configs.

```bash
make ood-synthetic-shapes
```

#### Common overrides (OOD-only run)

- **`OOD_CONFIG_DIR`**: config dir to scan (default `config/final/synthetic_shapes`)
- **`OOD_DATASET`**: dataset variant (default `synthetic_shapes_2class`)
- **`OOD_MODEL`**: model name (default `CNN`)
- **`OOD_TRIALS`**: number of trials (default `15`)
- **`OOD_INITIAL`**: number of random initial trials (default `5`)
- **`OOD_SEED`**: random seed (default `30`)
- **`OUT`**, **`SPACE`**, **`PYTHON`**: as above

Example:

```bash
make ood-synthetic-shapes OOD_DATASET=synthetic_shapes_3class OOD_TRIALS=5 OUT=out/ood
```

## CLI entrypoint (direct)

If you prefer not to use `make`, the single command entrypoint is:

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
  --metric-path auto \
  --output-root out \
  --n-trials 30
```

## Runner library (internal)

`optimisers/runner_lib.py` is the internal execution engine used by the optimiser to materialize per-trial configs,
run `src.api.run_pipeline()`, and write artifacts (metrics, raw features, logs).

## Compatibility notes (import paths)

The canonical implementation in this repo is `optimisers/`.

