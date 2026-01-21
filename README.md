## Topology & robustness in deep neural networks

This repository contains the code, experiment and results used to study how topological features change in local neighborhoods by adversarial attacks

The central idea is:

- Build a kNN graph around each query point in feature space.
- Compute graph-derived scores.
- Train/calibrate a detector from these features and evaluate detection metrics (AUROC, AUPRC, FPR@95TPR, etc.).
- Use **Gaussian-process Bayesian optimisation** to tune graph/topology/detector hyperparameters.

### Repository layout

- **`src/`**: core library (datasets, models, attacks generation, graph + topology scoring, detectors, evaluation, plotting)
- **`config/`**: YAML experiment configs (supports inheritance via `base:`; see `src.utils.ExperimentConfig.from_yaml`)
- **`optimisers/`**: GP optimiser + CLI endpoints via make
- **`post_analyses/`**: post analyses done (related to whats written in the results section of the report)
- **`out/`**: Output of the experiments

## Installation

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
```

## Quickstart: run the experiments

The Makefile wraps all experimentation / analyses endpoints for easy execution

### See available cmds

```bash
make help
```

### Run a full dataset sweep (recursive)

Each target runs **all YAML files under the dataset directory**, including subfolders such as `baseline/` & `topology_only/`:

```bash
make run-tabular
make run-mnist
make run-synthetic-shapes
make run-blobs
make run-nested-spheres
make run-torus-one-hole
make run-torus-two-holes
```

Run all datasets sequentially:

```bash
make run-all
```

### Default values

In the makefile some parameters have been predefined but can be overwritten, for our experiments we've kept these as defined in the makefile.

- **`PYTHON`**: interpreter to use (default: `python3`)
- **`OUT`**: output directory root passed to `--output-root` (default: `out`)
- **`SPACE`**: optimiser search space spec (default: `optimisers/spaces/constrains.yaml`)
- **`RUN_TRIALS`**, **`RUN_INITIAL`**, **`RUN_SEED`**: optimisation settings
- **`SYNTH_DATASET`**, **`SYNTH_MODEL`**: used by `run-synthetic-shapes` (defaults: `synthetic_shapes_2class`, `CNN`)

Example:

```bash
make run-tabular OUT=out RUN_TRIALS=30 RUN_INITIAL=10 RUN_SEED=1
```

### Post-analyses

The Makefile also exposes post-processing scripts that operate on an output directory (the same `OUT` you used for the runs):

```bash
make post-analyses OUT=out
```

## Configs (`config/`)

The main configs live under:

- `config/final/<dataset>/...`

Typical subfolders:

- **`baseline/`**: uses geometric features only
- **`topology_only/`**: uses topology features only

## Datasets and models

### Dataset registry keys

The pipeline uses the dataset registry in `src/data.py` (see `src.api.list_datasets()`), including:

- **`TABULAR`**: scikit-learn breast cancer (tabular)
- **`IMAGE`**: MNIST via torchvision (requires `cfg.data.root` and `cfg.data.download` if you want auto-download)
- **`synthetic_shapes_2class`**, **`synthetic_shapes_3class`**: synthetic RGB images generated in-memory
- **`VECTOR`**: synthetic 3D point clouds (controlled via `cfg.data.dataset_type`)

For `VECTOR`, common `data.dataset_type` values used in configs:

- `torus_one_hole`
- `torus_two_holes`
- `nested_spheres`
- `Blobs`

### Model keys

Built-in model factories (see `src.api.list_models()`):

- **`MLP`**: vector/tabular inputs
- **`CNN`**: image inputs (channels inferred from data when possible)