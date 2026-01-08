# Graph/Laplacian Manifold Methods for Detecting Off-Manifold Adversarial Examples

A research codebase for validating graph and Laplacian-based manifold methods for detecting adversarial examples in deep neural networks.

## Overview

This project implements a modular experimental framework to test the hypothesis that adversarial examples can be detected by measuring their conformity to the data manifold using graph-based and Laplacian-based scores. The experiment is conducted on the two moons dataset with a simple MLP classifier.

## Project Structure

```
Graph Manifold/
├── src/                      # Source code modules
│   ├── __init__.py
│   ├── data.py              # Dataset generation & loaders
│   ├── models.py            # MLP definition and training
│   ├── adv_attacks.py       # FGSM/PGD implementations
│   ├── graph_scoring.py     # Graph/Laplacian/topology construction & scores
│   ├── detectors.py         # Graph-based adversarial detector
│   ├── evaluation.py        # Evaluation metrics, ROC/AUC, calibration
│   ├── visualization.py     # Plotting functions
│   ├── utils.py             # Seeds, config, helpers
│   ├── api.py               # High-level public API (recommended entry point)
│   ├── types.py             # Result dataclasses used by the public API
│   ├── registry.py          # Registries for extending models/attacks
├── notebooks/               # Jupyter notebooks
│   └── 01_graph_manifold_two_moons.ipynb  # Main experiment notebook
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

1. Clone or download this repository.

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Python API (recommended)

Most users should start with the high-level API in `src.api` (also re-exported at the
package level as `src.*`).

#### External datasets (MNIST/CIFAR via torchvision)

If you install `torchvision`, the dataset registry also includes:
- `mnist`, `fashion_mnist` (1×28×28)
- `cifar10`, `cifar100` (3×32×32)

By default, **datasets are not auto-downloaded**. Point `cfg.data.root` at an existing
torchvision dataset directory, or set `cfg.data.download=True` explicitly.

```python
from src import ExperimentConfig, get_dataset, get_model

cfg = ExperimentConfig()
cfg.data.root = "./data"
cfg.data.download = False  # default (set True only if you want downloads)

bundle = get_dataset("cifar10", cfg)
model = get_model("minicnn", cfg, num_classes=bundle.meta["num_classes"], in_channels=bundle.meta["channels"])
```

#### End-to-end run (dataset → train → attack → scores → detector → metrics)

```python
from src import ExperimentConfig, run_pipeline

cfg = ExperimentConfig()
cfg.graph.space = "feature"
cfg.graph.use_topology = True
cfg.detector.detector_type = "topology_score"

result = run_pipeline(
    dataset_name="two_moons",
    model_name="two_moons_mlp",
    cfg=cfg,
    max_points_for_scoring=1000,  # optional convenience to keep runtime small
)

print(result.eval.metrics["roc_auc"])
```

#### Lower-level building blocks (compose your own experiment)

```python
import numpy as np

from src import (
    ExperimentConfig,
    get_dataset,
    get_model,
    train,
    generate_adversarial,
    compute_scores,
    fit_detector,
    evaluate_detection,
)

cfg = ExperimentConfig()
bundle = get_dataset("synthetic_shapes_2class", cfg)
model = get_model("minicnn", cfg, num_classes=bundle.meta["num_classes"])
model = train(model, bundle, cfg)

X_adv = generate_adversarial(model, bundle.X_test, bundle.y_test, cfg, clip=bundle.meta.get("clip"))
scores_clean = compute_scores(bundle.X_test, model, bundle=bundle, cfg=cfg)
scores_adv = compute_scores(X_adv, model, bundle=bundle, cfg=cfg)

# Combine for detector training/eval the same way notebooks do
scores_all = {k: np.concatenate([scores_clean[k], scores_adv[k]]) for k in scores_clean}
labels = np.concatenate([np.zeros(len(bundle.X_test)), np.ones(len(bundle.X_test))])

detector = fit_detector(scores_all, labels, cfg)
raw_scores = detector.score(scores_all)
metrics = evaluate_detection(labels, raw_scores)
print(metrics["roc_auc"])
```

### Running the Experiment (notebooks)

The main experiment is orchestrated in the Jupyter notebook:

```bash
jupyter notebook notebooks/01_graph_manifold_two_moons.ipynb
```

Or if using JupyterLab:

```bash
jupyter lab notebooks/01_graph_manifold_two_moons.ipynb
```

The notebook will:
1. Generate the two moons dataset
2. Train an MLP classifier
3. Generate adversarial examples using FGSM/PGD
4. Build k-NN graphs on training data
5. Compute graph-based manifold conformity scores
6. Train and evaluate a graph-based adversarial detector
7. Calibrate scores to error probabilities
8. Visualize results

### Using the Modules Programmatically

You can also use the modules directly in Python:

```python
from src.data import generate_two_moons
from src.models import TwoMoonsMLP, train_model
from src.adv_attacks import generate_adversarial_examples
from src.graph_scoring import build_knn_graph, compute_graph_scores
from src.utils import ExperimentConfig

# Set up configuration
config = ExperimentConfig()

# Generate data
X_train, y_train, X_val, y_val, X_test, y_test = generate_two_moons(
    n_samples=1000, noise=0.1, random_state=42
)

# Train model
model = TwoMoonsMLP(input_dim=2, hidden_dims=[64, 32], output_dim=2)
# ... training code ...

# Generate adversarial examples
X_adv_test = generate_adversarial_examples(
    model, X_test, y_test, config.attack
)

# Compute graph scores
scores = compute_graph_scores(
    X_test, model, X_train, f_train, config.graph
)
```

## Key Features

### Graph-Based Manifold Scores

1. **Degree Score**: Measures connectivity in k-NN graph (higher degree = more on-manifold)
2. **Laplacian Smoothness Score**: Measures Dirichlet energy increment (higher = less smooth = more suspicious)
3. **Diffusion Map Distance** (optional): Spectral embedding distance
4. **Topology (Persistent Homology) Features** (optional): Computes local persistent homology
   (e.g., H0/H1 persistence summaries) on neighborhoods and scores points by deviation from
   the clean topology distribution.

### Adversarial Attacks

- **FGSM**: Fast Gradient Sign Method
- **PGD**: Projected Gradient Descent (multi-step)

### Detector Types

- **Score-based**: Uses threshold on graph scores
- **Supervised**: Trains classifier on score features
- **Topology score-based (PH)**: Uses a topology feature vector (persistent homology summaries)
  and scores via Mahalanobis distance to the clean topology reference, then thresholds by a
  clean percentile.

### Evaluation Metrics

- ROC curves and AUC for adversarial detection
- FPR at fixed TPR (e.g., FPR@95%TPR)
- Error probability calibration (isotonic/logistic regression)

## Configuration

Hyperparameters can be configured through dataclasses in `src/utils.py`:

- `DataConfig`: Dataset parameters (n_samples, noise, split ratios)
- `ModelConfig`: Model architecture and training (hidden dims, learning rate, epochs)
- `AttackConfig`: Adversarial attack parameters (epsilon, num_steps)
- `GraphConfig`: Graph construction (k, sigma, space: input/feature)
- `DetectorConfig`: Detector type and calibration method

### Enabling the topology-based detector

1. Ensure PH dependency is installed (already included in `requirements.txt`):

```bash
pip install -r requirements.txt
```

2. In your experiment config:

```python
from src.utils import ExperimentConfig

config = ExperimentConfig()
config.graph.space = "feature"
config.graph.use_topology = True
config.graph.topo_k = 50
config.graph.topo_maxdim = 1

config.detector.detector_type = "topology_score"
# Optional: select which PH feature keys to use; otherwise defaults are chosen.
# config.detector.topo_feature_keys = ["topo_h0_total_persistence", "topo_h1_total_persistence", "topo_h1_entropy"]
```

## Extending the Codebase

This codebase is designed for extensibility. You can easily:

- Add new datasets by implementing a data generation function in `data.py`
- Swap model architectures by modifying `models.py`
- Implement new graph construction methods in `graph_manifold.py`
- Add new detector methods in `detectors.py`
- Compare with other detection baselines

## Citation

If you use this codebase in your research, please cite appropriately.

## License

This codebase is provided as-is for research purposes.


