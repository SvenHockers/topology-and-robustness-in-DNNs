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
│   ├── graph_manifold.py    # Graph/Laplacian construction & scores
│   ├── detectors.py         # Graph-based adversarial detector
│   ├── evaluation.py        # Evaluation metrics, ROC/AUC, calibration
│   ├── visualization.py     # Plotting functions
│   ├── utils.py             # Seeds, config, helpers
│   └── compute_combined_score.py  # Combined score computation
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

### Running the Experiment

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
from src.graph_manifold import build_knn_graph, compute_graph_scores
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

### Adversarial Attacks

- **FGSM**: Fast Gradient Sign Method
- **PGD**: Projected Gradient Descent (multi-step)

### Detector Types

- **Score-based**: Uses threshold on graph scores
- **Supervised**: Trains classifier on score features

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


