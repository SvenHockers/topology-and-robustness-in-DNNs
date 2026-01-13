"""
Sweep presets for this repo: **frozen models** + **focused sweep ranges**.

This file is intentionally *lightweight*: it does not implement a sweep runner.
It defines:
  - frozen per-dataset configs (model architectures/training are fixed)
  - the parameter ranges we actually want to sweep

Modalities in this repo:
  - tabular / generic vectors: X has shape (N, D)
  - pointcloud (still represented as vectors here): X has shape (N, 3)
  - images: X has shape (N, C, H, W)

Notes on the pipeline:
  - The default detector used by `api.run_pipeline(...)` is the topology-score detector.
  - Topology features are enabled by `cfg.graph.use_topology=True` and controlled by `cfg.graph.topo_*`.
  - `cfg.detector.topo_percentile` sets the clean-quantile threshold (e.g. 95 => ~5% clean FPR target).
  - Attack epsilons are in *input units*:
      - standardized tabular: eps ~ 0.03..0.30 is typical
      - images in [0,1]: eps ~ 2/255..16/255 (CIFAR-like) or 0.10..0.30 (MNIST-like)
      - point clouds: eps depends on dataset scale; geometrical-shapes defaults are ~0.2..0.6

Detector-layer convention for this repo:
  - We only consider the **penultimate layer** when `graph.space == "feature"`.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List


def _logspace10(lo_exp: float, hi_exp: float, n: int) -> List[float]:
    """Return n values in 10**linspace(lo_exp, hi_exp)."""
    if n <= 1:
        return [float(10 ** hi_exp)]
    step = (hi_exp - lo_exp) / float(n - 1)
    return [float(10 ** (lo_exp + i * step)) for i in range(n)]


# ---------------------------------------------------------------------
# Frozen per-dataset presets
# ---------------------------------------------------------------------

# Each preset is a single "experiment template" for `api.run_pipeline(...)`:
#   - model architecture/training is fixed
#   - detector uses `penultimate` features when in feature space
#
# A sweep runner should:
#   (a) start from `DATASET_PRESETS[key]["cfg"]`
#   (b) apply parameter overrides for:
#         - attack.epsilon
#         - graph.topo_k            (kNN neighborhood size used for PH)
#         - graph.topo_preprocess   ("pca" or "none")
#         - graph.topo_pca_dim      (only meaningful when preprocess == "pca")
#   (c) derive attack.step_size from epsilon (see derive_step_size)

DATASET_PRESETS: Dict[str, Dict[str, Any]] = {
    # Tabular (standardized) — topology in feature space + local PCA by default.
    "breast_cancer_tabular": {
        "dataset_name": "breast_cancer_tabular",
        "model_name": "MLP",
        "model_kwargs": {},  # input_dim/output_dim inferred in api.run_pipeline for vectors
        "cfg": {
            "seed": 42,
            "device": "cpu",
            "model": {
                # Freeze: "mlp_med_reg" from foolability search (good acc + attackable)
                "hidden_dims": [128, 64],
                "activation": "relu",
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "epochs": 80,
                "batch_size": 64,
            },
            "attack": {
                "attack_type": "pgd",
                "epsilon": 0.10,
                "num_steps": 30,
                "step_size": 0.01,  # epsilon/10; keep derived in sweeps
                "random_start": True,
            },
            "graph": {
                "space": "feature",
                "feature_layer": "penultimate",
                "use_topology": True,
                "topo_k": 40,
                "topo_maxdim": 1,
                "topo_preprocess": "pca",
                "topo_pca_dim": 10,
                "topo_min_persistence": 1e-6,
            },
            "detector": {
                "detector_type": "topology_score",
                "topo_percentile": 95.0,
                "topo_cov_shrinkage": 1e-3,
            },
        },
    },
    # Vector / point clouds (3D) — topology in input space; PCA optional.
    "geometrical-shapes": {
        "dataset_name": "geometrical-shapes",
        "model_name": "MLP",
        "model_kwargs": {},  # input_dim inferred as 3
        "cfg": {
            "seed": 42,
            "device": "cpu",
            "model": {
                # Freeze: "mlp_big" from foolability search (best foolability at high acc)
                "hidden_dims": [256, 128],
                "activation": "relu",
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "epochs": 40,
                "batch_size": 64,
            },
            "attack": {
                "attack_type": "pgd",
                "epsilon": 0.30,
                "num_steps": 50,
                "step_size": 0.03,  # epsilon/10; keep derived in sweeps
                "random_start": True,
            },
            "graph": {
                "space": "input",
                "feature_layer": "penultimate",  # ignored for input space; kept consistent
                "use_topology": True,
                "topo_k": 150,
                "topo_maxdim": 2,
                "topo_preprocess": "none",
                "topo_pca_dim": 3,  # ignored when preprocess == "none"
                "topo_min_persistence": 1e-6,
            },
            "detector": {
                "detector_type": "topology_score",
                "topo_percentile": 95.0,
                "topo_cov_shrinkage": 1e-3,
            },
        },
    },
    # Images (MNIST) — topology in penultimate feature space + PCA by default.
    "mnist": {
        "dataset_name": "mnist",
        "model_name": "CNN",
        "model_kwargs": {"in_channels": 1, "feat_dim": 128},
        "cfg": {
            "seed": 42,
            "device": "cpu",
            "data": {"root": "./data", "download": False},
            "model": {
                "output_dim": 10,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                # Freeze: "CNN_feat128_reg_20ep" (sanity-checked at eps=0.05..0.2)
                "epochs": 20,
                "batch_size": 64,
            },
            "attack": {
                "attack_type": "pgd",
                "epsilon": 0.20,  # MNIST in [0,1]
                "num_steps": 10,
                "step_size": 0.04,  # epsilon/5; keep derived in sweeps
                "random_start": True,
            },
            "graph": {
                "space": "feature",
                "feature_layer": "penultimate",
                "use_topology": True,
                "topo_k": 40,
                "topo_maxdim": 1,
                "topo_preprocess": "pca",
                "topo_pca_dim": 16,
                "topo_min_persistence": 1e-6,
            },
            "detector": {
                "detector_type": "topology_score",
                "topo_percentile": 95.0,
                "topo_cov_shrinkage": 1e-3,
            },
        },
    },
}


# ---------------------------------------------------------------------
# Focused sweep ranges (what you said you want to sweep)
# ---------------------------------------------------------------------

# We keep this intentionally narrow: models are frozen; only sweep:
#   - epsilon
#   - k (kNN neighborhood size used by PH) => graph.topo_k
#   - PCA dims
#   - and "no PCA" (graph.topo_preprocess="none")

FOCUSED_SWEEP_RANGES: Dict[str, Dict[str, List[Any]]] = {
    "breast_cancer_tabular": {
        "attack.epsilon": [0.03, 0.05, 0.10, 0.20, 0.30],
        "graph.topo_k": [20, 40, 80, 120],
        "graph.topo_preprocess": ["pca", "none"],
        "graph.topo_pca_dim": [5, 10, 15, 20],
        # Detector hyperparameters
        # Note: `topo_percentile` is in [0,100], so quantiles {0.9,0.95,0.99} => {90,95,99}.
        "detector.topo_percentile": [90.0, 95.0, 99.0],
        "detector.topo_cov_shrinkage": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        # step size should be derived from epsilon in the sweep runner
        "attack.step_size": ["__DERIVE_EPS_OVER_10__", "__DERIVE_EPS_OVER_5__"],
    },
    "geometrical-shapes": {
        "attack.epsilon": [0.10, 0.20, 0.30, 0.50, 0.70],
        "graph.topo_k": [60, 100, 150, 220],
        "graph.topo_preprocess": ["none", "pca"],
        "graph.topo_pca_dim": [2, 3],
        "detector.topo_percentile": [90.0, 95.0, 99.0],
        "detector.topo_cov_shrinkage": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        "attack.step_size": ["__DERIVE_EPS_OVER_10__", "__DERIVE_EPS_OVER_5__"],
    },
    "mnist": {
        # MNIST in [0,1]
        # Include a smaller eps regime (sanity check showed non-trivial but not-saturated success at 0.05).
        "attack.epsilon": [0.05, 0.08, 0.10, 0.20, 0.30],
        # CIFAR-like alternative (0..1): [2/255, 4/255, 8/255, 16/255]
        "graph.topo_k": [20, 40, 80, 120],
        # "no PCA" is still allowed, but usually much noisier in high-d feature clouds.
        "graph.topo_preprocess": ["pca", "none"],
        "graph.topo_pca_dim": [8, 16, 32, 64],
        "detector.topo_percentile": [90.0, 95.0, 99.0],
        "detector.topo_cov_shrinkage": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        "attack.step_size": ["__DERIVE_EPS_OVER_10__", "__DERIVE_EPS_OVER_5__"],
    },
}


def derive_step_size(epsilon: float, rule: str) -> float:
    """
    Helper for sweep runners: map a step-size rule string -> numeric step size.

    Accepted rules:
      - "__DERIVE_EPS_OVER_10__" -> epsilon / 10
      - "__DERIVE_EPS_OVER_5__"  -> epsilon / 5
    """
    eps = float(epsilon)
    if rule == "__DERIVE_EPS_OVER_10__":
        return eps / 10.0
    if rule == "__DERIVE_EPS_OVER_5__":
        return eps / 5.0
    raise ValueError(f"Unknown step-size rule: {rule!r}")


def approx_ph_cost_scale(topo_k: int, topo_maxdim: int) -> float:
    """
    Very rough proxy for PH runtime scaling to help you prune sweeps.

    Vietoris–Rips PH grows quickly with neighborhood size and maxdim.
    This is *not* a complexity proof; it’s a practical heuristic:
      cost ~ O(k^(d+2)) where d=maxdim (aggressive upper bound).
    """
    k = max(1, int(topo_k))
    d = max(0, int(topo_maxdim))
    return float(k ** (d + 2))

