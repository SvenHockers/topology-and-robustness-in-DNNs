"""
Graph/Laplacian Manifold Methods for Detecting Off-Manifold Adversarial Examples
"""

__version__ = "0.1.0"

# Public API re-exports (package-like interface)
from .api import (  # noqa: F401
    compute_scores,
    detect,
    evaluate_detection,
    fit_detector,
    generate_adversarial,
    get_dataset,
    get_model,
    load_config,
    list_datasets,
    list_models,
    predict,
    run_pipeline,
    train,
    wrap_feature_model,
)
from .types import AttackResult, DetectorEvalResult, RunResult  # noqa: F401
from .utils import (  # noqa: F401
    AttackConfig,
    DataConfig,
    DetectorConfig,
    ExperimentConfig,
    GraphConfig,
    ModelConfig,
)

__all__ = [
    "__version__",
    # configs
    "AttackConfig",
    "DataConfig",
    "DetectorConfig",
    "ExperimentConfig",
    "GraphConfig",
    "ModelConfig",
    # results/types
    "AttackResult",
    "DetectorEvalResult",
    "RunResult",
    # api
    "list_datasets",
    "get_dataset",
    "list_models",
    "get_model",
    "load_config",
    "wrap_feature_model",
    "train",
    "predict",
    "generate_adversarial",
    "compute_scores",
    "fit_detector",
    "detect",
    "evaluate_detection",
    "run_pipeline",
]

