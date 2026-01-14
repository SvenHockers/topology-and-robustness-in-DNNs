"""
Utility functions for reproducibility and configuration.
"""

import random
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Any, Dict, Union
from pathlib import Path


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across NumPy, PyTorch, and Python random.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class DataConfig:
    """Configuration for dataset generation."""
    n_samples: int = 1000
    noise: float = 0.1
    random_state: int = 42
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    # Optional: root directory for external datasets (e.g. torchvision IMAGE/CIFAR).
    root: str = "./data"
    # Optional: whether to download external datasets if missing.
    # Repo default is conservative (no auto-download); can be overridden by user code.
    download: bool = False


@dataclass
class ModelConfig:
    """Configuration for MLP model architecture and training."""
    input_dim: int = 2
    hidden_dims: Optional[list[int]] = None
    output_dim: int = 2
    activation: str = 'relu'  # 'relu' or 'tanh'
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    weight_decay: float = 0.0
    random_state: int = 42
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 32]


@dataclass
class AttackConfig:
    """Configuration for adversarial attacks."""
    attack_type: str = 'fgsm'  # 'fgsm' or 'pgd'
    epsilon: float = 0.1
    num_steps: int = 10  # for PGD
    step_size: float = 0.01  # for PGD
    random_start: bool = True  # for PGD


@dataclass
class OODConfig:
    """
    Configuration for out-of-distribution (OOD) generation.

    Notes:
      - `method` selects the OOD shift/corruption family.
      - `severity` is a generic scalar (interpretation depends on method).
    """

    enabled: bool = False
    method: str = "feature_shuffle"
    severity: float = 1.0

    # Determinism: if None, API falls back to ExperimentConfig.seed (+ offsets)
    seed: Optional[int] = None

    # Image-specific knobs (used by some methods)
    batch_size: int = 128
    patch_size: int = 4
    blur_kernel_size: int = 5
    blur_sigma: float = 1.0
    saltpepper_p: float = 0.05


@dataclass
class GraphConfig:
    """Configuration for graph construction and manifold scores."""
    k: int = 10  # number of nearest neighbors
    sigma: Optional[float] = None  # if None, will use median distance heuristic
    space: str = 'feature'  # 'input' or 'feature'
    # Which feature layer to use when `space == "feature"`.
    # Models may support multiple layers (e.g. `TwoMoonsMLP`: "penultimate" or "first").
    feature_layer: str = "penultimate"
    normalized_laplacian: bool = True
    use_diffusion: bool = False  # optional diffusion map embedding
    diffusion_components: int = 10  # for diffusion embedding
    # Local tangent manifold membership (PCA on neighborhood)
    use_tangent: bool = True
    tangent_k: int = 20  # neighborhood size for local PCA
    tangent_dim: Optional[int] = None  # if None, will use min(input_dim, 2)
    tangent_var_threshold: float = 0.9  # if tangent_dim is None, choose dims to explain this variance
    tangent_dim_min: int = 2  # lower bound when using explained-variance selection
    tangent_dim_max: Optional[int] = None  # upper bound when using explained-variance selection

    # --- Topology (persistent homology) scoring options ---
    # These options enable a topology-based detector score derived from persistent homology
    # features computed on a local neighborhood point cloud around each query point.
    use_topology: bool = False
    topo_k: int = 50  # neighborhood size for local PH (number of neighbors + query point)
    topo_maxdim: int = 1  # compute H0..H_topo_maxdim
    topo_metric: str = 'euclidean'
    topo_thresh: Optional[float] = None  # max filtration radius (None lets backend choose)
    topo_min_persistence: float = 1e-6  # ignore tiny lifetimes as numerical noise
    # Optional preprocessing of each local neighborhood point cloud before PH.
    # Motivation: in high-d tabular/vector settings, distances concentrate and VR PH can become
    # uninformative; projecting to a local low-d subspace often yields more meaningful topology.
    topo_preprocess: str = 'none'  # 'none' or 'pca'
    topo_pca_dim: int = 10  # used when topo_preprocess == 'pca'


@dataclass
class DetectorConfig:
    """Configuration for graph-based detector."""
    # score_type selects the scalar score used for detection.
    # For off-manifold detection, prefer:
    # - 'tangent_residual': local tangent-space projection residual (geometry)
    # - 'knn_radius': mean distance to kNN in representation space (density proxy)
    score_type: str = 'combined'  # 'degree', 'laplacian', 'diffusion', 'combined', 'tangent_residual', 'tangent_residual_z', 'knn_radius'
    alpha: float = 0.5  # weight for degree score in combined
    beta: float = 0.5  # weight for laplacian score in combined
    detector_type: str = 'topology_score'  # standardized: 'topology_score'
    calibration_method: str = 'isotonic'  # 'isotonic' or 'logistic'

    # Topology-score detector (uses PH feature vectors -> scalar score -> threshold).
    # When detector_type == 'topology_score', topo_feature_keys selects which score dict keys
    # become the feature vector.
    topo_feature_keys: Optional[list] = None
    topo_cov_shrinkage: float = 1e-3  # diagonal shrinkage for covariance stabilization
    topo_percentile: float = 95.0  # clean-score percentile threshold (FPR target ~ 5%)

    # --- Class-conditional topology scoring (optional) ---
    # Motivation: pooled (all-class) Gaussian scoring can be a poor approximation when
    # topology features are multi-modal across classes. Enabling this fits one Gaussian
    # per class on clean samples and scores using either:
    #   - 'min_over_classes' (default): min Mahalanobis distance over classes
    #   - 'predicted_class': Mahalanobis distance to the classifier's predicted class
    #
    # Defaults preserve current behavior (pooled scoring).
    topo_class_conditional: bool = False
    topo_class_scoring_mode: str = "min_over_classes"  # 'min_over_classes' | 'predicted_class' | 'true_class'
    topo_min_clean_per_class: int = 5  # below this, class covariance falls back to diagonal+shrinkage


@dataclass
class ExperimentConfig:
    """Master configuration class combining all configs."""
    data: Optional[DataConfig] = None
    model: Optional[ModelConfig] = None
    attack: Optional[AttackConfig] = None
    ood: Optional[OODConfig] = None
    graph: Optional[GraphConfig] = None
    detector: Optional[DetectorConfig] = None
    seed: int = 42
    device: str = 'cpu'
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.attack is None:
            self.attack = AttackConfig()
        if self.ood is None:
            self.ood = OODConfig()
        if self.graph is None:
            self.graph = GraphConfig()
        if self.detector is None:
            self.detector = DetectorConfig()
        
        # Set device
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Set seed
        set_seed(self.seed)

    @staticmethod
    def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two dicts without mutating either.

        - dict values are merged recursively
        - non-dict values in override replace base
        """
        out: Dict[str, Any] = dict(base)
        for k, v in (override or {}).items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = ExperimentConfig._deep_merge_dicts(out[k], v)
            else:
                out[k] = v
        return out

    @staticmethod
    def _normalize_config_dict(d: Dict[str, Any]) -> Dict[str, Any]:
        """
        Allow a couple of ergonomic aliases in YAML:
          - top-level pca_dim -> graph.topo_pca_dim
          - graph.pca_dim -> graph.topo_pca_dim
        """
        if not isinstance(d, dict):
            raise TypeError("Config must be a mapping/dict.")

        d = dict(d)
        graph = dict(d.get("graph") or {})

        if "pca_dim" in d and "topo_pca_dim" not in graph:
            graph["topo_pca_dim"] = d.pop("pca_dim")

        if "pca_dim" in graph and "topo_pca_dim" not in graph:
            graph["topo_pca_dim"] = graph.pop("pca_dim")

        if graph:
            d["graph"] = graph
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        """
        Build an ExperimentConfig from a (possibly partial) nested dict with keys:
          - data, model, attack, graph, detector, seed, device
        """
        d = cls._normalize_config_dict(d or {})

        data_cfg = DataConfig(**(d.get("data") or {}))
        model_cfg = ModelConfig(**(d.get("model") or {}))
        attack_cfg = AttackConfig(**(d.get("attack") or {}))
        ood_cfg = OODConfig(**(d.get("ood") or {}))
        graph_cfg = GraphConfig(**(d.get("graph") or {}))
        detector_cfg = DetectorConfig(**(d.get("detector") or {}))

        return cls(
            data=data_cfg,
            model=model_cfg,
            attack=attack_cfg,
            ood=ood_cfg,
            graph=graph_cfg,
            detector=detector_cfg,
            seed=d.get("seed", 42),
            device=d.get("device", "cpu"),
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ExperimentConfig":
        """
        Load config from a YAML file. Supports optional inheritance:

        - If the YAML contains: base: "base.yaml"
          then the base config is loaded first and merged with the current file.
        - Relative base paths are resolved relative to the current YAML's directory.
        """
        path = Path(path)
        try:
            import yaml  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Loading YAML configs requires PyYAML. Install with: pip install pyyaml"
            ) from e

        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise TypeError(f"YAML config must be a mapping/dict. Got: {type(raw)}")

        base_ref = raw.pop("base", None)
        merged = raw
        if base_ref:
            base_path = Path(base_ref)
            if not base_path.is_absolute():
                base_path = (path.parent / base_path).resolve()
            base_cfg = cls.from_yaml(base_path)
            merged = cls._deep_merge_dicts(base_cfg.to_dict(), raw)

        return cls.from_dict(merged)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain nested dict (useful for YAML merge/inheritance)."""
        return {
            "data": dict(self.data.__dict__),
            "model": dict(self.model.__dict__),
            "attack": dict(self.attack.__dict__),
            "ood": dict(self.ood.__dict__),
            "graph": dict(self.graph.__dict__),
            "detector": dict(self.detector.__dict__),
            "seed": int(self.seed),
            "device": str(self.device),
        }

