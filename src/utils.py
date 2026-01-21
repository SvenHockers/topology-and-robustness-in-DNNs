"""
Utils for setting configs
"""

import random
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Any, Dict, Union
from pathlib import Path


def set_seed(seed: int = 42):
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
    n_samples: int = 1000
    noise: float = 0.1
    random_state: int = 42
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    dataset_type: str = "torus_one_hole"
    n_points: Optional[int] = None
    root: str = "./data"
    download: bool = False


@dataclass
class ModelConfig:
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
    attack_type: str = 'fgsm'  # 'fgsm' or 'pgd'
    epsilon: float = 0.1
    num_steps: int = 10  # for PGD
    step_size: float = 0.01  # for PGD
    random_start: bool = True  # for PGD


@dataclass
class OODConfig:
    enabled: bool = False
    method: str = "feature_shuffle"
    severity: float = 1.0

    seed: Optional[int] = None
    batch_size: int = 128
    patch_size: int = 4
    blur_kernel_size: int = 5
    blur_sigma: float = 1.0
    saltpepper_p: float = 0.05


@dataclass
class GraphConfig:
    use_baseline_scores: bool = True
    k: int = 10  # k nearest neighbors
    sigma: Optional[float] = None  # if None, use median distance heuristic
    space: str = 'feature'  # 'input' or 'feature'
    feature_layer: str = "penultimate"
    normalized_laplacian: bool = True
    use_diffusion: bool = False  # diffusion map embedding
    diffusion_components: int = 10  
    use_tangent: bool = True
    tangent_k: int = 20  # neighborhood size for local PCA
    tangent_dim: Optional[int] = None 
    tangent_var_threshold: float = 0.9 
    tangent_dim_min: int = 2 
    tangent_dim_max: Optional[int] = None  

    use_topology: bool = False
    topo_k: int = 50  # neighborhood size for local PH
    topo_maxdim: int = 1  # compute H0 to H_topo_maxdim -> We keep this at 1 for the project
    topo_metric: str = 'euclidean'
    topo_thresh: Optional[float] = None 
    topo_min_persistence: float = 1e-6  
    topo_preprocess: str = 'none' 
    topo_pca_dim: int = 10  


@dataclass
class DetectorConfig:
    """Configuration for graph-based detector."""
    score_type: str = 'combined' 
    alpha: float = 0.5 
    beta: float = 0.5 
    detector_type: str = 'topology_score' 
    calibration_method: str = 'isotonic' 

    topo_feature_keys: Optional[list] = None
    topo_cov_shrinkage: float = 1e-3 
    topo_percentile: float = 95.0  
    topo_class_conditional: bool = False
    topo_class_scoring_mode: str = "min_over_classes"  
    topo_min_clean_per_class: int = 5  


@dataclass
class ExperimentConfig:
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
        Allow a couple of ergonomic aliases in YAML
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

