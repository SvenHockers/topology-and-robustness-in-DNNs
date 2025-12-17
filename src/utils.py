"""
Utility functions for reproducibility and configuration.
"""

import random
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional


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


@dataclass
class ModelConfig:
    """Configuration for MLP model architecture and training."""
    input_dim: int = 2
    hidden_dims: list = None
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
class GraphConfig:
    """Configuration for graph construction and manifold scores."""
    k: int = 10  # number of nearest neighbors
    sigma: Optional[float] = None  # if None, will use median distance heuristic
    space: str = 'input'  # 'input' or 'feature'
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
    detector_type: str = 'score'  # 'score' (anomaly) or 'supervised'
    calibration_method: str = 'isotonic'  # 'isotonic' or 'logistic'


@dataclass
class ExperimentConfig:
    """Master configuration class combining all configs."""
    data: DataConfig = None
    model: ModelConfig = None
    attack: AttackConfig = None
    graph: GraphConfig = None
    detector: DetectorConfig = None
    seed: int = 42
    device: str = 'cpu'
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.attack is None:
            self.attack = AttackConfig()
        if self.graph is None:
            self.graph = GraphConfig()
        if self.detector is None:
            self.detector = DetectorConfig()
        
        # Set device
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Set seed
        set_seed(self.seed)


