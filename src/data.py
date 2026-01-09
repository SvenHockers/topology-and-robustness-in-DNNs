"""
Dataset generation and data loaders.

This module centralizes dataset generation used in notebooks to avoid duplication:
- two moons toy data
- tabular/vector datasets (real + synthetic)
- synthetic "image-like" RGB shapes datasets (2-class / 3-class)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Sequence, Any, Dict


def _validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) >= 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")


def generate_two_moons(
    n_samples: int = 1000,
    noise: float = 0.1,
    random_state: int = 42,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate two moons dataset and split into train/val/test sets.
    
    Args:
        n_samples: Total number of samples to generate
        noise: Standard deviation of Gaussian noise added to the data
        random_state: Random seed for reproducibility
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test) as numpy arrays
    """
    _validate_split_ratios(train_ratio, val_ratio, test_ratio)
    
    # Generate two moons dataset
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    
    # First split: train + val vs test
    test_size = test_ratio
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train vs val
    val_size = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )

    # Help static type checkers (and ensure consistent dtypes)
    X_train = np.asarray(X_train, dtype=float)
    X_val = np.asarray(X_val, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    y_val = np.asarray(y_val, dtype=int)
    y_test = np.asarray(y_test, dtype=int)

    return X_train, y_train, X_val, y_val, X_test, y_test


class NumpyTensorDataset(Dataset):
    """PyTorch Dataset wrapper for generic (X,y) numpy arrays (any input shape)."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            X: Feature array of shape (n_samples, ...)
            y: Label array of shape (n_samples,)
        """
        # torch.as_tensor avoids an unconditional copy for already-contiguous float32 arrays
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# Backwards compatible alias (old name used throughout earlier notebooks).
TwoMoonsDataset = NumpyTensorDataset


def create_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train, validation, and test sets.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        y_test: Test labels
        batch_size: Batch size for data loaders
        shuffle_train: Whether to shuffle training data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = NumpyTensorDataset(X_train, y_train)
    val_dataset = NumpyTensorDataset(X_val, y_val)
    test_dataset = NumpyTensorDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle_train
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------
# Dataset base class + standardized return bundle
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetBundle:
    """
    Standard dataset output for experiments.

    - X_* are numpy arrays of shape (N, ...) (e.g., (N,D) for tabular, (N,C,H,W) for images)
    - y_* are integer class labels of shape (N,)
    - meta contains dataset-specific info (input_kind, num_classes, clip range, transforms, etc.)
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    meta: Dict[str, Any]

    def num_classes(self) -> int:
        y = np.asarray(self.y_train, dtype=int)
        return int(len(np.unique(y)))


class BaseDataset(ABC):
    """
    Base class for datasets used by the experiment pipeline.

    Subclasses should implement `load()` and return a DatasetBundle.
    """

    name: str = "base"

    @abstractmethod
    def load(self, cfg: Any) -> DatasetBundle:  # pragma: no cover
        raise NotImplementedError

    def loaders(
        self,
        cfg: Any,
        *,
        batch_size: int,
        shuffle_train: bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader, DatasetBundle]:
        bundle = self.load(cfg)
        train_loader, val_loader, test_loader = create_data_loaders(
            bundle.X_train,
            bundle.y_train,
            bundle.X_val,
            bundle.y_val,
            bundle.X_test,
            bundle.y_test,
            batch_size=int(batch_size),
            shuffle_train=bool(shuffle_train),
        )
        return train_loader, val_loader, test_loader, bundle


def _infer_num_classes(y: Any) -> int:
    """Return number of unique integer labels (robust to array-likes)."""
    return int(len(np.unique(np.asarray(y, dtype=int))))


def _as_float32(X: Any) -> np.ndarray:
    """Convert array-like to np.float32 numpy array (no copy when possible)."""
    X = np.asarray(X)
    return X.astype(np.float32, copy=False) if X.dtype != np.float32 else X


def _as_int64(y: Any) -> np.ndarray:
    """Convert array-like to np.int64 numpy array (no copy when possible)."""
    y = np.asarray(y)
    return y.astype(np.int64, copy=False) if y.dtype != np.int64 else y


class TwoMoonsDatasetSpec(BaseDataset):
    name = "two_moons"

    def load(self, cfg: Any) -> DatasetBundle:
        # Expect cfg to look like ExperimentConfig (cfg.data.*, cfg.seed), but keep it duck-typed.
        data = getattr(cfg, "data", cfg)
        X_train, y_train, X_val, y_val, X_test, y_test = generate_two_moons(
            n_samples=int(getattr(data, "n_samples", 1000)),
            noise=float(getattr(data, "noise", 0.1)),
            random_state=int(getattr(data, "random_state", getattr(cfg, "seed", 42))),
            train_ratio=float(getattr(data, "train_ratio", 0.6)),
            val_ratio=float(getattr(data, "val_ratio", 0.2)),
            test_ratio=float(getattr(data, "test_ratio", 0.2)),
        )
        meta = {
            "input_kind": "vector",
            "clip": None,
            "num_classes": _infer_num_classes(y_train),
        }
        return DatasetBundle(
            _as_float32(X_train), _as_int64(y_train),
            _as_float32(X_val), _as_int64(y_val),
            _as_float32(X_test), _as_int64(y_test),
            meta=meta,
        )


class GeometricalPointCloudDatasetSpec(BaseDataset):
    name = "geometrical_pointclouds"

    def load(self, cfg: Any) -> DatasetBundle:
        # Duck-typed config access (same pattern as TwoMoons)
        data = getattr(cfg, "data", cfg)

        dataset_type = getattr(data, "dataset_type", "torus_one_hole")
        noise = float(getattr(data, "noise", 0.1))
        n_points = getattr(data, "n_points", None)

        seed = int(getattr(data, "random_state", getattr(cfg, "seed", 42)))
        train_ratio = float(getattr(data, "train_ratio", 0.6))
        val_ratio = float(getattr(data, "val_ratio", 0.2))
        test_ratio = float(getattr(data, "test_ratio", 0.2))

        # Core generation call (your abstraction)
        X_train, y_train, X_val, y_val, X_test, y_test = generate_geometrical_dataset(
            dataset_type=dataset_type,
            n_points=n_points,
            noise=noise,
            random_state=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        # Metadata describing the dataset (important for downstream logic)
        meta = {
            "input_kind": "pointcloud",
            "input_dim": 3,
            "dataset_type": dataset_type,
            "noise": noise,
            "num_classes": _infer_num_classes(y_train),
            "clip": None,
        }

        return DatasetBundle(
            _as_float32(X_train), _as_int64(y_train),
            _as_float32(X_val), _as_int64(y_val),
            _as_float32(X_test), _as_int64(y_test),
            meta=meta,
        )

class BreastCancerTabularDatasetSpec(BaseDataset):
    name = "breast_cancer_tabular"

    def load(self, cfg: Any) -> DatasetBundle:
        X, y = load_breast_cancer_tabular(as_float32=False)
        data = getattr(cfg, "data", cfg)
        X_train, y_train, X_val, y_val, X_test, y_test, scaler = split_and_scale(
            X,
            y,
            seed=int(getattr(cfg, "seed", 42)),
            train_ratio=float(getattr(data, "train_ratio", 0.6)),
            val_ratio=float(getattr(data, "val_ratio", 0.2)),
            test_ratio=float(getattr(data, "test_ratio", 0.2)),
        )
        meta = {
            "input_kind": "vector",
            "clip": None,
            "num_classes": _infer_num_classes(y_train),
            "scaler": scaler,
        }
        return DatasetBundle(
            _as_float32(X_train), _as_int64(y_train),
            _as_float32(X_val), _as_int64(y_val),
            _as_float32(X_test), _as_int64(y_test),
            meta=meta,
        )


class SyntheticShapes2ClassDatasetSpec(BaseDataset):
    name = "synthetic_shapes_2class"

    def __init__(self, *, image_size: int = 32):
        self.image_size = int(image_size)

    def load(self, cfg: Any) -> DatasetBundle:
        data = getattr(cfg, "data", cfg)
        n_total = int(getattr(data, "n_samples", 1000))
        train_ratio = float(getattr(data, "train_ratio", 0.6))
        val_ratio = float(getattr(data, "val_ratio", 0.2))
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))
        n_test = int(n_total - n_train - n_val)

        X_train, y_train, X_val, y_val, X_test, y_test = load_synthetic_shapes_2class(
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            seed=int(getattr(cfg, "seed", 0)),
            image_size=int(self.image_size),
        )
        meta = {
            "input_kind": "image",
            "clip": (0.0, 1.0),
            "num_classes": _infer_num_classes(y_train),
        }
        return DatasetBundle(
            _as_float32(X_train), _as_int64(y_train),
            _as_float32(X_val), _as_int64(y_val),
            _as_float32(X_test), _as_int64(y_test),
            meta=meta,
        )


class SyntheticShapes3ClassDatasetSpec(BaseDataset):
    name = "synthetic_shapes_3class"

    def __init__(self, *, image_size: int = 32):
        self.image_size = int(image_size)

    def load(self, cfg: Any) -> DatasetBundle:
        data = getattr(cfg, "data", cfg)
        n_total = int(getattr(data, "n_samples", 1000))
        train_ratio = float(getattr(data, "train_ratio", 0.6))
        val_ratio = float(getattr(data, "val_ratio", 0.2))
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))
        n_test = int(n_total - n_train - n_val)

        X_train, y_train, X_val, y_val, X_test, y_test = load_synthetic_shapes_3class(
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            seed=int(getattr(cfg, "seed", 0)),
            image_size=int(self.image_size),
        )
        meta = {
            "input_kind": "image",
            "clip": (0.0, 1.0),
            "num_classes": _infer_num_classes(y_train),
        }
        return DatasetBundle(
            _as_float32(X_train), _as_int64(y_train),
            _as_float32(X_val), _as_int64(y_val),
            _as_float32(X_test), _as_int64(y_test),
            meta=meta,
        )


def _torchvision_dataset_to_numpy(ds: Any, *, batch_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a torchvision-style Dataset returning (tensor, label) into numpy arrays.

    Expected tensor shape:
      - images: (C,H,W) float in [0,1] (e.g., via transforms.ToTensor()).
    """
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False)
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    X = torch.cat(xs, dim=0).cpu().numpy()
    y = torch.cat(ys, dim=0).cpu().numpy()
    return _as_float32(X), _as_int64(y)


class TorchvisionDatasetSpec(BaseDataset):
    """
    Optional external datasets via torchvision (no auto-download by default).

    Uses `cfg.data.root` and `cfg.data.download` when present.
    """

    def __init__(self, *, name: str, dataset: str, num_classes: int):
        self.name = str(name)
        self.dataset = str(dataset)
        self._num_classes = int(num_classes)

    def load(self, cfg: Any) -> DatasetBundle:
        data = getattr(cfg, "data", cfg)
        root = str(getattr(data, "root", "./data"))
        download = bool(getattr(data, "download", False))
        seed = int(getattr(cfg, "seed", getattr(data, "random_state", 42)))

        train_ratio = float(getattr(data, "train_ratio", 0.8))
        val_ratio = float(getattr(data, "val_ratio", 0.2))
        if train_ratio <= 0 or val_ratio < 0 or (train_ratio + val_ratio) <= 0:
            raise ValueError("Need positive train_ratio and non-negative val_ratio for torchvision datasets.")
        val_size = val_ratio / (train_ratio + val_ratio)

        try:
            from torchvision import datasets, transforms  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Torchvision datasets require torchvision.\n"
                "Install with: pip install torchvision\n"
            ) from e

        if not hasattr(datasets, self.dataset):
            raise KeyError(f"torchvision.datasets has no dataset {self.dataset!r}")

        ds_cls = getattr(datasets, self.dataset)
        tfm = transforms.ToTensor()

        # Most torchvision datasets use train=True/False (MNIST/CIFAR*). This repo supports those first.
        try:
            ds_train = ds_cls(root=root, train=True, download=download, transform=tfm)
            ds_test = ds_cls(root=root, train=False, download=download, transform=tfm)
        except TypeError as e:  # pragma: no cover
            raise TypeError(
                f"Dataset {self.dataset} did not accept train=True/False constructor.\n"
                f"Original error: {e}"
            ) from e

        X_full, y_full = _torchvision_dataset_to_numpy(ds_train)
        X_test, y_test = _torchvision_dataset_to_numpy(ds_test)

        # Split torchvision train into train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X_full,
            y_full,
            test_size=float(val_size),
            random_state=seed,
            stratify=y_full,
        )

        meta = {
            "input_kind": "image",
            "clip": (0.0, 1.0),
            "num_classes": int(self._num_classes),
            "channels": int(X_train.shape[1]) if X_train.ndim == 4 else None,
            "source": "torchvision",
            "root": root,
            "download": download,
        }
        return DatasetBundle(
            _as_float32(X_train), _as_int64(y_train),
            _as_float32(X_val), _as_int64(y_val),
            _as_float32(X_test), _as_int64(y_test),
            meta=meta,
        )


DATASET_REGISTRY: Dict[str, BaseDataset] = {
    "two_moons": TwoMoonsDatasetSpec(),
    "breast_cancer_tabular": BreastCancerTabularDatasetSpec(),
    "synthetic_shapes_2class": SyntheticShapes2ClassDatasetSpec(image_size=32),
    "synthetic_shapes_3class": SyntheticShapes3ClassDatasetSpec(image_size=32),
    # External datasets (optional; require torchvision). Defaults: no auto-download.
    "mnist": TorchvisionDatasetSpec(name="mnist", dataset="MNIST", num_classes=10),
    "fashion_mnist": TorchvisionDatasetSpec(name="fashion_mnist", dataset="FashionMNIST", num_classes=10),
    "cifar10": TorchvisionDatasetSpec(name="cifar10", dataset="CIFAR10", num_classes=10),
    "cifar100": TorchvisionDatasetSpec(name="cifar100", dataset="CIFAR100", num_classes=100),
    "geometrical-shapes":GeometricalPointCloudDatasetSpec()
}

# ---------------------------------------------------------------------
# Tabular / vector data helpers (used by notebook 04)
# ---------------------------------------------------------------------

def split_and_scale(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
):
    """
    Stratified train/val/test split + StandardScaler fit on train.

    Returns:
      (X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, scaler)
    """
    from sklearn.preprocessing import StandardScaler

    _validate_split_ratios(train_ratio, val_ratio, test_ratio)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed, stratify=y
    )
    val_size = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=seed, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, scaler


def load_breast_cancer_tabular(*, as_float32: bool = False):
    """
    Load scikit-learn breast cancer dataset (X,y).
    """
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True)
    X = np.asarray(X, dtype=np.float32 if as_float32 else float)
    y = np.asarray(y, dtype=int)
    return X, y


def generate_synthetic_vector_classification(
    *,
    n_samples: int = 2000,
    n_features: int = 100,
    n_informative: int = 20,
    n_redundant: int = 20,
    n_clusters_per_class: int = 2,
    class_sep: float = 1.5,
    flip_y: float = 0.01,
    random_state: int = 123,
):
    """
    Synthetic high-dimensional binary classification dataset used in notebook 04.
    """
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=int(n_samples),
        n_features=int(n_features),
        n_informative=int(n_informative),
        n_redundant=int(n_redundant),
        n_clusters_per_class=int(n_clusters_per_class),
        class_sep=float(class_sep),
        flip_y=float(flip_y),
        random_state=int(random_state),
    )
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


# ---------------------------------------------------------------------
# Synthetic "image-like" RGB shapes datasets (used by notebooks 05/06)
# ---------------------------------------------------------------------

def _subsample_classwise(X: np.ndarray, y: np.ndarray, n_total: int, *, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    classes = sorted(int(c) for c in np.unique(y))
    if len(classes) < 2:
        raise ValueError("Need at least 2 classes to subsample classwise.")

    per = [n_total // len(classes)] * len(classes)
    for i in range(n_total - sum(per)):
        per[i] += 1

    idx_all = []
    for cls, n_take in zip(classes, per):
        idx = np.where(y == cls)[0]
        if len(idx) < n_take:
            raise ValueError(f"Not enough samples for class {cls}: have={len(idx)} need={n_take}")
        idx_all.append(rng.choice(idx, size=int(n_take), replace=False))
    idx = np.concatenate(idx_all)
    rng.shuffle(idx)
    return X[idx], y[idx]


def load_synthetic_shapes_2class(
    *,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int = 0,
    image_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthetic 2-class RGB dataset: circles (0) vs squares (1).

    Returns arrays:
      X_*: float32, shape (N, 3, H, W) in [0,1]
      y_*: int labels
    """
    rng = np.random.default_rng(seed)
    total = int(n_train + n_val + n_test)
    H = W = int(image_size)
    X = np.zeros((total, 3, H, W), dtype=np.float32)
    y = np.zeros((total,), dtype=int)

    yy, xx = np.mgrid[0:H, 0:W]
    for i in range(total):
        label = int(i % 2)
        y[i] = label

        img = rng.uniform(0.0, 0.15, size=(H, W, 3)).astype(np.float32)
        cx = int(rng.integers(W // 4, 3 * W // 4))
        cy = int(rng.integers(H // 4, 3 * H // 4))
        size = int(rng.integers(H // 8, H // 5))

        if label == 0:
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= (size ** 2)
        else:
            mask = (np.abs(xx - cx) <= size) & (np.abs(yy - cy) <= size)

        color = rng.uniform(0.6, 1.0, size=(1, 1, 3)).astype(np.float32)
        img[mask] = np.clip(img[mask] + color, 0.0, 1.0)
        X[i] = img.transpose(2, 0, 1)

    X, y = _subsample_classwise(X, y, total, seed=seed)
    return (
        X[:n_train], y[:n_train],
        X[n_train:n_train + n_val], y[n_train:n_train + n_val],
        X[n_train + n_val:], y[n_train + n_val:],
    )


def _balanced_counts(n: int, n_classes: int) -> Sequence[int]:
    n = int(n)
    n_classes = int(n_classes)
    base = n // n_classes
    counts = [base] * n_classes
    for i in range(n - base * n_classes):
        counts[i] += 1
    return counts


def _triangle_mask(xx: np.ndarray, yy: np.ndarray, p0, p1, p2) -> np.ndarray:
    """Vectorized point-in-triangle test using signed areas."""
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2

    def _sign(px, py, ax, ay, bx, by):
        return (px - bx) * (ay - by) - (ax - bx) * (py - by)

    b0 = _sign(xx, yy, x0, y0, x1, y1)
    b1 = _sign(xx, yy, x1, y1, x2, y2)
    b2 = _sign(xx, yy, x2, y2, x0, y0)

    has_neg = (b0 < 0) | (b1 < 0) | (b2 < 0)
    has_pos = (b0 > 0) | (b1 > 0) | (b2 > 0)
    return ~(has_neg & has_pos)


def _make_shape_image(*, label: int, rng: np.random.Generator, H: int, W: int) -> np.ndarray:
    """Return HxWx3 image in [0,1]. label: 0=circle, 1=square, 2=triangle."""
    yy, xx = np.mgrid[0:H, 0:W]
    img = rng.uniform(0.0, 0.15, size=(H, W, 3)).astype(np.float32)
    cx = int(rng.integers(W // 4, 3 * W // 4))
    cy = int(rng.integers(H // 4, 3 * H // 4))

    if label == 0:  # circle
        r = int(rng.integers(H // 8, H // 5))
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= (r ** 2)
    elif label == 1:  # square
        s = int(rng.integers(H // 8, H // 5))
        mask = (np.abs(xx - cx) <= s) & (np.abs(yy - cy) <= s)
    elif label == 2:  # triangle
        h = int(rng.integers(H // 8, H // 4))
        w = int(rng.integers(H // 8, H // 3))

        p0 = (cx, cy - h)
        p1 = (cx - w, cy + h)
        p2 = (cx + w, cy + h)

        p0 = (int(np.clip(p0[0], 0, W - 1)), int(np.clip(p0[1], 0, H - 1)))
        p1 = (int(np.clip(p1[0], 0, W - 1)), int(np.clip(p1[1], 0, H - 1)))
        p2 = (int(np.clip(p2[0], 0, W - 1)), int(np.clip(p2[1], 0, H - 1)))
        mask = _triangle_mask(xx, yy, p0, p1, p2)
    else:
        raise ValueError(f"Unknown label: {label}")

    color = rng.uniform(0.6, 1.0, size=(1, 1, 3)).astype(np.float32)
    img[mask] = np.clip(img[mask] + color, 0.0, 1.0)
    return img


def load_synthetic_shapes_3class(
    *,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int = 0,
    image_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthetic 3-class RGB dataset: circle (0), square (1), triangle (2).
    """
    H = W = int(image_size)

    def _make_split(n: int, seed_offset: int):
        rng_local = np.random.default_rng(int(seed) + int(seed_offset))
        counts = _balanced_counts(n, 3)
        X_list = []
        y_list = []
        for cls, c in enumerate(counts):
            for _ in range(int(c)):
                X_list.append(_make_shape_image(label=cls, rng=rng_local, H=H, W=W).transpose(2, 0, 1))
                y_list.append(cls)
        X = np.stack(X_list).astype(np.float32)
        y = np.asarray(y_list, dtype=int)
        idx = rng_local.permutation(len(X))
        return X[idx], y[idx]

    X_train, y_train = _make_split(int(n_train), 1)
    X_val, y_val = _make_split(int(n_val), 2)
    X_test, y_test = _make_split(int(n_test), 3)
    return X_train, y_train, X_val, y_val, X_test, y_test



# ---------------------------------------------------------------------
# Synthetic "Point Cloud"  datasets Doubletorus ,nested spheres and blobs within sphere
# ---------------------------------------------------------------------


def generate_geometrical_dataset(dataset_type, n_points=None, noise=0.1, random_state=42):
    np.random.seed(random_state)
    
    if dataset_type == 'torus_one_hole':
        # Your first torus code here
        points, labels = gen_torus_one_hole(noise)
        
    elif dataset_type == 'torus_two_holes':
        # Your second torus code here
        points, labels = gen_torus_two_holes(noise)
        
    elif dataset_type == 'nested_spheres':
        # Your sphere code here
        if n_points is None:
            n_points = 6000
        points, labels = gen_nested_spheres(n_points, noise)

    elif dataset_type == 'Blobs':
        # Your sphere code here
        if n_points is None:
            n_points = 6000
        points, labels = Blobs(n_points, noise)
    
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    X_train, X_t, y_train, y_t = train_test_split(
        points, labels, 
        test_size=0.3, 
        stratify=labels,  
        random_state=42
    )

    print(X_train.shape)
    print(X_t.shape)
    print(y_train.shape)
    print(y_t.shape)

    X_val, X_test, y_val, y_test = train_test_split(
        X_t, y_t, 
        test_size=0.5,  
        stratify=y_t, 
        random_state=42
    )
    return X_train, y_train, X_val, y_val, X_test, y_test

def gen_torus_one_hole(noise=0.1):
    """Generate double torus + torus through one hole."""
    R = 4
    r = 1
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, 2*np.pi, 40)
    u, v = np.meshgrid(u, v)
    
    #Double torus
    x=(R+r*np.cos(v))*np.cos(u)-2.8
    y=(R+r*np.cos(v))*np.sin(u)-2.8
    z=r*np.sin(v)

    noise_x = noise * np.random.randn(*x.shape)
    noise_y = noise * np.random.randn(*y.shape)
    noise_z = noise * np.random.randn(*z.shape)
    
    # Double torus (class 0)
    x1 = (R + r*np.cos(v)) * np.cos(u) - 2.8 + noise_x
    y1 = (R + r*np.cos(v)) * np.sin(u) - 2.8 + noise_y
    z1 = r*np.sin(v) + noise_z
    
    x2 = (R + r*np.cos(v)) * np.cos(u) + 2.8 + noise_x
    y2 = (R + r*np.cos(v)) * np.sin(u) + 2.8 + noise_y
    z2 = r*np.sin(v) + noise_z
    
    points1 = np.column_stack([x1.ravel(), y1.ravel(), z1.ravel()])
    points2 = np.column_stack([x2.ravel(), y2.ravel(), z2.ravel()])
    
    # Single torus (class 1) 
    x3 = r*np.sin(v) + noise_x -4
    y3 = (R + r*np.cos(v)) * np.cos(u) + noise_y -6
    z3 = (R + r*np.cos(v)) * np.sin(u) + noise_z
    points3 = np.column_stack([x3.ravel(), y3.ravel(), z3.ravel()])
    
    
    # Combine
    all_points = np.vstack([points1, points2, points3])
    labels = np.concatenate([
        np.zeros(len(points1)),
        np.zeros(len(points2)),
        np.ones(len(points3))
    ]).astype(int)
    
    return all_points, labels

def gen_torus_two_holes(noise=0.1):
    """Generate double torus + torus through both holes."""
    R = 4
    r = 1
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, 2*np.pi, 40)
    u, v = np.meshgrid(u, v)
    
    #Double torus
    x=(R+r*np.cos(v))*np.cos(u)-2.8
    y=(R+r*np.cos(v))*np.sin(u)-2.8
    z=r*np.sin(v)

    noise_x = noise * np.random.randn(*x.shape)
    noise_y = noise * np.random.randn(*y.shape)
    noise_z = noise * np.random.randn(*z.shape)
    
    # Double torus (class 0)
    x1 = (R + r*np.cos(v)) * np.cos(u) - 2.8 + noise_x
    y1 = (R + r*np.cos(v)) * np.sin(u) - 2.8 + noise_y
    z1 = r*np.sin(v) + noise_z
    
    x2 = (R + r*np.cos(v)) * np.cos(u) + 2.8 + noise_x
    y2 = (R + r*np.cos(v)) * np.sin(u) + 2.8 + noise_y
    z2 = r*np.sin(v) + noise_z
    
    points1 = np.column_stack([x1.ravel(), y1.ravel(), z1.ravel()])
    points2 = np.column_stack([x2.ravel(), y2.ravel(), z2.ravel()])
    
    # Single torus (class 1) - rotated
    x3 = r*np.sin(v) + noise_x
    y3 = (R + r*np.cos(v)) * np.cos(u) + noise_y
    z3 = (R + r*np.cos(v)) * np.sin(u) + noise_z
    points3 = np.column_stack([x3.ravel(), y3.ravel(), z3.ravel()])
    
    # Rotate
    theta = np.pi/2 + np.pi/3
    Z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    points3 = points3 @ Z.T
    
    # Combine
    all_points = np.vstack([points1, points2, points3])
    labels = np.concatenate([
        np.zeros(len(points1)),
        np.zeros(len(points2)),
        np.ones(len(points3))
    ]).astype(int)
    
    return all_points, labels

def gen_nested_spheres(n_points=6000, noise=0.1, R=4):
    # Sample sphere
    points = np.random.randn(n_points, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    radii = R * np.random.rand(n_points)**(1/3)
    points = points * radii[:, np.newaxis]
    
    # Add noise
    noise_array = noise * np.random.randn(*points.shape)
    points = points + noise_array
    
    # Define inclusions
    center1 = np.array([0.0, 0.0, 0.0])

    r_class2 = 3
    r_class1 = 1
    
    distances_to_center1 = np.linalg.norm(points - center1, axis=1)
    
    
    # Classify
    labels = np.ones(len(points), dtype=int)
    labels[(distances_to_center1 < r_class2) ]=0
    labels[distances_to_center1 < r_class1] = 1
    
    return points, labels

def Blobs(n_points=6000, noise=0.1, R=4):
    # Sample sphere
    points = np.random.randn(n_points, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    radii = R * np.random.rand(n_points)**(1/3)
    points = points * radii[:, np.newaxis]
    
    # Add noise
    noise_array = noise * np.random.randn(*points.shape)
    points = points + noise_array
    
    # Define inclusions
    center1 = np.array([2.0, 0.0, 0.0])
    center2 = np.array([-2.0, 1.0, 0.0])
    r_class2 = 1.5
    
    distances_to_center1 = np.linalg.norm(points - center1, axis=1)
    distances_to_center2 = np.linalg.norm(points - center2, axis=1)
    
    # Classify
    labels = np.ones(len(points), dtype=int)
    labels[(distances_to_center1 < r_class2) | (distances_to_center2 < r_class2)] = 0

    
    return points, labels

