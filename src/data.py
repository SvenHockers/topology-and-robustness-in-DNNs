"""
Dataset generation and data loaders for two moons dataset.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
from .utils import DataConfig


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
    # Validate split ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Train, validation, and test ratios must sum to 1.0"
    
    # Generate two moons dataset
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    
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
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def generate_two_moons_from_config(config: DataConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate two moons dataset using a DataConfig object.
    
    Args:
        config: DataConfig instance with dataset parameters
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test) as numpy arrays
    """
    return generate_two_moons(
        n_samples=config.n_samples,
        noise=config.noise,
        random_state=config.random_state,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio
    )


class TwoMoonsDataset(Dataset):
    """PyTorch Dataset wrapper for two moons data."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            X: Feature array of shape (n_samples, n_features)
            y: Label array of shape (n_samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


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
    train_dataset = TwoMoonsDataset(X_train, y_train)
    val_dataset = TwoMoonsDataset(X_val, y_val)
    test_dataset = TwoMoonsDataset(X_test, y_test)
    
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


