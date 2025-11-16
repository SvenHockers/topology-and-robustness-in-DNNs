from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from persim import plot_diagrams
from ripser import ripser
from .plot_style import new_figure


def _preprocess_points(points: np.ndarray, normalize: str, pca_dim: Optional[int]) -> np.ndarray:
    """
    points: (num_points, num_features)
    normalize: 'none' | 'zscore' | 'l2'
    pca_dim: reduce features via SVD to pca_dim if provided
    """
    X = points
    if normalize == "zscore":
        mu = X.mean(axis=0, keepdims=True)
        sigma = X.std(axis=0, keepdims=True) + 1e-12
        X = (X - mu) / sigma
    elif normalize == "l2":
        norms = np.linalg.norm(X, ord=2, axis=1, keepdims=True) + 1e-12
        X = X / norms
    if pca_dim is not None and pca_dim > 0 and X.shape[1] > pca_dim:
        # SVD-based PCA
        Xc = X - X.mean(axis=0, keepdims=True)
        # economy SVD
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        Vt_k = Vt[:pca_dim, :]
        X = Xc @ Vt_k.T
    return X


def compute_layer_topology(
    activations: torch.Tensor | np.ndarray,
    sample_size: int = 100,
    maxdim: int = 1,
    normalize: str = "none",
    pca_dim: Optional[int] = None,
    bootstrap_repeats: int = 1,
) -> Optional[List[np.ndarray]]:
    """
    Compute persistence diagrams for layer activations.

    Args:
        activations: tensor of shape (batch, points, features) or (batch, features)
        sample_size: subsample points if too many
        maxdim: maximum homology dimension

    Returns:
        Persistence diagram list as returned by ripser (for the last bootstrap); if bootstrap>1,
        caller should aggregate externally or call this multiple times.
    """
    # Convert to numpy
    if isinstance(activations, torch.Tensor):
        activations = activations.numpy()

    # Handle different shapes
    if len(activations.shape) == 3:  # (B, N, F)
        # Flatten batch and points: (B*N, F)
        _, _, n_features = activations.shape
        activations = activations.reshape(-1, n_features)
    elif len(activations.shape) == 2:  # (B, F)
        pass
    else:
        raise ValueError(f"Unexpected activation shape: {activations.shape}")

    # Bootstrap repeats: return last; typical use is multiple calls for noise-floor or averaging
    dgm_last = None
    for _ in range(max(1, bootstrap_repeats)):
        X = activations
        # Subsample if too many points
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X = X[indices]
        # Normalize / PCA
        X = _preprocess_points(X, normalize=normalize, pca_dim=pca_dim)
        # Compute persistence
        try:
            result = ripser(X, maxdim=maxdim)
            dgm_last = result['dgms']
        except Exception as e:
            print(f"Error computing persistence: {e}")
            dgm_last = None
    return dgm_last


def extract_persistence_stats(dgm) -> Dict[str, float]:
    """Extract summary statistics from a persistence diagram."""
    stats: Dict[str, float] = {}

    for dim, diagram in enumerate(dgm):
        # Remove infinite values
        finite_dgm = diagram[diagram[:, 1] != np.inf]

        if len(finite_dgm) == 0:
            stats[f'H{dim}_count'] = 0.0
            stats[f'H{dim}_mean_persistence'] = 0.0
            stats[f'H{dim}_max_persistence'] = 0.0
            stats[f'H{dim}_total_persistence'] = 0.0
            stats[f'H{dim}_entropy'] = 0.0
        else:
            persistence = finite_dgm[:, 1] - finite_dgm[:, 0]
            stats[f'H{dim}_count'] = float(len(finite_dgm))
            stats[f'H{dim}_mean_persistence'] = float(np.mean(persistence))
            stats[f'H{dim}_max_persistence'] = float(np.max(persistence))
            stats[f'H{dim}_total_persistence'] = float(np.sum(persistence))
            # Persistent entropy
            L = persistence
            total = float(np.sum(L))
            if total > 0:
                p = (L / total).clip(min=1e-12)
                entropy = float(-np.sum(p * np.log(p)))
            else:
                entropy = 0.0
            stats[f'H{dim}_entropy'] = entropy

    return stats


def analyze_layer_topology(model, data_loader, device, sample_batch: int = 0) -> Dict[str, Dict[str, float]]:
    """
    Analyze topology of representations at each layer.

    Args:
        model: trained model
        data_loader: DataLoader with data
        device: torch device
        sample_batch: which batch to analyze in detail

    Returns:
        Dictionary with topology stats per layer
    """
    model.eval()
    topology_stats: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(data_loader):
            x = x.to(device)

            # Forward pass with layer saving
            _ = model(x, save_layers=True)

            # Analyze each layer
            for layer_name, activations in model.layer_outputs.items():
                # Compute persistence diagram
                dgm = compute_layer_topology(activations, sample_size=200, maxdim=1)

                if dgm is not None:
                    stats = extract_persistence_stats(dgm)
                    topology_stats[layer_name].append(stats)

            # Only analyze a few batches to save time
            if batch_idx >= 2:
                break

    # Average statistics across batches
    avg_stats: Dict[str, Dict[str, float]] = {}
    for layer_name, stats_list in topology_stats.items():
        avg_stats[layer_name] = {}
        if stats_list:
            keys = stats_list[0].keys()
            for key in keys:
                values = [s[key] for s in stats_list]
                avg_stats[layer_name][key] = float(np.mean(values))

    return avg_stats


def visualize_layer_topology(model, data_loader, device, model_name: str):
    model.eval()

    with torch.no_grad():
        # Get one batch
        x, _ = next(iter(data_loader))
        x = x.to(device)
        _ = model(x, save_layers=True)

        # Get layer names (excluding output)
        layer_names = [k for k in model.layer_outputs.keys() if k != 'output']
        n_layers = len(layer_names)

        fig, axes = new_figure(kind="custom", figsize=(4 * n_layers, 8), nrows=2, ncols=n_layers)
        if n_layers == 1:
            axes = axes.reshape(2, 1)

        for idx, layer_name in enumerate(layer_names):
            activations = model.layer_outputs[layer_name]

            # Compute persistence
            dgm = compute_layer_topology(activations, sample_size=200, maxdim=1)

            if dgm is not None:
                # Plot persistence diagram
                ax = axes[0, idx]
                plot_diagrams(dgm, ax=ax, show=False)
                ax.set_title(f'{layer_name}')

                # Plot Betti numbers
                ax = axes[1, idx]
                stats = extract_persistence_stats(dgm)
                betti_numbers = [stats.get(f'H{i}_count', 0) for i in range(2)]
                ax.bar(['$H_0$', '$H_1$'], betti_numbers)
                ax.set_ylabel('Count')
                ax.set_title(f'{layer_name} - Betti Numbers')

        fig.tight_layout()
        fig.savefig(f'{model_name}_layer_topology.png', dpi=150, bbox_inches='tight')
        print(f"Saved layer topology visualization to '{model_name}_layer_topology.png'")
        plt.close(fig)


