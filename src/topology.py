from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from persim import plot_diagrams
from ripser import ripser


def compute_layer_topology(activations: torch.Tensor | np.ndarray, sample_size: int = 100, maxdim: int = 1):
    """
    Compute persistence diagrams for layer activations.

    Args:
        activations: tensor of shape (batch, points, features) or (batch, features)
        sample_size: subsample points if too many
        maxdim: maximum homology dimension

    Returns:
        Persistence diagram list as returned by ripser
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

    # Subsample if too many points
    if len(activations) > sample_size:
        indices = np.random.choice(len(activations), sample_size, replace=False)
        activations = activations[indices]

    # Compute persistence
    try:
        result = ripser(activations, maxdim=maxdim)
        return result['dgms']
    except Exception as e:
        print(f"Error computing persistence: {e}")
        return None


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
        else:
            persistence = finite_dgm[:, 1] - finite_dgm[:, 0]
            stats[f'H{dim}_count'] = float(len(finite_dgm))
            stats[f'H{dim}_mean_persistence'] = float(np.mean(persistence))
            stats[f'H{dim}_max_persistence'] = float(np.max(persistence))
            stats[f'H{dim}_total_persistence'] = float(np.sum(persistence))

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

        fig, axes = plt.subplots(2, n_layers, figsize=(4 * n_layers, 8))
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
                ax.bar(['H0', 'H1'], betti_numbers)
                ax.set_ylabel('Count')
                ax.set_title(f'{layer_name} - Betti Numbers')

        plt.tight_layout()
        plt.savefig(f'{model_name}_layer_topology.png', dpi=150, bbox_inches='tight')
        print(f"Saved layer topology visualization to '{model_name}_layer_topology.png'")
        plt.close()


