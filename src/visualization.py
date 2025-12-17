"""
Visualization functions for experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
from typing import Optional, Tuple


def plot_two_moons(
    X: np.ndarray,
    y: np.ndarray,
    title: str = "Two Moons Dataset",
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot two moons dataset colored by class.
    
    Args:
        X: Feature array
        y: Labels
        title: Plot title
        ax: Optional axes to plot on
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, s=50, alpha=0.6)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax, label='Class')
    
    return ax


def plot_decision_boundary(
    model,
    X: np.ndarray,
    y: np.ndarray,
    title: str = "Decision Boundary",
    device: str = 'cpu',
    resolution: int = 100,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot decision boundary of the model overlaid on data.
    
    Args:
        model: Trained PyTorch model
        X: Feature array
        y: Labels
        title: Plot title
        device: Device for model inference
        resolution: Resolution of decision boundary grid
        ax: Optional axes to plot on
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    
    # Flatten grid and make predictions
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.FloatTensor(grid_points).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(grid_tensor)
        probs = torch.softmax(outputs, dim=1)
        Z = probs[:, 1].cpu().numpy()  # Probability of class 1
    
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    contour = ax.contourf(xx, yy, Z, levels=50, cmap=plt.cm.RdYlBu, alpha=0.5)
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    
    # Plot data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, 
                        s=50, alpha=0.8, edgecolors='black', linewidths=1)
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title)
    plt.colorbar(contour, ax=ax, label='Probability of Class 1')
    
    return ax


def plot_score_distributions(
    scores_clean: np.ndarray,
    scores_adv: np.ndarray,
    score_name: str = "Score",
    title: str = "Score Distributions",
    bins: int = 50,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot histogram/KDE of scores for clean vs adversarial examples.
    
    Args:
        scores_clean: Scores for clean examples
        scores_adv: Scores for adversarial examples
        score_name: Name of the score for axis label
        title: Plot title
        bins: Number of histogram bins
        ax: Optional axes to plot on
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(scores_clean, bins=bins, alpha=0.5, label='Clean', density=True, color='blue')
    ax.hist(scores_adv, bins=bins, alpha=0.5, label='Adversarial', density=True, color='red')
    
    ax.set_xlabel(score_name)
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    title: str = "ROC Curve",
    ax: Optional[plt.Axes] = None,
    interpolate: bool = False,
    n_points: int = 200
) -> plt.Axes:
    """
    Plot ROC curve.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc_score: AUC score
        title: Plot title
        ax: Optional axes to plot on
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    if interpolate and len(fpr) > 1:
        grid = np.linspace(0.0, 1.0, int(n_points))
        # roc_curve output is monotone in fpr; interpolate tpr for smoother plotting
        tpr_i = np.interp(grid, fpr, tpr)
        ax.plot(grid, tpr_i, linewidth=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    else:
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_score_scatter(
    X: np.ndarray,
    scores: np.ndarray,
    title: str = "Score Visualization",
    cmap: str = 'viridis',
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot 2D scatter of data points colored by score values.
    
    Args:
        X: 2D feature array
        scores: Score values to color by
        title: Plot title
        cmap: Colormap name
        ax: Optional axes to plot on
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(X[:, 0], X[:, 1], c=scores, cmap=cmap, s=50, alpha=0.6)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax, label='Score')
    
    return ax


def plot_adversarial_examples(
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    y: np.ndarray,
    title: str = "Clean vs Adversarial Examples",
    n_samples: int = 100,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot clean and adversarial examples with arrows showing perturbations.
    
    Args:
        X_clean: Clean examples
        X_adv: Adversarial examples
        y: Labels
        title: Plot title
        n_samples: Number of samples to visualize (for clarity)
        ax: Optional axes to plot on
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sample subset if too many points
    if len(X_clean) > n_samples:
        indices = np.random.choice(len(X_clean), n_samples, replace=False)
        X_clean = X_clean[indices]
        X_adv = X_adv[indices]
        y = y[indices]
    
    # Plot clean points
    scatter_clean = ax.scatter(X_clean[:, 0], X_clean[:, 1], c=y, 
                              cmap=plt.cm.RdYlBu, s=50, alpha=0.6, 
                              marker='o', label='Clean')
    
    # Plot adversarial points
    scatter_adv = ax.scatter(X_adv[:, 0], X_adv[:, 1], c=y,
                            cmap=plt.cm.RdYlBu, s=50, alpha=0.6,
                            marker='x', label='Adversarial')
    
    # Draw arrows showing perturbations
    for i in range(len(X_clean)):
        dx = X_adv[i, 0] - X_clean[i, 0]
        dy = X_adv[i, 1] - X_clean[i, 1]
        ax.arrow(X_clean[i, 0], X_clean[i, 1], dx, dy,
                head_width=0.05, head_length=0.05, fc='gray', ec='gray', alpha=0.3)
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title)
    ax.legend()
    
    return ax


