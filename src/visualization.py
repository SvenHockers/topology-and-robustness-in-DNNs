"""
Visualization functions for experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
from typing import Optional, Tuple, Mapping, Any


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
    ax: Optional[plt.Axes] = None,
    *,
    threshold: Optional[float] = None,
    labels: Tuple[str, str] = ("Clean", "Adversarial"),
    alpha: float = 0.5,
    density: bool = True,
    colors: Tuple[str, str] = ("tab:blue", "tab:orange"),
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
        threshold: Optional score threshold to draw as a vertical line
        labels: Legend labels for (clean, adversarial)
        alpha: Histogram alpha
        density: Whether to plot density-normalized histograms
        colors: Histogram colors for (clean, adversarial)
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(
        np.asarray(scores_clean, dtype=float).ravel(),
        bins=bins,
        alpha=alpha,
        label=labels[0],
        density=density,
        color=colors[0],
    )
    ax.hist(
        np.asarray(scores_adv, dtype=float).ravel(),
        bins=bins,
        alpha=alpha,
        label=labels[1],
        density=density,
        color=colors[1],
    )

    thr = float(threshold) if threshold is not None else np.nan
    if np.isfinite(thr):
        ax.axvline(thr, color="k", linestyle="--", linewidth=1, label=f"thr={thr:.3f}")
    
    ax.set_xlabel(score_name)
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_score_distributions_figure(
    scores_clean: np.ndarray,
    scores_adv: np.ndarray,
    *,
    score_name: str = "Score",
    title: str = "Score Distributions",
    bins: int = 50,
    threshold: Optional[float] = None,
    labels: Tuple[str, str] = ("Clean", "Adversarial"),
    alpha: float = 0.5,
    density: bool = True,
    colors: Tuple[str, str] = ("tab:blue", "tab:orange"),
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (7, 4),
    show: bool = True,
):
    """
    Standardized wrapper around plot_score_distributions (mirrors plot_roc_from_metrics).

    Returns (fig, ax).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    plot_score_distributions(
        scores_clean,
        scores_adv,
        score_name=score_name,
        title=title,
        bins=bins,
        ax=ax,
        threshold=threshold,
        labels=labels,
        alpha=alpha,
        density=density,
        colors=colors,
    )

    if show:
        fig.tight_layout()
        plt.show()

    return fig, ax


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: Optional[float] = None,
    title: str = "ROC Curve",
    ax: Optional[plt.Axes] = None,
    interpolate: bool = False,
    n_points: int = 200,
    label: Optional[str] = None,
    *,
    auc_score: Optional[float] = None,
) -> plt.Axes:
    """
    Plot ROC curve.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        roc_auc: AUC score (kept for backwards compatibility; positional arg #3)
        title: Plot title
        ax: Optional axes to plot on
        interpolate: Whether to interpolate for smoother plotting
        n_points: Number of interpolation points when interpolate=True
        label: Optional legend label (defaults to AUC label)
        auc_score: Alias for roc_auc (keyword-only)
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    auc = float(auc_score if auc_score is not None else roc_auc) if (auc_score is not None or roc_auc is not None) else None

    if interpolate and len(fpr) > 1:
        grid = np.linspace(0.0, 1.0, int(n_points))
        # roc_curve output is monotone in fpr; interpolate tpr for smoother plotting
        tpr_i = np.interp(grid, fpr, tpr)
        if label is None:
            label = f"ROC curve (AUC = {auc:.3f})" if auc is not None else "ROC curve"
        ax.plot(grid, tpr_i, linewidth=2, label=label)
    else:
        if label is None:
            label = f"ROC curve (AUC = {auc:.3f})" if auc is not None else "ROC curve"
        ax.plot(fpr, tpr, linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_roc_from_metrics(
    metrics: Mapping[str, Any],
    *,
    title: str = "ROC curve",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    interpolate: bool = False,
    n_points: int = 200,
):
    """
    Plot an ROC curve from the dict returned by src.evaluation.evaluate_detector.

    Returns (fig, ax).
    """
    fpr = np.asarray(metrics.get("fpr"), dtype=float)
    tpr = np.asarray(metrics.get("tpr"), dtype=float)
    roc_auc = metrics.get("roc_auc", None)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    plot_roc_curve(
        fpr,
        tpr,
        roc_auc,
        title=title,
        ax=ax,
        interpolate=interpolate,
        n_points=n_points,
    )

    if show:
        fig.tight_layout()
        plt.show()

    return fig, ax


def plot_confusion_matrix(
    y_true,
    *,
    y_pred=None,
    y_scores=None,
    threshold: Optional[float] = None,
    labels: tuple = ("clean", "adv"),
    ax: Optional[plt.Axes] = None,
    show: bool = True,
):
    """
    Plot TP/FP/TN/FN confusion matrix (counts + row-normalized).

    Intended for binary problems like:
      - 0 = clean, 1 = adversarial

    You can pass either:
      - y_true + y_pred
      - y_true + y_scores + threshold  (y_pred computed as y_scores >= threshold)
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    if y_pred is None:
        if y_scores is None or threshold is None:
            raise ValueError("Provide either y_pred, or (y_scores and threshold).")
        y_scores = np.asarray(y_scores, dtype=float).ravel()
        y_pred = (y_scores >= float(threshold)).astype(int)
    else:
        y_pred = np.asarray(y_pred, dtype=int).ravel()

    # Counts
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    cm = np.array([[tn, fp], [fn, tp]], dtype=float)
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    else:
        # If a single axis is provided, draw counts only.
        fig = ax.figure
        axes = [ax]

    # Left: counts
    ax0 = axes[0]
    ax0.imshow(cm, cmap="Blues")
    ax0.set_title("Confusion (counts)")
    ax0.set_xticks([0, 1]); ax0.set_yticks([0, 1])
    ax0.set_xticklabels([f"pred {labels[0]}", f"pred {labels[1]}"])
    ax0.set_yticklabels([f"true {labels[0]}", f"true {labels[1]}"])
    for (i, j), val in np.ndenumerate(cm):
        ax0.text(j, i, f"{int(val)}", ha="center", va="center")

    # Right: normalized
    if len(axes) > 1:
        ax1 = axes[1]
        ax1.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
        ax1.set_title("Confusion (row-normalized)")
        ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1])
        ax1.set_xticklabels([f"pred {labels[0]}", f"pred {labels[1]}"])
        ax1.set_yticklabels([f"true {labels[0]}", f"true {labels[1]}"])
        for (i, j), val in np.ndenumerate(cm_norm):
            ax1.text(j, i, f"{val:.2f}", ha="center", va="center")

    if show:
        fig.tight_layout()
        plt.show()

    return {
        "fig": fig,
        "axes": axes,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "cm": cm,
        "cm_norm": cm_norm,
    }


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


