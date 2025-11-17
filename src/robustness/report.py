from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple
import csv
import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import cast
from matplotlib.axes import Axes
from ..plot_style import (
    new_figure,
    setup_axes,
    setup_bar_axes,
    setup_heatmap_axes,
    setup_line_axes,
    setup_scatter_axes,
    setup_violin_axes,
	get_heatmap_cmap,
)


@dataclass
class SampleRecord:
    sample_id: int
    true_label: int
    clean_pred: int
    clean_margin: float
    eps_star_linf: Optional[float] = None
    adv_pred_linf: Optional[int] = None
    eps_star_l2: Optional[float] = None
    adv_pred_l2: Optional[int] = None
    rot_deg_star: Optional[float] = None
    trans_x_star: Optional[float] = None
    trans_y_star: Optional[float] = None
    trans_z_star: Optional[float] = None
    jitter_std_star: Optional[float] = None
    dropout_star: Optional[float] = None
    alpha_star: Optional[float] = None
    chamfer_clean_adv: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LayerwiseRecord:
    sample_id: int
    condition: str  # 'clean','adv_linf_eps=0.1', 'rotation_deg=10', ...
    layer: str
    betti: str  # 'H0','H1'
    count: float
    mean_persistence: float
    max_persistence: float
    total_persistence: float

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiagramDistanceRecord:
    sample_id: int
    condition: str
    layer: str
    metric: str  # 'wasserstein' | 'bottleneck'
    H: int
    distance: float

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RobustnessSummary:
    # aggregates
    mean_eps_star_linf: Optional[float] = None
    mean_eps_star_l2: Optional[float] = None
    ra_curves: Dict[str, List[float]] = field(default_factory=dict)  # norm -> list over eps grid
    per_class: Dict[str, Any] = field(default_factory=dict)
    layerwise_avg: Dict[str, Any] = field(default_factory=dict)
    correlations: Dict[str, Any] = field(default_factory=dict)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


def write_csv(records: List[Dict[str, Any]], path: str) -> None:
    if not records:
        return
    keys = sorted(records[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in records:
            w.writerow(r)


def save_curve_png(xs: List[float], ys: List[float], title: str, path: str) -> None:
    fig, ax = new_figure(kind="single")
    ax = cast(Axes, ax)
    ax.plot(xs, ys, marker="o")
    # Use LaTeX for epsilon
    setup_line_axes(ax, xlabel=r"$\epsilon$", ylabel="Robust accuracy", title=title.replace("epsilon", r"$\epsilon$"))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt_close(fig=fig)


def save_hist_png(values: List[float], title: str, path: str, bins: int = 30) -> None:
    fig, ax = new_figure(kind="single")
    ax = cast(Axes, ax)
    clean_vals = [v for v in values if v is not None and not np.isnan(v)]
    ax.hist(clean_vals, bins=bins, alpha=0.8)
    # Build a LaTeX-safe title. If both eps* and a norm are present, compose a single math block to avoid nested $...$.
    t_low = title.lower()
    has_eps = "eps" in t_low or "epsilon" in t_low
    norm_label = None
    if "linf" in t_low or "l_inf" in t_low:
        norm_label = r"\ell_\infty"
    elif "l2" in t_low or "l_2" in t_low:
        norm_label = r"\ell_2"
    if has_eps and norm_label is not None:
        latex_title = rf"$\epsilon^\star\ ({norm_label})$"
    elif has_eps:
        latex_title = r"$\epsilon^\star$"
    else:
        latex_title = title
    # Axis labels
    xlabel = r"$\epsilon^\star$" if has_eps else None
    setup_axes(ax, xlabel=xlabel, ylabel="Count", title=latex_title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt_close(fig=fig)


def save_layer_distance_bar(avg_by_layer: Dict[str, float], title: str, path: str) -> None:
    layers = list(avg_by_layer.keys())
    vals = [avg_by_layer[k] for k in layers]
    fig, ax = new_figure(kind="custom", figsize=(max(6, len(layers)), 4))
    ax = cast(Axes, ax)
    ax.bar(layers, vals)
    setup_bar_axes(ax, title=title, rotate_xticks=True, rotation=45.0)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt_close(fig=fig)


def save_betti_counts_bar(layerwise_records: List[LayerwiseRecord], H: int, path: str, title: str | None = None) -> None:
    from collections import defaultdict
    agg = defaultdict(list)
    for r in layerwise_records:
        if r.betti == f"H{H}":
            agg[r.layer].append(r.count)
    if not agg:
        return
    layers = sorted(agg.keys())
    vals = [float(sum(agg[l]) / max(len(agg[l]), 1)) for l in layers]
    fig, ax = new_figure(kind="custom", figsize=(max(6, len(layers)), 4))
    ax = cast(Axes, ax)
    ax.bar(layers, vals)
    setup_bar_axes(ax, title=(title or f"Average Betti $H_{{{H}}}$ counts per layer"), rotate_xticks=True, rotation=45.0)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt_close(fig=fig)


def save_distance_heatmap(
    diagdist_records: List[DiagramDistanceRecord],
    metric: str,
    H: int,
    path: str,
    title: Optional[str] = None,
) -> None:
    from collections import defaultdict, OrderedDict
    # aggregate mean distance per (condition, layer)
    agg: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    layers_set, conds_set = set(), set()
    for r in diagdist_records:
        if r.metric != metric or r.H != H:
            continue
        if r.condition in {"noise_floor"}:
            continue
        if r.distance is None or (isinstance(r.distance, float) and np.isnan(r.distance)):
            continue
        agg[(r.condition, r.layer)].append(r.distance)
        layers_set.add(r.layer)
        conds_set.add(r.condition)
    if not agg:
        return
    layers = sorted(layers_set)
    conds = sorted(conds_set)
    M: np.ndarray = np.zeros((len(conds), len(layers)))
    for i, c in enumerate(conds):
        for j, l in enumerate(layers):
            vals = agg.get((c, l), [])
            M[i, j] = np.mean(vals) if vals else np.nan
    fig, ax = new_figure(kind="custom", figsize=(max(6, 1.2 * len(layers)), max(4, 0.6 * len(conds))))
    ax = cast(Axes, ax)
    im = ax.imshow(M, aspect="auto", interpolation="nearest", cmap=get_heatmap_cmap("sequential"))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers)
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels(conds)
    setup_heatmap_axes(ax, title=(title or f"{metric} $H_{H}$ distances (mean)"), rotate_xticks=True, rotation=45.0)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt_close(fig=fig)


def save_distance_heatmap_normalized(
    diagdist_records: List[DiagramDistanceRecord],
    metric: str,
    H: int,
    path: str,
    title: Optional[str] = None,
) -> None:
    from collections import defaultdict
    # mean noise floor per layer
    noise: Dict[str, List[float]] = defaultdict(list)
    for r in diagdist_records:
        if r.metric == metric and r.H == H and r.condition == "noise_floor":
            if r.distance is not None and not (isinstance(r.distance, float) and np.isnan(r.distance)):
                noise[r.layer].append(r.distance)
    noise_mean = {layer: (float(np.mean(vals)) if vals else 0.0) for layer, vals in noise.items()}
    # aggregate conditions
    layers = sorted({r.layer for r in diagdist_records})
    conds = sorted({r.condition for r in diagdist_records if r.condition != "noise_floor"})
    if not layers or not conds:
        return
    M: np.ndarray = np.full((len(conds), len(layers)), np.nan, dtype=float)
    for i, c in enumerate(conds):
        for j, l in enumerate(layers):
            vals = [r.distance for r in diagdist_records if r.metric == metric and r.H == H and r.layer == l and r.condition == c]
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
            if vals:
                v = float(np.mean(vals))
                v = max(0.0, v - noise_mean.get(l, 0.0))  # subtract noise floor
                M[i, j] = v
    fig, ax = new_figure(kind="custom", figsize=(max(6, 1.2 * len(layers)), max(4, 0.6 * len(conds))))
    ax = cast(Axes, ax)
    im = ax.imshow(M, aspect="auto", interpolation="nearest", cmap=get_heatmap_cmap("sequential"))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers)
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels(conds)
    # annotate
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    setup_heatmap_axes(ax, title=(title or f"{metric} $H_{H}$ (mean minus noise floor)"), rotate_xticks=True, rotation=45.0)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt_close(fig=fig)


def save_scatter_eps_vs_distance(
    sample_records: List[SampleRecord],
    diagdist_records: List[DiagramDistanceRecord],
    norm: str,
    layer: str,
    metric: str,
    H: int,
    condition_prefix: Optional[str] = None,
    path: str = "scatter_eps_vs_distance.png",
) -> None:
    # pick conditions that match e.g., adv_linf_eps=...
    cond_prefix = condition_prefix or f"adv_{norm}_eps="
    # Map distances per sample_id for this layer and conditions
    dist_by_sample: Dict[int, float] = {}
    for r in diagdist_records:
        if r.metric == metric and r.H == H and r.layer == layer and r.condition.startswith(cond_prefix):
            if r.distance is not None and not (isinstance(r.distance, float) and np.isnan(r.distance)):
                # If multiple eps per sample, prefer the largest eps (more effect)
                dist_by_sample[r.sample_id] = r.distance
    # Collect eps* from sample records
    xs, ys = [], []
    for s in sample_records:
        eps_star = s.eps_star_linf if norm == "linf" else s.eps_star_l2
        if eps_star is None:
            continue
        if s.sample_id in dist_by_sample:
            xs.append(float(eps_star))
            ys.append(float(dist_by_sample[s.sample_id]))
    if not xs:
        return
    X = np.array(xs); Y = np.array(ys)
    # add small jitter for readability when many points stack
    jitter = 1e-4
    Xj = X + np.random.randn(*X.shape) * 0.0  # no jitter in x by default
    Yj = Y + np.random.randn(*Y.shape) * jitter
    # correlation (Pearson) guarded
    if len(X) < 3 or np.std(X) == 0 or np.std(Y) == 0:
        rho = float("nan")
    else:
        C = cast(np.ndarray, np.corrcoef(X, Y))
        rho = float(C[0, 1])
    fig, ax = new_figure(kind="single")
    ax = cast(Axes, ax)
    ax.scatter(Xj, Yj, s=14, alpha=0.7)
    title_txt = f"eps* vs distance (r={rho:.2f})" if not np.isnan(rho) else "eps* vs distance (no variation)"
    norm_label = r"$\ell_\infty$" if norm == "linf" else r"$\ell_2$"
    setup_scatter_axes(
        ax,
        xlabel=rf"$\epsilon^\star$ ({norm_label})",
        ylabel=rf"{metric} $H_{H}$ distance @ {layer}",
        title=title_txt.replace("eps*", r"$\epsilon^\star$"),
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt_close(fig=fig)


def best_signal_layerH(
    diagdist_records: List[DiagramDistanceRecord],
    metric: str = "wasserstein",
) -> Optional[Tuple[str, int]]:
    """
    Return (layer, H) with largest std across conditions for available distances.
    Skips layers where all distances are zero/NaN.
    """
    from collections import defaultdict
    values: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for r in diagdist_records:
        if r.metric != metric or r.condition == "noise_floor":
            continue
        if r.distance is None or (isinstance(r.distance, float) and np.isnan(r.distance)):
            continue
        values[(r.layer, r.H)].append(r.distance)
    if not values:
        return None
    best_key, best_std = None, -1.0
    for key, vals in values.items():
        v = np.array(vals, dtype=float)
        s = float(np.std(v))
        if s > best_std and s > 0.0:
            best_std = s
            best_key = key
    return best_key


def save_distance_curves_with_ci(
    diagdist_records: List[DiagramDistanceRecord],
    metric: str,
    H: int,
    norm: str,
    output_path: str,
    top_k_layers: int = 4,
) -> None:
    """
    Plot mean ± 95% CI distance vs epsilon for top-k layers with largest mean distance.
    Uses conditions adv_{norm}_eps=x.
    """
    from collections import defaultdict
    # collect eps grid
    eps_vals = sorted({
        float(c.split("=")[1])
        for c in {r.condition for r in diagdist_records}
        if c.startswith(f"adv_{norm}_eps=")
    })
    if not eps_vals:
        return
    # aggregate per layer per eps
    by_layer: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in diagdist_records:
        if r.metric == metric and r.H == H and r.condition.startswith(f"adv_{norm}_eps="):
            try:
                eps = float(r.condition.split("=")[1])
            except Exception:
                continue
            if r.distance is None or (isinstance(r.distance, float) and np.isnan(r.distance)):
                continue
            by_layer[r.layer][eps].append(r.distance)
    if not by_layer:
        return
    # rank layers by overall mean
    layer_scores = []
    for layer, eps_map in by_layer.items():
        all_vals = [v for eps in eps_vals for v in eps_map.get(eps, [])]
        if not all_vals:
            continue
        layer_scores.append((layer, float(np.mean(all_vals))))
    layer_scores.sort(key=lambda x: x[1], reverse=True)
    selected = [l for l, _ in layer_scores[:top_k_layers]]
    if not selected:
        return
    # plot
    fig, ax = new_figure(kind="custom", figsize=(7, 4))
    ax = cast(Axes, ax)
    for layer in selected:
        means, cis = [], []
        for eps in eps_vals:
            vals = by_layer[layer].get(eps, [])
            if vals:
                m = float(np.mean(vals))
                se = float(np.std(vals) / np.sqrt(len(vals)))
                ci95 = 1.96 * se
            else:
                m, ci95 = np.nan, np.nan
            means.append(m)
            cis.append(ci95)
        means = np.array(means); cis = np.array(cis)
        ax.plot(eps_vals, means, marker="o", label=layer)
        # CI as shaded area
        if not np.all(np.isnan(cis)):
            ax.fill_between(eps_vals, means - cis, means + cis, alpha=0.15)
    norm_label = r"$\ell_\infty$" if norm == "linf" else r"$\ell_2$"
    setup_line_axes(
        ax,
        xlabel=rf"$\epsilon$ ({norm_label})",
        ylabel=rf"{metric} $H_{H}$",
        title=rf"Distance vs $\epsilon$ (top-{top_k_layers} layers)",
    )
    ax.legend()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt_close(fig=fig)


def save_violin_distance_by_layer(
    diagdist_records: List[DiagramDistanceRecord],
    metric: str,
    H: int,
    norm: str,
    pick: str = "max",  # "max" picks largest eps; or provide specific eps float as str
    output_path: str = "violin_distance_by_layer.png",
) -> None:
    from collections import defaultdict
    # choose condition
    eps_all = sorted({
        float(c.split("=")[1])
        for c in {r.condition for r in diagdist_records}
        if c.startswith(f"adv_{norm}_eps=")
    })
    if not eps_all:
        return
    eps_pick = max(eps_all) if pick == "max" else float(pick)
    cond = f"adv_{norm}_eps={eps_pick}"
    # gather
    by_layer: Dict[str, List[float]] = defaultdict(list)
    for r in diagdist_records:
        if r.metric == metric and r.H == H and r.condition == cond:
            if r.distance is not None and not (isinstance(r.distance, float) and np.isnan(r.distance)):
                by_layer[r.layer].append(r.distance)
    if not by_layer:
        return
    layers = sorted(by_layer.keys())
    data = [by_layer[l] for l in layers]
    fig, ax = new_figure(kind="custom", figsize=(max(6, 1.2 * len(layers)), 4))
    ax = cast(Axes, ax)
    ax.violinplot(data, showmeans=True, showextrema=False)
    ax.set_xticks(range(1, len(layers) + 1))
    ax.set_xticklabels(layers)
    setup_violin_axes(
        ax,
        ylabel=rf"{metric} $H_{H}$ @ $\epsilon$={eps_pick}",
        title="Distance distribution by layer",
        rotate_xticks=True,
        rotation=45.0,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt_close(fig=fig)


# ---------------------------
# Adversarial Example Visualizations
# ---------------------------
def save_adversarial_example(
    x_orig: torch.Tensor,
    x_adv: torch.Tensor,
    path: str,
    title_suffix: str = "",
) -> None:
    """
    Save adversarial example visualization (wrapper around plot_original_vs_adversarial).
    """
    from ..visualization import plot_original_vs_adversarial
    import matplotlib.pyplot as plt
    
    if isinstance(x_orig, torch.Tensor):
        x_orig = x_orig.detach().cpu().numpy()
    if isinstance(x_adv, torch.Tensor):
        x_adv = x_adv.detach().cpu().numpy()
    if x_adv.ndim == 3:
        x_adv = x_adv.squeeze(0)
    if x_orig.ndim == 3:
        x_orig = x_orig.squeeze(0)
    
    # Create figure manually to save instead of show
    from mpl_toolkits.mplot3d import Axes3D
    from ..plot_style import new_figure
    
    all_pts = np.vstack([x_orig, x_adv])
    x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
    z_min, z_max = all_pts[:, 2].min(), all_pts[:, 2].max()
    
    fig, _ = new_figure(kind="custom", figsize=(12, 4))
    
    # Left: overlay
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(x_orig[:, 0], x_orig[:, 1], x_orig[:, 2], s=8, alpha=0.8, label="original")
    ax1.scatter(x_adv[:, 0], x_adv[:, 1], x_adv[:, 2], s=8, alpha=0.8, marker="^", label="adversarial")
    ax1.set_title(f"Overlay {title_suffix}")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.set_xlim(x_min, x_max); ax1.set_ylim(y_min, y_max); ax1.set_zlim(z_min, z_max)
    ax1.legend()
    
    # Right: displacement
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(x_orig[:, 0], x_orig[:, 1], x_orig[:, 2], s=5, alpha=0.5, color="gray")
    ax2.scatter(x_adv[:, 0], x_adv[:, 1], x_adv[:, 2], s=8, alpha=0.9, color="tab:orange")
    for i in range(x_orig.shape[0]):
        xs = [x_orig[i, 0], x_adv[i, 0]]
        ys = [x_orig[i, 1], x_adv[i, 1]]
        zs = [x_orig[i, 2], x_adv[i, 2]]
        ax2.plot(xs, ys, zs, linewidth=0.5, color="tab:orange", alpha=0.7)
    ax2.set_title(f"Displacement vectors {title_suffix}")
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    ax2.set_xlim(x_min, x_max); ax2.set_ylim(y_min, y_max); ax2.set_zlim(z_min, z_max)
    
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt_close(fig=fig)


def save_sample_grid(
    sample_records: List[SampleRecord],
    original_samples: Dict[int, torch.Tensor],
    adversarial_examples: Dict[int, Dict[str, torch.Tensor]],
    output_dir: str,
    n_samples: int = 9,
    selection: str = "diverse",
) -> None:
    """
    Create a grid of clean vs adversarial visualizations for selected samples.
    """
    # Select samples based on strategy
    if selection == "most_vulnerable":
        # Select samples with lowest eps*
        candidates = [(r.sample_id, r.eps_star_linf or float('inf')) for r in sample_records if r.eps_star_linf is not None]
        candidates.sort(key=lambda x: x[1])
        selected_ids = [sid for sid, _ in candidates[:n_samples]]
    elif selection == "random":
        import random
        candidates = [r.sample_id for r in sample_records if r.sample_id in adversarial_examples]
        selected_ids = random.sample(candidates, min(n_samples, len(candidates)))
    else:  # diverse
        # Select diverse samples: mix of classes and eps* ranges
        from collections import defaultdict
        by_class = defaultdict(list)
        for r in sample_records:
            if r.sample_id in adversarial_examples:
                by_class[r.true_label].append((r.sample_id, r.eps_star_linf or 0.0))
        selected_ids = []
        for class_id, samples in by_class.items():
            samples.sort(key=lambda x: x[1])
            # Take from different parts of distribution
            n_per_class = max(1, n_samples // len(by_class))
            indices = np.linspace(0, len(samples) - 1, n_per_class, dtype=int)
            selected_ids.extend([samples[i][0] for i in indices])
        selected_ids = selected_ids[:n_samples]
    
    if not selected_ids:
        return
    
    # Create grid
    grid_size = int(np.ceil(np.sqrt(n_samples)))
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(grid_size * 4, grid_size * 2))
    axes = []
    for i in range(grid_size * grid_size * 2):
        axes.append(fig.add_subplot(grid_size, grid_size * 2, i + 1, projection='3d'))
    
    for idx, sample_id in enumerate(selected_ids[:n_samples]):
        if sample_id not in original_samples or sample_id not in adversarial_examples:
            continue
        x_orig = original_samples[sample_id].numpy()
        if "linf" in adversarial_examples[sample_id]:
            x_adv = adversarial_examples[sample_id]["linf"].numpy()
        elif "l2" in adversarial_examples[sample_id]:
            x_adv = adversarial_examples[sample_id]["l2"].numpy()
        else:
            continue
        
        if x_adv.ndim == 3:
            x_adv = x_adv.squeeze(0)
        if x_orig.ndim == 3:
            x_orig = x_orig.squeeze(0)
        
        # Clean
        ax_clean = axes[idx * 2]
        ax_clean.scatter(x_orig[:, 0], x_orig[:, 1], x_orig[:, 2], s=5, alpha=0.8)
        ax_clean.set_title(f"Sample {sample_id} (clean)")
        ax_clean.axis('off')
        
        # Adversarial
        ax_adv = axes[idx * 2 + 1]
        ax_adv.scatter(x_adv[:, 0], x_adv[:, 1], x_adv[:, 2], s=5, alpha=0.8, color="tab:orange")
        ax_adv.set_title(f"Sample {sample_id} (adv)")
        ax_adv.axis('off')
    
    # Hide unused subplots
    for idx in range(len(selected_ids) * 2, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "sample_grid.png"), dpi=150, bbox_inches="tight")
    plt_close(fig=fig)


# ---------------------------
# Per-Class Visualizations
# ---------------------------
def save_eps_star_by_class_boxplot(
    sample_records: List[SampleRecord],
    norm: str,
    path: str,
) -> None:
    """Box plot of eps* distribution per class."""
    from collections import defaultdict
    class_names = {0: "Circle", 1: "Sphere", 2: "Torus"}
    
    by_class: Dict[int, List[float]] = defaultdict(list)
    for r in sample_records:
        eps_star = r.eps_star_linf if norm == "linf" else r.eps_star_l2
        if eps_star is not None and not np.isnan(eps_star):
            by_class[r.true_label].append(float(eps_star))
    
    if not by_class:
        return
    
    classes = sorted(by_class.keys())
    data = [by_class[c] for c in classes]
    labels = [class_names.get(c, f"Class {c}") for c in classes]
    
    fig, ax = new_figure(kind="single")
    ax = cast(Axes, ax)
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    norm_label = r"$\ell_\infty$" if norm == "linf" else r"$\ell_2$"
    setup_axes(ax, xlabel="Class", ylabel=rf"$\epsilon^\star$ ({norm_label})", 
               title=rf"$\epsilon^\star$ distribution by class ({norm_label})")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt_close(fig=fig)


def save_class_confusion_matrix(
    sample_records: List[SampleRecord],
    norm: str,
    path: str,
) -> None:
    """Confusion matrix for adversarial predictions."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    class_names = {0: "Circle", 1: "Sphere", 2: "Torus"}
    true_labels = []
    pred_labels = []
    
    for r in sample_records:
        adv_pred = r.adv_pred_linf if norm == "linf" else r.adv_pred_l2
        if adv_pred is not None:
            true_labels.append(r.true_label)
            pred_labels.append(adv_pred)
    
    if not true_labels:
        return
    
    cm = confusion_matrix(true_labels, pred_labels)
    class_labels = [class_names.get(i, f"Class {i}") for i in sorted(set(true_labels + pred_labels))]
    
    fig, ax = new_figure(kind="custom", figsize=(6, 5))
    ax = cast(Axes, ax)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, 
                yticklabels=class_labels, ax=ax)
    norm_label = r"$\ell_\infty$" if norm == "linf" else r"$\ell_2$"
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (Adversarial, {norm_label})")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt_close(fig=fig)


def save_robust_accuracy_by_class(
    ra_curves_by_class: Dict[int, List[float]],
    eps_values: List[float],
    norm: str,
    path: str,
) -> None:
    """Robust accuracy curves per class."""
    class_names = {0: "Circle", 1: "Sphere", 2: "Torus"}
    
    if not ra_curves_by_class:
        return
    
    fig, ax = new_figure(kind="single")
    ax = cast(Axes, ax)
    
    for class_id, accuracies in sorted(ra_curves_by_class.items()):
        label = class_names.get(class_id, f"Class {class_id}")
        ax.plot(eps_values, accuracies, marker="o", label=label)
    
    norm_label = r"$\ell_\infty$" if norm == "linf" else r"$\ell_2$"
    setup_line_axes(ax, xlabel=rf"$\epsilon$ ({norm_label})", 
                    ylabel="Robust accuracy",
                    title=rf"Robust accuracy by class ({norm_label})")
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt_close(fig=fig)


def save_topology_distance_by_class(
    diagdist_records: List[DiagramDistanceRecord],
    sample_records: List[SampleRecord],
    metric: str,
    H: int,
    layer: str,
    condition: str,
    path: str,
) -> None:
    """Box plot of topology distances grouped by class."""
    from collections import defaultdict
    
    # Create sample_id -> class mapping
    sample_to_class = {r.sample_id: r.true_label for r in sample_records}
    class_names = {0: "Circle", 1: "Sphere", 2: "Torus"}
    
    by_class: Dict[int, List[float]] = defaultdict(list)
    for r in diagdist_records:
        if r.metric == metric and r.H == H and r.layer == layer and r.condition == condition:
            if r.distance is not None and not np.isnan(r.distance):
                class_id = sample_to_class.get(r.sample_id)
                if class_id is not None:
                    by_class[class_id].append(float(r.distance))
    
    if not by_class:
        return
    
    classes = sorted(by_class.keys())
    data = [by_class[c] for c in classes]
    labels = [class_names.get(c, f"Class {c}") for c in classes]
    
    fig, ax = new_figure(kind="single")
    ax = cast(Axes, ax)
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightgreen')
        patch.set_alpha(0.7)
    
    setup_axes(ax, xlabel="Class", ylabel=f"{metric} $H_{H}$ distance",
               title=f"{metric} $H_{H}$ by class ({layer}, {condition})")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt_close(fig=fig)


# ---------------------------
# Statistical Comparison Plots
# ---------------------------
def save_correlation_heatmap(
    sample_records: List[SampleRecord],
    diagdist_records: List[DiagramDistanceRecord],
    path: str,
) -> None:
    """Correlation matrix of all numeric metrics."""
    import pandas as pd
    import seaborn as sns
    
    # Collect metrics from sample_records
    data_dict = {}
    for r in sample_records:
        data_dict[r.sample_id] = {
            'eps_star_linf': r.eps_star_linf,
            'eps_star_l2': r.eps_star_l2,
            'clean_margin': r.clean_margin,
        }
    
    # Aggregate topology distances per sample (mean across layers/conditions)
    from collections import defaultdict
    topo_by_sample: Dict[int, List[float]] = defaultdict(list)
    for r in diagdist_records:
        if r.distance is not None and not np.isnan(r.distance):
            topo_by_sample[r.sample_id].append(float(r.distance))
    
    for sample_id, distances in topo_by_sample.items():
        if sample_id in data_dict:
            data_dict[sample_id]['mean_topology_distance'] = float(np.mean(distances))
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(data_dict, orient='index')
    df = df.dropna()  # Remove rows with any NaN
    
    if len(df) < 3:
        return
    
    corr = df.corr()
    
    fig, ax = new_figure(kind="custom", figsize=(7, 6))
    ax = cast(Axes, ax)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                square=True, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Metric Correlation Matrix")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt_close(fig=fig)


def save_percentile_curves(
    diagdist_records: List[DiagramDistanceRecord],
    metric: str,
    H: int,
    norm: str,
    layer: str,
    path: str,
) -> None:
    """Show 10th, 50th, 90th percentiles of distance vs epsilon."""
    from collections import defaultdict
    
    eps_vals = sorted({
        float(c.split("=")[1])
        for c in {r.condition for r in diagdist_records}
        if c.startswith(f"adv_{norm}_eps=")
    })
    if not eps_vals:
        return
    
    by_eps: Dict[float, List[float]] = defaultdict(list)
    for r in diagdist_records:
        if r.metric == metric and r.H == H and r.layer == layer and r.condition.startswith(f"adv_{norm}_eps="):
            try:
                eps = float(r.condition.split("=")[1])
            except Exception:
                continue
            if r.distance is not None and not np.isnan(r.distance):
                by_eps[eps].append(float(r.distance))
    
    if not by_eps:
        return
    
    percentiles = [10, 50, 90]
    p_data = {p: [] for p in percentiles}
    
    for eps in eps_vals:
        vals = by_eps.get(eps, [])
        if vals:
            for p in percentiles:
                p_data[p].append(float(np.percentile(vals, p)))
        else:
            for p in percentiles:
                p_data[p].append(np.nan)
    
    fig, ax = new_figure(kind="single")
    ax = cast(Axes, ax)
    for p in percentiles:
        ax.plot(eps_vals, p_data[p], marker="o", label=f"{p}th percentile")
    
    norm_label = r"$\ell_\infty$" if norm == "linf" else r"$\ell_2$"
    setup_line_axes(ax, xlabel=rf"$\epsilon$ ({norm_label})", 
                    ylabel=rf"{metric} $H_{H}$ distance",
                    title=rf"Distance percentiles vs $\epsilon$ ({layer})")
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt_close(fig=fig)


def save_outlier_analysis(
    sample_records: List[SampleRecord],
    norm: str,
    path: str,
) -> None:
    """Scatter plot highlighting outliers in eps*."""
    eps_stars = []
    margins = []
    is_outlier = []
    
    for r in sample_records:
        eps_star = r.eps_star_linf if norm == "linf" else r.eps_star_l2
        if eps_star is not None and not np.isnan(eps_star) and r.clean_margin is not None:
            eps_stars.append(float(eps_star))
            margins.append(float(r.clean_margin))
            # Outlier: eps* more than 2 std devs from mean
            is_outlier.append(False)  # Will compute after collecting all
    
    if len(eps_stars) < 3:
        return
    
    eps_arr = np.array(eps_stars)
    mean_eps = np.mean(eps_arr)
    std_eps = np.std(eps_arr)
    outlier_threshold = mean_eps + 2 * std_eps
    
    is_outlier = [eps > outlier_threshold for eps in eps_stars]
    
    fig, ax = new_figure(kind="single")
    ax = cast(Axes, ax)
    
    # Plot normal and outliers separately
    normal_eps = [eps_stars[i] for i in range(len(eps_stars)) if not is_outlier[i]]
    normal_margins = [margins[i] for i in range(len(margins)) if not is_outlier[i]]
    outlier_eps = [eps_stars[i] for i in range(len(eps_stars)) if is_outlier[i]]
    outlier_margins = [margins[i] for i in range(len(margins)) if is_outlier[i]]
    
    if normal_eps:
        ax.scatter(normal_eps, normal_margins, s=14, alpha=0.6, label="Normal", color="blue")
    if outlier_eps:
        ax.scatter(outlier_eps, outlier_margins, s=20, alpha=0.8, label="Outlier", color="red", marker="x")
    
    norm_label = r"$\ell_\infty$" if norm == "linf" else r"$\ell_2$"
    setup_scatter_axes(ax, xlabel=rf"$\epsilon^\star$ ({norm_label})", 
                       ylabel="Clean margin",
                       title=f"Outlier Analysis ({norm_label})")
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt_close(fig=fig)


# ---------------------------
# Intuitive Comparison Visualizations
# ---------------------------
def save_norm_comparison_bars(
    sample_records: List[SampleRecord],
    path: str,
) -> None:
    """Compare L∞ vs L2 eps* side-by-side for easy comparison."""
    class_names = {0: "Circle", 1: "Sphere", 2: "Torus"}
    
    # Aggregate by class
    from collections import defaultdict
    by_class: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: {"linf": [], "l2": []})
    
    for r in sample_records:
        if r.eps_star_linf is not None and not np.isnan(r.eps_star_linf):
            by_class[r.true_label]["linf"].append(float(r.eps_star_linf))
        if r.eps_star_l2 is not None and not np.isnan(r.eps_star_l2):
            by_class[r.true_label]["l2"].append(float(r.eps_star_l2))
    
    if not by_class:
        return
    
    classes = sorted(by_class.keys())
    class_labels = [class_names.get(c, f"Class {c}") for c in classes]
    
    linf_means = [np.mean(by_class[c]["linf"]) if by_class[c]["linf"] else 0.0 for c in classes]
    l2_means = [np.mean(by_class[c]["l2"]) if by_class[c]["l2"] else 0.0 for c in classes]
    
    x = np.arange(len(classes))
    width = 0.35
    
    fig, ax = new_figure(kind="custom", figsize=(8, 5))
    ax = cast(Axes, ax)
    
    bars1 = ax.bar(x - width/2, linf_means, width, label=r"$\ell_\infty$", alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, l2_means, width, label=r"$\ell_2$", alpha=0.8, color='coral')
    
    ax.set_xlabel("Class")
    ax.set_ylabel(r"Mean $\epsilon^\star$")
    ax.set_title("Robustness Comparison: L∞ vs L2")
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8)
    
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt_close(fig=fig)


def save_robustness_summary_cards(
    sample_records: List[SampleRecord],
    ra_curves: Dict[str, List[float]],
    eps_grid: List[float],
    path: str,
) -> None:
    """Create visual summary cards showing key robustness metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    # Card 1: Mean eps* by norm
    ax = axes[0]
    norms = []
    means = []
    for norm in ["linf", "l2"]:
        eps_stars = [r.eps_star_linf if norm == "linf" else r.eps_star_l2 
                    for r in sample_records]
        eps_stars = [e for e in eps_stars if e is not None and not np.isnan(e)]
        if eps_stars:
            norms.append(r"$\ell_\infty$" if norm == "linf" else r"$\ell_2$")
            means.append(np.mean(eps_stars))
    
    if norms:
        bars = ax.bar(norms, means, color=['steelblue', 'coral'], alpha=0.7)
        ax.set_ylabel(r"Mean $\epsilon^\star$")
        ax.set_title("Average Robustness")
        ax.grid(True, axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
    
    # Card 2: Robust accuracy at different epsilons
    ax = axes[1]
    if ra_curves:
        for norm, accs in ra_curves.items():
            label = r"$\ell_\infty$" if norm == "linf" else r"$\ell_2$"
            ax.plot(eps_grid, accs, marker='o', label=label, linewidth=2)
        ax.set_xlabel(r"$\epsilon$")
        ax.set_ylabel("Robust Accuracy")
        ax.set_title("Robust Accuracy Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Card 3: Success rate (samples that can be attacked)
    ax = axes[2]
    success_rates = []
    norm_labels = []
    for norm in ["linf", "l2"]:
        eps_stars = [r.eps_star_linf if norm == "linf" else r.eps_star_l2 
                    for r in sample_records]
        success = sum(1 for e in eps_stars if e is not None and not np.isnan(e))
        total = len(sample_records)
        if total > 0:
            success_rates.append(100 * success / total)
            norm_labels.append(r"$\ell_\infty$" if norm == "linf" else r"$\ell_2$")
    
    if norm_labels:
        bars = ax.bar(norm_labels, success_rates, color=['steelblue', 'coral'], alpha=0.7)
        ax.set_ylabel("Attack Success Rate (%)")
        ax.set_title("Vulnerability Rate")
        ax.set_ylim([0, 100])
        ax.grid(True, axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom')
    
    # Card 4: Distribution of eps* (overlapping histograms)
    ax = axes[3]
    linf_eps = []
    l2_eps = []
    for r in sample_records:
        if r.eps_star_linf is not None and not np.isnan(r.eps_star_linf):
            linf_eps.append(float(r.eps_star_linf))
        if r.eps_star_l2 is not None and not np.isnan(r.eps_star_l2):
            l2_eps.append(float(r.eps_star_l2))
    
    if linf_eps or l2_eps:
        # Determine bins that work for both distributions
        all_eps_combined = linf_eps + l2_eps
        if all_eps_combined:
            bins = np.linspace(min(all_eps_combined), max(all_eps_combined), 30)
            
            # Plot overlapping histograms with transparency
            if linf_eps:
                ax.hist(linf_eps, bins=bins, alpha=0.6, color='steelblue', 
                       label=r"$\ell_\infty$", density=True, edgecolor='darkblue', linewidth=0.5)
            if l2_eps:
                ax.hist(l2_eps, bins=bins, alpha=0.6, color='coral', 
                       label=r"$\ell_2$", density=True, edgecolor='darkred', linewidth=0.5)
            
            ax.set_ylabel("Density")
            ax.set_xlabel(r"$\epsilon^\star$")
            ax.set_title("Robustness Distribution")
            ax.legend()
            ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt_close(fig=fig)


def save_class_robustness_comparison(
    sample_records: List[SampleRecord],
    path: str,
) -> None:
    """Visual comparison of robustness across classes - easy to read."""
    class_names = {0: "Circle", 1: "Sphere", 2: "Torus"}
    
    from collections import defaultdict
    metrics = defaultdict(lambda: defaultdict(list))
    
    for r in sample_records:
        if r.eps_star_linf is not None and not np.isnan(r.eps_star_linf):
            metrics[r.true_label]["linf"].append(float(r.eps_star_linf))
        if r.eps_star_l2 is not None and not np.isnan(r.eps_star_l2):
            metrics[r.true_label]["l2"].append(float(r.eps_star_l2))
        if r.clean_margin is not None:
            metrics[r.true_label]["margin"].append(float(r.clean_margin))
    
    if not metrics:
        return
    
    classes = sorted(metrics.keys())
    class_labels = [class_names.get(c, f"Class {c}") for c in classes]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: eps* comparison
    x = np.arange(len(classes))
    width = 0.35
    
    linf_means = [np.mean(metrics[c]["linf"]) if metrics[c]["linf"] else 0 for c in classes]
    l2_means = [np.mean(metrics[c]["l2"]) if metrics[c]["l2"] else 0 for c in classes]
    
    bars1 = ax1.bar(x - width/2, linf_means, width, label=r"$\ell_\infty$", 
                    alpha=0.8, color='steelblue')
    bars2 = ax1.bar(x + width/2, l2_means, width, label=r"$\ell_2$", 
                    alpha=0.8, color='coral')
    
    ax1.set_xlabel("Class")
    ax1.set_ylabel(r"Mean $\epsilon^\star$")
    ax1.set_title("Robustness by Class")
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_labels)
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Right: Clean margin comparison
    margins = [np.mean(metrics[c]["margin"]) if metrics[c]["margin"] else 0 for c in classes]
    bars3 = ax2.bar(class_labels, margins, alpha=0.8, color='mediumseagreen')
    ax2.set_xlabel("Class")
    ax2.set_ylabel("Mean Clean Margin")
    ax2.set_title("Model Confidence by Class")
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax = bar.axes
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt_close(fig=fig)


def save_layer_sensitivity_comparison(
    diagdist_records: List[DiagramDistanceRecord],
    metric: str,
    H: int,
    norm: str,
    path: str,
) -> None:
    """Compare topology sensitivity across layers - intuitive visualization."""
    from collections import defaultdict
    
    # Get max epsilon condition
    eps_vals = sorted({
        float(c.split("=")[1])
        for c in {r.condition for r in diagdist_records}
        if c.startswith(f"adv_{norm}_eps=")
    })
    if not eps_vals:
        return
    
    max_eps = max(eps_vals)
    condition = f"adv_{norm}_eps={max_eps}"
    
    by_layer: Dict[str, List[float]] = defaultdict(list)
    for r in diagdist_records:
        if r.metric == metric and r.H == H and r.condition == condition:
            if r.distance is not None and not np.isnan(r.distance):
                by_layer[r.layer].append(float(r.distance))
    
    if not by_layer:
        return
    
    layers = sorted(by_layer.keys())
    means = [np.mean(by_layer[l]) for l in layers]
    stds = [np.std(by_layer[l]) for l in layers]
    
    fig, ax = new_figure(kind="custom", figsize=(10, 5))
    ax = cast(Axes, ax)
    
    x = np.arange(len(layers))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, 
                  color=plt.cm.viridis(np.linspace(0, 1, len(layers))))
    
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"{metric} $H_{H}$ Distance")
    norm_label = r"$\ell_\infty$" if norm == "linf" else r"$\ell_2$"
    ax.set_title(f"Topology Sensitivity by Layer ({norm_label}, $\\epsilon$={max_eps})")
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=45, ha='right')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
    
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt_close(fig=fig)


def save_robustness_radar(
    sample_records: List[SampleRecord],
    diagdist_records: List[DiagramDistanceRecord],
    path: str,
) -> None:
    """Radar chart showing multi-dimensional robustness profile."""
    from ..plot_style import new_radar_figure
    
    # Calculate metrics
    metrics = {}
    
    # Mean eps* for each norm
    for norm in ["linf", "l2"]:
        eps_stars = [r.eps_star_linf if norm == "linf" else r.eps_star_l2 
                    for r in sample_records]
        eps_stars = [e for e in eps_stars if e is not None and not np.isnan(e)]
        if eps_stars:
            metrics[f"eps* ({norm})"] = np.mean(eps_stars)
    
    # Mean topology distance
    topo_dists = [r.distance for r in diagdist_records 
                 if r.distance is not None and not np.isnan(r.distance)]
    if topo_dists:
        metrics["Topology Distance"] = np.mean(topo_dists)
    
    # Mean clean margin
    margins = [r.clean_margin for r in sample_records 
              if r.clean_margin is not None]
    if margins:
        metrics["Clean Margin"] = np.mean(margins)
    
    if len(metrics) < 3:
        return
    
    labels = list(metrics.keys())
    values = list(metrics.values())
    
    # Normalize values to 0-1 scale for radar chart
    max_vals = {k: max(v, 1.0) for k, v in metrics.items()}
    normalized = [v / max_vals[k] for k, v in zip(labels, values)]
    
    from ..plot_style import setup_radar_axes
    
    fig, ax, angles = new_radar_figure(labels, kind="custom", figsize=(8, 8))
    
    # angles is already returned from new_radar_figure and includes closing angle
    angles_list = angles.tolist()
    normalized += normalized[:1]
    
    ax.plot(angles_list, normalized, 'o-', linewidth=2, label="Model Profile")
    ax.fill(angles_list, normalized, alpha=0.25)
    ax.set_ylim(0, 1)
    setup_radar_axes(ax, labels, title="Robustness Profile")
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt_close(fig=fig)


def save_epsilon_impact_comparison(
    diagdist_records: List[DiagramDistanceRecord],
    metric: str,
    H: int,
    norm: str,
    path: str,
) -> None:
    """Show how topology distance changes with epsilon - intuitive line chart."""
    from collections import defaultdict
    
    eps_vals = sorted({
        float(c.split("=")[1])
        for c in {r.condition for r in diagdist_records}
        if c.startswith(f"adv_{norm}_eps=")
    })
    if not eps_vals:
        return
    
    # Aggregate across all layers
    by_eps: Dict[float, List[float]] = defaultdict(list)
    for r in diagdist_records:
        if r.metric == metric and r.H == H and r.condition.startswith(f"adv_{norm}_eps="):
            try:
                eps = float(r.condition.split("=")[1])
            except Exception:
                continue
            if r.distance is not None and not np.isnan(r.distance):
                by_eps[eps].append(float(r.distance))
    
    if not by_eps:
        return
    
    means = [np.mean(by_eps[eps]) for eps in eps_vals]
    stds = [np.std(by_eps[eps]) for eps in eps_vals]
    
    fig, ax = new_figure(kind="single")
    ax = cast(Axes, ax)
    
    ax.errorbar(eps_vals, means, yerr=stds, marker='o', linewidth=2, 
                markersize=8, capsize=5, capthick=2, alpha=0.8, color='steelblue')
    ax.fill_between(eps_vals, 
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.2, color='steelblue')
    
    norm_label = r"$\ell_\infty$" if norm == "linf" else r"$\ell_2$"
    ax.set_xlabel(rf"$\epsilon$ ({norm_label})")
    ax.set_ylabel(f"{metric} $H_{H}$ Distance")
    ax.set_title(f"Topology Sensitivity vs Attack Strength ({norm_label})")
    ax.grid(True, alpha=0.3)
    
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt_close(fig=fig)


# ---------------------------
# Internal utility
# ---------------------------
def plt_close(fig=None):
    """Utility: close a figure without relying on global plt state."""
    try:
        import matplotlib.pyplot as _plt
        if fig is not None:
            _plt.close(fig)
        else:
            _plt.close()
    except Exception:
        pass


