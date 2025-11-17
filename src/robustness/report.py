from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple, cast
import csv
import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
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
    # PGD (backward compatibility)
    eps_star_linf: Optional[float] = None
    adv_pred_linf: Optional[int] = None
    eps_star_l2: Optional[float] = None
    adv_pred_l2: Optional[int] = None
    # Other attacks (dynamic fields stored as dict in metadata)
    attack_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # {attack_type: {norm: {eps_star, adv_pred, ...}}}
    rot_deg_star: Optional[float] = None
    trans_x_star: Optional[float] = None
    trans_y_star: Optional[float] = None
    trans_z_star: Optional[float] = None
    jitter_std_star: Optional[float] = None
    dropout_star: Optional[float] = None
    alpha_star: Optional[float] = None
    chamfer_clean_adv: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Flatten attack_results for CSV export
        for attack_type, norms in self.attack_results.items():
            for norm, results in norms.items():
                d[f"eps_star_{attack_type}_{norm}"] = results.get("eps_star")
                d[f"adv_pred_{attack_type}_{norm}"] = results.get("adv_pred")
        return d
    
    def set_attack_result(self, attack_type: str, norm: str, eps_star: Optional[float], adv_pred: Optional[int], metadata: Optional[Dict[str, Any]] = None):
        """Set attack result for a specific attack type and norm."""
        if attack_type not in self.attack_results:
            self.attack_results[attack_type] = {}
        self.attack_results[attack_type][norm] = {
            "eps_star": eps_star,
            "adv_pred": adv_pred,
            "metadata": metadata or {},
        }


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
# Layer Transformation Visualizations
# ---------------------------
def save_layer_transformation_grid(
    model,
    x_sample: torch.Tensor,
    sample_id: int,
    true_label: int,
    output_path: str,
    layers: Optional[List[str]] = None,
    title_suffix: str = "",
    reduction_method: str = "pca",
    normalize: str = "zscore",
    pca_dim: Optional[int] = None,
) -> None:
    """
    Visualize how a point cloud transforms through each layer of the network.
    For high-dimensional layers, uses dimensionality reduction to 3D for visualization.
    
    Args:
        model: The neural network model
        x_sample: Input point cloud tensor (1, N, 3) or (N, 3)
        sample_id: ID of the sample for labeling
        true_label: True class label
        output_path: Path to save the visualization
        layers: List of layer names to visualize (if None, uses all available)
        title_suffix: Additional text for the title
        reduction_method: "pca" | "tsne" | "umap" - method for dimensionality reduction
        normalize: "none" | "zscore" | "l2" - normalization before reduction
        pca_dim: If provided, first reduce to this dimension via PCA, then to 3D
    """
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.decomposition import PCA
    
    class_names = {0: "Circle", 1: "Sphere", 2: "Torus"}
    class_name = class_names.get(true_label, f"Class {true_label}")
    
    # Ensure x_sample is in correct format (1, N, 3)
    if x_sample.dim() == 2:
        x_sample = x_sample.unsqueeze(0)
    if x_sample.dim() == 3 and x_sample.size(0) > 1:
        x_sample = x_sample[0:1]
    
    device = next(model.parameters()).device
    model.eval()
    
    # Forward pass with layer saving
    with torch.no_grad():
        _ = model(x_sample.to(device), save_layers=True)
    
    # Get available layers
    available_layers = list(model.layer_outputs.keys())
    if layers is None:
        layers = available_layers
    else:
        # Filter to only layers that exist
        layers = [l for l in layers if l in available_layers]
    
    if not layers:
        return
    
    # Create grid layout
    n_layers = len(layers)
    ncols = min(4, n_layers)
    nrows = int(np.ceil(n_layers / ncols))
    
    fig = plt.figure(figsize=(ncols * 3.5, nrows * 3.5))
    
    for idx, layer_name in enumerate(layers):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
        
        layer_act = model.layer_outputs[layer_name]
        
        # Handle different tensor shapes
        if layer_act.dim() == 3:
            # (B, N, D) - per-point features
            act = layer_act[0].detach().cpu().numpy()  # (N, D)
        elif layer_act.dim() == 2:
            # (B, D) - pooled/global features
            # For pooled layers, we can't visualize per-point, so skip or show a placeholder
            if layer_act.size(0) == 1:
                act = layer_act[0].detach().cpu().numpy()  # (D,)
                # For pooled layers, create a simple visualization
                ax.text(0.5, 0.5, 0.5, f"Pooled\n({act.shape[0]}D)", 
                       ha='center', va='center', fontsize=10)
                ax.set_title(f"{layer_name}\n(Global features)", fontsize=9)
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_zlim(-1, 1)
                continue
            else:
                act = layer_act.detach().cpu().numpy()  # (B, D)
        else:
            continue
        
        # Get feature dimension
        if act.ndim == 1:
            # Single vector - can't visualize as point cloud
            ax.text(0.5, 0.5, 0.5, f"Vector\n({act.shape[0]}D)", 
                   ha='center', va='center', fontsize=10)
            ax.set_title(f"{layer_name}\n(Global features)", fontsize=9)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            continue
        
        feature_dim = act.shape[1]
        
        # If already 3D, use directly
        if feature_dim == 3:
            points_3d = act
            variance_explained = None
        elif feature_dim < 3:
            # Pad with zeros
            points_3d = np.zeros((act.shape[0], 3))
            points_3d[:, :feature_dim] = act
            variance_explained = None
        else:
            # Normalize before reduction (consistent with topology computation)
            act_normalized = _normalize_activations(act, normalize)
            
            # Optional: first reduce to intermediate dimension via PCA
            if pca_dim is not None and pca_dim < feature_dim and pca_dim > 3:
                pca_intermediate = PCA(n_components=pca_dim)
                act_normalized = pca_intermediate.fit_transform(act_normalized)
            
            # Reduce to 3D using specified method
            points_3d, variance_explained = _reduce_to_3d(
                act_normalized, method=reduction_method, original_dim=feature_dim
            )
        
        # Plot the 3D point cloud
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                  s=20, alpha=0.7, c=points_3d[:, 2], cmap='viridis')
        
        # Set equal aspect ratio
        max_range = np.array([points_3d[:, 0].max() - points_3d[:, 0].min(),
                             points_3d[:, 1].max() - points_3d[:, 1].min(),
                             points_3d[:, 2].max() - points_3d[:, 2].min()]).max() / 2.0
        mid_x = (points_3d[:, 0].max() + points_3d[:, 0].min()) * 0.5
        mid_y = (points_3d[:, 1].max() + points_3d[:, 1].min()) * 0.5
        mid_z = (points_3d[:, 2].max() + points_3d[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Title with dimension info and variance explained
        if variance_explained is not None:
            dim_info = f"{feature_dim}D→3D ({variance_explained:.1f}% var)"
        else:
            dim_info = f"{feature_dim}D→3D" if feature_dim > 3 else f"{feature_dim}D"
        ax.set_title(f"{layer_name}\n({dim_info})", fontsize=9)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
    
    # Overall title
    fig.suptitle(f"Layer Transformations: Sample {sample_id} ({class_name}){title_suffix}", 
                 fontsize=12, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt_close(fig=fig)


def save_layer_transformation_comparison(
    model,
    x_clean: torch.Tensor,
    x_adv: torch.Tensor,
    sample_id: int,
    true_label: int,
    output_path: str,
    layers: Optional[List[str]] = None,
    norm: str = "linf",
    reduction_method: str = "pca",
    normalize: str = "zscore",
    pca_dim: Optional[int] = None,
) -> None:
    """
    Visualize how clean and adversarial point clouds transform through layers side-by-side.
    
    Args:
        model: The neural network model
        x_clean: Clean input point cloud (1, N, 3) or (N, 3)
        x_adv: Adversarial input point cloud (1, N, 3) or (N, 3)
        sample_id: ID of the sample
        true_label: True class label
        output_path: Path to save the visualization
        layers: List of layer names to visualize
        norm: Norm used for adversarial example (for labeling)
    """
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.decomposition import PCA
    
    class_names = {0: "Circle", 1: "Sphere", 2: "Torus"}
    class_name = class_names.get(true_label, f"Class {true_label}")
    
    # Ensure correct format
    if x_clean.dim() == 2:
        x_clean = x_clean.unsqueeze(0)
    if x_adv.dim() == 2:
        x_adv = x_adv.unsqueeze(0)
    
    device = next(model.parameters()).device
    model.eval()
    
    # Forward pass for clean
    with torch.no_grad():
        _ = model(x_clean.to(device), save_layers=True)
    clean_outputs = {k: v.clone() for k, v in model.layer_outputs.items()}
    
    # Forward pass for adversarial
    with torch.no_grad():
        _ = model(x_adv.to(device), save_layers=True)
    adv_outputs = {k: v.clone() for k, v in model.layer_outputs.items()}
    
    # Get available layers
    available_layers = list(clean_outputs.keys())
    if layers is None:
        layers = available_layers
    else:
        layers = [l for l in layers if l in available_layers]
    
    if not layers:
        return
    
    # Create grid: 2 columns (clean, adv) x n_layers rows
    n_layers = len(layers)
    fig = plt.figure(figsize=(7, n_layers * 3))
    
    for idx, layer_name in enumerate(layers):
        # Clean
        ax_clean = fig.add_subplot(n_layers, 2, idx * 2 + 1, projection='3d')
        _plot_layer_activation(
            ax_clean, clean_outputs[layer_name], layer_name, "Clean",
            reduction_method=reduction_method, normalize=normalize, pca_dim=pca_dim
        )
        
        # Adversarial
        ax_adv = fig.add_subplot(n_layers, 2, idx * 2 + 2, projection='3d')
        _plot_layer_activation(
            ax_adv, adv_outputs[layer_name], layer_name, "Adversarial",
            reduction_method=reduction_method, normalize=normalize, pca_dim=pca_dim
        )
    
    norm_label = r"$\ell_\infty$" if norm == "linf" else r"$\ell_2$"
    fig.suptitle(f"Layer Transformations: Clean vs Adversarial\nSample {sample_id} ({class_name}, {norm_label})", 
                 fontsize=12, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt_close(fig=fig)


def _normalize_activations(act: np.ndarray, normalize: str) -> np.ndarray:
    """Normalize activations using specified method."""
    if normalize == "zscore":
        mu = act.mean(axis=0, keepdims=True)
        sigma = act.std(axis=0, keepdims=True) + 1e-12
        return (act - mu) / sigma
    elif normalize == "l2":
        norms = np.linalg.norm(act, ord=2, axis=1, keepdims=True) + 1e-12
        return act / norms
    else:  # "none"
        return act


def _reduce_to_3d(act: np.ndarray, method: str = "pca", original_dim: Optional[int] = None) -> Tuple[np.ndarray, Optional[float]]:
    """
    Reduce activations to 3D using specified method.
    Returns (points_3d, variance_explained) where variance_explained is None for non-PCA methods.
    """
    from sklearn.decomposition import PCA
    
    if method == "pca":
        pca = PCA(n_components=3)
        points_3d = pca.fit_transform(act)
        variance_explained = sum(pca.explained_variance_ratio_) * 100
        # Normalize to similar scale as input
        if points_3d.std() > 0:
            points_3d = points_3d / (points_3d.std() + 1e-8) * 0.5
        return points_3d, variance_explained
    elif method == "tsne":
        try:
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(act) - 1))
            points_3d = tsne.fit_transform(act)
            return points_3d, None
        except ImportError:
            # Fallback to PCA if t-SNE not available
            pca = PCA(n_components=3)
            points_3d = pca.fit_transform(act)
            variance_explained = sum(pca.explained_variance_ratio_) * 100
            return points_3d, variance_explained
    elif method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=3, random_state=42)
            points_3d = reducer.fit_transform(act)
            return points_3d, None
        except ImportError:
            # Fallback to PCA if UMAP not available
            pca = PCA(n_components=3)
            points_3d = pca.fit_transform(act)
            variance_explained = sum(pca.explained_variance_ratio_) * 100
            return points_3d, variance_explained
    else:
        # Default to PCA
        pca = PCA(n_components=3)
        points_3d = pca.fit_transform(act)
        variance_explained = sum(pca.explained_variance_ratio_) * 100
        if points_3d.std() > 0:
            points_3d = points_3d / (points_3d.std() + 1e-8) * 0.5
        return points_3d, variance_explained


def _plot_layer_activation(
    ax, 
    layer_act: torch.Tensor, 
    layer_name: str, 
    condition: str,
    reduction_method: str = "pca",
    normalize: str = "zscore",
    pca_dim: Optional[int] = None,
):
    """Helper function to plot a single layer activation in 3D."""
    from sklearn.decomposition import PCA
    
    # Handle different tensor shapes
    if layer_act.dim() == 3:
        act = layer_act[0].detach().cpu().numpy()  # (N, D)
    elif layer_act.dim() == 2:
        if layer_act.size(0) == 1:
            act = layer_act[0].detach().cpu().numpy()  # (D,)
        else:
            act = layer_act.detach().cpu().numpy()  # (B, D)
    else:
        return
    
    # Handle pooled/global features
    if act.ndim == 1:
        ax.text(0.5, 0.5, 0.5, f"Pooled\n({act.shape[0]}D)", 
               ha='center', va='center', fontsize=9)
        ax.set_title(f"{layer_name} ({condition})", fontsize=8)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        return
    
    feature_dim = act.shape[1]
    
    # Reduce to 3D if needed
    if feature_dim == 3:
        points_3d = act
        variance_explained = None
    elif feature_dim < 3:
        points_3d = np.zeros((act.shape[0], 3))
        points_3d[:, :feature_dim] = act
        variance_explained = None
    else:
        # Normalize before reduction
        act_normalized = _normalize_activations(act, normalize)
        
        # Optional: first reduce to intermediate dimension via PCA
        if pca_dim is not None and pca_dim < feature_dim and pca_dim > 3:
            pca_intermediate = PCA(n_components=pca_dim)
            act_normalized = pca_intermediate.fit_transform(act_normalized)
        
        # Reduce to 3D
        points_3d, variance_explained = _reduce_to_3d(
            act_normalized, method=reduction_method, original_dim=feature_dim
        )
    
    # Plot
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
              s=15, alpha=0.7, c=points_3d[:, 2], cmap='viridis')
    
    # Set equal aspect
    max_range = np.array([points_3d[:, 0].max() - points_3d[:, 0].min(),
                         points_3d[:, 1].max() - points_3d[:, 1].min(),
                         points_3d[:, 2].max() - points_3d[:, 2].min()]).max() / 2.0
    mid_x = (points_3d[:, 0].max() + points_3d[:, 0].min()) * 0.5
    mid_y = (points_3d[:, 1].max() + points_3d[:, 1].min()) * 0.5
    mid_z = (points_3d[:, 2].max() + points_3d[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Title with variance explained
    if variance_explained is not None:
        dim_info = f"{feature_dim}D→3D ({variance_explained:.1f}% var)"
    else:
        dim_info = f"{feature_dim}D→3D" if feature_dim > 3 else f"{feature_dim}D"
    ax.set_title(f"{layer_name} ({condition})\n{dim_info}", fontsize=8)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


# ---------------------------
# Attack Comparison Visualizations
# ---------------------------

def save_attack_comparison_bars(
    recs: List[SampleRecord],
    norm: str,
    save_path: str,
) -> None:
    """
    Compare eps* across different attack types using box plot for better distribution visibility.
    Shows minimum perturbation needed to fool the model for each attack type.
    """
    from collections import defaultdict
    
    # Collect eps* values by attack type
    attack_eps: Dict[str, List[float]] = defaultdict(list)
    
    for rec in recs:
        # PGD (backward compatibility)
        if norm == "linf" and rec.eps_star_linf is not None:
            if not (isinstance(rec.eps_star_linf, float) and np.isnan(rec.eps_star_linf)):
                attack_eps["pgd"].append(float(rec.eps_star_linf))
        elif norm == "l2" and rec.eps_star_l2 is not None:
            if not (isinstance(rec.eps_star_l2, float) and np.isnan(rec.eps_star_l2)):
                attack_eps["pgd"].append(float(rec.eps_star_l2))
        
        # Other attacks from attack_results
        for attack_type, norms_dict in rec.attack_results.items():
            if norm in norms_dict:
                eps_star = norms_dict[norm].get("eps_star")
                if eps_star is not None and not (isinstance(eps_star, float) and np.isnan(eps_star)):
                    attack_eps[attack_type].append(float(eps_star))
    
    if not attack_eps:
        return
    
    # Use box plot instead of bar chart to show distributions
    fig, ax = new_figure(kind="custom", figsize=(10, 6))
    ax = cast(Axes, ax)
    
    attacks = sorted(attack_eps.keys())
    data = [attack_eps[at] for at in attacks]
    
    # Create box plot
    bp = ax.boxplot(data, labels=attacks, patch_artist=True, 
                    showmeans=True, meanline=False,
                    medianprops=dict(color='black', linewidth=1.5),
                    meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=6))
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(attacks)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    norm_label = r"$\ell_\infty$" if norm == "linf" else r"$\ell_2$"
    ax.set_ylabel(rf"$\epsilon^\star$ ({norm_label})", fontsize=12)
    ax.set_xlabel("Attack Type", fontsize=12)
    ax.set_title(f"Minimum Perturbation Required by Attack Type ({norm_label})\n" +
                 "Lower values = more effective attacks", fontsize=13, pad=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticklabels(attacks, rotation=45, ha='right')
    
    # Add sample count annotations
    for i, (at, vals) in enumerate(zip(attacks, data)):
        n = len(vals)
        mean_val = np.mean(vals)
        ax.text(i+1, mean_val, f'n={n}', ha='center', va='bottom', fontsize=9, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt_close(fig)


def save_attack_topology_comparison(
    diagdist_records: List[DiagramDistanceRecord],
    layer: str,
    metric: str,
    H: int,
    norm: str,
    save_path: str,
) -> None:
    """
    Compare topology distances across different attack types as epsilon increases.
    Shows how each attack affects topology at different perturbation magnitudes.
    """
    from collections import defaultdict
    
    # Group by attack type AND epsilon value
    # Format: "adv_{attack_type}_{norm}_eps={eps}"
    attack_eps_distances: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    
    for rec in diagdist_records:
        if rec.layer != layer or rec.metric != metric or rec.H != H:
            continue
        
        # Parse condition to extract attack type and epsilon
        if rec.condition.startswith("adv_"):
            parts = rec.condition.split("_")
            if len(parts) >= 3:
                attack_type = parts[1]  # e.g., "pgd", "fgsm", "cw"
                condition_norm = parts[2] if len(parts) > 2 else ""
                
                if condition_norm == norm and rec.distance is not None:
                    if not (isinstance(rec.distance, float) and np.isnan(rec.distance)):
                        # Extract epsilon value
                        try:
                            eps_str = rec.condition.split("eps=")[1] if "eps=" in rec.condition else None
                            if eps_str:
                                eps = float(eps_str)
                                attack_eps_distances[attack_type][eps].append(float(rec.distance))
                        except (ValueError, IndexError):
                            continue
    
    if not attack_eps_distances:
        return
    
    fig, ax = new_figure(kind="custom", figsize=(10, 6))
    ax = cast(Axes, ax)
    
    # Plot curves for each attack type
    colors = plt.cm.tab10(np.linspace(0, 1, len(attack_eps_distances)))
    attacks = sorted(attack_eps_distances.keys())
    
    for attack_type, color in zip(attacks, colors):
        eps_vals = sorted(attack_eps_distances[attack_type].keys())
        mean_dists = [np.mean(attack_eps_distances[attack_type][eps]) for eps in eps_vals]
        std_dists = [np.std(attack_eps_distances[attack_type][eps]) for eps in eps_vals]
        
        # Plot mean line with error bars
        ax.plot(eps_vals, mean_dists, 'o-', label=attack_type.upper(), 
                color=color, linewidth=2, markersize=6, alpha=0.8)
        ax.fill_between(eps_vals, 
                        [m - s for m, s in zip(mean_dists, std_dists)],
                        [m + s for m, s in zip(mean_dists, std_dists)],
                        alpha=0.2, color=color)
    
    norm_label = r"$\ell_\infty$" if norm == "linf" else r"$\ell_2$"
    metric_label = metric.capitalize()
    ax.set_xlabel(rf"Perturbation Magnitude $\epsilon$ ({norm_label})", fontsize=12)
    ax.set_ylabel(f"{metric_label} Distance (H{H})", fontsize=12)
    ax.set_title(f"Topology Change vs Perturbation by Attack Type\n{layer}, {metric_label} H{H}, {norm_label}",
                 fontsize=13, pad=10)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt_close(fig)


def save_attack_efficiency_scatter(
    recs: List[SampleRecord],
    diagdist_records: List[DiagramDistanceRecord],
    norm: str,
    layer: str,
    metric: str,
    H: int,
    save_path: str,
) -> None:
    """
    Scatter plot: Attack efficiency - topology change per unit perturbation.
    Shows which attacks cause more topology disruption for the same perturbation magnitude.
    Uses topology distance at eps* (minimum successful perturbation) for each attack.
    """
    from collections import defaultdict
    
    # Collect (eps_star, topology_distance_at_eps_star) pairs by attack type
    attack_data: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    
    # Build a map: (sample_id, attack_type, norm) -> eps_star
    eps_star_by_sample_attack: Dict[Tuple[int, str], float] = {}
    for rec in recs:
        # PGD
        if norm == "linf" and rec.eps_star_linf is not None:
            if not (isinstance(rec.eps_star_linf, float) and np.isnan(rec.eps_star_linf)):
                eps_star_by_sample_attack[(rec.sample_id, "pgd")] = float(rec.eps_star_linf)
        elif norm == "l2" and rec.eps_star_l2 is not None:
            if not (isinstance(rec.eps_star_l2, float) and np.isnan(rec.eps_star_l2)):
                eps_star_by_sample_attack[(rec.sample_id, "pgd")] = float(rec.eps_star_l2)
        
        # Other attacks
        for attack_type, norms_dict in rec.attack_results.items():
            if norm in norms_dict:
                eps_star = norms_dict[norm].get("eps_star")
                if eps_star is not None and not (isinstance(eps_star, float) and np.isnan(eps_star)):
                    eps_star_by_sample_attack[(rec.sample_id, attack_type)] = float(eps_star)
    
    # Get topology distances at eps* for each sample and attack
    # We'll find the closest epsilon value to eps* for each attack
    dist_by_sample_attack_eps: Dict[Tuple[int, str, float], float] = {}
    for rec in diagdist_records:
        if rec.layer == layer and rec.metric == metric and rec.H == H:
            if rec.condition.startswith("adv_") and norm in rec.condition:
                if rec.distance is not None and not (isinstance(rec.distance, float) and np.isnan(rec.distance)):
                    # Parse condition: "adv_{attack_type}_{norm}_eps={eps}"
                    parts = rec.condition.split("_")
                    if len(parts) >= 3:
                        attack_type = parts[1]
                        try:
                            eps_str = rec.condition.split("eps=")[1] if "eps=" in rec.condition else None
                            if eps_str:
                                eps = float(eps_str)
                                dist_by_sample_attack_eps[(rec.sample_id, attack_type, eps)] = float(rec.distance)
                        except (ValueError, IndexError):
                            continue
    
    # Match eps* with closest topology distance
    for (sample_id, attack_type), eps_star in eps_star_by_sample_attack.items():
        # Find topology distance at eps* (or closest epsilon)
        best_dist = None
        best_eps_diff = float('inf')
        
        for (sid, at, eps), dist in dist_by_sample_attack_eps.items():
            if sid == sample_id and at == attack_type:
                eps_diff = abs(eps - eps_star)
                if eps_diff < best_eps_diff:
                    best_eps_diff = eps_diff
                    best_dist = dist
        
        if best_dist is not None:
            attack_data[attack_type].append((eps_star, best_dist))
    
    if not attack_data:
        return
    
    fig, ax = new_figure(kind="custom", figsize=(10, 6))
    ax = cast(Axes, ax)
    
    norm_label = r"$\ell_\infty$" if norm == "linf" else r"$\ell_2$"
    metric_label = metric.capitalize()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(attack_data)))
    for (attack_type, data), color in zip(sorted(attack_data.items()), colors):
        if data:
            eps_vals, dist_vals = zip(*data)
            ax.scatter(eps_vals, dist_vals, label=attack_type.upper(), 
                      alpha=0.6, s=50, c=[color], edgecolors='black', linewidths=0.5)
    
    ax.set_xlabel(rf"Minimum Perturbation $\epsilon^\star$ ({norm_label})", fontsize=12)
    ax.set_ylabel(f"{metric_label} Distance at $\epsilon^\star$ (H{H})", fontsize=12)
    ax.set_title(f"Attack Efficiency: Topology Change vs Minimum Perturbation\n" +
                 f"{layer}, {metric_label} H{H}, {norm_label}\n" +
                 "Upper-right = more efficient (high topology change, low perturbation)",
                 fontsize=13, pad=10)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt_close(fig)


def save_attack_success_rate(
    recs: List[SampleRecord],
    norm: str,
    eps_max: float,
    save_path: str,
) -> None:
    """
    Compare attack success rates - percentage of samples where each attack succeeds.
    An attack succeeds if it finds an adversarial example (eps* is not None and <= eps_max).
    """
    from collections import defaultdict
    
    attack_successes: Dict[str, List[bool]] = defaultdict(list)
    
    for rec in recs:
        # PGD (backward compatibility)
        if norm == "linf" and rec.eps_star_linf is not None:
            if not (isinstance(rec.eps_star_linf, float) and np.isnan(rec.eps_star_linf)):
                attack_successes["pgd"].append(rec.eps_star_linf <= eps_max)
        elif norm == "l2" and rec.eps_star_l2 is not None:
            if not (isinstance(rec.eps_star_l2, float) and np.isnan(rec.eps_star_l2)):
                attack_successes["pgd"].append(rec.eps_star_l2 <= eps_max)
        
        # Other attacks from attack_results
        for attack_type, norms_dict in rec.attack_results.items():
            if norm in norms_dict:
                eps_star = norms_dict[norm].get("eps_star")
                if eps_star is not None and not (isinstance(eps_star, float) and np.isnan(eps_star)):
                    attack_successes[attack_type].append(float(eps_star) <= eps_max)
    
    if not attack_successes:
        return
    
    # Compute success rates
    attack_rates = {at: np.mean(vals) * 100.0 for at, vals in attack_successes.items() if vals}
    attack_counts = {at: len(vals) for at, vals in attack_successes.items() if vals}
    
    if not attack_rates:
        return
    
    fig, ax = new_figure(kind="custom", figsize=(10, 6))
    ax = cast(Axes, ax)
    
    attacks = sorted(attack_rates.keys())
    rates = [attack_rates[at] for at in attacks]
    counts = [attack_counts[at] for at in attacks]
    
    # Create horizontal bar chart for better readability
    y_pos = np.arange(len(attacks))
    bars = ax.barh(y_pos, rates, alpha=0.7, color=plt.cm.viridis(np.linspace(0, 1, len(attacks))))
    
    # Add value labels
    for i, (rate, count) in enumerate(zip(rates, counts)):
        ax.text(rate + 1, i, f'{rate:.1f}% (n={count})', 
                va='center', fontsize=10, fontweight='bold')
    
    norm_label = r"$\ell_\infty$" if norm == "linf" else r"$\ell_2$"
    ax.set_yticks(y_pos)
    ax.set_yticklabels([at.upper() for at in attacks])
    ax.set_xlabel(f"Success Rate (%)", fontsize=12)
    ax.set_title(f"Attack Success Rate ({norm_label}, ε_max={eps_max})\n" +
                 "Percentage of samples where attack finds adversarial example",
                 fontsize=13, pad=10)
    ax.set_xlim(0, 105)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt_close(fig)


def save_attack_agreement_matrix(
    recs: List[SampleRecord],
    norm: str,
    save_path: str,
) -> None:
    """
    Show agreement/correlation between attacks - do they agree on which samples are vulnerable?
    Computes pairwise correlation of eps* values between attack types.
    """
    from collections import defaultdict
    import pandas as pd
    
    # Collect eps* values by attack type for each sample
    attack_eps_by_sample: Dict[str, Dict[int, float]] = defaultdict(dict)
    
    for rec in recs:
        # PGD (backward compatibility)
        if norm == "linf" and rec.eps_star_linf is not None:
            if not (isinstance(rec.eps_star_linf, float) and np.isnan(rec.eps_star_linf)):
                attack_eps_by_sample["pgd"][rec.sample_id] = float(rec.eps_star_linf)
        elif norm == "l2" and rec.eps_star_l2 is not None:
            if not (isinstance(rec.eps_star_l2, float) and np.isnan(rec.eps_star_l2)):
                attack_eps_by_sample["pgd"][rec.sample_id] = float(rec.eps_star_l2)
        
        # Other attacks
        for attack_type, norms_dict in rec.attack_results.items():
            if norm in norms_dict:
                eps_star = norms_dict[norm].get("eps_star")
                if eps_star is not None and not (isinstance(eps_star, float) and np.isnan(eps_star)):
                    attack_eps_by_sample[attack_type][rec.sample_id] = float(eps_star)
    
    if len(attack_eps_by_sample) < 2:
        return
    
    # Build DataFrame with samples as rows, attacks as columns
    all_sample_ids = set()
    for attack_data in attack_eps_by_sample.values():
        all_sample_ids.update(attack_data.keys())
    
    data_dict = {}
    for sample_id in all_sample_ids:
        data_dict[sample_id] = {}
        for attack_type in attack_eps_by_sample.keys():
            data_dict[sample_id][attack_type] = attack_eps_by_sample[attack_type].get(sample_id, np.nan)
    
    df = pd.DataFrame.from_dict(data_dict, orient='index')
    df = df.dropna(how='all')  # Remove samples with no attack data
    
    if len(df) < 3 or len(df.columns) < 2:
        return
    
    # Compute correlation matrix
    corr = df.corr()
    
    fig, ax = new_figure(kind="custom", figsize=(8, 7))
    ax = cast(Axes, ax)
    
    # Use seaborn if available, otherwise matplotlib
    try:
        import seaborn as sns
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    square=True, ax=ax, vmin=-1, vmax=1, cbar_kws={'label': 'Correlation'})
    except ImportError:
        im = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr.columns)
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f'{corr.iloc[i, j]:.2f}', 
                       ha='center', va='center', color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black')
        plt.colorbar(im, ax=ax, label='Correlation')
    
    norm_label = r"$\ell_\infty$" if norm == "linf" else r"$\ell_2$"
    ax.set_title(f"Attack Agreement Matrix ({norm_label})\n" +
                 "Correlation of ε* values - High correlation = attacks find similar vulnerabilities",
                 fontsize=13, pad=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt_close(fig)


def save_attack_vs_margin(
    recs: List[SampleRecord],
    norm: str,
    save_path: str,
) -> None:
    """
    Scatter plot showing relationship between clean prediction margin and attack success (eps*).
    Tests if attacks work better on samples with smaller margins (lower confidence).
    """
    from collections import defaultdict
    
    attack_data: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    
    for rec in recs:
        if rec.clean_margin is None or (isinstance(rec.clean_margin, float) and np.isnan(rec.clean_margin)):
            continue
        
        margin = float(rec.clean_margin)
        
        # PGD (backward compatibility)
        if norm == "linf" and rec.eps_star_linf is not None:
            if not (isinstance(rec.eps_star_linf, float) and np.isnan(rec.eps_star_linf)):
                attack_data["pgd"].append((margin, float(rec.eps_star_linf)))
        elif norm == "l2" and rec.eps_star_l2 is not None:
            if not (isinstance(rec.eps_star_l2, float) and np.isnan(rec.eps_star_l2)):
                attack_data["pgd"].append((margin, float(rec.eps_star_l2)))
        
        # Other attacks
        for attack_type, norms_dict in rec.attack_results.items():
            if norm in norms_dict:
                eps_star = norms_dict[norm].get("eps_star")
                if eps_star is not None and not (isinstance(eps_star, float) and np.isnan(eps_star)):
                    attack_data[attack_type].append((margin, float(eps_star)))
    
    if not attack_data:
        return
    
    fig, ax = new_figure(kind="custom", figsize=(10, 6))
    ax = cast(Axes, ax)
    
    norm_label = r"$\ell_\infty$" if norm == "linf" else r"$\ell_2$"
    colors = plt.cm.tab10(np.linspace(0, 1, len(attack_data)))
    
    for (attack_type, data), color in zip(sorted(attack_data.items()), colors):
        if data:
            margins, eps_stars = zip(*data)
            ax.scatter(margins, eps_stars, label=attack_type.upper(), 
                      alpha=0.6, s=50, c=[color], edgecolors='black', linewidths=0.5)
            
            # Add trend line
            if len(data) > 1:
                z = np.polyfit(margins, eps_stars, 1)
                p = np.poly1d(z)
                margin_range = (min(margins), max(margins))
                ax.plot(margin_range, p(margin_range), "--", color=color, alpha=0.5, linewidth=2)
    
    ax.set_xlabel("Clean Prediction Margin", fontsize=12)
    ax.set_ylabel(rf"Minimum Perturbation $\epsilon^\star$ ({norm_label})", fontsize=12)
    ax.set_title(f"Attack Effectiveness vs Model Confidence ({norm_label})\n" +
                 "Lower margin (less confident) → Lower eps* (more vulnerable)",
                 fontsize=13, pad=10)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt_close(fig)


def save_topology_disruption_ranking(
    diagdist_records: List[DiagramDistanceRecord],
    norm: str,
    metric: str,
    H: int,
    save_path: str,
) -> None:
    """
    Rank layers by topology disruption for each attack type.
    Shows which layers are most affected by which attacks.
    """
    from collections import defaultdict
    
    # Group by attack type and layer
    # Format: "adv_{attack_type}_{norm}_eps={eps}"
    attack_layer_distances: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    
    for rec in diagdist_records:
        if rec.metric != metric or rec.H != H:
            continue
        
        if rec.condition.startswith("adv_"):
            parts = rec.condition.split("_")
            if len(parts) >= 3:
                attack_type = parts[1]
                condition_norm = parts[2] if len(parts) > 2 else ""
                
                if condition_norm == norm and rec.distance is not None:
                    if not (isinstance(rec.distance, float) and np.isnan(rec.distance)):
                        attack_layer_distances[attack_type][rec.layer].append(float(rec.distance))
    
    if not attack_layer_distances:
        return
    
    # Compute mean distances per attack per layer
    attack_layer_means: Dict[str, Dict[str, float]] = {}
    for attack_type, layer_data in attack_layer_distances.items():
        attack_layer_means[attack_type] = {
            layer: np.mean(distances) 
            for layer, distances in layer_data.items()
        }
    
    # Get all layers and attacks
    all_layers = set()
    for layer_data in attack_layer_distances.values():
        all_layers.update(layer_data.keys())
    all_layers = sorted(all_layers)
    attacks = sorted(attack_layer_distances.keys())
    
    if not all_layers or not attacks:
        return
    
    # Build matrix for heatmap
    matrix = []
    for attack_type in attacks:
        row = [attack_layer_means[attack_type].get(layer, 0.0) for layer in all_layers]
        matrix.append(row)
    
    fig, ax = new_figure(kind="custom", figsize=(max(8, len(all_layers) * 0.8), max(6, len(attacks) * 0.6)))
    ax = cast(Axes, ax)
    
    # Create heatmap
    try:
        import seaborn as sns
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlOrRd', 
                   xticklabels=all_layers, yticklabels=[at.upper() for at in attacks],
                   ax=ax, cbar_kws={'label': f'{metric.capitalize()} Distance'})
    except ImportError:
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(all_layers)))
        ax.set_yticks(range(len(attacks)))
        ax.set_xticklabels(all_layers, rotation=45, ha='right')
        ax.set_yticklabels([at.upper() for at in attacks])
        for i in range(len(attacks)):
            for j in range(len(all_layers)):
                ax.text(j, i, f'{matrix[i][j]:.3f}', 
                       ha='center', va='center', 
                       color='white' if matrix[i][j] > np.max(matrix) * 0.5 else 'black')
        plt.colorbar(im, ax=ax, label=f'{metric.capitalize()} Distance')
    
    norm_label = r"$\ell_\infty$" if norm == "linf" else r"$\ell_2$"
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Attack Type", fontsize=12)
    ax.set_title(f"Topology Disruption by Attack and Layer ({norm_label})\n" +
                 f"{metric.capitalize()} Distance H{H} - Darker = more disruption",
                 fontsize=13, pad=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt_close(fig)


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


