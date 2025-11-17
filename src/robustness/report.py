from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple
import csv
import json
import os
import numpy as np
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


