"""
Visualization functions for experiments.
"""

from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import ListedColormap
import torch
from pathlib import Path
from typing import Optional, Tuple, Mapping, Any, Dict, Sequence, Union

# -----------------------------------------------------------------------------
# Custom plotting style - such plots can be used in the report
# -----------------------------------------------------------------------------

# Matplotlib "tab10" palette
_TAB10_HEX: Tuple[str, ...] = (
    "#1f77b4",  # tab:blue
    "#ff7f0e",  # tab:orange
    "#2ca02c",  # tab:green
    "#d62728",  # tab:red
    "#9467bd",  # tab:purple
    "#8c564b",  # tab:brown
    "#e377c2",  # tab:pink
    "#7f7f7f",  # tab:gray
    "#bcbd22",  # tab:olive
    "#17becf",  # tab:cyan
)

_NEUTRAL: Dict[str, str] = {
    "text": "#333333",
    "dark": "#444444",
    "grid": "#B0B0B0",
    "light": "#DDDDDD",
}

_ALPHA: Dict[str, float] = {
    "fill": 0.60,
    "line": 1.00,
}

_STYLE_STATE: Dict[str, Any] = {"configured": False, "latex": None}


def get_palette() -> Dict[str, Any]:
    return {
        "clean": _TAB10_HEX[0],
        "adversarial": _TAB10_HEX[1],
        "tab10": list(_TAB10_HEX),
        "neutral": dict(_NEUTRAL),
        "alpha": dict(_ALPHA),
    }


def _latex_escape_text(s: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in s:
        out.append(repl.get(ch, ch))
    return "".join(out)


def _as_latex(s: Optional[Union[str, float, int]]) -> Optional[str]:
    if s is None:
        return None
    if not isinstance(s, str):
        s = str(s)
    # If LaTeX rendering is disabled, return plain strings so Matplotlib doesn't
    # try to parse LaTeX macros (e.g. \textnormal) via mathtext.
    if not bool(mpl.rcParams.get("text.usetex", False)):
        return s

    # If the user already provided math mode, respect it verbatim.
    if "$" in s:
        return s

    # Plain text: wrap in \textnormal{} so it renders as text with LaTeX.
    return r"\textnormal{" + _latex_escape_text(s) + "}"

def _sanity_check_latex() -> None:
    """
    Minimal runtime check that LaTeX rendering works (fail fast, no silent fallback).
    """
    # Use a tiny off-screen draw. This will fail if `latex`/`dvipng`/`gs` are missing.
    fig = plt.figure(figsize=(0.5, 0.5))
    try:
        fig.text(0.1, 0.5, r"$\alpha+\beta=\gamma$")
        fig.canvas.draw()
    finally:
        plt.close(fig)


def configure_mpl_style(latex: bool = True) -> None:
    """
    Configure Matplotlib rcParams for publication-quality figures.

    Call once at program start (recommended):
        >>> from src.visualization import configure_mpl_style
        >>> configure_mpl_style(latex=True)

    Notes:
    - When latex=True, this function performs a small sanity check and raises
      RuntimeError with actionable instructions if LaTeX is unavailable.
    - This function intentionally does NOT silently fall back to non-LaTeX.
    """
    pal = get_palette()
    base_font = 10

    rc = {
        # Layout/export
        "figure.constrained_layout.use": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.transparent": False,
        # Fonts
        "font.family": "serif",
        "font.size": base_font,
        "axes.labelsize": base_font + 1,
        "axes.titlesize": base_font + 2,
        "legend.fontsize": base_font,
        "xtick.labelsize": base_font - 1,
        "ytick.labelsize": base_font - 1,
        # Lines/spines/grid
        "axes.linewidth": 0.9,
        "lines.linewidth": 1.5,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.25,
        "grid.color": pal["neutral"]["grid"],
        "grid.linestyle": "-",
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        # Ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        # Legend
        "legend.frameon": False,
        "legend.borderaxespad": 0.2,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.6,
        "legend.labelspacing": 0.3,
        # Colors
        "axes.prop_cycle": cycler(color=pal["tab10"]),
        # LaTeX rendering
        "text.usetex": bool(latex),
        "axes.unicode_minus": False,
    }

    if latex:
        rc["text.latex.preamble"] = (
            r"\usepackage{amsmath}"
            r"\usepackage{amssymb}"
            r"\usepackage{siunitx}"
            r"\sisetup{detect-all}"
        )

    mpl.rcParams.update(rc)

    if latex:
        try:
            _sanity_check_latex()
        except Exception as e:  # pragma: no cover (depends on system LaTeX install)
            raise RuntimeError("Check if latex is installed") from e

    _STYLE_STATE["configured"] = True
    _STYLE_STATE["latex"] = bool(latex)


def set_latex_enabled(enabled: bool) -> None:
    """
    Convenience toggle for LaTeX rendering.

    - enabled=True  -> LaTeX required; fails fast if dependencies are missing.
    - enabled=False -> disable LaTeX (use Matplotlib's default text rendering).
    """
    configure_mpl_style(latex=bool(enabled))


def _ensure_style(latex: Optional[bool] = None) -> None:
    """
    Ensure style is configured before any figure is created.
    """
    if not _STYLE_STATE.get("configured", False):
        # Default to LaTeX-on unless the caller explicitly disables it.
        configure_mpl_style(latex=True if latex is None else bool(latex))
        return

    if latex is None:
        # Respect the already-configured state.
        return

    if _STYLE_STATE.get("latex") != bool(latex):
        configure_mpl_style(latex=bool(latex))


def new_figure(
    kind: str = "single",
    aspect: float = 0.62,
    nrows: int = 1,
    ncols: int = 1,
    sharex: bool = False,
    sharey: bool = False,
):
    """
    Create a new figure using standardized paper sizes.

    - kind="single": ~3.35 in width (single column)
    - kind="double": ~6.9  in width (double column)

    Height is chosen as: height = aspect * width * nrows.
    """
    _ensure_style()

    if kind not in {"single", "double"}:
        raise ValueError('kind must be "single" or "double".')

    width = 3.35 if kind == "single" else 6.9
    height = float(aspect) * float(width) * max(int(nrows), 1)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(width, height),
        sharex=sharex,
        sharey=sharey,
    )
    return (fig, axes) if (nrows * ncols) != 1 else (fig, axes)


def finalize_axes(
    ax: plt.Axes,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    legend: bool = True,
    legend_kwargs: Optional[Mapping[str, Any]] = None,
) -> plt.Axes:
    """
    Apply consistent formatting to an Axes (ticks, grid, labels, legend).

    Labels/titles are treated as LaTeX strings. Prefer:
      - raw strings: r"..."
      - math mode: r"$\mathrm{FPR}$"
    """
    _ensure_style()

    if xlabel is not None:
        ax.set_xlabel(_as_latex(xlabel))
    if ylabel is not None:
        ax.set_ylabel(_as_latex(ylabel))
    if title is not None:
        ax.set_title(_as_latex(title))

    # Light major y-grid by default; keep plot background clean.
    ax.grid(True, which="major", axis="y")
    ax.grid(False, which="minor")
    ax.minorticks_on()

    ax.tick_params(which="both", direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(mpl.rcParams.get("axes.linewidth", 0.9))

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            default_kwargs: Dict[str, Any] = {
                "frameon": False,
                "fontsize": mpl.rcParams.get("legend.fontsize", 10),
                "handlelength": mpl.rcParams.get("legend.handlelength", 1.6),
                "handletextpad": mpl.rcParams.get("legend.handletextpad", 0.6),
                "labelspacing": mpl.rcParams.get("legend.labelspacing", 0.3),
                "borderaxespad": mpl.rcParams.get("legend.borderaxespad", 0.2),
            }
            if legend_kwargs:
                default_kwargs.update(dict(legend_kwargs))
            ax.legend(**default_kwargs)

    return ax


def save_figure(
    fig: mpl.figure.Figure,
    path_no_ext: str,
    formats: Sequence[str] = ("pdf", "png"),
    dpi: int = 300,
    *,
    force_pdf: bool = True,
) -> Dict[str, str]:
    _ensure_style()

    fmts = [str(f).lower().lstrip(".") for f in formats]
    if bool(force_pdf) and "pdf" not in fmts:
        fmts = ["pdf"] + fmts

    out: Dict[str, str] = {}
    path = Path(path_no_ext)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)

    try:
        fig.tight_layout()
    except Exception:
        pass

    for fmt in fmts:
        out_path = str(path.with_suffix(f".{fmt}"))
        save_kwargs: Dict[str, Any] = {
            "bbox_inches": "tight",
            "pad_inches": mpl.rcParams.get("savefig.pad_inches", 0.02),
        }
        if fmt in {"png", "jpg", "jpeg", "tif", "tiff"}:
            save_kwargs["dpi"] = int(dpi)
        fig.savefig(out_path, **save_kwargs)
        out[fmt] = out_path

    return out


def plot_precision_recall_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    average_precision: Optional[float] = None,
    title: str = "Precision-Recall Curve",
    ax: Optional[plt.Axes] = None,
    *,
    interpolate: bool = False,
    n_points: int = 200,
    label: Optional[str] = None,
) -> plt.Axes:
    """
    Plot a precision-recall (PR) curve.

    """
    _ensure_style()
    precision = np.asarray(precision, dtype=float).ravel()
    recall = np.asarray(recall, dtype=float).ravel()
    if precision.size == 0 or recall.size == 0:
        raise ValueError("precision/recall must be non-empty arrays.")
    if precision.size != recall.size:
        raise ValueError("precision and recall must have the same length.")

    if ax is None:
        fig, ax = new_figure(kind="single", aspect=1.0)

    pal = get_palette()
    ap = float(average_precision) if average_precision is not None else None

    if interpolate and recall.size > 1:
        # recall is typically monotone increasing from 0..1; enforce monotone grid
        grid = np.linspace(0.0, 1.0, int(n_points))
        # np.interp expects x increasing; ensure it is.
        order = np.argsort(recall)
        rec_sorted = recall[order]
        prec_sorted = precision[order]
        prec_i = np.interp(grid, rec_sorted, prec_sorted)
        if label is None:
            label = f"PR curve (AP = {ap:.3f})" if ap is not None else "PR curve"
        ax.plot(grid, prec_i, color=pal["clean"], alpha=pal["alpha"]["line"], label=_as_latex(label))
    else:
        if label is None:
            label = f"PR curve (AP = {ap:.3f})" if ap is not None else "PR curve"
        ax.plot(recall, precision, color=pal["clean"], alpha=pal["alpha"]["line"], label=_as_latex(label))

    finalize_axes(
        ax,
        xlabel="Recall",
        ylabel="Precision",
        title=title,
        legend=True,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    return ax


def plot_pr_from_metrics(
    metrics: Mapping[str, Any],
    *,
    title: str = "Precision-Recall curve",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    interpolate: bool = False,
    n_points: int = 200,
):
    """
    Plot a PR curve from the dict returned by src.evaluation.evaluate_detector.

    Returns (fig, ax).
    """
    precision = np.asarray(metrics.get("pr_precision", metrics.get("precision")), dtype=float)
    recall = np.asarray(metrics.get("pr_recall", metrics.get("recall")), dtype=float)
    ap = metrics.get("pr_auc", None)

    if ax is None:
        fig, ax = new_figure(kind="single", aspect=1.0)
    else:
        fig = ax.figure

    plot_precision_recall_curve(
        precision,
        recall,
        average_precision=ap,
        title=title,
        ax=ax,
        interpolate=interpolate,
        n_points=n_points,
    )
    if show:
        plt.show()
    return fig, ax


def plot_reliability_diagram(
    predicted_probs: np.ndarray,
    y_true: np.ndarray,
    *,
    n_bins: int = 10,
    title: str = "Reliability diagram",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
):
    """
    Plot a standard reliability diagram (calibration curve).    
    """
    _ensure_style()
    p = np.asarray(predicted_probs, dtype=float).ravel()
    y = np.asarray(y_true, dtype=int).ravel()
    if p.size != y.size:
        raise ValueError("predicted_probs and y_true must have the same length.")
    p = np.clip(p, 0.0, 1.0)

    if ax is None:
        fig, ax = new_figure(kind="single", aspect=1.0)
    else:
        fig = ax.figure

    # Uniform bins in [0, 1]
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    bin_ids = np.digitize(p, edges[1:-1], right=True)

    bin_conf = np.zeros(int(n_bins), dtype=float)
    bin_acc = np.zeros(int(n_bins), dtype=float)
    bin_frac = np.zeros(int(n_bins), dtype=float)

    for b in range(int(n_bins)):
        m = (bin_ids == b)
        if not np.any(m):
            bin_conf[b] = (edges[b] + edges[b + 1]) / 2.0
            bin_acc[b] = np.nan
            bin_frac[b] = 0.0
            continue
        bin_conf[b] = float(np.mean(p[m]))
        bin_acc[b] = float(np.mean(y[m]))
        bin_frac[b] = float(np.mean(m))

    pal = get_palette()
    ax.plot([0, 1], [0, 1], "--", color=pal["neutral"]["dark"], linewidth=1.0, label=_as_latex("Ideal"))
    ax.plot(bin_conf, bin_acc, marker="o", color=pal["clean"], linewidth=1.5, label=_as_latex("Empirical"))

    finalize_axes(ax, xlabel="Predicted probability", ylabel="Empirical frequency", title=title, legend=True)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    if show:
        plt.show()
    return fig, ax


def plot_topology_feature_shift(
    topo_clean: Mapping[str, np.ndarray],
    topo_shifted: Mapping[str, np.ndarray],
    *,
    feature_keys: Optional[Sequence[str]] = None,
    title: str = "Topology feature shift",
    ax: Optional[plt.Axes] = None,
    top_k: int = 12,
    show: bool = True,
):
    """
    Visualize how topology features shift between two populations (e.g., clean vs adv/OOD).

    We plot standardized mean differences (Cohen's d)
    """
    _ensure_style()
    if feature_keys is None:
        keys = sorted(set(topo_clean.keys()) & set(topo_shifted.keys()))
        keys = [k for k in keys if str(k).startswith("topo_")]
    else:
        keys = [k for k in feature_keys if k in topo_clean and k in topo_shifted]

    if len(keys) == 0:
        raise ValueError("No shared topology feature keys found for shift plot.")

    dvals = []
    for k in keys:
        a = np.asarray(topo_clean[k], dtype=float).ravel()
        b = np.asarray(topo_shifted[k], dtype=float).ravel()
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
        ma, mb = float(np.mean(a)), float(np.mean(b))
        va, vb = float(np.var(a)), float(np.var(b))
        pooled = float(np.sqrt(0.5 * (va + vb) + 1e-12))
        d = (mb - ma) / pooled if pooled > 0 else 0.0
        dvals.append((k, d))

    dvals.sort(key=lambda x: abs(x[1]), reverse=True)
    dvals = dvals[: max(1, int(top_k))]

    names = [k.replace("topo_", "") for k, _ in dvals]
    vals = np.asarray([v for _, v in dvals], dtype=float)

    if ax is None:
        fig, ax = new_figure(kind="double", aspect=0.55)
    else:
        fig = ax.figure

    pal = get_palette()
    colors = [pal["adversarial"] if v >= 0 else pal["clean"] for v in vals]
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors, alpha=pal["alpha"]["fill"])
    ax.axvline(0.0, color=pal["neutral"]["dark"], linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([_as_latex(n) for n in names])
    ax.invert_yaxis()
    finalize_axes(ax, xlabel="Standardized mean difference (Cohen's d)", ylabel=None, title=title, legend=False)

    if show:
        plt.show()
    return fig, ax

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
    alpha: float = 0.60,
    density: bool = True,
    colors: Optional[Tuple[str, str]] = None,
) -> plt.Axes:
    _ensure_style()
    if ax is None:
        fig, ax = new_figure(kind="single", aspect=0.62)

    pal = get_palette()
    if colors is None:
        colors = (pal["clean"], pal["adversarial"])
    alpha = float(alpha)

    ax.hist(
        np.asarray(scores_clean, dtype=float).ravel(),
        bins=bins,
        alpha=alpha,
        label=_as_latex(labels[0]),
        density=density,
        color=colors[0],
    )
    ax.hist(
        np.asarray(scores_adv, dtype=float).ravel(),
        bins=bins,
        alpha=alpha,
        label=_as_latex(labels[1]),
        density=density,
        color=colors[1],
    )

    thr = float(threshold) if threshold is not None else np.nan
    if np.isfinite(thr):
        ax.axvline(
            thr,
            color=pal["neutral"]["dark"],
            linestyle="--",
            linewidth=1.0,
            label=rf"$\mathrm{{thr}}={thr:.3f}$",
        )

    finalize_axes(ax, xlabel=score_name, ylabel="Density", title=title, legend=True)
    
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
    alpha: float = 0.60,
    density: bool = True,
    colors: Optional[Tuple[str, str]] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    show: bool = True,
):
    _ensure_style()
    if ax is None:
        if figsize is None:
            fig, ax = new_figure(kind="single", aspect=0.62)
        else:
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
    _ensure_style()
    if ax is None:
        fig, ax = new_figure(kind="single", aspect=1.0)

    auc = float(auc_score if auc_score is not None else roc_auc) if (auc_score is not None or roc_auc is not None) else None
    pal = get_palette()

    if interpolate and len(fpr) > 1:
        grid = np.linspace(0.0, 1.0, int(n_points))
        # roc_curve output is monotone in fpr; interpolate tpr for smoother plotting
        tpr_i = np.interp(grid, fpr, tpr)
        if label is None:
            label = f"ROC curve (AUC = {auc:.3f})" if auc is not None else "ROC curve"
        ax.plot(grid, tpr_i, color=pal["clean"], alpha=pal["alpha"]["line"], label=_as_latex(label))
    else:
        if label is None:
            label = f"ROC curve (AUC = {auc:.3f})" if auc is not None else "ROC curve"
        ax.plot(fpr, tpr, color=pal["clean"], alpha=pal["alpha"]["line"], label=_as_latex(label))

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color=pal["neutral"]["dark"],
        linewidth=1.0,
        label=_as_latex("Random classifier"),
    )

    finalize_axes(
        ax,
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=title,
        legend=True,
    )
    
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
    fpr = np.asarray(metrics.get("fpr"), dtype=float)
    tpr = np.asarray(metrics.get("tpr"), dtype=float)
    roc_auc = metrics.get("roc_auc", None)

    if ax is None:
        fig, ax = new_figure(kind="single", aspect=1.0)
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
    _ensure_style()
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
        fig, axes = new_figure(kind="double", aspect=0.45, ncols=2)
    else:
        fig = ax.figure
        axes = [ax]

    # Left: counts
    ax0 = axes[0]
    ax0.imshow(cm, cmap="Blues")
    ax0.set_title(_as_latex("Confusion (counts)"))
    ax0.set_xticks([0, 1]); ax0.set_yticks([0, 1])
    ax0.set_xticklabels([_as_latex(f"pred {labels[0]}"), _as_latex(f"pred {labels[1]}")])
    ax0.set_yticklabels([_as_latex(f"true {labels[0]}"), _as_latex(f"true {labels[1]}")])
    for (i, j), val in np.ndenumerate(cm):
        ax0.text(j, i, f"{int(val)}", ha="center", va="center")
    finalize_axes(ax0, legend=False)
    ax0.grid(False)

    # Right: normalized
    if len(axes) > 1:
        ax1 = axes[1]
        ax1.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
        ax1.set_title(_as_latex("Confusion (row-normalized)"))
        ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1])
        ax1.set_xticklabels([_as_latex(f"pred {labels[0]}"), _as_latex(f"pred {labels[1]}")])
        ax1.set_yticklabels([_as_latex(f"true {labels[0]}"), _as_latex(f"true {labels[1]}")])
        for (i, j), val in np.ndenumerate(cm_norm):
            ax1.text(j, i, f"{val:.2f}", ha="center", va="center")
        finalize_axes(ax1, legend=False)
        ax1.grid(False)

    if show:
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
    _ensure_style()
    if ax is None:
        fig, ax = new_figure(kind="double", aspect=0.62)

    pal = get_palette()
    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=scores,
        cmap=cmap,
        s=26,
        alpha=pal["alpha"]["line"],
        linewidths=0.0,
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(_as_latex("Score"))

    finalize_axes(ax, xlabel="Feature 1", ylabel="Feature 2", title=title, legend=False)
    
    return ax


def plot_adversarial_examples(
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    y: np.ndarray,
    title: str = "Clean vs Adversarial Examples",
    n_samples: int = 100,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    _ensure_style()
    if ax is None:
        fig, ax = new_figure(kind="double", aspect=0.62)

    pal = get_palette()

    # Sample subset if too many points
    if len(X_clean) > n_samples:
        indices = np.random.choice(len(X_clean), n_samples, replace=False)
        X_clean = X_clean[indices]
        X_adv = X_adv[indices]
        y = y[indices]
    
    # Plot clean points
    cls_cmap = ListedColormap([pal["tab10"][0], pal["tab10"][1]])
    scatter_clean = ax.scatter(
        X_clean[:, 0],
        X_clean[:, 1],
        c=y,
        cmap=cls_cmap,
        s=26,
        alpha=pal["alpha"]["line"],
        marker="o",
        label=_as_latex("Clean"),
        edgecolors="none",
    )
    
    # Plot adversarial points
    scatter_adv = ax.scatter(
        X_adv[:, 0],
        X_adv[:, 1],
        c=y,
        cmap=cls_cmap,
        s=30,
        alpha=pal["alpha"]["line"],
        marker="x",
        label=_as_latex("Adversarial"),
    )
    
    # Draw arrows showing perturbations
    for i in range(len(X_clean)):
        dx = X_adv[i, 0] - X_clean[i, 0]
        dy = X_adv[i, 1] - X_clean[i, 1]
        ax.arrow(
            X_clean[i, 0],
            X_clean[i, 1],
            dx,
            dy,
            head_width=0.05,
            head_length=0.05,
            fc=pal["neutral"]["grid"],
            ec=pal["neutral"]["grid"],
            alpha=0.35,
            linewidth=0.6,
            length_includes_head=True,
        )

    finalize_axes(ax, xlabel="Feature 1", ylabel="Feature 2", title=title, legend=True)
    
    return ax


def plot_persistence_diagram(
    diagram: np.ndarray,
    dimension: int = 0,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    min_persistence: float = 1e-6,
    max_persistence: Optional[float] = None,
) -> plt.Axes:      
    _ensure_style()
    created_fig = False
    if ax is None:
        fig, ax = new_figure(kind="single", aspect=1.0)
        created_fig = True
    else:
        fig = ax.figure

    pal = get_palette()

    if diagram.size == 0:
        ax.text(
            0.5,
            0.5,
            _as_latex("No features"),
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        finalize_axes(
            ax,
            xlabel="Birth",
            ylabel="Death",
            title=title or f"H{dimension} Persistence Diagram (empty)",
            legend=False,
        )
        return ax
    
    # Filter by persistence
    births = diagram[:, 0]
    deaths = diagram[:, 1]
    # Some PH backends (e.g. ripser) encode essential features with death=inf.
    # Compute lifetimes in full shape, then mask with finite deaths.
    finite = np.isfinite(deaths)
    lifetimes = deaths - births
    lifetimes = np.where(np.isfinite(lifetimes), lifetimes, -np.inf)
    valid = finite & (lifetimes >= float(min_persistence))
    
    if valid.sum() == 0:
        ax.text(
            0.5,
            0.5,
            _as_latex("No features above min_persistence"),
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        finalize_axes(
            ax,
            xlabel="Birth",
            ylabel="Death",
            title=title or f"H{dimension} Persistence Diagram",
            legend=False,
        )
        return ax
    
    births_plot = births[valid]
    deaths_plot = deaths[valid]
    lifetimes_plot = lifetimes[valid]
    
    # Plot diagonal line (y=x)
    if max_persistence is None:
        max_val = max(deaths_plot.max(), births_plot.max()) if len(deaths_plot) > 0 else 1.0
    else:
        max_val = max_persistence
    ax.plot(
        [0, max_val],
        [0, max_val],
        linestyle="--",
        color=pal["neutral"]["dark"],
        alpha=0.8,
        linewidth=1.0,
        label=_as_latex("Diagonal"),
    )
    
    # Plot points with size proportional to persistence
    scatter = ax.scatter(
        births_plot, deaths_plot,
        s=50 + 100 * (lifetimes_plot / lifetimes_plot.max() if lifetimes_plot.max() > 0 else 1),
        c=lifetimes_plot,
        cmap="viridis",
        alpha=pal["alpha"]["line"],
        edgecolors=pal["neutral"]["dark"],
        linewidths=0.4,
    )
    
    # Colorbar
    # Use explicit sizing so the colorbar doesn't collide with the title in saved figures.
    cbar = fig.colorbar(scatter, ax=ax, pad=0.03, fraction=0.055, shrink=0.92)
    cbar.set_label(_as_latex("Persistence (lifetime)"))

    finalize_axes(
        ax,
        xlabel="Birth",
        ylabel="Death",
        title=title or f"H{dimension} Persistence Diagram",
        legend=True,
    )
    # Give the title a bit more breathing room from the top spine.
    ax.set_title(ax.get_title(), pad=12)
    ax.set_aspect('equal')

    # When we own the figure, reserve a touch more space for the colorbar and title.
    if created_fig:
        try:
            fig.subplots_adjust(right=0.86, top=0.90)
        except Exception:
            pass
    
    return ax


def plot_topology_summary_features(
    features: Dict[str, float],
    title: str = "Topology Summary Features",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot topology summary features as a bar chart.
    
    Args:
        features: Dictionary of topology features (e.g., from persistence_summary_features)
        title: Plot title
        ax: Optional axes to plot on
        
    Returns:
        Matplotlib axes
    """
    _ensure_style()
    if ax is None:
        fig, ax = new_figure(kind="double", aspect=0.50)

    pal = get_palette()

    # Extract feature names and values
    feature_names = sorted(features.keys())
    feature_values = [features[k] for k in feature_names]
    
    # Create bar plot
    bars = ax.bar(
        range(len(feature_names)),
        feature_values,
        alpha=pal["alpha"]["fill"],
        color=pal["clean"],
        edgecolor="none",
    )
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels([_as_latex(n) for n in feature_names], rotation=45, ha='right')
    finalize_axes(ax, xlabel=None, ylabel="Feature Value", title=title, legend=False)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, feature_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    return ax


def plot_local_neighborhood(
    point_cloud: np.ndarray,
    query_point: Optional[np.ndarray] = None,
    title: str = "Local Neighborhood Point Cloud",
    ax: Optional[plt.Axes] = None,
    max_dims: int = 2,
) -> plt.Axes:
    _ensure_style()
    if ax is None:
        fig, ax = new_figure(kind="single", aspect=1.0)

    pal = get_palette()

    n_points, d = point_cloud.shape
    
    # Project to 2D if needed
    if d > max_dims:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        point_cloud_2d = pca.fit_transform(point_cloud)
        if query_point is not None:
            query_point_2d = pca.transform(query_point.reshape(1, -1))[0]
        else:
            query_point_2d = None
        explained_var = pca.explained_variance_ratio_.sum()
        title_suffix = f" (PCA projection, {explained_var:.1%} variance explained)"
    else:
        point_cloud_2d = point_cloud[:, :2] if d >= 2 else point_cloud
        query_point_2d = query_point[:2] if query_point is not None and len(query_point) >= 2 else None
        title_suffix = ""
    
    # Plot neighborhood points
    ax.scatter(point_cloud_2d[:, 0], point_cloud_2d[:, 1], 
              s=22, alpha=pal["alpha"]["line"], c=pal["clean"], label=_as_latex("Neighborhood points"),
              edgecolors='none', linewidths=0.0)
    
    # Highlight query point if provided
    if query_point_2d is not None:
        ax.scatter(query_point_2d[0], query_point_2d[1], 
                  s=140, c=pal["adversarial"], marker='*', label=_as_latex("Query point"),
                  edgecolors=pal["neutral"]["dark"], linewidths=0.6, zorder=10)

    finalize_axes(
        ax,
        xlabel="Dimension 1",
        ylabel="Dimension 2",
        title=title + title_suffix,
        legend=True,
    )
    ax.set_aspect('equal')
    
    return ax


def plot_topology_feature_pca(
    topo_clean: Mapping[str, np.ndarray],
    topo_shifted: Mapping[str, np.ndarray],
    *,
    feature_keys: Sequence[str],
    scores_clean: Optional[np.ndarray] = None,
    scores_shifted: Optional[np.ndarray] = None,
    title: str = "Topology feature PCA (clean vs shifted)",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> Tuple[mpl.figure.Figure, plt.Axes]:
    _ensure_style()
    keys = [k for k in feature_keys if k in topo_clean and k in topo_shifted]
    if len(keys) == 0:
        raise ValueError("plot_topology_feature_pca: no valid feature_keys present in both dicts.")

    V0 = np.column_stack([np.asarray(topo_clean[k], dtype=float).ravel() for k in keys])
    V1 = np.column_stack([np.asarray(topo_shifted[k], dtype=float).ravel() for k in keys])
    V0 = np.nan_to_num(V0, nan=0.0, posinf=0.0, neginf=0.0)
    V1 = np.nan_to_num(V1, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize by clean stats for interpretability.
    mu = V0.mean(axis=0, keepdims=True)
    sd = V0.std(axis=0, keepdims=True) + 1e-12
    Z0 = (V0 - mu) / sd
    Z1 = (V1 - mu) / sd
    Z = np.vstack([Z0, Z1])

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    P = pca.fit_transform(Z)
    P0 = P[: Z0.shape[0]]
    P1 = P[Z0.shape[0] :]

    if ax is None:
        fig, ax = new_figure(kind="double", aspect=0.62)
    else:
        fig = ax.figure

    pal = get_palette()

    # Optionally size points by detector score (helps see where high-suspiciousness sits).
    def _sizes(s: Optional[np.ndarray], n: int) -> np.ndarray:
        if s is None:
            return np.full((n,), 20.0, dtype=float)
        s = np.asarray(s, dtype=float).ravel()
        if s.size != n:
            return np.full((n,), 20.0, dtype=float)
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        s = (s - s.min()) / (s.max() - s.min() + 1e-12)
        return 20.0 + 80.0 * s

    s0 = _sizes(scores_clean, P0.shape[0])
    s1 = _sizes(scores_shifted, P1.shape[0])

    ax.scatter(P0[:, 0], P0[:, 1], s=s0, alpha=0.65, c=pal["clean"], label=_as_latex("Clean"), edgecolors="none")
    ax.scatter(P1[:, 0], P1[:, 1], s=s1, alpha=0.65, c=pal["adversarial"], label=_as_latex("Shifted"), edgecolors="none")

    ev = float(np.sum(pca.explained_variance_ratio_))
    finalize_axes(
        ax,
        xlabel=f"PC1",
        ylabel=f"PC2",
        title=f"{title} ({ev:.1%} var explained)",
        legend=True,
    )

    if show:
        plt.show()
    return fig, ax


def plot_topology_explanation_panel(
    *,
    X_point: np.ndarray,
    model,
    Z_train: np.ndarray,
    f_train: Optional[np.ndarray],
    graph_params,
    device: str = "cpu",
    score: Optional[float] = None,
    threshold: Optional[float] = None,
    title: str = "Topology detector explanation",
) -> Tuple[mpl.figure.Figure, Dict[str, Any]]:  
    _ensure_style()
    from matplotlib.gridspec import GridSpec
    from .graph_scoring import compute_graph_scores_with_diagrams

    feats, diagrams, cloud = compute_graph_scores_with_diagrams(
        X_point=np.asarray(X_point),
        model=model,
        Z_train=np.asarray(Z_train),
        f_train=np.asarray(f_train) if f_train is not None else np.zeros((len(Z_train),), dtype=float),
        graph_params=graph_params,
        device=str(device),
    )

    fig = plt.figure(figsize=(6.9, 4.2))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.1, 1.0, 1.0], height_ratios=[1.0, 1.0])

    ax_cloud = fig.add_subplot(gs[:, 0])
    ax_pd0 = fig.add_subplot(gs[0, 1])
    ax_pd1 = fig.add_subplot(gs[0, 2])
    ax_feat = fig.add_subplot(gs[1, 1:])

    # Neighborhood plot (query point is the first row of cloud by construction).
    plot_local_neighborhood(
        point_cloud=np.asarray(cloud),
        query_point=np.asarray(cloud[0]),
        title="Local neighborhood",
        ax=ax_cloud,
        max_dims=2,
    )

    # Persistence diagrams
    if len(diagrams) >= 1:
        plot_persistence_diagram(np.asarray(diagrams[0]), dimension=0, title="H0 diagram", ax=ax_pd0)
    else:
        ax_pd0.text(0.5, 0.5, _as_latex("No diagram"), ha="center", va="center", transform=ax_pd0.transAxes)
        finalize_axes(ax_pd0, title="H0 diagram", legend=False)

    if len(diagrams) >= 2:
        plot_persistence_diagram(np.asarray(diagrams[1]), dimension=1, title="H1 diagram", ax=ax_pd1)
    else:
        ax_pd1.text(0.5, 0.5, _as_latex("No H1 (maxdim<1)"), ha="center", va="center", transform=ax_pd1.transAxes)
        finalize_axes(ax_pd1, title="H1 diagram", legend=False)

    # Feature bar chart
    plot_topology_summary_features(feats, title="Topology summary features", ax=ax_feat)

    # Global title / score annotation
    st = title
    if score is not None and threshold is not None:
        st += f"\nscore={float(score):.3f}, threshold={float(threshold):.3f}"
    elif score is not None:
        st += f"\nscore={float(score):.3f}"
    fig.suptitle(_as_latex(st))

    try:
        fig.tight_layout()
    except Exception:
        pass

    payload: Dict[str, Any] = {"features": feats, "diagrams": diagrams, "cloud": cloud}
    return fig, payload


def plot_persistence_diagram_comparison(
    diagram_clean: np.ndarray,
    diagram_shifted: np.ndarray,
    *,
    dimension: int = 0,
    title_clean: str = "Clean",
    title_shifted: str = "Shifted",
    min_persistence: float = 1e-6,
    max_persistence: Optional[float] = None,
    show: bool = True,
) -> Tuple[mpl.figure.Figure, Sequence[plt.Axes]]:
    _ensure_style()
    d0 = np.asarray(diagram_clean, dtype=float)
    d1 = np.asarray(diagram_shifted, dtype=float)

    # Choose a common max_persistence if not provided.
    def _auto_max(diag: np.ndarray) -> float:
        if diag.size == 0:
            return 1.0
        births = diag[:, 0]
        deaths = diag[:, 1]
        finite = np.isfinite(deaths)
        if not np.any(finite):
            return 1.0
        return float(np.max(np.maximum(births[finite], deaths[finite])))

    if max_persistence is None:
        max_val = max(_auto_max(d0), _auto_max(d1))
    else:
        max_val = float(max_persistence)

    fig, axes = new_figure(kind="double", aspect=0.6, ncols=2)
    ax0, ax1 = axes[0], axes[1]

    plot_persistence_diagram(
        d0,
        dimension=int(dimension),
        title=f"{title_clean} (H{int(dimension)})",
        ax=ax0,
        min_persistence=float(min_persistence),
        max_persistence=max_val,
    )
    plot_persistence_diagram(
        d1,
        dimension=int(dimension),
        title=f"{title_shifted} (H{int(dimension)})",
        ax=ax1,
        min_persistence=float(min_persistence),
        max_persistence=max_val,
    )

    # Make sure axes are identical (helps visual comparison).
    ax0.set_xlim(0.0, max_val)
    ax0.set_ylim(0.0, max_val)
    ax1.set_xlim(0.0, max_val)
    ax1.set_ylim(0.0, max_val)

    if show:
        plt.show()
    return fig, (ax0, ax1)


def plot_persistence_diagrams_grid(
    diagrams_clean: Sequence[np.ndarray],
    diagrams_shifted: Sequence[np.ndarray],
    *,
    maxdim: int = 1,
    min_persistence: float = 1e-6,
    title_clean: str = "Clean",
    title_shifted: str = "Shifted",
    show: bool = True,
) -> Tuple[mpl.figure.Figure, np.ndarray]:  
    _ensure_style()
    m = int(max(0, maxdim))
    nrows = m + 1
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(6.9, 2.4 * nrows))
    axes = np.asarray(axes)

    # Shared max for all plotted diagrams (ensures comparability across dims).
    def _auto_max(diags: Sequence[np.ndarray]) -> float:
        mv = 1.0
        for diag in diags:
            diag = np.asarray(diag, dtype=float)
            if diag.size == 0:
                continue
            births = diag[:, 0]
            deaths = diag[:, 1]
            finite = np.isfinite(deaths)
            if np.any(finite):
                mv = max(mv, float(np.max(np.maximum(births[finite], deaths[finite]))))
        return mv

    max_val = max(_auto_max(diagrams_clean[: nrows]), _auto_max(diagrams_shifted[: nrows]))

    for d in range(nrows):
        dc = np.asarray(diagrams_clean[d], dtype=float) if len(diagrams_clean) > d else np.zeros((0, 2), dtype=float)
        ds = np.asarray(diagrams_shifted[d], dtype=float) if len(diagrams_shifted) > d else np.zeros((0, 2), dtype=float)
        plot_persistence_diagram(dc, dimension=d, title=f"{title_clean} (H{d})", ax=axes[d, 0], min_persistence=min_persistence, max_persistence=max_val)
        plot_persistence_diagram(ds, dimension=d, title=f"{title_shifted} (H{d})", ax=axes[d, 1], min_persistence=min_persistence, max_persistence=max_val)
        axes[d, 0].set_xlim(0.0, max_val); axes[d, 0].set_ylim(0.0, max_val)
        axes[d, 1].set_xlim(0.0, max_val); axes[d, 1].set_ylim(0.0, max_val)

    try:
        fig.tight_layout()
    except Exception:
        pass
    if show:
        plt.show()
    return fig, axes