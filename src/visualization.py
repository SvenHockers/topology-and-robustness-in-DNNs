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
# Publication plotting style utilities
# -----------------------------------------------------------------------------

# Matplotlib "tab10" palette in hex (Tableau colors).
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
    # Use for text/thresholds/diagonals; keep legible but not harsh.
    "text": "#333333",
    "dark": "#444444",
    "grid": "#B0B0B0",
    "light": "#DDDDDD",
}

_ALPHA: Dict[str, float] = {
    # Convention:
    # - Filled elements (histograms, fills, confidence bands): alpha=0.60
    # - Lines/markers: alpha=1.0 unless the plot specifically needs transparency
    "fill": 0.60,
    "line": 1.00,
}

_STYLE_STATE: Dict[str, Any] = {"configured": False, "latex": None}


def get_palette() -> Dict[str, Any]:
    """
    Return the central color palette used across this repository.

    Intended usage:
    - Use `pal["clean"]` and `pal["adversarial"]` for binary comparisons.
    - Use `pal["tab10"]` (10 colors) as the default categorical cycle.
    - Use `pal["alpha"]["fill"]` (default 0.60) for filled elements.
    - Use `pal["alpha"]["line"]` (default 1.0) for lines/markers.
    """
    return {
        "clean": _TAB10_HEX[0],  # tab:blue
        "adversarial": _TAB10_HEX[1],  # tab:orange
        "tab10": list(_TAB10_HEX),
        "neutral": dict(_NEUTRAL),
        "alpha": dict(_ALPHA),
    }


def _latex_escape_text(s: str) -> str:
    """
    Escape LaTeX special characters in plain text (non-math) strings.
    """
    # Keep this minimal and predictable; users should prefer raw strings + math mode.
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
    """
    Ensure a label/title string is LaTeX-safe.

    Recommendations:
    - Prefer raw strings: r"..."
    - Use math mode when appropriate: r"$\alpha$" or r"$\mathrm{FPR}$"
    """
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


def _latex_dependency_error_message(original_exc: BaseException) -> str:
    return (
        "Matplotlib is configured to require LaTeX rendering (text.usetex=True), "
        "but LaTeX could not be executed successfully.\n\n"
        "Fix by installing a LaTeX distribution and required helpers, then rerun:\n"
        "- macOS (Homebrew): brew install --cask mactex-no-gui && brew install ghostscript\n"
        "- Ubuntu/Debian: sudo apt-get install texlive-latex-extra texlive-fonts-recommended "
        "texlive-science dvipng ghostscript\n"
        "- Conda: conda install -c conda-forge texlive-core dvipng ghostscript\n\n"
        "Original error:\n"
        f"{type(original_exc).__name__}: {original_exc}"
    )


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

    # Typography (paper-ready defaults; base font size 9–10 pt).
    base_font = 10

    rc = {
        # Layout/export
        # IMPORTANT: Use ONE layout system consistently. We default to tight_layout
        # (user code often calls plt.tight_layout()), because mixing it with
        # constrained_layout can raise RuntimeError when colorbars exist.
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
            raise RuntimeError(_latex_dependency_error_message(e)) from e

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
) -> Dict[str, str]:
    """
    Save a figure in standard paper-ready formats.

    Always includes PDF (vector). PNG is also produced for quick inspection.
    """
    _ensure_style()

    fmts = [str(f).lower().lstrip(".") for f in formats]
    if "pdf" not in fmts:
        fmts = ["pdf"] + fmts

    out: Dict[str, str] = {}
    path = Path(path_no_ext)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer tight_layout for spacing (safe with colorbars); bbox_inches="tight"
    # remains enabled to avoid label clipping in exports.
    try:
        fig.tight_layout()
    except Exception:
        # If a user configured a non-tight layout engine explicitly, don't fail
        # at export time—bbox_inches="tight" still prevents clipping.
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
    _ensure_style()
    if ax is None:
        fig, ax = new_figure(kind="single", aspect=0.62)

    pal = get_palette()
    cmap = ListedColormap([pal["tab10"][0], pal["tab10"][1]])
    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap=cmap,
        s=28,
        alpha=pal["alpha"]["line"],
        linewidths=0.0,
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(_as_latex("Class"))

    finalize_axes(ax, xlabel="Feature 1", ylabel="Feature 2", title=title, legend=False)
    
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
    _ensure_style()
    if ax is None:
        fig, ax = new_figure(kind="double", aspect=0.62)

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

    pal = get_palette()
    from matplotlib.colors import LinearSegmentedColormap

    prob_cmap = LinearSegmentedColormap.from_list(
        "clean_to_adv", [pal["clean"], pal["adversarial"]]
    )

    # Plot decision boundary / probability surface (filled element -> alpha=0.60)
    contour = ax.contourf(
        xx, yy, Z, levels=50, cmap=prob_cmap, alpha=pal["alpha"]["fill"]
    )
    ax.contour(
        xx, yy, Z, levels=[0.5], colors=pal["neutral"]["dark"], linewidths=1.5
    )

    # Plot data points (markers -> alpha=1.0)
    cls_cmap = ListedColormap([pal["tab10"][0], pal["tab10"][1]])
    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap=cls_cmap,
        s=22,
        alpha=pal["alpha"]["line"],
        edgecolors=pal["neutral"]["dark"],
        linewidths=0.4,
    )

    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label(_as_latex("Probability of Class 1"))

    finalize_axes(ax, xlabel="Feature 1", ylabel="Feature 2", title=title, legend=False)
    
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
    alpha: float = 0.60,
    density: bool = True,
    colors: Optional[Tuple[str, str]] = None,
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
    """
    Standardized wrapper around plot_score_distributions (mirrors plot_roc_from_metrics).

    Returns (fig, ax).
    """
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
    """
    Plot an ROC curve from the dict returned by src.evaluation.evaluate_detector.

    Returns (fig, ax).
    """
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
    """
    Plot TP/FP/TN/FN confusion matrix (counts + row-normalized).

    Intended for binary problems like:
      - 0 = clean, 1 = adversarial

    You can pass either:
      - y_true + y_pred
      - y_true + y_scores + threshold  (y_pred computed as y_scores >= threshold)
    """
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
        # If a single axis is provided, draw counts only.
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
    """
    Plot a persistence diagram.
    
    Args:
        diagram: Persistence diagram array of shape (n_features, 2) with (birth, death) pairs
        dimension: Homology dimension (for labeling)
        title: Plot title (defaults to "H{dimension} Persistence Diagram")
        ax: Optional axes to plot on
        min_persistence: Minimum persistence to display (filters noise)
        max_persistence: Maximum persistence for axis limits (None = auto)
        
    Returns:
        Matplotlib axes
    """
    _ensure_style()
    if ax is None:
        fig, ax = new_figure(kind="single", aspect=1.0)

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
    finite = np.isfinite(deaths)
    lifetimes = deaths[finite] - births[finite]
    valid = finite & (lifetimes >= min_persistence)
    
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
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(_as_latex("Persistence (lifetime)"))

    finalize_axes(
        ax,
        xlabel="Birth",
        ylabel="Death",
        title=title or f"H{dimension} Persistence Diagram",
        legend=True,
    )
    ax.set_aspect('equal')
    
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
    """
    Plot a local neighborhood point cloud (with optional PCA projection for high-dim data).
    
    Args:
        point_cloud: Point cloud array of shape (n_points, d)
        query_point: Optional query point to highlight (shape (d,))
        title: Plot title
        ax: Optional axes to plot on
        max_dims: Maximum dimensions to plot (if d > max_dims, use PCA projection)
        
    Returns:
        Matplotlib axes
    """
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