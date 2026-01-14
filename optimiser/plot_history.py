from __future__ import annotations

"""
Utility to visualize optimiser runs from `history.jsonl`.

Designed to be dependency-light (numpy + matplotlib only).

Example:
  python -m optimiser.plot_history \
    --history test_out/shapes/history.jsonl \
    --outdir test_out/shapes/figs \
    --space optimiser/spaces/topology_basic.yaml
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_any_spec(path: Path) -> Mapping[str, Any]:
    suf = path.suffix.lower()
    if suf == ".json":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
        return {"params": obj}
    if suf in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("YAML support requires PyYAML (pip install pyyaml).") from e
        with path.open("r", encoding="utf-8") as f:
            obj = yaml.safe_load(f)
        if isinstance(obj, dict):
            return obj
        return {"params": obj}
    raise ValueError(f"Unsupported spec format: {path.suffix} (use .json/.yaml/.yml)")


def _infer_param_order(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    keys: set[str] = set()
    for r in rows:
        p = r.get("params") or {}
        if isinstance(p, dict):
            keys.update(str(k) for k in p.keys())
    return sorted(keys)


def _encode_params(
    *,
    rows: Sequence[Mapping[str, Any]],
    param_order: Sequence[str],
    space_spec_path: Optional[Path],
) -> Tuple["np.ndarray", List[str], Dict[str, List[str]]]:
    """
    Returns:
      X: (n, d) numeric matrix suitable for plotting/embedding
      labels: parameter names aligned with columns
      cat_levels: mapping param_name -> sorted levels (only for categoricals)
    """
    import numpy as np  # type: ignore

    cat_levels: Dict[str, List[str]] = {}

    if space_spec_path is not None:
        # Use optimiser's search space spec to encode correctly (log/categorical).
        from optimiser.search_space import specs_from_dict

        spec = _load_any_spec(space_spec_path)
        space = specs_from_dict(spec)
        labels = [p.name for p in space.params]

        # Gather categorical levels from spec order.
        for p in space.params:
            if getattr(p, "kind", None) == "categorical" and getattr(p, "choices", None):
                cat_levels[p.name] = list(p.choices)  # type: ignore[arg-type]

        X = []
        for r in rows:
            params = r.get("params") or {}
            if not isinstance(params, dict):
                params = {}
            X.append(space.vectorize(params))
        return np.asarray(X, dtype=float), labels, cat_levels

    # Fallback: infer numeric/categorical from values (strings become categoricals).
    labels = list(param_order)
    # Determine categorical levels
    for k in labels:
        levels: set[str] = set()
        for r in rows:
            params = r.get("params") or {}
            if isinstance(params, dict) and k in params and isinstance(params[k], str):
                levels.add(params[k])
        if levels:
            cat_levels[k] = sorted(levels)

    X = np.full((len(rows), len(labels)), np.nan, dtype=float)
    for i, r in enumerate(rows):
        params = r.get("params") or {}
        if not isinstance(params, dict):
            continue
        for j, k in enumerate(labels):
            if k not in params:
                continue
            v = params[k]
            if isinstance(v, str) and k in cat_levels:
                X[i, j] = float(cat_levels[k].index(v))
            else:
                try:
                    X[i, j] = float(v)
                except Exception:
                    X[i, j] = np.nan
    return X, labels, cat_levels


def _extract_objective(rows: Sequence[Mapping[str, Any]], *, field: str) -> Tuple["np.ndarray", "np.ndarray"]:
    import numpy as np  # type: ignore

    trial_id = np.asarray([int(r.get("trial_id", i + 1)) for i, r in enumerate(rows)], dtype=int)
    y = np.asarray([float(r.get(field)) if r.get(field) is not None else float("nan") for r in rows], dtype=float)
    return trial_id, y


def _best_so_far(y: "np.ndarray") -> "np.ndarray":
    import numpy as np  # type: ignore

    out = np.empty_like(y)
    best = -float("inf")
    for i, v in enumerate(y.tolist()):
        if not np.isfinite(v):
            out[i] = best
            continue
        best = max(best, float(v))
        out[i] = best
    return out


def _rankdata_average_ties(x: "np.ndarray") -> "np.ndarray":
    """
    Rank data with average ranks for ties (1..n).
    Minimal replacement for scipy.stats.rankdata(method="average").
    """
    import numpy as np  # type: ignore

    x = np.asarray(x, dtype=float)
    n = int(x.shape[0])
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        # ranks are 1-indexed
        avg_rank = 0.5 * ((i + 1) + (j + 1))
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _spearmanr(x: "np.ndarray", y: "np.ndarray") -> float:
    """
    Spearman correlation with average-tie ranks; returns NaN if undefined.
    """
    import numpy as np  # type: ignore

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(m)) < 3:
        return float("nan")
    rx = _rankdata_average_ties(x[m])
    ry = _rankdata_average_ties(y[m])
    rx = rx - float(np.mean(rx))
    ry = ry - float(np.mean(ry))
    denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(rx * ry) / denom)


def _save_best_so_far(trial_id: "np.ndarray", y: "np.ndarray", out_path: Path, *, ylabel: str) -> None:
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore

    fig = plt.figure(figsize=(5.6, 3.4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(trial_id, y, color="#999999", linewidth=1.0, alpha=0.6, label="trial value")
    ax.plot(trial_id, _best_so_far(y), color="#1f77b4", linewidth=2.0, label="best-so-far")
    ax.set_xlabel("Trial")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_scatter_matrix(
    X: "np.ndarray",
    y: "np.ndarray",
    labels: Sequence[str],
    out_path: Path,
    *,
    max_dims: int = 6,
) -> None:
    """
    Pairwise scatter (upper triangle) + histograms on diagonal.
    For many dimensions, we keep the first `max_dims`.
    """
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore

    d = int(X.shape[1])
    keep = min(d, int(max_dims))
    Xk = X[:, :keep]
    labs = list(labels[:keep])

    fig, axes = plt.subplots(keep, keep, figsize=(2.0 * keep, 2.0 * keep), squeeze=False)
    vmin = float(np.nanmin(y)) if np.any(np.isfinite(y)) else 0.0
    vmax = float(np.nanmax(y)) if np.any(np.isfinite(y)) else 1.0

    for i in range(keep):
        for j in range(keep):
            ax = axes[i][j]
            if i == j:
                ax.hist(Xk[:, j][np.isfinite(Xk[:, j])], bins=12, color="#cccccc", edgecolor="white")
            else:
                ax.scatter(
                    Xk[:, j],
                    Xk[:, i],
                    c=y,
                    s=16,
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                    alpha=0.85,
                    linewidths=0.0,
                )
            if i == keep - 1:
                ax.set_xlabel(labs[j], fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(labs[i], fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.grid(True, alpha=0.15)

    # Add a shared colorbar
    mappable = axes[0][1].collections[0] if keep >= 2 and axes[0][1].collections else None
    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes, fraction=0.02, pad=0.02)
        cbar.set_label("objective", rotation=90)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_parallel_coordinates(
    X: "np.ndarray",
    y: "np.ndarray",
    labels: Sequence[str],
    out_path: Path,
    *,
    cat_levels: Mapping[str, List[str]],
) -> None:
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore

    # Normalize each column to 0..1 for visualization (ignoring NaNs)
    Xn = np.asarray(X, dtype=float).copy()
    for j in range(Xn.shape[1]):
        col = Xn[:, j]
        m = np.nanmin(col)
        M = np.nanmax(col)
        if not np.isfinite(m) or not np.isfinite(M) or abs(M - m) < 1e-12:
            Xn[:, j] = 0.5
        else:
            Xn[:, j] = (col - m) / (M - m)

    fig = plt.figure(figsize=(max(6.0, 0.9 * Xn.shape[1]), 3.8))
    ax = fig.add_subplot(1, 1, 1)
    xs = np.arange(Xn.shape[1], dtype=float)

    vmin = float(np.nanmin(y)) if np.any(np.isfinite(y)) else 0.0
    vmax = float(np.nanmax(y)) if np.any(np.isfinite(y)) else 1.0

    for i in range(Xn.shape[0]):
        yi = y[i]
        if not np.isfinite(yi):
            continue
        ax.plot(xs, Xn[i, :], color=plt.cm.viridis((yi - vmin) / (vmax - vmin + 1e-12)), alpha=0.35, linewidth=1.0)

    ax.set_xlim(xs.min(), xs.max())
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("normalized parameter value")
    ax.grid(True, alpha=0.2)

    # Add categorical tick annotations (levels) if present.
    # We annotate below the axis to avoid overcrowding.
    for j, name in enumerate(labels):
        if name in cat_levels and cat_levels[name]:
            levs = ", ".join(cat_levels[name])
            ax.text(j, -0.12, levs, ha="center", va="top", fontsize=7, transform=ax.get_xaxis_transform())

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap="viridis")
    sm.set_clim(vmin, vmax)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("objective", rotation=90)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_pca_embedding(
    X: "np.ndarray",
    y: "np.ndarray",
    out_path: Path,
) -> None:
    """
    2D PCA of the parameter vectors (after standardization), colored by objective.
    """
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore

    X2 = np.asarray(X, dtype=float)
    # Impute NaNs with column medians
    for j in range(X2.shape[1]):
        col = X2[:, j]
        med = np.nanmedian(col)
        if not np.isfinite(med):
            med = 0.0
        col[np.isnan(col)] = med
        X2[:, j] = col

    # Standardize
    mu = X2.mean(axis=0, keepdims=True)
    sd = X2.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Z = (X2 - mu) / sd

    # PCA via SVD
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    emb = Z @ Vt[:2].T

    vmin = float(np.nanmin(y)) if np.any(np.isfinite(y)) else 0.0
    vmax = float(np.nanmax(y)) if np.any(np.isfinite(y)) else 1.0

    fig = plt.figure(figsize=(4.6, 3.8))
    ax = fig.add_subplot(1, 1, 1)
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap="viridis", s=28, alpha=0.9, vmin=vmin, vmax=vmax)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Parameter space (PCA)")
    ax.grid(True, alpha=0.2)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
    cbar.set_label("objective", rotation=90)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_category_boxplot(
    *,
    rows: Sequence[Mapping[str, Any]],
    category_param: str,
    y: "np.ndarray",
    out_path: Path,
    ylabel: str,
) -> bool:
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore

    cats: List[str] = []
    for r in rows:
        p = r.get("params") or {}
        if isinstance(p, dict) and category_param in p and isinstance(p[category_param], str):
            cats.append(str(p[category_param]))
        else:
            cats.append("missing")

    uniq = [c for c in sorted(set(cats)) if c != "missing"]
    if not uniq:
        return False

    groups = []
    for u in uniq:
        idx = [i for i, c in enumerate(cats) if c == u and np.isfinite(y[i])]
        groups.append([float(y[i]) for i in idx])
    if not any(len(g) for g in groups):
        return False

    fig = plt.figure(figsize=(4.2, 3.4))
    ax = fig.add_subplot(1, 1, 1)
    ax.boxplot(groups, labels=uniq, showfliers=False)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{category_param} vs objective")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def _save_top_vs_rest_marginals(
    *,
    X: "np.ndarray",
    y: "np.ndarray",
    labels: Sequence[str],
    out_path: Path,
    top_fraction: float,
    cat_levels: Mapping[str, List[str]],
    ylabel: str,
) -> None:
    """
    For each parameter:
      - numeric: overlay hist for top-fraction vs rest
      - categorical: bar plot of frequency in top vs rest
    """
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore

    top_fraction = float(max(0.01, min(0.5, top_fraction)))
    m = np.isfinite(y)
    if not bool(np.any(m)):
        return
    y2 = y.copy()
    y2[~m] = -float("inf")
    thresh = float(np.quantile(y2[m], 1.0 - top_fraction))
    top = m & (y >= thresh)
    rest = m & ~top

    d = int(X.shape[1])
    ncols = 3
    nrows = int(np.ceil(d / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.7 * nrows), squeeze=False)

    for idx, name in enumerate(labels):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]
        col = X[:, idx]
        if name in cat_levels:
            levs = cat_levels[name]
            # convert encoded indices back to levels by nearest integer
            counts_top = []
            counts_rest = []
            for li in range(len(levs)):
                counts_top.append(int(np.sum(top & (np.round(col) == li))))
                counts_rest.append(int(np.sum(rest & (np.round(col) == li))))
            xs = np.arange(len(levs), dtype=float)
            w = 0.38
            ax.bar(xs - w / 2, counts_top, width=w, label=f"top {int(top_fraction*100)}%", color="#1f77b4")
            ax.bar(xs + w / 2, counts_rest, width=w, label="rest", color="#cccccc")
            ax.set_xticks(xs)
            ax.set_xticklabels(levs, rotation=0, fontsize=8)
            ax.set_ylabel("count")
        else:
            # numeric overlay
            v_top = col[top & np.isfinite(col)]
            v_rest = col[rest & np.isfinite(col)]
            if v_top.size == 0 and v_rest.size == 0:
                ax.axis("off")
                continue
            ax.hist(v_rest, bins=14, color="#cccccc", alpha=0.8, density=True, label="rest")
            ax.hist(v_top, bins=14, color="#1f77b4", alpha=0.65, density=True, label=f"top {int(top_fraction*100)}%")
            ax.set_ylabel("density")
        ax.set_title(name, fontsize=9)
        ax.grid(True, alpha=0.15)

    # hide unused axes
    for j in range(d, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    # single legend
    handles, leglabels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, leglabels, loc="upper right", frameon=False)
    fig.suptitle(f"Top-vs-rest parameter marginals (colored by {ylabel})", y=1.01, fontsize=11)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_binned_performance(
    *,
    X: "np.ndarray",
    y: "np.ndarray",
    labels: Sequence[str],
    out_path: Path,
    bins: int,
    cat_levels: Mapping[str, List[str]],
    ylabel: str,
) -> None:
    """
    For numeric params: bin along parameter axis and plot mean +/- 95% CI per bin.
    For categoricals: bar plot of mean +/- CI per category.
    """
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore

    bins = int(max(4, min(30, bins)))
    m = np.isfinite(y)
    if not bool(np.any(m)):
        return

    d = int(X.shape[1])
    ncols = 3
    nrows = int(np.ceil(d / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.7 * nrows), squeeze=False)

    def mean_ci(vals: "np.ndarray") -> Tuple[float, float]:
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return float("nan"), float("nan")
        mu = float(np.mean(vals))
        # normal approx CI on mean
        se = float(np.std(vals, ddof=1) / np.sqrt(max(1, int(vals.size)))) if vals.size >= 2 else 0.0
        return mu, 1.96 * se

    for idx, name in enumerate(labels):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]
        col = X[:, idx]

        if name in cat_levels:
            levs = cat_levels[name]
            mus = []
            cis = []
            for li in range(len(levs)):
                vals = y[m & (np.round(col) == li)]
                mu, ci = mean_ci(vals)
                mus.append(mu)
                cis.append(ci)
            xs = np.arange(len(levs), dtype=float)
            ax.bar(xs, mus, yerr=cis, color="#1f77b4", alpha=0.85)
            ax.set_xticks(xs)
            ax.set_xticklabels(levs, fontsize=8)
            ax.set_ylabel(ylabel)
        else:
            mm = m & np.isfinite(col)
            if int(np.sum(mm)) < 4:
                ax.axis("off")
                continue
            x = col[mm]
            yy = y[mm]
            qs = np.quantile(x, np.linspace(0, 1, bins + 1))
            # unique edges to avoid empty bins
            edges = np.unique(qs)
            if edges.size < 3:
                ax.axis("off")
                continue
            centers = 0.5 * (edges[:-1] + edges[1:])
            mus = []
            cis = []
            for a, b in zip(edges[:-1], edges[1:]):
                in_bin = (x >= a) & (x <= b) if b == edges[-1] else (x >= a) & (x < b)
                mu, ci = mean_ci(yy[in_bin])
                mus.append(mu)
                cis.append(ci)
            ax.errorbar(centers, mus, yerr=cis, fmt="-o", color="#1f77b4", markersize=3, linewidth=1.2)
            ax.set_ylabel(ylabel)
        ax.set_title(name, fontsize=9)
        ax.grid(True, alpha=0.15)

    for j in range(d, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(f"Binned performance (mean ± 95% CI) vs parameter", y=1.01, fontsize=11)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_rank_correlation(
    *,
    X: "np.ndarray",
    y: "np.ndarray",
    labels: Sequence[str],
    out_path: Path,
) -> List[Tuple[str, float]]:
    """
    Spearman correlation per parameter (on encoded X). Returns sorted list.
    """
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore

    corrs: List[Tuple[str, float]] = []
    for j, name in enumerate(labels):
        corrs.append((name, float(_spearmanr(X[:, j], y))))
    corrs2 = [(n, c) for (n, c) in corrs if np.isfinite(c)]
    corrs2.sort(key=lambda t: abs(t[1]), reverse=True)

    if not corrs2:
        return []

    names = [n for n, _ in corrs2]
    vals = [c for _, c in corrs2]
    fig = plt.figure(figsize=(max(6.0, 0.22 * len(vals)), 3.8))
    ax = fig.add_subplot(1, 1, 1)
    xs = np.arange(len(vals), dtype=float)
    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in vals]
    ax.bar(xs, vals, color=colors, alpha=0.9)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Parameter–objective rank correlation (Spearman)")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return corrs2


def _save_interaction_slices(
    *,
    X: "np.ndarray",
    y: "np.ndarray",
    labels: Sequence[str],
    out_path: Path,
    categorical_param: str,
    cat_levels: Mapping[str, List[str]],
    max_numeric: int = 6,
) -> bool:
    """
    Small multiples: objective vs each numeric param, split by a categorical param (2–3 panels).
    """
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore

    if categorical_param not in labels or categorical_param not in cat_levels:
        return False

    cat_idx = int(labels.index(categorical_param))
    levs = list(cat_levels[categorical_param])
    if len(levs) < 2:
        return False

    # pick numeric parameters (non-categorical)
    numeric_idxs = [i for i, n in enumerate(labels) if n != categorical_param and n not in cat_levels]
    numeric_idxs = numeric_idxs[: int(max_numeric)]
    if not numeric_idxs:
        return False

    ncols = len(levs)
    nrows = len(numeric_idxs)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.2 * nrows), squeeze=False)

    vmin = float(np.nanmin(y)) if np.any(np.isfinite(y)) else 0.0
    vmax = float(np.nanmax(y)) if np.any(np.isfinite(y)) else 1.0

    for r, j in enumerate(numeric_idxs):
        for c, lev in enumerate(levs):
            ax = axes[r][c]
            cat_mask = np.isfinite(y) & (np.round(X[:, cat_idx]) == c)
            ax.scatter(X[:, j][cat_mask], y[cat_mask], s=18, alpha=0.75, color="#1f77b4")
            if r == 0:
                ax.set_title(f"{categorical_param}={lev}", fontsize=9)
            if c == 0:
                ax.set_ylabel(str(labels[j]), fontsize=8)
            ax.grid(True, alpha=0.15)
            if r == nrows - 1:
                ax.set_xlabel("encoded value", fontsize=8)

    fig.suptitle("Objective vs parameter, split by categorical setting", y=1.01, fontsize=11)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def _fit_surrogate_rf(X: "np.ndarray", y: "np.ndarray"):
    """
    Fit a simple surrogate (RandomForestRegressor). Returns model or None if sklearn unavailable.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor  # type: ignore
    except Exception:
        return None
    import numpy as np  # type: ignore

    m = np.isfinite(y)
    if int(np.sum(m)) < 5:
        return None
    X2 = np.asarray(X[m], dtype=float)
    y2 = np.asarray(y[m], dtype=float)
    # simple imputation of NaNs with column median
    for j in range(X2.shape[1]):
        col = X2[:, j]
        med = np.nanmedian(col)
        if not np.isfinite(med):
            med = 0.0
        col[np.isnan(col)] = med
        X2[:, j] = col
    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=0,
        min_samples_leaf=2,
        n_jobs=-1,
    )
    rf.fit(X2, y2)
    return rf


def _partial_dependence_1d(
    model,
    X: "np.ndarray",
    feature_idx: int,
    grid: "np.ndarray",
) -> "np.ndarray":
    import numpy as np  # type: ignore

    Xb = np.asarray(X, dtype=float)
    # impute NaNs with column median
    for j in range(Xb.shape[1]):
        col = Xb[:, j]
        med = np.nanmedian(col)
        if not np.isfinite(med):
            med = 0.0
        col[np.isnan(col)] = med
        Xb[:, j] = col

    out = np.zeros_like(grid, dtype=float)
    for i, g in enumerate(grid.tolist()):
        Xtmp = Xb.copy()
        Xtmp[:, feature_idx] = float(g)
        out[i] = float(np.mean(model.predict(Xtmp)))
    return out


def _partial_dependence_2d(
    model,
    X: "np.ndarray",
    feature_i: int,
    grid_i: "np.ndarray",
    feature_j: int,
    grid_j: "np.ndarray",
) -> "np.ndarray":
    import numpy as np  # type: ignore

    Xb = np.asarray(X, dtype=float)
    for k in range(Xb.shape[1]):
        col = Xb[:, k]
        med = np.nanmedian(col)
        if not np.isfinite(med):
            med = 0.0
        col[np.isnan(col)] = med
        Xb[:, k] = col

    Z = np.zeros((grid_i.size, grid_j.size), dtype=float)
    for a, gi in enumerate(grid_i.tolist()):
        for b, gj in enumerate(grid_j.tolist()):
            Xtmp = Xb.copy()
            Xtmp[:, feature_i] = float(gi)
            Xtmp[:, feature_j] = float(gj)
            Z[a, b] = float(np.mean(model.predict(Xtmp)))
    return Z


def _save_partial_dependence(
    *,
    X: "np.ndarray",
    y: "np.ndarray",
    labels: Sequence[str],
    out1d: Path,
    out2d: Path,
    top_corrs: Sequence[Tuple[str, float]],
    grid_points: int = 30,
) -> None:
    """
    Fit a surrogate and plot:
      - 1D partial dependence for top correlated parameters
      - 2D PD heatmaps for a couple of common interaction pairs
    """
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore

    model = _fit_surrogate_rf(X, y)
    if model is None:
        return

    # Choose up to 6 parameters by |spearman|
    chosen = [n for (n, _c) in top_corrs[:6] if n in labels]
    if not chosen:
        chosen = list(labels[: min(6, len(labels))])

    # --- 1D
    n = len(chosen)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.6 * nrows), squeeze=False)
    for idx, name in enumerate(chosen):
        ax = axes[idx // ncols][idx % ncols]
        j = int(labels.index(name))
        col = X[:, j]
        mm = np.isfinite(col)
        if int(np.sum(mm)) < 3:
            ax.axis("off")
            continue
        lo = float(np.nanquantile(col, 0.05))
        hi = float(np.nanquantile(col, 0.95))
        grid = np.linspace(lo, hi, int(grid_points), dtype=float)
        pd = _partial_dependence_1d(model, X, j, grid)
        ax.plot(grid, pd, color="#1f77b4", linewidth=1.8)
        ax.set_title(name, fontsize=9)
        ax.set_ylabel("predicted objective")
        ax.grid(True, alpha=0.15)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("Partial dependence (1D) from random-forest surrogate", y=1.01, fontsize=11)
    fig.tight_layout()
    fig.savefig(str(out1d), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- 2D: try a couple of intuitive pairs if present, else top pairs by correlation
    candidate_pairs: List[Tuple[str, str]] = []
    for a, b in [
        ("graph.topo_k", "graph.k"),
        ("graph.topo_k", "graph.topo_pca_dim"),
        ("detector.topo_percentile", "detector.topo_cov_shrinkage"),
    ]:
        if a in labels and b in labels:
            candidate_pairs.append((a, b))

    if not candidate_pairs:
        names = [n for n, _c in top_corrs if n in labels][:6]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                candidate_pairs.append((names[i], names[j]))
                if len(candidate_pairs) >= 2:
                    break
            if len(candidate_pairs) >= 2:
                break

    candidate_pairs = candidate_pairs[:2]
    if not candidate_pairs:
        return

    fig, axes = plt.subplots(1, len(candidate_pairs), figsize=(5.0 * len(candidate_pairs), 3.8), squeeze=False)
    for pi, (a, b) in enumerate(candidate_pairs):
        ax = axes[0][pi]
        ia = int(labels.index(a))
        ib = int(labels.index(b))
        xa = X[:, ia]
        xb = X[:, ib]
        ga = np.linspace(float(np.nanquantile(xa, 0.05)), float(np.nanquantile(xa, 0.95)), 35)
        gb = np.linspace(float(np.nanquantile(xb, 0.05)), float(np.nanquantile(xb, 0.95)), 35)
        Z = _partial_dependence_2d(model, X, ia, ga, ib, gb)
        im = ax.imshow(
            Z,
            origin="lower",
            aspect="auto",
            extent=[float(gb.min()), float(gb.max()), float(ga.min()), float(ga.max())],
            cmap="viridis",
        )
        ax.set_xlabel(b)
        ax.set_ylabel(a)
        ax.set_title("2D partial dependence", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, label="predicted objective")
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(str(out2d), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_topk_table(
    *,
    rows: Sequence[Mapping[str, Any]],
    X: "np.ndarray",
    y: "np.ndarray",
    labels: Sequence[str],
    out_csv: Path,
    out_png: Path,
    top_k: int,
    cat_levels: Mapping[str, List[str]],
    ylabel: str,
) -> None:
    """
    Write a compact top-K table (CSV + rendered PNG) showing params + objective.
    Uses the original `params` dicts for readability.
    """
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore
    import csv

    # Sort by y descending
    idxs = [i for i in range(len(rows)) if np.isfinite(y[i])]
    idxs.sort(key=lambda i: float(y[i]), reverse=True)
    idxs = idxs[: int(max(1, min(len(idxs), top_k)))]

    # Determine columns: objective + all params (as in labels order)
    header = ["trial_id", ylabel] + list(labels)
    table_rows: List[List[str]] = []
    for i in idxs:
        r = rows[i]
        trial_id = str(r.get("trial_id", ""))
        params = r.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        row = [trial_id, f"{float(y[i]):.6g}"]
        for name in labels:
            v = params.get(name, "")
            if isinstance(v, float):
                row.append(f"{v:.6g}")
            else:
                row.append(str(v))
        table_rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(table_rows)

    # Consensus row: median/IQR for numerics; mode for categoricals
    consensus: List[str] = ["", "topK_median"]
    for j, name in enumerate(labels):
        col = X[idxs, j]
        if name in cat_levels:
            # mode of rounded indices
            vals = [int(round(v)) for v in col.tolist() if np.isfinite(v)]
            if not vals:
                consensus.append("")
            else:
                # mode
                best = max(set(vals), key=vals.count)
                consensus.append(cat_levels[name][best])
        else:
            vals = col[np.isfinite(col)]
            if vals.size == 0:
                consensus.append("")
            else:
                q50 = float(np.quantile(vals, 0.5))
                q25 = float(np.quantile(vals, 0.25))
                q75 = float(np.quantile(vals, 0.75))
                consensus.append(f"{q50:.3g} (IQR {q25:.3g}–{q75:.3g})")

    # Render PNG table (keep it compact: show only first ~8 params if too wide)
    max_cols = 10  # trial_id + objective + 8 params
    cols_to_show = header[: max_cols]
    # always include topo_preprocess if present
    if "graph.topo_preprocess" in header and "graph.topo_preprocess" not in cols_to_show:
        cols_to_show = cols_to_show[:-1] + ["graph.topo_preprocess"]

    col_idxs = [header.index(c) for c in cols_to_show]
    render_rows = [[row[k] for k in col_idxs] for row in table_rows]
    render_rows.append([consensus[k] for k in col_idxs])

    fig = plt.figure(figsize=(min(14.0, 0.9 * len(cols_to_show)), 0.6 + 0.35 * (len(render_rows) + 1)))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    tbl = ax.table(cellText=render_rows, colLabels=cols_to_show, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.0, 1.2)
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Plot optimiser parameter-space visualisations from history.jsonl")
    p.add_argument("--history", required=True, help="Path to history.jsonl")
    p.add_argument("--outdir", required=True, help="Directory to write figures into")
    p.add_argument(
        "--space",
        default=None,
        help="Optional search-space spec (same format as optimiser --space) for correct log/categorical encoding.",
    )
    p.add_argument("--field", default="objective_value", help="Which field to plot/optimize (default: objective_value)")
    p.add_argument("--ylabel", default="objective", help="Y-axis label (default: objective)")
    p.add_argument("--max-dims", type=int, default=6, help="Max dims in scatter matrix (default: 6)")
    p.add_argument("--top-fraction", type=float, default=0.2, help="Top fraction used for top-vs-rest plots (default: 0.2)")
    p.add_argument("--bins", type=int, default=10, help="Bin count for binned performance plots (default: 10)")
    p.add_argument("--top-k", type=int, default=10, help="Top-K table size (default: 10)")
    p.add_argument(
        "--category-param",
        default="graph.topo_preprocess",
        help='Categorical param for group boxplot (default: "graph.topo_preprocess")',
    )
    args = p.parse_args(argv)

    # Fail fast with a clear message if plotting deps are missing.
    try:
        import numpy as _np  # noqa: F401  # type: ignore
        import matplotlib.pyplot as _plt  # noqa: F401  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Missing plotting dependencies (numpy/matplotlib). "
            "Please run this script in the same environment you use for the repo, e.g.\n\n"
            "  pip install -r requirements.txt\n"
            "  python -m optimiser.plot_history --history <path> --outdir <dir>\n\n"
            f"Original import error: {type(e).__name__}: {e}"
        )

    history_path = Path(str(args.history)).resolve()
    outdir = Path(str(args.outdir)).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    space_path = None if args.space in (None, "", "none") else Path(str(args.space)).resolve()

    rows_all = _load_jsonl(history_path)
    rows = [r for r in rows_all if str(r.get("status", "")).lower() == "success"]
    if not rows:
        raise SystemExit("No successful trials found in history.jsonl")

    trial_id, y = _extract_objective(rows, field=str(args.field))
    param_order = _infer_param_order(rows)
    X, labels, cat_levels = _encode_params(rows=rows, param_order=param_order, space_spec_path=space_path)

    _save_best_so_far(trial_id, y, outdir / "best_so_far.png", ylabel=str(args.ylabel))
    _save_scatter_matrix(X, y, labels, outdir / "scatter_matrix.png", max_dims=int(args.max_dims))
    _save_parallel_coordinates(X, y, labels, outdir / "parallel_coordinates.png", cat_levels=cat_levels)
    _save_pca_embedding(X, y, outdir / "pca_embedding.png")
    _save_category_boxplot(
        rows=rows,
        category_param=str(args.category_param),
        y=y,
        out_path=outdir / "category_boxplot.png",
        ylabel=str(args.ylabel),
    )

    # New, more interpretable "what works well" plots
    _save_top_vs_rest_marginals(
        X=X,
        y=y,
        labels=labels,
        out_path=outdir / "top_vs_rest_marginals.png",
        top_fraction=float(args.top_fraction),
        cat_levels=cat_levels,
        ylabel=str(args.ylabel),
    )
    _save_binned_performance(
        X=X,
        y=y,
        labels=labels,
        out_path=outdir / "binned_performance.png",
        bins=int(args.bins),
        cat_levels=cat_levels,
        ylabel=str(args.ylabel),
    )
    corrs = _save_rank_correlation(X=X, y=y, labels=labels, out_path=outdir / "rank_correlation.png")
    _save_interaction_slices(
        X=X,
        y=y,
        labels=labels,
        out_path=outdir / "interaction_slices.png",
        categorical_param=str(args.category_param),
        cat_levels=cat_levels,
    )
    _save_partial_dependence(
        X=X,
        y=y,
        labels=labels,
        out1d=outdir / "partial_dependence_1d.png",
        out2d=outdir / "partial_dependence_2d.png",
        top_corrs=corrs,
    )
    _write_topk_table(
        rows=rows,
        X=X,
        y=y,
        labels=labels,
        out_csv=outdir / "topk_table.csv",
        out_png=outdir / "topk_table.png",
        top_k=int(args.top_k),
        cat_levels=cat_levels,
        ylabel=str(args.ylabel),
    )

    print(f"Wrote figures to: {outdir}")


if __name__ == "__main__":
    main()

