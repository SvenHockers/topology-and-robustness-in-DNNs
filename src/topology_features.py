"""
Topology (persistent homology) feature extraction utilities.

This module is intentionally optional: it requires a PH backend (recommended: ripser).
If the dependency is not installed, importing is fine, but calling the PH functions
will raise an informative error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA


def _require_ripser():
    try:
        from ripser import ripser  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Topology scoring requires a persistent-homology backend.\n"
            "Install one of:\n"
            "  - ripser (recommended for Vietoris–Rips PH on point clouds)\n"
            "  - gudhi (more general; alpha complexes, cubical, etc.)\n\n"
            "For this repo, add to your environment:\n"
            "  pip install ripser\n"
        ) from e
    return ripser


@dataclass
class TopologyConfig:
    """
    Configuration for local persistent homology feature extraction.

    Notes:
    - We compute PH on a *local neighborhood point cloud* around each query point.
    - The detector score is derived from PH features, not from distances directly.
    """

    neighborhood_k: int = 50
    maxdim: int = 1  # compute H0..H_maxdim
    metric: str = "euclidean"
    # ripser supports an optional distance_matrix mode; we use point-cloud mode by default.
    # thresh limits the filtration radius; leaving it None lets ripser choose.
    thresh: Optional[float] = None
    # small lifetimes are often numerical noise; ignore below this for counting/entropy
    min_persistence: float = 1e-6
    # Optional preprocessing of each local neighborhood point cloud before PH.
    # In high-dimensional settings (d >> n_points), projecting to a low-dimensional
    # subspace often yields more informative local topology than running VR PH in ambient space.
    preprocess: str = "none"  # "none" or "pca"
    pca_dim: int = 10


def _diagram_lifetimes(diagram: np.ndarray) -> np.ndarray:
    """
    Convert a persistence diagram array (birth, death) -> lifetimes (death-birth),
    dropping infinite deaths.
    """
    if diagram.size == 0:
        return np.zeros((0,), dtype=float)
    births = diagram[:, 0]
    deaths = diagram[:, 1]
    finite = np.isfinite(deaths)
    lifetimes = deaths[finite] - births[finite]
    lifetimes = lifetimes[np.isfinite(lifetimes)]
    return lifetimes.astype(float, copy=False)


def persistence_summary_features(
    diagrams: Sequence[np.ndarray],
    min_persistence: float = 1e-6,
) -> Dict[str, float]:
    """
    Compute lightweight, topology-driven summary features from persistence diagrams.

    For each homology dimension d, we compute:
    - count: number of features with lifetime >= min_persistence
    - total_persistence: sum(lifetime)
    - max_persistence: max(lifetime)
    - entropy: Shannon entropy of normalized lifetimes (0 if <2 features)
    - l2_persistence: sqrt(sum(lifetime^2))
    """
    feats: Dict[str, float] = {}
    for dim, diag in enumerate(diagrams):
        lifetimes = _diagram_lifetimes(diag)
        lifetimes = lifetimes[lifetimes >= float(min_persistence)]
        key = f"h{dim}"
        if lifetimes.size == 0:
            feats[f"topo_{key}_count"] = 0.0
            feats[f"topo_{key}_total_persistence"] = 0.0
            feats[f"topo_{key}_max_persistence"] = 0.0
            feats[f"topo_{key}_entropy"] = 0.0
            feats[f"topo_{key}_l2_persistence"] = 0.0
            continue

        total = float(lifetimes.sum())
        feats[f"topo_{key}_count"] = float(lifetimes.size)
        feats[f"topo_{key}_total_persistence"] = total
        feats[f"topo_{key}_max_persistence"] = float(lifetimes.max())
        feats[f"topo_{key}_l2_persistence"] = float(np.sqrt(np.sum(lifetimes ** 2)))

        if lifetimes.size < 2 or total <= 0:
            feats[f"topo_{key}_entropy"] = 0.0
        else:
            p = lifetimes / total
            p = np.clip(p, 1e-12, 1.0)
            feats[f"topo_{key}_entropy"] = float(-np.sum(p * np.log(p)))

    return feats


def local_persistence_features(
    point_cloud: np.ndarray,
    topo_cfg: TopologyConfig,
) -> Dict[str, float]:
    """
    Compute PH summary features for a local neighborhood point cloud.

    Args:
        point_cloud: Array of shape (n_points, d)
        topo_cfg: TopologyConfig
    """
    ripser = _require_ripser()

    # Optional: local PCA compression before PH (helps when d >> n_points).
    if str(topo_cfg.preprocess).lower() == "pca":
        X = np.asarray(point_cloud, dtype=float)
        n, d = X.shape
        # PCA rank is at most min(n-1, d). Keep dim <= that.
        r_max = max(1, min(n - 1, d))
        r = int(min(max(1, int(topo_cfg.pca_dim)), r_max))
        if r < d:
            X = PCA(n_components=r).fit_transform(X)
        point_cloud = X

    # ripser's Cython backend expects thresh to be a real number; passing None can error.
    # If thresh is unset, omit it (ripser will use its internal default).
    kwargs = {
        "maxdim": int(topo_cfg.maxdim),
        "metric": str(topo_cfg.metric),
    }
    if topo_cfg.thresh is not None:
        kwargs["thresh"] = float(topo_cfg.thresh)

    out = ripser(point_cloud, **kwargs)
    diagrams = out.get("dgms", [])
    return persistence_summary_features(diagrams, min_persistence=topo_cfg.min_persistence)


