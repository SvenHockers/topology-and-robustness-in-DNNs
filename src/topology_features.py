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
    # Filtration mode:
    # - 'standard': Vietoris–Rips PH on point cloud (ripser point-cloud mode)
    # - 'query_anchored': query-anchored distance-matrix filtration (ripser distance-matrix mode)
    filtration: str = "standard"
    # Query-anchored filtration parameters (used only when filtration == 'query_anchored').
    # Interpretation:
    # - query point is assumed to be point_cloud[0]
    # - neighbor-neighbor distances are rescaled by a function of their distances to the query
    query_lambda: float = 1.0
    query_gamma: float = 1.0


def _pairwise_distances(X: np.ndarray) -> np.ndarray:
    """Compute full pairwise Euclidean distance matrix for X (n,d) in O(n^2 d)."""
    X = np.asarray(X, dtype=float)
    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    s = np.sum(X * X, axis=1, keepdims=True)  # (n,1)
    D2 = s + s.T - 2.0 * (X @ X.T)
    D2 = np.maximum(D2, 0.0)
    return np.sqrt(D2, dtype=float)


def _query_anchored_distance_matrix(point_cloud: np.ndarray, *, query_lambda: float, query_gamma: float) -> np.ndarray:
    """
    Build a query-anchored distance matrix D' for ripser(distance_matrix=True).

    Assumptions:
    - query point is point_cloud[0]
    - D'[0,i] uses the true Euclidean distance (so the query-to-neighbor geometry is preserved)
    - D'[i,j] for i,j>0 is rescaled based on distances to the query:
        D'[i,j] = D[i,j] * ((1 + λ * d(q,i)) + (1 + λ * d(q,j))) / 2) ^ γ

    This makes the filtration ordering depend strongly on the query point's position (through d(q,i)).
    Note: D' need not be a metric; ripser accepts general symmetric distance matrices.
    """
    X = np.asarray(point_cloud, dtype=float)
    D = _pairwise_distances(X)
    n = int(D.shape[0])
    if n <= 1:
        return D

    lam = float(query_lambda)
    gam = float(query_gamma)
    dq = np.asarray(D[0, :], dtype=float)  # (n,)
    w = 1.0 + lam * dq
    # Rescale neighbor-neighbor distances; keep query edges unmodified.
    scale = ((w.reshape(-1, 1) + w.reshape(1, -1)) / 2.0) ** gam
    Dp = D * scale
    Dp[0, :] = D[0, :]
    Dp[:, 0] = D[:, 0]
    np.fill_diagonal(Dp, 0.0)
    # Ensure symmetry
    Dp = (Dp + Dp.T) / 2.0
    return Dp


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
    return_diagrams: bool = False,
) -> Dict[str, float] | Tuple[Dict[str, float], Sequence[np.ndarray]]:
    """
    Compute PH summary features for a local neighborhood point cloud.

    Args:
        point_cloud: Array of shape (n_points, d)
        topo_cfg: TopologyConfig
        return_diagrams: If True, also return raw persistence diagrams
        
    Returns:
        If return_diagrams=False: Dict of summary features
        If return_diagrams=True: Tuple of (features_dict, diagrams_list)
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

    filt = str(getattr(topo_cfg, "filtration", "standard")).strip().lower()
    if filt == "standard":
        out = ripser(point_cloud, **kwargs)
    elif filt == "query_anchored":
        # Use ripser's distance-matrix mode; metric is not used in this mode.
        Dp = _query_anchored_distance_matrix(
            point_cloud,
            query_lambda=float(getattr(topo_cfg, "query_lambda", 1.0)),
            query_gamma=float(getattr(topo_cfg, "query_gamma", 1.0)),
        )
        # Remove 'metric' for distance_matrix mode to avoid backend confusion.
        kwargs_dm = dict(kwargs)
        kwargs_dm.pop("metric", None)
        out = ripser(Dp, distance_matrix=True, **kwargs_dm)
    else:
        raise ValueError(f"Unknown topo_cfg.filtration={topo_cfg.filtration!r}. Expected 'standard' or 'query_anchored'.")

    diagrams = out.get("dgms", [])
    features = persistence_summary_features(diagrams, min_persistence=topo_cfg.min_persistence)
    
    if return_diagrams:
        return features, diagrams
    return features


