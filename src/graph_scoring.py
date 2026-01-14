"""
Graph construction and manifold conformity score computation.
"""

import numpy as np
import torch
from scipy.spatial.distance import pdist
from scipy.sparse.linalg import eigs
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from typing import Tuple, Optional, Dict, Sequence, List, Any
from .utils import GraphConfig
from .topology_features import TopologyConfig, local_persistence_features


def build_knn_graph(
    Z_train: np.ndarray,
    k: int = 10,
    sigma: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build k-NN graph with Gaussian edge weights.
    
    Args:
        Z_train: Training points array of shape (n_train, d)
        k: Number of nearest neighbors
        sigma: Scale parameter for Gaussian weights. If None, uses median distance heuristic.
        
    Returns:
        Tuple of (W, D, distances) where:
        - W: Weight matrix (adjacency with weights)
        - D: Degree matrix (diagonal matrix with row sums)
        - distances: Pairwise distances for k-NN
    """
    n_train = Z_train.shape[0]
    
    # Find k nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(Z_train)
    distances, indices = nbrs.kneighbors(Z_train)
    
    # Use median distance as sigma if not provided
    if sigma is None:
        sigma = np.median(distances[distances > 0])
    
    # Initialize weight matrix
    W = np.zeros((n_train, n_train))
    
    # Fill in weights for k-NN pairs
    for i in range(n_train):
        for j_idx, neighbor_idx in enumerate(indices[i]):
            if neighbor_idx != i:  # Exclude self-connections
                dist = distances[i, j_idx]
                weight = np.exp(-dist ** 2 / (2 * sigma ** 2))
                W[i, neighbor_idx] = weight
    
    # Make symmetric (undirected graph)
    W = (W + W.T) / 2
    
    # Compute degree matrix (diagonal matrix with row sums)
    D = np.diag(W.sum(axis=1))
    
    return W, D, distances


def compute_degree_score(
    z: np.ndarray,
    Z_train: np.ndarray,
    k: int = 10,
    sigma: Optional[float] = None
) -> float:
    """
    Compute degree-based manifold conformity score for a new point.
    
    Higher degree = more connected = more on-manifold.
    We return negative degree as score (higher score = more suspicious/off-manifold).
    
    Args:
        z: New point of shape (d,)
        Z_train: Training points array of shape (n_train, d)
        k: Number of nearest neighbors
        sigma: Scale parameter for Gaussian weights
        
    Returns:
        Degree score (negative degree, so higher = more suspicious)
    """
    # Find k nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(Z_train)
    distances, indices = nbrs.kneighbors(z.reshape(1, -1))
    distances = distances[0]
    indices = indices[0]
    
    # Use median distance as sigma if not provided
    if sigma is None:
        # Compute median distance among training points
        train_distances = pdist(Z_train)
        sigma = np.median(train_distances)
    
    # Compute weights to neighbors
    weights = np.exp(-distances ** 2 / (2 * sigma ** 2))
    degree = weights.sum()
    
    # Return negative degree (higher score = more suspicious)
    return -degree


def compute_laplacian_smoothness_score(
    z: np.ndarray,
    f_z: float,
    Z_train: np.ndarray,
    f_train: np.ndarray,
    k: int = 10,
    sigma: Optional[float] = None
) -> float:
    """
    Compute Laplacian smoothness / Dirichlet energy increment score.
    
    Measures how much adding this point would increase the Dirichlet energy.
    Higher score = more suspicious (less smooth).
    
    Args:
        z: New point of shape (d,)
        f_z: Model output (e.g., probability of class 1) for z
        Z_train: Training points array of shape (n_train, d)
        f_train: Model outputs for training points, shape (n_train,)
        k: Number of nearest neighbors
        sigma: Scale parameter for Gaussian weights
        
    Returns:
        Laplacian smoothness score (higher = more suspicious)
    """
    # Find k nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(Z_train)
    distances, indices = nbrs.kneighbors(z.reshape(1, -1))
    distances = distances[0]
    indices = indices[0]
    
    # Use median distance as sigma if not provided
    if sigma is None:
        train_distances = pdist(Z_train)
        sigma = np.median(train_distances)
    
    # Compute weights to neighbors
    weights = np.exp(-distances ** 2 / (2 * sigma ** 2))
    
    # Compute Dirichlet energy increment
    delta_E = np.sum(weights * (f_z - f_train[indices]) ** 2)
    
    return delta_E


def compute_diffusion_embedding(
    W: np.ndarray,
    n_components: int = 10,
    alpha: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute diffusion map embedding of the training data.
    
    Args:
        W: Weight matrix
        n_components: Number of diffusion map components
        alpha: Diffusion parameter (0 = standard, 1 = Laplace-Beltrami)
        
    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    # Compute degree matrix
    D = np.diag(W.sum(axis=1))
    
    # Normalize weights
    D_alpha_inv = np.diag(1.0 / (np.diag(D) ** alpha + 1e-10))
    W_normalized = D_alpha_inv @ W @ D_alpha_inv
    
    # Compute transition matrix (row-normalized)
    D_normalized = np.diag(W_normalized.sum(axis=1))
    D_normalized_inv = np.diag(1.0 / (np.diag(D_normalized) + 1e-10))
    P = D_normalized_inv @ W_normalized
    
    # Compute eigenvectors of transition matrix
    # Use largest eigenvalues (excluding the first one which is always 1)
    eigenvalues, eigenvectors = eigs(P.T, k=n_components + 1, which='LR')
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    
    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Return eigenvalues and eigenvectors (skip the first trivial one)
    return eigenvalues[1:n_components+1], eigenvectors[:, 1:n_components+1]


def embed_new_point_diffusion(
    z: np.ndarray,
    Z_train: np.ndarray,
    embedding_eigenvectors: np.ndarray,
    W_train: np.ndarray,
    k: int = 10,
    sigma: Optional[float] = None
) -> np.ndarray:
    """
    Embed a new point using Nyström extension for diffusion maps.
    
    Args:
        z: New point to embed
        Z_train: Training points
        embedding_eigenvectors: Eigenvectors from diffusion embedding
        W_train: Training weight matrix
        k: Number of nearest neighbors for Nyström extension
        sigma: Scale parameter
        
    Returns:
        Embedded point in diffusion space
    """
    # Find k nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(Z_train)
    distances, indices = nbrs.kneighbors(z.reshape(1, -1))
    distances = distances[0]
    indices = indices[0]
    
    if sigma is None:
        train_distances = pdist(Z_train)
        sigma = np.median(train_distances)
    
    # Compute weights
    weights = np.exp(-distances ** 2 / (2 * sigma ** 2))
    weights = weights / (weights.sum() + 1e-10)
    
    # Nyström extension: weighted average of neighbor embeddings
    z_embedding = weights @ embedding_eigenvectors[indices]
    
    return z_embedding


def compute_diffusion_distance_score(
    z: np.ndarray,
    Z_train: np.ndarray,
    embedding_eigenvectors: np.ndarray,
    W_train: np.ndarray,
    k: int = 10,
    sigma: Optional[float] = None
) -> float:
    """
    Compute diffusion distance-based score.
    
    Args:
        z: New point
        Z_train: Training points
        embedding_eigenvectors: Diffusion map eigenvectors
        W_train: Training weight matrix
        k: Number of neighbors
        sigma: Scale parameter
        
    Returns:
        Minimal diffusion distance to training points
    """
    # Embed new point
    z_embedding = embed_new_point_diffusion(
        z, Z_train, embedding_eigenvectors, W_train, k, sigma
    )
    
    # Compute distances to all training embeddings
    distances = np.linalg.norm(
        embedding_eigenvectors - z_embedding.reshape(1, -1),
        axis=1
    )
    
    # Return minimum distance (higher = more suspicious)
    return np.min(distances)


def compute_graph_scores(
    X_points: np.ndarray,
    model,
    Z_train: np.ndarray,
    f_train: np.ndarray,
    graph_params: GraphConfig,
    device: str = 'cpu',
    *,
    y_train: Optional[np.ndarray] = None,
    y_points: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute all graph-based manifold conformity scores for a batch of points.
    
    Args:
        X_points: Points to score, shape (n_points, input_dim)
        model: Trained PyTorch model
        Z_train: Training representations (in input or feature space)
        f_train: Training model outputs (probabilities or logits)
        graph_params: GraphConfig with hyperparameters
        device: Device for model inference
        
    Returns:
        Dictionary with score arrays:
        - 'degree': Degree scores
        - 'laplacian': Laplacian smoothness scores
        - 'diffusion': Diffusion distance scores (if enabled)
        - 'combined': Combined score (if requested)
    """
    from .models import get_model_logits, extract_features_batch
    
    n_points = X_points.shape[0]
    scores = {}
    
    # Extract representations if needed
    if graph_params.space == 'feature':
        # Keep query embedding space consistent with Z_train construction.
        # (Z_train is built upstream using cfg.graph.feature_layer.)
        layer = str(getattr(graph_params, "feature_layer", "penultimate"))
        Z_points = extract_features_batch(model, X_points, layer=layer, device=device)
    else:
        Z_points = X_points
    
    # Get model outputs for points
    #
    # For binary classification, we use P(class=1) as the scalar function f(x).
    # For multi-class, we use max softmax probability as a generic confidence scalar.
    # This scalar is only used by some graph scores (e.g., Laplacian smoothness);
    # topology features are computed purely from geometry in Z-space.
    logits = get_model_logits(model, X_points, device=device)
    probs = torch.softmax(torch.as_tensor(logits, dtype=torch.float32), dim=1).cpu().numpy()
    if probs.ndim != 2 or probs.shape[1] < 2:
        raise ValueError(f"Expected model outputs to be (N,C) with C>=2; got probs shape={probs.shape}")
    if probs.shape[1] == 2:
        f_points = probs[:, 1]  # Probability of class 1
    else:
        f_points = probs.max(axis=1)  # Confidence proxy

    # Predicted class labels (used by class-aware topology neighborhoods when enabled).
    y_pred_points = probs.argmax(axis=1).astype(int, copy=False)
    
    # Compute degree scores
    degree_scores = np.zeros(n_points)
    for i in range(n_points):
        degree_scores[i] = compute_degree_score(
            Z_points[i], Z_train, k=graph_params.k, sigma=graph_params.sigma
        )
    scores['degree'] = degree_scores
    
    # Compute Laplacian smoothness scores
    laplacian_scores = np.zeros(n_points)
    for i in range(n_points):
        laplacian_scores[i] = compute_laplacian_smoothness_score(
            Z_points[i], f_points[i], Z_train, f_train,
            k=graph_params.k, sigma=graph_params.sigma
        )
    scores['laplacian'] = laplacian_scores

    # Local tangent-space residual score (manifold membership test)
    # Intuition: if clean data lies on/near a low-dim manifold in Z-space, then
    # projecting onto the local tangent space should have small residual norm.
    # Backward-compatible defaults: if older GraphConfig objects are in memory (e.g. notebook
    # kernel not restarted), these attributes may be missing. We default to computing
    # tangent-based scores in that case.
    if getattr(graph_params, 'use_tangent', True):
        tangent_scores = np.zeros(n_points)
        tangent_z_scores = np.zeros(n_points)
        tangent_k = int(getattr(graph_params, 'tangent_k', 20))
        nbrs_tangent = NearestNeighbors(
            n_neighbors=min(tangent_k, len(Z_train)),
            metric='euclidean'
        ).fit(Z_train)

        # If tangent_dim is not provided, pick it adaptively from local PCA explained variance.
        # This is important in feature space where the manifold may not be 2D after embedding.
        tangent_dim = getattr(graph_params, 'tangent_dim', None)
        var_thr = float(getattr(graph_params, 'tangent_var_threshold', 0.9))
        dim_min = int(getattr(graph_params, 'tangent_dim_min', 2))
        dim_max = getattr(graph_params, 'tangent_dim_max', None)

        for i in range(n_points):
            _, idx = nbrs_tangent.kneighbors(Z_points[i].reshape(1, -1))
            neighborhood = Z_train[idx[0]]
            center = neighborhood.mean(axis=0, keepdims=True)
            Xc = neighborhood - center

            # Fit PCA on neighborhood
            # max components is limited by rank: min(n-1, d)
            max_components = max(1, min(Xc.shape[0] - 1, Xc.shape[1]))
            if tangent_dim is None:
                pca = PCA(n_components=max_components)
                pca.fit(Xc)
                cum = np.cumsum(pca.explained_variance_ratio_)
                r = int(np.searchsorted(cum, var_thr) + 1)
                r = max(dim_min, min(r, max_components))
                if dim_max is not None:
                    r = min(int(dim_max), r)
            else:
                r = max(1, min(int(tangent_dim), max_components))
                pca = PCA(n_components=r)
                pca.fit(Xc)

            # Use the first r components as the local tangent basis (rows)
            V = pca.components_[:r]  # (r, d)
            zc = (Z_points[i].reshape(1, -1) - center)
            z_proj = (zc @ V.T) @ V
            resid = zc - z_proj
            resid2 = float(np.sum(resid ** 2))
            tangent_scores[i] = resid2

            # Local normalization: z-score relative to neighborhood residual distribution.
            # This helps for small eps where absolute residuals can be tiny but still
            # atypical compared to local clean variation.
            Xc_proj = (Xc @ V.T) @ V
            Xc_resid = Xc - Xc_proj
            neigh_resid2 = np.sum(Xc_resid ** 2, axis=1)
            mu = float(neigh_resid2.mean())
            sigma = float(neigh_resid2.std() + 1e-12)
            tangent_z_scores[i] = (resid2 - mu) / sigma

        scores['tangent_residual'] = tangent_scores
        scores['tangent_residual_z'] = tangent_z_scores

        # Also provide a simple kNN radius score (density proxy)
        # Higher mean neighbor distance => lower local density => more suspicious.
        radius_scores = np.zeros(n_points)
        nbrs_radius = NearestNeighbors(
            n_neighbors=min(graph_params.k, len(Z_train)),
            metric='euclidean'
        ).fit(Z_train)
        dists, _ = nbrs_radius.kneighbors(Z_points)
        # exclude the first distance if it is 0 due to self-match (shouldn't happen for new points,
        # but keep robust)
        radius_scores = dists[:, 1:].mean(axis=1) if dists.shape[1] > 1 else dists[:, 0]
        scores['knn_radius'] = radius_scores

    # --- Topology features (persistent homology on local neighborhoods) ---
    if getattr(graph_params, 'use_topology', False):
        topo_k = int(getattr(graph_params, 'topo_k', 50))
        topo_k_list = getattr(graph_params, "topo_k_list", None)
        if topo_k_list is None:
            k_list: List[int] = [topo_k]
            prefix_keys = False  # legacy keys
        else:
            k_list = [int(k) for k in topo_k_list if int(k) > 0]
            if len(k_list) == 0:
                raise ValueError("graph.topo_k_list was provided but empty/invalid.")
            prefix_keys = True

        max_k = int(max(k_list))
        topo_cfg = TopologyConfig(
            neighborhood_k=max_k,
            maxdim=int(getattr(graph_params, 'topo_maxdim', 1)),
            metric=str(getattr(graph_params, 'topo_metric', 'euclidean')),
            thresh=getattr(graph_params, 'topo_thresh', None),
            min_persistence=float(getattr(graph_params, 'topo_min_persistence', 1e-6)),
            preprocess=str(getattr(graph_params, 'topo_preprocess', 'none')),
            pca_dim=int(getattr(graph_params, 'topo_pca_dim', 10)),
        )

        neighbor_mode = str(getattr(graph_params, "topo_neighbor_mode", "global")).strip().lower()
        metric_norm = str(getattr(graph_params, "topo_metric_normalization", "none")).strip().lower()
        whiten_ridge = float(getattr(graph_params, "topo_whiten_ridge", 1e-3))

        # Build neighbor indices for local neighborhoods in the scoring space.
        # Default: global kNN (legacy).
        global_nbrs = NearestNeighbors(
            n_neighbors=min(max_k, len(Z_train)),
            metric=topo_cfg.metric,
        ).fit(Z_train)

        # Optional: class-restricted kNN indices (built once per class for efficiency).
        class_nbrs: Dict[int, Any] = {}
        class_Z: Dict[int, np.ndarray] = {}
        if neighbor_mode in {"class_pred", "class_true"}:
            if y_train is None:
                raise ValueError(
                    "graph.topo_neighbor_mode requires y_train to restrict neighbors by class. "
                    "Pass y_train to compute_graph_scores(..., y_train=...) or set topo_neighbor_mode='global'."
                )
            y_train_i = np.asarray(y_train, dtype=int).ravel()
            if y_train_i.shape[0] != Z_train.shape[0]:
                raise ValueError(f"Z_train/y_train length mismatch: {Z_train.shape[0]} vs {y_train_i.shape[0]}")
            for c in np.unique(y_train_i).tolist():
                c = int(c)
                mask = (y_train_i == c)
                Zc = np.asarray(Z_train)[mask]
                if Zc.shape[0] < 2:
                    continue
                class_Z[c] = Zc
                class_nbrs[c] = NearestNeighbors(
                    n_neighbors=min(max_k, len(Zc)),
                    metric=topo_cfg.metric,
                ).fit(Zc)

        def _normalize_cloud(cloud: np.ndarray, *, dists: Optional[np.ndarray]) -> np.ndarray:
            """
            Apply local metric conditioning prior to PH.
            We treat the neighborhood (excluding the query point) as defining the local chart.
            """
            if metric_norm in {"none", ""}:
                return cloud

            X = np.asarray(cloud, dtype=float)
            if X.ndim != 2 or X.shape[0] < 2:
                return X

            # Use neighborhood mean (exclude query point at row 0) for centering.
            neigh = X[1:]
            mu = neigh.mean(axis=0, keepdims=True)
            Xc = X - mu

            if metric_norm == "center":
                return Xc

            if metric_norm == "local_scale":
                # Scale by a robust local radius proxy (median kNN distance of query to neighbors).
                if dists is None:
                    # Fallback: median pairwise distance within neighborhood.
                    r = np.median(pdist(neigh)) if neigh.shape[0] >= 3 else float(np.linalg.norm(neigh[0] - neigh.mean(axis=0)))
                else:
                    dd = np.asarray(dists, dtype=float).ravel()
                    dd = dd[np.isfinite(dd)]
                    r = float(np.median(dd[dd > 0])) if np.any(dd > 0) else float(np.median(dd))
                r = float(r) if np.isfinite(r) and r > 1e-12 else 1.0
                return Xc / r

            if metric_norm == "whiten":
                # Whiten using neighborhood covariance (ridge-stabilized).
                Y = neigh - mu  # (k, d)
                if Y.shape[0] < 2:
                    return Xc
                cov = (Y.T @ Y) / max(1, (Y.shape[0] - 1))
                cov = cov + float(whiten_ridge) * np.eye(cov.shape[0], dtype=float)
                # SVD for inverse sqrt
                U, S, _ = np.linalg.svd(cov, full_matrices=False)
                S = np.maximum(S, 1e-12)
                W = U @ np.diag(1.0 / np.sqrt(S)) @ U.T
                return Xc @ W

            # Unknown normalization: fall back to none (do not break experiments silently).
            raise ValueError(f"Unknown graph.topo_metric_normalization: {metric_norm!r}")

        # Compute PH features for each query point's neighborhood cloud.
        # Note: neighborhood selection uses metric geometry only to define the local patch;
        # the detection statistic itself is topology-derived (PH summaries).
        topo_feat_dicts: List[Dict[str, float]] = []
        for i in range(n_points):
            # Choose which neighbor index to use.
            chosen = "global"
            query_class: Optional[int] = None
            if neighbor_mode == "class_pred":
                query_class = int(y_pred_points[i])
                if query_class in class_nbrs:
                    chosen = "class"
            elif neighbor_mode == "class_true":
                if y_points is None:
                    raise ValueError(
                        "graph.topo_neighbor_mode='class_true' requires y_points aligned with X_points."
                    )
                yp = np.asarray(y_points, dtype=int).ravel()
                if yp.shape[0] != n_points:
                    raise ValueError(f"X_points/y_points length mismatch: {n_points} vs {yp.shape[0]}")
                query_class = int(yp[i])
                if query_class in class_nbrs:
                    chosen = "class"

            # Get maximum-k neighborhood (we'll slice per k in k_list).
            if chosen == "class" and query_class is not None:
                nbrs = class_nbrs[query_class]
                Zref = class_Z[query_class]
            else:
                nbrs = global_nbrs
                Zref = Z_train

            dists_max, idx_max = nbrs.kneighbors(Z_points[i].reshape(1, -1))
            dists_max = dists_max[0]
            idx_max = idx_max[0]

            # Compute features for each requested k and merge into one dict (possibly prefixed).
            feats_merged: Dict[str, float] = {}
            for K in k_list:
                kk = int(min(max(1, K), len(idx_max)))
                neigh = Zref[idx_max[:kk]]
                cloud = np.vstack([Z_points[i].reshape(1, -1), neigh])
                cloud = _normalize_cloud(cloud, dists=dists_max[:kk])
                feats_k = local_persistence_features(cloud, topo_cfg)
                if prefix_keys:
                    # Prefix keys so they remain `topo_*` and distinct per k.
                    for key, val in feats_k.items():
                        # key is like "topo_h0_count" -> "topo_k30_h0_count"
                        if str(key).startswith("topo_"):
                            k2 = f"topo_k{K}_" + str(key)[5:]
                        else:
                            k2 = f"topo_k{K}_" + str(key)
                        feats_merged[k2] = float(val)
                else:
                    feats_merged.update({str(k): float(v) for k, v in feats_k.items()})

            topo_feat_dicts.append(feats_merged)

        # Materialize into score arrays with stable keys.
        all_keys = sorted({k for d in topo_feat_dicts for k in d.keys()})
        for k in all_keys:
            scores[k] = np.array([d.get(k, 0.0) for d in topo_feat_dicts], dtype=float)
    
    # Optional: diffusion scores
    if graph_params.use_diffusion:
        # Build training graph
        W_train, _, _ = build_knn_graph(
            Z_train, k=graph_params.k, sigma=graph_params.sigma
        )
        
        # Compute diffusion embedding
        _, eigenvectors = compute_diffusion_embedding(
            W_train, n_components=graph_params.diffusion_components
        )
        
        # Compute diffusion distance scores
        diffusion_scores = np.zeros(n_points)
        for i in range(n_points):
            diffusion_scores[i] = compute_diffusion_distance_score(
                Z_points[i], Z_train, eigenvectors, W_train,
                k=graph_params.k, sigma=graph_params.sigma
            )
        scores['diffusion'] = diffusion_scores
    
    # Combined score is computed separately if needed (see detectors.py)
    
    return scores


def compute_graph_scores_with_diagrams(
    X_point: np.ndarray,
    model,
    Z_train: np.ndarray,
    f_train: np.ndarray,
    graph_params: GraphConfig,
    device: str = 'cpu'
) -> Tuple[Dict[str, float], Sequence[np.ndarray], np.ndarray]:
    """
    Compute graph scores with persistence diagrams for a single point (for visualization).
    
    Args:
        X_point: Single point to score, shape (input_dim,)
        model: Trained PyTorch model
        Z_train: Training representations (in input or feature space)
        f_train: Training model outputs (probabilities or logits)
        graph_params: GraphConfig with hyperparameters
        device: Device for model inference
        
    Returns:
        Tuple of (features_dict, diagrams_list, point_cloud):
        - features_dict: Dictionary of topology summary features
        - diagrams_list: List of persistence diagrams [H0, H1, ...]
        - point_cloud: The local neighborhood point cloud used for PH
    """
    from .models import get_model_logits, extract_features_batch
    
    # Extract representation if needed
    if graph_params.space == 'feature':
        layer = str(getattr(graph_params, "feature_layer", "penultimate"))
        # Support both vector and image inputs:
        # - vector: (D,) -> (1,D)
        # - image: (C,H,W) -> (1,C,H,W)
        # - already batched: (1,...) -> keep
        Xp = np.asarray(X_point)
        if Xp.ndim == 1:
            Xb = Xp.reshape(1, -1)
        elif Xp.ndim == 3:
            Xb = Xp.reshape(1, *Xp.shape)
        elif Xp.ndim == 4:
            if Xp.shape[0] != 1:
                raise ValueError(f"Expected a single point for diagrams; got batch shape={Xp.shape}")
            Xb = Xp
        else:
            raise ValueError(f"Unsupported X_point ndim={Xp.ndim}. Expected 1 (vector) or 3 (C,H,W).")
        Z_point = extract_features_batch(model, Xb, layer=layer, device=device)[0]
    else:
        Z_point = X_point
    
    # Get topology features with diagrams
    if not getattr(graph_params, 'use_topology', False):
        raise ValueError("compute_graph_scores_with_diagrams requires use_topology=True")
    
    topo_k = int(getattr(graph_params, 'topo_k', 50))
    topo_cfg = TopologyConfig(
        neighborhood_k=topo_k,
        maxdim=int(getattr(graph_params, 'topo_maxdim', 1)),
        metric=str(getattr(graph_params, 'topo_metric', 'euclidean')),
        thresh=getattr(graph_params, 'topo_thresh', None),
        min_persistence=float(getattr(graph_params, 'topo_min_persistence', 1e-6)),
        preprocess=str(getattr(graph_params, 'topo_preprocess', 'none')),
        pca_dim=int(getattr(graph_params, 'topo_pca_dim', 10)),
    )
    
    # Build neighbor index (visualization helper sticks to global neighborhoods for simplicity).
    nbrs_topo = NearestNeighbors(
        n_neighbors=min(topo_k, len(Z_train)),
        metric=topo_cfg.metric,
    ).fit(Z_train)
    
    # Get neighborhood
    _, idx = nbrs_topo.kneighbors(Z_point.reshape(1, -1))
    neighborhood = Z_train[idx[0]]
    # Ensure the query point participates in the complex
    cloud = np.vstack([Z_point.reshape(1, -1), neighborhood])
    # Optional local metric normalization for visualization parity.
    metric_norm = str(getattr(graph_params, "topo_metric_normalization", "none")).strip().lower()
    whiten_ridge = float(getattr(graph_params, "topo_whiten_ridge", 1e-3))
    if metric_norm not in {"none", ""}:
        neigh = cloud[1:]
        mu = neigh.mean(axis=0, keepdims=True)
        cloud_c = cloud - mu
        if metric_norm == "center":
            cloud = cloud_c
        elif metric_norm == "local_scale":
            r = np.median(pdist(neigh)) if neigh.shape[0] >= 3 else 1.0
            r = float(r) if np.isfinite(r) and r > 1e-12 else 1.0
            cloud = cloud_c / r
        elif metric_norm == "whiten":
            Y = neigh - mu
            cov = (Y.T @ Y) / max(1, (Y.shape[0] - 1))
            cov = cov + float(whiten_ridge) * np.eye(cov.shape[0], dtype=float)
            U, S, _ = np.linalg.svd(cov, full_matrices=False)
            S = np.maximum(S, 1e-12)
            W = U @ np.diag(1.0 / np.sqrt(S)) @ U.T
            cloud = cloud_c @ W
        else:
            raise ValueError(f"Unknown graph.topo_metric_normalization: {metric_norm!r}")
    
    # Compute PH features with diagrams
    features, diagrams = local_persistence_features(cloud, topo_cfg, return_diagrams=True)
    
    return features, diagrams, cloud

