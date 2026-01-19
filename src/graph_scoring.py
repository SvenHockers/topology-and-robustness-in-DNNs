"""
Graph construction and manifold conformity score computation.
"""

import numpy as np
import torch
from scipy.sparse.linalg import eigs
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from typing import Tuple, Optional, Dict, Sequence
from .utils import GraphConfig
from .topology_features import TopologyConfig, local_persistence_features


def _estimate_sigma_knn_median(
    Z_train: np.ndarray,
    *,
    k: int,
    max_samples: int = 2048,
    seed: int = 42,
) -> float:
    """
    Estimate the Gaussian kernel scale (sigma) using a *kNN-distance median* heuristic.

    IMPORTANT:
    We do **not** compute all-pairs distances (e.g. via scipy.spatial.distance.pdist),
    because that is O(n^2) memory and will crash on large datasets (e.g. IMAGE).
    """
    n = int(Z_train.shape[0])
    if n <= 1:
        return 1.0

    k_eff = int(max(1, min(int(k), n - 1)))
    rng = np.random.default_rng(int(seed))
    if n > int(max_samples):
        idx = rng.choice(n, size=int(max_samples), replace=False)
        Z_query = Z_train[idx]
    else:
        Z_query = Z_train

    nbrs = NearestNeighbors(n_neighbors=k_eff, metric="euclidean").fit(Z_train)
    distances, _indices = nbrs.kneighbors(Z_query)
    distances = np.asarray(distances, dtype=float)
    pos = distances[distances > 0]
    if pos.size == 0:
        return 1.0
    s = float(np.median(pos))
    if not np.isfinite(s) or s <= 0:
        return 1.0
    return s


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
        # Safe heuristic (kNN median) that does not allocate O(n^2) memory.
        sigma = _estimate_sigma_knn_median(Z_train, k=k, seed=42)
    
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
        sigma = _estimate_sigma_knn_median(Z_train, k=k, seed=42)
    
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
        sigma = _estimate_sigma_knn_median(Z_train, k=k, seed=42)
    
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
    device: str = 'cpu'
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
    scores: Dict[str, np.ndarray] = {}

    def _get_param(name: str, default=None):
        """
        Access graph params robustly whether `graph_params` is a dataclass (GraphConfig)
        or a plain dict (common in notebooks / serialized configs).
        """
        if isinstance(graph_params, dict):
            return graph_params.get(name, default)
        return getattr(graph_params, name, default)

    def _coerce_bool(v) -> bool:
        """
        Robust boolean parsing for config values that may come from YAML/JSON/CLI/notebooks.
        Handles common string forms like "false"/"0"/"no" and dict-based configs.
        """
        if isinstance(v, bool):
            return v
        # Numpy scalars (e.g., np.bool_) and numeric flags
        if isinstance(v, (int, float, np.integer, np.floating)):
            return bool(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"false", "0", "no", "n", "off"}:
                return False
            if s in {"true", "1", "yes", "y", "on"}:
                return True
            # Fall back: non-empty string is True, but be explicit about unknown tokens.
            return bool(s)
        return bool(v)

    # Backwards compatibility: some notebooks may pass `graph_params` as a plain dict.
    raw_use_baseline = _get_param("use_baseline_scores", True)
    use_baseline = _coerce_bool(raw_use_baseline)

    # Pull frequently used knobs once (works for dict or dataclass configs).
    k = int(_get_param("k", 10))
    sigma = _get_param("sigma", None)
    space = str(_get_param("space", "feature"))
    
    # Extract representations if needed
    if space == 'feature':
        # Keep query embedding space consistent with Z_train construction.
        # (Z_train is built upstream using cfg.graph.feature_layer.)
        layer = str(_get_param("feature_layer", "penultimate"))
        Z_points = extract_features_batch(model, X_points, layer=layer, device=device)
    else:
        Z_points = X_points
    if use_baseline:
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

        # Compute degree scores
        degree_scores = np.zeros(n_points)
        for i in range(n_points):
            degree_scores[i] = compute_degree_score(
                Z_points[i], Z_train, k=k, sigma=sigma
            )
        scores['degree'] = degree_scores

        # Compute Laplacian smoothness scores
        laplacian_scores = np.zeros(n_points)
        for i in range(n_points):
            laplacian_scores[i] = compute_laplacian_smoothness_score(
                Z_points[i], f_points[i], Z_train, f_train,
                k=k, sigma=sigma
            )
        scores['laplacian'] = laplacian_scores

        # Local tangent-space residual score (manifold membership test)
        # Intuition: if clean data lies on/near a low-dim manifold in Z-space, then
        # projecting onto the local tangent space should have small residual norm.
        # Backward-compatible defaults: if older GraphConfig objects are in memory (e.g. notebook
        # kernel not restarted), these attributes may be missing. We default to computing
        # tangent-based scores in that case.
        if _coerce_bool(_get_param('use_tangent', True)):
            tangent_scores = np.zeros(n_points)
            tangent_z_scores = np.zeros(n_points)
            tangent_k = int(_get_param('tangent_k', 20))
            nbrs_tangent = NearestNeighbors(
                n_neighbors=min(tangent_k, len(Z_train)),
                metric='euclidean'
            ).fit(Z_train)

            # If tangent_dim is not provided, pick it adaptively from local PCA explained variance.
            # This is important in feature space where the manifold may not be 2D after embedding.
            tangent_dim = _get_param('tangent_dim', None)
            var_thr = float(_get_param('tangent_var_threshold', 0.9))
            dim_min = int(_get_param('tangent_dim_min', 2))
            dim_max = _get_param('tangent_dim_max', None)

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
            nbrs_radius = NearestNeighbors(
                n_neighbors=min(k, len(Z_train)),
                metric='euclidean'
            ).fit(Z_train)
            dists, _ = nbrs_radius.kneighbors(Z_points)
            # exclude the first distance if it is 0 due to self-match (shouldn't happen for new points,
            # but keep robust)
            radius_scores = dists[:, 1:].mean(axis=1) if dists.shape[1] > 1 else dists[:, 0]
            scores['knn_radius'] = np.asarray(radius_scores, dtype=float)

    # --- Topology features (persistent homology on local neighborhoods) ---
    if _coerce_bool(_get_param('use_topology', False)):
        topo_k = int(_get_param('topo_k', 50))
        topo_cfg = TopologyConfig(
            neighborhood_k=topo_k,
            maxdim=int(_get_param('topo_maxdim', 1)),
            metric=str(_get_param('topo_metric', 'euclidean')),
            thresh=_get_param('topo_thresh', None),
            min_persistence=float(_get_param('topo_min_persistence', 1e-6)),
            preprocess=str(_get_param('topo_preprocess', 'none')),
            pca_dim=int(_get_param('topo_pca_dim', 10)),
            filtration=str(_get_param("topo_filtration", "standard")),
            query_lambda=float(_get_param("topo_query_lambda", 1.0)),
            query_gamma=float(_get_param("topo_query_gamma", 1.0)),
        )

        # Build a neighbor index for local neighborhoods in the scoring space.
        nbrs_topo = NearestNeighbors(
            n_neighbors=min(topo_k, len(Z_train)),
            metric=topo_cfg.metric,
        ).fit(Z_train)

        # Compute PH features for each query point's neighborhood cloud.
        # Note: neighborhood selection uses metric geometry only to define the local patch;
        # the detection statistic itself is topology-derived (PH summaries).
        topo_feat_dicts = []
        for i in range(n_points):
            _, idx = nbrs_topo.kneighbors(Z_points[i].reshape(1, -1))
            neighborhood = Z_train[idx[0]]
            # Ensure the query point participates in the complex (stability wrt insertion).
            cloud = np.vstack([Z_points[i].reshape(1, -1), neighborhood])
            feats_i = local_persistence_features(cloud, topo_cfg)
            topo_feat_dicts.append(feats_i)

        # Materialize into score arrays with stable keys.
        all_keys = sorted({k for d in topo_feat_dicts for k in d.keys()})
        for k in all_keys:
            scores[k] = np.array([d.get(k, 0.0) for d in topo_feat_dicts], dtype=float)
    
    # Optional: diffusion scores (baseline family)
    if use_baseline and _coerce_bool(_get_param("use_diffusion", False)):
        # Build training graph
        W_train, _, _ = build_knn_graph(
            Z_train, k=k, sigma=sigma
        )

        # Compute diffusion embedding
        _, eigenvectors = compute_diffusion_embedding(
            W_train, n_components=int(_get_param("diffusion_components", 10))
        )

        # Compute diffusion distance scores
        diffusion_scores = np.zeros(n_points)
        for i in range(n_points):
            diffusion_scores[i] = compute_diffusion_distance_score(
                Z_points[i], Z_train, eigenvectors, W_train,
                k=k, sigma=sigma
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

    def _get_param(name: str, default=None):
        if isinstance(graph_params, dict):
            return graph_params.get(name, default)
        return getattr(graph_params, name, default)

    def _coerce_bool(v) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float, np.integer, np.floating)):
            return bool(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"false", "0", "no", "n", "off"}:
                return False
            if s in {"true", "1", "yes", "y", "on"}:
                return True
            return bool(s)
        return bool(v)
    
    # Extract representation if needed
    if str(_get_param("space", "feature")) == 'feature':
        layer = str(_get_param("feature_layer", "penultimate"))
        # IMPORTANT: preserve input shape for non-vector modalities (e.g., images for CNNs).
        # `reshape(1, -1)` breaks CNN forward passes by flattening HxW.
        X_point = np.asarray(X_point)
        if X_point.ndim <= 1:
            X_batch = X_point.reshape(1, -1)
        else:
            X_batch = X_point[None, ...]
        Z_point = extract_features_batch(model, X_batch, layer=layer, device=device)[0]
    else:
        Z_point = X_point
    
    # Get topology features with diagrams
    if not _coerce_bool(_get_param('use_topology', False)):
        raise ValueError("compute_graph_scores_with_diagrams requires use_topology=True")
    
    topo_k = int(_get_param('topo_k', 50))
    topo_cfg = TopologyConfig(
        neighborhood_k=topo_k,
        maxdim=int(_get_param('topo_maxdim', 1)),
        metric=str(_get_param('topo_metric', 'euclidean')),
        thresh=_get_param('topo_thresh', None),
        min_persistence=float(_get_param('topo_min_persistence', 1e-6)),
        preprocess=str(_get_param('topo_preprocess', 'none')),
        pca_dim=int(_get_param('topo_pca_dim', 10)),
        filtration=str(_get_param("topo_filtration", "standard")),
        query_lambda=float(_get_param("topo_query_lambda", 1.0)),
        query_gamma=float(_get_param("topo_query_gamma", 1.0)),
    )
    
    # Build neighbor index
    nbrs_topo = NearestNeighbors(
        n_neighbors=min(topo_k, len(Z_train)),
        metric=topo_cfg.metric,
    ).fit(Z_train)
    
    # Get neighborhood
    _, idx = nbrs_topo.kneighbors(Z_point.reshape(1, -1))
    neighborhood = Z_train[idx[0]]
    # Ensure the query point participates in the complex
    cloud = np.vstack([Z_point.reshape(1, -1), neighborhood])
    
    # Compute PH features with diagrams
    features, diagrams = local_persistence_features(cloud, topo_cfg, return_diagrams=True)
    
    return features, diagrams, cloud

