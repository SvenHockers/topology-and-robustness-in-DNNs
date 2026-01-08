r"""
Detectors for adversarial / OOD detection.

This repo's *standard* detector is the **topological (persistent homology) scoring
detector** described in `theory.md`.

Conceptual pipeline (see `theory.md`):
- Represent: \(z=\phi(x)\) (typically a neural embedding; computed upstream)
- Localize: \(P(z)=\{z\}\cup N_k(z)\) from training representations
- Topologize: persistent homology on \(P(z)\) (optionally after local PCA)
- Vectorize: diagram summaries → fixed-length topology feature vector \(v(x)\)
- Score: suspiciousness score \(s(x)\) = (shrunk) Mahalanobis distance of \(v(x)\)
         to the clean reference distribution
- Decide: flag if \(s(x) > \tau\), where \(\tau\) is a clean-quantile threshold

Implementation mapping:
- Topology feature computation: `src/topology_features.py`
- Feature emission for model inputs: `src/graph_manifold.py` (keys like `topo_h{0,1}_*`)
- Scoring detector + thresholding: this file (`TopologyScoreDetector`)
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from .utils import DetectorConfig


class TopologyScoreDetector:
    r"""
    Topological (persistent-homology) scoring detector.

    This implements the Part B scoring rule in `theory.md`:

    - **Inputs**: a dict of topology summary arrays (emitted by `compute_graph_scores(...)`)
      where each key corresponds to one coordinate of the topology feature vector \(v(x)\).
    - **Calibration on clean data**: estimate \(\mu\) and \(\Sigma\) for \(v(x)\), apply
      diagonal shrinkage \(\Sigma_\lambda=\Sigma+\lambda I\).
    - **Score**: suspiciousness \(s(x)=\sqrt{(v(x)-\mu)^\top \Sigma_\lambda^\dagger (v(x)-\mu)}\).
    - **Threshold**: \(\tau\) is set as a clean quantile (percentile), targeting an
      approximate false-positive rate under the clean calibration distribution.
    """

    def __init__(
        self,
        feature_keys: list,
        cov_shrinkage: float = 1e-3,
        percentile: float = 95.0,
    ):
        self.feature_keys = list(feature_keys)
        self.cov_shrinkage = float(cov_shrinkage)
        self.percentile = float(percentile)

        self.mu_: Optional[np.ndarray] = None
        self.inv_cov_: Optional[np.ndarray] = None
        self.threshold: Optional[float] = None

    def _feature_matrix(self, scores: Dict[str, np.ndarray]) -> np.ndarray:
        r"""
        Construct the topology feature matrix \(V\) by column-stacking selected keys.

        `scores` is expected to include keys like `topo_h0_*`, `topo_h1_*` as produced by
        `src.graph_scoring.compute_graph_scores(..., use_topology=True)`.
        """
        X = np.column_stack([scores[k] for k in self.feature_keys]).astype(float)
        # Replace any NaNs/Infs defensively; PH backends can emit edge-case values.
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    def fit(self, scores: Dict[str, np.ndarray], labels: np.ndarray):
        r"""
        Fit the clean reference distribution and calibrate \(\tau\).

        `labels` are used only to select the clean calibration set \(\mathcal{C}\)
        (i.e., points with label 0). This keeps thresholding tied to the clean
        distribution rather than a specific threat model.
        """
        if len(self.feature_keys) == 0:
            raise ValueError("TopologyScoreDetector requires non-empty feature_keys.")
        for k in self.feature_keys:
            if k not in scores:
                raise KeyError(f"Missing topology feature key {k!r} in scores dict.")

        X = self._feature_matrix(scores)
        clean_mask = (labels == 0)
        X0 = X[clean_mask]
        if X0.shape[0] < 5:
            raise ValueError("Need at least 5 clean samples to fit topology detector.")

        mu = X0.mean(axis=0, keepdims=False)
        Xc = X0 - mu.reshape(1, -1)
        cov = (Xc.T @ Xc) / max(1, (Xc.shape[0] - 1))
        # Diagonal shrinkage for stability in small-sample / correlated-feature regimes.
        cov = cov + self.cov_shrinkage * np.eye(cov.shape[0], dtype=float)
        inv_cov = np.linalg.pinv(cov)

        self.mu_ = mu
        self.inv_cov_ = inv_cov

        clean_scores = self.score(scores)[clean_mask]
        self.threshold = float(np.percentile(clean_scores, self.percentile))
        return self

    def score(self, scores: Dict[str, np.ndarray]) -> np.ndarray:
        r"""Compute the suspiciousness score \(s(x)\) (Mahalanobis distance in feature space)."""
        if self.mu_ is None or self.inv_cov_ is None:
            raise ValueError("TopologyScoreDetector must be fitted first.")
        X = self._feature_matrix(scores)
        Xc = X - self.mu_.reshape(1, -1)
        # Squared Mahalanobis distances (monotone in \(s(x)\)).
        m2 = np.einsum("bi,ij,bj->b", Xc, self.inv_cov_, Xc)
        m2 = np.maximum(m2, 0.0)
        return np.sqrt(m2)

    def predict(self, scores: Dict[str, np.ndarray]) -> np.ndarray:
        r"""Apply the calibrated decision rule \(\mathbf{1}\{s(x)>\tau\}\)."""
        if self.threshold is None:
            raise ValueError("TopologyScoreDetector must be fitted first.")
        s = self.score(scores)
        return (s > self.threshold).astype(int)

    def predict_proba(self, scores: Dict[str, np.ndarray]) -> np.ndarray:
        r"""
        Return a **monotone confidence proxy** derived from \(s(x)\).

        Important: this is **not** a calibrated probability. The detector is a
        *scoring + quantile-thresholding* method (see `theory.md` B8).
        """
        if self.threshold is None:
            raise ValueError("TopologyScoreDetector must be fitted first.")
        s = self.score(scores)
        distances = s - self.threshold
        probs = 1.0 / (1.0 + np.exp(-distances))
        return probs


def train_graph_detector(
    scores: Dict[str, np.ndarray],
    labels: np.ndarray,
    config: DetectorConfig
):
    r"""
    Train / calibrate the detector.

    In this repository, this function **standardizes** to the topological detector:
    it selects topology summary keys, fits the clean reference distribution in
    topology-feature space, and sets a clean-quantile threshold.
    
    Args:
        scores: Dict of score arrays, including PH summary keys like `topo_h{0,1}_*`.
        labels: Binary labels (1 = adversarial, 0 = clean); used only to select clean points.
        config: DetectorConfig (controls feature selection + shrinkage + percentile).
        
    Returns:
        Calibrated detector object (TopologyScoreDetector).
    """
    # Standardized in this project: always train the topology-score detector.
    # This keeps configuration consistent across datasets and notebooks.
    #
    # Choose topology feature keys.
    # Default: use standard PH summary keys if present; else all keys prefixed with "topo_".
    topo_keys = getattr(config, 'topo_feature_keys', None)
    if topo_keys is not None:
        feature_keys = list(topo_keys)
    else:
        default_order = [
            'topo_h0_count',
            'topo_h0_total_persistence',
            'topo_h0_entropy',
            'topo_h1_count',
            'topo_h1_total_persistence',
            'topo_h1_max_persistence',
            'topo_h1_l2_persistence',
            'topo_h1_entropy',
        ]
        feature_keys = [k for k in default_order if k in scores]
        if len(feature_keys) == 0:
            feature_keys = sorted([k for k in scores.keys() if str(k).startswith('topo_')])

    if len(feature_keys) == 0:
        raise ValueError(
            "No topology features found in scores.\n"
            "Enable topology feature computation by setting:\n"
            "  config.graph.use_topology = True\n"
            "and ensure you have a PH backend installed (e.g., ripser)."
        )

    detector = TopologyScoreDetector(
        feature_keys=feature_keys,
        cov_shrinkage=float(getattr(config, 'topo_cov_shrinkage', 1e-3)),
        percentile=float(getattr(config, 'topo_percentile', 95.0)),
    )
    detector.fit(scores, labels)
    return detector


def predict_graph_detector(detector, scores: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Make predictions with a trained detector.

    Returns:
    - `predictions`: binary flags \(\mathbf{1}\{s(x)>\tau\}\)
    - `probabilities`: a *monotone proxy* derived from the underlying score
      (not a calibrated probability; see `theory.md` B7.1 / B8).
    
    Args:
        detector: Trained detector object
        scores: Dictionary of score arrays
        
    Returns:
        Tuple of (predictions, probabilities)
    """
    if isinstance(detector, TopologyScoreDetector):
        predictions = detector.predict(scores)
        probabilities = detector.predict_proba(scores)
        return predictions, probabilities
    # Fallback: allow duck-typed detectors used in ad-hoc notebook experiments.
    # We keep this minimal to avoid maintaining unused detector classes in src/.
    if hasattr(detector, "predict") and hasattr(detector, "predict_proba"):
        predictions = detector.predict(scores)  # type: ignore[misc]
        probabilities = detector.predict_proba(scores)  # type: ignore[misc]
        return np.asarray(predictions, dtype=int), np.asarray(probabilities, dtype=float)
    else:
        raise ValueError(f"Unknown detector type: {type(detector)}")


