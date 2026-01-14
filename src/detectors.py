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
from typing import Tuple, Optional, Dict, Any, Iterable
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


class ClassConditionalTopologyScoreDetector:
    r"""
    Class-conditional variant of TopologyScoreDetector.

    Instead of fitting a single Gaussian to all clean samples, this fits one Gaussian
    per class on clean samples and scores points via:
      - mode='min_over_classes': s(x) = min_c d_M(v(x), mu_c, Sigma_c)
      - mode='predicted_class': s(x) = d_M(v(x), mu_{y_hat}, Sigma_{y_hat})
      - mode='true_class': s(x) = d_M(v(x), mu_{y}, Sigma_{y})  (analysis/oracle)

    Notes:
    - This is a detector in *topology-feature* space. It does not guarantee that the
      base classifier is correct; it measures typicality under class-conditional
      clean topology statistics.
    - Defaults are chosen to keep inference compatible with existing call sites:
      `score(scores)` works without labels when mode='min_over_classes'.
    """

    def __init__(
        self,
        feature_keys: list,
        cov_shrinkage: float = 1e-3,
        percentile: float = 95.0,
        *,
        mode: str = "min_over_classes",
        min_clean_per_class: int = 5,
    ):
        self.feature_keys = list(feature_keys)
        self.cov_shrinkage = float(cov_shrinkage)
        self.percentile = float(percentile)
        self.mode = str(mode)
        self.min_clean_per_class = int(min_clean_per_class)

        self.classes_: Optional[np.ndarray] = None
        self.mu_by_class_: Optional[Dict[int, np.ndarray]] = None
        self.inv_cov_by_class_: Optional[Dict[int, np.ndarray]] = None
        self.threshold: Optional[float] = None

    def _feature_matrix(self, scores: Dict[str, np.ndarray]) -> np.ndarray:
        X = np.column_stack([scores[k] for k in self.feature_keys]).astype(float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    @staticmethod
    def _unique_int_classes(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=int).ravel()
        return np.unique(y)

    def fit(
        self,
        scores: Dict[str, np.ndarray],
        labels: np.ndarray,
        *,
        y_true: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
    ):
        """
        Fit per-class clean reference distributions and calibrate threshold.

        Args:
            scores: dict of topology features (same format as TopologyScoreDetector).
            labels: binary (0=clean, 1=adversarial/OOD) used only to select clean points.
            y_true: class labels aligned with `scores` rows (needed for per-class fit).
        """
        if len(self.feature_keys) == 0:
            raise ValueError("ClassConditionalTopologyScoreDetector requires non-empty feature_keys.")
        for k in self.feature_keys:
            if k not in scores:
                raise KeyError(f"Missing topology feature key {k!r} in scores dict.")

        X = self._feature_matrix(scores)
        labels = np.asarray(labels, dtype=int).ravel()
        if X.shape[0] != labels.shape[0]:
            raise ValueError(f"Scores/labels length mismatch: X has {X.shape[0]} rows, labels has {labels.shape[0]}.")

        y_true = np.asarray(y_true, dtype=int).ravel()
        if y_true.shape[0] != X.shape[0]:
            raise ValueError(f"Scores/y_true length mismatch: X has {X.shape[0]} rows, y_true has {y_true.shape[0]}.")

        clean_mask = (labels == 0)
        X0 = X[clean_mask]
        y0 = y_true[clean_mask]
        if X0.shape[0] < 5:
            raise ValueError("Need at least 5 clean samples to fit class-conditional topology detector.")

        classes = self._unique_int_classes(y0)
        mu_by: Dict[int, np.ndarray] = {}
        inv_by: Dict[int, np.ndarray] = {}

        for c in classes.tolist():
            idx = (y0 == int(c))
            Xc = X0[idx]
            if Xc.shape[0] == 0:
                continue
            mu = Xc.mean(axis=0, keepdims=False)
            Z = Xc - mu.reshape(1, -1)

            # If a class has very few clean points, full covariance is unstable.
            # Fallback: diagonal covariance from per-feature variance + shrinkage.
            if Xc.shape[0] < int(self.min_clean_per_class):
                var = np.var(Xc, axis=0, ddof=1) if Xc.shape[0] > 1 else np.zeros((Xc.shape[1],), dtype=float)
                cov = np.diag(var.astype(float, copy=False))
            else:
                cov = (Z.T @ Z) / max(1, (Z.shape[0] - 1))

            cov = cov + self.cov_shrinkage * np.eye(cov.shape[0], dtype=float)
            inv = np.linalg.pinv(cov)

            mu_by[int(c)] = mu
            inv_by[int(c)] = inv

        if len(mu_by) == 0:
            raise ValueError("No class-conditional statistics were fitted (check y_true/labels alignment).")

        self.classes_ = np.array(sorted(mu_by.keys()), dtype=int)
        self.mu_by_class_ = mu_by
        self.inv_cov_by_class_ = inv_by

        # Calibrate threshold on clean scores using the configured mode.
        clean_scores = self.score(scores, y_pred=y_pred, y_true=y_true, mode=self.mode)[clean_mask]
        self.threshold = float(np.percentile(clean_scores, self.percentile))
        return self

    def _score_against_class(self, X: np.ndarray, *, c: int) -> np.ndarray:
        if self.mu_by_class_ is None or self.inv_cov_by_class_ is None:
            raise ValueError("Detector must be fitted first.")
        mu = self.mu_by_class_[int(c)]
        inv = self.inv_cov_by_class_[int(c)]
        Xc = X - mu.reshape(1, -1)
        m2 = np.einsum("bi,ij,bj->b", Xc, inv, Xc)
        m2 = np.maximum(m2, 0.0)
        return np.sqrt(m2)

    def score(
        self,
        scores: Dict[str, np.ndarray],
        *,
        y_pred: Optional[np.ndarray] = None,
        y_true: Optional[np.ndarray] = None,
        mode: Optional[str] = None,
    ) -> np.ndarray:
        if self.classes_ is None or self.mu_by_class_ is None or self.inv_cov_by_class_ is None:
            raise ValueError("ClassConditionalTopologyScoreDetector must be fitted first.")
        X = self._feature_matrix(scores)

        m = str(mode if mode is not None else self.mode).strip().lower()
        if m == "min_over_classes":
            # Compute per-class distances and take min.
            dists = []
            for c in self.classes_.tolist():
                dists.append(self._score_against_class(X, c=int(c)))
            return np.min(np.stack(dists, axis=0), axis=0)

        if m in {"predicted_class", "pred"}:
            if y_pred is None:
                raise ValueError("mode='predicted_class' requires y_pred.")
            y_pred = np.asarray(y_pred, dtype=int).ravel()
            if y_pred.shape[0] != X.shape[0]:
                raise ValueError(f"Scores/y_pred length mismatch: X has {X.shape[0]} rows, y_pred has {y_pred.shape[0]}.")
            out = np.empty((X.shape[0],), dtype=float)
            for c in self.classes_.tolist():
                mask = (y_pred == int(c))
                if not np.any(mask):
                    continue
                out[mask] = self._score_against_class(X[mask], c=int(c))
            # Handle labels outside fitted set (e.g., new class id) by scoring to closest class.
            unseen = ~np.isin(y_pred, self.classes_)
            if np.any(unseen):
                # Conservative: use min_over_classes for those points.
                out[unseen] = self.score({k: np.asarray(v)[unseen] for k, v in scores.items()}, mode="min_over_classes")
            return out

        if m in {"true_class", "true"}:
            if y_true is None:
                raise ValueError("mode='true_class' requires y_true.")
            y_true = np.asarray(y_true, dtype=int).ravel()
            if y_true.shape[0] != X.shape[0]:
                raise ValueError(f"Scores/y_true length mismatch: X has {X.shape[0]} rows, y_true has {y_true.shape[0]}.")
            out = np.empty((X.shape[0],), dtype=float)
            for c in self.classes_.tolist():
                mask = (y_true == int(c))
                if not np.any(mask):
                    continue
                out[mask] = self._score_against_class(X[mask], c=int(c))
            unseen = ~np.isin(y_true, self.classes_)
            if np.any(unseen):
                out[unseen] = self.score({k: np.asarray(v)[unseen] for k, v in scores.items()}, mode="min_over_classes")
            return out

        raise ValueError(f"Unknown class-conditional scoring mode: {mode!r}.")

    def predict(
        self,
        scores: Dict[str, np.ndarray],
        *,
        y_pred: Optional[np.ndarray] = None,
        y_true: Optional[np.ndarray] = None,
        mode: Optional[str] = None,
    ) -> np.ndarray:
        if self.threshold is None:
            raise ValueError("Detector must be fitted first (threshold missing).")
        s = self.score(scores, y_pred=y_pred, y_true=y_true, mode=mode)
        return (s > float(self.threshold)).astype(int)

    def predict_proba(
        self,
        scores: Dict[str, np.ndarray],
        *,
        y_pred: Optional[np.ndarray] = None,
        y_true: Optional[np.ndarray] = None,
        mode: Optional[str] = None,
    ) -> np.ndarray:
        if self.threshold is None:
            raise ValueError("Detector must be fitted first (threshold missing).")
        s = self.score(scores, y_pred=y_pred, y_true=y_true, mode=mode)
        distances = s - float(self.threshold)
        probs = 1.0 / (1.0 + np.exp(-distances))
        return probs


def train_graph_detector(
    scores: Dict[str, np.ndarray],
    labels: np.ndarray,
    config: DetectorConfig,
    *,
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
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

    use_cc = bool(getattr(config, "topo_class_conditional", False))
    if use_cc:
        if y_true is None:
            raise ValueError(
                "Class-conditional topology scoring requires class labels `y_true`.\n"
                "Pass y_true to train_graph_detector(..., y_true=...) or disable:\n"
                "  cfg.detector.topo_class_conditional = False"
            )
        detector = ClassConditionalTopologyScoreDetector(
            feature_keys=feature_keys,
            cov_shrinkage=float(getattr(config, "topo_cov_shrinkage", 1e-3)),
            percentile=float(getattr(config, "topo_percentile", 95.0)),
            mode=str(getattr(config, "topo_class_scoring_mode", "min_over_classes")),
            min_clean_per_class=int(getattr(config, "topo_min_clean_per_class", 5)),
        )
        detector.fit(
            scores,
            labels,
            y_true=np.asarray(y_true, dtype=int),
            y_pred=None if y_pred is None else np.asarray(y_pred, dtype=int),
        )
        return detector

    detector = TopologyScoreDetector(
        feature_keys=feature_keys,
        cov_shrinkage=float(getattr(config, "topo_cov_shrinkage", 1e-3)),
        percentile=float(getattr(config, "topo_percentile", 95.0)),
    )
    detector.fit(scores, labels)
    return detector


def predict_graph_detector(
    detector,
    scores: Dict[str, np.ndarray],
    *,
    y_pred: Optional[np.ndarray] = None,
    y_true: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
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
    if isinstance(detector, ClassConditionalTopologyScoreDetector):
        predictions = detector.predict(scores, y_pred=y_pred, y_true=y_true)
        probabilities = detector.predict_proba(scores, y_pred=y_pred, y_true=y_true)
        return predictions, probabilities
    # Fallback: allow duck-typed detectors used in ad-hoc notebook experiments.
    # We keep this minimal to avoid maintaining unused detector classes in src/.
    if hasattr(detector, "predict") and hasattr(detector, "predict_proba"):
        # If the duck-typed detector supports y_pred/y_true, pass them; else fall back.
        try:
            predictions = detector.predict(scores, y_pred=y_pred, y_true=y_true)  # type: ignore[misc]
            probabilities = detector.predict_proba(scores, y_pred=y_pred, y_true=y_true)  # type: ignore[misc]
        except TypeError:
            predictions = detector.predict(scores)  # type: ignore[misc]
            probabilities = detector.predict_proba(scores)  # type: ignore[misc]
        return np.asarray(predictions, dtype=int), np.asarray(probabilities, dtype=float)
    else:
        raise ValueError(f"Unknown detector type: {type(detector)}")


