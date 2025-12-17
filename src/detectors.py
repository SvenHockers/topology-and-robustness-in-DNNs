"""
Graph-based adversarial detection methods.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from typing import Tuple, Optional, Dict
from .utils import DetectorConfig
from .compute_combined_score import compute_combined_score


class ScoreBasedDetector:
    """
    Simple score-based anomaly detector using graph manifold scores.
    """
    
    def __init__(self, score_type: str = 'combined'):
        """
        Initialize score-based detector.
        
        Args:
            score_type: Which score to use ('degree', 'laplacian', 'diffusion', 'combined',
                'tangent_residual', 'knn_radius')
        """
        self.score_type = score_type
        self.threshold = None
    
    def fit(self, scores: Dict[str, np.ndarray], labels: np.ndarray, percentile: float = 95):
        """
        Fit detector by computing threshold on validation set.
        
        Args:
            scores: Dictionary of score arrays
            labels: Binary labels (1 = adversarial, 0 = clean)
            percentile: Percentile of clean scores to use as threshold
        """
        clean_mask = (labels == 0)
        clean_scores = scores[self.score_type][clean_mask]
        
        # Set threshold at percentile of clean scores
        self.threshold = np.percentile(clean_scores, percentile)
    
    def predict(self, scores: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Predict whether points are adversarial based on scores.
        
        Args:
            scores: Dictionary of score arrays
            
        Returns:
            Binary predictions (1 = adversarial, 0 = clean)
        """
        if self.threshold is None:
            raise ValueError("Detector must be fitted first")
        
        predictions = (scores[self.score_type] > self.threshold).astype(int)
        return predictions
    
    def predict_proba(self, scores: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Predict adversarial probabilities (approximated from scores).
        
        Args:
            scores: Dictionary of score arrays
            
        Returns:
            Probability array (probability of being adversarial)
        """
        if self.threshold is None:
            raise ValueError("Detector must be fitted first")
        
        # Simple sigmoid-like transformation around threshold
        score_values = scores[self.score_type]
        distances = score_values - self.threshold
        
        # Normalize to [0, 1] range using softmax-like function
        probs = 1.0 / (1.0 + np.exp(-distances))
        
        return probs


class SupervisedGraphDetector:
    """
    Supervised detector that learns from graph-based score features.
    """
    
    def __init__(self, detector_type: str = 'logistic'):
        """
        Initialize supervised detector.
        
        Args:
            detector_type: Type of classifier ('logistic' or 'isolation_forest')
        """
        self.detector_type = detector_type
        self.detector = None
    
    def fit(self, score_features: np.ndarray, labels: np.ndarray):
        """
        Train detector on score features.
        
        Args:
            score_features: Feature matrix of shape (n_samples, n_score_types)
            labels: Binary labels (1 = adversarial, 0 = clean)
        """
        if self.detector_type == 'logistic':
            self.detector = LogisticRegression(random_state=42, max_iter=1000)
            self.detector.fit(score_features, labels)
        elif self.detector_type == 'isolation_forest':
            # For isolation forest, we treat adversarial as anomalies
            # Invert labels: 1 = anomaly, -1 = normal
            self.detector = IsolationForest(random_state=42, contamination=0.1)
            self.detector.fit(score_features)
            self.is_adversarial = (labels == 1)
        else:
            raise ValueError(f"Unknown detector type: {self.detector_type}")
    
    def predict(self, score_features: np.ndarray) -> np.ndarray:
        """
        Predict whether points are adversarial.
        
        Args:
            score_features: Feature matrix
            
        Returns:
            Binary predictions (1 = adversarial, 0 = clean)
        """
        if self.detector is None:
            raise ValueError("Detector must be fitted first")
        
        if self.detector_type == 'logistic':
            return self.detector.predict(score_features)
        if self.detector_type == 'isolation_forest':
            anomaly_pred = self.detector.predict(score_features)
            # Convert: 1 = normal, -1 = anomaly -> 0 = clean, 1 = adversarial
            return (anomaly_pred == -1).astype(int)
        raise ValueError(f"Unknown detector type: {self.detector_type}")
    
    def predict_proba(self, score_features: np.ndarray) -> np.ndarray:
        """
        Predict adversarial probabilities.
        
        Args:
            score_features: Feature matrix
            
        Returns:
            Probability array (probability of being adversarial)
        """
        if self.detector is None:
            raise ValueError("Detector must be fitted first")
        
        if self.detector_type == 'logistic':
            probs = self.detector.predict_proba(score_features)
            # Return probability of class 1 (adversarial)
            if probs.shape[1] == 2:
                return probs[:, 1]
            else:
                return probs[:, 0]
        elif self.detector_type == 'isolation_forest':
            # Isolation forest doesn't provide probabilities directly
            # Use decision function as proxy
            decision_scores = self.detector.decision_function(score_features)
            # Normalize to [0, 1]
            probs = 1.0 / (1.0 + np.exp(-decision_scores))
            return 1.0 - probs  # Invert: lower decision score = more adversarial
        else:
            raise ValueError(f"Unknown detector type: {self.detector_type}")


def train_graph_detector(
    scores: Dict[str, np.ndarray],
    labels: np.ndarray,
    config: DetectorConfig
):
    """
    Train a graph-based detector.
    
    Args:
        scores: Dictionary of score arrays
        labels: Binary labels (1 = adversarial, 0 = clean)
        config: DetectorConfig with detector parameters
        
    Returns:
        Trained detector object
    """
    # Ensure 'combined' score exists if requested
    if config.score_type == 'combined' and 'combined' not in scores:
        scores = dict(scores)  # avoid mutating caller
        scores['combined'] = compute_combined_score(
            scores,
            alpha=config.alpha,
            beta=config.beta,
            normalize=True,
        )

    if config.detector_type == 'score':
        detector = ScoreBasedDetector(score_type=config.score_type)
        detector.fit(scores, labels)
        return detector
    elif config.detector_type == 'supervised':
        # Prepare feature matrix
        score_types = ['degree', 'laplacian']
        if 'diffusion' in scores:
            score_types.append('diffusion')
        
        feature_matrix = np.column_stack([scores[st] for st in score_types])
        
        detector = SupervisedGraphDetector(detector_type='logistic')
        detector.fit(feature_matrix, labels)
        return detector
    else:
        raise ValueError(f"Unknown detector type: {config.detector_type}")


def predict_graph_detector(detector, scores: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions with a trained detector.
    
    Args:
        detector: Trained detector object
        scores: Dictionary of score arrays
        
    Returns:
        Tuple of (predictions, probabilities)
    """
    # Ensure 'combined' score exists if the detector expects it
    if isinstance(detector, ScoreBasedDetector) and detector.score_type == 'combined' and 'combined' not in scores:
        scores = dict(scores)  # avoid mutating caller
        scores['combined'] = compute_combined_score(scores, normalize=True)

    if isinstance(detector, ScoreBasedDetector):
        predictions = detector.predict(scores)
        probabilities = detector.predict_proba(scores)
        return predictions, probabilities
    elif isinstance(detector, SupervisedGraphDetector):
        # Prepare feature matrix
        score_types = ['degree', 'laplacian']
        if 'diffusion' in scores:
            score_types.append('diffusion')
        
        feature_matrix = np.column_stack([scores[st] for st in score_types])
        
        predictions = detector.predict(feature_matrix)
        probabilities = detector.predict_proba(feature_matrix)
        return predictions, probabilities
    else:
        raise ValueError(f"Unknown detector type: {type(detector)}")


