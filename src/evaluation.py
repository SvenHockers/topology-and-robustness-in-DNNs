"""
Evaluation metrics and calibration functions for the threshold
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, auc, 
    precision_recall_curve, average_precision_score
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from typing import Tuple, Dict, Optional, Any


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute standard classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with metrics: accuracy, precision, recall, f1
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def evaluate_detector(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Evaluate adversarial detector performance.
    
    Args:
        y_true: True binary labels (1 = adversarial, 0 = clean)
        y_scores: Detector scores (higher = more likely adversarial)
        threshold: Optional threshold for binary predictions
        
    Returns:
        Dictionary with metrics: auc, fpr_at_tpr95, etc.
    """
    # ROC curve
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Find FPR at TPR = 0.95
    tpr_target = 0.95
    idx = np.argmax(tpr >= tpr_target)
    fpr_at_tpr95 = fpr[idx] if idx > 0 else 1.0
    
    # Precision-Recall curve
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    
    # If threshold provided, compute binary metrics
    binary_metrics = {}
    if threshold is not None:
        y_pred = (y_scores >= threshold).astype(int)
        binary_metrics = compute_classification_metrics(y_true, y_pred)
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'fpr_at_tpr95': fpr_at_tpr95,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds_roc': thresholds_roc,
        # Store PR curve arrays under non-colliding keys (binary metrics use 'precision'/'recall').
        'pr_precision': pr_precision,
        'pr_recall': pr_recall,
        'pr_thresholds': pr_thresholds,
        **binary_metrics
    }


def calibrate_error_probability(
    scores: np.ndarray,
    is_wrong: np.ndarray,
    method: str = 'isotonic'
) -> object:
    """
    Calibrate detector scores to error probabilities.
    
    Maps score -> P(model is wrong | score).
    
    Args:
        scores: Detector scores
        is_wrong: Binary labels (1 = model prediction is wrong, 0 = correct)
        method: Calibration method ('isotonic' or 'logistic')
        
    Returns:
        Calibrated model object
    """
    if method == 'isotonic':
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(scores, is_wrong)
        return calibrator
    elif method == 'logistic':
        calibrator = LogisticRegression()
        calibrator.fit(scores.reshape(-1, 1), is_wrong)
        return calibrator
    else:
        raise ValueError(f"Unknown calibration method: {method}")


def predict_error_probability(
    calibration_model: object,
    scores: np.ndarray,
    method: str = 'isotonic'
) -> np.ndarray:
    """
    Predict error probabilities using calibrated model.
    
    Args:
        calibration_model: Trained calibration model
        scores: Detector scores to calibrate
        method: Calibration method used
        
    Returns:
        Error probabilities (P(wrong | score))
    """
    if method == 'isotonic':
        probs = calibration_model.predict(scores)
        # Ensure probabilities are in [0, 1]
        probs = np.clip(probs, 0.0, 1.0)
        return probs
    elif method == 'logistic':
        probs = calibration_model.predict_proba(scores.reshape(-1, 1))[:, 1]
        return probs
    else:
        raise ValueError(f"Unknown calibration method: {method}")


def compute_calibration_metrics(
    predicted_probs: np.ndarray,
    true_labels: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Compute calibration metrics (ECE, MCE).
    
    Args:
        predicted_probs: Predicted probabilities
        true_labels: True binary labels
        n_bins: Number of bins for calibration
        
    Returns:
        Dictionary with Expected Calibration Error (ECE) and Max Calibration Error (MCE)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    mce = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (predicted_probs > bin_lower) & (predicted_probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = true_labels[in_bin].mean()
            avg_confidence_in_bin = predicted_probs[in_bin].mean()
            
            bin_error = abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += prop_in_bin * bin_error
            mce = max(mce, bin_error)
    
    return {
        'ece': ece,
        'mce': mce
    }


