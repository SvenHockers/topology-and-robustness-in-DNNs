"""
Helper function to compute combined graph scores.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional


def compute_combined_score(
    scores: Dict[str, np.ndarray],
    alpha: float = 0.5,
    beta: float = 0.5,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute combined score from multiple graph-based scores.
    
    Args:
        scores: Dictionary of score arrays
        alpha: Weight for degree score
        beta: Weight for laplacian score
        normalize: Whether to normalize scores before combining
        
    Returns:
        Combined score array
    """
    degree_scores = scores.get('degree')
    laplacian_scores = scores.get('laplacian')
    
    if degree_scores is None or laplacian_scores is None:
        raise ValueError("Both 'degree' and 'laplacian' scores are required for combined score")
    
    if normalize:
        scaler_deg = StandardScaler()
        scaler_lap = StandardScaler()
        
        degree_norm = scaler_deg.fit_transform(degree_scores.reshape(-1, 1)).flatten()
        laplacian_norm = scaler_lap.fit_transform(
            laplacian_scores.reshape(-1, 1)
        ).flatten()
    else:
        degree_norm = degree_scores
        laplacian_norm = laplacian_scores
    
    combined = alpha * degree_norm + beta * laplacian_norm
    return combined


