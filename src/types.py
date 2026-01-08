"""
Public API types (dataclasses).

These are thin, structural containers for results returned by `src.api`.
They intentionally do not implement logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class AttackResult:
    """Container for adversarial examples and optional metadata."""

    X_adv: np.ndarray
    meta: Dict[str, Any]


@dataclass(frozen=True)
class DetectorEvalResult:
    """Container for detector evaluation outputs."""

    labels: np.ndarray
    raw_scores: np.ndarray
    metrics: Dict[str, Any]
    # Optional matplotlib plots, as returned by `src.visualization`.
    plots: Dict[str, Any]


@dataclass(frozen=True)
class RunResult:
    """
    End-to-end pipeline outputs.

    Note: to avoid import cycles, `bundle`, `model`, and `detector` are typed as Any.
    """

    cfg: Any
    bundle: Any
    model: Any
    detector: Any

    attack_val: Optional[AttackResult]
    attack_test: Optional[AttackResult]

    scores_val_clean: Dict[str, np.ndarray]
    scores_val_adv: Dict[str, np.ndarray]
    scores_test_clean: Dict[str, np.ndarray]
    scores_test_adv: Dict[str, np.ndarray]

    eval: DetectorEvalResult

