from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class AttackResult:
    X_adv: np.ndarray
    meta: Dict[str, Any]


@dataclass(frozen=True)
class OODResult:
    X_ood: np.ndarray
    meta: Dict[str, Any]


@dataclass(frozen=True)
class DetectorEvalResult:
    labels: np.ndarray
    raw_scores: np.ndarray
    metrics: Dict[str, Any]
    # Optional matplotlib plots, as returned by `src.visualization`.
    plots: Dict[str, Any]


@dataclass(frozen=True)
class RunResult:
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

    ood_val: Optional[OODResult] = None
    ood_test: Optional[OODResult] = None
    scores_val_ood: Optional[Dict[str, np.ndarray]] = None
    scores_test_ood: Optional[Dict[str, np.ndarray]] = None
    eval_ood: Optional[DetectorEvalResult] = None

