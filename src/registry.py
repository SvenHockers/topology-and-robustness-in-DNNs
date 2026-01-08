"""
Registries for pluggable components (models, attacks).

The repo already has a dataset registry in `src.data.DATASET_REGISTRY`.
This module adds the analogous pattern for models/attacks so users can extend
the library without editing core code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch.nn as nn


ModelFactory = Callable[..., nn.Module]
AttackFn = Callable[..., np.ndarray]


MODEL_REGISTRY: Dict[str, ModelFactory] = {}
ATTACK_REGISTRY: Dict[str, AttackFn] = {}


def register_model(name: str) -> Callable[[ModelFactory], ModelFactory]:
    """Decorator to register a model factory under a stable name."""

    def _decorator(fn: ModelFactory) -> ModelFactory:
        key = str(name).lower()
        MODEL_REGISTRY[key] = fn
        return fn

    return _decorator


def register_attack(name: str) -> Callable[[AttackFn], AttackFn]:
    """Decorator to register an attack generator under a stable name."""

    def _decorator(fn: AttackFn) -> AttackFn:
        key = str(name).lower()
        ATTACK_REGISTRY[key] = fn
        return fn

    return _decorator


def list_models() -> list[str]:
    return sorted(MODEL_REGISTRY.keys())


def list_attacks() -> list[str]:
    return sorted(ATTACK_REGISTRY.keys())


def get_model_factory(name: str) -> ModelFactory:
    key = str(name).lower()
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available: {list_models()}")
    return MODEL_REGISTRY[key]


def get_attack_fn(name: str) -> AttackFn:
    key = str(name).lower()
    if key not in ATTACK_REGISTRY:
        raise KeyError(f"Unknown attack {name!r}. Available: {list_attacks()}")
    return ATTACK_REGISTRY[key]


