"""
Mechanistic synthetic datasets for hypothesis validation.

This file defines *data generators* (pure numpy) that are intended to test a
specific, topology-aligned mechanism in a controlled setting.

## Fundamental research question (mechanistic)
When do persistent-homology (PH) summaries of local kNN neighborhoods in
representation space become *comparable* across the data manifold, and when do
they expose "decision–geometry inconsistency" induced by adversarial
perturbations?

## Design principle
We want a dataset where:
1) Clean data lie on a simple low-d manifold (so the "true" neighborhood geometry
   is interpretable).
2) Local neighborhoods are *anisotropic* in a location-dependent way, so raw
   Euclidean distances can confound PH features (density / scaling / anisotropy).
3) A local metric conditioning step (e.g., whitening) should remove the nuisance
   anisotropy and make PH summaries more stable/comparable.

This lets us empirically test whether "metric-conditioned, class-conditional
local geometry" is the key ingredient (as suggested by the strong V3 result on
synthetic_shapes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class AnisotropicMoonsParams:
    """
    Parameters for `generate_anisotropic_moons`.

    The key knobs are `sigma_tangent` and `sigma_normal`, which define an
    anisotropic noise ellipse aligned with a *local* direction field (rotating
    with position). `class_aniso_scale` lets class 1 have a different anisotropy
    level than class 0 (multi-modality across classes).
    """

    sigma_tangent: float = 0.25
    sigma_normal: float = 0.05
    class_aniso_scale: float = 1.5
    warp_strength: float = 0.0


def generate_anisotropic_moons(
    *,
    n_samples: int = 1000,
    noise: float = 0.02,
    seed: int = 42,
    params: AnisotropicMoonsParams = AnisotropicMoonsParams(),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a 2D "two moons" dataset with *location-dependent anisotropic noise*.

    Returns:
      X: (N, 2) float64
      y: (N,) int

    Notes:
    - We start from a low-noise two-moons base manifold.
    - We then add anisotropic Gaussian noise in a *rotating local frame* defined
      by theta = atan2(y, x). This creates neighborhoods whose covariance varies
      strongly with location even on the same class manifold.
    - Class 1 gets its anisotropy scaled by `class_aniso_scale` to introduce a
      class-conditional multi-modality in local geometry.
    """

    from sklearn.datasets import make_moons

    rng = np.random.default_rng(int(seed))

    X0, y = make_moons(n_samples=int(n_samples), noise=float(noise), random_state=int(seed))
    X0 = np.asarray(X0, dtype=float)
    y = np.asarray(y, dtype=int)

    # Optional mild warp to create a varying local metric (nuisance) without
    # changing the global topology.
    if float(params.warp_strength) != 0.0:
        theta0 = np.arctan2(X0[:, 1], X0[:, 0])
        rscale = 1.0 + float(params.warp_strength) * np.sin(2.0 * theta0)
        X0 = X0 * rscale[:, None]

    # Define a rotating local frame from position angle.
    theta = np.arctan2(X0[:, 1], X0[:, 0])
    t = np.stack([np.cos(theta), np.sin(theta)], axis=1)  # tangent-like direction field
    n = np.stack([-np.sin(theta), np.cos(theta)], axis=1)  # normal (perpendicular)

    # Base anisotropic scales (class 0).
    sig_t = float(params.sigma_tangent)
    sig_n = float(params.sigma_normal)

    # Sample anisotropic noise in (t, n) coordinates.
    eps_t = rng.normal(size=(len(X0), 1))
    eps_n = rng.normal(size=(len(X0), 1))

    # Class-conditional scaling (class 1 is "more/less anisotropic").
    scale = np.ones((len(X0), 1), dtype=float)
    scale[y == 1] = float(params.class_aniso_scale)

    noise_aniso = scale * (sig_t * eps_t * t + sig_n * eps_n * n)
    X = X0 + noise_aniso
    return X, y

