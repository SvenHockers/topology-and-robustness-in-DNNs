from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import math
import random


def _normal_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def expected_improvement(
    *,
    mu: Sequence[float],
    sigma: Sequence[float],
    best_y: float,
    xi: float = 0.01,
) -> List[float]:
    out: List[float] = []
    for m, s in zip(mu, sigma):
        s2 = float(max(0.0, s))
        if s2 <= 1e-12:
            out.append(0.0)
            continue
        imp = float(m - best_y - xi)
        z = imp / s2
        out.append(float(imp * _normal_cdf(z) + s2 * _normal_pdf(z)))
    return out


@dataclass(frozen=True)
class GpConfig:

    noise: float = 1e-6
    n_restarts_optimizer: int = 3
    random_state: int = 42


class GaussianProcessSurrogate:

    def __init__(self, cfg: GpConfig):
        self.cfg = cfg
        self._gp = None

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[float]) -> None:
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore
            from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "GaussianProcessSurrogate requires scikit-learn. "
                "Install dependencies (e.g. `pip install -r requirements.txt`)."
            ) from e

        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        if len(X) == 0:
            raise ValueError("Need at least one observation to fit GP.")

        # Matern is a robust default for BO; WhiteKernel models observation noise.
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=2.5) + WhiteKernel(
            noise_level=float(self.cfg.noise), noise_level_bounds=(1e-10, 1e1)
        )
        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=float(self.cfg.noise),
            normalize_y=True,
            n_restarts_optimizer=int(self.cfg.n_restarts_optimizer),
            random_state=int(self.cfg.random_state),
        )
        self._gp.fit(list(map(list, X)), list(map(float, y)))

    def predict(self, X: Sequence[Sequence[float]]) -> Tuple[List[float], List[float]]:
        if self._gp is None:
            raise RuntimeError("GP not fitted yet.")
        mu, std = self._gp.predict(list(map(list, X)), return_std=True)
        return [float(m) for m in mu.tolist()], [float(s) for s in std.tolist()]


def propose_by_random_ei(
    *,
    rng: random.Random,
    surrogate: GaussianProcessSurrogate,
    candidates: Sequence[Sequence[float]],
    best_y: float,
    xi: float,
) -> Tuple[int, float]:
    mu, sigma = surrogate.predict(candidates)
    eis = expected_improvement(mu=mu, sigma=sigma, best_y=best_y, xi=xi)
    best_idx = int(max(range(len(eis)), key=lambda i: eis[i]))
    return best_idx, float(eis[best_idx])

