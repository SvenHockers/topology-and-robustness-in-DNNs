from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple

import math
import random


ParamKind = Literal["float", "int", "categorical"]


@dataclass(frozen=True)
class ParamSpec:
    """
    One optimisable parameter that maps onto a dotted config path.

    Example:
      - name="graph.k", path="graph.k", kind="int", min=5, max=80
      - name="detector.topo_cov_shrinkage", path="detector.topo_cov_shrinkage", kind="float", min=1e-6, max=1e-1, log=True
    """

    name: str
    path: str
    kind: ParamKind
    min: float
    max: float
    log: bool = False
    choices: Optional[Tuple[str, ...]] = None

    def sample(self, rng: random.Random) -> float | int:
        if self.kind == "categorical":
            if not self.choices:
                raise ValueError(f"categorical param {self.name!r} requires non-empty choices")
            return str(rng.choice(list(self.choices)))

        lo = float(self.min)
        hi = float(self.max)
        if lo > hi:
            lo, hi = hi, lo

        if self.kind == "int":
            if self.log:
                # log-uniform on a continuous scale, then round to int bounds
                x = _log_uniform(rng, lo, hi)
                return int(_clamp_int(int(round(x)), int(math.floor(lo)), int(math.ceil(hi))))
            return int(rng.randint(int(math.floor(lo)), int(math.ceil(hi))))

        # float
        if self.log:
            return float(_log_uniform(rng, lo, hi))
        return float(rng.uniform(lo, hi))

    def encode(self, value: float | int) -> float:
        """
        Encode a parameter into a scalar feature for GP fitting.

        For log-scaled parameters we encode log10(value) (clamped to bounds).
        """
        if self.kind == "categorical":
            if not self.choices:
                raise ValueError(f"categorical param {self.name!r} requires non-empty choices")
            s = str(value)
            try:
                return float(self.choices.index(s))
            except ValueError:
                raise ValueError(f"Value {s!r} not in choices for {self.name!r}: {self.choices!r}")

        v = float(value)
        lo = float(min(self.min, self.max))
        hi = float(max(self.min, self.max))
        if self.log:
            # clamp to avoid log domain errors
            v = float(max(lo, min(hi, v)))
            v = max(v, 1e-300)
            return float(math.log10(v))
        return float(max(lo, min(hi, v)))

    def decode(self, encoded: float) -> float | int:
        """
        Inverse of encode() for convenience (used when generating candidates).
        """
        if self.kind == "categorical":
            if not self.choices:
                raise ValueError(f"categorical param {self.name!r} requires non-empty choices")
            idx = int(round(float(encoded)))
            idx = max(0, min(idx, len(self.choices) - 1))
            return str(self.choices[idx])

        lo = float(min(self.min, self.max))
        hi = float(max(self.min, self.max))
        x = float(encoded)
        if self.log:
            v = 10.0**x
        else:
            v = x
        v = float(max(lo, min(hi, v)))
        if self.kind == "int":
            return int(_clamp_int(int(round(v)), int(math.floor(lo)), int(math.ceil(hi))))
        return float(v)


@dataclass(frozen=True)
class SearchSpace:
    params: Tuple[ParamSpec, ...]

    def __post_init__(self) -> None:
        if not self.params:
            raise ValueError("SearchSpace must have at least one parameter.")
        # Ensure unique names
        names = [p.name for p in self.params]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate parameter names in search space: {names!r}")

    @property
    def dim(self) -> int:
        return int(len(self.params))

    def sample_params(self, rng: random.Random) -> Dict[str, float | int]:
        return {p.name: p.sample(rng) for p in self.params}

    def vectorize(self, params: Mapping[str, float | int]) -> List[float]:
        """
        Convert parameter dict into a numeric vector consistent with `self.params` order.
        """
        out: List[float] = []
        for p in self.params:
            if p.name not in params:
                raise KeyError(f"Missing parameter {p.name!r} in params dict.")
            out.append(p.encode(params[p.name]))
        return out

    def unvectorize(self, x: Sequence[float]) -> Dict[str, float | int]:
        if len(x) != len(self.params):
            raise ValueError(f"Vector length {len(x)} does not match space dim {len(self.params)}")
        out: Dict[str, float | int] = {}
        for p, xi in zip(self.params, x):
            out[p.name] = p.decode(float(xi))
        return out

    def apply_to_config_dict(self, cfg: MutableMapping[str, Any], params: Mapping[str, float | int]) -> None:
        """
        Mutate a nested config dict in-place, setting dotted-path values.
        """
        for p in self.params:
            if p.name not in params:
                raise KeyError(f"Missing parameter {p.name!r} in params dict.")
            _set_dotted(cfg, p.path, params[p.name])


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, v)))


def _log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    lo2 = float(min(lo, hi))
    hi2 = float(max(lo, hi))
    if lo2 <= 0 or hi2 <= 0:
        raise ValueError(f"log-uniform bounds must be > 0, got lo={lo}, hi={hi}")
    a = math.log10(lo2)
    b = math.log10(hi2)
    return float(10.0 ** rng.uniform(a, b))


def _set_dotted(d: MutableMapping[str, Any], dotted: str, value: Any) -> None:
    """
    Set `d['a']['b']['c'] = value` from dotted path "a.b.c".
    Creates intermediate dicts as needed.
    """
    if not dotted or not isinstance(dotted, str):
        raise ValueError("dotted path must be a non-empty string")
    keys = dotted.split(".")
    cur: MutableMapping[str, Any] = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]  # type: ignore[assignment]
    cur[keys[-1]] = value


def specs_from_dict(obj: Mapping[str, Any]) -> SearchSpace:
    """
    Parse a search-space spec from a dict.

    Expected formats:

    1) {"params": [ {ParamSpec fields...}, ... ]}
    2) [ {ParamSpec fields...}, ... ]  (top-level list)
    """
    if isinstance(obj, list):
        params_obj = obj
    else:
        params_obj = obj.get("params")

    if not isinstance(params_obj, list):
        raise TypeError("Search space spec must be a list or a dict with key 'params' as a list.")

    specs: List[ParamSpec] = []
    for i, p in enumerate(params_obj):
        if not isinstance(p, dict):
            raise TypeError(f"Param spec at index {i} must be a dict; got {type(p)}")
        kind = str(p.get("kind") or "float")
        choices_raw = p.get("choices", None)
        choices: Optional[Tuple[str, ...]] = None
        if choices_raw is not None:
            if not isinstance(choices_raw, list) or not all(isinstance(x, (str, int, float, bool)) for x in choices_raw):
                raise TypeError(f"choices for param {p.get('name') or p.get('path') or i} must be a list of scalars")
            choices = tuple(str(x) for x in choices_raw)
        specs.append(
            ParamSpec(
                name=str(p.get("name") or p.get("path") or f"p{i}"),
                path=str(p.get("path") or p.get("name") or f"p{i}"),
                kind=kind,  # type: ignore[arg-type]
                min=float(p.get("min", 0.0)),
                max=float(p.get("max", 0.0)),
                log=bool(p.get("log", False)),
                choices=choices,
            )
        )

    return SearchSpace(params=tuple(specs))

