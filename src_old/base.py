# Base classes used throughout the project
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional, Sequence, Dict, Any

# Type aliases for clarity
Vec = Sequence[float]
VertexList = Sequence[Vec]
Cell = Sequence[int]
CellList = Sequence[Cell]
ParametricFn = Callable[[float], Vec]

class Boundary(Enum):
    """Represents how a shapes boundary is specified.

    - CONTINUOUS: analytic/parametric closed curve or hypersurface
    - POLYGON: discrete boundary via vertices joined by straight segments (polyline/polygon)
    - MESH: interior represented by a simplicial/cellular mesh (vertices + cells)
    """
    CONTINUOUS = auto()
    POLYGON = auto()
    MESH = auto()


class InvalidShapeError(ValueError):
    """This is a custom error I've defined if shape validation fails"""


@dataclass(frozen=True)
class Shape:
    """Base class for an n-dimensional geometric object (n => 2).

    A Shape describes *how* a region and/or its boundary is represented. It is intentionally
    generic so it can back multiple downstream tasks (visualization, topology, etc.).

    Attributes
    -----------
    dim: int
        The ambient dimension of the shape (must be => 2).

    boundary: Boundary
        How the boundary is specified: CONTINUOUS | POLYGON | MESH.

    filled: bool
        If True, the region has interior; if False, the shape is boundary-only.

    data: dict
        Representation payload. Expected keys depend on `boundary`:
        - CONTINUOUS: {"parametric": Callable[[float], Sequence[float]], "period": float (optional)}
        - POLYGON:    {"vertices": Sequence[Sequence[float]]}
        - MESH:       {"vertices": Sequence[Sequence[float]], "cells": Sequence[Sequence[int]]}

    Notes
    * Validation is very basic, we only check dimensional consistency and basic structure.
    * For POLYGON the vertices are assumed to be ordered to trace the boundary.
    * For CONTINUOUS the parametric function should be closed over its domain (e.g. [0,1]).
    """

    dim: int
    boundary: Boundary
    filled: bool = False
    data: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_continuous(parametric: ParametricFn, *, dim: Optional[int] = None, period: float = 1.0, filled: bool = False) -> "Shape":
        sample = parametric(0.0)
        inferred_dim = len(sample)
        d = dim if dim is not None else inferred_dim
        return Shape(dim=d, boundary=Boundary.CONTINUOUS, filled=filled, data={"parametric": parametric, "period": period})

    @staticmethod
    def from_polygon(vertices: VertexList, *, filled: bool = False) -> "Shape":
        if vertices is None or len(vertices) == 0:
            raise InvalidShapeError("POLYGON requires a non-empty list of vertices.")
        dim = len(vertices[0])
        return Shape(dim=dim, boundary=Boundary.POLYGON, filled=filled, data={"vertices": list(vertices)})

    @staticmethod
    def from_mesh(vertices: VertexList, cells: CellList, *, filled: bool = True) -> "Shape":
        if vertices is None or len(vertices) == 0:
            raise InvalidShapeError("MESH requires a non-empty list of vertices.")
        if cells is None or len(cells) == 0:
            raise InvalidShapeError("MESH requires a non-empty list of cells (simplices).")
        dim = len(vertices[0])
        return Shape(dim=dim, boundary=Boundary.MESH, filled=filled, data={"vertices": list(vertices), "cells": list(cells)})

    def validate(self) -> None:
        if self.dim < 2:
            raise InvalidShapeError("Shape dimension must be > 2.")

        if self.boundary is Boundary.CONTINUOUS:
            fn = self.data.get("parametric")
            if fn is None or not callable(fn):
                raise InvalidShapeError("CONTINUOUS boundary requires a callable 'parametric' in data.")
            # Basic dimensionality sanity check
            try:
                v = fn(0.0)
            except Exception as e:
                raise InvalidShapeError(f"Parametric function failed at t=0: {e}")
            if not hasattr(v, "__len__"):
                raise InvalidShapeError("Parametric function must return a sequence of floats.")
            if len(v) != self.dim:
                raise InvalidShapeError(f"Parametric function returns dimension {len(v)} but Shape.dim is {self.dim}.")

        elif self.boundary is Boundary.POLYGON:
            verts = self.data.get("vertices")
            if not isinstance(verts, (list, tuple)) or len(verts) < 2:
                raise InvalidShapeError("POLYGON requires at least 2 vertices.")
            for i, p in enumerate(verts):
                if len(p) != self.dim:
                    raise InvalidShapeError(f"Vertex {i} has dimension {len(p)} but expected {self.dim}.")

        elif self.boundary is Boundary.MESH:
            verts = self.data.get("vertices")
            cells = self.data.get("cells")
            if not isinstance(verts, (list, tuple)) or not verts:
                raise InvalidShapeError("MESH requires non-empty 'vertices'.")
            if not isinstance(cells, (list, tuple)) or not cells:
                raise InvalidShapeError("MESH requires non-empty 'cells'.")
            for i, p in enumerate(verts):
                if len(p) != self.dim:
                    raise InvalidShapeError(f"Vertex {i} has dimension {len(p)} but expected {self.dim}.")
            nverts = len(verts)
            for j, c in enumerate(cells):
                if not isinstance(c, (list, tuple)) or not c:
                    raise InvalidShapeError(f"Cell {j} must be a non-empty sequence of vertex indices.")
                if any((idx < 0 or idx >= nverts) for idx in c):
                    raise InvalidShapeError(f"Cell {j} contains out-of-range vertex indices (0..{nverts-1}).")
        else:
            raise InvalidShapeError("Unknown Boundary type.")

    def is_void(self) -> bool:
        return not self.filled

    def is_filled(self) -> bool:
        return self.filled

    def boundary_kind(self) -> Boundary:
        return self.boundary

    def vertices(self) -> Optional[List[Vec]]:
        return self.data.get("vertices")  # type: ignore[return-value]

    def cells(self) -> Optional[List[Cell]]:
        return self.data.get("cells")  # type: ignore[return-value]

    def parametric(self) -> Optional[ParametricFn]:
        return self.data.get("parametric")  # type: ignore[return-value]

    def __repr__(self) -> str:
        core = f"dim={self.dim}, boundary={self.boundary.name}, filled={self.filled}"
        if self.boundary is Boundary.CONTINUOUS:
            extras = "parametric=True"
        elif self.boundary is Boundary.POLYGON:
            v = self.data.get("vertices"); n = len(v) if isinstance(v, (list, tuple)) else 0
            extras = f"vertices={n}"
        else:
            v = self.data.get("vertices"); c = self.data.get("cells")
            nv = len(v) if isinstance(v, (list, tuple)) else 0
            nc = len(c) if isinstance(c, (list, tuple)) else 0
            extras = f"vertices={nv}, cells={nc}"
        return f"Shape({core}; {extras})"

    @classmethod
    def create(cls, *, dim: int, boundary: Boundary, filled: bool = False, data: Optional[Dict[str, Any]] = None) -> "Shape":
        """Generic constructor with validation.

        Use this when you already have a boundary type and a payload prepared.
        """
        shape = cls(dim=dim, boundary=boundary, filled=filled, data=data or {})
        object.__setattr__(shape, "_validated", True) # we store the validation such we know if its valid after creation
        shape.validate()
        return shape