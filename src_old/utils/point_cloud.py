from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..base import Boundary, Shape


ArrayLike = Union[Sequence[float], np.ndarray]


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def _edge_lengths(vertices: np.ndarray) -> np.ndarray:
    diffs = np.diff(vertices, axis=0)
    closed_diffs = np.vstack([diffs, vertices[0] - vertices[-1]])
    return np.linalg.norm(closed_diffs, axis=1)


def _sample_polygon_boundary(vertices: np.ndarray, num_points: int, *, seed: Optional[int]) -> np.ndarray:
    n = vertices.shape[0]
    if n < 2:
        raise ValueError("Polygon requires at least 2 vertices.")

    lengths = _edge_lengths(vertices)
    perimeter = float(np.sum(lengths))
    if perimeter == 0.0:
        return np.repeat(vertices[:1], num_points, axis=0)

    cumulative = np.cumsum(lengths) / perimeter
    rng = _rng(seed)
    u = rng.random(num_points)
    edge_indices = np.searchsorted(cumulative, u, side="right")

    pts = np.empty((num_points, vertices.shape[1]), dtype=float)
    for i, ei in enumerate(edge_indices):
        a = vertices[ei % n]
        b = vertices[(ei + 1) % n]
        # local proportion along the chosen edge
        if ei == 0:
            prev_cum = 0.0
        else:
            prev_cum = cumulative[ei - 1]
        local_u = (u[i] - prev_cum) / (cumulative[ei] - prev_cum)
        pts[i] = (1.0 - local_u) * a + local_u * b
    return pts


def _point_in_polygon(point: np.ndarray, vertices: np.ndarray) -> bool:
    # Ray casting algorithm, supports 2D polygons (ambient dim can be >=2, we use first two axes)
    x, y = float(point[0]), float(point[1])
    xs = vertices[:, 0]
    ys = vertices[:, 1]
    inside = False
    j = len(vertices) - 1
    for i in range(len(vertices)):
        xi, yi = xs[i], ys[i]
        xj, yj = xs[j], ys[j]
        if ((yi > y) != (yj > y)):
            x_intersect = (xj - xi) * (y - yi) / (yj - yi + 1e-20) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def _sample_polygon_interior(vertices: np.ndarray, num_points: int, *, seed: Optional[int]) -> np.ndarray:
    if vertices.shape[1] < 2:
        raise ValueError("Polygon interior sampling requires at least 2D coordinates.")

    mins = np.min(vertices, axis=0)
    maxs = np.max(vertices, axis=0)
    rng = _rng(seed)

    points: List[np.ndarray] = []
    trials = 0
    max_trials = max(1000, 20 * num_points)
    while len(points) < num_points and trials < max_trials:
        trials += 1
        candidate = rng.uniform(mins, maxs)
        if _point_in_polygon(candidate, vertices[:, :2]):
            points.append(candidate)
    if len(points) < num_points:
        # Fall back: if rejection is inefficient (e.g., highly concave), duplicate accepted points
        if points:
            needed = num_points - len(points)
            idx = rng.integers(0, len(points), size=needed)
            points.extend(points[i] for i in idx)
        else:
            # As a last resort, return boundary samples
            return _sample_polygon_boundary(vertices, num_points, seed=seed)
    return np.asarray(points, dtype=float)


def _simplex_volume(vertices: np.ndarray) -> float:
    # vertices: (k+1, d) simplex in R^d; volume = sqrt(det(G)) / k!
    base = vertices[1:] - vertices[0]
    gram = base @ base.T
    det = float(np.linalg.det(gram))
    if det <= 0:
        return 0.0
    k = vertices.shape[0] - 1
    k_fact = float(np.math.factorial(k))
    return np.sqrt(det) / k_fact


def _sample_points_in_simplex(vertices: np.ndarray, m: int, *, seed: Optional[int]) -> np.ndarray:
    # Dirichlet trick: sample barycentric coordinates ~ Dirichlet(1,...,1)
    rng = _rng(seed)
    d = vertices.shape[1]
    k1 = vertices.shape[0]
    w = rng.random((m, k1))
    w = -np.log(w)
    w /= np.sum(w, axis=1, keepdims=True)
    return w @ vertices


def _sample_mesh_interior(vertices: np.ndarray, cells: Sequence[Sequence[int]], num_points: int, *, seed: Optional[int]) -> np.ndarray:
    # Weight simplices by volume for uniform sampling across the mesh
    simplices = [vertices[np.asarray(cell, dtype=int)] for cell in cells]
    volumes = np.asarray([_simplex_volume(s) for s in simplices], dtype=float)
    total = float(np.sum(volumes))
    rng = _rng(seed)
    if total == 0.0:
        # Degenerate; sample barycentric uniformly across simplices evenly
        counts = np.full(len(simplices), max(1, num_points // max(1, len(simplices))), dtype=int)
    else:
        probs = volumes / total
        counts = rng.multinomial(num_points, probs)

    pts_list: List[np.ndarray] = []
    for s, c in zip(simplices, counts):
        if c <= 0:
            continue
        pts_list.append(_sample_points_in_simplex(s, c, seed=int(rng.integers(0, 2**31 - 1))))
    if not pts_list:
        return np.repeat(vertices[:1], num_points, axis=0)
    pts = np.vstack(pts_list)
    # If due to rounding we have fewer or more, adjust by random choice
    if pts.shape[0] != num_points:
        idx = rng.permutation(pts.shape[0])[:num_points]
        pts = pts[idx]
    return pts


def _sample_continuous_boundary(shape: Shape, num_points: int, *, seed: Optional[int]) -> np.ndarray:
    fn = shape.parametric()
    assert fn is not None
    period = float(shape.data.get("period", 1.0))
    rng = _rng(seed)
    t = rng.random(num_points) * period
    samples = [np.asarray(fn(float(tt)), dtype=float) for tt in t]
    return np.vstack(samples)


def shape_to_point_cloud(
    shape: Shape,
    num_points: int,
    *,
    sample: str = "auto",
    seed: Optional[int] = None,
) -> np.ndarray:
    """Convert a Shape into a point cloud.

    Parameters
    ----------
    shape: Shape
        The geometric shape to sample.
    num_points: int
        Number of points to generate.
    sample: {"auto", "boundary", "interior"}
        What to sample. "auto" chooses "interior" if the shape is filled, otherwise "boundary".
    seed: Optional[int]
        Random seed for reproducibility.
    """
    if num_points <= 0:
        return np.empty((0, shape.dim), dtype=float)

    mode = sample
    if mode == "auto":
        mode = "interior" if shape.is_filled() else "boundary"

    if shape.boundary_kind() is Boundary.POLYGON:
        verts = np.asarray(shape.vertices(), dtype=float)
        if mode == "boundary":
            return _sample_polygon_boundary(verts, num_points, seed=seed)
        else:
            return _sample_polygon_interior(verts, num_points, seed=seed)

    if shape.boundary_kind() is Boundary.MESH:
        verts = np.asarray(shape.vertices(), dtype=float)
        cells = shape.cells()
        if not cells:
            raise ValueError("MESH shape must contain cells.")
        return _sample_mesh_interior(verts, cells, num_points, seed=seed)

    if shape.boundary_kind() is Boundary.CONTINUOUS:
        return _sample_continuous_boundary(shape, num_points, seed=seed)

    raise ValueError("Unsupported shape boundary kind.")


def add_gaussian_noise(points: np.ndarray, sigma: float, *, seed: Optional[int] = None) -> np.ndarray:
    if sigma <= 0:
        return points.copy()
    rng = _rng(seed)
    return points + rng.normal(0.0, sigma, size=points.shape)


def rotate_points(points: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    if rotation.shape[0] != rotation.shape[1]:
        raise ValueError("Rotation matrix must be square.")
    if rotation.shape[0] != points.shape[1]:
        raise ValueError("Rotation matrix dimensionality must match point dimensionality.")
    return points @ rotation.T


def random_rotation_matrix(dim: int, *, seed: Optional[int] = None) -> np.ndarray:
    rng = _rng(seed)
    A = rng.normal(size=(dim, dim))
    # QR decomposition then adjust to ensure a proper rotation (determinant = +1)
    Q, R = np.linalg.qr(A)
    d = np.diag(R)
    ph = np.sign(d)
    Q = Q * ph
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def scale_points(points: np.ndarray, scale: Union[float, ArrayLike], *, center: Optional[ArrayLike] = None) -> np.ndarray:
    pts = points
    if center is not None:
        c = np.asarray(center, dtype=float)
        pts = pts - c
    s = np.asarray(scale, dtype=float)
    if s.ndim == 0:
        scaled = pts * float(s)
    else:
        scaled = pts * s.reshape(1, -1)
    if center is not None:
        scaled = scaled + c
    return scaled


def translate_points(points: np.ndarray, translation: ArrayLike) -> np.ndarray:
    t = np.asarray(translation, dtype=float).reshape(1, -1)
    return points + t


@dataclass
class TransformSpec:
    rotation: Optional[np.ndarray] = None
    scale: Optional[Union[float, ArrayLike]] = None
    translation: Optional[ArrayLike] = None
    noise_sigma: float = 0.0
    seed: Optional[int] = None


def apply_transforms(points: np.ndarray, spec: TransformSpec) -> np.ndarray:
    out = points.copy()
    if spec.rotation is not None:
        out = rotate_points(out, spec.rotation)
    if spec.scale is not None:
        out = scale_points(out, spec.scale)
    if spec.translation is not None:
        out = translate_points(out, spec.translation)
    if spec.noise_sigma > 0:
        out = add_gaussian_noise(out, spec.noise_sigma, seed=spec.seed)
    return out


def generate_point_cloud(
    shape: Shape,
    num_points: int,
    *,
    sample: str = "auto",
    transforms: Optional[TransformSpec] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """High-level convenience: sample a shape and apply optional transforms."""
    pts = shape_to_point_cloud(shape, num_points, sample=sample, seed=seed)
    if transforms is not None:
        pts = apply_transforms(pts, transforms)
    return pts


