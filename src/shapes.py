from __future__ import annotations
from typing import Callable, List, Sequence, Tuple
import numpy as np
from .base import Boundary, Shape

"""
2D Shapes
"""
def _normalize_vertices(vertices: np.ndarray) -> np.ndarray:
    mins = np.min(vertices, axis=0)
    maxs = np.max(vertices, axis=0)
    span = np.maximum(maxs - mins, 1e-9)
    return (vertices - mins) / span

def make_line(p0: Sequence[float] = (-0.5, -0.5), p1: Sequence[float] = (0.5, 0.5)) -> Shape:
    v = [list(p0), list(p1)]
    return Shape.from_polygon(v, filled=False)


def make_circle(radius: float = 1.0, center: Sequence[float] = (0.0, 0.0)) -> Shape:
    cx, cy = float(center[0]), float(center[1])

    def param(t: float) -> Sequence[float]:
        ang = 2.0 * np.pi * t
        return [cx + radius * np.cos(ang), cy + radius * np.sin(ang)]

    return Shape.from_continuous(parametric=param, dim=2, period=1.0, filled=False)


def make_figure8(radius: float = 0.5) -> Shape:
    # Two touching circles of radius r centered at (-r, 0) and (r, 0). Touch at origin.
    def param(t: float) -> Sequence[float]:
        u = (t % 1.0)
        if u < 0.5:
            s = u * 2.0
            ang = 2.0 * np.pi * s
            return [-radius + radius * np.cos(ang), radius * np.sin(ang)]
        else:
            s = (u - 0.5) * 2.0
            ang = 2.0 * np.pi * s
            return [radius + radius * np.cos(ang), radius * np.sin(ang)]

    return Shape.from_continuous(parametric=param, dim=2, period=1.0, filled=False)


def make_two_circles(radius: float = 0.35, separation: float = 0.9) -> Shape:
    # Two disjoint circles centered at (-sep/2, 0) and (sep/2, 0)
    cx_left = -separation * 0.5
    cx_right = separation * 0.5

    def param(t: float) -> Sequence[float]:
        u = (t % 1.0)
        if u < 0.5:
            s = u * 2.0
            ang = 2.0 * np.pi * s
            return [cx_left + radius * np.cos(ang), radius * np.sin(ang)]
        else:
            s = (u - 0.5) * 2.0
            ang = 2.0 * np.pi * s
            return [cx_right + radius * np.cos(ang), radius * np.sin(ang)]

    return Shape.from_continuous(parametric=param, dim=2, period=1.0, filled=False)


def make_spiral(turns: float = 2.0, r0: float = 0.1, r1: float = 1.0) -> Shape:
    # Archimedean spiral from radius r0 to r1 over `turns` turns.
    a = r0
    b = (r1 - r0) / (2.0 * np.pi * max(turns, 1e-6))

    def param(t: float) -> Sequence[float]:
        u = (t % 1.0)
        theta = u * (2.0 * np.pi * turns)
        r = a + b * theta
        return [r * np.cos(theta), r * np.sin(theta)]

    return Shape.from_continuous(parametric=param, dim=2, period=1.0, filled=False)


def make_swiss_roll_2d(width: float = 0.15, turns: float = 2.0) -> Shape:
    # A single-curve proxy of a swiss roll centerline (for simplicity)
    # Users can inflate via raster distance transform.
    center = make_spiral(turns=turns, r0=0.2, r1=0.95)
    return center


"""
3D Shapes
"""
def _icosahedron() -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = []
    for coords in [
        (-1,  phi, 0), (1,  phi, 0), (-1, -phi, 0), (1, -phi, 0),
        (0, -1,  phi), (0, 1,  phi), (0, -1, -phi), (0, 1, -phi),
        ( phi, 0, -1), ( phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
    ]:
        v.append(np.array(coords, dtype=float))
    vertices = np.stack(v, axis=0)
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    return vertices, faces


def _subdivide(vertices: np.ndarray, faces: List[Tuple[int, int, int]]) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    midpoint_cache: dict[Tuple[int, int], int] = {}
    verts = vertices.tolist()
    new_faces: List[Tuple[int, int, int]] = []

    def midpoint(i: int, j: int) -> int:
        key = (i, j) if i < j else (j, i)
        if key in midpoint_cache:
            return midpoint_cache[key]
        m = (vertices[i] + vertices[j]) * 0.5
        m /= np.linalg.norm(m) + 1e-12
        verts.append(m.tolist())
        idx = len(verts) - 1
        midpoint_cache[key] = idx
        return idx

    for (i, j, k) in faces:
        a = midpoint(i, j)
        b = midpoint(j, k)
        c = midpoint(k, i)
        new_faces.extend([(i, a, c), (a, j, b), (c, b, k), (a, b, c)])

    return np.asarray(verts, dtype=float), new_faces


def make_sphere(subdivisions: int = 2, radius: float = 1.0) -> Shape:
    v, f = _icosahedron()
    for _ in range(max(0, subdivisions)):
        v, f = _subdivide(v, f)
    v = (v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)) * radius
    return Shape.from_mesh(v.tolist(), [list(t) for t in f], filled=True)


def make_two_spheres(subdivisions: int = 1, radius: float = 0.9, separation: float = 2.5) -> Shape:
    left = make_sphere(subdivisions=subdivisions, radius=radius)
    right = make_sphere(subdivisions=subdivisions, radius=radius)
    vl = np.asarray(left.vertices(), dtype=float)
    vr = np.asarray(right.vertices(), dtype=float)
    cl = left.cells() or []
    cr = right.cells() or []
    vl[:, 0] -= separation * 0.5
    vr[:, 0] += separation * 0.5
    v = np.vstack([vl, vr])
    off = vl.shape[0]
    faces = (cl + [[i + off for i in tri] for tri in cr])
    return Shape.from_mesh(v.tolist(), faces, filled=True)


def make_torus(R: float = 1.5, r: float = 0.5, nu: int = 64, nv: int = 32) -> Shape:
    # Parametric torus grid triangulated
    us = np.linspace(0.0, 2.0 * np.pi, num=nu, endpoint=False)
    vs = np.linspace(0.0, 2.0 * np.pi, num=nv, endpoint=False)
    vertices: List[Tuple[float, float, float]] = []
    for v in vs:
        for u in us:
            x = (R + r * np.cos(v)) * np.cos(u)
            y = (R + r * np.cos(v)) * np.sin(u)
            z = r * np.sin(v)
            vertices.append((x, y, z))
    vertices_np = np.asarray(vertices, dtype=float)

    def idx(i: int, j: int) -> int:
        return (i % nv) * nu + (j % nu)

    faces: List[Tuple[int, int, int]] = []
    for i in range(nv):
        for j in range(nu):
            a = idx(i, j)
            b = idx(i, j + 1)
            c = idx(i + 1, j)
            d = idx(i + 1, j + 1)
            faces.append((a, b, c))
            faces.append((b, d, c))
    return Shape.from_mesh(vertices_np.tolist(), [list(t) for t in faces], filled=True)


def make_trefoil_knot(scale: float = 1.0) -> Shape:
    def param(t: float) -> Sequence[float]:
        u = (t % 1.0) * 2.0 * np.pi
        x = (2 + np.cos(3 * u)) * np.cos(2 * u)
        y = (2 + np.cos(3 * u)) * np.sin(2 * u)
        z = np.sin(3 * u)
        return [scale * x, scale * y, scale * z]

    return Shape.from_continuous(parametric=param, dim=3, period=1.0, filled=False)


# We can use these dicts to generate shapes easily
SHAPES_2D = {
    "line": make_line,
    "circle": make_circle,
    "figure8": make_figure8,
    "two_circles": make_two_circles,
    "spiral": make_spiral,
    "swiss_roll_2d": make_swiss_roll_2d,
}

SHAPES_3D = {
    "sphere": make_sphere,
    "torus": make_torus,
    "two_spheres": make_two_spheres,
    "trefoil_knot": make_trefoil_knot,
}


