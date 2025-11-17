from __future__ import annotations

from typing import Tuple, Optional
import math
import torch


def rotate_euler(x: torch.Tensor, angles_deg: Tuple[float, float, float]) -> torch.Tensor:
    """Rotate point cloud by Euler angles (degrees) in XYZ order."""
    ax, ay, az = [math.radians(a) for a in angles_deg]
    Rx = torch.tensor([[1, 0, 0],
                       [0, math.cos(ax), -math.sin(ax)],
                       [0, math.sin(ax),  math.cos(ax)]], dtype=x.dtype, device=x.device)
    Ry = torch.tensor([[ math.cos(ay), 0, math.sin(ay)],
                       [0,             1, 0],
                       [-math.sin(ay), 0, math.cos(ay)]], dtype=x.dtype, device=x.device)
    Rz = torch.tensor([[math.cos(az), -math.sin(az), 0],
                       [math.sin(az),  math.cos(az), 0],
                       [0,             0,            1]], dtype=x.dtype, device=x.device)
    R = Rz @ Ry @ Rx
    return x @ R.T


def translate(x: torch.Tensor, t: Tuple[float, float, float]) -> torch.Tensor:
    """Translate point cloud by (tx, ty, tz)."""
    tx, ty, tz = t
    shift = torch.tensor([tx, ty, tz], dtype=x.dtype, device=x.device).view(1, 1, 3)
    return x + shift


def scale(x: torch.Tensor, s: float | Tuple[float, float, float]) -> torch.Tensor:
    """Scale point cloud uniformly (float) or per-axis (sx, sy, sz)."""
    if isinstance(s, tuple):
        s_vec = torch.tensor(s, dtype=x.dtype, device=x.device).view(1, 1, 3)
    else:
        s_vec = torch.tensor([s, s, s], dtype=x.dtype, device=x.device).view(1, 1, 3)
    return x * s_vec


def jitter(x: torch.Tensor, std: float, clip: Optional[float] = None) -> torch.Tensor:
    """Add Gaussian jitter per point; optionally clip."""
    noise = torch.randn_like(x) * std
    if clip is not None:
        noise = torch.clamp(noise, -clip, clip)
    return x + noise


def dropout_points(x: torch.Tensor, ratio: float, keep_size: bool = True) -> torch.Tensor:
    """
    Randomly drop a fraction of points. If keep_size, duplicate remaining points to restore size.
    x: (B?, N, 3) or (N, 3)
    """
    if x.dim() == 2:
        x = x.unsqueeze(0)
        squeeze_back = True
    else:
        squeeze_back = False
    b, n, _ = x.shape
    keep_n = max(1, int(round(n * (1.0 - ratio))))
    idx = torch.rand(b, n, device=x.device).argsort(dim=1)[:, :keep_n]
    batch_idx = torch.arange(b, device=x.device).unsqueeze(-1).expand_as(idx)
    kept = x[batch_idx, idx]  # (B, keep_n, 3)
    if keep_size and keep_n < n:
        # tile and trim to N
        reps = math.ceil(n / keep_n)
        kept_tiled = kept.repeat(1, reps, 1)[:, :n, :]
        out = kept_tiled
    else:
        out = kept
    if squeeze_back:
        out = out.squeeze(0)
    return out


def apply_transform(
    x: torch.Tensor,
    transform: str,
    value: float,
    axis: Optional[str] = None,
    jitter_clip: Optional[float] = None,
    keep_size: bool = True,
) -> torch.Tensor:
    """
    Apply a parameterized transform by name.
    transform: 'rotation' | 'translation' | 'jitter' | 'dropout'
    axis: for rotation/translation in {'x','y','z'} or None for isotropic rotation (uses same angle on all axes)
    """
    if transform == "rotation":
        if axis is None:
            angles = (value, value, value)
        else:
            angles = {"x": (value, 0.0, 0.0), "y": (0.0, value, 0.0), "z": (0.0, 0.0, value)}[axis]
        return rotate_euler(x, angles)
    elif transform == "translation":
        t = {"x": (value, 0.0, 0.0), "y": (0.0, value, 0.0), "z": (0.0, 0.0, value)}[axis or "x"]
        return translate(x, t)
    elif transform == "jitter":
        return jitter(x, std=value, clip=jitter_clip)
    elif transform == "dropout":
        return dropout_points(x, ratio=value, keep_size=keep_size)
    else:
        raise ValueError(f"Unknown transform: {transform}")


