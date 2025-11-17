from __future__ import annotations

from typing import Optional, Tuple
import torch


def chamfer_distance(a: torch.Tensor, b: torch.Tensor, squared: bool = True, max_points: Optional[int] = None) -> float:
    """
    Symmetric Chamfer distance between two point sets (N,3) and (M,3).
    Optional subsampling to max_points to bound cost.
    """
    if a.dim() == 3:
        a = a.squeeze(0)
    if b.dim() == 3:
        b = b.squeeze(0)
    if max_points is not None:
        if a.shape[0] > max_points:
            idx = torch.randperm(a.shape[0], device=a.device)[:max_points]
            a = a[idx]
        if b.shape[0] > max_points:
            idx = torch.randperm(b.shape[0], device=b.device)[:max_points]
            b = b[idx]

    # pairwise distances
    a2 = (a * a).sum(dim=1, keepdim=True)  # (N,1)
    b2 = (b * b).sum(dim=1, keepdim=True)  # (M,1)
    # (N,M): ||a||^2 + ||b||^2 - 2 a.b
    d2 = a2 + b2.transpose(0, 1) - 2.0 * (a @ b.transpose(0, 1))
    d2 = torch.clamp(d2, min=0.0)
    if not squared:
        d = torch.sqrt(d2 + 1e-12)
        min_ab = d.min(dim=1)[0].mean()
        min_ba = d.min(dim=0)[0].mean()
        return float((min_ab + min_ba).item())
    else:
        min_ab = d2.min(dim=1)[0].mean()
        min_ba = d2.min(dim=0)[0].mean()
        return float((min_ab + min_ba).item())


def emd_distance(*args, **kwargs) -> float:
    """
    Placeholder for Earth Mover's Distance. Install POT (Python Optimal Transport)
    or a differentiable EMD implementation to enable this feature.
    """
    raise RuntimeError("EMD distance not available. Install POT or a suitable EMD library.")


