from __future__ import annotations

from typing import List, Tuple, Optional, Dict
import math
import numpy as np
import torch
import torch.nn.functional as F
from persim import wasserstein, bottleneck

from .transforms import apply_transform
from ..topology import compute_layer_topology, extract_persistence_stats


def _linf_project(x_adv: torch.Tensor, x_orig: torch.Tensor, eps: float) -> torch.Tensor:
    eta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
    return x_orig + eta


def _l2_project(x_adv: torch.Tensor, x_orig: torch.Tensor, eps: float) -> torch.Tensor:
    delta = x_adv - x_orig
    norm = torch.norm(delta.view(delta.size(0), -1), p=2, dim=1, keepdim=True) + 1e-12
    factor = torch.clamp(eps / norm, max=1.0).view(-1, 1, 1)
    return x_orig + delta * factor


def pgd_attack(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    norm: str,
    eps: float,
    steps: int,
    step_frac: float = 1.0,
    random_start: bool = True,
) -> torch.Tensor:
    """
    PGD attack in L_inf or L2.
    x: (1, N, 3), y: scalar tensor
    """
    x_orig = x.detach()
    x_adv = x_orig.clone()
    if random_start:
        if norm == "linf":
            x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
        elif norm == "l2":
            noise = torch.randn_like(x_orig)
            noise = noise / (torch.norm(noise.view(1, -1), p=2, dim=1, keepdim=True) + 1e-12).view(1, 1, 1)
            x_adv = x_orig + eps * noise
    x_adv.requires_grad_(True)

    alpha = step_frac * (eps / max(steps, 1))
    for _ in range(steps):
        logits = model(x_adv, save_layers=False)
        loss = F.cross_entropy(logits, y.unsqueeze(0))
        model.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.zero_()
        loss.backward()
        grad = x_adv.grad
        if norm == "linf":
            x_adv = x_adv + alpha * grad.sign()
            x_adv = _linf_project(x_adv, x_orig, eps)
        elif norm == "l2":
            grad_norm = torch.norm(grad.view(grad.size(0), -1), p=2, dim=1, keepdim=True) + 1e-12
            x_adv = x_adv + alpha * (grad / grad_norm).view_as(x_adv)
            x_adv = _l2_project(x_adv, x_orig, eps)
        else:
            raise ValueError(f"Unsupported norm: {norm}")
        x_adv = x_adv.detach().requires_grad_(True)
    return x_adv.detach()


def estimate_min_eps(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    norm: str,
    eps_max: float,
    steps: int,
    tol: float,
    max_outer: int = 12,
) -> Tuple[Optional[float], Optional[torch.Tensor]]:
    """
    Bisection on epsilon using PGD as inner oracle. Returns (eps_star, x_adv_at_eps_star).
    """
    device = next(model.parameters()).device
    x = x.unsqueeze(0).to(device)
    y = y.to(device)

    with torch.no_grad():
        clean_pred = model(x, save_layers=False).argmax(1).item()
    if clean_pred != int(y.item()):
        return 0.0, x.detach().cpu()

    lo, hi = 0.0, eps_max
    found = None
    x_best = None
    for _ in range(max_outer):
        mid = 0.5 * (lo + hi)
        x_adv = pgd_attack(model, x.clone(), y, norm=norm, eps=mid, steps=steps, step_frac=1.0, random_start=True)
        with torch.no_grad():
            pred = model(x_adv, save_layers=False).argmax(1).item()
        if pred != int(y.item()):
            found = mid
            x_best = x_adv
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return found, (x_best.detach().cpu() if x_best is not None else None)


def robust_accuracy_curve(model, loader, norm: str, eps_values: List[float], steps: int) -> List[float]:
    """
    Compute robust accuracy over an epsilon grid using PGD at each epsilon.
    """
    device = next(model.parameters()).device
    correct_counts = [0 for _ in eps_values]
    total = 0
    model.eval()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        total += x.size(0)
        # run per sample to reuse pgd function
        for i in range(x.size(0)):
            xi = x[i : i + 1]
            yi = y[i]
            for j, eps in enumerate(eps_values):
                x_adv = pgd_attack(model, xi.clone(), yi, norm=norm, eps=eps, steps=steps, step_frac=1.0, random_start=True)
                with torch.no_grad():
                    pred = model(x_adv, save_layers=False).argmax(1).item()
                correct_counts[j] += int(pred == int(yi.item()))
    return [c / max(total, 1) for c in correct_counts]


def find_geometric_flip_threshold(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    family: str,
    param_range: Tuple[float, float],
    tol: float,
    axis: Optional[str] = None,
    max_iter: int = 20,
    jitter_clip: Optional[float] = 0.5,
) -> Tuple[Optional[float], List[Tuple[float, int]]]:
    """
    Find minimal parameter for which prediction flips under a single-parameter transform.
    Returns (threshold, trace[(param, pred)]).
    """
    device = next(model.parameters()).device
    x = x.unsqueeze(0).to(device)
    y = y.to(device)

    # coarse bracket
    a, b = param_range
    grid = np.linspace(a, b, num=7)
    trace = []
    model.eval()
    with torch.no_grad():
        for val in grid:
            x_t = apply_transform(x, family, value=val, axis=axis, jitter_clip=jitter_clip)
            pred = model(x_t, save_layers=False).argmax(1).item()
            trace.append((float(val), int(pred)))
    # find first change from baseline pred
    baseline = trace[0][1]
    bracket = None
    for i in range(1, len(trace)):
        if trace[i][1] != baseline:
            bracket = (trace[i - 1][0], trace[i][0])
            break
    if bracket is None:
        return None, trace

    lo, hi = bracket
    # bisection
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        x_t = apply_transform(x, family, value=mid, axis=axis, jitter_clip=jitter_clip)
        with torch.no_grad():
            pred = model(x_t, save_layers=False).argmax(1).item()
        if pred != baseline:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return hi, trace


def interpolate_clouds(x_a: torch.Tensor, x_b: torch.Tensor, alpha: float, match: str = "index") -> torch.Tensor:
    """
    Interpolate two point clouds by index. For NN matching, a one-time nearest neighbor map can be added later.
    Requires same N for 'index'.
    """
    assert x_a.shape[0] == x_b.shape[0], "interpolate_clouds requires same number of points for 'index' match"
    return (1.0 - alpha) * x_a + alpha * x_b


def search_interpolation_boundary(
    model, x_a: torch.Tensor, x_b: torch.Tensor, y: torch.Tensor, steps: int = 25, tol: float = 1e-3
) -> Optional[float]:
    """
    Find alpha in [0,1] where prediction flips along interpolation path x(alpha).
    """
    device = next(model.parameters()).device
    x_a = x_a.unsqueeze(0).to(device)
    x_b = x_b.unsqueeze(0).to(device)
    y = y.to(device)

    model.eval()
    with torch.no_grad():
        preds = []
        alphas = torch.linspace(0.0, 1.0, steps, device=device)
        for a in alphas:
            x_ab = interpolate_clouds(x_a, x_b, float(a.item()))
            pred = model(x_ab, save_layers=False).argmax(1).item()
            preds.append(int(pred))
    baseline = preds[0]
    bracket = None
    for i in range(1, len(preds)):
        if preds[i] != baseline:
            bracket = (float(alphas[i - 1].item()), float(alphas[i].item()))
            break
    if bracket is None:
        return None
    lo, hi = bracket
    # refine
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        x_ab = interpolate_clouds(x_a, x_b, mid)
        with torch.no_grad():
            pred = model(x_ab, save_layers=False).argmax(1).item()
        if pred != baseline:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return hi


def compare_layer_topology(
    model,
    x_clean: torch.Tensor,
    x_alt: torch.Tensor,
    device: torch.device,
    layers: List[str],
    maxdim: int,
    sample_size: int,
    distances: List[str],
    normalize: str = "none",
    pca_dim: Optional[int] = None,
    bootstrap_repeats: int = 1,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-layer topology stats for clean vs alt, and distances between diagrams.
    Returns dict keyed by layer with stats and distances.
    """
    out: Dict[str, Dict[str, float]] = {}
    model.eval()
    with torch.no_grad():
        # clean pass
        _ = model(x_clean.to(device), save_layers=True)
        clean_layers = {k: v for k, v in model.layer_outputs.items() if k in layers}
        # alt pass
        _ = model(x_alt.to(device), save_layers=True)
        alt_layers = {k: v for k, v in model.layer_outputs.items() if k in layers}

    for layer in layers:
        clean_act = clean_layers[layer]
        alt_act = alt_layers[layer]
        dgm_clean = compute_layer_topology(
            clean_act, sample_size=sample_size, maxdim=maxdim, normalize=normalize, pca_dim=pca_dim, bootstrap_repeats=bootstrap_repeats
        )
        dgm_alt = compute_layer_topology(
            alt_act, sample_size=sample_size, maxdim=maxdim, normalize=normalize, pca_dim=pca_dim, bootstrap_repeats=bootstrap_repeats
        )
        if dgm_clean is None or dgm_alt is None:
            continue
        stats_clean = extract_persistence_stats(dgm_clean)
        stats_alt = extract_persistence_stats(dgm_alt)
        entry: Dict[str, float] = {}
        # add stats with prefixes
        for k, v in stats_clean.items():
            entry[f"clean_{k}"] = float(v)
        for k, v in stats_alt.items():
            entry[f"alt_{k}"] = float(v)
        # distances
        for metric in distances:
            for h in range(min(len(dgm_clean), len(dgm_alt))):
                A = dgm_clean[h]
                B = dgm_alt[h]
                A = A[np.isfinite(A[:, 1])]
                B = B[np.isfinite(B[:, 1])]
                if len(A) == 0 and len(B) == 0:
                    dist_val = 0.0
                elif len(A) == 0 or len(B) == 0:
                    dist_val = float("nan")
                else:
                    if metric == "wasserstein":
                        dist_val = float(wasserstein(A, B, matching=False))
                    elif metric == "bottleneck":
                        # persim.bottleneck returns a scalar when matching=False (default),
                        # or a (distance, matching) tuple when matching=True.
                        bn = bottleneck(A, B)  # default matching=False
                        dist_val = float(bn if not isinstance(bn, (tuple, list)) else bn[0])
                    else:
                        raise ValueError(f"Unknown diagram distance metric: {metric}")
                entry[f"{metric}_H{h}"] = dist_val
        out[layer] = entry
    return out


