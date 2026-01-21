"""
OOD generator
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .utils import OODConfig


def _clip_if_needed(x: np.ndarray, clip: Optional[Tuple[float, float]]) -> np.ndarray:
    if clip is None:
        return x
    lo, hi = float(clip[0]), float(clip[1])
    return np.clip(x, lo, hi)


# -----------------------------------------------------------------------------
# Vector / tabular OOD
# -----------------------------------------------------------------------------


def generate_ood_examples(
    X: np.ndarray,
    *,
    config: OODConfig,
) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"generate_ood_examples expects X.ndim==2; got {X.ndim} with shape {X.shape}")

    seed = int(0 if config.seed is None else config.seed)
    rng = np.random.default_rng(seed)
    method = str(config.method).lower()
    sev = float(config.severity)

    if method in {"feature_shuffle", "shuffle_features", "permute_features"}:
        X_ood = np.array(X, copy=True)
        for j in range(X_ood.shape[1]):
            rng.shuffle(X_ood[:, j])
        return X_ood

    if method in {"gaussian_noise", "noise"}:
        Xf = X.astype(np.float32, copy=False)
        std = Xf.std(axis=0, keepdims=True)
        std = np.where(std > 0, std, 1.0)
        noise = rng.normal(loc=0.0, scale=sev * std, size=Xf.shape).astype(np.float32)
        return (Xf + noise).astype(X.dtype, copy=False)

    if method in {"extrapolate", "extrapolation", "mixup_extrapolate"}:
        n = X.shape[0]
        if n < 2:
            return np.array(X, copy=True)
        idx_i = rng.integers(0, n, size=n)
        idx_j = rng.integers(0, n, size=n)
        x_i = X[idx_i]
        x_j = X[idx_j]
        lam = rng.uniform(1.0, 1.0 + max(sev, 0.0), size=(n, 1)).astype(np.float32)
        Xf = X.astype(np.float32, copy=False)
        out = x_i.astype(np.float32) + lam * (x_i.astype(np.float32) - x_j.astype(np.float32))
        # Keep dtype consistent with input
        return out.astype(Xf.dtype, copy=False)

    if method in {"uniform_wide", "wide_uniform", "box"}:
        Xf = X.astype(np.float32, copy=False)
        lo = Xf.min(axis=0, keepdims=True)
        hi = Xf.max(axis=0, keepdims=True)
        span = hi - lo
        span = np.where(span > 0, span, 1.0)
        lo2 = lo - sev * span
        hi2 = hi + sev * span
        return rng.uniform(low=lo2, high=hi2, size=Xf.shape).astype(Xf.dtype)

    raise ValueError(f"Unknown OOD method for vectors: {config.method!r}")


# -----------------------------------------------------------------------------
# Image OOD
# -----------------------------------------------------------------------------


def _gaussian_kernel2d(kernel_size: int, sigma: float, *, device: torch.device) -> torch.Tensor:
    k = int(kernel_size)
    if k <= 0 or (k % 2) == 0:
        raise ValueError("blur_kernel_size must be a positive odd integer.")
    sig = float(sigma)
    if sig <= 0:
        raise ValueError("blur_sigma must be > 0.")

    ax = torch.arange(k, device=device, dtype=torch.float32) - (k - 1) / 2.0
    g1 = torch.exp(-(ax**2) / (2.0 * (sig**2)))
    g1 = g1 / g1.sum()
    g2 = torch.outer(g1, g1)
    g2 = g2 / g2.sum()
    return g2


def _blur_images_torch(X: torch.Tensor, *, kernel_size: int, sigma: float) -> torch.Tensor:
    """
    X: (N,C,H,W) float tensor
    """
    if X.ndim != 4:
        raise ValueError(f"Expected X.ndim==4, got {X.ndim}")
    n, c, _h, _w = X.shape
    device = X.device
    k2 = _gaussian_kernel2d(int(kernel_size), float(sigma), device=device)  # (k,k)
    w = k2.view(1, 1, *k2.shape).repeat(c, 1, 1, 1)  # (C,1,k,k)
    pad = int(kernel_size) // 2
    return F.conv2d(X, w, bias=None, stride=1, padding=pad, groups=c)


def _patch_shuffle_numpy(X: np.ndarray, *, patch_size: int, seed: int) -> np.ndarray:
    """
    Shuffle non-overlapping patches per image.
    X: (N,C,H,W)
    """
    X = np.asarray(X)
    if X.ndim != 4:
        raise ValueError(f"Expected X.ndim==4, got {X.ndim}")
    n, c, h, w = X.shape
    p = int(patch_size)
    if p <= 0:
        raise ValueError("patch_size must be >= 1")
    if (h % p) != 0 or (w % p) != 0:
        # If dimensions don't divide cleanly, fall back to a safe no-op.
        return np.array(X, copy=True)

    rng = np.random.default_rng(int(seed))
    gh, gw = h // p, w // p
    out = np.empty_like(X)
    for i in range(n):
        patches = X[i].reshape(c, gh, p, gw, p).transpose(1, 3, 0, 2, 4)  # (gh,gw,C,p,p)
        flat = patches.reshape(gh * gw, c, p, p)
        perm = rng.permutation(gh * gw)
        flat2 = flat[perm]
        patches2 = flat2.reshape(gh, gw, c, p, p).transpose(2, 0, 3, 1, 4)  # (C,gh,p,gw,p)
        out[i] = patches2.reshape(c, h, w)
    return out


def generate_ood_examples_images(
    X: np.ndarray,
    *,
    config: OODConfig,
    clip: Optional[Tuple[float, float]] = (0.0, 1.0),
    batch_size: int = 128,
    device: str = "cpu",
) -> np.ndarray:
    """
    Generate OOD-like images from in-distribution images.

    Supported methods:
      - "gaussian_noise": add pixel noise (severity controls std, relative to data std)
      - "salt_pepper": random pixels set to min/max (severity scales probability)
      - "invert": x -> (max+min) - x
      - "blur": gaussian blur (uses torch conv2d)
      - "patch_shuffle": shuffle non-overlapping patches
    """
    X = np.asarray(X)
    if X.ndim != 4:
        raise ValueError(f"generate_ood_examples_images expects X.ndim==4; got {X.ndim} with shape {X.shape}")

    seed = int(0 if config.seed is None else config.seed)
    rng = np.random.default_rng(seed)
    method = str(config.method).lower()
    sev = float(config.severity)

    if method in {"patch_shuffle", "shuffle_patches"}:
        out = _patch_shuffle_numpy(X, patch_size=int(config.patch_size), seed=seed)
        return _clip_if_needed(out, clip)

    # Methods below are easiest in float32 for stability
    Xf = X.astype(np.float32, copy=False)

    if method in {"invert", "inversion"}:
        if clip is None:
            # Infer range from X (robust but can be surprising); prefer explicit clip for images.
            lo, hi = float(Xf.min()), float(Xf.max())
        else:
            lo, hi = float(clip[0]), float(clip[1])
        out = (lo + hi) - Xf
        return _clip_if_needed(out, clip).astype(X.dtype, copy=False)

    if method in {"gaussian_noise", "noise"}:
        # scale relative to the dataset std to make severity roughly comparable across datasets
        base = float(Xf.std()) if float(Xf.std()) > 0 else 1.0
        sigma = sev * 0.25 * base
        noise = rng.normal(loc=0.0, scale=sigma, size=Xf.shape).astype(np.float32)
        out = Xf + noise
        return _clip_if_needed(out, clip).astype(X.dtype, copy=False)

    if method in {"salt_pepper", "saltpepper", "impulse"}:
        p = float(config.saltpepper_p) * max(sev, 0.0)
        p = float(np.clip(p, 0.0, 0.5))
        if clip is None:
            lo, hi = float(Xf.min()), float(Xf.max())
        else:
            lo, hi = float(clip[0]), float(clip[1])
        u = rng.random(size=Xf.shape, dtype=np.float32)
        out = Xf.copy()
        out[u < (p / 2.0)] = lo
        out[(u >= (p / 2.0)) & (u < p)] = hi
        return _clip_if_needed(out, clip).astype(X.dtype, copy=False)

    if method in {"blur", "gaussian_blur"}:
        dev = torch.device(str(device))
        bs = int(max(1, batch_size))
        out_chunks: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(Xf), bs):
                bx = torch.as_tensor(Xf[i:i + bs], device=dev, dtype=torch.float32)
                by = _blur_images_torch(
                    bx,
                    kernel_size=int(config.blur_kernel_size),
                    sigma=float(config.blur_sigma) * max(sev, 1e-6),
                )
                out_chunks.append(by.cpu().numpy())
        out = np.concatenate(out_chunks, axis=0)
        return _clip_if_needed(out, clip).astype(X.dtype, copy=False)

    raise ValueError(f"Unknown OOD method for images: {config.method!r}")

