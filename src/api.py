"""
Public, stable API for this repository.

Goal: expose a small set of functions/classes so users can run experiments without
copying notebook glue code, while keeping the underlying implementations unchanged.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, cast
import json
import shutil
import warnings

import numpy as np
import torch
import torch.nn as nn

from .adv_attacks import (
    generate_adversarial_examples,
    generate_adversarial_examples_images,
)
from .OOD import generate_ood_examples, generate_ood_examples_images
from .data import DATASET_REGISTRY, DatasetBundle, create_data_loaders
from .detectors import TopologyScoreDetector, predict_graph_detector, train_graph_detector
from .evaluation import evaluate_detector
from .graph_scoring import compute_graph_scores
from .models import (
    HookedFeatureModel,
    CNN,
    TwoMoonsMLP,
    extract_features_batch,
    get_model_logits,
    get_model_predictions,
    get_submodule_by_name,
    train_model,
)
from .utils import ExperimentConfig, OODConfig
from .types import AttackResult, DetectorEvalResult, OODResult, RunResult


# ---------------------------------------------------------------------
# Config API
# ---------------------------------------------------------------------


def load_config(
    name_or_path: str,
    *,
    config_dir: str | Path = "config",
) -> ExperimentConfig:
    """
    Load an ExperimentConfig from a YAML file.

    Convenience wrapper around `ExperimentConfig.from_yaml` that defaults to the repo's
    `config/` directory.

    Args:
        name_or_path:
            - A filename inside `config/` (e.g. 'base.yaml', 'pgd_eps0p05_pca10.yaml')
            - Or a bare name without suffix (e.g. 'base' -> 'config/base.yaml')
            - Or an explicit path to a YAML file.
        config_dir: Directory containing config files (default: 'config').

    Returns:
        ExperimentConfig
    """
    def _find_repo_root() -> Path:
        """
        Best-effort repo root discovery.

        We prefer a directory that contains both `src/` and the requested `config_dir/`,
        and we search upwards from the current working directory first (so notebooks
        can run from `experiments/`), then fall back to the package location.
        """
        cfg_dirname = Path(config_dir).name

        # 1) Search upwards from CWD
        for base in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
            if (base / "src").is_dir() and (base / cfg_dirname).is_dir():
                return base

        # 2) Fall back to where this file lives: <repo>/src/api.py -> <repo>
        return Path(__file__).resolve().parents[1]

    p = Path(str(name_or_path))
    if p.suffix == "":
        p = p.with_suffix(".yaml")

    if not p.is_absolute() and not p.exists():
        root = _find_repo_root()
        p = (root / Path(config_dir) / p).resolve()

    return ExperimentConfig.from_yaml(p)


# ---------------------------------------------------------------------
# Dataset API
# ---------------------------------------------------------------------


def list_datasets() -> list[str]:
    """Return available dataset names from the built-in dataset registry."""
    return sorted(DATASET_REGISTRY.keys())


def get_dataset(
    name: str,
    cfg: Optional[ExperimentConfig] = None,
    **overrides: Any,
) -> DatasetBundle:
    """
    Load a dataset bundle by name.

    Args:
        name: Dataset registry key (see `list_datasets()`).
        cfg: ExperimentConfig (optional). If omitted, a default config is used.
        **overrides: Optional overrides for `cfg.data` (e.g. n_samples, noise, root, download).

    Returns:
        DatasetBundle (numpy arrays + meta).
    """
    if cfg is None:
        cfg = ExperimentConfig()

    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset {name!r}. Available: {list_datasets()}")

    if overrides:
        # Backwards/ergonomic compatibility:
        # - allow `get_dataset(..., overrides={...})` (older notebooks)
        # - allow common uppercase aliases used in notebooks/docs (DOWNLOAD, DATA_ROOT)
        nested = overrides.pop("overrides", None)
        if nested is not None:
            if not isinstance(nested, dict):
                raise TypeError(
                    f"`overrides` must be a dict when provided; got {type(nested)!r}"
                )
            merged = dict(nested)
            merged.update(overrides)  # explicit kwargs win
            overrides = merged

        normalized: Dict[str, Any] = {}
        for k, v in overrides.items():
            ks = str(k)
            ku = ks.upper()
            kl = ks.lower()
            if ku == "DOWNLOAD" or kl == "download":
                normalized["download"] = v
            elif ku in {"DATA_ROOT", "ROOT"} or kl in {"data_root", "root"}:
                normalized["root"] = v
            else:
                normalized[k] = v

        # Avoid mutating caller-owned config; keep it simple and deterministic.
        # NOTE: ExperimentConfig.__post_init__ sets seeds; we preserve cfg.seed.
        data_dict = dict(cfg.data.__dict__)
        data_dict.update(normalized)
        cfg = replace(cfg, data=type(cfg.data)(**data_dict))

    return DATASET_REGISTRY[name].load(cfg)


# ---------------------------------------------------------------------
# Model API
# ---------------------------------------------------------------------


def list_models() -> list[str]:
    """Return built-in model factory names."""
    return sorted(["MLP", "CNN"])


def get_model(
    name: str,
    cfg: Optional[ExperimentConfig] = None,
    **kwargs: Any,
) -> nn.Module:
    """
    Construct a model by name.

    Built-ins:
      - 'MLP' -> TwoMoonsMLP
      - 'CNN' -> CNN

    Args:
        name: Model key (see `list_models()`).
        cfg: ExperimentConfig (optional).
        **kwargs: Model constructor kwargs (e.g. input_dim/output_dim, num_classes, feat_dim, in_channels).
    """
    if cfg is None:
        cfg = ExperimentConfig()

    # Normalize for ergonomic CLI/config usage.
    # We keep list_models() returning uppercase ("MLP"/"CNN") but accept any casing here.
    name = str(name).strip().lower()
    if name == "mlp":
        mc = cast(Any, cfg.model)
        return TwoMoonsMLP(
            input_dim=int(kwargs.get("input_dim", mc.input_dim)),
            hidden_dims=list(kwargs.get("hidden_dims", mc.hidden_dims or [64, 32])),
            output_dim=int(kwargs.get("output_dim", mc.output_dim)),
            activation=str(kwargs.get("activation", mc.activation)),
        )
    if name == "cnn":
        # CNN config is intentionally lightweight; most training hyperparams come from cfg.model.
        mc = cast(Any, cfg.model)
        num_classes = int(kwargs.get("num_classes", mc.output_dim))
        feat_dim = int(kwargs.get("feat_dim", 128))
        in_channels = int(kwargs.get("in_channels", 3))
        return CNN(num_classes=num_classes, feat_dim=feat_dim, in_channels=in_channels)

    raise KeyError(f"Unknown model {name!r}. Available: {list_models()}")


def wrap_feature_model(model: nn.Module, feature_module: str | nn.Module) -> HookedFeatureModel:
    """
    Wrap an arbitrary nn.Module and expose `extract_features()` via a forward hook.

    Args:
        model: Base model.
        feature_module: Module (or dotted path inside `model`) whose output should be treated as features.
    """
    if isinstance(feature_module, str):
        fm = get_submodule_by_name(model, feature_module)
    else:
        fm = feature_module
    return HookedFeatureModel(model, feature_module=fm)


# ---------------------------------------------------------------------
# Training / inference API
# ---------------------------------------------------------------------


def train(
    model: nn.Module,
    bundle: DatasetBundle,
    cfg: ExperimentConfig,
    *,
    device: Optional[str] = None,
    verbose: bool = True,
    return_history: bool = False,
):
    """
    Train a model using the repo's existing training utilities.

    Returns:
        model (and optionally history if return_history=True).
    """
    dev = str(device or cfg.device)
    mc = cast(Any, cfg.model)
    train_loader, val_loader, _test_loader = create_data_loaders(
        bundle.X_train,
        bundle.y_train,
        bundle.X_val,
        bundle.y_val,
        bundle.X_test,
        bundle.y_test,
        batch_size=int(mc.batch_size),
        shuffle_train=True,
    )
    history = train_model(model, train_loader, val_loader, cast(Any, cfg.model), device=dev, verbose=bool(verbose))
    return (model, history) if return_history else model


def predict(
    model: nn.Module,
    X: np.ndarray,
    *,
    device: Optional[str] = None,
    return_probs: bool = False,
) -> np.ndarray:
    """Get model predictions (or probabilities) for inputs using existing helpers."""
    return get_model_predictions(model, X, device=str(device or "cpu"), return_probs=bool(return_probs))


# ---------------------------------------------------------------------
# Attack API
# ---------------------------------------------------------------------


def generate_adversarial(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    cfg: ExperimentConfig,
    *,
    clip: Optional[Tuple[float, float]] = None,
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """
    Generate adversarial examples for either vectors (N,D) or images (N,C,H,W).

    Vector inputs call `generate_adversarial_examples`.
    Image inputs call `generate_adversarial_examples_images` and clamp to a valid range.
    """
    mc = cast(Any, cfg.model)
    ac = cast(Any, cfg.attack)
    bs = int(batch_size or mc.batch_size)
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim == 2:
        return generate_adversarial_examples(model, X, y, ac, device=str(cfg.device), batch_size=bs)

    if X.ndim == 4:
        if clip is None:
            # Default image range used across notebooks.
            clip = (0.0, 1.0)
        clip_min, clip_max = float(clip[0]), float(clip[1])
        return generate_adversarial_examples_images(
            model,
            X,
            y,
            attack_type=str(ac.attack_type),
            epsilon=float(ac.epsilon),
            num_steps=int(ac.num_steps),
            step_size=float(ac.step_size),
            device=str(cfg.device),
            batch_size=bs,
            clip_min=clip_min,
            clip_max=clip_max,
        )

    raise ValueError(f"Unsupported input rank X.ndim={X.ndim}. Expected 2 (vector) or 4 (image).")


def generate_ood(
    X: np.ndarray,
    cfg: Optional[ExperimentConfig] = None,
    *,
    method: Optional[str] = None,
    severity: Optional[float] = None,
    seed: Optional[int] = None,
    clip: Optional[Tuple[float, float]] = None,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
    patch_size: Optional[int] = None,
    blur_kernel_size: Optional[int] = None,
    blur_sigma: Optional[float] = None,
    saltpepper_p: Optional[float] = None,
) -> np.ndarray:
    """
    Generate OOD examples for either vectors (N,D) / point clouds (N,D) or images (N,C,H,W).

    This is intentionally lightweight: it does not require a model; it generates shifted/corrupted
    samples from an existing split to probe OOD sensitivity.
    """
    if cfg is None:
        cfg = ExperimentConfig()

    X = np.asarray(X)
    base = cfg.ood if getattr(cfg, "ood", None) is not None else OODConfig()

    # Resolve config defaults from cfg.ood, then allow explicit overrides.
    resolved_method = str(method if method is not None else base.method)
    resolved_severity = float(severity if severity is not None else base.severity)
    # Seed precedence:
    #   explicit seed arg > cfg.ood.seed > cfg.seed
    s = int(seed if seed is not None else (base.seed if base.seed is not None else cfg.seed))
    resolved_bs = int(batch_size if batch_size is not None else base.batch_size)
    resolved_patch = int(patch_size if patch_size is not None else base.patch_size)
    resolved_bk = int(blur_kernel_size if blur_kernel_size is not None else base.blur_kernel_size)
    resolved_sig = float(blur_sigma if blur_sigma is not None else base.blur_sigma)
    resolved_sp = float(saltpepper_p if saltpepper_p is not None else base.saltpepper_p)

    # OOD.py expects a config object with method/severity/seed + image knobs.
    # (OODConfig also contains `enabled`; generators ignore it.)
    oc = OODConfig(
        enabled=bool(getattr(base, "enabled", False)),
        method=resolved_method,
        severity=resolved_severity,
        seed=s,
        batch_size=resolved_bs,
        patch_size=resolved_patch,
        blur_kernel_size=resolved_bk,
        blur_sigma=resolved_sig,
        saltpepper_p=resolved_sp,
    )

    if X.ndim == 2:
        return generate_ood_examples(X, config=oc)

    if X.ndim == 4:
        if clip is None:
            # Keep consistent with adversarial/image datasets in this repo.
            clip = (0.0, 1.0)
        return generate_ood_examples_images(
            X,
            config=oc,
            clip=clip,
            batch_size=int(resolved_bs),
            device=str(device or cfg.device),
        )

    raise ValueError(f"Unsupported input rank X.ndim={X.ndim}. Expected 2 (vector) or 4 (image).")


# ---------------------------------------------------------------------
# Scoring / detector API
# ---------------------------------------------------------------------


def _softmax_probs(logits: np.ndarray) -> np.ndarray:
    t = torch.as_tensor(logits, dtype=torch.float32)
    return torch.softmax(t, dim=1).cpu().numpy()


def _scalar_f_from_probs(probs: np.ndarray) -> np.ndarray:
    if probs.ndim != 2 or probs.shape[1] < 2:
        raise ValueError(f"Expected probs shape (N,C) with C>=2; got {probs.shape}")
    return probs[:, 1] if probs.shape[1] == 2 else probs.max(axis=1)


def compute_scores(
    X_points: np.ndarray,
    model: nn.Module,
    *,
    bundle: DatasetBundle,
    cfg: ExperimentConfig,
) -> Dict[str, np.ndarray]:
    """
    Compute graph/topology scores for X_points using the training reference set from `bundle`.

    This follows the same logic as notebooks:
      - Z_train is either X_train (input space) or penultimate features (feature space)
      - f_train is a scalar confidence/function derived from model outputs on X_train
    """
    gc = cast(Any, cfg.graph)
    if gc.space == "feature":
        layer = str(getattr(gc, "feature_layer", "penultimate"))
        Z_train = extract_features_batch(model, bundle.X_train, layer=layer, device=str(cfg.device))
    else:
        Z_train = bundle.X_train

    logits_train = get_model_logits(model, bundle.X_train, device=str(cfg.device))
    probs_train = _softmax_probs(logits_train)
    f_train = _scalar_f_from_probs(probs_train)

    return compute_graph_scores(
        X_points=np.asarray(X_points),
        model=model,
        Z_train=np.asarray(Z_train),
        f_train=np.asarray(f_train),
        graph_params=gc,
        device=str(cfg.device),
    )


def fit_detector(
    scores: Dict[str, np.ndarray],
    labels: np.ndarray,
    cfg: ExperimentConfig,
) -> TopologyScoreDetector:
    """Fit/calibrate the detector on (scores, labels) using existing detector training."""
    det = train_graph_detector(scores, np.asarray(labels, dtype=int), cast(Any, cfg.detector))
    if not isinstance(det, TopologyScoreDetector):
        # In this repo, `train_graph_detector` standardizes to TopologyScoreDetector.
        raise TypeError(f"Unexpected detector type: {type(det)}")
    return det


def detect(
    detector: Any,
    scores: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (predictions, probability_proxy) using the existing detector inference helper."""
    return predict_graph_detector(detector, scores)


def evaluate_detection(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate detection performance (AUROC/AUPRC/FPR@95TPR, etc.)."""
    return evaluate_detector(np.asarray(y_true, dtype=int), np.asarray(y_score, dtype=float), threshold=threshold)


# ---------------------------------------------------------------------
# Orchestration helpers (optional convenience; no new algorithms)
# ---------------------------------------------------------------------


def concat_scores(scores_a: Dict[str, np.ndarray], scores_b: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Concatenate two score dicts key-wise, filling missing keys with zeros."""
    keys = sorted(set(scores_a.keys()) | set(scores_b.keys()))
    if len(keys) == 0:
        return {}

    # Choose lengths for zero fill
    any_a = next(iter(scores_a.values())) if scores_a else None
    any_b = next(iter(scores_b.values())) if scores_b else None
    n_a = int(len(any_a)) if any_a is not None else 0
    n_b = int(len(any_b)) if any_b is not None else 0

    out: Dict[str, np.ndarray] = {}
    for k in keys:
        a = scores_a.get(k)
        b = scores_b.get(k)
        if a is None:
            a = np.zeros((n_a,), dtype=float)
        if b is None:
            b = np.zeros((n_b,), dtype=float)
        out[k] = np.concatenate([np.asarray(a, dtype=float), np.asarray(b, dtype=float)], axis=0)
    return out


def subsample_masked(
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    n_max: int,
    *,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    idx = np.where(np.asarray(mask, dtype=bool))[0]
    if len(idx) == 0:
        return X[:0], y[:0]
    if len(idx) > int(n_max):
        idx = rng.choice(idx, size=int(n_max), replace=False)
    return np.asarray(X)[idx], np.asarray(y)[idx]


def attack_success_mask(
    model: nn.Module,
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    y_true: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    """
    Boolean mask for successful attacks:
      model(X_clean) correct AND model(X_adv) incorrect (wrt y_true).
    """
    pred_clean = get_model_predictions(model, X_clean, device=device, return_probs=False)
    pred_adv = get_model_predictions(model, X_adv, device=device, return_probs=False)
    y_true = np.asarray(y_true, dtype=int)
    return (pred_clean == y_true) & (pred_adv != y_true)


def run_pipeline(
    dataset_name: str,
    model_name: str,
    cfg: Optional[ExperimentConfig] = None,
    *,
    dataset_overrides: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    filter_clean_to_correct: bool = False,
    eval_only_successful_attacks: bool = False,
    max_points_for_scoring: Optional[int] = None,
    seed: Optional[int] = None,
    make_plots: bool = True,
    # Optional OOD evaluation
    run_ood: Optional[bool] = None,
    ood_method: Optional[str] = None,
    ood_severity: Optional[float] = None,
    ood_batch_size: Optional[int] = None,
    all_vis: bool = False,
) -> RunResult:
    """
    Run an end-to-end experiment:
      dataset -> train model -> generate attacks -> compute scores -> fit detector -> evaluate.

    This is a convenience wrapper around the API functions above. It does not change
    algorithms; it packages common notebook wiring into one call.
    """
    if cfg is None:
        cfg = ExperimentConfig()
    if dataset_overrides:
        bundle = get_dataset(dataset_name, cfg, **dataset_overrides)
    else:
        bundle = get_dataset(dataset_name, cfg)

    mk = dict(model_kwargs or {})
    # Default num_classes inference from dataset meta when available.
    mc = cast(Any, cfg.model)
    num_classes = int(bundle.meta.get("num_classes", mc.output_dim))
    mk.setdefault("num_classes", num_classes)
    mk.setdefault("output_dim", num_classes)  # for MLP factory

    # Per-dataset ergonomics: infer input_dim for vector/pointcloud/tabular MLPs if caller didn't supply it.
    # This prevents common shape-mismatch errors when sweeping datasets/models.
    #
    # NOTE: `get_model()` normalizes name casing, so we check the lowercased form here.
    if str(model_name).strip().lower() == "mlp":
        if getattr(bundle.X_train, "ndim", None) == 2:
            mk.setdefault("input_dim", int(bundle.X_train.shape[1]))

    # Per-dataset ergonomics: infer input channels for CNNs from the dataset.
    # This prevents common shape-mismatch errors (e.g., MNIST is 1-channel, but CNN defaults to 3).
    if str(model_name).strip().lower() == "cnn":
        if getattr(bundle.X_train, "ndim", None) == 4:
            mk.setdefault("in_channels", int(bundle.X_train.shape[1]))
    model = get_model(model_name, cfg, **mk)

    trained = cast(nn.Module, train(model, bundle, cfg, device=str(cfg.device), verbose=True, return_history=False))

    clip = bundle.meta.get("clip", None)
    X_adv_val = generate_adversarial(trained, bundle.X_val, bundle.y_val, cfg, clip=clip)
    X_adv_test = generate_adversarial(trained, bundle.X_test, bundle.y_test, cfg, clip=clip)

    # Optional filtering/subsampling logic, mirroring notebook toggles.
    s = int(seed if seed is not None else cfg.seed)

    clean_mask_val = np.ones(len(bundle.X_val), dtype=bool)
    clean_mask_test = np.ones(len(bundle.X_test), dtype=bool)
    if filter_clean_to_correct:
        pred_val = get_model_predictions(trained, bundle.X_val, device=str(cfg.device), return_probs=False)
        pred_test = get_model_predictions(trained, bundle.X_test, device=str(cfg.device), return_probs=False)
        clean_mask_val = pred_val == np.asarray(bundle.y_val, dtype=int)
        clean_mask_test = pred_test == np.asarray(bundle.y_test, dtype=int)

    adv_mask_val = np.ones(len(X_adv_val), dtype=bool)
    adv_mask_test = np.ones(len(X_adv_test), dtype=bool)
    if eval_only_successful_attacks:
        adv_mask_val = attack_success_mask(
            trained, bundle.X_val, X_adv_val, bundle.y_val, device=str(cfg.device)
        )
        adv_mask_test = attack_success_mask(
            trained, bundle.X_test, X_adv_test, bundle.y_test, device=str(cfg.device)
        )

    # Robustness: it's common (especially for small eps / small datasets) to have
    # very few or even zero "successful" attacks after filtering. The notebooks
    # typically warn and fall back to evaluating on all adversarial points.
    if filter_clean_to_correct:
        if not np.any(clean_mask_val):
            clean_mask_val = np.ones(len(bundle.X_val), dtype=bool)
        if not np.any(clean_mask_test):
            clean_mask_test = np.ones(len(bundle.X_test), dtype=bool)

    # Track attack success rates + warn on degenerate evaluation sets.
    val_succ_rate = float(np.mean(np.asarray(adv_mask_val, dtype=bool))) if len(adv_mask_val) else 0.0
    test_succ_rate = float(np.mean(np.asarray(adv_mask_test, dtype=bool))) if len(adv_mask_test) else 0.0
    fallback_all_adv_val = False
    fallback_all_adv_test = False

    if eval_only_successful_attacks:
        if not np.any(adv_mask_val):
            fallback_all_adv_val = True
            warnings.warn(
                "No successful VAL attacks after filtering (eval_only_successful_attacks=True). "
                "Falling back to evaluating on ALL adversarial points for VAL. "
                "Detection AUROC may collapse toward ~0.5 if adv≈clean. "
                "Consider increasing attack strength (e.g., switch to PGD / increase epsilon/steps) "
                "or disabling successful-attack-only evaluation.",
                RuntimeWarning,
            )
            adv_mask_val = np.ones(len(X_adv_val), dtype=bool)
        if not np.any(adv_mask_test):
            fallback_all_adv_test = True
            warnings.warn(
                "No successful TEST attacks after filtering (eval_only_successful_attacks=True). "
                "Falling back to evaluating on ALL adversarial points for TEST. "
                "Detection AUROC may collapse toward ~0.5 if adv≈clean. "
                "Consider increasing attack strength (e.g., switch to PGD / increase epsilon/steps) "
                "or disabling successful-attack-only evaluation.",
                RuntimeWarning,
            )
            adv_mask_test = np.ones(len(X_adv_test), dtype=bool)

    X_val_clean_used, y_val_clean_used = bundle.X_val, bundle.y_val
    X_test_clean_used, y_test_clean_used = bundle.X_test, bundle.y_test
    X_val_adv_used, y_val_adv_used = X_adv_val, bundle.y_val
    X_test_adv_used, y_test_adv_used = X_adv_test, bundle.y_test

    if max_points_for_scoring is not None:
        X_val_clean_used, y_val_clean_used = subsample_masked(
            bundle.X_val, bundle.y_val, clean_mask_val, int(max_points_for_scoring), seed=s
        )
        X_test_clean_used, y_test_clean_used = subsample_masked(
            bundle.X_test, bundle.y_test, clean_mask_test, int(max_points_for_scoring), seed=s + 1
        )
        X_val_adv_used, y_val_adv_used = subsample_masked(
            X_adv_val, bundle.y_val, adv_mask_val, int(max_points_for_scoring), seed=s + 2
        )
        X_test_adv_used, y_test_adv_used = subsample_masked(
            X_adv_test, bundle.y_test, adv_mask_test, int(max_points_for_scoring), seed=s + 3
        )

    scores_val_clean = compute_scores(X_val_clean_used, trained, bundle=bundle, cfg=cfg)
    scores_val_adv = compute_scores(X_val_adv_used, trained, bundle=bundle, cfg=cfg)
    scores_test_clean = compute_scores(X_test_clean_used, trained, bundle=bundle, cfg=cfg)
    scores_test_adv = compute_scores(X_test_adv_used, trained, bundle=bundle, cfg=cfg)

    scores_val_all = concat_scores(scores_val_clean, scores_val_adv)
    any_key = next(iter(scores_val_all.keys()))
    labels_val = np.concatenate(
        [np.zeros(len(scores_val_clean[any_key]), dtype=int), np.ones(len(scores_val_adv[any_key]), dtype=int)]
    )

    detector = fit_detector(scores_val_all, labels_val, cfg)

    scores_test_all = concat_scores(scores_test_clean, scores_test_adv)
    any_key_t = next(iter(scores_test_all.keys()))
    labels_test = np.concatenate(
        [np.zeros(len(scores_test_clean[any_key_t]), dtype=int), np.ones(len(scores_test_adv[any_key_t]), dtype=int)]
    )

    raw_scores_test = np.asarray(detector.score(scores_test_all), dtype=float)
    # Include threshold-based binary metrics as well (confusion matrix, etc.).
    thr_val = getattr(detector, "threshold", None)
    if thr_val is None:
        thr_val = detector.threshold
    if thr_val is None:
        raise ValueError("Detector has no calibrated threshold; fit must set detector.threshold.")
    threshold = float(thr_val)
    metrics = evaluate_detector(labels_test, raw_scores_test, threshold=threshold)

    # Optional plots (no display by default; notebooks can display returned figures).
    #
    # IMPORTANT: Some Matplotlib GUI backends (notably MacOSX) can hard-abort the
    # interpreter when used in headless contexts. Since `run_pipeline()` is meant
    # to be usable from scripts/CI as well as notebooks, we skip plot creation in
    # those contexts by default (or when explicitly disabled).
    plots: Dict[str, Any] = {}
    if bool(make_plots):
        try:
            import os
            import matplotlib

            backend = str(matplotlib.get_backend()).lower()
            is_headless = (os.environ.get("DISPLAY", "") == "") and (os.environ.get("MPLBACKEND", "") == "")
            if is_headless and backend in {"macosx"}:
                plots["plot_skipped"] = f"Skipped plotting in headless mode (backend={backend!r})."
            else:
                from .visualization import (
                    plot_confusion_matrix,
                    plot_roc_from_metrics,
                    plot_score_distributions_figure,
                    plot_persistence_diagram,
                    plot_topology_summary_features,
                )
                import matplotlib.pyplot as plt

                roc_fig, roc_ax = plot_roc_from_metrics(metrics, title="ROC curve", show=False, interpolate=True)
                plots["roc_fig"] = roc_fig
                plots["roc_ax"] = roc_ax

                y_pred_test = np.asarray(detector.predict(scores_test_all), dtype=int)
                cm_out = plot_confusion_matrix(labels_test, y_pred=y_pred_test, show=False)
                plots["confusion"] = cm_out
                plots["confusion_fig"] = cm_out.get("fig")
                plots["confusion_axes"] = cm_out.get("axes")

                # Score distributions (clean vs adversarial) using the detector's raw score.
                s_clean = np.asarray(detector.score(scores_test_clean), dtype=float)
                s_adv = np.asarray(detector.score(scores_test_adv), dtype=float)
                dist_fig, dist_ax = plot_score_distributions_figure(
                    s_clean,
                    s_adv,
                    score_name="Detector score",
                    title="Detector score distributions (test)",
                    bins=50,
                    threshold=threshold,
                    show=False,
                )
                plots["score_dist_fig"] = dist_fig
                plots["score_dist_ax"] = dist_ax

                # Optional, richer visualizations for research reporting.
                if bool(all_vis) and bool(getattr(cfg.graph, "use_topology", False)):
                    plots["all_vis"] = {}

                    # Reconstruct training representations and f_train (mirrors compute_scores)
                    gc = cast(Any, cfg.graph)
                    if gc.space == "feature":
                        layer = str(getattr(gc, "feature_layer", "penultimate"))
                        Z_train = extract_features_batch(trained, bundle.X_train, layer=layer, device=str(cfg.device))
                    else:
                        Z_train = bundle.X_train
                    logits_train = get_model_logits(trained, bundle.X_train, device=str(cfg.device))
                    probs_train = _softmax_probs(logits_train)
                    f_train = _scalar_f_from_probs(probs_train)

                    from .graph_scoring import compute_graph_scores_with_diagrams

                    def _one_example_pd(X_point, label: str):
                        feats, diagrams, cloud = compute_graph_scores_with_diagrams(
                            np.asarray(X_point, dtype=float),
                            trained,
                            Z_train=np.asarray(Z_train),
                            f_train=np.asarray(f_train),
                            graph_params=gc,
                            device=str(cfg.device),
                        )
                        pd_figs = []
                        for dim, diag in enumerate(diagrams[:2]):  # only H0/H1 for brevity
                            ax = plot_persistence_diagram(diag, dimension=dim, title=f"{label} H{dim} PD", ax=None)
                            pd_figs.append(ax.figure)
                        bar_ax = plot_topology_summary_features(feats, title=f"{label} topology features")
                        bar_fig = bar_ax.figure
                        return {"features": feats, "diagrams": diagrams, "pd_figs": pd_figs, "bar_fig": bar_fig, "cloud": cloud}

                    if len(X_test_clean_used) > 0:
                        plots["all_vis"]["clean_example"] = _one_example_pd(X_test_clean_used[0], "Clean sample")
                    if len(X_test_adv_used) > 0:
                        plots["all_vis"]["adv_example"] = _one_example_pd(X_test_adv_used[0], "Adversarial sample")
        except Exception as e:
            # Plotting should never break the pipeline; expose the error for debugging.
            plots["plot_error"] = repr(e)

    # -----------------------------------------------------------------
    # Optional: OOD evaluation (separate metrics/plots)
    # -----------------------------------------------------------------
    ood_val: Optional[OODResult] = None
    ood_test: Optional[OODResult] = None
    scores_val_ood: Optional[Dict[str, np.ndarray]] = None
    scores_test_ood: Optional[Dict[str, np.ndarray]] = None
    eval_ood: Optional[DetectorEvalResult] = None

    # If cfg.ood.enabled=True, we run OOD by default (unless user explicitly sets run_ood=False).
    cfg_ood_enabled = bool(getattr(getattr(cfg, "ood", None), "enabled", False))
    run_ood_effective = cfg_ood_enabled if run_ood is None else bool(run_ood)

    if bool(run_ood_effective):
        # When cfg.ood.enabled=True, config controls defaults; function args can still override.
        # Generate OOD samples from the same splits (val/test) to avoid data leakage.
        X_ood_val = generate_ood(
            bundle.X_val,
            cfg,
            method=ood_method,
            severity=ood_severity,
            seed=s + 10_001,
            clip=clip,
            batch_size=ood_batch_size,
        )
        X_ood_test = generate_ood(
            bundle.X_test,
            cfg,
            method=ood_method,
            severity=ood_severity,
            seed=s + 10_002,
            clip=clip,
            batch_size=ood_batch_size,
        )

        # Use the same clean filtering/subsampling knobs as the adversarial path.
        X_val_ood_used, y_val_ood_used = X_ood_val, bundle.y_val
        X_test_ood_used, y_test_ood_used = X_ood_test, bundle.y_test

        if max_points_for_scoring is not None:
            # For OOD, treat all OOD points as eligible (mask=all True).
            ood_mask_val = np.ones(len(X_ood_val), dtype=bool)
            ood_mask_test = np.ones(len(X_ood_test), dtype=bool)
            X_val_ood_used, y_val_ood_used = subsample_masked(
                X_ood_val, bundle.y_val, ood_mask_val, int(max_points_for_scoring), seed=s + 4
            )
            X_test_ood_used, y_test_ood_used = subsample_masked(
                X_ood_test, bundle.y_test, ood_mask_test, int(max_points_for_scoring), seed=s + 5
            )

        scores_val_ood = compute_scores(X_val_ood_used, trained, bundle=bundle, cfg=cfg)
        scores_test_ood = compute_scores(X_test_ood_used, trained, bundle=bundle, cfg=cfg)

        # Evaluate on test: clean vs OOD, using the SAME detector (trained on clean vs adversarial).
        scores_test_ood_all = concat_scores(scores_test_clean, scores_test_ood)
        any_key_ood = next(iter(scores_test_clean.keys()))
        labels_test_ood = np.concatenate(
            [
                np.zeros(len(scores_test_clean[any_key_ood]), dtype=int),
                np.ones(len(scores_test_ood[any_key_ood]), dtype=int),
            ]
        )
        raw_scores_test_ood = np.asarray(detector.score(scores_test_ood_all), dtype=float)
        metrics_ood = evaluate_detector(labels_test_ood, raw_scores_test_ood, threshold=threshold)

        plots_ood: Dict[str, Any] = {}
        if bool(make_plots):
            try:
                import os
                import matplotlib

                backend = str(matplotlib.get_backend()).lower()
                is_headless = (os.environ.get("DISPLAY", "") == "") and (os.environ.get("MPLBACKEND", "") == "")
                if is_headless and backend in {"macosx"}:
                    plots_ood["plot_skipped"] = f"Skipped plotting in headless mode (backend={backend!r})."
                else:
                    from .visualization import (
                        plot_confusion_matrix,
                        plot_roc_from_metrics,
                        plot_score_distributions_figure,
                    )

                    roc_fig, roc_ax = plot_roc_from_metrics(
                        metrics_ood, title="ROC curve (OOD)", show=False, interpolate=True
                    )
                    plots_ood["roc_fig"] = roc_fig
                    plots_ood["roc_ax"] = roc_ax

                    y_pred_ood = (raw_scores_test_ood >= float(threshold)).astype(int)
                    cm_out_ood = plot_confusion_matrix(
                        labels_test_ood, y_pred=y_pred_ood, labels=("clean", "ood"), show=False
                    )
                    plots_ood["confusion"] = cm_out_ood
                    plots_ood["confusion_fig"] = cm_out_ood.get("fig")
                    plots_ood["confusion_axes"] = cm_out_ood.get("axes")

                    s_clean_ood = np.asarray(detector.score(scores_test_clean), dtype=float)
                    s_ood = np.asarray(detector.score(scores_test_ood), dtype=float)
                    dist_fig_ood, dist_ax_ood = plot_score_distributions_figure(
                        s_clean_ood,
                        s_ood,
                        score_name="Detector score",
                        title="Detector score distributions (test: clean vs OOD)",
                        bins=50,
                        threshold=threshold,
                        labels=("clean", "OOD"),
                        show=False,
                    )
                    plots_ood["score_dist_fig"] = dist_fig_ood
                    plots_ood["score_dist_ax"] = dist_ax_ood
            except Exception as e:
                plots_ood["plot_error"] = repr(e)

        ood_val = OODResult(
            X_ood=np.asarray(X_ood_val),
            meta={
                "method": str(ood_method),
                "severity": float(ood_severity),
                "seed": int(s + 10_001),
                "y_ref": np.asarray(bundle.y_val, dtype=int),
            },
        )
        ood_test = OODResult(
            X_ood=np.asarray(X_ood_test),
            meta={
                "method": str(ood_method),
                "severity": float(ood_severity),
                "seed": int(s + 10_002),
                "y_ref": np.asarray(bundle.y_test, dtype=int),
            },
        )
        eval_ood = DetectorEvalResult(
            labels=np.asarray(labels_test_ood, dtype=int),
            raw_scores=np.asarray(raw_scores_test_ood, dtype=float),
            metrics=dict(metrics_ood),
            plots=plots_ood,
        )

    attack_val = AttackResult(
        X_adv=np.asarray(X_adv_val),
        meta={
            "adv_mask": np.asarray(adv_mask_val, dtype=bool),
            "clean_mask": np.asarray(clean_mask_val, dtype=bool),
            "y_true": np.asarray(bundle.y_val, dtype=int),
            "success_rate": float(val_succ_rate),
            "fallback_all_adv": bool(fallback_all_adv_val),
        },
    )
    attack_test = AttackResult(
        X_adv=np.asarray(X_adv_test),
        meta={
            "adv_mask": np.asarray(adv_mask_test, dtype=bool),
            "clean_mask": np.asarray(clean_mask_test, dtype=bool),
            "y_true": np.asarray(bundle.y_test, dtype=int),
            "success_rate": float(test_succ_rate),
            "fallback_all_adv": bool(fallback_all_adv_test),
        },
    )
    eval_out = DetectorEvalResult(
        labels=np.asarray(labels_test, dtype=int),
        raw_scores=np.asarray(raw_scores_test, dtype=float),
        metrics=dict(metrics),
        plots=plots,
    )
    return RunResult(
        cfg=cfg,
        bundle=bundle,
        model=trained,
        detector=detector,
        attack_val=attack_val,
        attack_test=attack_test,
        scores_val_clean=scores_val_clean,
        scores_val_adv=scores_val_adv,
        scores_test_clean=scores_test_clean,
        scores_test_adv=scores_test_adv,
        eval=eval_out,
        ood_val=ood_val,
        ood_test=ood_test,
        scores_val_ood=scores_val_ood,
        scores_test_ood=scores_test_ood,
        eval_ood=eval_ood,
    )


# ---------------------------------------------------------------------
# Batch experiment runner (configs -> outputs)
# ---------------------------------------------------------------------


def run_experiment(
    *,
    dataset_name: str,
    model_name: str,
    config_dir: str | Path = "config",
    output_root: str | Path = "outputs",
    config_glob: str = "*.yaml",
    exclude_prefixes: Iterable[str] = ("base",),
    run_name: Optional[str] = None,
    # Forwarded to `run_pipeline()`
    dataset_overrides: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    filter_clean_to_correct: bool = False,
    eval_only_successful_attacks: bool = False,
    max_points_for_scoring: Optional[int] = None,
    seed: Optional[int] = None,
    # Optional OOD evaluation (forwarded to run_pipeline)
    run_ood: Optional[bool] = None,
    ood_method: Optional[str] = None,
    ood_severity: Optional[float] = None,
    # What to write
    save_npz: bool = True,
    save_metrics_json: bool = True,
    copy_config_yaml: bool = True,
) -> Dict[str, Any]:
    """
    Run `run_pipeline()` for *all* YAML configs in `config_dir`, excluding `base*`,
    and write results into a newly-created output folder.

    Output structure:

      <output_root>/<run_name-or-timestamp>/
        summary.json
        summary.csv
        <config_stem>/
          config.yaml                (copy of the YAML used)
          metrics.json               (JSON-serializable summary)
          eval.npz                   (labels + raw_scores)
          scores_*.npz               (score dicts; key-wise arrays)

    Returns:
        dict with keys:
          - run_dir: Path (string)
          - results: list of per-config summaries (dict)
    """

    def _find_repo_root() -> Path:
        cfg_dirname = Path(config_dir).name
        out_dirname = Path(output_root).name
        for base in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
            if (base / "src").is_dir() and (base / cfg_dirname).is_dir():
                return base
            if (base / "src").is_dir() and (base / out_dirname).is_dir():
                return base
        return Path(__file__).resolve().parents[1]

    def _resolve_under_root(p: str | Path, *, root: Path) -> Path:
        pp = Path(p)
        if pp.is_absolute():
            return pp
        return (root / pp).resolve()

    def _make_run_dir(base: Path, name: str) -> Path:
        base.mkdir(parents=True, exist_ok=True)
        run_dir = base / name
        if not run_dir.exists():
            run_dir.mkdir(parents=False, exist_ok=False)
            return run_dir
        # Avoid collisions by suffixing.
        for i in range(1, 10_000):
            cand = base / f"{name}_{i:04d}"
            if not cand.exists():
                cand.mkdir(parents=False, exist_ok=False)
                return cand
        raise RuntimeError("Could not create a unique run directory name.")

    def _json_safe(x: Any) -> Any:
        # Basic types
        if x is None or isinstance(x, (bool, int, float, str)):
            return x
        # numpy scalars
        if isinstance(x, (np.generic,)):
            return x.item()
        # numpy arrays
        if isinstance(x, np.ndarray):
            return x.tolist()
        # torch tensors
        if torch.is_tensor(x):
            return x.detach().cpu().numpy().tolist()
        # mappings / sequences
        if isinstance(x, dict):
            return {str(k): _json_safe(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_json_safe(v) for v in x]
        # fallback
        return repr(x)

    def _save_score_dict_npz(path: Path, scores: Dict[str, np.ndarray]) -> None:
        arrs = {str(k): np.asarray(v) for k, v in (scores or {}).items()}
        np.savez_compressed(path, **arrs)

    root = _find_repo_root()
    cfg_dir = _resolve_under_root(config_dir, root=root)
    out_root = _resolve_under_root(output_root, root=root)

    if not cfg_dir.is_dir():
        raise FileNotFoundError(f"Config directory not found: {cfg_dir}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dirname = str(run_name or f"run_{ts}")
    run_dir = _make_run_dir(out_root, run_dirname)

    # Discover configs
    exclude_prefixes = tuple(str(p) for p in exclude_prefixes or ())
    cfg_paths = sorted(cfg_dir.glob(str(config_glob)))
    cfg_paths = [
        p
        for p in cfg_paths
        if p.is_file() and not any(p.stem.startswith(pref) for pref in exclude_prefixes)
    ]

    results: list[Dict[str, Any]] = []

    for cfg_path in cfg_paths:
        cfg_name = cfg_path.stem
        exp_dir = run_dir / cfg_name
        exp_dir.mkdir(parents=False, exist_ok=False)

        cfg = ExperimentConfig.from_yaml(cfg_path)
        res = run_pipeline(
            dataset_name=dataset_name,
            model_name=model_name,
            cfg=cfg,
            dataset_overrides=dataset_overrides,
            model_kwargs=model_kwargs,
            filter_clean_to_correct=filter_clean_to_correct,
            eval_only_successful_attacks=eval_only_successful_attacks,
            max_points_for_scoring=max_points_for_scoring,
            seed=seed,
            run_ood=run_ood,
            ood_method=ood_method,
            ood_severity=ood_severity,
        )

        # Copy YAML used (helps reproducibility even if inheritance/base is used).
        if copy_config_yaml:
            shutil.copy2(cfg_path, exp_dir / "config.yaml")

        # Save compact, JSON-safe summary.
        # We intentionally do NOT serialize `res.eval.plots` (matplotlib objects).
        threshold_val = getattr(res.detector, "threshold", None)
        summary = {
            "config_name": cfg_name,
            "config_path": str(cfg_path),
            "dataset_name": str(dataset_name),
            "model_name": str(model_name),
            "seed": int(seed if seed is not None else cfg.seed),
            "threshold": None if threshold_val is None else float(threshold_val),
            "metrics": _json_safe(res.eval.metrics),
            "metrics_ood": None if res.eval_ood is None else _json_safe(res.eval_ood.metrics),
            "cfg": _json_safe(cfg.to_dict()) if hasattr(cfg, "to_dict") else _json_safe(cfg),
        }
        results.append(summary)

        if save_metrics_json:
            with (exp_dir / "metrics.json").open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, sort_keys=True)

        if save_npz:
            np.savez_compressed(
                exp_dir / "eval.npz",
                labels=np.asarray(res.eval.labels),
                raw_scores=np.asarray(res.eval.raw_scores),
            )
            if res.eval_ood is not None:
                np.savez_compressed(
                    exp_dir / "eval_ood.npz",
                    labels=np.asarray(res.eval_ood.labels),
                    raw_scores=np.asarray(res.eval_ood.raw_scores),
                )
            _save_score_dict_npz(exp_dir / "scores_val_clean.npz", res.scores_val_clean)
            _save_score_dict_npz(exp_dir / "scores_val_adv.npz", res.scores_val_adv)
            _save_score_dict_npz(exp_dir / "scores_test_clean.npz", res.scores_test_clean)
            _save_score_dict_npz(exp_dir / "scores_test_adv.npz", res.scores_test_adv)
            if res.scores_val_ood is not None:
                _save_score_dict_npz(exp_dir / "scores_val_ood.npz", res.scores_val_ood)
            if res.scores_test_ood is not None:
                _save_score_dict_npz(exp_dir / "scores_test_ood.npz", res.scores_test_ood)

    # Write run-level summaries
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"run_dir": str(run_dir), "results": results}, f, indent=2, sort_keys=True)

    # Also emit a small CSV for quick comparison.
    # We pick a few common keys when present.
    try:
        import csv

        fieldnames = [
            "config_name",
            "roc_auc",
            "pr_auc",
            "fpr_at_tpr95",
            "roc_auc_ood",
            "pr_auc_ood",
            "fpr_at_tpr95_ood",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "threshold",
        ]
        with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for s in results:
                m = s.get("metrics") or {}
                mo = s.get("metrics_ood") or {}
                w.writerow(
                    {
                        "config_name": s.get("config_name"),
                        "roc_auc": m.get("roc_auc"),
                        "pr_auc": m.get("pr_auc"),
                        "fpr_at_tpr95": m.get("fpr_at_tpr95"),
                        "roc_auc_ood": mo.get("roc_auc"),
                        "pr_auc_ood": mo.get("pr_auc"),
                        "fpr_at_tpr95_ood": mo.get("fpr_at_tpr95"),
                        "accuracy": m.get("accuracy"),
                        "precision": m.get("precision"),
                        "recall": m.get("recall"),
                        "f1": m.get("f1"),
                        "threshold": s.get("threshold"),
                    }
                )
    except Exception:
        # CSV is a convenience; never fail the whole run for it.
        pass

    return {"run_dir": str(run_dir), "results": results}

