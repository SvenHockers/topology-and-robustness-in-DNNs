"""
Quick experimentation: find a "good but attackable" classifier per dataset.

Why this exists:
  If the classifier is *too hard to fool*, adversarial examples look like clean
  examples (attack success rate ~0), and detector AUROC collapses toward ~0.5.

This script:
  - trains a small set of candidate model/training configs
  - reports clean accuracy and attack success rates for a small epsilon grid
  - (optional) reports OOD misclassification rates for simple OOD transforms

It does NOT tune detector parameters. It's for freezing a model before the detector sweep.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# Ensure repo root on path when running as a script (python scripts/..).
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import api
from src.data import DatasetBundle
from src.sweep_specs import DATASET_PRESETS, derive_step_size
from src.utils import ExperimentConfig, set_seed


def _subsample_bundle(bundle: DatasetBundle, *, n_train: int, n_val: int, n_test: int, seed: int) -> DatasetBundle:
    rng = np.random.default_rng(int(seed))

    def _take(X: np.ndarray, y: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
        n = int(min(n, len(X)))
        idx = rng.choice(len(X), size=n, replace=False) if n < len(X) else np.arange(len(X))
        return np.asarray(X)[idx], np.asarray(y)[idx]

    Xtr, ytr = _take(bundle.X_train, bundle.y_train, n_train)
    Xva, yva = _take(bundle.X_val, bundle.y_val, n_val)
    Xte, yte = _take(bundle.X_test, bundle.y_test, n_test)
    return DatasetBundle(Xtr, ytr, Xva, yva, Xte, yte, meta=dict(bundle.meta))


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(y_true == y_pred))


def _attack_success_rate(
    model,
    *,
    X: np.ndarray,
    y: np.ndarray,
    X_adv: np.ndarray,
    device: str,
) -> float:
    """
    Success definition (same as api.attack_success_mask):
      correct on clean AND incorrect on adv
    """
    if len(X) == 0:
        return float("nan")
    mask = api.attack_success_mask(model, X, X_adv, y, device=device)
    return float(np.mean(mask))


def _ood_misclass_rate(model, *, X: np.ndarray, y: np.ndarray, X_ood: np.ndarray, device: str) -> float:
    if len(X) == 0:
        return float("nan")
    pred = api.predict(model, X_ood, device=device, return_probs=False)
    return float(np.mean(np.asarray(pred, dtype=int) != np.asarray(y, dtype=int)))


def _candidate_grid(dataset_name: str) -> List[Dict[str, Any]]:
    """
    Small, high-signal candidate sets. Goal: vary *margin/robustness* without destroying accuracy.
    """
    ds = str(dataset_name).lower()

    if ds == "TABULAR":
        return [
            {"name": "mlp_small_light_reg", "model": {"hidden_dims": [64, 32], "weight_decay": 1e-5, "epochs": 80}},
            {"name": "mlp_small_no_reg", "model": {"hidden_dims": [64, 32], "weight_decay": 0.0, "epochs": 80}},
            {"name": "mlp_med_reg", "model": {"hidden_dims": [128, 64], "weight_decay": 1e-4, "epochs": 80}},
            {"name": "mlp_med_no_reg", "model": {"hidden_dims": [128, 64], "weight_decay": 0.0, "epochs": 80}},
            # fewer epochs can reduce margins but still keep decent accuracy
            {"name": "mlp_med_early", "model": {"hidden_dims": [128, 64], "weight_decay": 1e-4, "epochs": 30}},
            # tanh sometimes yields smoother boundaries; include as a probe
            {"name": "mlp_med_tanh", "model": {"hidden_dims": [128, 64], "activation": "tanh", "weight_decay": 1e-4, "epochs": 80}},
            {"name": "mlp_big_reg", "model": {"hidden_dims": [256, 128], "weight_decay": 1e-4, "epochs": 80}},
        ]

    if ds == "VECTOR":
        return [
            {"name": "mlp_small", "model": {"hidden_dims": [64, 32], "weight_decay": 0.0, "epochs": 40}},
            {"name": "mlp_med", "model": {"hidden_dims": [128, 64], "weight_decay": 0.0, "epochs": 40}},
            {"name": "mlp_big", "model": {"hidden_dims": [256, 128], "weight_decay": 0.0, "epochs": 40}},
            {"name": "mlp_med_early", "model": {"hidden_dims": [128, 64], "weight_decay": 0.0, "epochs": 15}},
            {"name": "mlp_med_reg", "model": {"hidden_dims": [128, 64], "weight_decay": 1e-4, "epochs": 40}},
        ]

    if ds == "IMAGE":
        return [
            # With IMAGE subsampling, you usually need ~10-20 epochs to reach "respectable" accuracy.
            {"name": "CNN_feat64_reg", "model_kwargs": {"feat_dim": 64}, "model": {"weight_decay": 1e-4, "epochs": 10}},
            {"name": "CNN_feat128_reg", "model_kwargs": {"feat_dim": 128}, "model": {"weight_decay": 1e-4, "epochs": 10}},
            {"name": "CNN_feat256_reg", "model_kwargs": {"feat_dim": 256}, "model": {"weight_decay": 1e-4, "epochs": 10}},
            # fewer epochs: often still "respectable" but more attackable
            {"name": "CNN_feat128_early", "model_kwargs": {"feat_dim": 128}, "model": {"weight_decay": 1e-4, "epochs": 5}},
            # remove weight decay: can increase confidence; include as probe
            {"name": "CNN_feat128_no_reg", "model_kwargs": {"feat_dim": 128}, "model": {"weight_decay": 0.0, "epochs": 10}},
            # a slightly "better" baseline for reference
            {"name": "CNN_feat128_reg_20ep", "model_kwargs": {"feat_dim": 128}, "model": {"weight_decay": 1e-4, "epochs": 20}},
        ]

    raise KeyError(f"Unknown dataset for candidate grid: {dataset_name!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASET_PRESETS.keys()))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default=None, help="CSV output path (default: results/model_foolability_<dataset>.csv)")
    ap.add_argument(
        "--candidates",
        type=str,
        default=None,
        help="Comma-separated candidate names to run (default: all for dataset).",
    )

    # Attack evaluation settings
    ap.add_argument("--attack-type", type=str, default="pgd", choices=["fgsm", "pgd"])
    ap.add_argument("--eps", type=str, default=None, help="Comma-separated eps list (default: dataset-specific)")
    ap.add_argument("--pgd-steps", type=int, default=30)
    ap.add_argument("--pgd-step-rule", type=str, default="__DERIVE_EPS_OVER_10__", choices=["__DERIVE_EPS_OVER_10__", "__DERIVE_EPS_OVER_5__"])
    ap.add_argument(
        "--n-attack",
        type=int,
        default=0,
        help="If >0, subsample this many points from val/test for attack evaluation (big speedup for images).",
    )

    # Optional speed controls (subsampling)
    ap.add_argument("--n-train", type=int, default=0, help="If >0, subsample training set to this size")
    ap.add_argument("--n-val", type=int, default=0)
    ap.add_argument("--n-test", type=int, default=0)

    # Optional OOD checks
    ap.add_argument("--ood", action="store_true", help="Also compute simple OOD misclassification rates")
    ap.add_argument("--ood-methods", type=str, default=None, help="Comma-separated methods (default: dataset-specific)")
    ap.add_argument("--ood-severities", type=str, default=None, help="Comma-separated severities (default: dataset-specific)")

    args = ap.parse_args()
    ds = args.dataset
    seed = int(args.seed)
    set_seed(seed)

    preset = DATASET_PRESETS[ds]
    cfg = ExperimentConfig.from_dict(preset["cfg"])
    cfg = replace(cfg, seed=seed, device=str(args.device))

    bundle = api.get_dataset(preset["dataset_name"], cfg)
    if args.n_train and args.n_val and args.n_test:
        bundle = _subsample_bundle(bundle, n_train=args.n_train, n_val=args.n_val, n_test=args.n_test, seed=seed)

    # Default eps grids per dataset (in input units)
    if args.eps is not None:
        eps_list = [float(x) for x in args.eps.split(",") if x.strip()]
    else:
        if ds == "TABULAR":
            eps_list = [0.03, 0.10, 0.20]
        elif ds == "VECTOR":
            eps_list = [0.20, 0.30, 0.50]
        elif ds == "IMAGE":
            eps_list = [0.10, 0.20, 0.30]
        else:
            eps_list = [0.10, 0.20]

    if args.ood_methods is not None:
        ood_methods = [m.strip() for m in args.ood_methods.split(",") if m.strip()]
    else:
        ood_methods = ["gaussian_noise", "patch_shuffle"] if ds == "IMAGE" else ["gaussian_noise", "extrapolate", "uniform_wide"]

    if args.ood_severities is not None:
        ood_sevs = [float(x) for x in args.ood_severities.split(",") if x.strip()]
    else:
        ood_sevs = [0.5, 1.0, 2.0] if ds != "IMAGE" else [0.5, 1.0, 2.0]

    candidates = _candidate_grid(ds)
    if args.candidates:
        want = {c.strip() for c in str(args.candidates).split(",") if c.strip()}
        candidates = [c for c in candidates if str(c.get("name", "")).strip() in want]
        if len(candidates) == 0:
            raise ValueError(f"--candidates filtered everything out. Requested: {sorted(want)}")

    out_path = Path(args.out) if args.out else Path("results") / f"model_foolability_{ds}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for cand in candidates:
        # Build candidate-specific cfg and model_kwargs
        cfg_i = cfg
        model_kwargs = dict(preset.get("model_kwargs") or {})

        if "model" in cand:
            md = dict(cfg_i.model.__dict__)
            md.update(cand["model"])
            cfg_i = replace(cfg_i, model=type(cfg_i.model)(**md))

        if "model_kwargs" in cand:
            model_kwargs.update(cand["model_kwargs"])

        # Ensure attack config is set per evaluation (type+steps); epsilon/step_size vary per eps.
        ac_base = dict(cfg_i.attack.__dict__)
        ac_base["attack_type"] = str(args.attack_type)
        ac_base["num_steps"] = int(args.pgd_steps)
        ac_base["random_start"] = True

        # Match api.run_pipeline ergonomics: infer model I/O dims from dataset when possible.
        num_classes = int(bundle.meta.get("num_classes", cfg_i.model.output_dim))
        model_kwargs.setdefault("num_classes", num_classes)
        model_kwargs.setdefault("output_dim", num_classes)  # for MLP factory
        if str(preset["model_name"]).lower() == "MLP" and getattr(bundle.X_train, "ndim", None) == 2:
            model_kwargs.setdefault("input_dim", int(bundle.X_train.shape[1]))

        # Construct + train
        model = api.get_model(preset["model_name"], cfg_i, **(model_kwargs or {}))
        model = api.train(model, bundle, cfg_i, device=str(cfg_i.device), verbose=False, return_history=False)

        # Clean accuracy
        pred_val = api.predict(model, bundle.X_val, device=str(cfg_i.device), return_probs=False)
        pred_test = api.predict(model, bundle.X_test, device=str(cfg_i.device), return_probs=False)
        acc_val = _accuracy(bundle.y_val, pred_val)
        acc_test = _accuracy(bundle.y_test, pred_test)

        row: Dict[str, Any] = {
            "dataset": ds,
            "candidate": cand.get("name", "candidate"),
            "model_kwargs": json.dumps(model_kwargs, sort_keys=True),
            "cfg_model": json.dumps(cfg_i.model.__dict__, sort_keys=True),
            "acc_val": acc_val,
            "acc_test": acc_test,
        }

        # Attacks at multiple eps
        clip = bundle.meta.get("clip", None)
        # Optional subsampling for attack evaluation (especially useful for IMAGE/CIFAR).
        if int(args.n_attack) > 0:
            X_val_a, y_val_a = api.subsample_masked(bundle.X_val, bundle.y_val, np.ones(len(bundle.X_val), dtype=bool), int(args.n_attack), seed=seed + 1)
            X_test_a, y_test_a = api.subsample_masked(bundle.X_test, bundle.y_test, np.ones(len(bundle.X_test), dtype=bool), int(args.n_attack), seed=seed + 2)
        else:
            X_val_a, y_val_a = bundle.X_val, bundle.y_val
            X_test_a, y_test_a = bundle.X_test, bundle.y_test

        for eps in eps_list:
            ac = dict(ac_base)
            ac["epsilon"] = float(eps)
            ac["step_size"] = derive_step_size(float(eps), str(args.pgd_step_rule))
            cfg_eps = replace(cfg_i, attack=type(cfg_i.attack)(**ac))

            X_adv_val = api.generate_adversarial(model, X_val_a, y_val_a, cfg_eps, clip=clip)
            X_adv_test = api.generate_adversarial(model, X_test_a, y_test_a, cfg_eps, clip=clip)

            sr_val = _attack_success_rate(model, X=X_val_a, y=y_val_a, X_adv=X_adv_val, device=str(cfg_i.device))
            sr_test = _attack_success_rate(model, X=X_test_a, y=y_test_a, X_adv=X_adv_test, device=str(cfg_i.device))

            row[f"attack_succ_val_eps{eps:g}"] = sr_val
            row[f"attack_succ_test_eps{eps:g}"] = sr_test

        # Optional OOD
        if args.ood:
            for method in ood_methods:
                for sev in ood_sevs:
                    X_ood = api.generate_ood(
                        bundle.X_test,
                        cfg_i,
                        method=method,
                        severity=float(sev),
                        seed=seed + 123,
                        clip=clip,
                        device=str(cfg_i.device),
                    )
                    mr = _ood_misclass_rate(model, X=bundle.X_test, y=bundle.y_test, X_ood=X_ood, device=str(cfg_i.device))
                    row[f"ood_misclass_{method}_sev{sev:g}"] = mr

        rows.append(row)
        print(
            f"[{ds}] {row['candidate']}: acc_test={acc_test:.3f} | "
            + ", ".join([f"succ@{e:g}={row[f'attack_succ_test_eps{e:g}']:.3f}" for e in eps_list])
        )

    # Write CSV (union of keys)
    all_keys: List[str] = sorted({k for r in rows for k in r.keys()})
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()

