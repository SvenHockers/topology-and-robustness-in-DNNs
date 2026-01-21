from __future__ import annotations

"""
Batch CLI Endpoints for the optimiser"""

import argparse
import csv
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from optimisers.gp import GpConfig
from optimisers.gp_optimiser import OptimiserConfig, run_gp_optimisation
from optimisers.runner_bridge import ObjectiveSpec
from optimisers.search_space import specs_from_dict


def _load_any_spec(path: Path) -> Dict[str, Any]:
    suf = path.suffix.lower()
    if suf == ".json":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
        return {"params": obj}
    if suf in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("YAML support requires PyYAML (pip install pyyaml).") from e
        with path.open("r", encoding="utf-8") as f:
            obj = yaml.safe_load(f)
        if isinstance(obj, dict):
            return obj
        return {"params": obj}
    raise ValueError(f"Unsupported spec format: {path.suffix} (use .json/.yaml/.yml)")


def _prune_overrides_where_base_sets_value(overrides: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a copy of `overrides` where any key already present in `base` is removed.
    """
    out: Dict[str, Any] = {}
    for k, v in (overrides or {}).items():
        if k in base:
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                child = _prune_overrides_where_base_sets_value(v, base.get(k) or {})  # type: ignore[arg-type]
                if child:
                    out[k] = child
            else:
                continue
        else:
            out[k] = v
    return out


def _should_ignore(path: Path, *, ignore_globs: Sequence[str], ignore_baseline: bool) -> bool:
    p = str(path).replace("\\", "/")
    if ignore_baseline and "/baseline/" in ("/" + p + "/"):
        return True
    for pat in ignore_globs:
        import fnmatch

        if fnmatch.fnmatch(path.name, pat) or fnmatch.fnmatch(p, pat):
            return True
    return False


def discover_configs(
    config_dir: Path,
    *,
    extensions: Sequence[str],
    ignore_globs: Sequence[str],
    ignore_baseline: bool,
) -> List[Path]:
    exts = {("." + e.lstrip(".").lower()) for e in extensions}
    out: List[Path] = []
    for p in config_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        if _should_ignore(p, ignore_globs=ignore_globs, ignore_baseline=bool(ignore_baseline)):
            continue
        out.append(p)
    out.sort(key=lambda x: str(x).replace("\\", "/").lower())
    return out


def _safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_")


def compute_study_dir(*, output_root: Path, config_dir: Path, config_path: Path) -> Path:
    rel = config_path.resolve().relative_to(config_dir.resolve())
    rel_parent = rel.parent
    stem = _safe_stem(config_path)
    return output_root / config_dir.name / rel_parent / stem


def _best_trial(trials: Sequence[Any]) -> Optional[Dict[str, Any]]:
    best = None
    best_val = -float("inf")
    for t in trials:
        if getattr(t, "status", None) != "success":
            continue
        v = getattr(t, "objective_value", None)
        if v is None:
            continue
        vv = float(v)
        if vv > best_val:
            best_val = vv
            best = t
    return None if best is None else asdict(best)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch GP optimiser over many YAML configs in a directory.")
    p.add_argument("--config-dir", required=True, help="Directory of YAML configs to optimise (recursively).")
    p.add_argument("--dataset-name", required=True, help='Dataset name passed to src.api.run_pipeline (e.g. "TABULAR").')
    p.add_argument("--model-name", required=True, help='Model name passed to src.api.run_pipeline (e.g. "MLP").')
    p.add_argument("--space", required=True, help="Search space spec file (.json/.yaml) used for all studies.")

    p.add_argument(
        "--metric-path",
        default="metrics_adv.roc_auc",
        help=(
            "Objective metric dotted path. "
            'Special value: "auto" chooses metrics_ood.roc_auc for configs named "ood_*.yaml" '
            "or located under an OOD/ folder, otherwise metrics_adv.roc_auc."
        ),
    )
    p.add_argument("--minimize", action="store_true", help="Minimize the metric instead of maximizing it.")

    p.add_argument(
        "--output-root",
        default="optimiser_outputs/final",
        help="Root directory for all batch outputs (default: optimiser_outputs/final).",
    )
    p.add_argument("--overwrite", action="store_true", help="Delete existing per-config study dirs before running.")

    p.add_argument("--extensions", default="yaml,yml", help="Comma-separated config extensions (default: yaml,yml).")
    p.add_argument("--ignore", action="append", default=[], help="Glob to ignore (repeatable).")
    p.add_argument("--ignore-baseline", action="store_true", help='Skip any config under a "baseline/" folder.')
    p.add_argument(
        "--force-fixed-overrides",
        action="store_true",
        help=(
            "Force the batch's fixed graph/detector overrides even if the base YAML sets them. "
            "By default, fixed overrides are only applied when the base YAML does not define the field."
        ),
    )

    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--n-initial", type=int, default=8)
    p.add_argument("--n-candidates", type=int, default=256)
    p.add_argument("--xi", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--gp-noise", type=float, default=1e-6)
    p.add_argument("--gp-restarts", type=int, default=3)

    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default=None)
    p.add_argument("--enable-latex", action="store_true")
    p.add_argument("--export-features", choices=["npy", "npy+csv", "npy+parquet"], default="npy")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no-filter-clean-to-correct", dest="filter_clean_to_correct", action="store_false")
    p.add_argument("--max-points-for-scoring", type=int, default=400)
    p.set_defaults(filter_clean_to_correct=True)

    p.add_argument("--data-dataset-type", default=None, help="Fixed override for data.dataset_type.")
    p.add_argument("--data-n-points", type=int, default=None, help="Fixed override for data.n_points.")

    p.add_argument("--make-plots", action="store_true", help="Run optimisers.plot_history for each study.")

    return p


def _parse_extensions(ext_csv: str) -> List[str]:
    return [e.strip().lstrip(".") for e in str(ext_csv).split(",") if e.strip()]


def _auto_metric_path_for_config(cfg_path: Path) -> str:
    name = cfg_path.name.lower()
    parts = [p.lower() for p in cfg_path.parts]
    if name.startswith("ood_") or "ood" in parts:
        return "metrics_ood.roc_auc"
    return "metrics_adv.roc_auc"


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    config_dir = Path(str(args.config_dir)).resolve()
    if not config_dir.is_dir():
        raise SystemExit(f"--config-dir is not a directory: {config_dir}")

    output_root = Path(str(args.output_root)).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    space_spec = _load_any_spec(Path(str(args.space)).resolve())
    space = specs_from_dict(space_spec)

    opt_cfg = OptimiserConfig(
        n_trials=int(args.n_trials),
        n_initial=int(args.n_initial),
        n_candidates=int(args.n_candidates),
        xi=float(args.xi),
        seed=int(args.seed),
    )
    gp_cfg = GpConfig(noise=float(args.gp_noise), n_restarts_optimizer=int(args.gp_restarts), random_state=int(args.seed))

    fixed_overrides: Dict[str, Any] = {
        "graph": {"space": "feature", "feature_layer": "penultimate", "use_topology": True},
        "detector": {"detector_type": "topology_score"},
    }
    if args.data_dataset_type is not None or args.data_n_points is not None:
        fixed_overrides["data"] = {}
        if args.data_dataset_type is not None:
            fixed_overrides["data"]["dataset_type"] = str(args.data_dataset_type)
        if args.data_n_points is not None:
            fixed_overrides["data"]["n_points"] = int(args.data_n_points)

    cfgs = discover_configs(
        config_dir,
        extensions=_parse_extensions(args.extensions),
        ignore_globs=list(args.ignore or []),
        ignore_baseline=bool(args.ignore_baseline),
    )
    if not cfgs:
        raise SystemExit(f"No configs found under {config_dir} (extensions={args.extensions})")

    agg_dir = output_root / config_dir.name / "_aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, cfg_path in enumerate(cfgs, start=1):
        rel = str(cfg_path.resolve().relative_to(config_dir.resolve())).replace("\\", "/")
        study_dir = compute_study_dir(output_root=output_root, config_dir=config_dir, config_path=cfg_path)

        metric_path = str(args.metric_path)
        if metric_path.strip().lower() == "auto":
            metric_path = _auto_metric_path_for_config(cfg_path)
        objective = ObjectiveSpec(metric_path=metric_path, maximize=not bool(args.minimize))

        if bool(args.overwrite) and study_dir.exists():
            shutil.rmtree(study_dir, ignore_errors=True)

        print(f"[{i:03d}/{len(cfgs):03d}] START {rel} (metric={metric_path})")
        start = time.time()
        status = "success"
        err: Optional[str] = None
        best: Optional[Dict[str, Any]] = None
        try:
            effective_fixed = fixed_overrides
            if not bool(args.force_fixed_overrides):
                from optimisers.runner_lib import load_config_any

                base_cfg = load_config_any(cfg_path)
                base_dict = base_cfg.to_dict() if hasattr(base_cfg, "to_dict") else {}
                effective_fixed = _prune_overrides_where_base_sets_value(dict(fixed_overrides), dict(base_dict))

            trials = run_gp_optimisation(
                base_config_path=cfg_path,
                dataset_name=str(args.dataset_name),
                model_name=str(args.model_name),
                study_dir=study_dir,
                space=space,
                fixed_overrides=effective_fixed,
                objective=objective,
                opt_cfg=opt_cfg,
                gp_cfg=gp_cfg,
                device=None if args.device is None else str(args.device),
                enable_latex=bool(args.enable_latex),
                export_features=str(args.export_features),
                verbose=bool(args.verbose),
                filter_clean_to_correct=bool(args.filter_clean_to_correct),
                max_points_for_scoring=int(args.max_points_for_scoring),
            )
            best = _best_trial(trials)
        except Exception as e:
            status = "failed"
            err = f"{type(e).__name__}: {e}"
        dur = time.time() - start

        if bool(args.make_plots) and status == "success":
            try:
                from optimisers.plot_history import main as plot_main

                plot_main(
                    [
                        "--history",
                        str(study_dir / "history.jsonl"),
                        "--outdir",
                        str(study_dir / "figs"),
                        "--space",
                        str(Path(str(args.space)).resolve()),
                    ]
                )
            except Exception:
                pass

        if status == "success":
            best_val = None if not best else best.get("metric_value")
            print(f"[{i:03d}/{len(cfgs):03d}] DONE  {rel} ({dur:.1f}s) best={best_val}")
        else:
            print(f"[{i:03d}/{len(cfgs):03d}] FAIL  {rel} ({dur:.1f}s) err={err}")

        summary_rows.append(
            {
                "config_path": str(cfg_path),
                "config_relpath": rel,
                "study_dir": str(study_dir),
                "status": status,
                "duration_s": float(dur),
                "best": best,
                "error": err,
            }
        )

        with (agg_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "config_dir": str(config_dir),
                    "dataset_name": str(args.dataset_name),
                    "model_name": str(args.model_name),
                    "metric_path": str(args.metric_path),
                    "minimize": bool(args.minimize),
                    "n_trials": int(args.n_trials),
                    "n_initial": int(args.n_initial),
                    "seed": int(args.seed),
                    "timestamp_unix_s": time.time(),
                    "elapsed_s": float(time.time() - t0),
                    "runs": summary_rows,
                },
                f,
                indent=2,
                sort_keys=True,
            )

    csv_path = agg_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "config_relpath",
                "status",
                "duration_s",
                "study_dir",
                "best_metric_value",
                "best_params_json",
                "error",
            ],
        )
        w.writeheader()
        for r in summary_rows:
            best = r.get("best") or {}
            w.writerow(
                {
                    "config_relpath": r.get("config_relpath"),
                    "status": r.get("status"),
                    "duration_s": r.get("duration_s"),
                    "study_dir": r.get("study_dir"),
                    "best_metric_value": None if not best else best.get("metric_value"),
                    "best_params_json": None if not best else json.dumps(best.get("params", {}), sort_keys=True),
                    "error": r.get("error"),
                }
            )


if __name__ == "__main__":
    main()

