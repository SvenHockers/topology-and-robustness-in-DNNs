from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from optimisers.gp import GpConfig
from optimisers.gp_optimiser import OptimiserConfig, run_gp_optimisation
from optimisers.runner_bridge import ObjectiveSpec
from optimisers.search_space import specs_from_dict


def _load_any(path: Path) -> Mapping[str, Any]:
    suf = path.suffix.lower()
    if suf in {".json"}:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
        # Allow top-level list for params; wrap.
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


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Gaussian-process Bayesian optimiser for runner configs.")
    p.add_argument("--base-config", required=True, help="Base YAML/JSON config to optimise (e.g. config/IMAGE/base.yaml)")
    p.add_argument("--dataset-name", required=True, help='Dataset name used by src.api.run_pipeline (e.g. "IMAGE")')
    p.add_argument("--model-name", required=True, help='Model name used by src.api.run_pipeline (e.g. "CNN")')
    p.add_argument(
        "--study-dir",
        default="optimiser_outputs/study",
        help="Where to write optimiser outputs (history + runner outputs).",
    )
    p.add_argument(
        "--space",
        required=True,
        help="Search space spec file (.json/.yaml) describing params and bounds.",
    )

    p.add_argument(
        "--metric-path",
        default="metrics_adv.roc_auc",
        help=(
            'Metric dotted path inside run metrics.json (default: "metrics_adv.roc_auc"). '
            'Special value: "auto" chooses metrics_ood.roc_auc when the base config filename starts with "ood_" '
            "or is located under an OOD/ folder; otherwise metrics_adv.roc_auc."
        ),
    )
    p.add_argument("--minimize", action="store_true", help="Minimize the metric instead of maximizing it.")

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
    # Dataset overrides (useful for VECTOR dataset variants).
    p.add_argument(
        "--data-dataset-type",
        default=None,
        help=(
            "Optional override for data.dataset_type (VECTOR pointcloud variants), "
            'e.g. "torus_one_hole", "torus_two_holes", "nested_spheres", "Blobs".'
        ),
    )
    p.add_argument(
        "--data-n-points",
        type=int,
        default=None,
        help="Optional override for data.n_points (VECTOR pointcloud size).",
    )
    p.set_defaults(filter_clean_to_correct=True)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    # Ensure imports resolve when running as a script: `python optimisers/cli.py ...`
    import sys

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    base_config = Path(str(args.base_config)).resolve()
    study_dir = Path(str(args.study_dir)).resolve()
    space_spec = _load_any(Path(str(args.space)).resolve())
    space = specs_from_dict(space_spec)

    metric_path = str(args.metric_path)
    if metric_path.strip().lower() == "auto":
        base_parts = [p.lower() for p in base_config.parts]
        if base_config.name.lower().startswith("ood_") or "ood" in base_parts:
            metric_path = "metrics_ood.roc_auc"
        else:
            metric_path = "metrics_adv.roc_auc"
    objective = ObjectiveSpec(metric_path=str(metric_path), maximize=not bool(args.minimize))

    opt_cfg = OptimiserConfig(
        n_trials=int(args.n_trials),
        n_initial=int(args.n_initial),
        n_candidates=int(args.n_candidates),
        xi=float(args.xi),
        seed=int(args.seed),
    )
    gp_cfg = GpConfig(noise=float(args.gp_noise), n_restarts_optimizer=int(args.gp_restarts), random_state=int(args.seed))

    # Enforce your fixed context assumptions for optimisation.
    fixed_overrides: Dict[str, Any] = {
        "graph": {
            "space": "feature",
            "feature_layer": "penultimate",
            "use_topology": True,
        },
        "detector": {
            "detector_type": "topology_score",
        },
    }
    if args.data_dataset_type is not None or args.data_n_points is not None:
        fixed_overrides["data"] = {}
        if args.data_dataset_type is not None:
            fixed_overrides["data"]["dataset_type"] = str(args.data_dataset_type)
        if args.data_n_points is not None:
            fixed_overrides["data"]["n_points"] = int(args.data_n_points)

    run_gp_optimisation(
        base_config_path=base_config,
        dataset_name=str(args.dataset_name),
        model_name=str(args.model_name),
        study_dir=study_dir,
        space=space,
        fixed_overrides=fixed_overrides,
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


if __name__ == "__main__":
    main()

