from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import json
import time


@dataclass(frozen=True)
class ObjectiveSpec:
    """
    Objective definition based on the runner's metrics.json output.

    metric_path examples:
      - "metrics_adv.roc_auc"
      - "metrics_adv.fpr_at_tpr95"
      - "metrics_ood.roc_auc"

    maximize:
      - True for metrics like roc_auc / pr_auc / f1
      - False for metrics like fpr_at_tpr95 (lower is better)
    """

    metric_path: str
    maximize: bool = True


def run_trial_via_runner_lib(
    *,
    trial_id: int,
    base_config_path: Path,
    overrides_config_dict: Mapping[str, Any],
    dataset_name: str,
    model_name: str,
    study_dir: Path,
    objective: ObjectiveSpec,
    device: Optional[str],
    enable_latex: bool,
    export_features: str,
    verbose: bool,
    filter_clean_to_correct: bool,
    max_points_for_scoring: Optional[int],
) -> Tuple[str, Optional[float], Optional[float], str, Optional[str], float]:
    """
    Materialize a per-trial config and execute it using the runner library.

    Returns:
      (status, metric_value, objective_value, run_dir, error, duration_s)
    """
    from optimisers import runner_lib

    start = time.time()

    # Create a private config root under the study so runner_lib can mirror paths deterministically.
    config_root = study_dir / "configs"
    trials_cfg_dir = config_root / "trials"
    trials_cfg_dir.mkdir(parents=True, exist_ok=True)

    # Build the per-trial config dict (base merged + overrides).
    base_cfg = runner_lib.load_config_any(base_config_path)
    d = base_cfg.to_dict() if hasattr(base_cfg, "to_dict") else {}

    # Deep merge overrides (lightweight).
    merged = _deep_merge_dicts(d, dict(overrides_config_dict))

    trial_cfg_path = trials_cfg_dir / f"trial_{trial_id:06d}.json"
    with trial_cfg_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, sort_keys=True)

    # Output root under the study
    output_root = study_dir / "runs"
    output_root.mkdir(parents=True, exist_ok=True)

    row = runner_lib.run_one_config(
        config_path=trial_cfg_path,
        config_root=config_root,
        output_root=output_root,
        dataset_name=str(dataset_name),
        model_name=str(model_name),
        device=device,
        enable_latex=bool(enable_latex),
        extensions=("json",),
        export_features=str(export_features),  # type: ignore[arg-type]
        dry_run=False,
        verbose=bool(verbose),
        filter_clean_to_correct=bool(filter_clean_to_correct),
        max_points_for_scoring=max_points_for_scoring,
    )

    run_dir = Path(row.output_dir)
    metric_value, objective_value, note = _extract_objective_from_run(run_dir, objective)

    dur = time.time() - start
    status = str(row.status)
    err = row.error
    if status != "success":
        # If run failed, propagate runner error (objective may be None).
        return status, metric_value, objective_value, str(run_dir), err, float(dur)

    # Successful run but missing metric is still considered "failed" for optimisation.
    if metric_value is None or objective_value is None:
        return "failed", metric_value, objective_value, str(run_dir), f"objective metric missing ({note})", float(dur)

    return "success", float(metric_value), float(objective_value), str(run_dir), None, float(dur)


def _extract_objective_from_run(run_dir: Path, objective: ObjectiveSpec) -> Tuple[Optional[float], Optional[float], str]:
    metrics_path = run_dir / "metrics" / "metrics.json"
    if not metrics_path.exists():
        return None, None, "metrics.json missing"
    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            d = json.load(f) or {}
    except Exception as e:
        return None, None, f"metrics.json unreadable: {type(e).__name__}: {e}"

    v = _get_dotted(d, objective.metric_path)
    if v is None:
        return None, None, f"metric_path not found: {objective.metric_path}"
    try:
        mv = float(v)
    except Exception as e:
        return None, None, f"metric not numeric: {objective.metric_path}: {repr(e)}"

    obj = mv if bool(objective.maximize) else -mv
    return mv, float(obj), "ok"


def _get_dotted(d: Mapping[str, Any], dotted: str) -> Any:
    cur: Any = d
    for k in str(dotted).split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_dicts(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out

