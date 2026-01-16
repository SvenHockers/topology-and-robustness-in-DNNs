from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import random
import time

from optimisers.gp import GaussianProcessSurrogate, GpConfig, propose_by_random_ei
from optimisers.search_space import SearchSpace
from optimisers.study import StudyStore, TrialRecord
from optimisers.runner_bridge import ObjectiveSpec, run_trial_via_runner_lib


@dataclass(frozen=True)
class OptimiserConfig:
    n_trials: int = 30
    n_initial: int = 8
    n_candidates: int = 256
    xi: float = 0.01
    seed: int = 42


def run_gp_optimisation(
    *,
    base_config_path: Path,
    dataset_name: str,
    model_name: str,
    study_dir: Path,
    space: SearchSpace,
    fixed_overrides: Mapping[str, Any],
    objective: ObjectiveSpec,
    opt_cfg: OptimiserConfig,
    gp_cfg: GpConfig,
    device: Optional[str],
    enable_latex: bool,
    export_features: str,
    verbose: bool,
    filter_clean_to_correct: bool,
    max_points_for_scoring: Optional[int],
) -> List[TrialRecord]:
    """
    Run Bayesian optimisation with a GP surrogate.

    Trials are executed through `optimisers.runner_lib` so datasets/models/artifacts match your
    existing runner pipeline.
    """
    store = StudyStore(study_dir)
    store.ensure()

    rng = random.Random(int(opt_cfg.seed))

    trials = store.load()
    next_id = 1 + max((t.trial_id for t in trials), default=0)

    # Gather existing observations (resume support)
    X: List[List[float]] = []
    y: List[float] = []
    for t in trials:
        if t.status == "success" and t.objective_value is not None:
            X.append(space.vectorize(t.params))
            y.append(float(t.objective_value))

    # Main loop
    while len(trials) < int(opt_cfg.n_trials):
        trial_id = next_id
        next_id += 1

        # Suggest parameters
        params: Dict[str, Any]
        if len(y) < int(opt_cfg.n_initial) or len(y) < 2:
            params = dict(space.sample_params(rng))
            suggest_note = "random"
        else:
            surrogate = GaussianProcessSurrogate(gp_cfg)
            surrogate.fit(X, y)
            # random candidate pool
            candidates = [space.vectorize(space.sample_params(rng)) for _ in range(int(opt_cfg.n_candidates))]
            best_idx, _best_ei = propose_by_random_ei(
                rng=rng, surrogate=surrogate, candidates=candidates, best_y=max(y), xi=float(opt_cfg.xi)
            )
            params = space.unvectorize(candidates[best_idx])
            suggest_note = "gp_ei"

        # Build overrides dict and execute trial
        # Apply params onto a nested dict using their `path`s
        sampled_overrides: Dict[str, Any] = {}
        space.apply_to_config_dict(sampled_overrides, params)  # type: ignore[arg-type]
        overrides: Dict[str, Any] = _deep_merge_dicts(dict(fixed_overrides), sampled_overrides)

        status: str
        metric_value: Optional[float]
        objective_value: Optional[float]
        run_dir: Optional[str]
        error: Optional[str]
        duration_s: Optional[float]

        status, metric_value, objective_value, run_dir, error, duration_s = run_trial_via_runner_lib(
            trial_id=trial_id,
            base_config_path=base_config_path,
            overrides_config_dict=overrides,
            dataset_name=dataset_name,
            model_name=model_name,
            study_dir=study_dir,
            objective=objective,
            device=device,
            enable_latex=bool(enable_latex),
            export_features=str(export_features),
            verbose=bool(verbose),
            filter_clean_to_correct=bool(filter_clean_to_correct),
            max_points_for_scoring=max_points_for_scoring,
        )

        rec = TrialRecord(
            trial_id=int(trial_id),
            status="success" if status == "success" else "failed",
            params=dict(params),
            metric_value=metric_value,
            objective_value=objective_value,
            run_dir=run_dir,
            notes=f"suggest={suggest_note}",
            error=error,
            duration_s=duration_s,
        )
        store.append(rec)
        trials = store.load()

        # Update training data
        if rec.status == "success" and rec.objective_value is not None:
            X.append(space.vectorize(rec.params))
            y.append(float(rec.objective_value))

    return trials


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_dicts(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out

