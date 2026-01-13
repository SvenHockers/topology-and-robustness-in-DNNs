"""
Batch runner utilities for executing the repo pipeline across many configs.

Design goals:
  - Mirror the `config/` directory structure under an `outputs/` root.
  - For each config file, create a run folder with a consistent artifact layout.
  - Persist *raw* detector feature arrays (score dict arrays) unmodified.
  - Extract adversarial/OOD "success" counts robustly:
      - Prefer in-memory outputs from `src.api.run_pipeline()`.
      - Fall back to a best-effort shim that parses common artifacts if present.
  - Never let one failed run stop the batch; failures are recorded in metadata + aggregate.

The repo already depends on NumPy; we use it for `.npy` output (default).
Parquet export is optional; if requested but unavailable, we log a warning and continue.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union, cast

import argparse
import csv
import fnmatch
import json
import logging
import os
import subprocess
import sys
import time
import traceback
import warnings

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np  # type: ignore
    from src.types import RunResult  # type: ignore
    from src.utils import ExperimentConfig  # type: ignore


ExportFeatures = Literal["npy", "npy+csv", "npy+parquet"]


@dataclass(frozen=True)
class SuccessCounts:
    adversarial_total: Optional[int]
    adversarial_success: Optional[int]
    ood_total: Optional[int]
    ood_success: Optional[int]
    notes: str


@dataclass(frozen=True)
class RunRow:
    """One row suitable for aggregate CSV/JSON."""

    config_path: str
    output_dir: str
    status: Literal["success", "failed"]
    duration_s: float
    adversarial_total: Optional[int]
    adversarial_success: Optional[int]
    ood_total: Optional[int]
    ood_success: Optional[int]
    notes: str
    error: Optional[str] = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_relpath(path: Path, start: Path) -> str:
    try:
        return str(path.resolve().relative_to(start.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _get_git_commit(repo_root: Path) -> Optional[str]:
    """
    Best-effort git commit detection. Returns None if unavailable.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def _should_ignore(path: Path, ignore_globs: Sequence[str]) -> bool:
    if not ignore_globs:
        return False
    name = path.name
    # Normalize to forward slashes so globs are stable on Windows.
    s = str(path).replace("\\", "/")
    return any(fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(s, pat) for pat in ignore_globs)


def discover_configs(
    subdir_path: Path,
    *,
    extensions: Sequence[str],
    ignore_globs: Sequence[str],
) -> List[Path]:
    """
    Recursively find configuration files in `subdir_path`.
    """
    exts = {("." + e.lstrip(".").lower()) for e in extensions}
    out: List[Path] = []
    for p in subdir_path.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        if _should_ignore(p, ignore_globs):
            continue
        out.append(p)
    out.sort(key=lambda x: str(x).lower())
    return out


def compute_run_dir(
    *,
    config_path: Path,
    config_root: Path,
    output_root: Path,
) -> Path:
    """
    Mirror config structure under `output_root` and strip the config extension.

    Example:
      config/IMAGE/sweeps/run_00.yaml -> outputs/IMAGE/sweeps/run_00/
    """
    rel = config_path.resolve().relative_to(config_root.resolve())
    # rel = <subdir>/.../<file.ext>
    rel_parent = rel.parent
    stem = config_path.stem
    return output_root / rel_parent / stem


def prepare_run_folders(run_dir: Path) -> Dict[str, Path]:
    """
    Create the required output structure for a single run folder.
    """
    images = run_dir / "images"
    raw = run_dir / "raw"
    raw_features = raw / "features"
    metrics = run_dir / "metrics"
    logs = run_dir / "logs"
    for p in (images, raw, raw_features, metrics, logs):
        _ensure_dir(p)
    return {
        "run_dir": run_dir,
        "images": images,
        "raw": raw,
        "raw_features": raw_features,
        "metrics": metrics,
        "logs": logs,
    }


def _setup_run_logger(log_dir: Path, *, verbose: bool) -> logging.Logger:
    """
    Create a per-run logger that writes full logs to `logs/run.log`.
    """
    _ensure_dir(log_dir)
    logger = logging.getLogger(f"run.{log_dir.as_posix()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicating handlers if reusing the same process.
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        fh = logging.FileHandler(str(log_dir / "run.log"), encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        if verbose:
            sh = logging.StreamHandler(stream=sys.stdout)
            sh.setLevel(logging.INFO)
            sh.setFormatter(fmt)
            logger.addHandler(sh)

    return logger


def load_config_any(path: Path) -> ExperimentConfig:
    """
    Load a config from YAML/YML/JSON into ExperimentConfig.

    Notes:
      - YAML uses ExperimentConfig.from_yaml(), which supports `base:` inheritance.
      - JSON is interpreted as ExperimentConfig.from_dict().
    """
    # Import lazily so `--dry-run` works in minimal environments.
    from src.utils import ExperimentConfig  # type: ignore

    def _coerce_scalar(x: Any) -> Any:
        """
        Best-effort coercion for config values that arrive as strings.

        This is primarily to handle YAML 1.1 edge cases where scientific notation like
        `1e-04` may be parsed as a string in some environments, which later breaks
        numeric comparisons inside the pipeline (e.g., `k <= 0` checks).
        """
        if not isinstance(x, str):
            return x
        s = x.strip()
        if s == "":
            return x

        lo = s.lower()
        if lo == "true":
            return True
        if lo == "false":
            return False
        if lo == "null" or lo == "none":
            return None

        # int
        if s.isdigit() or (s.startswith(("+", "-")) and s[1:].isdigit()):
            try:
                return int(s)
            except Exception:
                return x

        # float (incl scientific)
        try:
            # Only accept if float() consumes the string in a numeric way.
            # This will correctly parse "1e-04", "0.2", ".5", etc.
            f = float(s)
            # Guard against strings like "nan" / "inf" accidentally getting through
            # from user configs; keep them as strings unless explicitly needed.
            if lo in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
                return x
            return f
        except Exception:
            return x

    def _coerce_tree(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: _coerce_tree(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_coerce_tree(v) for v in x]
        if isinstance(x, tuple):
            return tuple(_coerce_tree(v) for v in x)
        return _coerce_scalar(x)

    suf = path.suffix.lower()
    if suf in {".yaml", ".yml"}:
        cfg = ExperimentConfig.from_yaml(path)
        # Rebuild via dict to coerce any stringly-typed numbers.
        d = cfg.to_dict() if hasattr(cfg, "to_dict") else {}
        return ExperimentConfig.from_dict(_coerce_tree(d))
    if suf == ".json":
        with path.open("r", encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict):
            raise TypeError(f"JSON config must be an object/dict; got {type(d)!r}")
        return ExperimentConfig.from_dict(_coerce_tree(cast(Dict[str, Any], d)))
    raise ValueError(f"Unsupported config extension: {path.suffix}")


def run_pipeline_from_config(
    *,
    config_path: Path,
    output_dir: Path,
    dataset_name: str,
    model_name: str,
    device: Optional[str] = None,
    enable_latex: bool = False,
    eval_only_successful_attacks: bool = True,
    make_plots: bool = True,
    run_ood: Optional[bool] = None,
) -> RunResult:
    """
    Runner-facing pipeline interface.

    This matches the "config_path + output_dir" calling convention used by batch runners.
    The underlying repo pipeline (`src.api.run_pipeline`) does not take an output directory;
    the runner uses `output_dir` for artifact persistence.
    """
    _ = output_dir  # reserved for future: pipeline-side artifact writing
    # Import lazily so `--dry-run` works without third-party deps installed.
    from src.api import run_pipeline  # type: ignore

    # Keep batch runs headless + avoid LaTeX popups (e.g. missing `siunitx.sty`).
    #
    # IMPORTANT: `src.visualization._ensure_style()` defaults to latex=True when not configured.
    # So simply changing matplotlib.rcParams isn't enough; we must configure the repo style
    # to latex=False up-front (unless the user opts in via --enable-latex).
    if not bool(enable_latex):
        os.environ["MPLBACKEND"] = "Agg"
        try:
            import matplotlib  # type: ignore

            matplotlib.rcParams["text.usetex"] = False
        except Exception:
            pass

        try:
            from src.visualization import configure_mpl_style  # type: ignore

            configure_mpl_style(latex=False)
        except Exception:
            # Visualization is optional; pipeline should continue even if unavailable.
            pass

    cfg = load_config_any(config_path)
    if device is not None:
        # Override config. Support "auto" consistently with ExperimentConfig.__post_init__.
        dev = str(device).lower()
        if dev == "auto":
            try:
                import torch  # type: ignore

                cfg.device = "cuda" if bool(torch.cuda.is_available()) else "cpu"
            except Exception:
                cfg.device = "cpu"
        else:
            cfg.device = dev
    # ripser can emit a noisy warning when n_features > n_points, which can happen
    # in feature-space topology experiments (e.g., tabular/image embeddings).
    # This warning is not actionable during batch runs; suppress it for cleaner CLI logs.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*more columns than rows; did you mean to transpose.*",
            category=UserWarning,
            module=r"ripser\..*",
        )
        return run_pipeline(
            dataset_name=str(dataset_name),
            model_name=str(model_name),
            cfg=cfg,
            eval_only_successful_attacks=bool(eval_only_successful_attacks),
            make_plots=bool(make_plots),
            run_ood=run_ood,
        )


def _try_save_plot(obj: Any, out_path: Path, logger: logging.Logger) -> bool:
    """
    Best-effort: save matplotlib-like figures without importing matplotlib directly.
    """
    try:
        savefig = getattr(obj, "savefig", None)
        if callable(savefig):
            savefig(str(out_path), dpi=200, bbox_inches="tight")
            return True
    except Exception as e:
        logger.info("Plot save failed for %s: %s", out_path.name, repr(e))
    return False


def _save_score_dict_as_npy(
    scores: Dict[str, Any],
    *,
    prefix: str,
    out_dir: Path,
    export: ExportFeatures,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """
    Persist every array in the score dict to individual `.npy` files (unmodified).

    Returns metadata entries for each exported file.
    """
    import numpy as np  # type: ignore

    meta: List[Dict[str, Any]] = []
    if not scores:
        return meta

    for k, v in scores.items():
        key = str(k)
        arr = np.asarray(v)
        safe_key = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in key)
        npy_path = out_dir / f"{prefix}{safe_key}.npy"
        np.save(str(npy_path), arr, allow_pickle=False)

        entry: Dict[str, Any] = {
            "key": key,
            "file": str(npy_path),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "count": int(arr.shape[0]) if arr.ndim >= 1 else int(arr.size),
        }
        meta.append(entry)

        if export in {"npy+csv", "npy+parquet"}:
            # CSV export: best-effort and conservative.
            if export == "npy+csv":
                csv_path = out_dir / f"{prefix}{safe_key}.csv"
                a = arr
                if a.ndim == 0:
                    a2 = a.reshape((1, 1))
                elif a.ndim == 1:
                    a2 = a.reshape((-1, 1))
                elif a.ndim == 2:
                    a2 = a
                else:
                    # Flatten trailing dims to keep row count stable.
                    a2 = a.reshape((a.shape[0], int(np.prod(a.shape[1:]))))
                np.savetxt(str(csv_path), a2, delimiter=",")
                entry["csv"] = str(csv_path)

            if export == "npy+parquet":
                # Optional dependency; never fail the run if missing.
                try:
                    import pyarrow as pa  # type: ignore
                    import pyarrow.parquet as pq  # type: ignore

                    a = arr
                    if a.ndim == 1:
                        cols = {"v": a.astype(float)}
                    elif a.ndim == 2:
                        cols = {f"c{i}": a[:, i] for i in range(a.shape[1])}
                    else:
                        a2 = a.reshape((a.shape[0], int(np.prod(a.shape[1:]))))
                        cols = {f"c{i}": a2[:, i] for i in range(a2.shape[1])}
                    table = pa.table(cols)
                    pq_path = out_dir / f"{prefix}{safe_key}.parquet"
                    pq.write_table(table, str(pq_path))
                    entry["parquet"] = str(pq_path)
                except Exception as e:
                    logger.info("Parquet export unavailable for %s (%s)", key, repr(e))

    return meta


def extract_success_counts_from_result(res: RunResult) -> SuccessCounts:
    """
    Primary path: derive success counts from in-memory pipeline outputs.

    Adversarial:
      - total: number of adversarial test points (`len(res.attack_test.X_adv)`)
      - success: derived from `attack_test.meta`:
          - If `fallback_all_adv=False`: sum(adv_mask)
          - If `fallback_all_adv=True`: round(success_rate * total)

    OOD:
      - available only if `res.eval_ood` exists
      - total: number of OOD-labelled points (label==1)
      - success: number of OOD points with raw_score >= detector.threshold
    """
    import numpy as np  # type: ignore

    notes: List[str] = []

    adv_total: Optional[int] = None
    adv_succ: Optional[int] = None
    if res.attack_test is not None:
        adv_total = int(len(res.attack_test.X_adv))
        meta = res.attack_test.meta or {}
        adv_mask = meta.get("adv_mask", None)
        fallback = bool(meta.get("fallback_all_adv", False))
        succ_rate = meta.get("success_rate", None)
        if isinstance(adv_mask, np.ndarray) and adv_mask.dtype == bool and adv_mask.shape[0] == adv_total and not fallback:
            adv_succ = int(np.sum(adv_mask))
            notes.append("adversarial_success derived from attack_test.meta.adv_mask")
        elif succ_rate is not None and adv_total is not None:
            # When pipeline falls back to evaluating all adv points, it overwrites adv_mask,
            # but preserves the original success_rate.
            try:
                sr = float(succ_rate)
                adv_succ = int(round(sr * adv_total))
                notes.append("adversarial_success derived from attack_test.meta.success_rate (fallback path)")
            except Exception:
                adv_succ = None
                notes.append("adversarial_success unavailable (success_rate unparsable)")
        else:
            notes.append("adversarial_success unavailable (missing adv_mask/success_rate)")
    else:
        notes.append("adversarial_success unavailable (no attack_test in result)")

    ood_total: Optional[int] = None
    ood_succ: Optional[int] = None
    if res.eval_ood is not None:
        labels = np.asarray(res.eval_ood.labels, dtype=int).ravel()
        scores = np.asarray(res.eval_ood.raw_scores, dtype=float).ravel()
        thr = getattr(res.detector, "threshold", None)
        if thr is None:
            notes.append("ood_success unavailable (detector.threshold missing)")
        else:
            thr_f = float(thr)
            ood_mask = labels == 1
            ood_total = int(np.sum(ood_mask))
            ood_succ = int(np.sum((scores >= thr_f) & ood_mask))
            notes.append("ood_success derived from eval_ood labels/raw_scores + detector.threshold")
    else:
        notes.append("ood_success unavailable (no eval_ood in result)")

    return SuccessCounts(
        adversarial_total=adv_total,
        adversarial_success=adv_succ,
        ood_total=ood_total,
        ood_success=ood_succ,
        notes="; ".join(notes),
    )


def extract_success_counts_from_artifacts(run_dir: Path) -> SuccessCounts:
    """
    Fallback shim: attempt to derive success counts from common artifact files.

    Supported (best-effort):
      - raw/records.jsonl (written by these runners)
      - metrics/success_counts.json (if already present)
      - metrics/metrics.json (if it contains obvious fields)
      - predictions.jsonl / results.json / metrics.json (flat fields)

    If unavailable, returns null counts with a note; never raises.
    """
    notes: List[str] = []
    metrics_dir = run_dir / "metrics"
    sc_path = metrics_dir / "success_counts.json"
    if sc_path.exists():
        try:
            with sc_path.open("r", encoding="utf-8") as f:
                d = json.load(f) or {}
            return SuccessCounts(
                adversarial_total=d.get("adversarial_total"),
                adversarial_success=d.get("adversarial_success"),
                ood_total=d.get("ood_total"),
                ood_success=d.get("ood_success"),
                notes=str(d.get("notes") or "loaded from existing success_counts.json"),
            )
        except Exception as e:
            notes.append(f"failed to read existing success_counts.json: {repr(e)}")

    # Minimal heuristics: look for any JSON file with these fields.
    for cand in [
        metrics_dir / "metrics.json",
        run_dir / "results.json",
        run_dir / "metrics.json",
    ]:
        if not cand.exists():
            continue
        try:
            with cand.open("r", encoding="utf-8") as f:
                d = json.load(f) or {}
            keys = {"adversarial_total", "adversarial_success", "ood_total", "ood_success"}
            if isinstance(d, dict) and keys.issubset(set(d.keys())):
                return SuccessCounts(
                    adversarial_total=d.get("adversarial_total"),
                    adversarial_success=d.get("adversarial_success"),
                    ood_total=d.get("ood_total"),
                    ood_success=d.get("ood_success"),
                    notes=f"derived from {cand.name}",
                )
        except Exception as e:
            notes.append(f"failed to parse {cand.name}: {repr(e)}")

    # JSONL heuristic: count booleans if present.
    for jsonl in [run_dir / "raw" / "records.jsonl", run_dir / "predictions.jsonl"]:
        if not jsonl.exists():
            continue
        adv_total = adv_succ = ood_total = ood_succ = 0
        seen_any = False
        try:
            with jsonl.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if not isinstance(rec, dict):
                        continue
                    if "is_adversarial" in rec and "adversarial_success" in rec:
                        seen_any = True
                        if bool(rec["is_adversarial"]):
                            adv_total += 1
                            if bool(rec["adversarial_success"]):
                                adv_succ += 1
                    if "is_ood" in rec and "ood_success" in rec:
                        seen_any = True
                        if bool(rec["is_ood"]):
                            ood_total += 1
                            if bool(rec["ood_success"]):
                                ood_succ += 1
            if seen_any:
                return SuccessCounts(
                    adversarial_total=int(adv_total),
                    adversarial_success=int(adv_succ),
                    ood_total=int(ood_total),
                    ood_success=int(ood_succ),
                    notes=f"derived from {jsonl.name} boolean fields",
                )
        except Exception as e:
            notes.append(f"failed to parse {jsonl.name}: {repr(e)}")

    return SuccessCounts(
        adversarial_total=None,
        adversarial_success=None,
        ood_total=None,
        ood_success=None,
        notes="; ".join(notes) if notes else "success counts unavailable (no recognizable artifacts)",
    )


def write_sample_records_jsonl(run_dir: Path, res: "RunResult", logger: logging.Logger) -> None:
    """
    Emit per-sample records in a stable, parseable format:

      raw/records.jsonl with fields:
        - sample_id
        - is_adversarial
        - adversarial_success
        - is_ood
        - ood_success

    This enables downstream tooling to compute success counts even if metrics formats change.
    """
    import numpy as np  # type: ignore

    out_path = run_dir / "raw" / "records.jsonl"
    _ensure_dir(out_path.parent)

    # Adversarial (test) records
    adv_success_mask: Optional[np.ndarray] = None
    adv_fallback = False
    if res.attack_test is not None:
        meta = res.attack_test.meta or {}
        adv_fallback = bool(meta.get("fallback_all_adv", False))
        m = meta.get("adv_mask", None)
        if isinstance(m, np.ndarray) and m.dtype == bool:
            adv_success_mask = m

    # OOD records (from eval_ood: labels are 0=clean, 1=ood)
    ood_labels: Optional[np.ndarray] = None
    ood_scores: Optional[np.ndarray] = None
    thr = getattr(res.detector, "threshold", None)
    if res.eval_ood is not None and thr is not None:
        ood_labels = np.asarray(res.eval_ood.labels, dtype=int).ravel()
        ood_scores = np.asarray(res.eval_ood.raw_scores, dtype=float).ravel()
        thr = float(thr)

    wrote = 0
    with out_path.open("w", encoding="utf-8") as f:
        if res.attack_test is not None:
            n = int(len(res.attack_test.X_adv))
            for i in range(n):
                adv_succ: Optional[bool]
                if adv_success_mask is not None and (not adv_fallback) and i < int(adv_success_mask.shape[0]):
                    adv_succ = bool(adv_success_mask[i])
                else:
                    adv_succ = None
                rec = {
                    "sample_id": int(i),
                    "split": "test_adv",
                    "is_adversarial": True,
                    "adversarial_success": adv_succ,
                    "is_ood": False,
                    "ood_success": None,
                }
                f.write(json.dumps(rec) + "\n")
                wrote += 1

        if ood_labels is not None and ood_scores is not None and thr is not None:
            for i, (lab, sc) in enumerate(zip(ood_labels.tolist(), ood_scores.tolist())):
                is_ood = bool(int(lab) == 1)
                ood_succ: Optional[bool]
                if is_ood:
                    ood_succ = bool(float(sc) >= float(thr))
                else:
                    ood_succ = None
                rec = {
                    "sample_id": int(i),
                    "split": "test_clean+ood",
                    "is_adversarial": False,
                    "adversarial_success": None,
                    "is_ood": is_ood,
                    "ood_success": ood_succ,
                }
                f.write(json.dumps(rec) + "\n")
                wrote += 1

    logger.info("Wrote %d per-sample records to %s", wrote, str(out_path))

def write_success_counts(run_dir: Path, counts: SuccessCounts) -> None:
    metrics_dir = run_dir / "metrics"
    _ensure_dir(metrics_dir)
    payload = asdict(counts)
    with (metrics_dir / "success_counts.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    with (metrics_dir / "success_counts.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "adversarial_total",
                "adversarial_success",
                "ood_total",
                "ood_success",
                "notes",
            ],
        )
        w.writeheader()
        w.writerow(payload)


def _write_summary_log(run_dir: Path, counts: SuccessCounts) -> None:
    logs_dir = run_dir / "logs"
    _ensure_dir(logs_dir)

    def _frac(a: Optional[int], b: Optional[int]) -> str:
        if a is None or b is None:
            return "unavailable"
        return f"{a}/{b}"

    adv = _frac(counts.adversarial_success, counts.adversarial_total)
    ood = _frac(counts.ood_success, counts.ood_total)
    line = f"adversarial_success={adv}, ood_success={ood}\n"
    with (logs_dir / "summary.log").open("a", encoding="utf-8") as f:
        f.write(line)


def run_one_config(
    *,
    config_path: Path,
    config_root: Path,
    output_root: Path,
    dataset_name: str,
    model_name: str,
    device: Optional[str],
    enable_latex: bool,
    extensions: Sequence[str],
    export_features: ExportFeatures,
    dry_run: bool,
    verbose: bool,
) -> RunRow:
    """
    Execute the pipeline for a single config file and write required artifacts.
    """
    start = time.time()
    run_dir = compute_run_dir(config_path=config_path, config_root=config_root, output_root=output_root)
    folders = prepare_run_folders(run_dir)
    logger = _setup_run_logger(folders["logs"], verbose=verbose)

    repo_root = config_root.parent
    metadata_path = run_dir / "metadata.json"
    meta: Dict[str, Any] = {
        "timestamp_utc": _utc_now_iso(),
        "git_commit": _get_git_commit(repo_root),
        "config_path": str(config_path),
        "config_relpath": _safe_relpath(config_path, config_root),
        "dataset_name": str(dataset_name),
        "model_name": str(model_name),
        "device": None if device is None else str(device),
        "output_dir": str(run_dir),
        "status": "started",
        "duration_s": None,
        "export_features": str(export_features),
    }

    # Always write metadata early so failures still leave a breadcrumb.
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    if dry_run:
        dur = time.time() - start
        meta.update({"status": "success", "duration_s": dur, "dry_run": True})
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        counts = SuccessCounts(None, None, None, None, "dry-run (pipeline not executed)")
        write_success_counts(run_dir, counts)
        _write_summary_log(run_dir, counts)
        return RunRow(
            config_path=str(config_path),
            output_dir=str(run_dir),
            status="success",
            duration_s=float(dur),
            adversarial_total=None,
            adversarial_success=None,
            ood_total=None,
            ood_success=None,
            notes=counts.notes,
            error=None,
        )

    # Progress line (console-friendly).
    print(f"[START] {config_path} -> {run_dir}")
    logger.info("Starting run for config=%s", str(config_path))

    try:
        import numpy as np  # type: ignore
        import math

        # Stage 1: config + pipeline execution
        dev_note = "" if device is None else f", device={device}"
        print(f"[PIPE ] running pipeline (dataset={dataset_name}, model={model_name}{dev_note})")
        logger.info("Stage: run_pipeline (dataset=%s, model=%s, device=%s)", str(dataset_name), str(model_name), str(device))

        # Keep plotting enabled; we save plots generically if returned.
        # Important for a meaningful adversarial-success count:
        # the pipeline computes attack success only when `eval_only_successful_attacks=True`.
        res = run_pipeline_from_config(
            config_path=config_path,
            output_dir=run_dir,
            dataset_name=str(dataset_name),
            model_name=str(model_name),
            device=device,
            enable_latex=bool(enable_latex),
            eval_only_successful_attacks=True,
            make_plots=True,
            run_ood=None,  # follow cfg.ood.enabled unless user changes YAML
        )

        logger.info("Stage: pipeline returned; saving artifacts")

        # Persist raw feature arrays ("feature vectors") to raw/features/
        print(f"[SAVE ] features -> {folders['raw_features']}")
        logger.info("Stage: save feature vectors to %s (export=%s)", str(folders["raw_features"]), str(export_features))
        feature_meta: List[Dict[str, Any]] = []
        feature_meta += _save_score_dict_as_npy(
            res.scores_val_clean,
            prefix="val_clean__",
            out_dir=folders["raw_features"],
            export=export_features,
            logger=logger,
        )
        feature_meta += _save_score_dict_as_npy(
            res.scores_val_adv,
            prefix="val_adv__",
            out_dir=folders["raw_features"],
            export=export_features,
            logger=logger,
        )
        feature_meta += _save_score_dict_as_npy(
            res.scores_test_clean,
            prefix="test_clean__",
            out_dir=folders["raw_features"],
            export=export_features,
            logger=logger,
        )
        feature_meta += _save_score_dict_as_npy(
            res.scores_test_adv,
            prefix="test_adv__",
            out_dir=folders["raw_features"],
            export=export_features,
            logger=logger,
        )
        if res.scores_val_ood is not None:
            feature_meta += _save_score_dict_as_npy(
                res.scores_val_ood,
                prefix="val_ood__",
                out_dir=folders["raw_features"],
                export=export_features,
                logger=logger,
            )
        if res.scores_test_ood is not None:
            feature_meta += _save_score_dict_as_npy(
                res.scores_test_ood,
                prefix="test_ood__",
                out_dir=folders["raw_features"],
                export=export_features,
                logger=logger,
            )

        # Save a compact metrics.json (repo already has richer reporting utilities; we keep it simple).
        print(f"[SAVE ] metrics/logs -> {folders['metrics']} , {folders['logs']}")
        logger.info("Stage: write metrics.json to %s", str(folders["metrics"]))
        metrics_payload: Dict[str, Any] = {
            "config_path": str(config_path),
            "dataset_name": str(dataset_name),
            "model_name": str(model_name),
            "threshold": float(getattr(res.detector, "threshold", math.nan)),
            "metrics_adv": res.eval.metrics,
            "metrics_ood": None if res.eval_ood is None else res.eval_ood.metrics,
        }
        with (folders["metrics"] / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2, sort_keys=True, default=str)

        # Save eval arrays (raw artifacts).
        logger.info("Stage: write eval npz to %s", str(folders["raw"]))
        np.savez_compressed(
            str(folders["raw"] / "eval_adv.npz"),
            labels=np.asarray(res.eval.labels),
            raw_scores=np.asarray(res.eval.raw_scores),
        )
        if res.eval_ood is not None:
            np.savez_compressed(
                str(folders["raw"] / "eval_ood.npz"),
                labels=np.asarray(res.eval_ood.labels),
                raw_scores=np.asarray(res.eval_ood.raw_scores),
            )

        # Save any plots returned by the pipeline to images/
        logger.info("Stage: save plots to %s", str(folders["images"]))
        for k, obj in (res.eval.plots or {}).items():
            if obj is None:
                continue
            _try_save_plot(obj, folders["images"] / f"adv_{k}.png", logger)
        if res.eval_ood is not None:
            for k, obj in (res.eval_ood.plots or {}).items():
                if obj is None:
                    continue
                _try_save_plot(obj, folders["images"] / f"ood_{k}.png", logger)

        # Success counts (primary: from RunResult)
        print("[METR] computing adversarial/OOD success counts")
        logger.info("Stage: compute/write success counts")
        counts = extract_success_counts_from_result(res)
        write_success_counts(run_dir, counts)
        _write_summary_log(run_dir, counts)

        # Per-sample record emission (enables robust "shim" extraction later).
        logger.info("Stage: write per-sample records.jsonl")
        write_sample_records_jsonl(run_dir, res, logger)

        dur = time.time() - start
        meta.update(
            {
                "status": "success",
                "duration_s": dur,
                "feature_files": feature_meta,
                "feature_file_count": int(len(feature_meta)),
                "success_counts": asdict(counts),
            }
        )
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)

        logger.info("Completed run successfully in %.2fs", dur)
        print(f"[DONE ] {config_path} ({dur:.2f}s)")

        return RunRow(
            config_path=str(config_path),
            output_dir=str(run_dir),
            status="success",
            duration_s=float(dur),
            adversarial_total=counts.adversarial_total,
            adversarial_success=counts.adversarial_success,
            ood_total=counts.ood_total,
            ood_success=counts.ood_success,
            notes=counts.notes,
            error=None,
        )

    except Exception as e:
        dur = time.time() - start
        err_txt = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        logger.info("Run failed: %s", err_txt)
        err_path = folders["logs"] / "error.txt"
        print(f"[FAIL] {config_path} ({dur:.2f}s) - {type(e).__name__}: {e}")
        print(f"[FAIL] wrote traceback to {err_path}")

        # Error breadcrumbs
        with err_path.open("w", encoding="utf-8") as f:
            f.write(err_txt)

        counts = extract_success_counts_from_artifacts(run_dir)
        write_success_counts(run_dir, counts)
        _write_summary_log(run_dir, counts)

        meta.update(
            {
                "status": "failed",
                "duration_s": dur,
                "error": {"type": type(e).__name__, "message": str(e)},
                "success_counts": asdict(counts),
            }
        )
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)

        return RunRow(
            config_path=str(config_path),
            output_dir=str(run_dir),
            status="failed",
            duration_s=float(dur),
            adversarial_total=counts.adversarial_total,
            adversarial_success=counts.adversarial_success,
            ood_total=counts.ood_total,
            ood_success=counts.ood_success,
            notes=counts.notes,
            error=f"{type(e).__name__}: {e}",
        )


def run_subdir(
    *,
    subdir: str,
    config_root: Path,
    output_root: Path,
    extensions: Sequence[str],
    ignore_globs: Sequence[str],
    dataset_name: str,
    model_name: str,
    device: Optional[str],
    enable_latex: bool,
    dry_run: bool,
    max_workers: int,
    export_features: ExportFeatures,
    verbose: bool,
) -> List[RunRow]:
    """
    Run all configs under `config_root/<subdir>/`.
    """
    subdir_path = config_root / subdir
    if not subdir_path.is_dir():
        raise FileNotFoundError(f"Config subdir not found: {subdir_path}")

    cfgs = discover_configs(subdir_path, extensions=extensions, ignore_globs=ignore_globs)

    # Create output root eagerly.
    _ensure_dir(output_root)

    if dry_run:
        print(f"[DRY ] subdir={subdir} configs={len(cfgs)}")
        for p in cfgs:
            out_dir = compute_run_dir(config_path=p, config_root=config_root, output_root=output_root)
            print(f"  - {p} -> {out_dir}")

    if max_workers <= 1:
        rows: List[RunRow] = []
        for p in cfgs:
            rows.append(
                run_one_config(
                    config_path=p,
                    config_root=config_root,
                    output_root=output_root,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    device=device,
                    enable_latex=bool(enable_latex),
                    extensions=extensions,
                    export_features=export_features,
                    dry_run=dry_run,
                    verbose=verbose,
                )
            )
        return rows

    # Concurrency: use threads (safe for I/O + avoids Windows process spawn costs).
    from concurrent.futures import ThreadPoolExecutor, as_completed

    rows2: List[RunRow] = []
    with ThreadPoolExecutor(max_workers=int(max_workers)) as ex:
        futs = [
            ex.submit(
                run_one_config,
                config_path=p,
                config_root=config_root,
                output_root=output_root,
                dataset_name=dataset_name,
                model_name=model_name,
                device=device,
                enable_latex=bool(enable_latex),
                extensions=extensions,
                export_features=export_features,
                dry_run=dry_run,
                verbose=verbose,
            )
            for p in cfgs
        ]
        for fut in as_completed(futs):
            rows2.append(fut.result())

    # Make deterministic output ordering for aggregate files.
    rows2.sort(key=lambda r: r.config_path.lower())
    return rows2


def write_aggregate(
    *,
    output_root: Path,
    rows: Sequence[RunRow],
) -> Tuple[Path, Path]:
    """
    Write required aggregate outputs:
      - outputs/_aggregate/aggregate_success_counts.csv
      - outputs/_aggregate/aggregate_success_counts.json
    """
    agg_dir = output_root / "_aggregate"
    _ensure_dir(agg_dir)
    csv_path = agg_dir / "aggregate_success_counts.csv"
    json_path = agg_dir / "aggregate_success_counts.json"

    # Totals (ignore None entries)
    def _sum_opt(vals: Iterable[Optional[int]]) -> Optional[int]:
        xs = [v for v in vals if v is not None]
        return int(sum(xs)) if xs else None

    totals = {
        "adversarial_total": _sum_opt(r.adversarial_total for r in rows),
        "adversarial_success": _sum_opt(r.adversarial_success for r in rows),
        "ood_total": _sum_opt(r.ood_total for r in rows),
        "ood_success": _sum_opt(r.ood_success for r in rows),
        "run_count": int(len(list(rows))),
        "failed_count": int(sum(1 for r in rows if r.status == "failed")),
    }

    # JSON
    payload = {"totals": totals, "runs": [asdict(r) for r in rows], "timestamp_utc": _utc_now_iso()}
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    # CSV (one row per run)
    fieldnames = [
        "config_path",
        "output_dir",
        "status",
        "duration_s",
        "adversarial_total",
        "adversarial_success",
        "ood_total",
        "ood_success",
        "notes",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

    return csv_path, json_path


def build_arg_parser(*, default_config_root: str = "config", default_output_root: str = "outputs") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run pipeline across configs and capture artifacts.")
    p.add_argument("--config-root", default=default_config_root, help="Root config directory (default: config/)")
    p.add_argument("--output-root", default=default_output_root, help="Root outputs directory (default: outputs/)")
    p.add_argument("--extensions", default="yaml,yml,json", help="Comma-separated extensions (default: yaml,yml,json)")
    p.add_argument("--ignore", action="append", default=[], help="Glob pattern to ignore (repeatable)")
    p.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default=None,
        help='Override device for all runs (choices: cpu,cuda,auto). If omitted, uses config value.',
    )
    p.add_argument(
        "--enable-latex",
        action="store_true",
        help="Enable LaTeX rendering for plots (may require a full TeX install, e.g. siunitx).",
    )
    p.add_argument("--dry-run", action="store_true", help="Discover and print, but do not execute pipeline")
    p.add_argument("--max-workers", type=int, default=1, help="Parallel workers (default: 1)")
    p.add_argument(
        "--export-features",
        choices=["npy", "npy+csv", "npy+parquet"],
        default="npy",
        help="Feature export formats (default: npy)",
    )
    # Optional: where the runner writes machine-readable results for master aggregation.
    p.add_argument("--results-json", default=None, help=argparse.SUPPRESS)
    p.add_argument("--verbose", action="store_true", help="Also stream per-run logs to stdout")
    return p


def parse_extensions(ext_csv: str) -> List[str]:
    return [e.strip().lstrip(".") for e in str(ext_csv).split(",") if e.strip()]


def run_and_optionally_emit_results(
    *,
    subdir: str,
    dataset_name: str,
    model_name: str,
    args: argparse.Namespace,
) -> List[RunRow]:
    config_root = Path(str(args.config_root)).resolve()
    output_root = Path(str(args.output_root)).resolve()
    exts = parse_extensions(args.extensions)
    ignore = list(args.ignore or [])
    rows = run_subdir(
        subdir=str(subdir),
        config_root=config_root,
        output_root=output_root,
        extensions=exts,
        ignore_globs=ignore,
        dataset_name=str(dataset_name),
        model_name=str(model_name),
        device=None if getattr(args, "device", None) is None else str(args.device),
        enable_latex=bool(getattr(args, "enable_latex", False)),
        dry_run=bool(args.dry_run),
        max_workers=int(args.max_workers),
        export_features=cast(ExportFeatures, str(args.export_features)),
        verbose=bool(args.verbose),
    )

    # Emit results for a master process (optional).
    if args.results_json:
        out_path = Path(str(args.results_json)).resolve()
        _ensure_dir(out_path.parent)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in rows], f, indent=2, sort_keys=True)

    # Also print JSON to stdout so subprocess callers can consume it.
    print(json.dumps([asdict(r) for r in rows]))
    return rows

