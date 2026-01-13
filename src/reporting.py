"""
Reporting utilities for paper-ready figures/tables.

These functions are thin wrappers that compose existing visualization helpers
and metrics into consistent artifacts for multiple datasets/models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
import csv

import numpy as np

from .evaluation import compute_calibration_metrics
from .visualization import (
    plot_confusion_matrix,
    plot_pr_from_metrics,
    plot_reliability_diagram,
    plot_roc_from_metrics,
    plot_score_distributions_figure,
    plot_topology_feature_shift,
    save_figure,
    set_latex_enabled,
)


@dataclass(frozen=True)
class ReportArtifacts:
    """Structured container for detector report assets."""

    figures: Dict[str, Any]
    figure_paths: Dict[str, str]
    tables: Dict[str, Any]
    table_paths: Dict[str, str]


def _safe_list(x: Any) -> list:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _to_array(x: Any) -> np.ndarray:
    try:
        arr = np.asarray(x)
    except Exception:
        arr = np.array([])
    return arr


def _filter_prefix(scores: Mapping[str, np.ndarray], prefix: str) -> Dict[str, np.ndarray]:
    return {str(k): np.asarray(v) for k, v in (scores or {}).items() if str(k).startswith(str(prefix))}


def _cohens_d(clean: np.ndarray, shifted: np.ndarray) -> float:
    a = np.asarray(clean, dtype=float).ravel()
    b = np.asarray(shifted, dtype=float).ravel()
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
    ma, mb = float(np.mean(a)), float(np.mean(b))
    va, vb = float(np.var(a)), float(np.var(b))
    pooled = float(np.sqrt(0.5 * (va + vb) + 1e-12))
    if pooled <= 0:
        return 0.0
    return (mb - ma) / pooled


def build_detector_report(
    *,
    dataset_name: str,
    model_name: str,
    metrics: Mapping[str, Any],
    labels: np.ndarray,
    scores: np.ndarray,
    probabilities: np.ndarray,
    y_pred: np.ndarray,
    score_clean: np.ndarray,
    score_adv: np.ndarray,
    topology_scores_clean: Dict[str, np.ndarray],
    topology_scores_adv: Dict[str, np.ndarray],
    feature_keys: Sequence[str],
    threshold: Optional[float],
    out_dir: Optional[str | Path] = None,
    prefix: Optional[str] = None,
    use_latex: bool = False,
    save_png_only: bool = True,
) -> ReportArtifacts:
    """
    Build a paper-ready detector report for a single run.
    """
    # Ensure plotting works even when LaTeX is unavailable (CI/headless defaults to False).
    try:
        set_latex_enabled(bool(use_latex))
    except Exception:
        # Fall back to non-LaTeX without failing report generation.
        set_latex_enabled(False)

    out_dir_path = Path(out_dir).expanduser().resolve() if out_dir is not None else None
    if out_dir_path is not None:
        out_dir_path.mkdir(parents=True, exist_ok=True)
    pref = f"{prefix}_" if prefix else ""

    figs: Dict[str, Any] = {}
    fig_paths: Dict[str, str] = {}
    errors: Dict[str, str] = {}

    try:
        roc_fig, roc_ax = plot_roc_from_metrics(metrics, title=f"ROC ({dataset_name}, {model_name})", show=False, interpolate=True)
        figs["roc"] = roc_fig
    except Exception as e:
        errors["roc"] = repr(e)

    try:
        pr_fig, pr_ax = plot_pr_from_metrics(
            metrics,
            title=f"Precision-Recall ({dataset_name}, {model_name})",
            show=False,
            interpolate=True,
        )
        figs["pr"] = pr_fig
    except Exception as e:
        errors["pr"] = repr(e)

    try:
        cm_out = plot_confusion_matrix(labels, y_pred=y_pred, show=False)
        figs["confusion"] = cm_out.get("fig")
    except Exception as e:
        errors["confusion"] = repr(e)

    try:
        dist_fig, _ = plot_score_distributions_figure(
            score_clean,
            score_adv,
            score_name="Detector score",
            title=f"Detector score distributions ({dataset_name}, {model_name})",
            threshold=threshold,
            show=False,
        )
        figs["score_distribution"] = dist_fig
    except Exception as e:
        errors["score_distribution"] = repr(e)

    try:
        calib_fig, _ = plot_reliability_diagram(
            probabilities,
            labels,
            title=f"Reliability (score→proxy prob) ({dataset_name}, {model_name})",
            show=False,
        )
        figs["reliability"] = calib_fig
    except Exception as e:
        errors["reliability"] = repr(e)

    try:
        fk = list(feature_keys) if feature_keys is not None else []
        feat_fig, _ = plot_topology_feature_shift(
            topology_scores_clean,
            topology_scores_adv,
            feature_keys=fk,
            title=f"Topology feature shift ({dataset_name}, {model_name})",
        )
        figs["topology_shift"] = feat_fig
    except Exception as e:
        errors["topology_shift"] = repr(e)

    if out_dir_path is not None:
        for key, fig in figs.items():
            fmts = ("png",) if bool(save_png_only) else ("pdf", "png")
            saved = save_figure(fig, str(out_dir_path / f"{pref}{key}"), formats=fmts, force_pdf=not bool(save_png_only))
            fig_paths[key] = saved.get("png", next(iter(saved.values())))

    # Metrics table (single-row)
    table_row = {
        "dataset": dataset_name,
        "model": model_name,
        "auroc": float(metrics.get("roc_auc", np.nan)),
        "aupr": float(metrics.get("pr_auc", np.nan)),
        "fpr_at_tpr95": float(metrics.get("fpr_at_tpr95", np.nan)),
        "threshold": float(threshold) if threshold is not None else np.nan,
    }
    # Calibration summary
    calib_metrics = compute_calibration_metrics(probabilities, labels, n_bins=10)
    table_row.update({
        "ece": float(calib_metrics.get("ece", np.nan)),
        "mce": float(calib_metrics.get("mce", np.nan)),
    })

    tables = {"metrics": table_row}
    table_paths: Dict[str, str] = {}
    if out_dir_path is not None:
        metrics_csv = out_dir_path / f"{pref}metrics.csv"
        with metrics_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table_row.keys()))
            writer.writeheader()
            writer.writerow(table_row)
        table_paths["metrics"] = str(metrics_csv)

        # Topology feature shift table (per-feature Cohen's d).
        try:
            fk = list(feature_keys) if feature_keys is not None else []
            rows = []
            for k in fk:
                if k in topology_scores_clean and k in topology_scores_adv:
                    rows.append({"feature": k, "cohens_d": _cohens_d(topology_scores_clean[k], topology_scores_adv[k])})
            rows.sort(key=lambda r: abs(float(r.get("cohens_d", 0.0))), reverse=True)
            shift_csv = out_dir_path / f"{pref}topology_shift.csv"
            with shift_csv.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["feature", "cohens_d"])
                w.writeheader()
                for r in rows:
                    w.writerow(r)
            table_paths["topology_shift"] = str(shift_csv)
            tables["topology_shift"] = rows
        except Exception as e:
            errors["topology_shift_table"] = repr(e)

    if errors:
        tables["errors"] = errors
    return ReportArtifacts(figures=figs, figure_paths=fig_paths, tables=tables, table_paths=table_paths)


def build_detector_report_from_scores(
    *,
    dataset_name: str,
    model_name: str,
    metrics: Mapping[str, Any],
    labels: np.ndarray,
    raw_scores: np.ndarray,
    threshold: float,
    scores_clean_dict: Mapping[str, np.ndarray],
    scores_shifted_dict: Mapping[str, np.ndarray],
    feature_keys: Optional[Sequence[str]] = None,
    out_dir: Optional[str | Path] = None,
    prefix: Optional[str] = None,
    use_latex: bool = False,
    save_png_only: bool = True,
) -> ReportArtifacts:
    """
    Convenience wrapper when you only have:
      - (labels, raw_scores, threshold, metrics)
      - score dicts for clean and shifted populations
    """
    labels = np.asarray(labels, dtype=int).ravel()
    raw_scores = np.asarray(raw_scores, dtype=float).ravel()
    thr = float(threshold)
    y_pred = (raw_scores >= thr).astype(int)
    # Use the same proxy as TopologyScoreDetector.predict_proba: sigmoid(score - threshold).
    probabilities = 1.0 / (1.0 + np.exp(-(raw_scores - thr)))

    score_clean = np.asarray(scores_clean_dict.get("__detector_score__", []), dtype=float)
    score_shifted = np.asarray(scores_shifted_dict.get("__detector_score__", []), dtype=float)

    # If caller did not provide precomputed detector scores, attempt to pull a common key.
    if score_clean.size == 0:
        score_clean = np.asarray(scores_clean_dict.get("detector_score", []), dtype=float)
    if score_shifted.size == 0:
        score_shifted = np.asarray(scores_shifted_dict.get("detector_score", []), dtype=float)

    topo_clean = _filter_prefix(scores_clean_dict, "topo_")
    topo_shifted = _filter_prefix(scores_shifted_dict, "topo_")
    fk = list(feature_keys) if feature_keys is not None else sorted(set(topo_clean.keys()) & set(topo_shifted.keys()))

    return build_detector_report(
        dataset_name=str(dataset_name),
        model_name=str(model_name),
        metrics=metrics,
        labels=labels,
        scores=raw_scores,
        probabilities=probabilities,
        y_pred=y_pred,
        score_clean=score_clean,
        score_adv=score_shifted,
        topology_scores_clean=topo_clean,
        topology_scores_adv=topo_shifted,
        feature_keys=fk,
        threshold=thr,
        out_dir=out_dir,
        prefix=prefix,
        use_latex=use_latex,
        save_png_only=save_png_only,
    )
