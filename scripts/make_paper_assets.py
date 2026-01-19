"""
Build paper-ready tables and figure bundles from this repo's `out*/` run artifacts.

What it does
------------
- Scans one or more output roots (e.g. `out_1`, `out`) for runs:
    <out_root>/**/runs/trials/trial_*/metrics/metrics.json
- Extracts:
    - dataset family (e.g. mnist, tabular, torus_one_hole, ...)
    - config group (e.g. base_e_0.1, baseline/base_e_0.1, topology_only/..., query_anchor/...)
    - detector "regime": baseline vs topology_only vs combined vs query_anchor (best-effort)
    - attack epsilon (from copied config json when available; otherwise from path)
    - metrics (ROC-AUC, PR-AUC, FPR@TPR95) for adversarial and OOD (when present)
    - attack/OOD success counts when available
- Writes:
    paper_assets/runs.csv                         (one row per trial)
    paper_assets/tables/table_adv_main.tex        (LaTeX table: adv detection performance)
    paper_assets/tables/table_ood_main.tex        (LaTeX table: OOD detection performance; if present)
    paper_assets/figures/...                      (copies key PNGs from best trial per group)

This is intentionally dependency-light: stdlib + numpy only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from glob import glob
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None


def _mean_std(xs: Sequence[Optional[float]]) -> Tuple[Optional[float], Optional[float], int]:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    n = len(vals)
    if n == 0:
        return None, None, 0
    mu = sum(vals) / n
    var = sum((v - mu) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return mu, math.sqrt(var), n


def _fmt_pm(mu: Optional[float], sd: Optional[float], n: int, *, digits: int = 3) -> str:
    if mu is None or sd is None or n <= 0:
        return "—"
    return f"{mu:.{digits}f} $\\pm$ {sd:.{digits}f} ({n})"


def _parse_out_root_and_group(run_dir: str) -> Tuple[str, str, str]:
    """
    For a run dir like:
      .../out_1/mnist/base_e_0.1/runs/trials/trial_000001
    return:
      out_root_name='out_1', dataset='mnist', group='base_e_0.1'
    For nested group like:
      .../out_1/mnist/baseline/base_e_0.1/runs/...
    group='baseline/base_e_0.1'
    """
    parts = os.path.normpath(run_dir).split(os.sep)
    out_i = None
    for i, p in enumerate(parts):
        if p.startswith("out"):
            # match out, out_1, out_something
            out_i = i
            break
    if out_i is None or out_i + 2 >= len(parts):
        return "unknown", "unknown", "unknown"
    out_root_name = parts[out_i]
    dataset = parts[out_i + 1]

    # group = everything between dataset and "runs"
    try:
        runs_i = parts.index("runs", out_i + 2)
        group_parts = parts[out_i + 2 : runs_i]
        group = "/".join(group_parts) if group_parts else "unknown"
    except Exception:
        group = "unknown"
    return out_root_name, dataset, group


def _infer_regime(group: str) -> str:
    g = group.lower()
    if "query_anchor" in g:
        return "query_anchor"
    if "topology_only" in g:
        return "topology_only"
    if "baseline" in g:
        return "baseline"
    return "combined"


_EPS_RE = re.compile(r"e[_=]([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def _infer_epsilon_from_group(group: str) -> Optional[float]:
    m = _EPS_RE.search(group.replace("__", "_"))
    if not m:
        return None
    return _safe_float(m.group(1))


def _load_trial_config_from_run_dir(run_dir: str) -> Optional[Dict[str, Any]]:
    # Copied config lives at: run_dir/config/*.json
    cfg_dir = os.path.join(run_dir, "config")
    if not os.path.isdir(cfg_dir):
        return None
    cands = sorted(glob(os.path.join(cfg_dir, "*.json")))
    if not cands:
        return None
    try:
        return _read_json(cands[0])
    except Exception:
        return None


def _extract_epsilon(run_dir: str, group: str) -> Optional[float]:
    cfg = _load_trial_config_from_run_dir(run_dir)
    if cfg is not None:
        eps = _safe_float(((cfg.get("attack") or {}).get("epsilon")))
        if eps is not None:
            return eps
    return _infer_epsilon_from_group(group)


def _extract_seed(run_dir: str) -> Optional[int]:
    cfg = _load_trial_config_from_run_dir(run_dir)
    if cfg is None:
        return None
    try:
        s = cfg.get("seed")
        return int(s) if s is not None else None
    except Exception:
        return None


def _extract_ood_method(run_dir: str) -> Optional[str]:
    cfg = _load_trial_config_from_run_dir(run_dir)
    if cfg is None:
        return None
    ood = cfg.get("ood") or {}
    if not isinstance(ood, dict):
        return None
    if bool(ood.get("enabled")):
        m = ood.get("method")
        return str(m) if m is not None else "enabled"
    return None


def _extract_counts_from_success_counts(run_dir: str) -> Dict[str, Optional[int]]:
    out: Dict[str, Optional[int]] = {
        "adv_total": None,
        "adv_success": None,
        "ood_total": None,
        "ood_success": None,
    }
    p = os.path.join(run_dir, "metrics", "success_counts.json")
    if not os.path.exists(p):
        p = os.path.join(run_dir, "metrics", "success_counts.csv")
    if not os.path.exists(p):
        return out
    try:
        if p.endswith(".json"):
            d = _read_json(p)
            out["adv_total"] = int(d.get("adversarial_total")) if d.get("adversarial_total") is not None else None
            out["adv_success"] = int(d.get("adversarial_success")) if d.get("adversarial_success") is not None else None
            out["ood_total"] = int(d.get("ood_total")) if d.get("ood_total") is not None else None
            out["ood_success"] = int(d.get("ood_success")) if d.get("ood_success") is not None else None
            return out

        # csv with one row
        with open(p, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            row = next(iter(r), None)
        if not row:
            return out
        out["adv_total"] = int(row.get("adversarial_total")) if row.get("adversarial_total") else None
        out["adv_success"] = int(row.get("adversarial_success")) if row.get("adversarial_success") else None
        out["ood_total"] = int(row.get("ood_total")) if row.get("ood_total") else None
        out["ood_success"] = int(row.get("ood_success")) if row.get("ood_success") else None
        return out
    except Exception:
        return out


def _copy_if_exists(src: str, dst: str) -> bool:
    if not os.path.exists(src):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return True


FIG_KEYS = [
    # performance
    ("adv_roc_fig.png", "adv_roc.png"),
    ("adv_pr_curve.png", "adv_pr.png"),
    ("adv_confusion_fig.png", "adv_confusion.png"),
    ("adv_score_dist_fig.png", "adv_score_dist.png"),
    # topology interpretation
    ("adv_clean_pd_h0_fig.png", "pd_clean_h0.png"),
    ("adv_clean_pd_h1_fig.png", "pd_clean_h1.png"),
    ("adv_adv_pd_h0_fig.png", "pd_adv_h0.png"),
    ("adv_adv_pd_h1_fig.png", "pd_adv_h1.png"),
]


@dataclass(frozen=True)
class TrialRow:
    out_root: str
    dataset: str
    group: str
    regime: str
    epsilon: Optional[float]
    seed: Optional[int]
    ood_method: Optional[str]
    run_dir: str
    trial_id: str
    adv_roc_auc: Optional[float]
    adv_pr_auc: Optional[float]
    adv_fpr95: Optional[float]
    ood_roc_auc: Optional[float]
    ood_pr_auc: Optional[float]
    ood_fpr95: Optional[float]
    adv_total: Optional[int]
    adv_success: Optional[int]
    ood_total: Optional[int]
    ood_success: Optional[int]


def _trial_id_from_run_dir(run_dir: str) -> str:
    return os.path.basename(os.path.normpath(run_dir))


def collect_trials(out_root: str) -> List[TrialRow]:
    metric_paths = sorted(glob(os.path.join(out_root, "**", "runs", "trials", "trial_*", "metrics", "metrics.json"), recursive=True))
    rows: List[TrialRow] = []
    for mp in metric_paths:
        run_dir = os.path.dirname(os.path.dirname(mp))  # .../trial_xxx
        out_root_name, dataset, group = _parse_out_root_and_group(run_dir)
        regime = _infer_regime(group)
        eps = _extract_epsilon(run_dir, group)
        seed = _extract_seed(run_dir)
        ood_method = _extract_ood_method(run_dir)
        d = _read_json(mp)
        m_adv = d.get("metrics_adv") or {}
        m_ood = d.get("metrics_ood") or {}

        counts = _extract_counts_from_success_counts(run_dir)

        rows.append(
            TrialRow(
                out_root=out_root_name,
                dataset=str(dataset),
                group=str(group),
                regime=str(regime),
                epsilon=eps,
                seed=seed,
                ood_method=ood_method,
                run_dir=str(run_dir),
                trial_id=_trial_id_from_run_dir(run_dir),
                adv_roc_auc=_safe_float(m_adv.get("roc_auc")),
                adv_pr_auc=_safe_float(m_adv.get("pr_auc")),
                adv_fpr95=_safe_float(m_adv.get("fpr_at_tpr95")),
                ood_roc_auc=_safe_float(m_ood.get("roc_auc")),
                ood_pr_auc=_safe_float(m_ood.get("pr_auc")),
                ood_fpr95=_safe_float(m_ood.get("fpr_at_tpr95")),
                adv_total=counts["adv_total"],
                adv_success=counts["adv_success"],
                ood_total=counts["ood_total"],
                ood_success=counts["ood_success"],
            )
        )
    return rows


def write_runs_csv(rows: List[TrialRow], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "out_root",
                "dataset",
                "group",
                "regime",
                "epsilon",
                "seed",
                "ood_method",
                "trial_id",
                "adv_roc_auc",
                "adv_pr_auc",
                "adv_fpr95",
                "ood_roc_auc",
                "ood_pr_auc",
                "ood_fpr95",
                "adv_total",
                "adv_success",
                "ood_total",
                "ood_success",
                "run_dir",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.out_root,
                    r.dataset,
                    r.group,
                    r.regime,
                    "" if r.epsilon is None else f"{r.epsilon}",
                    "" if r.seed is None else str(r.seed),
                    "" if r.ood_method is None else r.ood_method,
                    r.trial_id,
                    "" if r.adv_roc_auc is None else f"{r.adv_roc_auc:.6f}",
                    "" if r.adv_pr_auc is None else f"{r.adv_pr_auc:.6f}",
                    "" if r.adv_fpr95 is None else f"{r.adv_fpr95:.6f}",
                    "" if r.ood_roc_auc is None else f"{r.ood_roc_auc:.6f}",
                    "" if r.ood_pr_auc is None else f"{r.ood_pr_auc:.6f}",
                    "" if r.ood_fpr95 is None else f"{r.ood_fpr95:.6f}",
                    "" if r.adv_total is None else str(r.adv_total),
                    "" if r.adv_success is None else str(r.adv_success),
                    "" if r.ood_total is None else str(r.ood_total),
                    "" if r.ood_success is None else str(r.ood_success),
                    r.run_dir,
                ]
            )


def _group_key(r: TrialRow) -> Tuple[str, str, str, str]:
    # dataset + group + regime + (ood_method or empty)
    return (r.dataset, r.group, r.regime, r.ood_method or "")


def select_best_trials(rows: List[TrialRow], *, metric: str) -> Dict[Tuple[str, str, str, str], TrialRow]:
    best: Dict[Tuple[str, str, str, str], TrialRow] = {}
    for r in rows:
        key = _group_key(r)
        val = getattr(r, metric)
        if val is None:
            continue
        cur = best.get(key)
        if cur is None:
            best[key] = r
            continue
        cur_val = getattr(cur, metric)
        if cur_val is None or float(val) > float(cur_val):
            best[key] = r
    return best


def write_latex_table_adv(rows: List[TrialRow], out_tex: str) -> None:
    """
    Aggregates by (dataset, group, regime) and prints mean±std over trials.
    """
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    grouped: Dict[Tuple[str, str, str], List[TrialRow]] = {}
    for r in rows:
        k = (r.dataset, r.group, r.regime)
        grouped.setdefault(k, []).append(r)

    keys = sorted(grouped.keys())
    lines: List[str] = []
    lines.append(r"\begin{tabular}{lllccc}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Config & Regime & ROC-AUC & PR-AUC & FPR@TPR95 \\")
    lines.append(r"\midrule")
    for (ds, group, regime) in keys:
        rs = grouped[(ds, group, regime)]
        mu_auc, sd_auc, n_auc = _mean_std([x.adv_roc_auc for x in rs])
        mu_pr, sd_pr, n_pr = _mean_std([x.adv_pr_auc for x in rs])
        mu_f, sd_f, n_f = _mean_std([x.adv_fpr95 for x in rs])
        # Avoid backslashes inside f-string expressions (SyntaxError).
        ds_tex = str(ds).replace("_", "\\_")
        group_tex = str(group).replace("_", "\\_")
        regime_tex = str(regime).replace("_", "\\_")
        lines.append(
            f"{ds_tex} & {group_tex} & {regime_tex} & "
            f"{_fmt_pm(mu_auc, sd_auc, n_auc)} & {_fmt_pm(mu_pr, sd_pr, n_pr)} & {_fmt_pm(mu_f, sd_f, n_f)} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_latex_table_ood(rows: List[TrialRow], out_tex: str) -> None:
    ood_rows = [r for r in rows if r.ood_roc_auc is not None or r.ood_pr_auc is not None or r.ood_fpr95 is not None]
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    if not ood_rows:
        with open(out_tex, "w", encoding="utf-8") as f:
            f.write("% No OOD metrics found in scanned runs.\n")
        return

    grouped: Dict[Tuple[str, str, str, str], List[TrialRow]] = {}
    for r in ood_rows:
        k = (r.dataset, r.group, r.regime, r.ood_method or "")
        grouped.setdefault(k, []).append(r)

    keys = sorted(grouped.keys())
    lines: List[str] = []
    lines.append(r"\begin{tabular}{llllccc}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Config & Regime & OOD & ROC-AUC & PR-AUC & FPR@TPR95 \\")
    lines.append(r"\midrule")
    for (ds, group, regime, ood_method) in keys:
        rs = grouped[(ds, group, regime, ood_method)]
        mu_auc, sd_auc, n_auc = _mean_std([x.ood_roc_auc for x in rs])
        mu_pr, sd_pr, n_pr = _mean_std([x.ood_pr_auc for x in rs])
        mu_f, sd_f, n_f = _mean_std([x.ood_fpr95 for x in rs])
        ood_label = ood_method if ood_method else "—"
        ds_tex = str(ds).replace("_", "\\_")
        group_tex = str(group).replace("_", "\\_")
        regime_tex = str(regime).replace("_", "\\_")
        ood_tex = str(ood_label).replace("_", "\\_")
        lines.append(
            f"{ds_tex} & {group_tex} & {regime_tex} & {ood_tex} & "
            f"{_fmt_pm(mu_auc, sd_auc, n_auc)} & {_fmt_pm(mu_pr, sd_pr, n_pr)} & {_fmt_pm(mu_f, sd_f, n_f)} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def copy_best_figures(best_trials: Dict[Tuple[str, str, str, str], TrialRow], out_dir: str) -> int:
    copied = 0
    for (ds, group, regime, ood_method), r in best_trials.items():
        fig_dir = os.path.join(r.run_dir, "images")
        if not os.path.isdir(fig_dir):
            continue
        safe_group = group.replace("/", "__")
        safe_ood = (ood_method or "no_ood").replace("/", "__")
        base = os.path.join(out_dir, ds, safe_group, regime, safe_ood)
        for src_name, dst_name in FIG_KEYS:
            src = os.path.join(fig_dir, src_name)
            dst = os.path.join(base, dst_name)
            if _copy_if_exists(src, dst):
                copied += 1
    return copied


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-roots",
        nargs="+",
        default=["out_1", "out"],
        help="One or more output roots to scan (relative or absolute paths).",
    )
    ap.add_argument(
        "--paper-assets-dir",
        default="paper_assets",
        help="Directory to write tables/figures into.",
    )
    ap.add_argument(
        "--select-metric",
        default="adv_roc_auc",
        choices=["adv_roc_auc", "adv_pr_auc", "ood_roc_auc", "ood_pr_auc"],
        help="Metric used to select the representative ('best') trial per (dataset, group, regime, ood_method).",
    )
    args = ap.parse_args()

    all_rows: List[TrialRow] = []
    for root in args.out_roots:
        root_abs = os.path.abspath(root)
        if not os.path.isdir(root_abs):
            continue
        all_rows.extend(collect_trials(root_abs))

    out_dir = os.path.abspath(args.paper_assets_dir)
    runs_csv = os.path.join(out_dir, "runs.csv")
    write_runs_csv(all_rows, runs_csv)

    write_latex_table_adv(all_rows, os.path.join(out_dir, "tables", "table_adv_main.tex"))
    write_latex_table_ood(all_rows, os.path.join(out_dir, "tables", "table_ood_main.tex"))

    best = select_best_trials(all_rows, metric=args.select_metric)
    copied = copy_best_figures(best, os.path.join(out_dir, "figures"))

    print(f"[paper_assets] wrote {runs_csv} ({len(all_rows)} trials)")
    print(f"[paper_assets] wrote tables to {os.path.join(out_dir, 'tables')}")
    print(f"[paper_assets] copied {copied} figure files to {os.path.join(out_dir, 'figures')}")


if __name__ == "__main__":
    main()

