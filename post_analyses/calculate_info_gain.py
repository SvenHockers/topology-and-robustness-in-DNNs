"""
This script does the following (and aligns with section 5.3 of the paper):
- Ablation: how separable is Y using
    G only (geometric kNN features),
    T only (topology features),
    G+T (combined features, when available).
- Residualization: does topology contain signal beyond geometry?
    Fit T_hat = f(G) on val; compute residuals T_perp = T - T_hat; then evaluate AUROC using T_perp.

Data source:
- Uses best trials referenced in out/<dataset>/_aggregate/summary.json when available.
- Only considers dataset-level aggregates: out/<dataset>/_aggregate/summary.{json,csv}
- Skips OOD experiments (paths containing 'OOD').

Output:
- out/_analysis/knn_vs_topo_adv/ablation_adv_summary.csv
- out/_analysis/knn_vs_topo_adv/fig_ablation_heatmaps.png
- out/_analysis/knn_vs_topo_adv/fig_delta_vs_residual.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from glob import glob
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge  
from sklearn.metrics import roc_auc_score  


G_KEYS = ["degree", "laplacian"]
EPS_RE = re.compile(r"(?:base_e_|e[_=])([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE) #regex to match epsilon values in the config


def _read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return list(r)


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _normalize_run_dir(path: str, *, out_root: str) -> str:
    """
    Handle Windows paths stored in some summary.json files by extracting the suffix after 'out\\'
    and mapping it to the local out_root.
    """
    p = str(path)
    if os.path.exists(p):
        return p
    p2 = p.replace("\\", "/")
    # find '/out/' segment
    idx = p2.lower().find("/out/")
    if idx >= 0:
        suffix = p2[idx + 5 :]  # after '/out/'
        return os.path.join(out_root, *suffix.split("/"))
    return p2


def _dataset_from_aggregate_path(summary_path: str, *, out_root: str) -> Optional[str]:
    rel = os.path.relpath(os.path.abspath(summary_path), os.path.abspath(out_root))
    parts = rel.split(os.sep)
    # Only accept dataset-level aggregates: <dataset>/_aggregate/summary.*
    if len(parts) == 3 and parts[1] == "_aggregate":
        return parts[0]
    return None


def _infer_eps(s: str) -> Optional[str]:
    m = EPS_RE.search(str(s).replace("__", "_"))
    if not m:
        return None
    return m.group(1)


def _infer_regime(s: str) -> str:
    ss = str(s).lower()
    if "ood" in ss:
        return "ood"
    if "topology_only" in ss:
        return "topology_only"
    if "baseline" in ss:
        return "baseline"
    return "combined"


def _is_adv_experiment(config_relpath: str) -> bool:
    # treat any config path containing 'OOD' as non-adv experiment
    return "ood" not in str(config_relpath).lower()


def _list_trial_dirs(study_dir: str) -> List[str]:
    return sorted(glob(os.path.join(study_dir, "runs", "trials", "trial_*")))


def _trial_auc(run_dir: str) -> Optional[float]:
    mp = os.path.join(run_dir, "metrics", "metrics.json")
    if not os.path.exists(mp):
        return None
    d = _read_json(mp)
    m_adv = (d.get("metrics_adv") or {}) if isinstance(d, dict) else {}
    return _safe_float(m_adv.get("roc_auc"))


def _pick_best_trial_by_auc(study_dir: str) -> Optional[str]:
    """
    Fallback for cases where we only have study_dir: pick max metrics_adv.roc_auc over trials.
    """
    best_dir = None
    best_auc = None
    for rd in _list_trial_dirs(study_dir):
        auc = _trial_auc(rd)
        if auc is None:
            continue
        if best_auc is None or auc > best_auc:
            best_auc = auc
            best_dir = rd
    return best_dir


@dataclass(frozen=True)
class BestTrial:
    dataset: str
    eps: Optional[str]
    regime: str
    config_relpath: str
    run_dir: str


def collect_best_trials(out_root: str) -> List[BestTrial]:
    out_root = os.path.abspath(out_root)
    best: List[BestTrial] = []

    json_summaries = glob(os.path.join(out_root, "*", "_aggregate", "summary.json"))
    csv_summaries = glob(os.path.join(out_root, "*", "_aggregate", "summary.csv"))
    csv_by_ds = {os.path.basename(os.path.dirname(os.path.dirname(p))): p for p in csv_summaries}

    for p in sorted(json_summaries):
        ds = _dataset_from_aggregate_path(p, out_root=out_root)
        if ds is None:
            continue
        d = _read_json(p)
        for run in d.get("runs", []):
            if run.get("status") != "success":
                continue
            cfg_rel = str(run.get("config_relpath", ""))
            if not _is_adv_experiment(cfg_rel):
                continue
            best_obj = run.get("best") or {}
            rd = best_obj.get("run_dir")
            if not rd:
                # fallback: best trial dir via study_dir scan
                study_dir = run.get("study_dir")
                if study_dir:
                    study_dir = _normalize_run_dir(study_dir, out_root=out_root)
                    rd = _pick_best_trial_by_auc(study_dir)
            if not rd:
                continue
            rd = _normalize_run_dir(str(rd), out_root=out_root)
            eps = _infer_eps(cfg_rel) or _infer_eps(rd)
            regime = _infer_regime(cfg_rel)
            if regime == "ood":
                continue
            best.append(BestTrial(dataset=str(ds), eps=eps, regime=regime, config_relpath=cfg_rel, run_dir=rd))

    for ds, p in sorted(csv_by_ds.items()):
        # If dataset already had summary.json, skip (we prefer json's explicit run_dir)
        if any(bt.dataset == ds for bt in best):
            continue
        rows = _read_csv(p)
        for r in rows:
            if r.get("status") != "success":
                continue
            cfg_rel = str(r.get("config_relpath", ""))
            if not _is_adv_experiment(cfg_rel):
                continue
            study_dir = r.get("study_dir")
            if not study_dir:
                continue
            study_dir = _normalize_run_dir(study_dir, out_root=out_root)
            rd = _pick_best_trial_by_auc(study_dir)
            if not rd:
                continue
            eps = _infer_eps(cfg_rel) or _infer_eps(study_dir)
            regime = _infer_regime(cfg_rel)
            if regime == "ood":
                continue
            best.append(BestTrial(dataset=str(ds), eps=eps, regime=regime, config_relpath=cfg_rel, run_dir=str(rd)))

    return best


def _feature_dir(run_dir: str) -> str:
    return os.path.join(run_dir, "raw", "features")


def _available_topo_keys(run_dir: str, split: str = "test_clean") -> List[str]:
    feat_dir = _feature_dir(run_dir)
    if not os.path.isdir(feat_dir):
        return []
    pref = f"{split}__topo_"
    keys: List[str] = []
    for fn in os.listdir(feat_dir):
        if fn.startswith(pref) and fn.endswith(".npy"):
            keys.append(fn.replace(f"{split}__", "").replace(".npy", ""))
    keys.sort()
    return keys


def _load_col(run_dir: str, split: str, key: str) -> np.ndarray:
    p = os.path.join(_feature_dir(run_dir), f"{split}__{key}.npy")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return np.load(p).astype(float, copy=False).reshape(-1, 1)


def _load_block(run_dir: str, split: str, keys: Sequence[str]) -> np.ndarray:
    cols = [_load_col(run_dir, split, k) for k in keys]
    if not cols:
        raise ValueError(f"No keys for split={split}")
    return np.concatenate(cols, axis=1)


def _make_xy(run_dir: str, split_clean: str, split_adv: str, keys: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    x0 = _load_block(run_dir, split_clean, keys)
    x1 = _load_block(run_dir, split_adv, keys)
    y = np.concatenate([np.zeros(x0.shape[0], dtype=int), np.ones(x1.shape[0], dtype=int)])
    x = np.concatenate([x0, x1], axis=0)
    return x, y


def _standardize_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(x, axis=0)
    sd = np.std(x, axis=0, ddof=0)
    sd = np.where(sd > 0, sd, 1.0)
    return mu, sd


def _standardize_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x - mu) / sd


def _auc_from_probe(
    run_dir: str,
    keys: Sequence[str],
    *,
    train_clean: str = "val_clean",
    train_adv: str = "val_adv",
    test_clean: str = "test_clean",
    test_adv: str = "test_adv",
) -> Tuple[float, int, int, int, int]:
    """
    Train logistic regression on val and evaluate AUROC on test.
    Returns (auc, n_train_clean, n_train_adv, n_test_clean, n_test_adv).
    """
    xtr, ytr = _make_xy(run_dir, train_clean, train_adv, keys)
    xte, yte = _make_xy(run_dir, test_clean, test_adv, keys)

    n_tr0 = int(np.sum(ytr == 0))
    n_tr1 = int(np.sum(ytr == 1))
    n_te0 = int(np.sum(yte == 0))
    n_te1 = int(np.sum(yte == 1))
    if n_tr0 < 5 or n_tr1 < 5 or n_te0 < 5 or n_te1 < 1:
        return float("nan"), n_tr0, n_tr1, n_te0, n_te1

    mu, sd = _standardize_fit(xtr)
    xtr = _standardize_apply(xtr, mu, sd)
    xte = _standardize_apply(xte, mu, sd)

    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        n_jobs=None,
    )
    clf.fit(xtr, ytr)
    p1 = clf.predict_proba(xte)[:, 1]
    auc = float(roc_auc_score(yte, p1)) if len(np.unique(yte)) == 2 else float("nan")
    return auc, n_tr0, n_tr1, n_te0, n_te1


def _auc_residual_topo(
    run_dir: str,
    g_keys: Sequence[str],
    topo_keys: Sequence[str],
    *,
    train_clean: str = "val_clean",
    train_adv: str = "val_adv",
    test_clean: str = "test_clean",
    test_adv: str = "test_adv",
) -> Tuple[float, int, int, int, int]:
    """
    Residualization:
      - Fit ridge: T_hat = f(G) on val
      - Residuals: T_perp = T - T_hat (for val and test)
      - Train LR on val residuals, evaluate AUROC on test residuals.
    """
    xg_tr, ytr = _make_xy(run_dir, train_clean, train_adv, g_keys)
    xt_tr, _ = _make_xy(run_dir, train_clean, train_adv, topo_keys)
    xg_te, yte = _make_xy(run_dir, test_clean, test_adv, g_keys)
    xt_te, _ = _make_xy(run_dir, test_clean, test_adv, topo_keys)

    n_tr0 = int(np.sum(ytr == 0))
    n_tr1 = int(np.sum(ytr == 1))
    n_te0 = int(np.sum(yte == 0))
    n_te1 = int(np.sum(yte == 1))
    if n_tr0 < 5 or n_tr1 < 5 or n_te0 < 5 or n_te1 < 1:
        return float("nan"), n_tr0, n_tr1, n_te0, n_te1

    # Standardize G for ridge
    mu_g, sd_g = _standardize_fit(xg_tr)
    xg_tr_s = _standardize_apply(xg_tr, mu_g, sd_g)
    xg_te_s = _standardize_apply(xg_te, mu_g, sd_g)

    # Fit multi-output ridge
    reg = Ridge(alpha=1.0)
    reg.fit(xg_tr_s, xt_tr)
    t_hat_tr = reg.predict(xg_tr_s)
    t_hat_te = reg.predict(xg_te_s)
    r_tr = xt_tr - t_hat_tr
    r_te = xt_te - t_hat_te

    # Train LR on residuals
    mu_r, sd_r = _standardize_fit(r_tr)
    r_tr_s = _standardize_apply(r_tr, mu_r, sd_r)
    r_te_s = _standardize_apply(r_te, mu_r, sd_r)

    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        n_jobs=None,
    )
    clf.fit(r_tr_s, ytr)
    p1 = clf.predict_proba(r_te_s)[:, 1]
    auc = float(roc_auc_score(yte, p1)) if len(np.unique(yte)) == 2 else float("nan")
    return auc, n_tr0, n_tr1, n_te0, n_te1


def _write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        raise ValueError("No rows to write")
    cols = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _plot_heatmap_delta(rows: List[Dict[str, object]], *, out_path: str) -> None:
    import matplotlib.pyplot as plt

    # collect unique axes
    datasets = sorted({str(r["dataset"]) for r in rows})
    eps_vals = sorted({str(r["eps"]) for r in rows if r.get("eps") is not None})
    if not eps_vals:
        return

    # matrices
    delta = np.full((len(datasets), len(eps_vals)), np.nan, dtype=float)

    idx_ds = {d: i for i, d in enumerate(datasets)}
    idx_eps = {e: j for j, e in enumerate(eps_vals)}

    for r in rows:
        ds = str(r["dataset"])
        e = str(r["eps"])
        i = idx_ds[ds]
        j = idx_eps[e]
        delta[i, j] = float(r["delta_auc_topo"]) if r.get("delta_auc_topo") is not None else np.nan

    fig, ax = plt.subplots(figsize=(6.1, max(3.2, 0.35 * len(datasets))), constrained_layout=True)
    im = ax.imshow(delta, aspect="auto", cmap="coolwarm", vmin=-0.2, vmax=0.2)
    ax.set_title(r"$\Delta$AUROC")
    ax.set_xlabel(r"attack strength $\epsilon$")
    ax.set_ylabel("dataset")
    ax.set_xticks(np.arange(len(eps_vals)))
    ax.set_xticklabels(eps_vals)
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_yticklabels(datasets)
    c = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    c.set_label(r"$\Delta$AUROC")

    # annotate compact values
    for i in range(len(datasets)):
        for j in range(len(eps_vals)):
            if np.isfinite(delta[i, j]):
                ax.text(j, i, f"{delta[i,j]:+.2f}", ha="center", va="center", fontsize=7, color="black")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap_residual(rows: List[Dict[str, object]], *, out_path: str) -> None:
    import matplotlib.pyplot as plt

    datasets = sorted({str(r["dataset"]) for r in rows})
    eps_vals = sorted({str(r["eps"]) for r in rows if r.get("eps") is not None})
    if not eps_vals:
        return

    resid = np.full((len(datasets), len(eps_vals)), np.nan, dtype=float)
    idx_ds = {d: i for i, d in enumerate(datasets)}
    idx_eps = {e: j for j, e in enumerate(eps_vals)}

    for r in rows:
        ds = str(r["dataset"])
        e = str(r["eps"])
        i = idx_ds[ds]
        j = idx_eps[e]
        resid[i, j] = float(r["auc_T_resid"]) if r.get("auc_T_resid") is not None else np.nan

    fig, ax = plt.subplots(figsize=(6.1, max(3.2, 0.35 * len(datasets))), constrained_layout=True)
    im = ax.imshow(resid, aspect="auto", cmap="viridis", vmin=0.5, vmax=1.0)
    ax.set_title(r"AUROC using residual topology $T_{\perp}$")
    ax.set_xlabel(r"attack strength $\epsilon$")
    ax.set_ylabel("dataset")
    ax.set_xticks(np.arange(len(eps_vals)))
    ax.set_xticklabels(eps_vals)
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_yticklabels(datasets)
    c = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    c.set_label("AUROC")

    for i in range(len(datasets)):
        for j in range(len(eps_vals)):
            if np.isfinite(resid[i, j]):
                ax.text(j, i, f"{resid[i,j]:.2f}", ha="center", va="center", fontsize=7, color="white")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_scatter(rows: List[Dict[str, object]], *, out_path: str) -> None:
    import matplotlib.pyplot as plt

    xs = []
    ys = []
    labs = []
    for r in rows:
        d = r.get("delta_auc_topo")
        a = r.get("auc_T_resid")
        if d is None or a is None:
            continue
        dd = float(d)
        aa = float(a)
        if not (math.isfinite(dd) and math.isfinite(aa)):
            continue
        xs.append(dd)
        ys.append(aa)
        labs.append(f"{r['dataset']} (e={r['eps']})")
    if not xs:
        return

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.scatter(xs, ys, s=28, alpha=0.8, color="#4C78A8", edgecolor="none")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel(r"$\Delta$AUROC (G+T minus G)")
    ax.set_ylabel(r"AUROC($T_{\perp}$)")
    ax.set_title("Gain vs residual-topology separability")
    ax.set_xlim(min(xs) - 0.02, max(xs) + 0.02)
    ax.set_ylim(0.45, 1.02)
    ax.grid(alpha=0.15, linewidth=0.6)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default="out", help="Root output directory (default: out)")
    ap.add_argument("--save-dir", default="out/_analysis/knn_vs_topo_adv", help="Where to write outputs")
    args = ap.parse_args()

    out_root = os.path.abspath(str(args.out_root))
    save_dir = os.path.abspath(str(args.save_dir))
    os.makedirs(save_dir, exist_ok=True)

    best = collect_best_trials(out_root)
    # index by (dataset, eps, regime)
    by_key: Dict[Tuple[str, Optional[str], str], BestTrial] = {}
    for bt in best:
        k = (bt.dataset, bt.eps, bt.regime)
        # keep first occurrence; in practice each config_relpath is unique
        by_key.setdefault(k, bt)

    datasets = sorted({bt.dataset for bt in best})
    eps_vals = sorted({bt.eps for bt in best if bt.eps is not None})

    rows: List[Dict[str, object]] = []
    for ds in datasets:
        for eps in eps_vals:
            b_g = by_key.get((ds, eps, "baseline"))
            b_t = by_key.get((ds, eps, "topology_only"))
            b_gt = by_key.get((ds, eps, "combined"))

            # Ablation analysis
            # we prefer computing G/T+G using the same hyperparms
            auc_g_baseline = float("nan")
            auc_t_topology_only = float("nan")

            auc_g = float("nan") 
            auc_t = float("nan")  
            auc_gt = float("nan") 

            n_te0 = n_te1 = n_tr0 = n_tr1 = None
            topo_cnt = None

            # Baseline-only run (G-only)
            if b_g is not None:
                try:
                    auc_g_baseline, *_ = _auc_from_probe(b_g.run_dir, G_KEYS)
                except Exception:
                    pass

            # Topology-only run (T-only)
            if b_t is not None:
                try:
                    topo_keys_t = _available_topo_keys(b_t.run_dir, split="test_clean")
                    if topo_keys_t:
                        auc_t_topology_only, *_ = _auc_from_probe(b_t.run_dir, topo_keys_t)
                except Exception:
                    pass

            if b_gt is not None:
                try:
                    topo_keys_gt = _available_topo_keys(b_gt.run_dir, split="test_clean")
                    topo_cnt = int(len(topo_keys_gt))
                    if topo_keys_gt:
                        auc_g, tr0, tr1, te0, te1 = _auc_from_probe(b_gt.run_dir, G_KEYS)
                        n_tr0, n_tr1, n_te0, n_te1 = tr0, tr1, te0, te1
                        auc_t, *_ = _auc_from_probe(b_gt.run_dir, topo_keys_gt)
                        auc_gt, *_ = _auc_from_probe(b_gt.run_dir, list(G_KEYS) + list(topo_keys_gt))
                except Exception:
                    pass

            # Fallback if no combined run exists
            if b_gt is None:
                if math.isfinite(auc_g_baseline):
                    auc_g = auc_g_baseline
                if math.isfinite(auc_t_topology_only):
                    auc_t = auc_t_topology_only

            # Residualization analysis (requires combined run with both blocks)
            auc_resid = float("nan")
            if b_gt is not None:
                try:
                    topo_keys_gt = _available_topo_keys(b_gt.run_dir, split="test_clean")
                    topo_cnt = int(len(topo_keys_gt))
                    if topo_keys_gt:
                        auc_resid, *_ = _auc_residual_topo(b_gt.run_dir, G_KEYS, topo_keys_gt)
                except Exception:
                    pass

            delta = auc_gt - auc_g if (math.isfinite(auc_gt) and math.isfinite(auc_g)) else float("nan")

            # Only keep rows where we have at least one meaningful result (no nan)  
            if not any(math.isfinite(x) for x in [auc_g, auc_t, auc_gt, auc_resid, auc_g_baseline, auc_t_topology_only]):
                continue

            rows.append(
                {
                    "dataset": ds,
                    "eps": eps,
                    "run_dir_baseline": b_g.run_dir if b_g else "",
                    "run_dir_topology_only": b_t.run_dir if b_t else "",
                    "run_dir_combined": b_gt.run_dir if b_gt else "",
                    "auc_G_baseline": auc_g_baseline,
                    "auc_T_topology_only": auc_t_topology_only,
                    "auc_G": auc_g,
                    "auc_T": auc_t,
                    "auc_GT": auc_gt,
                    "delta_auc_topo": delta,
                    "auc_T_resid": auc_resid,
                    "topo_feature_count_GT": topo_cnt if topo_cnt is not None else "",
                    "n_test_clean_G": n_te0 if n_te0 is not None else "",
                    "n_test_adv_G": n_te1 if n_te1 is not None else "",
                }
            )

    csv_path = os.path.join(save_dir, "ablation_adv_summary.csv")
    _write_csv(csv_path, rows)

    fig_delta = os.path.join(save_dir, "fig_ablation_delta_heatmap.png")
    _plot_heatmap_delta(rows, out_path=fig_delta)

    fig_resid = os.path.join(save_dir, "fig_ablation_residual_heatmap.png")
    _plot_heatmap_residual(rows, out_path=fig_resid)

    fig2 = os.path.join(save_dir, "fig_delta_vs_residual.png")
    _plot_scatter(rows, out_path=fig2)

    print(f"[saved] {csv_path}")
    print(f"[saved] {fig_delta}")
    print(f"[saved] {fig_resid}")
    print(f"[saved] {fig2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

