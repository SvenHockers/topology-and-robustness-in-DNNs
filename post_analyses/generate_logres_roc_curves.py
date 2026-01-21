"""
Plot ROC curves to assess how much separability is available in T and how much is retained by Mahalanobis distance.

Data format assumptions:
- Per trial: <run_dir>/raw/features contains per-split .npy columns:
    val_clean__topo_*.npy, val_adv__topo_*.npy, test_clean__topo_*.npy, test_adv__topo_*.npy
- Splits used:
    fit detectors on val_{clean,adv} (and val_clean for Mahalanobis moments - since we need to fit the distribution)
    evaluate ROC on test_{clean,adv}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from glob import glob
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

def _ensure_cache_dirs() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(repo_root, ".mplcache"))
    os.environ.setdefault("XDG_CACHE_HOME", os.path.join(repo_root, ".cache"))
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    os.makedirs(os.path.join(os.environ["XDG_CACHE_HOME"], "fontconfig"), exist_ok=True)


_ensure_cache_dirs()

from sklearn.covariance import LedoitWolf  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import auc as _auc  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402


EPS_RE = re.compile(r"(?:base_e_|e[_=])([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def _read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_eps(s: str) -> Optional[str]:
    m = EPS_RE.search(str(s).replace("__", "_"))
    return m.group(1) if m else None


def _normalize_run_dir(path: str, *, out_root: str) -> str:
    """
    Handle Windows paths stored in some summary.json files by extracting the suffix after 'out\\'
    and mapping it to the local out_root.
    """
    p = str(path)
    if os.path.exists(p):
        return p
    p2 = p.replace("\\", "/")
    idx = p2.lower().find("/out/")
    if idx >= 0:
        suffix = p2[idx + 5 :]  # after '/out/'
        return os.path.join(out_root, *suffix.split("/"))
    return p2


def _dataset_from_aggregate_path(summary_path: str, *, out_root: str) -> Optional[str]:
    rel = os.path.relpath(os.path.abspath(summary_path), os.path.abspath(out_root))
    parts = rel.split(os.sep)
    if len(parts) == 3 and parts[1] == "_aggregate":
        return parts[0]
    return None


@dataclass(frozen=True)
class BestTrial:
    dataset: str
    eps: Optional[str]
    regime: str
    config_relpath: str
    run_dir: str


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
    return "ood" not in str(config_relpath).lower()


def collect_best_trials(out_root: str) -> List[BestTrial]:
    """
    Collect best trials from out/<dataset>/_aggregate/summary.json.
    We keep only ADV experiments (exclude OOD) and return (dataset, eps, regime, run_dir).
    """
    out_root = os.path.abspath(out_root)
    best: List[BestTrial] = []
    json_summaries = glob(os.path.join(out_root, "*", "_aggregate", "summary.json"))
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
                continue
            rd = _normalize_run_dir(str(rd), out_root=out_root)
            eps = _infer_eps(cfg_rel) or _infer_eps(rd)
            regime = _infer_regime(cfg_rel)
            if regime == "ood":
                continue
            best.append(BestTrial(dataset=str(ds), eps=eps, regime=regime, config_relpath=cfg_rel, run_dir=rd))
    return best


def _feature_dir(run_dir: str) -> str:
    return os.path.join(run_dir, "raw", "features")


def _available_keys(run_dir: str, *, split: str, prefix: str) -> List[str]:
    """
    List keys for a given split and prefix. Example:
      split='test_clean', prefix='topo_' => returns ['topo_h0_count', ...]
    """
    feat_dir = _feature_dir(run_dir)
    if not os.path.isdir(feat_dir):
        return []
    pref = f"{split}__{prefix}"
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
        raise ValueError(f"No keys to load for split={split}")
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


def _score_logreg(
    run_dir: str,
    keys: Sequence[str],
    *,
    train_clean: str = "val_clean",
    train_adv: str = "val_adv",
    test_clean: str = "test_clean",
    test_adv: str = "test_adv",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Upper bound (linear): standardize on val, train LR on val_{clean,adv}, score test.
    Returns (y_test, score) where score = P(adv|x).
    """
    xtr, ytr = _make_xy(run_dir, train_clean, train_adv, keys)
    xte, yte = _make_xy(run_dir, test_clean, test_adv, keys)
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return yte, np.full_like(yte, fill_value=np.nan, dtype=float)
    mu, sd = _standardize_fit(xtr)
    xtr = _standardize_apply(xtr, mu, sd)
    xte = _standardize_apply(xte, mu, sd)
    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=3000)
    clf.fit(xtr, ytr)
    s = clf.predict_proba(xte)[:, 1]
    return yte, s.astype(float, copy=False)


def _score_mahalanobis(
    run_dir: str,
    topo_keys: Sequence[str],
    *,
    fit_split: str = "val_clean",
    test_clean: str = "test_clean",
    test_adv: str = "test_adv",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scalar score: Mahalanobis distance to clean distribution in topo-feature space.
    - Fit mean/cov on val_clean only.
    - Score test_{clean,adv}: higher => more "non-clean".
    """
    x_fit = _load_block(run_dir, fit_split, topo_keys)
    x0 = _load_block(run_dir, test_clean, topo_keys)
    x1 = _load_block(run_dir, test_adv, topo_keys)
    y = np.concatenate([np.zeros(x0.shape[0], dtype=int), np.ones(x1.shape[0], dtype=int)])
    x = np.concatenate([x0, x1], axis=0)

    # Standardize using clean moments (important for mixed-scale topo stats).
    mu_s, sd_s = _standardize_fit(x_fit)
    x_fit_s = _standardize_apply(x_fit, mu_s, sd_s)
    x_s = _standardize_apply(x, mu_s, sd_s)

    # Shrinkage covariance for stability when n is not large.
    lw = LedoitWolf().fit(x_fit_s)
    mu = lw.location_
    prec = lw.precision_

    d = x_s - mu.reshape(1, -1)
    # quadratic form per sample: (x-mu)^T Prec (x-mu)
    s = np.einsum("bi,ij,bj->b", d, prec, d)
    return y, s.astype(float, copy=False)


def _plot_roc_panel(
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    *,
    title: str,
    out_path: str,
    use_tex: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    if use_tex:
        # Requires a LaTeX installation on the system. If unavailable, matplotlib will error.
        plt.rcParams.update({"text.usetex": True})

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.4, linewidth=1.0)

    for name, (y, s) in curves.items():
        y = np.asarray(y, dtype=int)
        s = np.asarray(s, dtype=float)
        ok = np.isfinite(s)
        if ok.sum() < 2 or len(np.unique(y[ok])) < 2:
            continue
        fpr, tpr, _thr = roc_curve(y[ok], s[ok])
        a = _auc(fpr, tpr)
        # Use mathtext for a consistent "LaTeX-like" look without requiring TeX.
        ax.plot(fpr, tpr, linewidth=2.0, label=rf"{name} (AUROC={a:.3f})")

    ax.set_xlabel(r"$\mathrm{FPR}$")
    ax.set_ylabel(r"$\mathrm{TPR}$")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.15, linewidth=0.6)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default="out", help="Root output directory (default: out)")
    ap.add_argument(
        "--save-dir",
        default="out/_analysis/topo_upper_bound_rocs",
        help="Where to write ROC plots",
    )
    ap.add_argument(
        "--eps-values",
        default="0.1,0.2,0.3",
        help="Comma-separated eps values to plot (default: 0.1,0.2,0.3).",
    )
    ap.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Optional explicit trial run_dir (e.g. out/synthetic_shapes/base_e_0.2/runs/trials/trial_000006). "
            "If set, bypasses _aggregate lookup and plots only this run."
        ),
    )
    ap.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset filter (exact name, e.g. 'synthetic_shapes'). If omitted, plots all datasets found.",
    )
    ap.add_argument(
        "--eps",
        default=None,
        help="Optional epsilon filter (e.g. '0.2'). If omitted, plots all eps found.",
    )
    ap.add_argument(
        "--regime",
        default="combined",
        choices=["combined", "topology_only", "baseline"],
        help="Which best-trial regime to use as data source (default: combined).",
    )
    ap.add_argument(
        "--usetex",
        action="store_true",
        help="Use full LaTeX rendering (requires a LaTeX installation). Otherwise uses matplotlib mathtext.",
    )
    ap.add_argument(
        "--allow-mix",
        action="store_true",
        help=(
            "Allow a single figure to mix multiple datasets/eps. Not recommended; "
            "kept only for quick exploratory debugging."
        ),
    )
    args = ap.parse_args()

    def _norm_eps(e: Optional[str]) -> Optional[str]:
        if e is None:
            return None
        try:
            # %g normalizes "0.10" -> "0.1"
            return ("%g" % float(e)).strip()
        except Exception:
            return str(e).strip()

    eps_allow = {_norm_eps(s) for s in str(args.eps_values).split(",") if str(s).strip()}
    eps_allow.discard(None)

    out_root = os.path.abspath(str(args.out_root))
    save_dir = os.path.abspath(str(args.save_dir))
    os.makedirs(save_dir, exist_ok=True)

    if args.run_dir:
        run_dir = os.path.abspath(str(args.run_dir))
        if not os.path.isdir(run_dir):
            raise SystemExit(f"--run-dir does not exist: {run_dir}")
        eps = _norm_eps(_infer_eps(run_dir) or (str(args.eps) if args.eps else None) or "unknown")
        if eps_allow and eps not in eps_allow:
            # Respect --eps-values even in --run-dir mode.
            return 0
        ds = str(args.dataset) if args.dataset else os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(run_dir))))

        topo_keys = _available_keys(run_dir, split="test_clean", prefix="topo_")
        if not topo_keys:
            raise SystemExit(f"No topo features found under {os.path.join(run_dir, 'raw', 'features')}")

        curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        y_m, s_m = _score_mahalanobis(run_dir, topo_keys)
        curves[r"Mahalanobis"] = (y_m, s_m)
        y_t, s_t = _score_logreg(run_dir, topo_keys)
        curves[r"LogReg"] = (y_t, s_t)

        title = rf"{ds} adv detection"
        out_path = os.path.join(save_dir, f"roc__{ds}__eps_{eps}__RUN_DIR.png")
        _plot_roc_panel(curves, title=title, out_path=out_path, use_tex=bool(args.usetex))
        return 0

    best = [bt for bt in collect_best_trials(out_root) if bt.regime == str(args.regime)]
    best = [BestTrial(bt.dataset, _norm_eps(bt.eps), bt.regime, bt.config_relpath, bt.run_dir) for bt in best]
    if args.dataset:
        best = [bt for bt in best if bt.dataset == str(args.dataset)]
    if args.eps:
        best = [bt for bt in best if bt.eps == _norm_eps(str(args.eps))]
    if eps_allow:
        best = [bt for bt in best if bt.eps in eps_allow]

    if not best:
        raise SystemExit("No matching best trials found. Try adjusting --dataset/--eps/--regime.")

    # Group by dataset+eps (default, recommended)
    groups: Dict[Tuple[str, str], List[BestTrial]] = {}
    for bt in best:
        if bt.eps is None:
            continue
        groups.setdefault((bt.dataset, bt.eps), []).append(bt)

    if not groups:
        raise SystemExit("No trials with inferable eps found.")

    # Either produce one plot per group (recommended) or one big mixed plot (discouraged).
    if not args.allow_mix:
        for (ds, eps), bts in sorted(groups.items()):
            bt = bts[0]  # one per (ds, eps, regime)
            run_dir = bt.run_dir

            topo_keys = _available_keys(run_dir, split="test_clean", prefix="topo_")
            if not topo_keys:
                # fall back to 'topo_' inferred from file names without explicit prefix arg
                topo_keys = _available_keys(run_dir, split="test_clean", prefix="topo_")
            if not topo_keys:
                continue

            curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            try:
                y_m, s_m = _score_mahalanobis(run_dir, topo_keys)
                curves[r"Mahalanobis"] = (y_m, s_m)
            except Exception:
                pass

            try:
                y_t, s_t = _score_logreg(run_dir, topo_keys)
                curves[r"LogReg"] = (y_t, s_t)
            except Exception:
                pass

            if not curves:
                continue

            title = f"{ds} adv detection"
            out_path = os.path.join(save_dir, f"roc__{ds}__eps_{eps}__{args.regime}.png")
            _plot_roc_panel(curves, title=title, out_path=out_path, use_tex=bool(args.usetex))

        return 0

    mixed: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for (ds, eps), bts in sorted(groups.items()):
        bt = bts[0]
        run_dir = bt.run_dir
        topo_keys = _available_keys(run_dir, split="test_clean", prefix="topo_")
        if not topo_keys:
            continue
        try:
            y, s = _score_mahalanobis(run_dir, topo_keys)
            mixed[rf"Mahalanobis | {ds}"] = (y, s)
        except Exception:
            pass
        try:
            y, s = _score_logreg(run_dir, topo_keys)
            mixed[rf"LogReg | {ds}"] = (y, s)
        except Exception:
            pass

    if not mixed:
        raise SystemExit("No curves could be computed in --allow-mix mode.")

    out_path = os.path.join(save_dir, f"roc__MIXED__{args.regime}.png")
    _plot_roc_panel(
        mixed,
        title=r"MIXED (NOT comparable): per-dataset curves",
        out_path=out_path,
        use_tex=bool(args.usetex),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

