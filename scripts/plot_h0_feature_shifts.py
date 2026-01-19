"""
Plot how strongly H0/H1 topology features differ between clean and adversarial samples.

This script is designed to *visualize the conclusion* we've reached in analysis:
in many (but not all) experiments, H0/H1 persistence summary features are shifted
between clean and successful adversarial samples.

It relies only on numpy + matplotlib and on the repo's existing run artifacts:
  out/**/runs/trials/trial_*/raw/features/test_clean__topo_h{0,1}_*.npy
  out/**/runs/trials/trial_*/raw/features/test_adv__topo_h{0,1}_*.npy

Notes / assumptions:
- In this repo, many runs export `test_adv__*` arrays corresponding to *successful*
  adversarial examples (see records.jsonl / success_counts.json). When that's true,
  these comparisons are already "successful-only".
- Some runs (e.g. baseline-only) won't export topo_*; they are automatically skipped.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from glob import glob
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


H0_KEYS_DEFAULT = [
    "topo_h0_total_persistence",
    "topo_h0_l2_persistence",
    "topo_h0_max_persistence",
    "topo_h0_entropy",
]

H0_LABELS_LATEX: Dict[str, str] = {
    "topo_h0_total_persistence": r"$\mathrm{H}_0$ total persistence",
    "topo_h0_l2_persistence": r"$\mathrm{H}_0$ $\ell_2$ persistence",
    "topo_h0_max_persistence": r"$\mathrm{H}_0$ max persistence",
    "topo_h0_entropy": r"$\mathrm{H}_0$ entropy",
}

H1_KEYS_DEFAULT = [
    "topo_h1_total_persistence",
    "topo_h1_l2_persistence",
    "topo_h1_max_persistence",
    "topo_h1_entropy",
]

H1_LABELS_LATEX: Dict[str, str] = {
    "topo_h1_total_persistence": r"$\mathrm{H}_1$ total persistence",
    "topo_h1_l2_persistence": r"$\mathrm{H}_1$ $\ell_2$ persistence",
    "topo_h1_max_persistence": r"$\mathrm{H}_1$ max persistence",
    "topo_h1_entropy": r"$\mathrm{H}_1$ entropy",
}


def _dim_prefix(dim: int) -> str:
    if int(dim) not in (0, 1):
        raise ValueError(f"Unsupported dim={dim}; expected 0 or 1")
    return f"topo_h{int(dim)}_"


def _keys_and_labels_for_dim(dim: int) -> Tuple[List[str], Dict[str, str]]:
    if int(dim) == 0:
        return list(H0_KEYS_DEFAULT), dict(H0_LABELS_LATEX)
    if int(dim) == 1:
        return list(H1_KEYS_DEFAULT), dict(H1_LABELS_LATEX)
    raise ValueError(f"Unsupported dim={dim}; expected 0 or 1")


def _roc_auc_binary(y: np.ndarray, score: np.ndarray) -> float:
    """
    Compute AUROC for binary labels without sklearn.
    Uses rank statistic; handles ties by average rank.
    """
    y = np.asarray(y, dtype=int).ravel()
    s = np.asarray(score, dtype=float).ravel()
    if y.size != s.size:
        raise ValueError("y/score length mismatch")
    if y.size == 0:
        return float("nan")
    if len(np.unique(y)) < 2:
        return float("nan")

    # Average ranks for ties
    uniq, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    rank_mean = np.zeros_like(uniq, dtype=float)
    start = 0
    for i, c in enumerate(counts):
        rank_mean[i] = (start + (start + c - 1)) / 2.0
        start += c
    ranks = rank_mean[inv]

    n1 = float(np.sum(y == 1))
    n0 = float(np.sum(y == 0))
    auc = (np.sum(ranks[y == 1]) - n1 * (n1 - 1) / 2.0) / (n1 * n0)
    return float(auc)


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or b.size == 0:
        return float("nan")
    sa = float(np.std(a))
    sb = float(np.std(b))
    pooled = float(np.sqrt((sa * sa + sb * sb) / 2.0)) if (sa > 0 or sb > 0) else 0.0
    return float((float(np.mean(b)) - float(np.mean(a))) / pooled) if pooled > 0 else 0.0


def _parse_dataset_and_config(run_dir: str, *, out_root: str) -> Tuple[str, str]:
    """
    Best-effort parse of (dataset, config) from a path like:
      out/<dataset>/<config>/runs/trials/trial_000001
    """
    try:
        # Primary: parse relative to out_root so this works for out/, out_1/, etc.
        rel = os.path.relpath(os.path.normpath(run_dir), os.path.normpath(out_root))
        parts = rel.split(os.sep)
        if len(parts) >= 2:
            dataset = parts[0]
            config = parts[1]
            return str(dataset), str(config)
    except Exception:
        pass

    # Fallback: old behavior for paths that include a literal "out" segment.
    parts2 = os.path.normpath(run_dir).split(os.sep)
    try:
        i = parts2.index("out")
        dataset = parts2[i + 1]
        config = parts2[i + 2]
        return str(dataset), str(config)
    except Exception:
        return "unknown", "unknown"


def _dataset_to_modality(dataset: str) -> str:
    ds = str(dataset).lower()
    if ds in {"mnist"}:
        return "image"
    if ds in {"tabular"}:
        return "tabular"
    return "vector"


def _load_feature_pair(run_dir: str, *, split_a: str, split_b: str, key: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    feat_dir = os.path.join(run_dir, "raw", "features")
    a_path = os.path.join(feat_dir, f"{split_a}__{key}.npy")
    b_path = os.path.join(feat_dir, f"{split_b}__{key}.npy")
    if not (os.path.exists(a_path) and os.path.exists(b_path)):
        return None
    a = np.load(a_path).astype(float, copy=False)
    b = np.load(b_path).astype(float, copy=False)
    return a, b


@dataclass(frozen=True)
class Record:
    dataset: str
    config: str
    run_dir: str
    key: str
    n_clean: int
    n_adv: int
    auc: float
    d: float


def collect_records(*, out_root: str, keys: List[str], min_adv: int, dim: int) -> List[Record]:
    run_dirs = sorted(glob(os.path.join(out_root, "**", "runs", "trials", "trial_*"), recursive=True))
    recs: List[Record] = []

    for rd in run_dirs:
        dataset, config = _parse_dataset_and_config(rd, out_root=str(out_root))

        # require at least one topo feature file for speed
        feat_dir = os.path.join(rd, "raw", "features")
        if not os.path.isdir(feat_dir):
            continue
        pref = f"test_clean__{_dim_prefix(int(dim))}"
        if not any(fn.startswith(pref) and fn.endswith(".npy") for fn in os.listdir(feat_dir)):
            continue

        for key in keys:
            pair = _load_feature_pair(rd, split_a="test_clean", split_b="test_adv", key=key)
            if pair is None:
                continue
            a, b = pair
            if int(len(b)) < int(min_adv) or int(len(a)) < 5:
                continue

            y = np.concatenate([np.zeros(len(a), dtype=int), np.ones(len(b), dtype=int)])
            x = np.concatenate([a, b]).astype(float, copy=False)
            auc = _roc_auc_binary(y, x)
            d = _cohen_d(a, b)

            recs.append(
                Record(
                    dataset=str(dataset),
                    config=str(config),
                    run_dir=str(rd),
                    key=str(key),
                    n_clean=int(len(a)),
                    n_adv=int(len(b)),
                    auc=float(auc),
                    d=float(d),
                )
            )

    return recs


def _group_by(items: Iterable[Record], key_fn) -> Dict[str, List[Record]]:
    out: Dict[str, List[Record]] = {}
    for it in items:
        k = str(key_fn(it))
        out.setdefault(k, []).append(it)
    return out


def plot_summary(recs: List[Record], *, out_path: str, auc_good: float) -> None:
    if len(recs) == 0:
        raise SystemExit("No records found. Are topo features exported under out/**/raw/features/?")

    # Unique datasets and keys
    datasets = sorted(set(r.dataset for r in recs))
    # Prefer the script's default key order if present; otherwise fall back to observed keys.
    default_order = [*H0_KEYS_DEFAULT, *H1_KEYS_DEFAULT]
    keys = [k for k in default_order if any(r.key == k for r in recs)]
    if len(keys) == 0:
        keys = sorted(set(r.key for r in recs))

    # Prepare per-dataset arrays for each key
    fig = plt.figure(figsize=(14, 3.5 * len(keys)))
    gs = fig.add_gridspec(len(keys), 2, width_ratios=[3, 1])

    for i, key in enumerate(keys):
        sub = [r for r in recs if r.key == key]
        by_ds = _group_by(sub, lambda r: r.dataset)

        # Left: boxplots of AUROC per dataset
        ax = fig.add_subplot(gs[i, 0])
        data = [np.array([r.auc for r in by_ds.get(ds, [])], dtype=float) for ds in datasets]
        ax.boxplot(data, tick_labels=datasets, showfliers=False)
        ax.axhline(0.5, color="k", linestyle="--", alpha=0.5, linewidth=1)
        ax.axhline(float(auc_good), color="tab:green", linestyle="--", alpha=0.8, linewidth=1)
        ax.set_ylim(0.0, 1.0)
        label_map = {**H0_LABELS_LATEX, **H1_LABELS_LATEX}
        ax.set_title(f"{label_map.get(key, key)}: AUROC(clean vs adv) by dataset (each point = one trial/run)")
        ax.set_ylabel("AUROC")
        ax.tick_params(axis="x", rotation=25)

        # Right: fraction of runs above threshold per dataset
        ax2 = fig.add_subplot(gs[i, 1])
        fracs = []
        for ds in datasets:
            a = np.array([r.auc for r in by_ds.get(ds, [])], dtype=float)
            if a.size == 0:
                fracs.append(np.nan)
            else:
                fracs.append(float(np.mean(a >= float(auc_good))))
        ax2.barh(np.arange(len(datasets)), np.nan_to_num(fracs, nan=0.0), color="tab:blue", alpha=0.8)
        ax2.set_yticks(np.arange(len(datasets)))
        ax2.set_yticklabels(datasets)
        ax2.set_xlim(0.0, 1.0)
        ax2.set_title(f"Frac ≥ {auc_good:.2f}")
        for j, v in enumerate(fracs):
            if np.isnan(v):
                txt = "n/a"
                vv = 0.0
            else:
                txt = f"{v:.2f}"
                vv = float(v)
            ax2.text(min(0.98, vv + 0.02), j, txt, va="center", fontsize=9)

    fig.suptitle("H0 topology features often differ between clean and adversarial samples (across runs)", y=1.02, fontsize=14)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"[saved] {out_path}")


def plot_overall_shift(recs: List[Record], *, out_path: str, d_ref: float = 0.5) -> None:
    """
    Report-style figure focused on the *overall distribution shift*:
    for each H0 feature, show the distribution of Cohen's d across runs.

    Interpretation:
    - d > 0: adv values tend to be larger than clean
    - d < 0: adv values tend to be smaller than clean
    - |d|: effect size magnitude (shift strength)
    """
    if len(recs) == 0:
        raise SystemExit("No records found.")

    default_order = [*H0_KEYS_DEFAULT, *H1_KEYS_DEFAULT]
    keys = [k for k in default_order if any(r.key == k for r in recs)]
    if len(keys) == 0:
        keys = sorted(set(r.key for r in recs))

    # Gather d-values per key
    data = []
    ns = []
    for k in keys:
        d = np.array([r.d for r in recs if r.key == k and np.isfinite(r.d)], dtype=float)
        data.append(d)
        ns.append(int(d.size))

    fig, ax = plt.subplots(figsize=(12, 5))
    parts = ax.violinplot(
        data,
        positions=np.arange(1, len(keys) + 1),
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for pc in parts["bodies"]:
        pc.set_facecolor("#4C78A8")
        pc.set_edgecolor("black")
        pc.set_alpha(0.45)

    ax.axhline(0.0, color="k", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(+float(d_ref), color="tab:green", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(-float(d_ref), color="tab:green", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xticks(np.arange(1, len(keys) + 1))
    label_map = {**H0_LABELS_LATEX, **H1_LABELS_LATEX}
    ax.set_xticklabels([label_map.get(k, k) for k in keys], rotation=0, ha="center")
    ax.set_ylabel(r"$\Delta$")
    ax.set_title("Overall shift of H0 topology features under attacks (across runs)")

    # Annotate n per feature
    ymin, ymax = ax.get_ylim()
    y_annot = ymin + 0.05 * (ymax - ymin)
    for i, n in enumerate(ns, start=1):
        ax.text(i, y_annot, f"n={n}", ha="center", va="bottom", fontsize=9, alpha=0.85)

    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="k", linestyle="--", alpha=0.5, label="no shift (d=0)"),
            plt.Line2D([0], [0], color="tab:green", linestyle="--", alpha=0.5, label=f"moderate shift (|d|≈{d_ref})"),
        ],
        loc="upper right",
        frameon=False,
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[saved] {out_path}")


def _max_abs_d_per_run(records: List[Record], *, keys: List[str]) -> Dict[Tuple[str, str], float]:
    """
    Compute max(|d|) over selected keys per (dataset, run_dir).
    """
    keys_set = set(keys)
    by_run: Dict[Tuple[str, str], List[float]] = {}
    for r in records:
        if r.key not in keys_set:
            continue
        if not np.isfinite(r.d):
            continue
        by_run.setdefault((r.dataset, r.run_dir), []).append(abs(float(r.d)))

    out: Dict[Tuple[str, str], float] = {}
    for k, vals in by_run.items():
        v = np.asarray(vals, dtype=float)
        if v.size:
            out[k] = float(np.max(v))
    return out


def split_datasets_by_h0_shift(
    recs: List[Record],
    *,
    keys_for_split: List[str],
    d_ref: float,
    min_runs_per_dataset: int = 5,
) -> Tuple[List[str], List[str], Dict[str, float], Dict[str, int]]:
    """
    Split datasets into 'shift' vs 'no_shift' based on median max(|d|) across runs.
    """
    max_abs = _max_abs_d_per_run(recs, keys=keys_for_split)
    by_ds: Dict[str, List[float]] = {}
    for (ds, _rd), v in max_abs.items():
        by_ds.setdefault(ds, []).append(float(v))

    med_by: Dict[str, float] = {}
    n_by: Dict[str, int] = {}
    for ds, vals in by_ds.items():
        n_by[ds] = int(len(vals))
        if len(vals) >= int(min_runs_per_dataset):
            med_by[ds] = float(np.median(np.asarray(vals, dtype=float)))

    shift = [ds for ds, med in med_by.items() if med >= float(d_ref)]
    no_shift = [ds for ds, med in med_by.items() if med < float(d_ref)]
    shift.sort()
    no_shift.sort()
    return shift, no_shift, med_by, n_by


def plot_overall_shift_for_subset(
    recs: List[Record],
    *,
    datasets_keep: List[str],
    out_path: str,
    title: str,
    d_ref: float = 0.5,
) -> None:
    sub = [r for r in recs if r.dataset in set(datasets_keep)]
    if len(sub) == 0:
        raise SystemExit(f"No records for subset: {datasets_keep}")

    default_order = [*H0_KEYS_DEFAULT, *H1_KEYS_DEFAULT]
    keys = [k for k in default_order if any(r.key == k for r in sub)]
    data = []
    ns = []
    for k in keys:
        d = np.array([r.d for r in sub if r.key == k and np.isfinite(r.d)], dtype=float)
        data.append(d)
        ns.append(int(d.size))

    fig, ax = plt.subplots(figsize=(12, 5))
    parts = ax.violinplot(
        data,
        positions=np.arange(1, len(keys) + 1),
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for pc in parts["bodies"]:
        pc.set_facecolor("#4C78A8")
        pc.set_edgecolor("black")
        pc.set_alpha(0.45)

    ax.axhline(0.0, color="k", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(+float(d_ref), color="tab:green", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(-float(d_ref), color="tab:green", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xticks(np.arange(1, len(keys) + 1))
    label_map = {**H0_LABELS_LATEX, **H1_LABELS_LATEX}
    ax.set_xticklabels([label_map.get(k, k) for k in keys], rotation=0, ha="center")
    ax.set_ylabel(r"$\Delta$")
    ax.set_title(title)

    ymin, ymax = ax.get_ylim()
    y_annot = ymin + 0.05 * (ymax - ymin)
    for i, n in enumerate(ns, start=1):
        ax.text(i, y_annot, f"n={n}", ha="center", va="bottom", fontsize=9, alpha=0.85)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[saved] {out_path}")


def _safe_filename(s: str) -> str:
    s = str(s).strip().lower()
    out = []
    for ch in s:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        elif ch in {" ", "/", "\\"}:
            out.append("_")
    return "".join(out) or "dataset"


def plot_overall_shift_per_dataset(
    recs: List[Record],
    *,
    out_dir: str,
    file_tag: str,
    d_ref: float = 0.5,
    min_runs_per_dataset: int = 1,
    datasets: Optional[List[str]] = None,
) -> None:
    """
    Generate one violin plot per dataset showing the distribution of Δ (Cohen's d)
    across runs for each H0 feature.
    """
    if len(recs) == 0:
        raise SystemExit("No records found.")

    ds_all = sorted(set(r.dataset for r in recs))
    ds_keep = ds_all if not datasets else [d for d in datasets if d in set(ds_all)]
    os.makedirs(out_dir, exist_ok=True)

    for ds in ds_keep:
        sub = [r for r in recs if r.dataset == ds]
        if len(sub) == 0:
            continue

        # Require at least N distinct runs (trial directories) to avoid noisy single-run plots.
        n_runs = len(set(r.run_dir for r in sub))
        if n_runs < int(min_runs_per_dataset):
            continue

        out_path = os.path.join(out_dir, f"{_safe_filename(file_tag)}_feature_shift_overall__{_safe_filename(ds)}.png")
        title = f"H0 feature shift under attacks ({ds})"
        plot_overall_shift_for_subset(sub, datasets_keep=[ds], out_path=out_path, title=title, d_ref=float(d_ref))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default="out", help="Root output directory to scan (default: out)")
    ap.add_argument("--dim", type=int, default=0, choices=[0, 1], help="Homology dimension to analyze (0=H0, 1=H1). Default: 0.")
    ap.add_argument("--min-adv", type=int, default=5, help="Require at least this many adversarial samples for a run (default: 5)")
    ap.add_argument("--auc-good", type=float, default=0.7, help="AUROC threshold to mark as 'good separation' (default: 0.7)")
    ap.add_argument("--save", default="out/_analysis/h0_feature_shift_summary.png", help="Path to save plot PNG")
    ap.add_argument(
        "--mode",
        choices=["auc_by_dataset", "overall_shift", "overall_shift_split", "overall_shift_per_dataset"],
        default="overall_shift",
        help="Which report figure to generate (default: overall_shift)",
    )
    ap.add_argument("--save-shift", default="out/_analysis/h0_feature_shift_overall.png", help="Save path for overall shift plot")
    ap.add_argument("--save-shift-yes", default="out/_analysis/h0_feature_shift_overall_yes.png", help="Save path (datasets with meaningful H0 shift)")
    ap.add_argument("--save-shift-no", default="out/_analysis/h0_feature_shift_overall_no.png", help="Save path (datasets without meaningful H0 shift)")
    ap.add_argument(
        "--save-per-dataset-dir",
        default=None,
        help="Directory to save per-dataset plots (mode=overall_shift_per_dataset). "
        "Default: <out-root>/_analysis/h0_feature_shift_per_dataset/",
    )
    ap.add_argument(
        "--per-dataset-min-runs",
        type=int,
        default=5,
        help="Min distinct runs per dataset to emit a per-dataset plot (default: 5)",
    )
    ap.add_argument(
        "--datasets",
        default="",
        help="Optional comma-separated dataset allowlist for per-dataset mode (e.g. 'mnist,tabular'). Default: all.",
    )
    ap.add_argument("--d-ref", type=float, default=0.5, help="Reference |d| line for 'moderate shift' (default: 0.5)")
    ap.add_argument("--split-min-runs", type=int, default=5, help="Min runs per dataset for splitting (default: 5)")
    args = ap.parse_args()

    keys_default, _labels = _keys_and_labels_for_dim(int(args.dim))
    recs = collect_records(out_root=str(args.out_root), keys=list(keys_default), min_adv=int(args.min_adv), dim=int(args.dim))
    print("records:", len(recs))

    # Quick textual summary: for each key, fraction of runs with AUROC>=threshold
    for key in keys_default:
        a = np.array([r.auc for r in recs if r.key == key], dtype=float)
        if a.size == 0:
            continue
        frac = float(np.mean(a >= float(args.auc_good)))
        med = float(np.median(a))
        print(f"{key:26s} n={a.size:5d} median_auc={med:.3f} frac_auc>={args.auc_good:.2f}={frac:.3f}")

    if str(args.mode) == "auc_by_dataset":
        plot_summary(recs, out_path=str(args.save), auc_good=float(args.auc_good))
    elif str(args.mode) == "overall_shift":
        plot_overall_shift(recs, out_path=str(args.save_shift), d_ref=float(args.d_ref))
    elif str(args.mode) == "overall_shift_per_dataset":
        dim_tag = f"h{int(args.dim)}"
        out_dir = (
            str(args.save_per_dataset_dir)
            if args.save_per_dataset_dir
            else os.path.join(str(args.out_root), "_analysis", f"{dim_tag}_feature_shift_per_dataset")
        )
        ds_allow = [d.strip() for d in str(args.datasets).split(",") if d.strip()]
        plot_overall_shift_per_dataset(
            recs,
            out_dir=out_dir,
            file_tag=dim_tag,
            d_ref=float(args.d_ref),
            min_runs_per_dataset=int(args.per_dataset_min_runs),
            datasets=ds_allow if ds_allow else None,
        )
    else:
        dim = int(args.dim)
        split_keys = [f"topo_h{dim}_total_persistence", f"topo_h{dim}_l2_persistence", f"topo_h{dim}_max_persistence"]
        shift, no_shift, med_by, n_by = split_datasets_by_h0_shift(
            recs,
            keys_for_split=split_keys,
            d_ref=float(args.d_ref),
            min_runs_per_dataset=int(args.split_min_runs),
        )
        print("\nDataset split by median max(|d|) over H0 magnitude features:")
        for ds in sorted(med_by.keys()):
            print(f"  {ds:15s} median_max|d|={med_by[ds]:.3f} (n_runs={n_by.get(ds)})  modality={_dataset_to_modality(ds)}")
        print("SHIFT datasets:", shift)
        print("NO-SHIFT datasets:", no_shift)

        if shift:
            plot_overall_shift_for_subset(
                recs,
                datasets_keep=shift,
                out_path=str(args.save_shift_yes),
                title="Overall H0 feature shift (datasets with meaningful shift)",
                d_ref=float(args.d_ref),
            )
        if no_shift:
            plot_overall_shift_for_subset(
                recs,
                datasets_keep=no_shift,
                out_path=str(args.save_shift_no),
                title="Overall H0 feature shift (datasets with no meaningful shift)",
                d_ref=float(args.d_ref),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

