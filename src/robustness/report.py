from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
import csv
import json
import os
import numpy as np


@dataclass
class SampleRecord:
    sample_id: int
    true_label: int
    clean_pred: int
    clean_margin: float
    eps_star_linf: Optional[float] = None
    adv_pred_linf: Optional[int] = None
    eps_star_l2: Optional[float] = None
    adv_pred_l2: Optional[int] = None
    rot_deg_star: Optional[float] = None
    trans_x_star: Optional[float] = None
    trans_y_star: Optional[float] = None
    trans_z_star: Optional[float] = None
    jitter_std_star: Optional[float] = None
    dropout_star: Optional[float] = None
    alpha_star: Optional[float] = None
    chamfer_clean_adv: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LayerwiseRecord:
    sample_id: int
    condition: str  # 'clean','adv_linf_eps=0.1', 'rotation_deg=10', ...
    layer: str
    betti: str  # 'H0','H1'
    count: float
    mean_persistence: float
    max_persistence: float
    total_persistence: float

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiagramDistanceRecord:
    sample_id: int
    condition: str
    layer: str
    metric: str  # 'wasserstein' | 'bottleneck'
    H: int
    distance: float

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RobustnessSummary:
    # aggregates
    mean_eps_star_linf: Optional[float] = None
    mean_eps_star_l2: Optional[float] = None
    ra_curves: Dict[str, List[float]] = field(default_factory=dict)  # norm -> list over eps grid
    per_class: Dict[str, Any] = field(default_factory=dict)
    layerwise_avg: Dict[str, Any] = field(default_factory=dict)
    correlations: Dict[str, Any] = field(default_factory=dict)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


def write_csv(records: List[Dict[str, Any]], path: str) -> None:
    if not records:
        return
    keys = sorted(records[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in records:
            w.writerow(r)


def save_curve_png(xs: List[float], ys: List[float], title: str, path: str) -> None:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.title(title)
    plt.xlabel("epsilon")
    plt.ylabel("robust accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_hist_png(values: List[float], title: str, path: str, bins: int = 30) -> None:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist([v for v in values if v is not None and not np.isnan(v)], bins=bins, alpha=0.8)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_layer_distance_bar(avg_by_layer: Dict[str, float], title: str, path: str) -> None:
    import matplotlib.pyplot as plt
    layers = list(avg_by_layer.keys())
    vals = [avg_by_layer[k] for k in layers]
    plt.figure(figsize=(max(6, len(layers)), 4))
    plt.bar(layers, vals)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_betti_counts_bar(layerwise_records: List[LayerwiseRecord], H: int, path: str, title: str | None = None) -> None:
    import matplotlib.pyplot as plt
    from collections import defaultdict
    agg = defaultdict(list)
    for r in layerwise_records:
        if r.betti == f"H{H}":
            agg[r.layer].append(r.count)
    if not agg:
        return
    layers = sorted(agg.keys())
    vals = [float(sum(agg[l]) / max(len(agg[l]), 1)) for l in layers]
    plt.figure(figsize=(max(6, len(layers)), 4))
    plt.bar(layers, vals, color="tab:blue" if H == 0 else "tab:orange")
    plt.title(title or f"Average Betti H{H} counts per layer")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


