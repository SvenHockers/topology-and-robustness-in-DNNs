"""
Generate YAML experiment configs under config/<dataset>/...

This script is optional convenience: it regenerates the same structure we created:
  - config/<dataset>/base.yaml
  - config/<dataset>/sweeps/run_00.yaml .. (many)

All YAML files are compatible with `ExperimentConfig.from_yaml(...)` (supports `base:` inheritance).

Notes:
  - The repo supports optional OOD evaluation when `cfg.ood.enabled: true`.
  - This generator can emit a small set of OOD sweep YAMLs so you can test OOD
    behavior alongside adversarial evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List


def _fmt_scalar(v: Any) -> str:
    """
    Format YAML scalars with a stable, human-friendly style.

    - Small floats are written in scientific notation (e.g. 1e-06) to match how
      these configs are typically edited by hand.
    - Other scalars fall back to `str(v)`.
    """
    if isinstance(v, float):
        if v != 0.0 and abs(v) < 1e-3:
            # e.g. 0.000001 -> 1e-06
            return f"{v:.0e}"
        # Keep concise decimals (e.g. 0.10 -> 0.1)
        s = f"{v:.12g}"
        return s
    return str(v)


def _dump_yaml_text(d: Dict[str, Any], indent: int = 0) -> str:
    """Minimal YAML writer (avoids requiring PyYAML in the generator)."""
    lines: List[str] = []
    pad = " " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{pad}{k}:")
            lines.append(_dump_yaml_text(v, indent=indent + 2))
        elif isinstance(v, list):
            # inline list for simplicity
            inner = ", ".join(_fmt_scalar(x) for x in v)
            lines.append(f"{pad}{k}: [{inner}]")
        elif isinstance(v, bool):
            lines.append(f"{pad}{k}: {'true' if v else 'false'}")
        else:
            lines.append(f"{pad}{k}: {_fmt_scalar(v)}")
    return "\n".join(lines)


def write_yaml(path: Path, d: Dict[str, Any], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return
    path.write_text(_dump_yaml_text(d) + "\n", encoding="utf-8")


BASES: Dict[str, Dict[str, Any]] = {
    "breast_cancer_tabular": {
        "seed": 42,
        "device": "cpu",
        "model": {
            "hidden_dims": [128, 64],
            "activation": "relu",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "epochs": 80,
            "batch_size": 64,
        },
        "attack": {
            "attack_type": "pgd",
            "epsilon": 0.10,
            "num_steps": 30,
            "step_size": 0.01,
            "random_start": True,
        },
        "graph": {
            "space": "feature",
            "feature_layer": "penultimate",
            "use_topology": True,
            "use_tangent": False,
            # "k in kNN" (used by tangent + knn-radius); keep aligned with topo_k in sweeps
            "k": 40,
            "tangent_k": 40,
            "topo_k": 40,
            "topo_maxdim": 1,
            "topo_preprocess": "pca",
            "topo_pca_dim": 10,
            "topo_min_persistence": 0.000001,
        },
        "detector": {
            "detector_type": "topology_score",
            "topo_percentile": 95.0,
            "topo_cov_shrinkage": 0.001,
        },
    },
    "mnist": {
        "seed": 42,
        "device": "cpu",
        "data": {"root": "./data", "download": False, "train_ratio": 0.9, "val_ratio": 0.1},
        "model": {
            "output_dim": 10,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "epochs": 20,
            "batch_size": 64,
        },
        "attack": {
            "attack_type": "pgd",
            "epsilon": 0.10,
            "num_steps": 20,
            "step_size": 0.01,
            "random_start": True,
        },
        "graph": {
            "space": "feature",
            "feature_layer": "penultimate",
            "use_topology": True,
            "use_tangent": False,
            "k": 40,
            "tangent_k": 40,
            "topo_k": 40,
            "topo_maxdim": 1,
            "topo_preprocess": "pca",
            "topo_pca_dim": 16,
            "topo_min_persistence": 0.000001,
        },
        "detector": {"detector_type": "topology_score", "topo_percentile": 95.0, "topo_cov_shrinkage": 0.001},
    },
    "geometrical-shapes": {
        "seed": 42,
        "device": "cpu",
        "model": {
            "hidden_dims": [256, 128],
            "activation": "relu",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "epochs": 40,
            "batch_size": 64,
        },
        "attack": {"attack_type": "pgd", "epsilon": 0.30, "num_steps": 50, "step_size": 0.03, "random_start": True},
        "graph": {
            "space": "input",
            "feature_layer": "penultimate",
            "use_topology": True,
            "use_tangent": False,
            "k": 150,
            "tangent_k": 150,
            "topo_k": 150,
            "topo_maxdim": 2,
            "topo_preprocess": "none",
            "topo_pca_dim": 3,
            "topo_min_persistence": 0.000001,
        },
        "detector": {"detector_type": "topology_score", "topo_percentile": 95.0, "topo_cov_shrinkage": 0.001},
    },
}


def ood_runs(dataset: str) -> List[Dict[str, Any]]:
    """
    Return a small set of realistic OOD configurations per dataset.

    The OOD generators live in `src/OOD.py` and are enabled in the pipeline when
    `cfg.ood.enabled` is true.
    """
    if dataset == "mnist":
        # Realistic digit corruptions.
        presets: List[Dict[str, Any]] = [
            {"method": "gaussian_noise", "severity": 1.0},
            {"method": "blur", "severity": 1.0, "blur_kernel_size": 5, "blur_sigma": 1.0},
            {"method": "patch_shuffle", "severity": 1.0, "patch_size": 4},
        ]
    elif dataset == "breast_cancer_tabular":
        # Realistic tabular shifts: correlation break, measurement noise, support widening.
        presets = [
            {"method": "feature_shuffle", "severity": 1.0},
            {"method": "gaussian_noise", "severity": 0.5},
            {"method": "uniform_wide", "severity": 0.25},
        ]
    elif dataset == "geometrical-shapes":
        # Point clouds (treated as vectors): geometry/sensor-like shifts.
        presets = [
            {"method": "gaussian_noise", "severity": 0.15},
            {"method": "extrapolate", "severity": 0.5},
            {"method": "uniform_wide", "severity": 0.25},
        ]
    else:
        raise KeyError(dataset)

    runs: List[Dict[str, Any]] = []
    for p in presets:
        runs.append({"base": "../base.yaml", "ood": {"enabled": True, **p}})
    return runs


def sweep_runs(dataset: str) -> List[Dict[str, Any]]:
    """
    9-run orthogonal-ish coverage over:
      - epsilon (3 levels)
      - k/topo_k (3 levels)
      - use_tangent (2 levels)
      - pca vs none (+ pca_dim)
      - min_persistence (3 levels)
      - percentile (90/95)
      - cov_shrinkage (1e-6/1e-4/1e-2)

    Then we **double** the set by emitting an FGSM copy of every PGD config
    (same parameters, only `attack.attack_type` changes).
    """
    if dataset == "breast_cancer_tabular":
        eps = [(0.05, 0.005), (0.10, 0.01), (0.20, 0.02)]
        # include a larger k option (PH neighborhoods can change a lot here)
        ks = [20, 40, 120]
        pca_dim = [10, 20]
    elif dataset == "mnist":
        eps = [(0.05, 0.005), (0.10, 0.01), (0.20, 0.02)]
        # include a larger k option (PH neighborhoods can change a lot here)
        ks = [20, 40, 120]
        pca_dim = [16, 32]
    elif dataset == "geometrical-shapes":
        eps = [(0.20, 0.02), (0.30, 0.03), (0.50, 0.05)]
        ks = [60, 150, 220]
        pca_dim = [2, 3]
    else:
        raise KeyError(dataset)

    minp = [0.000001, 0.00001, 0.0001]
    shrink = [0.01, 0.000001, 0.0001]
    perc = [95.0, 90.0]
    # Manually encode the same 9 patterns used in the checked-in files.
    base_runs: List[Dict[str, Any]] = []
    patterns = [
        # (eps_idx, k_idx, tangent, preprocess, pca_idx, minp_idx, perc_idx, shrink_idx)
        (0, 0, True, "none", 0, 1, 0, 0),
        (1, 0, False, "pca", 0, 0, 1, 1),
        (2, 0, True, "pca", 1, 2, 0, 2),
        (0, 1, False, "pca", 0, 1, 1, 0),
        (1, 1, True, "pca", 1, 0, 0, 1),
        (2, 1, False, "none", 0, 2, 1, 2),
        (0, 2, True, "pca", 1, 1, 0, 0),
        (1, 2, False, "none", 0, 0, 1, 1),
        (2, 2, True, "pca", 0, 2, 0, 2),
    ]

    for (ei, ki, tan, prep, pi, mi, pe, si) in patterns:
        e, step = eps[ei]
        k = ks[ki]
        r: Dict[str, Any] = {
            "base": "../base.yaml",
            # attack_type is set in the expanded runs below (pgd + fgsm copy)
            "attack": {"epsilon": e, "step_size": step},
            "graph": {
                "k": k,
                "topo_k": k,
                "use_tangent": bool(tan),
                "tangent_k": k,
                "topo_preprocess": prep,
                "topo_min_persistence": minp[mi],
            },
            "detector": {"topo_percentile": perc[pe], "topo_cov_shrinkage": shrink[si]},
        }
        if prep == "pca":
            r["graph"]["topo_pca_dim"] = pca_dim[pi]
        base_runs.append(r)

    # Expand: for every base run, emit (1) PGD config and (2) FGSM copy.
    expanded: List[Dict[str, Any]] = []
    for r in base_runs:
        # PGD version
        pgd = {**r, "attack": {**r["attack"], "attack_type": "pgd"}}
        expanded.append(pgd)

        # FGSM version (same epsilon, but set FGSM-specific knobs)
        e = float(r["attack"]["epsilon"])
        fgsm_attack = {
            **r["attack"],
            "attack_type": "fgsm",
            "num_steps": 1,
            "random_start": False,
            # step_size isn't used by FGSM; set equal to epsilon for readability
            "step_size": e,
        }
        fgsm = {**r, "attack": fgsm_attack}
        expanded.append(fgsm)

    return expanded


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", help="Dataset key or 'all'")
    ap.add_argument("--config-dir", default="config", help="Root config directory")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing YAML files (default: keep existing, skip writes).",
    )
    ap.add_argument(
        "--no-ood",
        action="store_true",
        help="Do not emit additional OOD sweep configs (default: include OOD configs).",
    )
    args = ap.parse_args()

    root = Path(args.config_dir)
    datasets = list(BASES.keys()) if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        if ds not in BASES:
            raise SystemExit(f"Unknown dataset: {ds}. Known: {sorted(BASES.keys())}")
        ds_dir = root / ds
        write_yaml(ds_dir / "base.yaml", BASES[ds], overwrite=bool(args.overwrite))
        sweep_dir = ds_dir / "sweeps"
        runs = list(sweep_runs(ds))
        if not bool(args.no_ood):
            runs.extend(ood_runs(ds))
        for i, run in enumerate(runs):
            write_yaml(sweep_dir / f"run_{i:02d}.yaml", run, overwrite=bool(args.overwrite))

    print("Generated configs under:", root.resolve())


if __name__ == "__main__":
    main()

