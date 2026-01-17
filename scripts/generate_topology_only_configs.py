"""
Generate "topology_only" config variants for each dataset under config/final/.

Rationale:
- Existing configs compute baseline (degree/laplacian/tangent/...) scores by default.
- Even if the detector uses topology features, baseline computations can make it hard
  to reason about isolation and can confuse comparisons.
- This script creates a sibling directory per dataset: config/final/<dataset>/topology_only/
  with YAMLs that inherit from the existing base*.yaml and override:
    graph.use_topology = true
    graph.use_baseline_scores = false

Usage:
  python scripts/generate_topology_only_configs.py
"""

from __future__ import annotations

from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    final_dir = repo_root / "config" / "final"
    if not final_dir.exists():
        raise SystemExit(f"Missing expected directory: {final_dir}")

    datasets = [p for p in final_dir.iterdir() if p.is_dir()]
    created = 0

    for ds_dir in sorted(datasets, key=lambda p: p.name):
        # Consider only base YAMLs in the dataset root (not baseline/ or OOD/).
        base_yamls = sorted(
            [
                p
                for p in ds_dir.glob("base*.yaml")
                if p.is_file() and p.parent == ds_dir
            ],
            key=lambda p: p.name,
        )
        if not base_yamls:
            continue

        out_dir = ds_dir / "topology_only"
        out_dir.mkdir(parents=True, exist_ok=True)

        for base_yaml in base_yamls:
            target = out_dir / base_yaml.name
            rel = Path("..") / base_yaml.name  # relative from topology_only/ -> dataset root
            content = (
                f"base: {rel.as_posix()}\n"
                f"graph:\n"
                f"  use_topology: true\n"
                f"  use_baseline_scores: false\n"
            )

            # Only write if missing or content differs (idempotent).
            if target.exists():
                old = target.read_text(encoding="utf-8")
                if old == content:
                    continue

            target.write_text(content, encoding="utf-8")
            created += 1

    print(f"Wrote/updated {created} topology_only config files under {final_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

