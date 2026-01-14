"""
Master runner: execute pipeline across *all* immediate subdirectories of `config/`.

Behavior:
  - Discover immediate subdirs under `--config-root`.
  - For each subdir, invoke `runners/run_<subdir>.py` via subprocess if it exists.
    (Subprocess is used so config subdirs with hyphens can map to runnable filenames.)
  - Collect per-run rows and write aggregate success-count outputs to:
      outputs/_aggregate/aggregate_success_counts.{csv,json}
  - Continue even if a subdir runner fails; failures are included with null metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import subprocess
import sys


def _forwarded_args(argv: Any) -> List[str]:
    """
    Forward the shared CLI args to sub-runners.
    """
    out: List[str] = []
    out += ["--config-root", str(argv.config_root)]
    out += ["--output-root", str(argv.output_root)]
    out += ["--extensions", str(argv.extensions)]
    for pat in (argv.ignore or []):
        out += ["--ignore", str(pat)]
    if getattr(argv, "device", None):
        out += ["--device", str(argv.device)]
    if bool(getattr(argv, "enable_latex", False)):
        out += ["--enable-latex"]
    if bool(argv.dry_run):
        out += ["--dry-run"]
    out += ["--max-workers", str(int(argv.max_workers))]
    out += ["--export-features", str(argv.export_features)]
    if bool(argv.verbose):
        out += ["--verbose"]
    # Match per-subdir runner defaults/flags.
    if not bool(getattr(argv, "filter_clean_to_correct", True)):
        out += ["--no-filter-clean-to-correct"]
    if getattr(argv, "max_points_for_scoring", None) is not None:
        out += ["--max-points-for-scoring", str(int(argv.max_points_for_scoring))]
    return out


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from runners.runner_lib import (
        RunRow,
        build_arg_parser,
        compute_run_dir,
        discover_configs,
        parse_extensions,
        run_subdir,
        write_aggregate,
    )

    parser = build_arg_parser()
    args = parser.parse_args()

    config_root = Path(str(args.config_root)).resolve()
    output_root = Path(str(args.output_root)).resolve()
    exts = parse_extensions(args.extensions)
    ignore = list(args.ignore or [])

    subdirs = sorted([p.name for p in config_root.iterdir() if p.is_dir()], key=lambda s: s.lower())

    agg_tmp = output_root / "_aggregate" / "_tmp"
    agg_tmp.mkdir(parents=True, exist_ok=True)

    all_rows: List[RunRow] = []

    for subdir in subdirs:
        runner_path = repo_root / "runners" / f"run_{subdir}.py"
        results_json = agg_tmp / f"runs_{subdir}.json"

        # Prefer the dedicated per-subdir runner if present.
        if runner_path.exists():
            cmd = [sys.executable, str(runner_path), *_forwarded_args(args), "--results-json", str(results_json)]
            print(f"[SUB ] {subdir} -> {runner_path.name}")
            try:
                subprocess.run(cmd, cwd=str(repo_root), check=True)
            except subprocess.CalledProcessError as e:
                # Attempt to load partial results if any were written; otherwise, create failure rows.
                err = f"subdir runner failed (exit={e.returncode})"
                if results_json.exists():
                    try:
                        data = json.loads(results_json.read_text(encoding="utf-8"))
                        for d in data:
                            all_rows.append(RunRow(**d))
                        continue
                    except Exception:
                        pass

                # Fallback: discover configs and mark each as failed with null metrics.
                subdir_path = config_root / subdir
                cfgs = discover_configs(subdir_path, extensions=exts, ignore_globs=ignore)
                for p in cfgs:
                    out_dir = compute_run_dir(config_path=p, config_root=config_root, output_root=output_root)
                    all_rows.append(
                        RunRow(
                            config_path=str(p),
                            output_dir=str(out_dir),
                            status="failed",
                            duration_s=0.0,
                            adversarial_total=None,
                            adversarial_success=None,
                            ood_total=None,
                            ood_success=None,
                            notes=err,
                            error=err,
                        )
                    )
                continue

            # Load the per-subdir results file.
            try:
                data = json.loads(results_json.read_text(encoding="utf-8"))
                for d in data:
                    all_rows.append(RunRow(**d))
            except Exception as e:
                # If subrunner succeeded but results couldn't be parsed, treat as failure.
                err = f"failed to parse subdir results ({subdir}): {type(e).__name__}: {e}"
                subdir_path = config_root / subdir
                cfgs = discover_configs(subdir_path, extensions=exts, ignore_globs=ignore)
                for p in cfgs:
                    out_dir = compute_run_dir(config_path=p, config_root=config_root, output_root=output_root)
                    all_rows.append(
                        RunRow(
                            config_path=str(p),
                            output_dir=str(out_dir),
                            status="failed",
                            duration_s=0.0,
                            adversarial_total=None,
                            adversarial_success=None,
                            ood_total=None,
                            ood_success=None,
                            notes=err,
                            error=err,
                        )
                    )
            continue

        # Generic fallback (covers any number of subdirs even without a dedicated runner file).
        print(f"[SUB ] {subdir} -> (generic runner)")
        dataset_name = subdir
        # Normalize casing for robust matching (Windows folders are often uppercase).
        model_name = "CNN" if str(subdir).strip().lower() == "image" else "MLP"
        rows = run_subdir(
            subdir=subdir,
            config_root=config_root,
            output_root=output_root,
            extensions=exts,
            ignore_globs=ignore,
            dataset_name=dataset_name,
            model_name=model_name,
            device=None if getattr(args, "device", None) is None else str(args.device),
            enable_latex=bool(getattr(args, "enable_latex", False)),
            dry_run=bool(args.dry_run),
            max_workers=int(args.max_workers),
            export_features=str(args.export_features),  # type: ignore[arg-type]
            verbose=bool(args.verbose),
            filter_clean_to_correct=bool(getattr(args, "filter_clean_to_correct", True)),
            max_points_for_scoring=None
            if getattr(args, "max_points_for_scoring", None) is None
            else int(args.max_points_for_scoring),
        )
        all_rows.extend(rows)

    # Write aggregate outputs
    csv_path, json_path = write_aggregate(output_root=output_root, rows=all_rows)
    print(f"[AGG ] wrote {csv_path}")
    print(f"[AGG ] wrote {json_path}")


if __name__ == "__main__":
    main()

