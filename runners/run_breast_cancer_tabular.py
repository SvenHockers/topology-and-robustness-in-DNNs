"""
Runner for configs under `config/breast_cancer_tabular/`.

This is a thin CLI wrapper around `runners.runner_lib`.
"""

from __future__ import annotations

from pathlib import Path
import sys


def main() -> None:
    # Ensure repo root is importable even when invoked from another CWD.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from runners.runner_lib import build_arg_parser, run_and_optionally_emit_results

    parser = build_arg_parser()
    args = parser.parse_args()

    run_and_optionally_emit_results(
        subdir="breast_cancer_tabular",
        dataset_name="breast_cancer_tabular",
        model_name="MLP",
        args=args,
    )


if __name__ == "__main__":
    main()

