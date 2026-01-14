"""
Runner for configs under `config/IMAGE/`.

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

    # Dataset/model mapping for this config subtree (see `model_config.md`).
    run_and_optionally_emit_results(
        subdir="IMAGE",
        dataset_name="IMAGE",
        model_name="CNN",
        args=args,
    )


if __name__ == "__main__":
    main()

