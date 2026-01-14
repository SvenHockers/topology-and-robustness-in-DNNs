"""
Runner for configs under `config/IMAGE/`.

This filename matches the `config/` subdir name so `runners/run_all.py` can
dispatch to it automatically.
"""

from __future__ import annotations

from pathlib import Path
import sys


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from runners.runner_lib import build_arg_parser, run_and_optionally_emit_results

    parser = build_arg_parser()
    args = parser.parse_args()

    run_and_optionally_emit_results(
        subdir="IMAGE",
        dataset_name="IMAGE",
        model_name="CNN",
        args=args,
    )


if __name__ == "__main__":
    main()

