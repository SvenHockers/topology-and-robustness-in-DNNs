from __future__ import annotations

import sys
from typing import List, Optional


def _print_help() -> None:
    sys.stdout.write(
        "Topology & robustness optimiser\n\n"
        "Single command entrypoint:\n"
        "  python -m optimisers <args>\n\n"
        "Subcommands:\n"
        "  batch         Run batch GP optimisation over a config directory\n"
        "  plot-history  Plot figures from a study history.jsonl\n\n"
        "Default behavior (no subcommand):\n"
        "  Runs single-config GP optimisation (same args as `optimisers.cli`).\n\n"
        "Examples:\n"
        "  python -m optimisers --help\n"
        "  python -m optimisers --base-config <cfg> --dataset-name TABULAR --model-name MLP --space optimisers/spaces/constrains.yaml\n"
        "  python -m optimisers batch --config-dir config/final/tabular --dataset-name TABULAR --model-name MLP --space optimisers/spaces/constrains.yaml\n"
        "  python -m optimisers plot-history --history <study>/history.jsonl --outdir <study>/figs --space optimisers/spaces/constrains.yaml\n"
    )


def main(argv: Optional[List[str]] = None) -> None:
    args = sys.argv[1:] if argv is None else list(argv)

    if not args or args[0] in {"-h", "--help", "help"}:
        _print_help()
        return

    cmd = str(args[0]).lower()
    if cmd in {"batch"}:
        from optimisers.cli_batch import main as batch_main

        batch_main(args[1:])
        return
    if cmd in {"plot-history", "plot_history", "plot"}:
        from optimisers.plot_history import main as plot_main

        plot_main(args[1:])
        return

    # Fallback: treat argv as single-config optimiser args.
    from optimisers.cli import main as single_main

    single_main(args)


if __name__ == "__main__":
    main()

