"""
Compatibility shim for the production CLI entrypoint.

The "real" implementation is `optimiser.cli_batch` (British spelling, singular).
This module exists so:

  python -m optimisers.cli_batch ...

works unchanged.
"""

from __future__ import annotations

from typing import Optional

from optimiser.cli_batch import build_arg_parser as build_arg_parser
from optimiser.cli_batch import main as _main


def main(argv: Optional[list[str]] = None) -> None:
    _main(argv)


if __name__ == "__main__":
    main()

