"""
Compatibility shim for the production CLI entrypoint.

The "real" implementation is `optimiser.cli` (British spelling). This module exists so:

  python -m optimizers.cli ...

works unchanged.
"""

from __future__ import annotations

from typing import Optional

from optimiser.cli import build_arg_parser as build_arg_parser
from optimiser.cli import main as _main


def main(argv: Optional[list[str]] = None) -> None:
    _main(argv)


if __name__ == "__main__":
    main()

