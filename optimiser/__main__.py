from __future__ import annotations

"""
This file just here for exposing entrypoints for the makefile to get proper CLI functionality.
"""

from optimisers.__main__ import main as _main


def main() -> None:
    _main()


if __name__ == "__main__":
    main()

