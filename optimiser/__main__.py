from __future__ import annotations

"""
Compatibility entrypoint.

This forwards to the canonical `optimisers` CLI so both of the following work:
  - python -m optimisers ...
  - python -m optimiser ...
"""

from optimisers.__main__ import main as _main


def main() -> None:
    _main()


if __name__ == "__main__":
    main()

