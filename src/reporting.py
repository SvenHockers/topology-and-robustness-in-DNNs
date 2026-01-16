"""
Compatibility shim.

`src.reporting` is not used by the repo's deployable entrypoints, but it may be
imported by external code. The implementation lives in `archive/src_unused/`.
"""

from __future__ import annotations

from archive.src_unused.reporting import *  # noqa: F401,F403
