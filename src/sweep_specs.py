"""
Compatibility shim.

`src.sweep_specs` is not used by the repo's deployable entrypoints, but it may be
imported by external code. The implementation lives in `archive/src_unused/`.
"""

from __future__ import annotations

from archive.src_unused.sweep_specs import *  # noqa: F401,F403
