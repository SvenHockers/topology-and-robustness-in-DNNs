"""
Gaussian-process Bayesian optimisation for this repo's runner pipeline.

The optimiser reuses `runners/runner_lib.py` to execute trials so dataset/model
construction and artifact writing stay consistent with existing runners.
"""

from __future__ import annotations

