"""Skinnydiffing — cell-level batched table diffing."""

from .api import diff, diff_lazyframes
from .core import batch_diff_tbls, diff_tbls
from .result import DiffResult
from .runner import run_comparison, run_config

__all__ = [
    "DiffResult",
    "diff",
    "diff_lazyframes",
    "diff_tbls",
    "batch_diff_tbls",
    "run_comparison",
    "run_config",
]