from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .api import diff
from .config import load_config
from .result import DiffResult

logger = logging.getLogger(__name__)


def run_comparison(
    job: dict[str, Any],
    *,
    loaders: Mapping[str, Callable[[Any], Any]] | None = None,
    collect_lazy_inputs: bool | None = None,
) -> DiffResult:
    """
    Execute a single comparison job defined by a configuration dictionary.

    The configuration parameters are unpacked and passed directly to the main `diff`
    function. High-level summaries of the source-only columns, target-only columns, and
    total cell differences are logged to the console.

    Args:
        job: A dictionary containing the arguments required by the `diff` function
            (e.g., `source`, `target`, `keys`).
        loaders: A dictionary mapping a string name to a custom loading function.
        collect_lazy_inputs: Whether to force non-Polars lazy dataframes to fully load
            into memory before converting them. Overrides the value specified inside
            the `job` dictionary if provided.

    Returns:
        DiffResult: The complete output of the comparison.
    """
    name = job.get("name", "unnamed")
    logger.info("Running comparison: %s", name)

    result = diff(
        job["source"],
        job["target"],
        keys=job["keys"],
        compare=job.get("compare"),
        exclude=job.get("exclude"),
        normalise=job.get("normalise"),
        batch_size=job.get("batch_size", 50),
        check_keys=job.get("check_keys", True),
        source_options=job.get("source_options"),
        target_options=job.get("target_options"),
        loaders=loaders,
        collect_lazy_inputs=(
            job.get("collect_lazy_inputs", False)
            if collect_lazy_inputs is None
            else collect_lazy_inputs
        ),
        name=name,
    )

    if result.source_only_cols:
        logger.info(
            "  %d source-only column(s): %s",
            len(result.source_only_cols),
            result.source_only_cols,
        )
    if result.target_only_cols:
        logger.info(
            "  %d target-only column(s): %s",
            len(result.target_only_cols),
            result.target_only_cols,
        )

    logger.info("  %d difference(s) found", result.n_diffs)
    return result


def run_config(
    config_path: str | Path,
    *,
    loaders: Mapping[str, Callable[[Any], Any]] | None = None,
    collect_lazy_inputs: bool | None = None,
) -> list[DiffResult]:
    """
    Execute multiple comparison jobs defined in a YAML configuration file.

    The YAML file is parsed and expanded into individual, standalone job configurations.
    Each job is run sequentially, and its outputs are automatically written to CSV files
    in the specified output directory.

    Args:
        config_path: The file path to the YAML configuration.
        loaders: A dictionary mapping a string name to a custom loading function.
        collect_lazy_inputs: Whether to force non-Polars lazy dataframes to fully load
            into memory.

    Returns:
        list[DiffResult]: A list containing the results of every job executed.
    """
    jobs = load_config(config_path)
    results: list[DiffResult] = []
    for job in jobs:
        result = run_comparison(
            job,
            loaders=loaders,
            collect_lazy_inputs=collect_lazy_inputs,
        )
        result.write(job.get("output_dir", "./diff_output"))
        results.append(result)
    return results
