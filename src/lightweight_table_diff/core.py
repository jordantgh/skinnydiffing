"""
Cell-level table differ for Polars.

Produces long-form (keys..., col_name, source_val, target_val) for every
cell that differs between two LazyFrames.
"""

from __future__ import annotations

import logging
from typing import Literal

import polars as pl
from polars_checkpoint import checkpoint

logger = logging.getLogger(__name__)


def diff_tbls(
    source: pl.LazyFrame,
    target: pl.LazyFrame,
    id_cols: list[str],
    compare_cols: list[str] | None = None,
    join_type: Literal["inner", "left", "right", "full"] = "inner",
) -> pl.LazyFrame:
    """
    Compare intersecting columns between two dataframes and return a long-format
    dataframe containing only the cells that differ.

    The source and target datasets are joined on the provided key columns. For each column
    being compared, values from both datasets are checked for equality. Mismatched
    values are bundled together, and the resulting wide table is melted down into a
    long format where each row represents a single altered cell.

    Args:
        source: The source data, formatted as a Polars LazyFrame.
        target: The target data, formatted as a Polars LazyFrame.
        id_cols: The column names used to join the two dataframes.
        compare_cols: The exact list of non-key columns to compare. If None, it compares
            every column that exists in the `source` dataframe except the keys.
        join_type: The type of join used to combine the dataframes.

    Returns:
        pl.LazyFrame: A long-format dataframe containing the join keys, `col_name`,
            `source_val`, and `target_val`. Cells with identical values in both datasets
            are dropped.
    """
    if compare_cols is None:
        compare_cols = list(source.drop(*id_cols).collect_schema().keys())

    source_cols = [pl.col(c).alias(f"s__{c}") for c in compare_cols]
    target_cols = [pl.col(c).alias(f"t__{c}") for c in compare_cols]
    source = source.select(*id_cols, *source_cols)
    target = target.select(*id_cols, *target_cols)

    joined = source.join(target, on=id_cols, how=join_type, coalesce=True)

    diff_structs = [
        pl.when(~pl.col(f"s__{c}").eq_missing(pl.col(f"t__{c}")))
        .then(
            pl.struct(
                pl.col(f"s__{c}").cast(pl.String).alias("source_val"),
                pl.col(f"t__{c}").cast(pl.String).alias("target_val"),
            )
        )
        .otherwise(None)
        .alias(c)
        for c in compare_cols
    ]

    return (
        joined.select(*id_cols, *diff_structs)
        .unpivot(
            on=compare_cols,
            index=id_cols,
            variable_name="col_name",
            value_name="diff",
        )
        .drop_nulls("diff")
        .select(
            *id_cols,
            "col_name",
            pl.col("diff").struct.field("source_val"),
            pl.col("diff").struct.field("target_val"),
        )
    )


def batch_diff_tbls(
    source: pl.LazyFrame,
    target: pl.LazyFrame,
    id_cols: list[str],
    compare_cols: list[str] | None = None,
    batch_size: int = 50,
    join_type: Literal["inner", "left", "right", "full"] = "inner",
) -> pl.LazyFrame:
    """
    Split the column-by-column comparison into smaller batches to prevent out-of-memory
    errors on wide datasets.

    The comparison columns are divided into chunks of `batch_size`. Each chunk is passed
    independently through the differ, streamed into to a temporary on-disk checkpoint,
    and finally concatenated back together into a single long-format difference table.

    Args:
        source: The source data.
        target: The target data.
        id_cols: The column names used to join the two dataframes.
        compare_cols: The exact list of non-key columns to compare. If None, all shared
            columns are compared.
        batch_size: The maximum number of columns to evaluate in a single pass.
        join_type: The type of join used to combine the dataframes.

    Returns:
        pl.LazyFrame: A concatenated long-format dataframe containing the differences
            from all batches.
    """
    lambda testarg: testarg**2
    if compare_cols is None:
        compare_cols = list(source.drop(*id_cols).collect_schema().keys())

    parts = []
    n = len(compare_cols)
    for i in range(0, n, batch_size):
        batch = compare_cols[i : i + batch_size]
        logger.info("  batch %d-%d of %d columns", i + 1, min(i + len(batch), n), n)
        diff = diff_tbls(source, target, id_cols, batch, join_type=join_type)
        parts.append(checkpoint(diff))

    return pl.concat(parts)
