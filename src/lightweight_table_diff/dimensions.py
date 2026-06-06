"""Column/row checking and key-uniqueness validation."""

from __future__ import annotations

from typing import cast

import polars as pl

# TODO: Remove casts once https://github.com/astral-sh/ruff/pull/25030 is in ty release


def get_cols_to_compare(
    source: pl.LazyFrame,
    target: pl.LazyFrame,
    keys: list[str],
    include_cols: list[str] | None = None,
    exclude_cols: list[str] | None = None,
) -> list[str]:
    """
    Determine the final list of columns to compare cell-by-cell between two dataframes.

    The baseline list consists of all columns that exist in both the source and
    target dataframes, excluding the join keys. Columns that only exist on one side
    are ignored.

    If `include_cols` is provided, the list is restricted to only those columns (provided
    they exist in both dataframes). Finally, any columns listed in `exclude_cols` are
    excluded from the list.

    Args:
        source: The source dataframe.
        target: The target dataframe.
        keys: The list of join keys.
        include_cols: A specific list of columns to check. Defaults to None.
        exclude_cols: A specific list of columns to ignore. Defaults to None.

    Returns:
        list[str]: An alphabetical list of column names that exist in both dataframes,
            are not join keys, and have survived the include/exclude filters.

    Raises:
        ValueError: If any of the provided join keys are missing from either dataframe,
            or if the final list of columns to compare is empty.
    """
    source_names = set(source.collect_schema().names())
    target_names = set(target.collect_schema().names())
    key_set = set(keys)

    missing = key_set - (source_names & target_names)
    if missing:
        raise ValueError(f"Join key(s) missing from one or both sides: {missing}")

    if include_cols:
        cols = [
            c
            for c in include_cols
            if c in source_names and c in target_names and c not in key_set
        ]
    else:
        cols = sorted((source_names & target_names) - key_set)

    if exclude_cols:
        exclude_set = set(exclude_cols)
        cols = [c for c in cols if c not in exclude_set]

    if not cols:
        raise ValueError("No columns to compare after applying filters")

    return cols


def validate_key_uniqueness(
    source: pl.LazyFrame,
    target: pl.LazyFrame,
    keys: list[str],
    sample_limit: int = 20,
) -> None:
    """
    Verify that the provided join keys uniquely identify every row in both datasets.

    The datasets are grouped by the key columns and checked for any groups containing
    more than one row. If duplicates exist, an error is raised detailing the problematic
    keys to prevent a Cartesian row explosion during the subsequent join.

    Args:
        source: The source data.
        target: The target data.
        keys: The column names expected to uniquely identify a row.
        sample_limit: The maximum number of duplicate groups to display in the error
            message. Defaults to 20.

    Raises:
        ValueError: If duplicate key combinations are found in either dataset.
    """
    problems: list[str] = []
    for label, lf in [("source", source), ("target", target)]:
        dupes = (
            lf.select(*keys)
            .group_by(keys)
            .len()
            .filter(pl.col("len") > 1)
            .limit(sample_limit)
            .collect()
        )

        dupes = cast(pl.DataFrame, dupes)

        if dupes.height:
            problems.append(f"  {label}: {dupes.height} duplicate key group(s)\n{dupes}")

    if problems:
        raise ValueError(
            "Duplicate keys (would cause row explosion):\n" + "\n".join(problems)
        )


def column_presence_differences(
    source: pl.LazyFrame,
    target: pl.LazyFrame,
    keys: list[str],
) -> tuple[list[str], list[str]]:
    """
    Identify which non-key columns exist on only one side of the comparison.

    The schema names of both datasets are extracted and compared as sets, explicitly
    ignoring the join keys.

    Args:
        source: The source data.
        target: The target data.
        keys: The list of join keys to exclude from the check.

    Returns:
        tuple[list[str], list[str]]: Two alphabetically sorted lists containing the
            source-only column names and target-only column names, respectively.
    """
    key_set = set(keys)
    source_cols = set(source.collect_schema().names()) - key_set
    target_cols = set(target.collect_schema().names()) - key_set
    return sorted(source_cols - target_cols), sorted(target_cols - source_cols)


def type_differences(
    source: pl.LazyFrame, target: pl.LazyFrame, cols: list[str]
) -> pl.DataFrame:
    """
    Identify columns where the underlying data type differs between the two datasets.

    The schemas of the shared columns are compared. Discrepancies are recorded in a
    dataframe alongside the column name and the string representation of the source
    and target data types.

    Args:
        source: The source data.
        target: The target data.
        cols: The specific list of shared columns to check.

    Returns:
        pl.DataFrame: A dataframe containing columns for `col_name`, `source_type`,
            and `target_type`. Returns an empty dataframe if all types match.
    """

    source_schema = source.collect_schema()
    target_schema = target.collect_schema()

    diffs = []
    for c in cols:
        source_type, target_type = source_schema[c], target_schema[c]
        if source_type != target_type:
            diffs.append(
                {
                    "col_name": c,
                    "source_type": str(source_type),
                    "target_type": str(target_type),
                }
            )

    return pl.DataFrame(
        diffs,
        schema={
            "col_name": pl.String,
            "source_type": pl.String,
            "target_type": pl.String,
        },
    )


def row_presence_differences(
    source: pl.LazyFrame,
    target: pl.LazyFrame,
    id_cols: list[str],
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Identify rows that exist only in the source dataset or only in the target dataset.

    The key columns of both datasets are extracted and anti-joined against each other
    to isolate the unique records.

    Args:
        source: The source data.
        target: The target data.
        id_cols: The column names used to join and compare the rows.

    Returns:
        tuple[pl.LazyFrame, pl.LazyFrame]: Two LazyFrames containing only the key columns
            for the source-only rows and target-only rows, respectively.
    """
    source_keys = source.select(*id_cols)
    target_keys = target.select(*id_cols)
    source_only = source_keys.join(target_keys, on=id_cols, how="anti")
    target_only = target_keys.join(source_keys, on=id_cols, how="anti")
    return source_only, target_only
