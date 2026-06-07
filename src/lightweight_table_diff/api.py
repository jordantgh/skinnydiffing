"""Public diff API."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, cast

import polars as pl

from .core import batch_diff_tbls, diff_tbls
from .dimensions import (
    column_presence_differences,
    get_cols_to_compare,
    row_presence_differences,
    type_differences,
    validate_key_uniqueness,
)
from .normalisers import REGISTRY as NORMALISER_REGISTRY
from .normalisers import NormaliserFn
from .resolver import TableLike, into_lazyframe
from .result import DiffResult

logger = logging.getLogger(__name__)


def diff(
    source: TableLike | Callable[[], TableLike],
    target: TableLike | Callable[[], TableLike],
    keys: str | Sequence[str],
    *,
    compare: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    normalise: str | NormaliserFn | Sequence[str | NormaliserFn] | None = None,
    batch_size: int | None = 50,
    join_type: Literal["inner", "left", "full", "outer"] = "inner",
    check_keys: bool = True,
    name: str = "diff",
    source_options: Mapping[str, Any] | None = None,
    target_options: Mapping[str, Any] | None = None,
    loaders: Mapping[str, Callable[[Any], TableLike | Callable[[], TableLike]]]
    | None = None,
    collect_lazy_inputs: bool = False,
) -> DiffResult:
    """
    Compare two datasets and return a detailed record of every cell, row, and column
    that differs.

    The source and target datasets are loaded, standardised using any provided
    normalisation rules, and checked to ensure the join keys do not contain duplicates.
    The output separates structural differences (shape and data type differences) from
    actual data differences, which are reported at the individual cell level.

    Args:
        source: The source dataset. Can be a file path, a Polars or Pandas dataframe,
            a zero-argument function that returns data, or a single-item dictionary
            triggering a custom loader (e.g. `{"spark": "my_table"}`).
        target: The target dataset to compare against the source. Accepts the same formats
            as the `source` argument.
        keys: The column name(s) used to uniquely identify a row across both datasets.
        compare: A specific list of columns to compare. If provided, any columns not in
            this list are completely ignored. Defaults to None (compare all shared
            columns).
        exclude: A list of columns to ignore during the comparison. Defaults to None.
        normalise: The name of a registered normalisation function, or a custom function,
            to standardise the data before comparing. Use this to ignore formatting
            differences like trailing zeroes or casing.
        batch_size: The number of columns to process in memory at the same time during
            the cell-level comparison. Lowering this prevents out-of-memory errors on
            very wide tables. Defaults to 50.
        join_type: The type of join used to align the datasets. 'inner' only compares rows
            present in both datasets. 'full' compares shared rows and also identifies rows
            unique to either side. Defaults to 'inner'.
        check_keys: Whether to verify that the key columns strictly uniquely identify
            rows before diffing. If False, duplicate keys will cause a row explosion
            during the join. Defaults to True.
        name: A label for this comparison, used as a prefix when saving output files.
        source_options: Extra keyword arguments passed directly to the reader for the
            `source` data (e.g., `infer_schema_length` for CSVs).
        target_options: Extra keyword arguments passed directly to the reader for the
            `target` data.
        loaders: A dictionary mapping a string name to a custom loading function.
            Required if passing dictionaries like `{"spark": "my_table"}` as inputs.
        collect_lazy_inputs: If True, forces dataframes that are not natively supported by
            the Polars lazy engine to be fully loaded into memory before diffing.

    Returns:
        DiffResult: An object containing individual dataframes for the differing cells,
            source-only rows, target-only rows, source-only columns, target-only columns,
            and data type differences.
    """

    source_lf = into_lazyframe(
        source,
        loaders=loaders,
        collect_lazy=collect_lazy_inputs,
        **dict(source_options or {}),
    )
    target_lf = into_lazyframe(
        target,
        loaders=loaders,
        collect_lazy=collect_lazy_inputs,
        **dict(target_options or {}),
    )

    return diff_lazyframes(
        source_lf,
        target_lf,
        keys=keys,
        compare=compare,
        exclude=exclude,
        normalise=normalise,
        batch_size=batch_size,
        join_type=join_type,
        check_keys=check_keys,
        name=name,
    )


def diff_lazyframes(
    source: pl.LazyFrame,
    target: pl.LazyFrame,
    *,
    keys: str | Sequence[str] | None = None,
    compare: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    normalise: str | NormaliserFn | Sequence[str | NormaliserFn] | None = None,
    batch_size: int | None = 50,
    join_type: Literal["inner", "left", "full", "outer"] = "inner",
    check_keys: bool = True,
    name: str = "diff",
) -> DiffResult:
    """
    Calculate cell, row, and column differences between two pre-resolved Polars
    LazyFrames.

    Structural differences (shape and data type differences) are computed first.
    The intersecting columns are then aligned, passed through any requested normalisation
    functions, and validated for unique keys. Finally, source-only and target-only rows
    are identified before the datasets are joined to generate the cell-by-cell difference
    table.

    Args:
        source: The source data, formatted as a Polars LazyFrame.
        target: The target data, formatted as a Polars LazyFrame.
        keys: The column name(s) used to uniquely identify a row across both datasets.
        compare: A specific list of columns to compare. Defaults to None (compare all
            shared columns).
        exclude: A list of columns to ignore during the comparison. Defaults to None.
        normalise: The name of a registered normalisation function, or a custom function,
            to standardise the data before comparing.
        batch_size: The number of columns to process in memory at the same time during
            the cell-level comparison. Defaults to 50.
        join_type: The type of join used to align the datasets. Defaults to 'inner'.
        check_keys: Whether to verify that the key columns strictly uniquely identify rows
            before diffing. Defaults to True.
        name: A label for this comparison. Defaults to "diff".

    Returns:
        DiffResult: An object containing individual dataframes for the differing cells,
            source-only rows, target-only rows, source-only columns, target-only columns,
            and data type differences.
    """

    if keys is None:
        logger.info("No keys provided. Falling back to strict row-index positional diffing.")
        row_key = "__td_idx"
        source = source.with_row_index(row_key)
        target = target.with_row_index(row_key)
        key_list = [row_key]
    else:
        key_list = [keys] if isinstance(keys, str) else list(keys)

    if not key_list:
        raise ValueError("keys must contain at least one column, or be None to use row indices")

    join_type = _validate_join_type(join_type)

    source_only_cols, target_only_cols = column_presence_differences(
        source, target, key_list
    )
    cols = get_cols_to_compare(
        source,
        target,
        key_list,
        include_cols=list(compare) if compare is not None else None,
        exclude_cols=list(exclude) if exclude is not None else None,
    )

    type_diffs = type_differences(source, target, cols)

    source, target = _apply_normalisation(
        source,
        target,
        keys=key_list,
        cols=cols,
        normalisation=normalise,
    )

    if check_keys:
        validate_key_uniqueness(source, target, key_list)

    source_only_rows, target_only_rows = row_presence_differences(
        source, target, key_list
    )

    if batch_size is None or batch_size <= 0 or batch_size >= len(cols):
        diff_table = diff_tbls(source, target, key_list, cols, join_type=join_type)
    else:
        diff_table = batch_diff_tbls(
            source,
            target,
            key_list,
            cols,
            batch_size=batch_size,
            join_type=join_type,
        )

    return DiffResult(
        name=name,
        diff=diff_table,
        source_only_rows=source_only_rows,
        target_only_rows=target_only_rows,
        source_only_cols=source_only_cols,
        target_only_cols=target_only_cols,
        type_differences=type_diffs,
    )


def _validate_join_type(
    join_type: Literal["inner", "left", "full", "outer"],
) -> Literal["inner", "left", "full"]:
    if join_type == "outer":
        return "full"
    if join_type not in {"inner", "left", "full"}:
        raise ValueError("join_type must be one of: 'inner', 'left', 'full', 'outer'")
    return join_type


def _apply_normalisation(
    source: pl.LazyFrame,
    target: pl.LazyFrame,
    *,
    keys: list[str],
    cols: list[str],
    normalisation: str | NormaliserFn | Sequence[str | NormaliserFn] | None,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    if not normalisation:
        return source, target

    items = (
        [normalisation]
        if isinstance(normalisation, str) or callable(normalisation)
        else list(normalisation)
    )

    for item in items:
        if callable(item):
            norm_fn = cast(NormaliserFn, item)
        elif item not in NORMALISER_REGISTRY:
            raise ValueError(
                f"Unknown normaliser: {item!r}. Registered: {sorted(NORMALISER_REGISTRY)}"
            )
        else:
            norm_fn = NORMALISER_REGISTRY[item]

        source = norm_fn(source, keys, cols)
        target = norm_fn(target, keys, cols)

    return source, target
