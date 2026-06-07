"""
Normalisation transforms applied *before* diffing to suppress irrelevant
format differences between disparate sources.

Each normaliser has signature:  (lf, keys, cols) -> lf
"""

from __future__ import annotations

from collections.abc import Callable

import polars as pl

NormaliserFn = Callable[[pl.LazyFrame, list[str], list[str]], pl.LazyFrame]

NULLISH = ["", "nan", "none", "<na>", "null", "na", "n/a"]


def _norm_expr(expr: pl.Expr) -> pl.Expr:
    s = expr.cast(pl.String).str.strip_chars()
    s_lower = s.str.to_lowercase()
    s = (
        pl.when(s.is_null() | s_lower.is_in(NULLISH))
        .then(pl.lit(None, dtype=pl.String))
        .otherwise(s)
    )
    return s.str.replace(r"^(-?\d+)\.0+$", "${1}")


def normalise_float_strings(
    lf: pl.LazyFrame, keys: list[str], cols: list[str]
) -> pl.LazyFrame:
    """
    Standardise data formatting to prevent superficial text differences from being
    flagged as differences.

    All specified columns are cast to strings. Leading and trailing whitespace is
    stripped, trailing `.0` characters are removed from integers parsed as floats, and
    various string representations of null (like 'NaN', 'N/A', or empty strings) are
    mapped to true nulls.

    Args:
        lf: The input data.
        keys: The list of key columns to normalise.
        cols: The list of data columns to normalise.

    Returns:
        pl.LazyFrame: The dataframe with the normalisation transformations applied to the
            specified columns.
    """
    return lf.select(
        *[_norm_expr(pl.col(k)).alias(k) for k in keys],
        *[_norm_expr(pl.col(c)).alias(c) for c in cols],
    )


REGISTRY: dict[str, NormaliserFn] = {
    "float_strings": normalise_float_strings,
}
