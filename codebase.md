# Codebase: codebase

Scanned: `.`

## Structure

```
lightweight_table_diff/
├── examples/
│   └── cloudera_hive/
│       └── run_diff.py
├── src/
│   └── lightweight_table_diff/
│       ├── __init__.py
│       ├── __main__.py
│       ├── api.py
│       ├── config.py
│       ├── core.py
│       ├── dimensions.py
│       ├── normalisers.py
│       ├── resolver.py
│       ├── result.py
│       └── runner.py
├── tests/
│   ├── test_api.py
│   ├── test_core.py
│   ├── test_dimensions.py
│   ├── test_integration.py
│   └── test_normalisers.py
└── pyproject.toml
```

---

## `examples\cloudera_hive\run_diff.py`

```python
"""
Example runner for lightweight-table-diff using a custom Hive/S3 downloader.
"""

import logging
from pathlib import Path
from urllib.parse import urlparse

import boto3
import polars as pl
from pyspark.sql import SparkSession

from lightweight_table_diff import run_config

logger = logging.getLogger(__name__)


def load_hive(
    table: str,
    cache_dir: str | None = None,
    *,
    spark: SparkSession,
    ssl_cert: str | None = None,
) -> pl.LazyFrame:
    """
    Uses Spark to find a Hive table's S3 path, downloads the raw Parquet files
    locally, and returns a native Polars LazyFrame.
    """
    if cache_dir is None:
        cache_dir = f"/tmp/hive_{table}"

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    rows = spark.sql(f"DESCRIBE FORMATTED {table}").collect()
    location = next(
        (r[1].strip() for r in rows if r[0] and "Location" in r[0]), None
    )
    if not location:
        raise RuntimeError(f"Could not resolve S3 location for '{table}'")

    parsed = urlparse(str(location).replace("s3a://", "s3://"))
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/").rstrip("/") + "/"

    client = boto3.client("s3")
    try:
        import raz_client

        if ssl_cert:
            raz_client.configure_ranger_raz(client, ssl_file=ssl_cert)
    except ImportError:
        pass

    logger.info("Downloading %s -> %s", location, cache_path)
    n_files = 0
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=prefix
    ):
        for obj in page.get("Contents", []):
            if not obj["Key"].endswith(".parquet"):
                continue
            n_files += 1
            relative = (
                obj["Key"][len(prefix) :].lstrip("/")
                if obj["Key"].startswith(prefix)
                else Path(obj["Key"]).name
            )
            dest = cache_path / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, obj["Key"], str(dest))

    if not n_files:
        raise FileNotFoundError(f"No parquet files found at {location}")

    return pl.scan_parquet(
        str(cache_path / "**/*.parquet"), hive_partitioning=True
    )


def main():
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        level=logging.INFO,
    )

    logging.info("Initialising SparkSession...")
    spark = (
        SparkSession.builder.appName("TableDiff_Hive_S3")
        .enableHiveSupport()
        .getOrCreate()
    )

    current_dir = Path(__file__).parent
    config_path = current_dir / "config.yml"

    logging.info("Loading config from %s", config_path)

    loaders = {
        "hive": lambda payload: load_hive(
            **payload, spark=spark, ssl_cert="/etc/pki/tls/certs/ca-bundle.crt"
        )
    }

    results = run_config(
        config_path,
        loaders=loaders,
    )

    logging.info("Completed %d comparison(s).", len(results))
    spark.stop()


if __name__ == "__main__":
    main()
```

## `pyproject.toml`

```toml
[project]
name = "lightweight-table-diff"
version = "0.1.0"
description = "Cell-level table diffing with a Polars engine and Narwhals dataframe input support."
requires-python = ">=3.11.1"
dependencies = [
    "narwhals>=1.0",
    "polars>=1.0",
    "polars-checkpoint",
    "pyyaml",
]

[dependency-groups]
dev = [
    "pandas>=3.0.3",
    "pyarrow>=24.0.0",
    "pyrefly>=1.0.0",
    "pytest>=9.0.3",
    "pytest-cov>=7.1.0",
    "ty>=0.0.44",
]

[project.optional-dependencies]
readstat = ["polars-readstat"]
examples = ["pyspark", "raz_client", "types-boto3[s3]"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"



[tool.ruff]
target-version = "py311"
line-length = 90
src = ["src"]

exclude = [
    ".venv",
    ".pytest_cache",
    "tests",
    "**/__init__.py",
]

[tool.ruff.lint]
select = ["E", "W", "F", "B", "UP", "I", "SIM", "C90", "S", "PTH", "RUF"]


[tool.ruff.format]
docstring-code-format = true

[tool.pyrefly]

project-includes = ["src", "examples"]
project-excludes = [
    "**/tests",
    "**/.venv",
]

[tool.ty.src]
include = ["src", "examples"]
exclude = [
    "**/tests/",
    "**/.venv/",
]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.coverage.run]
source = ["src"]
```

## `src\lightweight_table_diff\__init__.py`

```python
"""lightweight_table_diff — cell-level table diffing."""

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
```

## `src\lightweight_table_diff\__main__.py`

```python
"""python -m lightweight_table_diff config.yml"""
import logging
import sys

from .runner import run_config

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    level=logging.INFO,
)

if len(sys.argv) < 2:
    print("Usage: python -m lightweight_table_diff <config.yml>", file=sys.stderr)
    sys.exit(1)

run_config(sys.argv[1])
```

## `src\lightweight_table_diff\api.py`

```python
"""Public diff API."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

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

    The source and target datasets are loaded, standardised using any provided normalisation
    rules, and checked to ensure the join keys do not contain duplicates. The output
    separates structural differences (shape and data type differences) from actual data
    differences, which are reported at the individual cell level.

    Args:
        source: The source dataset. Can be a file path, a Polars or Pandas dataframe,
            a zero-argument function that returns data, or a single-item dictionary
            triggering a custom loader (e.g. `{"spark": "my_table"}`).
        target: The target dataset to compare against the source. Accepts the same formats
            as the `source` argument.
        keys: The column name(s) used to uniquely identify a row across both datasets.
        compare: A specific list of columns to compare. If provided, any columns not in
            this list are completely ignored. Defaults to None (compare all shared columns).
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
    keys: str | Sequence[str],
    compare: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    normalise: str | NormaliserFn | Sequence[str | NormaliserFn] | None = None,
    batch_size: int | None = 50,
    join_type: Literal["inner", "left", "full", "outer"] = "inner",
    check_keys: bool = True,
    name: str = "diff",
) -> DiffResult:
    """
    Calculate cell, row, and column differences between two pre-resolved Polars LazyFrames.

    Structural differences (shape and data type differences) are computed first.
    The intersecting columns are then aligned, passed through any requested normalisation
    functions, and validated for unique keys. Finally, source-only and target-only rows are
    identified before the datasets are joined to generate the cell-by-cell difference table.

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

    key_list = [keys] if isinstance(keys, str) else list(keys)
    if not key_list:
        raise ValueError("keys must contain at least one column")

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
        diff_table = diff_tbls(
            source, target, key_list, cols, join_type=join_type
        )
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
        raise ValueError(
            "join_type must be one of: 'inner', 'left', 'full', 'outer'"
        )
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
            norm_fn = item
        elif item not in NORMALISER_REGISTRY:
            raise ValueError(
                f"Unknown normaliser: {item!r}. Registered: {sorted(NORMALISER_REGISTRY)}"
            )
        else:
            norm_fn = NORMALISER_REGISTRY[item]

        source = norm_fn(source, keys, cols)
        target = norm_fn(target, keys, cols)

    return source, target
```

## `src\lightweight_table_diff\config.py`

```python
"""YAML config loading and deep-merge expansion."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """
    Combine two dictionaries recursively, prioritising values from the override dictionary.

    Nested dictionaries are traversed and merged at every level. If a key exists in both
    dictionaries but the values are not dictionaries, the value from the override dictionary
    completely replaces the value from the base dictionary. Lists are replaced entirely,
    not concatenated. The original dictionaries are not modified.

    Args:
        base: The foundation dictionary containing default values.
        override: The dictionary containing specific values that replace the defaults.

    Returns:
        dict: A new dictionary containing the merged result.
    """
    result = copy.deepcopy(base)
    _merge_in_place(result, override)
    return result


def _is_loader_call(x: Any) -> bool:
    return isinstance(x, dict) and len(x) == 1


def _merge_in_place(base: dict, override: dict) -> None:
    for k, v in override.items():
        # Prevent merging different loaders (e.g. merging 'spark' with 'extract')
        if (
            k in {"source", "target"}
            and k in base
            and _is_loader_call(base[k])
            and _is_loader_call(v)
            and next(iter(base[k])) != next(iter(v))
        ):
            base[k] = copy.deepcopy(v)

        elif k in base and isinstance(v, dict) and isinstance(base[k], dict):
            _merge_in_place(base[k], v)
        else:
            base[k] = copy.deepcopy(v)


def expand_comparisons(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Expand configuration with global defaults and distinct jobs into standalone job configurations.

    Top-level keys and keys nested under a `defaults` block are deep-merged into every
    dictionary listed in the `comparisons` block. Overrides in the individual comparison
    blocks take precedence over the defaults.

    For example, an input like this::

        defaults:
            keys: ["id"]
            batch_size: 100
        comparisons:
            - source: "source_data.csv"
              target: "target_data.csv"
            - source: "source_other.csv"
              target: "target_other.csv"
              batch_size: 50

    Produces the following output::

        [
            {
                "keys": ["id"],
                "batch_size": 100,
                "source": "source_data.csv",
                "target": "target_data.csv",
            },
            {
                "keys": ["id"],
                "batch_size": 50,
                "source": "source_other.csv",
                "target": "target_other.csv",
            },
        ]

    Args:
        raw (dict): The raw dictionary parsed from the YAML configuration file.

    Returns:
        list[dict]: A list of fully expanded configuration dictionaries, one for each
            item in the `comparisons` list.

    Raises:
        TypeError: If the 'defaults' key is not a dictionary, or the 'comparisons' key
            is not a list.
    """
    top_level_defaults = {
        k: v for k, v in raw.items() if k not in {"defaults", "comparisons"}
    }
    named_defaults = raw.get("defaults", {})
    if named_defaults is None:
        named_defaults = {}
    if not isinstance(named_defaults, dict):
        raise TypeError(
            f"'defaults' must be a mapping, got {type(named_defaults).__name__}"
        )

    base = deep_merge(named_defaults, top_level_defaults)
    items = raw.get("comparisons", [{}])
    if not isinstance(items, list):
        raise TypeError(
            f"'comparisons' must be a list, got {type(items).__name__}"
        )
    return [deep_merge(base, item) for item in items]


def load_config(path: str | Path) -> list[dict[str, Any]]:
    """
    Read a YAML configuration file from disk and expand it into a list of standalone
    job dictionaries.

    The file is parsed into a Python dictionary and immediately passed through the
    expansion logic to merge any global or block-level defaults into the individual
    comparison jobs.

    Args:
        path: The file path to the YAML configuration file.

    Returns:
        list[dict]: A list of fully expanded configuration dictionaries, one for each
            comparison job defined in the file.

    Raises:
        ValueError: If the file is completely empty.
        TypeError: If the parsed YAML does not evaluate to a top-level dictionary.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ValueError(f"Config file {path!s} is empty")
    if not isinstance(raw, dict):
        raise TypeError(
            f"Config root must be a mapping, got {type(raw).__name__}"
        )
    return expand_comparisons(raw)
```

## `src\lightweight_table_diff\core.py`

```python
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
    lambda testarg: testarg ** 2
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
```

## `src\lightweight_table_diff\dimensions.py`

```python
"""Column/row checking and key-uniqueness validation."""

from __future__ import annotations

import polars as pl


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
        raise ValueError(
            f"Join key(s) missing from one or both sides: {missing}"
        )

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
        if dupes.height:
            problems.append(
                f"  {label}: {dupes.height} duplicate key group(s)\n{dupes}"
            )

    if problems:
        raise ValueError(
            "Duplicate keys (would cause row explosion):\n"
            + "\n".join(problems)
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
    source: pl.LazyFrame, 
    target: pl.LazyFrame, 
    cols: list[str]
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
            diffs.append({
                "col_name": c, 
                "source_type": str(source_type), 
                "target_type": str(target_type)
            })
            
    return pl.DataFrame(
        diffs, 
        schema={"col_name": pl.String, "source_type": pl.String, "target_type": pl.String}
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
```

## `src\lightweight_table_diff\normalisers.py`

```python
"""
Normalisation transforms applied *before* diffing to suppress irrelevant
format differences between disparate sources.

Each normaliser has signature:  (lf, keys, cols) -> lf
"""

from __future__ import annotations

from typing import Callable

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

    All specified columns are cast to strings. Leading and trailing whitespace is stripped, 
    trailing `.0` characters are removed from integers parsed as floats, and various string 
    representations of null (like 'NaN', 'N/A', or empty strings) are mapped to true nulls.

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
```

## `src\lightweight_table_diff\resolver.py`

```python
"""Resolve public table inputs into Polars LazyFrames."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import narwhals as nw
import polars as pl
from narwhals.typing import IntoFrame

TableLike = str | Path | PathLike | IntoFrame | Mapping[str, Any]

SUFFIX_TO_FORMAT = {
    ".parquet": "parquet",
    ".pq": "parquet",
    ".csv": "csv",
    # ReadStat formats (SAS, Stata, SPSS)
    ".sav": "readstat",
    ".zsav": "readstat",
    ".por": "readstat",
    ".sas7bdat": "readstat",
    ".xpt": "readstat",
    ".xpt5": "readstat",
    ".xpt8": "readstat",
    ".dta": "readstat",
}

FORMAT_ALIASES = {
    "parquet": "parquet",
    "pq": "parquet",
    "csv": "csv",
    "sav": "readstat",
    "zsav": "readstat",
    "por": "readstat",
    "spss": "readstat",
    "sas": "readstat",
    "sas7bdat": "readstat",
    "xpt": "readstat",
    "xpt5": "readstat",
    "xpt8": "readstat",
    "stata": "readstat",
    "dta": "readstat",
    "readstat": "readstat",
}


def into_lazyframe(
    obj: TableLike | Callable[[], TableLike],
    *,
    loaders: Mapping[str, Callable[[Any], TableLike | Callable[[], TableLike]]] | None = None,
    collect_lazy: bool = False,
    file_format: str | None = None,
    glob: str | None = None,
    **scan_options: Any,
) -> pl.LazyFrame:
    """
    Convert a file, dataframe object (or callable returning either of those) into a Polars LazyFrame.

    It supports direct file paths (e.g., CSV, Parquet, SPSS), Pandas and Narwhals-compatible 
    dataframes, zero-argument functions (can also be partial functions), or dictionary-based
    loader configurations  (e.g., `{'spark': 'db.table'}`). If a loader dictionary is 
    provided, it extracts the  data using the matching function provided in the `loaders`
    argument.

    Args:
        obj: The input data. Can be a path, a dataframe, a function, or a loader dictionary.
            The loader dictionary has the form {loader_name: payload}, where loader_name is 
            used in the `loaders` mapping to find the corresponding loading function, and 
            payload is passed to that function.
        loaders: A dictionary mapping a string loader name to a custom loading function.
        collect_lazy: Whether to force non-Polars lazy dataframes to fully load into 
            memory before converting them.
        file_format: An override to force reading a file path as a specific format 
            (e.g., 'csv' or 'readstat') regardless of its file extension.
        glob: A glob pattern to use if `obj` points to a directory (e.g. `*.parquet`).
        **scan_options: Additional arguments passed directly to the underlying Polars 
            scanning function.

    Returns:
        pl.LazyFrame: The input data converted into a Polars LazyFrame.

    Raises:
        TypeError: If the input type is unrecognised or cannot be converted.
        ValueError: If a custom loader is requested but not found in the `loaders` dictionary.
    """
    if callable(obj):
        return into_lazyframe(
            obj(),
            loaders=loaders,
            collect_lazy=collect_lazy,
            file_format=file_format,
            glob=glob,
            **scan_options,
        )

    if isinstance(obj, Mapping):
        return _load_from_call(
            obj,
            loaders=loaders,
            collect_lazy=collect_lazy,
            file_format=file_format,
            glob=glob,
            **scan_options,
        )

    if isinstance(obj, pl.LazyFrame):
        return obj

    if isinstance(obj, pl.DataFrame):
        return obj.lazy()

    if isinstance(obj, (str, Path, PathLike)):
        return scan_path(
            Path(obj),
            file_format=file_format,
            glob=glob,
            **scan_options,
        )

    return _narwhals_to_polars_lazy(obj, collect_lazy=collect_lazy)


def scan_path(
    path: Path,
    *,
    file_format: str | None = None,
    glob: str | None = None,
    **options: Any,
) -> pl.LazyFrame:
    """
    Read a local file or directory path into a Polars LazyFrame.

    The file format is inferred from the extension unless explicitly overridden. Parquet 
    and CSV files are scanned directly using native Polars readers. SAS, Stata, and SPSS 
    files are parsed via the `polars_readstat` plugin.

    Args:
        path: The file or directory path to read.
        file_format: An override to force parsing the path as a specific format 
            (e.g., 'csv' or 'readstat') regardless of its file extension.
        glob: A glob pattern to use if `path` points to a directory (e.g., `*.parquet`).
        **options: Additional arguments passed directly to the underlying scanning function 
            (e.g., `ignore_errors=True` for CSVs).

    Returns:
        pl.LazyFrame: The scanned data.

    Raises:
        ValueError: If the file format is completely unsupported or if a directory/glob is 
            passed to a format that only supports single files (like SPSS).
        ImportError: If reading a ReadStat format but the `polars_readstat` library is 
            not installed.
    """
    fmt = _normalise_format(file_format) if file_format else _infer_format(path, glob)

    if fmt == "parquet":
        scan_target = _path_with_glob(path, glob or "*.parquet")
        return pl.scan_parquet(
            scan_target,
            hive_partitioning=options.get("hive_partitioning", False),
        )

    if fmt == "csv":
        scan_target = _path_with_glob(path, glob or "*.csv")
        return pl.scan_csv(
            scan_target,
            infer_schema_length=options.get("infer_schema_length", 10_000),
            ignore_errors=options.get("ignore_errors", True),
        )

    if fmt == "readstat":
        if path.is_dir():
            raise ValueError(
                "SAS/Stata/SPSS inputs must be a specific file path, not a directory."
            )
        if glob is not None or any(ch in str(path) for ch in "*?[]"):
            raise ValueError(
                "SAS/Stata/SPSS inputs must be a specific file path, not a glob."
            )

        try:
            from polars_readstat import scan_readstat
        except ImportError:
            raise ImportError(
                "polars-readstat is required for SAS, Stata, and SPSS files: "
                "pip install 'lightweight-table-diff[readstat]'"
            ) from None

        return scan_readstat(str(path), **options)

    raise ValueError(f"Unsupported table format {fmt!r}")


def _load_from_call(
    call: Mapping[str, Any],
    *,
    loaders: Mapping[str, Callable[[Any], TableLike | Callable[[], TableLike]]] | None = None,
    collect_lazy: bool,
    file_format: str | None,
    glob: str | None,
    **scan_options: Any,
) -> pl.LazyFrame:
    """Resolve a one-item loader call, e.g. ``{"spark": "db.table"}``."""
    if len(call) != 1:
        raise TypeError(
            "Mapping inputs are reserved for one-item loader calls such as "
            "{'spark': 'db.table'} or {'spark_sql': {'query': '...'}}. "
            "For plain in-memory data, pass a dataframe object instead."
        )

    name, payload = next(iter(call.items()))
    if not isinstance(name, str):
        raise TypeError("Loader call keys must be strings")

    if loaders is None or name not in loaders:
        available = sorted(loaders or {})
        raise ValueError(f"Unknown loader {name!r}. Available loaders: {available}")

    loaded = loaders[name](payload)
    return into_lazyframe(
        loaded,
        loaders=loaders,
        collect_lazy=collect_lazy,
        file_format=file_format,
        glob=glob,
        **scan_options,
    )


def _narwhals_to_polars_lazy(obj: Any, *, collect_lazy: bool) -> pl.LazyFrame:
    try:
        wrapped = nw.from_native(obj, allow_series=False, pass_through=False)
    except Exception as e:
        raise TypeError(
            f"Cannot convert {type(obj).__name__!r} into a Polars LazyFrame. "
            "Pass a path, Polars frame, Narwhals-supported dataframe, "
            "zero-argument callable, or one-item loader call such as "
            "{'spark': 'db.table'}."
        ) from e

    if hasattr(wrapped, "to_polars"):
        return _polars_object_to_lazyframe(wrapped.to_polars())

    if hasattr(wrapped, "collect"):
        if not collect_lazy:
            raise TypeError(
                f"{type(obj).__name__!r} was recognised as a non-Polars lazy "
                "dataframe. Converting it to the Polars diff engine requires "
                "collecting it first. Pass collect_lazy_inputs=True if that is "
                "what you want."
            )
        eager = wrapped.collect()
        if not hasattr(eager, "to_polars"):
            raise TypeError(
                f"Narwhals collected {type(obj).__name__!r}, but the result "
                "could not be converted to Polars."
            )
        return _polars_object_to_lazyframe(eager.to_polars())

    raise TypeError(
        f"Narwhals wrapped {type(obj).__name__!r}, but it did not expose a "
        "DataFrame or LazyFrame interface."
    )


def _polars_object_to_lazyframe(obj: Any) -> pl.LazyFrame:
    if isinstance(obj, pl.LazyFrame):
        return obj
    if isinstance(obj, pl.DataFrame):
        return obj.lazy()
    raise TypeError(
        "Expected Narwhals conversion to produce a Polars DataFrame, got "
        f"{type(obj).__name__!r}."
    )


def _path_with_glob(path: Path, glob: str) -> str:
    return str(path / glob) if path.is_dir() else str(path)


def _infer_format(path: Path, glob: str | None) -> str:
    if path.is_dir():
        if glob:
            inferred = _format_from_suffix(Path(glob).suffix)
            if inferred is not None:
                return inferred
        return "parquet"

    inferred = _format_from_suffix(path.suffix)
    if inferred is None:
        raise ValueError(
            f"Cannot infer table format from path {path!s}. "
            "Pass file_format='parquet', 'csv', or 'readstat'."
        )
    return inferred


def _format_from_suffix(suffix: str) -> str | None:
    return SUFFIX_TO_FORMAT.get(suffix.lower())


def _normalise_format(file_format: str) -> str:
    key = file_format.lower().lstrip(".")
    try:
        return FORMAT_ALIASES[key]
    except KeyError as e:
        raise ValueError(
            f"Unsupported table format {file_format!r}. "
            f"Supported: {sorted(FORMAT_ALIASES)}"
        ) from e
```

## `src\lightweight_table_diff\result.py`

```python
"""Result object returned by the public diff API."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class DiffResult:
    """
    Container for all calculated differences between two datasets.

    Holds individual Polars dataframes containing the differing cells, removed rows, 
    added rows, and data type differences, along with lists of the added and removed 
    columns. Provides helper methods to evaluate the total number of differences,
    generate summaries, and write the outputs to disk.
    """

    diff: pl.LazyFrame
    source_only_rows: pl.LazyFrame
    target_only_rows: pl.LazyFrame
    source_only_cols: list[str]
    target_only_cols: list[str]
    type_differences: pl.DataFrame
    name: str = "diff"
    _n_diffs: int | None = field(default=None, init=False, repr=False)

    @property
    def n_diffs(self) -> int:
        """
        Calculate the total number of individual cells that contain different values.
        """
        if self._n_diffs is None:
            self._n_diffs = self.diff.select(pl.len()).collect().item()
        return self._n_diffs

    def summary(self) -> pl.LazyFrame:
        """
        Group identical cell-level differences together and count their frequency.

        Groups the cell difference table by the column name, the source value, and 
        the target value. It calculates the total occurrences of each unique diff and 
        sorts the output in descending order of frequency.

        Returns:
            pl.LazyFrame: A dataframe containing `col_name`, `source_val`, `target_val`, 
                and `count`.
        """
        return (
            self.diff.group_by("col_name", "source_val", "target_val")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )

    def is_empty(self) -> bool:
        """
        Determine if the two datasets are completely identical.

        Checks for the presence of any data type differences, added or removed columns, 
        cell-level value differences, or added/removed rows. Triggers a count execution 
        on the lazy dataframes to verify row counts if the structural checks pass.

        Returns:
            bool: True if absolutely no structural or data differences exist, False otherwise.
        """
        if not self.type_differences.is_empty():
            return False
        if self.source_only_cols or self.target_only_cols:
            return False
        if self.n_diffs > 0:
            return False
        if self.source_only_rows.select(pl.len()).collect().item() > 0:
            return False
        if self.target_only_rows.select(pl.len()).collect().item() > 0:
            return False
        return True

    def write(self, output_dir: str | Path, *, basename: str | None = None) -> dict[str, Path]:
        """
        Save all non-empty difference tables to CSV files in a specified directory.

        Structural differences (shape and data type differences) and the detailed cell-level
        differences are written to disk. A summary table grouping identical cell diffs by
        frequency is also generated. Files are only created if differences actually exist in
        that category.

        Args:
            output_dir: The directory where the CSV files will be saved. Created if it 
                does not exist.
            basename: A prefix for the filenames. Defaults to the `name` attribute of 
                the DiffResult, or "diff".

        Returns:
            dict[str, Path]: A dictionary mapping the logical name of the output 
                (e.g., 'cells', 'source_only_rows') to the absolute path of the written file.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = basename or self.name or "diff"
        written: dict[str, Path] = {}

        if not self.type_differences.is_empty():
            path = out / f"{stem}_type_differences.csv"
            self.type_differences.write_csv(path)
            written["type_differences"] = path
            logger.info("  Wrote %s", path)

        if self.source_only_cols or self.target_only_cols:
            path = out / f"{stem}_column_presence.csv"
            max_len = max(len(self.source_only_cols), len(self.target_only_cols))
            pl.DataFrame(
                {
                    "source_only_columns": self.source_only_cols
                    + [""] * (max_len - len(self.source_only_cols)),
                    "target_only_columns": self.target_only_cols
                    + [""] * (max_len - len(self.target_only_cols)),
                }
            ).write_csv(path)
            written["column_presence"] = path
            logger.info("  Wrote %s", path)

        for label, lf in [
            ("source_only_rows", self.source_only_rows),
            ("target_only_rows", self.target_only_rows),
        ]:
            n = lf.select(pl.len()).collect().item()
            if n > 0:
                path = out / f"{stem}_{label}.csv"
                lf.sink_csv(path)
                written[label] = path
                logger.info("  %d %s -> %s", n, label.replace("_", " "), path)
            else:
                logger.info("  No %s", label.replace("_", " "))

        if self.n_diffs == 0:
            logger.info("  %s: no cell differences", stem)
            return written

        detail_path = out / f"{stem}_detailed.csv"
        self.diff.sink_csv(detail_path)
        written["cells"] = detail_path
        logger.info("  Wrote %s", detail_path)

        summary_path = out / f"{stem}_summary.csv"
        self.summary().sink_csv(summary_path)
        written["summary"] = summary_path
        logger.info("  Wrote %s", summary_path)

        return written
```

## `src\lightweight_table_diff\runner.py`

```python
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
    function. High-level summaries of the source-only columns, target-only columns, and total 
    cell differences are logged to the console.

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
        join_type=job.get("join_type", "inner"),
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
```

## `tests\test_api.py`

```python
from functools import partial

import polars as pl
import pytest

from lightweight_table_diff import diff


def test_diff_accepts_polars_dataframes():
    source = pl.DataFrame({"id": [1, 2], "x": ["a", "b"]})
    target = pl.DataFrame({"id": [1, 2], "x": ["a", "c"]})

    result = diff(source, target, keys="id")

    assert result.n_diffs == 1
    assert result.diff.collect().to_dict(as_series=False) == {
        "id": [2],
        "col_name": ["x"],
        "source_val": ["b"],
        "target_val": ["c"],
    }


def test_diff_reports_source_only_and_target_only_rows_and_columns():
    source = pl.DataFrame({"id": [1, 2], "x": ["a", "b"], "source_only_col": [1, 2]})
    target = pl.DataFrame({"id": [2, 3], "x": ["b", "c"], "target_only_col": [3, 4]})

    result = diff(source, target, keys="id")

    assert result.source_only_cols == ["source_only_col"]
    assert result.target_only_cols == ["target_only_col"]
    assert result.source_only_rows.collect().to_dict(as_series=False) == {"id": [1]}
    assert result.target_only_rows.collect().to_dict(as_series=False) == {"id": [3]}


def test_diff_accepts_zero_arg_callables_returning_supported_inputs():
    source = lambda: pl.DataFrame({"id": [1, 2], "x": ["a", "b"]})
    target = lambda: pl.DataFrame({"id": [1, 2], "x": ["a", "c"]})

    result = diff(source, target, keys="id")

    assert result.n_diffs == 1


def test_diff_accepts_partials_as_zero_arg_callables():
    def load_table(name, *, tables):
        return tables[name]

    tables = {
        "source": pl.DataFrame({"id": [1, 2], "x": ["a", "b"]}),
        "target": pl.DataFrame({"id": [1, 2], "x": ["a", "c"]}),
    }

    result = diff(
        partial(load_table, "source", tables=tables),
        partial(load_table, "target", tables=tables),
        keys="id",
    )

    assert result.n_diffs == 1


def test_diff_accepts_one_item_loader_calls_with_scalar_payloads():
    tables = {
        "source": pl.DataFrame({"id": [1, 2], "x": ["a", "b"]}),
        "target": pl.DataFrame({"id": [1, 2], "x": ["a", "c"]}),
    }

    def load_table(name, *, tables):
        return tables[name]

    result = diff(
        {"table": "source"},
        {"table": "target"},
        keys="id",
        loaders={"table": partial(load_table, tables=tables)},
    )

    assert result.n_diffs == 1


def test_diff_accepts_one_item_loader_calls_with_structured_payloads():
    tables = {
        ("people", "source"): pl.DataFrame({"id": [1, 2], "x": ["a", "b"]}),
        ("people", "target"): pl.DataFrame({"id": [1, 2], "x": ["a", "c"]}),
    }

    def load_extract(args, *, tables):
        return tables[(args["dataset"], args["side"])]

    result = diff(
        {"extract": {"dataset": "people", "side": "source"}},
        {"extract": {"dataset": "people", "side": "target"}},
        keys="id",
        loaders={"extract": partial(load_extract, tables=tables)},
    )

    assert result.n_diffs == 1


def test_loader_call_must_be_one_item_mapping():
    with pytest.raises(TypeError, match="one-item loader calls"):
        diff(
            {"table": "source", "extra": "not allowed here"},
            {"table": "target"},
            keys="id",
            loaders={"table": lambda name: pl.DataFrame({"id": [1], "x": [name]})},
        )


def test_normalisation_runs_before_key_validation_and_diffing():
    source = pl.DataFrame({"id": ["1.0"], "x": [" A "]})
    target = pl.DataFrame({"id": ["1"], "x": ["A"]})

    result = diff(source, target, keys="id", normalise="float_strings")

    assert result.n_diffs == 0
    assert result.source_only_rows.collect().height == 0
    assert result.target_only_rows.collect().height == 0
```

## `tests\test_core.py`

```python
import polars as pl
from polars.testing import assert_frame_equal

from lightweight_table_diff.core import diff_tbls


class TestCoreDiffing:
    def test_diff_tbls_unpivots_mismatched_cells_and_ignores_identical_ones(self):
        # Arrange
        source_data = pl.LazyFrame(
            [
                # pk_1, pk_2, name,      status,     score
                (1,     "A",  "Alice",   "Active",   "10"),
                (2,     "B",  "Bob",     "Active",   "20"),
                (3,     "C",  "Charlie", "Inactive", "30"),
                (4,     "D",  "Diana",   None,       "40"),
                (5,     "E",  "Eve",     "Active",   "50"),
            ],
            schema=["pk_1", "pk_2", "name", "status", "score"],
            orient="row"
        )

        target_data = pl.LazyFrame(
            [
                # pk_1, pk_2, name,      status,     score
                (1,     "A",  "Alice",   "Active",   "10"),   # Completely identical -> should be excluded from output
                (2,     "B",  "Robert",  "Active",   "20"),   # One column changed (name)
                (3,     "C",  "Charlie", "Active",   "35"),   # Two columns changed (status, score)
                (4,     "D",  "Diana",   "Active",   "40"),   # Null in source populated in target (status)
                (5,     "E",  "Eve",     "Active",   None),   # Populated in source cleared to Null in target (score)
            ],
            schema=["pk_1", "pk_2", "name", "status", "score"],
            orient="row"
        )

        expected = pl.DataFrame(
            [
                # pk_1, pk_2, col_name, source_val, target_val
                (2,     "B",  "name",   "Bob",      "Robert"),
                (3,     "C",  "status", "Inactive", "Active"),
                (3,     "C",  "score",  "30",       "35"),
                (4,     "D",  "status", None,       "Active"),
                (5,     "E",  "score",  "50",       None),
            ],
            schema=["pk_1", "pk_2", "col_name", "source_val", "target_val"],
            orient="row"
        )

        # Act
        actual = diff_tbls(
            source=source_data,
            target=target_data,
            id_cols=["pk_1", "pk_2"],
            compare_cols=["name", "status", "score"],
            join_type="inner",
        ).collect()

        # Assert
        # Sort both to ensure the assertion does not fail on deterministic but arbitrary row ordering
        actual_sorted = actual.sort(["pk_1", "col_name"])
        expected_sorted = expected.sort(["pk_1", "col_name"])

        assert_frame_equal(actual_sorted, expected_sorted)
```

## `tests\test_dimensions.py`

```python
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from lightweight_table_diff.dimensions import (
    validate_key_uniqueness,
    row_presence_differences,
    column_presence_differences,
)


class TestDimensions:
    def test_validate_key_uniqueness_raises_error_on_duplicates(self):
        # Arrange
        source_data = pl.LazyFrame(
            [
                # id, val
                (1,   "A"),
                (2,   "B"),
                (2,   "C"),  # Duplicate key '2'
                (3,   "D"),
            ],
            schema=["id", "val"],
            orient="row"
        )
        
        target_data = pl.LazyFrame(
            [
                # id, val
                (1,   "A"),
                (4,   "B"),
                (4,   "C"),  # Duplicate key '4'
                (5,   "D"),
            ],
            schema=["id", "val"],
            orient="row"
        )

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            validate_key_uniqueness(source_data, target_data, keys=["id"])

        error_msg = str(exc_info.value)
        assert "Duplicate keys (would cause row explosion)" in error_msg
        assert "source: 1 duplicate key group" in error_msg
        assert "target: 1 duplicate key group" in error_msg

    def test_row_and_column_presence_differences_identify_structural_changes(self):
        # Arrange
        source_data = pl.LazyFrame(
            [
                # id, shared_col, source_only_col
                (1,   "A",        10),
                (2,   "B",        20),
                (3,   "C",        30),  # Row 3 exists only in the source
            ],
            schema=["id", "shared_col", "source_only_col"],
            orient="row"
        )
        
        target_data = pl.LazyFrame(
            [
                # id, shared_col, target_only_col
                (1,   "A",        100),
                (2,   "B",        200),
                (4,   "D",        400),  # Row 4 exists only in the target
            ],
            schema=["id", "shared_col", "target_only_col"],
            orient="row"
        )

        expected_source_only_rows = pl.DataFrame({"id": [3]})
        expected_target_only_rows = pl.DataFrame({"id": [4]})

        # Act
        source_only_cols, target_only_cols = column_presence_differences(source_data, target_data, keys=["id"])
        source_only_rows_lf, target_only_rows_lf = row_presence_differences(source_data, target_data, id_cols=["id"])

        # Assert
        assert source_only_cols == ["source_only_col"]
        assert target_only_cols == ["target_only_col"]
        
        assert_frame_equal(source_only_rows_lf.collect(), expected_source_only_rows)
        assert_frame_equal(target_only_rows_lf.collect(), expected_target_only_rows)
```

## `tests\test_integration.py`

```python
import polars as pl
from polars.testing import assert_frame_equal

from lightweight_table_diff.api import diff_lazyframes


class TestIntegration:
    def test_diff_lazyframes_captures_all_structural_and_data_changes_simultaneously(self):
        # Arrange
        source_data = pl.LazyFrame(
            [
                # employee_id, name,      department,    salary, legacy_code
                (1,            "Alice",   "HR",          50000,  "A1"), # No changes at all.
                (2,            "Bob",     "Engineering", 60000,  "B2"), # Cell changes in department and salary.
                (3,            "Charlie", "Sales",       40000,  "C3"), # Exists only in source.
                (4,            "Diana",   "Marketing",   70000,  "D4"), # Salary data type changes from int to float.
            ],
            schema={"employee_id": pl.Int64, "name": pl.String, "department": pl.String, "salary": pl.Int64, "legacy_code": pl.String},
            orient="row"
        )

        target_data = pl.LazyFrame(
            [
                # employee_id, name,      department,   salary,  bonus_eligible
                (1,            "Alice",   "HR",         50000.0, True),
                (2,            "Bob",     "Management", 85000.0, True),
                (4,            "Diana",   "Marketing",  70000.0, False),
                (5,            "Eve",     "Sales",      45000.0, True),  # Eve exists only in the target
            ],
            schema={"employee_id": pl.Int64, "name": pl.String, "department": pl.String, "salary": pl.Float64, "bonus_eligible": pl.Boolean},
            orient="row"
        )

        expected_source_only_cols = ["legacy_code"]
        expected_target_only_cols = ["bonus_eligible"]

        expected_type_changes = pl.DataFrame(
            [
                # col_name, source_type, target_type
                ("salary",  "Int64",     "Float64"),
            ],
            schema=["col_name", "source_type", "target_type"],
            orient="row"
        )

        expected_source_only_rows = pl.DataFrame({"employee_id": [3]})
        expected_target_only_rows = pl.DataFrame({"employee_id": [5]})

        # Cell differences are value differences with Polars comparison.
        # Diana's salary changed type from Int64 to Float64, which is captured in
        # type_changes, but 70000 and 70000.0 compare equal, so it is not a cell diff.
        expected_cells = pl.DataFrame(
            [
                # employee_id, col_name,     source_val,    target_val
                (2,            "department", "Engineering", "Management"),
                (2,            "salary",     "60000",       "85000.0"),
            ],
            schema={"employee_id": pl.Int64, "col_name": pl.String, "source_val": pl.String, "target_val": pl.String},
            orient="row"
        )

        # Act
        result = diff_lazyframes(
            source=source_data,
            target=target_data,
            keys=["employee_id"],
            join_type="inner",
        )

        # Assert
        assert result.source_only_cols == expected_source_only_cols
        assert result.target_only_cols == expected_target_only_cols

        assert_frame_equal(result.type_differences, expected_type_changes)
        
        assert_frame_equal(result.source_only_rows.collect(), expected_source_only_rows)
        assert_frame_equal(result.target_only_rows.collect(), expected_target_only_rows)

        actual_cells = result.diff.collect().sort(["employee_id", "col_name"])
        assert_frame_equal(actual_cells, expected_cells)
```

## `tests\test_normalisers.py`

```python
import polars as pl
from polars.testing import assert_frame_equal

from lightweight_table_diff.normalisers import normalise_float_strings


class TestNormalisers:
    def test_normalise_float_strings(self):
        # Arrange
        input_data = pl.LazyFrame(
            [
                # id, value_a,  value_b
                ("1", "100.0",  "NaN"),   # Valid floats stripped of trailing .0; 'NaN' string maps to true null
                ("2", "100.00", "N/A"),   # Multiple trailing zeros stripped; 'N/A' maps to true null
                ("3", "0.0",    "null"),  # Zero stripped; 'null' maps to true null
                ("4", "-5.0",   "   "),   # Negatives supported; Whitespace-only strings map to true null
                ("5", "3.14",   "<NA>"),  # Genuine decimals left untouched; '<NA>' maps to true null
            ],
            schema=["id", "value_a", "value_b"],
            orient="row"
        )

        expected = pl.DataFrame(
            [
                # id, value_a, value_b
                ("1", "100",   None),
                ("2", "100",   None),
                ("3", "0",     None),
                ("4", "-5",    None),
                ("5", "3.14",  None),
            ],
            # Explicit schema forces value_b to be recognised as String type despite containing all nulls
            schema={"id": pl.String, "value_a": pl.String, "value_b": pl.String},
            orient="row"
        )

        # Act
        actual = normalise_float_strings(
            lf=input_data, 
            keys=["id"], 
            cols=["value_a", "value_b"]
        ).collect()

        # Assert
        assert_frame_equal(actual, expected)
```

