"""Resolve public table inputs into Polars LazyFrames."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from os import PathLike
from pathlib import Path
from typing import Any, cast

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
    loaders: Mapping[str, Callable[[Any], TableLike | Callable[[], TableLike]]]
    | None = None,
    collect_lazy: bool = False,
    file_format: str | None = None,
    glob: str | None = None,
    **scan_options: Any,
) -> pl.LazyFrame:
    """
    Convert a file, dataframe object (or callable returning either of those) into a Polars
    LazyFrame.

    It supports direct file paths (e.g., CSV, Parquet, SPSS), Pandas and Narwhals
    compatible dataframes, zero-argument functions (can also be partial functions), or
    dictionary based loader configurations  (e.g., `{'spark': 'db.table'}`). If a loader
    dictionary is provided, it extracts the  data using the matching function provided in
    the `loaders` argument.

    Args:
        obj: The input data. Can be a path, a dataframe, a function, or a loader
            dictionary. The loader dictionary has the form {loader_name: payload}, where
            loader_name is used in the `loaders` mapping to find the corresponding loading
            function, and payload is passed to that function.
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
        ValueError: If a custom loader is requested but not found in the `loaders`
            dictionary.
    """
    if callable(obj):
        callable_obj = cast(Callable[[], TableLike], obj)
        return into_lazyframe(
            callable_obj(),
            loaders=loaders,
            collect_lazy=collect_lazy,
            file_format=file_format,
            glob=glob,
            **scan_options,
        )

    if isinstance(obj, Mapping):
        return _load_from_call(
            cast(Mapping[str, Any], obj),
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
            Path(cast(Any, obj)),
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
        **options: Additional arguments passed directly to the underlying scanning
            function (e.g., `ignore_errors=True` for CSVs).

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
        options.setdefault("hive_partitioning", False)
        return pl.scan_parquet(scan_target, **options)

    if fmt == "csv":
        scan_target = _path_with_glob(path, glob or "*.csv")
        options.setdefault("ignore_errors", False),
        return pl.scan_csv(scan_target, **options)

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
                "pip install 'skinnydiffing[readstat]'"
            ) from None

        return scan_readstat(str(path), **options)

    raise ValueError(f"Unsupported table format {fmt!r}")


def _load_from_call(
    call: Mapping[str, Any],
    *,
    loaders: Mapping[str, Callable[[Any], TableLike | Callable[[], TableLike]]]
    | None = None,
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
