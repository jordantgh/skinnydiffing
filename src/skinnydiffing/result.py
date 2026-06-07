"""Result object returned by the public diff API."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

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
            self._n_diffs = cast(
                pl.DataFrame, self.diff.select(pl.len()).collect()
            ).item()
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
            bool: True if absolutely no structural or data differences exist, False
                otherwise.
        """
        if not self.type_differences.is_empty():
            return False
        if self.source_only_cols or self.target_only_cols:
            return False
        if self.n_diffs > 0:
            return False
        if (
            cast(pl.DataFrame, self.source_only_rows.select(pl.len()).collect()).item()
            > 0
        ):
            return False
        return (
            not cast(
                pl.DataFrame, self.target_only_rows.select(pl.len()).collect()
            ).item()
            > 0
        )

    def write(
        self, output_dir: str | Path, *, basename: str | None = None
    ) -> dict[str, Path]:
        """
        Save all non-empty difference tables to CSV files in a specified directory.

        Structural differences (shape and data type differences) and the detailed
        cell-level differences are written to disk. A summary table grouping identical
        cell diffs by frequency is also generated. Files are only created if differences
        actually exist in that category.

        Args:
            output_dir: The directory where the CSV files will be saved. Created if it
                does not exist.
            basename: A prefix for the filenames. Defaults to the `name` attribute of
                the DiffResult, or "diff".

        Returns:
            dict[str, Path]: A dictionary mapping the logical name of the output
                (e.g., 'cells', 'source_only_rows') to the absolute path of the written
                file.
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
            n = cast(pl.DataFrame, lf.select(pl.len()).collect()).item()
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
