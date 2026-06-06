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
