import polars as pl
from polars.testing import assert_frame_equal

from skinnydiffing.api import diff_lazyframes


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
        )

        # Assert
        assert result.source_only_cols == expected_source_only_cols
        assert result.target_only_cols == expected_target_only_cols

        assert_frame_equal(result.type_differences, expected_type_changes)
        
        assert_frame_equal(result.source_only_rows.collect(), expected_source_only_rows)
        assert_frame_equal(result.target_only_rows.collect(), expected_target_only_rows)

        actual_cells = result.diff.collect().sort(["employee_id", "col_name"])
        assert_frame_equal(actual_cells, expected_cells)
