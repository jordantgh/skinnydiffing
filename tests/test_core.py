import polars as pl
from polars.testing import assert_frame_equal

from skinnydiffing.core import diff_tbls


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
