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