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
