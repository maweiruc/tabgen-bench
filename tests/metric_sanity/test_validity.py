import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from tabgen_bench.core.schema import ColumnSchema, DatasetSchema
from tabgen_bench.interfaces.api import evaluate_only
from tests.fakes import toy_table
from tests.support_metrics import context


def validity(schema, frame, **kwargs):
    result = evaluate_only(context(schema, frame, **kwargs), seed=11, metric_ids=())
    return {r.metric_id.split(".")[1]: r for r in result}


def test_validity_all_valid_zero_rates(schema):
    results = validity(schema, toy_table())
    assert results["schema_pass"].value == 1
    assert results["invalid_cell_rate"].value == results["invalid_row_rate"].value == 0
    assert results["schema_pass"].details.to_dict()["column_reasons"] == []


def test_validity_noninteger_out_of_range_unknown_category_null(schema):
    frame = pd.DataFrame({"age": [-12.5, None], "group": ["Unknown", "A"], "label": [True, False]})
    results = validity(schema, frame)
    report = results["schema_pass"].details.to_dict()
    assert results["schema_pass"].value == 0
    assert {v["reason_code"] for v in report["column_reasons"]} == {
        "non_integer",
        "out_of_range",
        "unknown_category",
        "null_not_allowed",
    }
    assert all(v["constraint_source"] and v["rule_version"] for v in report["column_reasons"])


def test_validity_invalid_cell_vs_row_denominators(schema):
    frame = pd.DataFrame({"age": [-12.5, 35], "group": ["Unknown", "A"], "label": [True, None]})
    report = validity(schema, frame)["invalid_cell_rate"].details.to_dict()
    assert report["invalid_cell_count"] == 3 and report["invalid_row_count"] == 2
    assert report["checked_cell_count"] == 6
    assert report["invalid_cell_rate"] == 0.5 and report["invalid_row_rate"] == 1


def test_validity_train_observed_max_not_domain_bound(schema):
    train = toy_table()
    frame = train.copy()
    frame.loc[0, "age"] = 75
    assert validity(schema, frame, real_train=train)["schema_pass"].value == 1


@pytest.mark.parametrize(
    "kind", ["missing", "extra", "order", "duplicate", "empty", "row_count", "unsupported_object"]
)
def test_validity_structural_error_na_rates(schema, kind):
    frame, kwargs = toy_table(), {}
    if kind == "missing":
        frame = frame.drop(columns="group")
    elif kind == "extra":
        frame["extra"] = 1
    elif kind == "order":
        frame = frame[["label", "group", "age"]]
    elif kind == "duplicate":
        frame.columns = ["age", "age", "label"]
    elif kind == "empty":
        frame = frame.iloc[:0]
    elif kind == "row_count":
        kwargs["expected_n_rows"] = 4
    else:
        frame.at[0, "group"] = {"nested": [1]}
    results = validity(schema, frame, **kwargs)
    assert results["schema_pass"].value == 0
    assert results["schema_pass"].details.to_dict()["structural_errors"]
    for name in ("invalid_cell_rate", "invalid_row_rate"):
        assert results[name].status == "not_applicable" and results[name].value is None


def test_validity_no_implicit_repair(schema):
    frame = toy_table()[["group", "age", "label"]]
    before = frame.copy(deep=True)
    snapshot = context(schema, frame)
    results = evaluate_only(snapshot, seed=11)
    assert_frame_equal(frame, before)
    assert_frame_equal(snapshot.synthetic, before)
    assert all(r.status == "not_applicable" for r in results if r.metric_id.startswith("marginal."))
    assert (
        next(r for r in results if r.metric_id == "marginal.ks_mean").notes == "blocked_by_validity"
    )


def test_validity_undeclared_category_not_invalid():
    schema = DatasetSchema(
        columns=(ColumnSchema(name="c", role="feature", logical_type="categorical", nullable=True),)
    )
    assert (
        validity(schema, pd.DataFrame({"c": ["unseen", "unknown", "nan", None]}))[
            "schema_pass"
        ].value
        == 1
    )


@pytest.mark.parametrize("value", ["35", True, np.inf, -np.inf])
def test_numeric_type_mismatch_is_not_coerced_or_repaired(schema, value):
    frame = toy_table().astype({"age": object})
    frame.loc[0, "age"] = value
    report = validity(schema, frame)["schema_pass"].details.to_dict()
    assert not report["structural_errors"]
    assert any(r["reason_code"] == "type_mismatch" for r in report["column_reasons"])
    assert report["invalid_cell_count"] == 1


def test_nullable_integer_float_storage_and_duplicate_rows_are_legal():
    schema = DatasetSchema(
        columns=(ColumnSchema(name="n", role="feature", logical_type="integer", nullable=True),)
    )
    result = validity(schema, pd.DataFrame({"n": [1.0, 1.0, None]}))
    assert result["schema_pass"].value == 1
