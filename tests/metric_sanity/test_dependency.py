import pandas as pd
import pytest

from tabgen_bench.core.schema import ColumnSchema, DatasetSchema
from tabgen_bench.interfaces.api import evaluate_only
from tests.support_metrics import context


def score(ctx):
    return evaluate_only(ctx, metric_ids=("dependency.pairwise_tv_mean",), seed=11)[-1]


def test_pairwise_tv_detects_destroyed_correlation():
    schema = DatasetSchema(
        columns=tuple(
            ColumnSchema(name=n, role="feature", logical_type="categorical", nullable=False)
            for n in ("a", "b")
        )
    )
    real = pd.DataFrame({"a": ["A", "B"] * 2, "b": ["A", "B"] * 2})
    synthetic = pd.DataFrame({"a": ["A", "A", "B", "B"], "b": ["A", "B", "A", "B"]})
    result = score(context(schema, synthetic, real_train=real, real_test=real))
    assert result.value == pytest.approx(0.5)
    assert result.details.to_dict()["bin_fit_input"] == "real_train"


def test_pairwise_train_bins_exclude_test_extremes_and_encode_overflow_na():
    schema = DatasetSchema(
        columns=tuple(
            ColumnSchema(name=n, role="feature", logical_type="numeric", nullable=True)
            for n in ("x", "y")
        )
    )
    train = pd.DataFrame({"x": [0, 1, 1], "y": [0, 0, 1]})
    test = pd.DataFrame({"x": [100, None], "y": [100, None]})
    result = score(context(schema, test, real_train=train, real_test=test))
    assert result.value == 0
    assert max(result.details.to_dict()["bins"]["x"]["quantile_edges"]) == 1
    from tabgen_bench.metrics.fidelity.dependency import encode_column

    values, _ = encode_column(train.x, pd.Series([-1, 100, None]), "numeric")
    assert values == ["UNDERFLOW", "OVERFLOW", "NA"]


def test_reference_only_bins_are_explicit_and_holdout_only_without_train_is_na():
    schema = DatasetSchema(
        columns=tuple(
            ColumnSchema(name=n, role="feature", logical_type="numeric", nullable=False)
            for n in ("x", "y")
        )
    )
    real = pd.DataFrame({"x": [0, 1], "y": [1, 0]})
    assert score(context(schema, real, real_reference=real)).value == 0
    missing = score(context(schema, real, real_test=real))
    assert missing.status == "not_applicable" and missing.value is None


def test_pair_cap_and_train_top_twenty_are_recorded():
    schema = DatasetSchema(
        columns=tuple(
            ColumnSchema(name=f"c{i}", role="feature", logical_type="categorical", nullable=False)
            for i in range(12)
        )
    )
    real = pd.DataFrame({c.name: [f"v{i}" for i in range(30)] for c in schema.columns})
    result = score(context(schema, real, real_train=real, real_test=real))
    details = result.details.to_dict()
    assert len(details["pairs"]) == 45
    assert len(details["bins"]["c0"]["top_categories"]) == 20
    assert result.value == 0
