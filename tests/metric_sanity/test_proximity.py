import numpy as np
import pandas as pd
import pytest

from tabgen_bench.core.schema import ColumnSchema, DatasetSchema
from tabgen_bench.interfaces.api import evaluate_only
from tabgen_bench.metrics.privacy import proximity
from tests.support_metrics import context


def schema_for(*names):
    return DatasetSchema(
        columns=tuple(
            ColumnSchema(name=n, role="feature", logical_type="numeric", nullable=True)
            for n in names
        )
    )


def test_dcr_known_distances_train_scale_and_strict_closer():
    schema = schema_for("x")
    train = pd.DataFrame({"x": [0, 10]})
    test = pd.DataFrame({"x": [2, 8]})
    results = {
        r.metric_id: r
        for r in evaluate_only(context(schema, train, real_train=train, real_test=test), seed=11)
    }
    assert results["proximity.dcr_train_median"].value == 0
    assert results["proximity.dcr_holdout_median"].value == pytest.approx(0.2)
    assert results["proximity.train_closer_rate"].value == 1
    assert results["proximity.dcr_tie_rate"].value == 0
    assert results["proximity.exact_train_copy_rate"].value == 1
    assert results["proximity.dcr_train_median"].direction == "diagnostic"


def test_dcr_constants_missing_and_duplicates():
    schema = DatasetSchema(
        columns=(
            ColumnSchema(name="x", role="feature", logical_type="numeric", nullable=True),
            ColumnSchema(name="c", role="feature", logical_type="categorical", nullable=True),
        )
    )
    train = pd.DataFrame({"x": [5.0, 5.0, None, 5.0], "c": ["A", "A", None, None]})
    matrix = proximity.distance_block(
        pd.DataFrame({"x": [6, None], "c": ["A", None]}), train, schema.columns, {"x": 0}
    )
    assert matrix[0, 0] == 0.5 and matrix[1, 2] == 0
    assert np.isfinite(matrix).all()
    results = {
        r.metric_id: r
        for r in evaluate_only(
            context(schema, train, real_train=train, real_test=train.copy()), seed=11
        )
    }
    assert results["proximity.dcr_tie_rate"].value == 1
    assert results["proximity.train_closer_rate"].value == 0
    assert results["proximity.exact_train_copy_rate"].value == 1


def test_dcr_equal_reference_caps_and_bounded_blocks(monkeypatch):
    schema = schema_for("x")
    train = pd.DataFrame({"x": np.arange(1100)})
    test = pd.DataFrame({"x": np.arange(1200)})
    synthetic = pd.DataFrame({"x": np.arange(600)})
    blocks = []
    original = proximity.distance_block

    def capture(left, right, *args):
        blocks.append((len(left), len(right)))
        return original(left, right, *args)

    monkeypatch.setattr(proximity, "distance_block", capture)
    result = evaluate_only(
        context(schema, synthetic, real_train=train, real_test=test),
        seed=11,
        metric_ids=("proximity.dcr_train_median",),
    )[-1]
    assert result.status == "ok"
    audit = result.details.to_dict()
    assert audit["n_ref_each"] == 1024 and audit["n_synthetic_used"] == 512
    assert len(audit["train_positions"]) == len(audit["holdout_positions"]) == 1024
    assert all(rows <= 64 and refs <= 128 for rows, refs in blocks)
    assert "privacy guarantee" in audit["disclaimer"]


def test_exact_copy_checks_full_train_and_lossless_numeric_storage_with_na():
    schema = schema_for("x")
    train = pd.DataFrame({"x": [*range(1100), None]})
    synthetic = pd.DataFrame({"x": [1099.0, None] * 300})
    result = evaluate_only(
        context(schema, synthetic, real_train=train),
        seed=23,
        metric_ids=("proximity.exact_train_copy_rate",),
    )[-1]
    assert result.value == 1 and result.details.to_dict()["n_checked"] == 600
