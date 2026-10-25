import json

import pytest
from pydantic import ValidationError

from tabgen_bench.core.hashing import config_hash
from tabgen_bench.core.results import MetricResult


def record(**patch):
    return MetricResult(
        **{
            "run_id": "test-only",
            "dataset_id": "toy",
            "metric_id": "marginal.ks_mean",
            "metric_domain": "fidelity",
            "metric_version": "test-1",
            "value": 0.25,
            "direction": "minimize",
            "unit": "distance",
            "status": "ok",
            "reference_split": "test",
            "n_real": 3,
            "n_synthetic": 3,
            "method_config_hash": config_hash({}),
            **patch,
        }
    )


def test_result_json_roundtrip_and_identity():
    result = record()
    assert MetricResult.model_validate_json(result.model_dump_json()) == result
    assert result.metric_id == "marginal.ks_mean" and result.reference_split == "test"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_result_becomes_explicit_failed_null(value):
    data = json.loads(record(value=value).model_dump_json())
    assert data["value"] is None
    assert data["status"] == "failed" and data["error_type"] == "non_finite_value"


@pytest.mark.parametrize(
    "patch",
    [
        {"status": "failed", "value": 0},
        {"status": "failed", "value": None},
        {"status": "not_applicable", "value": None},
        {"status": "ok", "value": None},
        {"metric_domain": "statistical"},
        {"n_real": -1},
    ],
)
def test_invalid_result_states_rejected(patch):
    with pytest.raises(ValidationError):
        record(**patch)


def test_missing_input_and_exception_records_roundtrip():
    for result in (
        record(status="not_applicable", value=None, notes="missing_input: real_test"),
        record(
            status="failed", value=None, error_type="test_error", error_message="fake plugin failed"
        ),
    ):
        assert MetricResult.model_validate_json(result.model_dump_json()) == result
