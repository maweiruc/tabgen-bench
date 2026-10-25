from dataclasses import replace
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tabgen_bench.core.config import FrozenConfig
from tabgen_bench.core.results import MetricResult
from tabgen_bench.metrics.registry import PluginRegistry
from tests.fakes import FakeMetric, toy_table
from tests.support_metrics import context
from tests.unit.test_results import record


def test_m0_result_json_still_loads_and_keeps_explicit_version():
    data = record().model_dump(mode="json")
    for key in ("details", "started_at", "ended_at"):
        del data[key]
    data["result_schema_version"] = "1.0.0"
    old = MetricResult.model_validate(data)
    assert old.result_schema_version == "1.0.0" and old.details.to_dict() == {}
    assert MetricResult.model_validate_json(old.model_dump_json()) == old


def test_metric_diagnostics_frozen_json_roundtrip_and_times():
    payload = {"columns": [{"reason_code": "non_integer", "count": 1}]}
    now = datetime.now(UTC)
    result = record(details=payload, started_at=now, ended_at=now)
    payload["columns"][0]["count"] = 999
    detached = result.details.to_dict()
    detached["columns"][0]["count"] = 333
    assert result.details.to_dict()["columns"][0]["count"] == 1
    assert MetricResult.model_validate_json(result.model_dump_json()) == result
    with pytest.raises(ValidationError):
        record(started_at=datetime(2026, 10, 25))


def test_optional_inputs_are_opt_in_and_missing_optional_is_allowed(schema):
    from tabgen_bench.metrics.validity.schema_checks import plugins

    plugin = plugins()[0]
    ctx = context(schema, toy_table(), expected_n_rows=4, real_train=toy_table())
    scoped = ctx.for_plugin(plugin)
    assert scoped.expected_n_rows == 4 and scoped.real_train is None
    assert not context(schema, toy_table()).missing_inputs(plugin)
    assert ctx.for_plugin(FakeMetric()).expected_n_rows is None
    assert isinstance(plugin.params, FrozenConfig)


def test_registry_rejects_invalid_optional_input_declarations():
    from tabgen_bench.core.errors import InvalidPluginError
    from tabgen_bench.metrics.validity.schema_checks import plugins

    for invalid in ({"real_train"}, frozenset({"secret_input"})):
        with pytest.raises(InvalidPluginError):
            PluginRegistry([replace(plugins()[0], optional_inputs=invalid)])


def test_execution_policy_rejects_unscheduled_gate_and_duplicate_preflight():
    from tabgen_bench.core.execution import EvaluationPolicy, ResultGate

    with pytest.raises(ValidationError):
        EvaluationPolicy(preflight_metric_ids=())
    with pytest.raises(ValidationError):
        EvaluationPolicy(preflight_metric_ids=("example.gate", "example.gate"), gates=())
    with pytest.raises(ValidationError):
        ResultGate(metric_id="example.gate", required_input="undeclared", reason="test")
