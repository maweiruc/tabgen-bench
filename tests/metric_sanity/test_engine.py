from dataclasses import dataclass

import pytest

from tabgen_bench.core.base import FrozenModel
from tabgen_bench.core.context import ContextExtension, ExtensionRequirement
from tabgen_bench.core.execution import EvaluationPolicy, ResultGate
from tabgen_bench.core.hashing import config_hash
from tabgen_bench.core.results import MetricResult
from tabgen_bench.evaluation.engine import EvaluationEngine
from tabgen_bench.interfaces.api import build_metric_registry, evaluate_only
from tests.fakes import FakeMetric, toy_table
from tests.support_metrics import binary_fixture, context


@dataclass(frozen=True)
class FutureMetric(FakeMetric):
    def compute(self, ctx, *, seed):
        assert ctx.real_train is None and ctx.real_test is None
        for requirement in self.required_capabilities:
            assert ctx.extension(requirement, FuturePayload).test_value == 7
        return MetricResult(
            run_id="fake",
            dataset_id="fake",
            metric_id=self.id,
            metric_domain=self.domain,
            metric_version=self.version,
            direction=self.direction,
            value=7,
            status="ok",
            unit="test-only",
            reference_split="none",
            method_config_hash=config_hash({}),
            notes="test-only interface result",
        )


class FuturePayload(FrozenModel):
    test_value: int


@pytest.mark.parametrize("domain", ["statistical", "privacy"])
def test_future_plugin_runs_without_engine_change(schema, domain):
    requirement = ExtensionRequirement(namespace=f"example.future_{domain}", version="test-1")
    registry = build_metric_registry()
    plugin = FutureMetric(
        id=f"example.future_{domain}", domain=domain, required_capabilities=frozenset({requirement})
    )
    registry.register(plugin)
    ctx = context(
        schema,
        toy_table(),
        real_train=toy_table(),
        extensions=(ContextExtension(requirement, FuturePayload(test_value=7)),),
    )
    results = EvaluationEngine(registry, run_id="real-evaluation", dataset_id="toy").evaluate(
        ctx, metric_ids=(plugin.id,), seed=11
    )
    result = results[-1]
    assert result.status == "ok" and result.value == 7 and result.metric_domain == domain
    assert result.run_id == "real-evaluation" and result.dataset_id == "toy"
    assert result.started_at is not None and result.ended_at >= result.started_at


def test_failed_metric_isolated(schema):
    class Broken(FakeMetric):
        def compute(self, context, *, seed):
            raise RuntimeError("deliberate test failure")

    registry = build_metric_registry()
    registry.register(Broken(id="example.broken"))
    results = EvaluationEngine(registry).evaluate(
        context(schema, toy_table(), real_reference=toy_table()),
        metric_ids=("example.broken", "marginal.ks_mean"),
        seed=11,
    )
    broken, good = results[-2:]
    assert broken.status == "failed" and broken.value is None
    assert broken.error_type == "RuntimeError" and "deliberate" in broken.error_message
    assert good.status == "ok" and good.value == 0


def test_missing_input_is_not_applicable(schema):
    results = evaluate_only(
        context(schema, toy_table()),
        seed=11,
        metric_ids=("utility.tstr_auc", "proximity.dcr_train_median"),
    )
    for result in results[-2:]:
        assert result.status == "not_applicable" and result.value is None
        assert result.notes.startswith("missing_input:")


def test_metric_missing_inputs_not_applicable(schema):
    test_missing_input_is_not_applicable(schema)


def test_missing_capability_is_not_applicable(schema):
    registry = build_metric_registry()
    requirement = ExtensionRequirement(namespace="example.future", version="test-1")
    plugin = FutureMetric(id="example.future", required_capabilities=frozenset({requirement}))
    registry.register(plugin)
    result = EvaluationEngine(registry).evaluate(
        context(schema, toy_table()), metric_ids=(plugin.id,), seed=11
    )[-1]
    assert result.status == "not_applicable" and result.value is None
    assert result.notes.startswith("missing_capability:")


def test_gate_prevents_invalid_synthetic_plugin_execution(schema):
    class MustNotRun(FakeMetric):
        def compute(self, ctx, *, seed):
            raise AssertionError("blocked plugin must not execute")

    registry = build_metric_registry()
    registry.register(MustNotRun(id="example.needs_synthetic"))
    frame = toy_table()
    frame.loc[0, "age"] = -12
    result = EvaluationEngine(registry).evaluate(
        context(schema, frame), metric_ids=("example.needs_synthetic",), seed=11
    )[-1]
    assert result.status == "not_applicable" and result.notes == "blocked_by_validity"


def test_generic_injected_policy_does_not_need_metric_domain_branch(schema):
    registry = build_metric_registry()
    registry.register(FutureMetric(id="example.preflight", domain="statistical"))
    registry.register(FutureMetric(id="example.dependent", domain="privacy"))
    policy = EvaluationPolicy(
        preflight_metric_ids=("example.preflight",),
        gates=(
            ResultGate(
                metric_id="example.preflight",
                required_input="synthetic",
                passed_values=(7,),
                exempt_metric_ids=("example.preflight",),
                reason="test_gate",
            ),
        ),
    )
    results = EvaluationEngine(registry, policy=policy).evaluate(
        context(schema, toy_table()), metric_ids=("example.dependent",), seed=11
    )
    assert [r.status for r in results] == ["ok", "ok"]


def test_nonfinite_and_identity_mismatch_plugins_fail_without_hiding_other_results(schema):
    class BadIdentity(FakeMetric):
        def compute(self, ctx, *, seed):
            result = super().compute(ctx, seed=seed)
            return result.model_copy(update={"metric_id": "example.wrong"})

    registry = build_metric_registry()
    registry.register(BadIdentity(id="example.bad_identity"))
    results = EvaluationEngine(registry).evaluate(
        context(schema, toy_table()), metric_ids=("example.bad_identity",), seed=11
    )
    assert results[-1].status == "failed" and results[-1].error_type == "ValueError"
    assert len(results) == 4


def test_all_v1_metrics_compute_and_serialize_without_sdv():
    schema, target, train, test, validation = binary_fixture()
    results = evaluate_only(
        context(
            schema,
            train,
            target=target,
            real_train=train,
            real_test=test,
            real_validation=validation,
        ),
        seed=11,
    )
    assert len(results) == 17
    assert all(r.status == "ok" for r in results), [
        (r.metric_id, r.notes, r.error_message) for r in results if r.status != "ok"
    ]
    assert all(MetricResult.model_validate_json(r.model_dump_json()) == r for r in results)
    assert {r.metric_domain for r in results} == {"validity", "fidelity", "utility", "privacy"}


def test_evaluation_only_reference_path_computes_fidelity_without_train():
    schema, _, train, test, _ = binary_fixture()
    results = {
        r.metric_id: r for r in evaluate_only(context(schema, train, real_reference=test), seed=11)
    }
    for metric_id in ("marginal.ks_mean", "dependency.pairwise_tv_mean", "joint.c2st_auc"):
        assert (
            results[metric_id].status == "ok" and results[metric_id].reference_split == "reference"
        )
    assert results["joint.c2st_real_vs_real_auc"].status == "not_applicable"


def test_failed_preflight_blocks_dependent_but_keeps_independent_metric(schema):
    from tabgen_bench.metrics.registry import PluginRegistry
    from tabgen_bench.metrics.validity.schema_checks import plugins

    class BrokenGate(FakeMetric):
        def compute(self, ctx, *, seed):
            raise RuntimeError("preflight failed")

    registry = PluginRegistry(
        [BrokenGate(id="validity.schema_pass", domain="validity"), *plugins()[1:]]
    )
    registry.register(FutureMetric(id="example.dependent"))
    registry.register(FutureMetric(id="example.independent", required_inputs=frozenset({"schema"})))
    results = EvaluationEngine(registry).evaluate(
        context(schema, toy_table()),
        metric_ids=("example.dependent", "example.independent"),
        seed=11,
    )
    assert results[0].status == "failed"
    assert results[-2].notes == "blocked_by_validity"
    assert results[-1].status == "ok"
    assert results[0].details.to_dict()["evaluation"]["validity_rules_hash"]
