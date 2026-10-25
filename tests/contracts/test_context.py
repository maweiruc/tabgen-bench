import inspect
from dataclasses import FrozenInstanceError, replace

import pandas as pd
import pytest

from tabgen_bench.core.base import FrozenModel
from tabgen_bench.core.context import (
    ContextExtension,
    EvaluationContext,
    ExtensionRequirement,
    InputIdentity,
)
from tabgen_bench.core.contracts import EvaluationEngine, MetricRegistry
from tabgen_bench.core.errors import MissingCapabilityError
from tabgen_bench.core.hashing import config_hash
from tabgen_bench.metrics.registry import PluginRegistry
from tests.fakes import FakeMetric, toy_table


def identity(name, split):
    return InputIdentity(
        input_name=name,
        source_id="local-toy-v1",
        split_id="fixed-test-split",
        reference_split=split,
        content_hash=config_hash({"test_input": name}),
    )


def test_evaluation_context_detaches_mutable_tables(schema):
    original = toy_table()
    context = EvaluationContext(
        schema=schema, synthetic=original, identities=(identity("synthetic", "synthetic"),)
    )
    original.iloc[0, 0] = 999
    assert context.synthetic.iloc[0, 0] == 25
    read = context.synthetic
    read.iloc[0, 0] = 888
    assert context.synthetic.iloc[0, 0] == 25
    with pytest.raises(FrozenInstanceError):
        context.target = None


def test_context_preserves_invalid_structure_without_repair(schema):
    invalid = pd.DataFrame([[{"invalid": [1]}, "A"]], columns=["same", "same"])
    context = EvaluationContext(
        schema=schema, synthetic=invalid, identities=(identity("synthetic", "synthetic"),)
    )
    invalid.iat[0, 0]["invalid"].append(2)
    detached = context.synthetic
    detached.iat[0, 0]["invalid"].append(3)
    assert context.synthetic.iat[0, 0] == {"invalid": [1]}
    assert list(context.synthetic.columns) == ["same", "same"]


def test_required_inputs_scope_tables_and_source_identities(schema):
    context = EvaluationContext(
        schema=schema,
        synthetic=toy_table(),
        real_train=toy_table(),
        real_test=toy_table(),
        expected_n_rows=3,
        identities=(
            identity("synthetic", "synthetic"),
            identity("real_train", "train"),
            identity("real_test", "test"),
        ),
    )
    plugin = FakeMetric()
    restricted = context.for_plugin(plugin)
    assert restricted.real_train is None and restricted.real_test is None
    assert restricted.expected_n_rows is None
    assert len(restricted.identities) == 1
    assert restricted.synthetic.equals(toy_table())
    assert not restricted.missing_inputs(plugin)
    missing = replace(plugin, required_inputs=frozenset({"real_reference", "target"}))
    assert context.missing_inputs(missing) == frozenset({"real_reference", "target"})
    assert "real_test" not in restricted.available_inputs


class FakeFuturePayload(FrozenModel):
    test_token: str


class OtherPayload(FrozenModel):
    other: int


class NestedPayload(FrozenModel):
    test_values: list[int]


def test_extension_detaches_nested_payloads():
    requirement = ExtensionRequirement(namespace="example.test", version="1")
    payload = NestedPayload(test_values=[1])
    extension = ContextExtension(requirement, payload)
    payload.test_values.append(2)
    extension.payload.test_values.append(3)
    assert extension.payload.test_values == [1]


@pytest.mark.parametrize("domain", ["statistical", "privacy"])
def test_future_capability_typed_versioned_and_scoped(schema, domain):
    requirement = ExtensionRequirement(namespace=f"example.{domain}_fixture", version="test-1")
    extension = ContextExtension(requirement, FakeFuturePayload(test_token="interface-only"))
    context = EvaluationContext(schema=schema, extensions=(extension,))
    plugin = replace(FakeMetric(), domain=domain, required_capabilities=frozenset({requirement}))
    registry = PluginRegistry([plugin])
    scoped = context.for_plugin(registry.get(plugin.id))
    assert scoped.extension(requirement, FakeFuturePayload).test_token == "interface-only"
    assert not context.missing_capabilities(plugin)
    with pytest.raises(TypeError):
        scoped.extension(requirement, OtherPayload)
    with pytest.raises(MissingCapabilityError):
        context.for_plugin(FakeMetric()).extension(requirement, FakeFuturePayload)
    changed = requirement.model_copy(update={"version": "test-2"})
    with pytest.raises(MissingCapabilityError):
        scoped.extension(changed, FakeFuturePayload)
    assert context.missing_capabilities(replace(plugin, required_capabilities=frozenset({changed})))
    assert extension.fingerprint() != ContextExtension(changed, extension.payload).fingerprint()


def test_context_requires_explicit_source_and_split(schema):
    with pytest.raises(ValueError, match="identity"):
        EvaluationContext(schema=schema, synthetic=toy_table())
    with pytest.raises(ValueError, match="split"):
        EvaluationContext(
            schema=schema, real_train=toy_table(), identities=(identity("real_train", "test"),)
        )


def test_engine_constructor_contract_is_core_injected_registry():
    constructor = inspect.signature(EvaluationEngine.__init__)
    assert tuple(constructor.parameters) == ("self", "metric_registry")
    assert constructor.parameters["metric_registry"].annotation is MetricRegistry
    assert EvaluationContext.__module__ == "tabgen_bench.core.context"
    assert MetricRegistry.__module__ == "tabgen_bench.core.contracts"


def test_same_injected_consumer_accepts_future_plugin_without_domain_branch(schema):
    # An interface consumer, not an Engine implementation: no suite, execution
    # policy, error isolation, or applicability logic is supplied by this consumer.
    def consume(registry: MetricRegistry, metric_id):
        plugin = registry.get(metric_id)
        return plugin.compute(EvaluationContext(schema=schema).for_plugin(plugin), seed=11)

    registry = PluginRegistry([FakeMetric(id="test.base")])
    registry.register(FakeMetric(id="test.future", domain="statistical"))
    assert consume(registry, "test.future").metric_domain == "statistical"
