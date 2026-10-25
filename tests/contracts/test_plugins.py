from dataclasses import replace

import pytest

from tabgen_bench.core.context import EvaluationContext
from tabgen_bench.core.contracts import (
    METRIC_DOMAINS,
    GeneratorAdapter,
    MetricPlugin,
    MetricRegistry,
)
from tabgen_bench.core.errors import DuplicateMetricError, InvalidPluginError, UnknownMetricError
from tabgen_bench.metrics.registry import PluginRegistry
from tests.fakes import FakeGenerator, FakeMetric, toy_table


@pytest.mark.parametrize("domain", sorted(METRIC_DOMAINS))
def test_metric_domain_registry_fake(domain, schema):
    fake = FakeMetric(id=f"test.{domain}", domain=domain)
    registry = PluginRegistry([fake])
    assert isinstance(fake, MetricPlugin)
    assert isinstance(registry, MetricRegistry)
    result = registry.get(fake.id).compute(EvaluationContext(schema=schema), seed=11)
    assert result.metric_domain == domain
    assert result.status == "not_applicable" and result.value is None


@pytest.mark.parametrize("domain", ["statistical", "privacy"])
def test_fake_future_domain_plugin_registration(domain):
    registry = PluginRegistry()
    fake = FakeMetric(id=f"external_lab.future_{domain}", domain=domain)
    registry.register(fake)
    assert registry.get(fake.id) is fake


@pytest.mark.parametrize(
    "metric_id,domain",
    [
        ("marginal.ks_mean", "fidelity"),
        ("dependency.pairwise_tv_mean", "fidelity"),
        ("joint.c2st_auc", "fidelity"),
        ("proximity.dcr_train_median", "privacy"),
        ("validity.invalid_row_rate", "validity"),
        ("utility.tstr_auc", "utility"),
    ],
)
def test_stable_metric_ids_preserve_domain(metric_id, domain):
    registry = PluginRegistry([FakeMetric(id=metric_id, domain=domain)])
    assert registry.get(metric_id).id == metric_id
    with pytest.raises(InvalidPluginError):
        PluginRegistry([FakeMetric(id=metric_id, domain="diversity")])


def test_registry_duplicate_unknown_and_no_global_state():
    fake = FakeMetric()
    registry = PluginRegistry([fake])
    with pytest.raises(DuplicateMetricError):
        registry.register(fake)
    with pytest.raises(UnknownMetricError):
        PluginRegistry().get(fake.id)
    with pytest.raises(InvalidPluginError):
        registry.register(object())


@pytest.mark.parametrize(
    "patch",
    [
        {"domain": "unknown"},
        {"version": ""},
        {"required_inputs": frozenset({"attack_data"})},
        {"required_inputs": {"schema"}},
        {"required_capabilities": frozenset({"lab.future"})},
        {"direction": "arbitrary"},
    ],
)
def test_invalid_plugin_declarations_rejected(patch):
    with pytest.raises(InvalidPluginError):
        PluginRegistry([replace(FakeMetric(), **patch)])


def test_generator_contract_fake(schema):
    fake = FakeGenerator()
    assert isinstance(fake, GeneratorAdapter)
    train = toy_table()
    original = train.copy(deep=True)
    fake.fit(train, schema.generation_view(), seed=11)
    samples = fake.sample(7, seed=11)
    assert samples.shape == (7, 3)
    assert list(samples.columns) == [c.name for c in schema.columns]
    assert train.equals(original)
    assert fake.environment_info()["training"] == "none"
    assert not hasattr(fake.schema.columns[0], "maximum")


def test_fake_generator_contract(schema):
    # Keep the compatibility alias covered.
    test_generator_contract_fake(schema)
