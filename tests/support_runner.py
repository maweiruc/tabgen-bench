"""Offline orchestration fakes only; never replace the real SDV acceptance gates."""

from dataclasses import dataclass, field

import pandas as pd

from tabgen_bench.core.config import FrozenConfig
from tabgen_bench.core.schema import (
    ColumnSchema,
    DatasetSchema,
    DatasetSource,
    DatasetSpec,
    TargetSpec,
)
from tabgen_bench.datasets.canonicalize import normalized_content_hash
from tabgen_bench.datasets.splits import split_dataset
from tabgen_bench.experiments.artifacts import ArtifactStore
from tabgen_bench.experiments.cache import Cache
from tabgen_bench.experiments.contracts import (
    DatasetBundle,
    EvaluationSpec,
    GeneratorSpec,
    MetricSpec,
)
from tabgen_bench.experiments.planner import Planner, make_cell, seed_spec
from tabgen_bench.experiments.runner import Runner
from tabgen_bench.generators.bootstrap import BootstrapGenerator
from tabgen_bench.metrics.base import Measurement, Metric
from tabgen_bench.metrics.registry import PluginRegistry
from tabgen_bench.metrics.validity.schema_checks import plugins


class Datasets:
    def __init__(self, *bundles):
        self.bundles = {b.spec.id: b for b in bundles}

    def get(self, name):
        return self.bundles[name]


def bundle(dataset_id="toy", *, maximum=1000, split_seed=20261025):
    schema = DatasetSchema(
        columns=(
            ColumnSchema(
                name="x",
                role="feature",
                logical_type="integer",
                nullable=False,
                minimum=0,
                maximum=maximum,
                constraint_source="explicit_toy_rule",
            ),
            ColumnSchema(name="group", role="feature", logical_type="categorical", nullable=False),
            ColumnSchema(name="label", role="target", logical_type="boolean", nullable=False),
        )
    )
    frame = pd.DataFrame(
        {"x": range(120), "group": ["A", "B", "C"] * 40, "label": [False, True] * 60}
    )
    spec = DatasetSpec(
        id=dataset_id,
        version="toy_v1",
        dataset_schema=schema,
        canonicalization_version="toy_v1",
        expected_rows=len(frame),
        source=DatasetSource(provider="test", source_id=dataset_id, version="1"),
        target=TargetSpec(column="label", positive_label=True, negative_label=False),
    )
    split = split_dataset(
        frame, "label", content_hash=normalized_content_hash(frame), seed=split_seed
    )
    return DatasetBundle(
        spec, frame, split, provenance=FrozenConfig.from_mapping({"toy_only": True})
    )


class CountingGenerator(BootstrapGenerator):
    def __init__(self, spec, counts):
        self.id, self.version, self.params, self.counts = spec.id, spec.version, spec.params, counts

    def fit(self, train, schema, *, seed):
        self.counts["fit"] += 1
        if self.counts.get("fail_fit"):
            raise RuntimeError("test-only fit failure")
        self.counts["train_rows"] = len(train)
        super().fit(train, schema, seed=seed)

    def fit_with_constraints(self, train, schema, rules, *, seed):
        self.counts["consumed_rules"] = rules.fingerprint()
        self.fit(train, schema, seed=seed)

    def sample(self, n_rows, *, seed):
        self.counts["sample"] += 1
        if self.counts.get("ignore_seed"):
            return self._train.iloc[:n_rows].reset_index(drop=True).copy(deep=True)
        result = super().sample(n_rows, seed=seed)
        result["x"] += self.params.to_dict().get("offset", 0)
        return result


@dataclass(frozen=True)
class FakeMetric(Metric):
    calls: dict = field(default_factory=dict)

    def calculate(self, context, *, seed):
        self.calls["metric"] = self.calls.get("metric", 0) + 1
        if self.calls.get("fail_metric"):
            raise RuntimeError("test-only future plugin failure")
        return Measurement(1.0, details={"test_only_fake": True})


def registry_and_suite(*, domain="statistical", version="1.0.0", params=None, calls=None):
    fake = FakeMetric(
        id="fake.future",
        domain=domain,
        direction="diagnostic",
        unit="test_only",
        required_inputs=frozenset({"synthetic"}),
        version=version,
        params=params or FrozenConfig(),
        calls=calls if calls is not None else {},
    )
    registry = PluginRegistry((*plugins(), fake))
    metrics = tuple(
        MetricSpec(id=p.id, version=p.version, domain=p.domain, params=p.params)
        for p in (*plugins(), fake)
    )
    return registry, EvaluationSpec(id="test_only_fake_suite", metrics=metrics)


def setup(tmp_path, *, data=None, generator=None, registry=None, suite=None, counts=None):
    data = data or bundle()
    counts = {"fit": 0, "sample": 0} if counts is None else counts
    if registry is None:
        registry, suite = registry_and_suite(calls=counts)
    generator = generator or GeneratorSpec(id="bootstrap")
    provider = Datasets(data)
    store, cache = ArtifactStore(tmp_path / "runs"), Cache(tmp_path / "cache")
    planner = Planner(dependency_lock_hash="1" * 64, runtime_fingerprint="2" * 64)
    cell = planner.expand([data], [generator], suite, replicate_seeds=(11,))[0]
    runner = Runner(
        datasets=provider,
        generators=lambda spec: CountingGenerator(spec, counts),
        metric_registry=registry,
        cache=cache,
        store=store,
    )
    return runner, cell, counts


def changed(
    cell,
    *,
    dataset=None,
    generator=None,
    evaluation=None,
    seeds=None,
    dependency_lock_hash=None,
    runtime_fingerprint=None,
    n_synthetic=None,
    report_template_version=None,
):
    return make_cell(
        dataset or cell.dataset,
        generator or cell.generator,
        evaluation or cell.evaluation,
        seeds or cell.seeds,
        dependency_lock_hash=dependency_lock_hash or cell.dependency_lock_hash,
        runtime_fingerprint=runtime_fingerprint or cell.runtime_fingerprint,
        n_synthetic=n_synthetic or cell.n_synthetic,
        report_template_version=report_template_version or cell.report_template_version,
    )


def for_seed(cell, seed):
    return changed(cell, seeds=seed_spec(seed, cell.seeds.split_seed))
