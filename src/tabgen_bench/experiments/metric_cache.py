"""Registry decorator: cache valid plugin outputs while retaining Engine gates and isolation."""

from tabgen_bench.core.config import FrozenConfig
from tabgen_bench.core.results import MetricResult
from tabgen_bench.evaluation.engine import EvaluationEngine

from .artifacts import json_bytes
from .manifest import utc_now


class CachedPlugin:
    def __init__(self, plugin, spec, node, cache, on_result, run_id):
        self.plugin, self.spec, self.node = plugin, spec, node
        self.cache, self.on_result = cache, on_result
        self.entry = cache.get("metric", node.key)
        self.called = False
        self.run_id = run_id

    def __getattr__(self, name):
        return getattr(self.plugin, name)

    def compute(self, context, *, seed):
        started = utc_now().isoformat()
        self.called = True
        hit = self.entry is not None
        if hit:
            result = MetricResult.model_validate_json(self.entry.read("result.json"))
        else:
            result = self.plugin.compute(context, seed=seed)
        if not isinstance(result, MetricResult) or (
            result.metric_id,
            result.metric_version,
            result.metric_domain,
            result.direction,
        ) != (self.plugin.id, self.plugin.version, self.plugin.domain, self.plugin.direction):
            raise ValueError("plugin result identity/version/domain/direction mismatch")
        result = MetricResult.model_validate(result.model_dump())
        if result.status != "failed":
            if not hit:
                self.entry = self.cache.publish(
                    "metric",
                    self.node.key,
                    {"result.json": json_bytes(result.model_dump(mode="json"))},
                    metadata={
                        "metric_id": self.id,
                        "metric_version": self.version,
                        "origin_run_id": self.run_id,
                        "computed_started_at": started,
                        "computed_ended_at": utc_now().isoformat(),
                    },
                )
            self.on_result(
                "metric:" + self.id,
                self.node.key,
                hit,
                entry=self.entry,
                metadata={"started_at": started},
            )
        return result


class CachedRegistry:
    def __init__(self, registry, cell, cache, on_result, run_id):
        self.plugins = {}
        nodes = {n.id: n for n in cell.nodes}
        for spec in cell.evaluation.metrics:
            plugin = registry.get(spec.id)
            params = getattr(plugin, "params", FrozenConfig())
            if (
                (plugin.id, plugin.version, plugin.domain) != (spec.id, spec.version, spec.domain)
                or not isinstance(params, FrozenConfig)
                or params != spec.params
                or getattr(plugin, "preprocessing_version", "1.0.0") != spec.preprocessing_version
            ):
                raise ValueError(f"frozen MetricSpec differs from injected plugin: {spec.id}")
            self.plugins[spec.id] = CachedPlugin(
                plugin, spec, nodes["metric:" + spec.id], cache, on_result, run_id
            )

    def get(self, metric_id):
        return self.plugins[metric_id]


def evaluate_cached(
    registry, context, cell, cache, on_result, run_id, engine_factory=EvaluationEngine
):
    cached = CachedRegistry(registry, cell, cache, on_result, run_id)
    engine = engine_factory(
        cached,
        policy=cell.evaluation.policy,
        run_id=run_id,
        dataset_id=cell.dataset.spec.id,
        generator_id=cell.generator.id,
        replicate_seed=cell.seeds.replicate_seed,
    )
    results = engine.evaluate(
        context,
        metric_ids=tuple(m.id for m in cell.evaluation.metrics),
        seed=cell.seeds.evaluation_seed,
    )
    for result in results:
        proxy = cached.get(result.metric_id)
        if not proxy.called or result.status == "failed":
            on_result(
                "metric:" + result.metric_id,
                proxy.node.key,
                False,
                metadata={"status": result.status, "reason": "engine_gate_or_failure_not_cached"},
            )
    return results
