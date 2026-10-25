"""Independent, registry-injected execution with generic preflight/result gates."""

from datetime import UTC, datetime
from uuid import uuid4

from tabgen_bench.core.config import FrozenConfig
from tabgen_bench.core.context import EvaluationContext
from tabgen_bench.core.contracts import MetricPlugin, MetricRegistry
from tabgen_bench.core.execution import EvaluationPolicy
from tabgen_bench.core.hashing import config_hash
from tabgen_bench.core.results import MetricResult


class EvaluationEngine:
    def __init__(
        self,
        metric_registry: MetricRegistry,
        *,
        policy: EvaluationPolicy | None = None,
        run_id: str | None = None,
        dataset_id: str = "external",
        generator_id: str | None = None,
        replicate_seed: int | None = None,
    ):
        self._registry = metric_registry
        self._policy = policy if policy is not None else EvaluationPolicy()
        self._identity = {
            "run_id": run_id or str(uuid4()),
            "dataset_id": dataset_id,
            "generator_id": generator_id,
            "replicate_seed": replicate_seed,
        }

    def _unavailable(self, plugin: MetricPlugin, reason: str, *, failure: Exception | None = None):
        params = getattr(plugin, "params", None)
        parameters = params.to_dict() if isinstance(params, FrozenConfig) else {}
        return MetricResult(
            **self._identity,
            metric_id=plugin.id,
            metric_domain=plugin.domain,
            metric_version=plugin.version,
            value=None,
            direction=plugin.direction,
            unit=getattr(plugin, "unit", "unspecified"),
            reference_split="none",
            status="failed" if failure is not None else "not_applicable",
            notes=reason,
            error_type=type(failure).__name__ if failure is not None else None,
            error_message=(str(failure) or repr(failure)) if failure is not None else None,
            method_config_hash=config_hash(
                {
                    "id": plugin.id,
                    "version": plugin.version,
                    "domain": plugin.domain,
                    "params": parameters,
                    "preprocessing_version": "1.0.0",
                }
            ),
            details={
                "reason": reason,
                "stage": "evaluation",
                "metric_id": plugin.id,
                "parameters": parameters,
            },
        )

    def evaluate(
        self, context: EvaluationContext, *, metric_ids: tuple[str, ...], seed: int
    ) -> tuple[MetricResult, ...]:
        if len(metric_ids) != len(set(metric_ids)):
            raise ValueError("duplicate requested metric IDs")
        ids = tuple(dict.fromkeys((*self._policy.preflight_metric_ids, *metric_ids)))
        # Unknown IDs are configuration errors: no invented plugin domain/version.
        plugins = {metric_id: self._registry.get(metric_id) for metric_id in ids}
        results = {}
        for metric_id, plugin in plugins.items():
            started = datetime.now(UTC)
            scoped = None
            try:
                scoped = context.for_plugin(plugin)
                missing = context.missing_inputs(plugin)
                capabilities = context.missing_capabilities(plugin)
                blocked = None
                for gate in self._policy.gates:
                    if (
                        gate.required_input in scoped.available_inputs
                        and metric_id not in gate.exempt_metric_ids
                    ):
                        prerequisite = results.get(gate.metric_id)
                        if (
                            prerequisite is None
                            or prerequisite.status not in {"ok", "warning"}
                            or prerequisite.value not in gate.passed_values
                        ):
                            blocked = gate.reason
                            break
                if missing or capabilities:
                    reason = (
                        "missing_input: " + ",".join(sorted(missing))
                        if missing
                        else (
                            "missing_capability: "
                            + ",".join(sorted(f"{c.namespace}@{c.version}" for c in capabilities))
                        )
                    )
                    result = self._unavailable(plugin, reason)
                elif blocked:
                    result = self._unavailable(plugin, blocked)
                else:
                    result = plugin.compute(scoped, seed=seed)
                    if not isinstance(result, MetricResult):
                        raise TypeError("plugin must return MetricResult")
                    if (
                        result.metric_id,
                        result.metric_domain,
                        result.metric_version,
                        result.direction,
                    ) != (plugin.id, plugin.domain, plugin.version, plugin.direction):
                        raise ValueError("plugin result identity/version/domain/direction mismatch")
            except Exception as exc:
                result = self._unavailable(plugin, "plugin_exception", failure=exc)
            details = result.details.to_dict()
            details["evaluation"] = {
                "seed": seed,
                "policy_hash": self._policy.fingerprint(),
                "generation_schema_hash": context.schema.generation_view().fingerprint(),
                "validity_rules_hash": context.schema.validity_rules().fingerprint(),
                "available_input_identities": [
                    i.model_dump(mode="json") for i in (() if scoped is None else scoped.identities)
                ],
            }
            result = MetricResult.model_validate(
                {
                    **result.model_dump(),
                    **self._identity,
                    "details": details,
                    "started_at": started,
                    "ended_at": datetime.now(UTC),
                }
            )
            results[metric_id] = result
        return tuple(results.values())
