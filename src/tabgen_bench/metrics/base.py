"""Shared result construction; no registry state or algorithm dispatch."""

from dataclasses import dataclass, field
from typing import Any

from tabgen_bench.core.config import FrozenConfig
from tabgen_bench.core.context import EvaluationContext, ExtensionRequirement
from tabgen_bench.core.contracts import MetricDirection, MetricDomain
from tabgen_bench.core.hashing import config_hash
from tabgen_bench.core.results import MetricResult


class NotApplicable(Exception):
    def __init__(self, reason: str, references: tuple[str, ...] = ()):
        super().__init__(reason)
        self.references = references


@dataclass
class Measurement:
    value: float | int | None
    references: tuple[str, ...] = ()
    n_real: int | None = None
    details: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    notes: str | None = None


@dataclass(frozen=True)
class Metric:
    id: str
    domain: MetricDomain
    direction: MetricDirection
    unit: str
    required_inputs: frozenset[str]
    optional_inputs: frozenset[str] = frozenset()
    required_capabilities: frozenset[ExtensionRequirement] = frozenset()
    version: str = "1.0.0"
    params: FrozenConfig = field(default_factory=FrozenConfig)

    def calculate(self, context: EvaluationContext, *, seed: int) -> Measurement:
        raise NotImplementedError

    def compute(self, context: EvaluationContext, *, seed: int) -> MetricResult:
        try:
            measurement = self.calculate(context, seed=seed)
        except NotApplicable as exc:
            measurement = Measurement(
                None,
                references=exc.references,
                status="not_applicable",
                notes=str(exc),
                details={"reason": str(exc)},
            )
        identities = tuple(i for i in context.identities if i.input_name in measurement.references)
        splits = {i.reference_split for i in identities}
        split = next(iter(splits)) if len(splits) == 1 else "multiple" if splits else "none"
        synthetic = context.synthetic
        details = {"parameters": self.params.to_dict(), "seed": seed, **measurement.details}
        return MetricResult(
            run_id="unbound",
            dataset_id="unbound",
            metric_id=self.id,
            metric_domain=self.domain,
            metric_version=self.version,
            value=measurement.value,
            direction=self.direction,
            unit=self.unit,
            status=measurement.status,
            notes=measurement.notes,
            reference_split=split,
            references=identities,
            n_real=measurement.n_real,
            n_synthetic=None if synthetic is None else len(synthetic),
            details=details,
            method_config_hash=config_hash(
                {
                    "id": self.id,
                    "version": self.version,
                    "domain": self.domain,
                    "params": self.params.to_dict(),
                    "preprocessing_version": "1.0.0",
                }
            ),
        )
