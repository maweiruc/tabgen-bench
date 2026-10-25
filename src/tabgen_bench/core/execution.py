"""Data-only scheduling contracts; gate evaluation does not know any algorithm."""

from typing import Self

from pydantic import Field, model_validator

from .base import FrozenModel
from .contracts import CONTEXT_INPUTS
from .schema import NonEmpty

VALIDITY_PREFLIGHT = (
    "validity.schema_pass",
    "validity.invalid_cell_rate",
    "validity.invalid_row_rate",
)


class ResultGate(FrozenModel):
    metric_id: NonEmpty
    required_input: NonEmpty
    passed_values: tuple[float, ...] = Field(default=(1.0,), min_length=1)
    exempt_metric_ids: tuple[str, ...] = ()
    reason: NonEmpty

    @model_validator(mode="after")
    def declared_input(self) -> Self:
        if self.required_input not in CONTEXT_INPUTS:
            raise ValueError("gate must name a declared context input")
        return self


class EvaluationPolicy(FrozenModel):
    policy_version: NonEmpty = "1.0.0"
    preflight_metric_ids: tuple[str, ...] = VALIDITY_PREFLIGHT
    gates: tuple[ResultGate, ...] = Field(
        default_factory=lambda: (
            ResultGate(
                metric_id="validity.schema_pass",
                required_input="synthetic",
                exempt_metric_ids=VALIDITY_PREFLIGHT,
                reason="blocked_by_validity",
            ),
        )
    )

    @model_validator(mode="after")
    def valid_preflight(self) -> Self:
        if len(self.preflight_metric_ids) != len(set(self.preflight_metric_ids)):
            raise ValueError("preflight metric IDs must be unique")
        if any(gate.metric_id not in self.preflight_metric_ids for gate in self.gates):
            raise ValueError("gate prerequisite must be scheduled as a preflight")
        return self
