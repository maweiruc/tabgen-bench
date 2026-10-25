"""Auditable metric records, independent of algorithms and persistence."""

import math
from datetime import datetime
from typing import Any, Literal, Self

from pydantic import (
    Field,
    StrictFloat,
    StrictInt,
    field_serializer,
    field_validator,
    model_validator,
)

from .base import FrozenModel
from .config import FrozenConfig
from .context import InputIdentity
from .contracts import MetricDirection, MetricDomain, declared_domain_matches
from .schema import NonEmpty, Sha256


class MetricResult(FrozenModel):
    result_schema_version: NonEmpty = "1.1.0"
    run_id: NonEmpty
    dataset_id: NonEmpty
    generator_id: str | None = None
    replicate_seed: int | None = None
    metric_id: NonEmpty
    metric_domain: MetricDomain
    metric_version: NonEmpty
    value: StrictInt | StrictFloat | None
    direction: MetricDirection
    unit: NonEmpty
    status: Literal["ok", "not_applicable", "warning", "failed"]
    reference_split: Literal["train", "validation", "test", "reference", "multiple", "none"]
    references: tuple[InputIdentity, ...] = ()
    n_real: int | None = Field(default=None, ge=0)
    n_synthetic: int | None = Field(default=None, ge=0)
    method_config_hash: Sha256
    notes: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    details: FrozenConfig = Field(default_factory=FrozenConfig)
    started_at: datetime | None = None
    ended_at: datetime | None = None

    @field_validator("details", mode="before")
    @classmethod
    def parse_details(cls, value):
        return FrozenConfig.from_mapping(value) if isinstance(value, dict) else value

    @field_serializer("details")
    def serialize_details(self, value: FrozenConfig):
        return value.to_dict()

    @model_validator(mode="before")
    @classmethod
    def normalize_nonfinite(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("value"), float):
            if not math.isfinite(data["value"]):
                return {
                    **data,
                    "value": None,
                    "status": "failed",
                    "error_type": "non_finite_value",
                    "error_message": "Metric produced NaN or infinity; serialized as null",
                }
        return data

    @model_validator(mode="after")
    def check_result(self) -> Self:
        for timestamp in (self.started_at, self.ended_at):
            if timestamp is not None and timestamp.utcoffset() is None:
                raise ValueError("metric timestamps must include a timezone")
        if self.ended_at is not None and (
            self.started_at is None or self.ended_at < self.started_at
        ):
            raise ValueError("ended_at requires an earlier started_at")
        if not declared_domain_matches(self.metric_id, self.metric_domain):
            raise ValueError("metric domain conflicts with stable metric ID")
        if self.status == "ok" and self.value is None:
            raise ValueError("ok result requires a numeric value")
        if self.status in {"not_applicable", "failed"} and self.value is not None:
            raise ValueError("unavailable/failed results cannot contain a fabricated score")
        if self.status == "failed" and not (self.error_type and self.error_message):
            raise ValueError("failed result requires error_type and error_message")
        if self.status in {"not_applicable", "warning"} and not self.notes:
            raise ValueError("not_applicable/warning requires explanatory notes")
        if any(i.input_name == "synthetic" for i in self.references):
            raise ValueError("references must identify real inputs")
        if self.references:
            splits = {i.reference_split for i in self.references}
            expected = next(iter(splits)) if len(splits) == 1 else "multiple"
            if self.reference_split != expected:
                raise ValueError("reference_split must match explicit references")
        if self.reference_split == "none" and (self.n_real not in {None, 0}):
            raise ValueError("no real reference cannot have positive n_real")
        return self
