"""Declared schema contracts, never inferred from empirical train/test ranges."""

import math
from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr, model_validator

from .base import FrozenModel

NonEmpty = Annotated[str, Field(min_length=1)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
LogicalType = Literal["numeric", "integer", "categorical", "boolean"]
ColumnRole = Literal["feature", "target"]
Scalar = StrictStr | StrictBool | StrictInt | StrictFloat


class GenerationColumn(FrozenModel):
    name: NonEmpty
    role: ColumnRole
    logical_type: LogicalType
    nullable: StrictBool
    missing_representation: Literal["pandas_na"] = "pandas_na"


class ColumnSchema(GenerationColumn):
    allowed_values: tuple[Scalar, ...] | None = None
    minimum: StrictInt | StrictFloat | None = None
    maximum: StrictInt | StrictFloat | None = None
    constraint_source: NonEmpty = "explicit_schema"
    rule_version: NonEmpty = "1.0.0"

    @model_validator(mode="after")
    def check_declared_rules(self) -> Self:
        bounds = (self.minimum, self.maximum)
        if any(v is not None for v in bounds) and self.logical_type not in {"numeric", "integer"}:
            raise ValueError("bounds require a numeric or integer logical type")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("minimum must not exceed maximum")
        if self.allowed_values is not None:
            if not self.allowed_values:
                raise ValueError("allowed_values must be nonempty when declared")
            from .hashing import canonical_json

            if len({canonical_json(v) for v in self.allowed_values}) != len(self.allowed_values):
                raise ValueError("allowed_values contains duplicates")
            for value in self.allowed_values:
                if self.logical_type == "boolean" and type(value) is not bool:
                    raise ValueError("boolean allowed_values must be booleans")
                if self.logical_type in {"numeric", "integer"}:
                    if type(value) not in {int, float}:
                        raise ValueError("numeric allowed_values must be numeric")
                    if self.logical_type == "integer" and not float(value).is_integer():
                        raise ValueError("integer allowed_values must be integral")
                    if self.minimum is not None and value < self.minimum:
                        raise ValueError("allowed value below minimum")
                    if self.maximum is not None and value > self.maximum:
                        raise ValueError("allowed value above maximum")
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError("allowed_values must be finite")
        return self


class GenerationSchemaView(FrozenModel):
    view_version: NonEmpty = "1.0.0"
    columns: tuple[GenerationColumn, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def check_columns(self) -> Self:
        if any(type(column) is not GenerationColumn for column in self.columns):
            raise ValueError("generation view must use rule-free GenerationColumn objects")
        _validate_columns(self.columns)
        return self


class ValidityColumnRule(FrozenModel):
    name: NonEmpty
    logical_type: LogicalType
    nullable: StrictBool
    allowed_values: tuple[Scalar, ...] | None
    minimum: StrictInt | StrictFloat | None
    maximum: StrictInt | StrictFloat | None
    constraint_source: NonEmpty
    rule_version: NonEmpty

    @model_validator(mode="after")
    def check_declared_rules(self) -> Self:
        ColumnSchema(role="feature", **self.model_dump())
        return self


class ValidityRuleSet(FrozenModel):
    ruleset_version: NonEmpty
    columns: tuple[ValidityColumnRule, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_columns(self) -> Self:
        if len({c.name for c in self.columns}) != len(self.columns):
            raise ValueError("validity rule columns must be unique")
        return self


class DatasetSchema(FrozenModel):
    schema_version: NonEmpty = "1.0.0"
    constraint_rules_version: NonEmpty = "1.0.0"
    columns: tuple[ColumnSchema, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def check_columns(self) -> Self:
        _validate_columns(self.columns)
        return self

    def generation_view(self) -> GenerationSchemaView:
        # Explicit field selection prevents future evaluation metadata from leaking.
        return GenerationSchemaView(
            columns=tuple(
                GenerationColumn(
                    name=c.name,
                    role=c.role,
                    logical_type=c.logical_type,
                    nullable=c.nullable,
                    missing_representation=c.missing_representation,
                )
                for c in self.columns
            )
        )

    def validity_rules(self) -> ValidityRuleSet:
        return ValidityRuleSet(
            ruleset_version=self.constraint_rules_version,
            columns=tuple(
                ValidityColumnRule(
                    name=c.name,
                    logical_type=c.logical_type,
                    nullable=c.nullable,
                    allowed_values=c.allowed_values,
                    minimum=c.minimum,
                    maximum=c.maximum,
                    constraint_source=c.constraint_source,
                    rule_version=c.rule_version,
                )
                for c in self.columns
            ),
        )


def _validate_columns(columns: tuple[GenerationColumn, ...]) -> None:
    if len({c.name for c in columns}) != len(columns):
        raise ValueError("schema column names must be unique")
    if sum(c.role == "target" for c in columns) > 1:
        raise ValueError("V1 supports at most one target")


class DatasetSource(FrozenModel):
    provider: NonEmpty
    source_id: NonEmpty
    version: NonEmpty
    upstream_url: str | None = None
    citation: str | None = None
    license_note: str | None = None


class TargetSpec(FrozenModel):
    column: NonEmpty
    task: Literal["binary_classification"] = "binary_classification"
    positive_label: Scalar
    negative_label: Scalar

    @model_validator(mode="after")
    def distinct_labels(self) -> Self:
        if self.positive_label == self.negative_label:
            raise ValueError("target labels must differ")
        return self


class DatasetSpec(FrozenModel):
    id: NonEmpty
    version: NonEmpty
    source: DatasetSource
    dataset_schema: DatasetSchema
    canonicalization_version: NonEmpty
    expected_rows: Annotated[int, Field(gt=0)] | None = None
    target: TargetSpec | None = None

    @model_validator(mode="after")
    def check_target(self) -> Self:
        if self.target is not None:
            names = {c.name for c in self.dataset_schema.columns if c.role == "target"}
            if names != {self.target.column}:
                raise ValueError("target must identify the schema's declared target column")
        return self
