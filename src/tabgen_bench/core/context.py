"""Defensive table snapshots and explicitly scoped, versioned extension access."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Generic, TypeVar

import pandas as pd
from pydantic import Field

from .base import FrozenModel
from .contracts import TABLE_INPUTS, OptionalInputPlugin
from .errors import MissingCapabilityError
from .schema import DatasetSchema, NonEmpty, Sha256, TargetSpec

if TYPE_CHECKING:
    from .contracts import MetricPlugin


class ExtensionRequirement(FrozenModel):
    namespace: str = Field(pattern=r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")
    version: NonEmpty


Payload = TypeVar("Payload", bound=FrozenModel)


@dataclass(frozen=True, init=False)
class ContextExtension(Generic[Payload]):
    requirement: ExtensionRequirement
    _payload_type: type[Payload]
    _payload_json: str

    def __init__(self, requirement: ExtensionRequirement, payload: Payload) -> None:
        if not isinstance(requirement, ExtensionRequirement):
            raise TypeError("extension requires a namespaced, versioned requirement")
        if not isinstance(payload, FrozenModel):
            raise TypeError("extension payload must use a typed frozen JSON contract")
        object.__setattr__(self, "requirement", requirement)
        object.__setattr__(self, "_payload_type", type(payload))
        object.__setattr__(self, "_payload_json", payload.model_dump_json())

    @property
    def payload(self) -> Payload:
        """Return a detached typed payload, including any nested collections."""
        return self._payload_type.model_validate_json(self._payload_json)

    def fingerprint(self) -> str:
        from .hashing import config_hash

        return config_hash(
            {
                "requirement": self.requirement.model_dump(mode="json"),
                "payload": self.payload.model_dump(mode="json"),
            }
        )


class InputIdentity(FrozenModel):
    input_name: str = Field(
        pattern=r"^(real_reference|real_train|real_validation|real_test|synthetic)$"
    )
    source_id: NonEmpty
    split_id: NonEmpty
    reference_split: str = Field(pattern=r"^(train|validation|test|reference|synthetic)$")
    content_hash: Sha256


def _copy_table(frame: pd.DataFrame) -> pd.DataFrame:
    # Preserve invalid data for validity diagnostics. Pandas deep=True alone shares
    # object cells; copy those recursively, using positions to retain duplicate columns.
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("table inputs must be pandas DataFrames")
    detached = frame.copy(deep=True)
    for column, dtype in enumerate(frame.dtypes):
        if pd.api.types.is_object_dtype(dtype):
            for row in range(len(frame)):
                detached.iat[row, column] = deepcopy(frame.iat[row, column])
    return detached


@dataclass(frozen=True, init=False)
class EvaluationContext:
    schema: DatasetSchema
    target: TargetSpec | None
    expected_n_rows: int | None
    identities: tuple[InputIdentity, ...]
    _tables: Mapping[str, pd.DataFrame]
    _extensions: tuple[ContextExtension, ...]

    def __init__(
        self,
        *,
        schema: DatasetSchema,
        synthetic: pd.DataFrame | None = None,
        real_reference: pd.DataFrame | None = None,
        real_train: pd.DataFrame | None = None,
        real_validation: pd.DataFrame | None = None,
        real_test: pd.DataFrame | None = None,
        target: TargetSpec | None = None,
        expected_n_rows: int | None = None,
        identities: tuple[InputIdentity, ...] = (),
        extensions: tuple[ContextExtension, ...] = (),
    ) -> None:
        tables = {
            "synthetic": synthetic,
            "real_reference": real_reference,
            "real_train": real_train,
            "real_validation": real_validation,
            "real_test": real_test,
        }
        tables = {name: _copy_table(frame) for name, frame in tables.items() if frame is not None}
        if expected_n_rows is not None and (
            type(expected_n_rows) is not int or expected_n_rows < 1
        ):
            raise ValueError("expected_n_rows must be positive")
        if len({i.input_name for i in identities}) != len(identities):
            raise ValueError("duplicate input identity")
        if {i.input_name for i in identities} != set(tables):
            raise ValueError("every supplied table must have exactly one explicit input identity")
        expected_splits = {
            "real_train": "train",
            "real_validation": "validation",
            "real_test": "test",
            "synthetic": "synthetic",
        }
        for identity in identities:
            if identity.input_name in expected_splits:
                if identity.reference_split != expected_splits[identity.input_name]:
                    raise ValueError("input identity split does not match its table")
        if len({e.requirement for e in extensions}) != len(extensions):
            raise ValueError("duplicate extension namespace/version")
        if target is not None and not any(
            c.name == target.column and c.role == "target" for c in schema.columns
        ):
            raise ValueError("target must match the schema's target column")
        for name, value in (
            ("schema", schema),
            ("target", target),
            ("expected_n_rows", expected_n_rows),
            ("identities", tuple(identities)),
            ("_tables", MappingProxyType(tables)),
            ("_extensions", tuple(extensions)),
        ):
            object.__setattr__(self, name, value)

    def table(self, name: str) -> pd.DataFrame | None:
        if name not in TABLE_INPUTS:
            raise KeyError(name)
        frame = self._tables.get(name)
        return None if frame is None else _copy_table(frame)

    @property
    def synthetic(self) -> pd.DataFrame | None:
        return self.table("synthetic")

    @property
    def real_reference(self) -> pd.DataFrame | None:
        return self.table("real_reference")

    @property
    def real_train(self) -> pd.DataFrame | None:
        return self.table("real_train")

    @property
    def real_validation(self) -> pd.DataFrame | None:
        return self.table("real_validation")

    @property
    def real_test(self) -> pd.DataFrame | None:
        return self.table("real_test")

    @property
    def available_inputs(self) -> frozenset[str]:
        return (
            frozenset(self._tables)
            | {"schema"}
            | ({"target"} if self.target is not None else set())
            | ({"expected_n_rows"} if self.expected_n_rows is not None else set())
        )

    def missing_inputs(self, plugin: "MetricPlugin") -> frozenset[str]:
        return plugin.required_inputs - self.available_inputs

    def missing_capabilities(self, plugin: "MetricPlugin") -> frozenset[ExtensionRequirement]:
        return plugin.required_capabilities - {e.requirement for e in self._extensions}

    def for_plugin(self, plugin: "MetricPlugin") -> "EvaluationContext":
        """Engine must pass this restricted view, never the unscoped context.

        Schema is public metadata. Undeclared tables, identities, target, row count,
        and extensions are absent. Missing inputs remain absent for applicability checks.
        """
        inputs = plugin.required_inputs | (
            plugin.optional_inputs if isinstance(plugin, OptionalInputPlugin) else frozenset()
        )
        return EvaluationContext(
            schema=self.schema,
            **{name: frame for name, frame in self._tables.items() if name in inputs},
            target=self.target if "target" in inputs else None,
            expected_n_rows=self.expected_n_rows if "expected_n_rows" in inputs else None,
            identities=tuple(i for i in self.identities if i.input_name in inputs),
            extensions=tuple(
                e for e in self._extensions if e.requirement in plugin.required_capabilities
            ),
        )

    def extension(self, requirement: ExtensionRequirement, payload_type: type[Payload]) -> Payload:
        for extension in self._extensions:
            if extension.requirement == requirement:
                if not isinstance(extension.payload, payload_type):
                    raise TypeError("extension payload type does not match requested contract")
                # Re-parse JSON to avoid passing through any caller-owned mutable payload fields.
                return payload_type.model_validate_json(extension.payload.model_dump_json())
        raise MissingCapabilityError(f"{requirement.namespace}@{requirement.version}")
