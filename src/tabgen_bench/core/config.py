"""Frozen JSON object for plugin parameters and resolved configuration."""

import json
from collections.abc import Mapping
from typing import Any

from pydantic import field_validator

from .base import FrozenModel
from .hashing import canonical_json, config_hash


class FrozenConfig(FrozenModel):
    canonical_json: str = "{}"

    @field_validator("canonical_json")
    @classmethod
    def validate_document(cls, value: str) -> str:
        data = json.loads(value)
        if not isinstance(data, dict):
            raise ValueError("configuration must be a JSON object")
        return canonical_json(data)

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, runtime_keys: frozenset[str] = frozenset()
    ) -> "FrozenConfig":
        return cls(
            canonical_json=canonical_json({k: v for k, v in value.items() if k not in runtime_keys})
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached copy; nested mutation cannot change the contract."""
        return json.loads(self.canonical_json)

    def fingerprint(self) -> str:
        return config_hash(self.to_dict())
