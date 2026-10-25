"""Canonical JSON v1: deterministic keys/numbers, strict JSON, SHA256.

Paths and arbitrary objects are rejected. Resolve semantic file inputs to content
hashes, and explicitly remove runtime-only paths before hashing a configuration.
Array order and JSON scalar types remain meaningful; -0.0 is normalized to 0.0.
"""

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

CANONICAL_JSON_VERSION = "1.0.0"


def _normalize(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite numbers cannot be hashed")
        return 0.0 if value == 0 else value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("configuration keys must be strings")
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    raise TypeError(f"unsupported canonical JSON type: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        _normalize(value),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def config_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
