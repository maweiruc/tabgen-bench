from pathlib import Path

import pytest

from tabgen_bench.core.config import FrozenConfig
from tabgen_bench.core.hashing import canonical_json, config_hash


def test_hash_stable_key_order():
    assert config_hash({"b": {"y": 2, "x": 1}, "a": [1, 2]}) == (
        config_hash({"a": [1, 2], "b": {"x": 1, "y": 2}})
    )


@pytest.mark.parametrize("changed", [{"seed": 2}, {"seed": "1"}, {"seed": True}])
def test_hash_changes_on_semantic_change(changed):
    assert config_hash({"seed": 1}) != config_hash(changed)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), Path("/tmp/machine"), {1: "x"}])
def test_hash_rejects_ambiguous_or_nonjson_input(value):
    with pytest.raises((ValueError, TypeError)):
        config_hash(value)


def test_hash_preserves_array_order_and_unicode():
    assert config_hash([1, 2]) != config_hash([2, 1])
    assert canonical_json({"中文": -0.0}) == '{"中文":0.0}'


def test_config_immutable_and_runtime_path_exclusion():
    original = {"params": {"seed": 11}, "output_dir": "/tmp/a"}
    config = FrozenConfig.from_mapping(original, runtime_keys=frozenset({"output_dir"}))
    original["params"]["seed"] = 99
    detached = config.to_dict()
    detached["params"]["seed"] = 100
    assert config.to_dict() == {"params": {"seed": 11}}
    assert config.fingerprint() == config_hash({"params": {"seed": 11}})
