from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tabgen_bench.core.config import FrozenConfig
from tabgen_bench.core.hashing import config_hash
from tabgen_bench.core.manifest import (
    ContractHashes,
    EnvironmentInfo,
    RunManifest,
    SeedSpec,
    VersionInfo,
)
from tabgen_bench.core.schema import DatasetSource


def manifest_data(schema):
    h = config_hash({"test-only": True})
    config = FrozenConfig.from_mapping({"seed": 11})
    return dict(
        run_id="test-snapshot",
        status="pending",
        created_at=datetime(2026, 10, 25, tzinfo=UTC),
        command=("test-only",),
        git_commit=None,
        git_dirty=None,
        environment=EnvironmentInfo(
            python="3.11",
            operating_system="test-only",
            cpu="test",
            actual_device="cpu",
            reproducibility_level="test-only",
            libraries=(VersionInfo(name="test", version="1"),),
        ),
        dataset_id="toy",
        dataset_version="1",
        source=DatasetSource(provider="local", source_id="toy.csv", version="1"),
        hashes=ContractHashes(
            data=h,
            split=h,
            schema_hash=schema.fingerprint(),
            generation_schema_view=schema.generation_view().fingerprint(),
            validity_rules=schema.validity_rules().fingerprint(),
            validity_rules_version=schema.constraint_rules_version,
            dependency_lock=h,
        ),
        seeds=SeedSpec(
            replicate_seed=11,
            split_seed=20261025,
            fit_seed=11,
            sample_seed=23,
            evaluation_seed=37,
            reference_sampling_seed=53,
        ),
        generator_id=None,
        generator_version=None,
        suite_id="test-only",
        suite_version="1",
        metric_versions=(),
        resolved_config=config,
        config_hash=config.fingerprint(),
    )


def test_manifest_immutable_json_snapshot(schema):
    manifest = RunManifest(**manifest_data(schema))
    assert RunManifest.model_validate_json(manifest.model_dump_json()) == manifest
    with pytest.raises(ValidationError):
        manifest.status = "completed"
    with pytest.raises(ValidationError):
        manifest.hashes.validity_rules_version = "changed"


@pytest.mark.parametrize(
    "patch",
    [
        {"config_hash": "0" * 64},
        {"created_at": datetime(2026, 10, 25)},
        {"status": "completed"},
        {"status": "running"},
        {"generator_id": "fake"},
    ],
)
def test_manifest_rejects_inconsistent_snapshot(schema, patch):
    with pytest.raises(ValidationError):
        RunManifest(**{**manifest_data(schema), **patch})
