"""Immutable manifest snapshot contract."""

from datetime import datetime
from typing import Literal, Self

from pydantic import Field, model_validator

from .base import FrozenModel
from .config import FrozenConfig
from .schema import DatasetSource, NonEmpty, Sha256


class SeedSpec(FrozenModel):
    replicate_seed: int
    split_seed: int
    fit_seed: int
    sample_seed: int
    evaluation_seed: int
    reference_sampling_seed: int


class VersionInfo(FrozenModel):
    name: NonEmpty
    version: NonEmpty


class EnvironmentInfo(FrozenModel):
    python: NonEmpty
    operating_system: NonEmpty
    cpu: NonEmpty
    gpu: str | None = None
    actual_device: NonEmpty
    reproducibility_level: NonEmpty
    libraries: tuple[VersionInfo, ...]


class ArtifactInfo(FrozenModel):
    path: NonEmpty
    sha256: Sha256
    size_bytes: int = Field(ge=0)


class CacheRecord(FrozenModel):
    stage: NonEmpty
    key: Sha256
    hit: bool


class ContractHashes(FrozenModel):
    data: Sha256
    split: Sha256
    schema_hash: Sha256
    generation_schema_view: Sha256
    validity_rules: Sha256
    validity_rules_version: NonEmpty
    dependency_lock: Sha256


class RunManifest(FrozenModel):
    manifest_schema_version: NonEmpty = "1.0.0"
    run_id: NonEmpty
    status: Literal["pending", "running", "completed", "partial_failure", "failed"]
    created_at: datetime
    started_at: datetime | None = None
    ended_at: datetime | None = None
    command: tuple[str, ...]
    git_commit: str | None
    git_dirty: bool | None
    environment: EnvironmentInfo
    dataset_id: NonEmpty
    dataset_version: NonEmpty
    source: DatasetSource
    hashes: ContractHashes
    seeds: SeedSpec
    generator_id: str | None
    generator_version: str | None
    generator_params: FrozenConfig = Field(default_factory=FrozenConfig)
    suite_id: NonEmpty
    suite_version: NonEmpty
    metric_versions: tuple[VersionInfo, ...]
    resolved_config: FrozenConfig
    config_hash: Sha256
    n_synthetic: int | None = Field(default=None, ge=0)
    artifacts: tuple[ArtifactInfo, ...] = ()
    cache: tuple[CacheRecord, ...] = ()
    metric_success_fraction: float | None = Field(default=None, ge=0, le=1)
    error_type: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def consistent_snapshot(self) -> Self:
        if self.config_hash != self.resolved_config.fingerprint():
            raise ValueError("config_hash does not match resolved_config")
        for timestamp in (self.created_at, self.started_at, self.ended_at):
            if timestamp is not None and timestamp.utcoffset() is None:
                raise ValueError("manifest timestamps must include a timezone")
        if self.started_at is not None and self.started_at < self.created_at:
            raise ValueError("started_at predates created_at")
        if self.ended_at is not None and (
            self.started_at is None or self.ended_at < self.started_at
        ):
            raise ValueError("ended_at requires an earlier started_at")
        if self.status != "pending" and self.started_at is None:
            raise ValueError("non-pending manifest requires started_at")
        if self.status in {"completed", "partial_failure", "failed"} and self.ended_at is None:
            raise ValueError("terminal manifest requires ended_at")
        if self.status in {"pending", "running"} and self.ended_at is not None:
            raise ValueError("active manifest cannot have ended_at")
        if self.status == "failed" and not (self.error_type and self.error_message):
            raise ValueError("failed manifest requires failure details")
        if (self.generator_id is None) != (self.generator_version is None):
            raise ValueError("generator ID and version must both be present or absent")
        return self
