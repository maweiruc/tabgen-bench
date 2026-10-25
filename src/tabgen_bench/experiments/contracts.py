"""Frozen orchestration inputs and structural ports."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

import pandas as pd
from pydantic import Field, model_validator

from tabgen_bench.core.base import FrozenModel
from tabgen_bench.core.config import FrozenConfig
from tabgen_bench.core.context import ContextExtension
from tabgen_bench.core.contracts import GeneratorAdapter, MetricDomain
from tabgen_bench.core.execution import EvaluationPolicy
from tabgen_bench.core.manifest import SeedSpec
from tabgen_bench.core.schema import DatasetSpec, GenerationSchemaView, NonEmpty, ValidityRuleSet
from tabgen_bench.datasets.canonicalize import normalized_content_hash, parquet_bytes
from tabgen_bench.datasets.openml import sha256
from tabgen_bench.datasets.splits import DatasetSplit


class GeneratorSpec(FrozenModel):
    id: NonEmpty
    version: NonEmpty = "1.0.0"
    adapter_version: NonEmpty = "1.0.0"
    params: FrozenConfig = Field(default_factory=FrozenConfig)
    consumes_validity_rules: bool = False


class MetricSpec(FrozenModel):
    id: NonEmpty
    version: NonEmpty
    domain: MetricDomain
    params: FrozenConfig = Field(default_factory=FrozenConfig)
    preprocessing_version: NonEmpty = "1.0.0"


class EvaluationSpec(FrozenModel):
    id: NonEmpty
    version: NonEmpty = "1.0.0"
    metrics: tuple[MetricSpec, ...] = Field(min_length=1)
    policy: EvaluationPolicy = Field(default_factory=EvaluationPolicy)

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [m.id for m in self.metrics]
        if len(ids) != len(set(ids)) or not set(self.policy.preflight_metric_ids) <= set(ids):
            raise ValueError("suite requires unique metrics including all policy prerequisites")
        return self


class DatasetDescriptor(FrozenModel):
    spec: DatasetSpec
    canonical_bytes_sha256: str
    content_hash: str
    split_hash: str
    split_seed: int
    splitter_version: str
    split_counts: FrozenConfig
    references: FrozenConfig
    provenance: FrozenConfig = Field(default_factory=FrozenConfig)
    extensions_hash: str


@dataclass(frozen=True, init=False)
class DatasetBundle:
    spec: DatasetSpec
    split: DatasetSplit
    provenance: FrozenConfig
    extensions: tuple[ContextExtension, ...]
    _frame: pd.DataFrame

    def __init__(self, spec, frame, split, *, provenance=None, extensions=()):
        for name, value in (
            ("spec", spec),
            ("split", split),
            ("provenance", provenance or FrozenConfig()),
            ("extensions", tuple(extensions)),
            ("_frame", frame.copy(deep=True)),
        ):
            object.__setattr__(self, name, value)
        split.frames(frame)  # Check content/order and row positions before planning or training.

    @property
    def frame(self):
        return self._frame.copy(deep=True)

    def describe(self):
        from tabgen_bench.core.hashing import config_hash

        frames = self.split.frames(self._frame)
        return DatasetDescriptor(
            spec=self.spec,
            canonical_bytes_sha256=sha256(parquet_bytes(self._frame)),
            content_hash=normalized_content_hash(self._frame),
            split_hash=config_hash(self.split.to_dict()),
            split_seed=self.split.seed,
            splitter_version=self.split.to_dict()["split_version"],
            split_counts=FrozenConfig.from_mapping(
                {k: len(v) for k, v in zip(("train", "validation", "test"), frames, strict=True)}
            ),
            references=FrozenConfig.from_mapping(
                {
                    k: normalized_content_hash(v)
                    for k, v in zip(
                        ("real_train", "real_validation", "real_test"), frames, strict=True
                    )
                }
            ),
            provenance=self.provenance,
            extensions_hash=config_hash([e.fingerprint() for e in self.extensions]),
        )


class DatasetProvider(Protocol):
    def get(self, dataset_id: str) -> DatasetBundle: ...


class ConstraintConsumer(Protocol):
    def fit_with_constraints(
        self,
        train: pd.DataFrame,
        schema: GenerationSchemaView,
        rules: ValidityRuleSet,
        *,
        seed: int,
    ) -> None: ...


GeneratorFactory = Callable[[GeneratorSpec], GeneratorAdapter]
Stage = Literal["data", "split", "fit", "sample", "metric", "report_input"]


class StageNode(FrozenModel):
    id: str
    stage: Stage
    key: str | None
    depends_on: tuple[str, ...] = ()


class CellPlan(FrozenModel):
    plan_version: str = "1.0.0"
    dataset: DatasetDescriptor
    generator: GeneratorSpec
    evaluation: EvaluationSpec
    seeds: SeedSpec
    dependency_lock_hash: str
    n_synthetic: int = Field(gt=0)
    sampling_config: FrozenConfig = Field(default_factory=FrozenConfig)
    report_template_version: NonEmpty = "report_input_v1"
    runtime_fingerprint: str
    nodes: tuple[StageNode, ...]

    @model_validator(mode="after")
    def acyclic_nodes(self):
        seen = set()
        for node in self.nodes:
            if node.id in seen or not set(node.depends_on) <= seen:
                raise ValueError("DAG must be unique and topologically ordered")
            seen.add(node.id)
        return self
