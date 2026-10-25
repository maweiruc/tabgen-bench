"""Test-only plugins: no real algorithm, benchmark score, or model training."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tabgen_bench.core.context import EvaluationContext, ExtensionRequirement
from tabgen_bench.core.contracts import MetricDirection, MetricDomain
from tabgen_bench.core.hashing import config_hash
from tabgen_bench.core.results import MetricResult
from tabgen_bench.core.schema import GenerationSchemaView


def toy_table():
    return pd.read_csv(Path(__file__).parent / "fixtures" / "toy.csv")


class FakeGenerator:
    id = "test.fixed_rows"
    version = "test-only-1"

    def fit(self, train: pd.DataFrame, schema: GenerationSchemaView, *, seed: int) -> None:
        assert isinstance(schema, GenerationSchemaView)
        self.schema = schema
        self.columns = tuple(train.columns)
        assert self.columns == tuple(c.name for c in schema.columns)
        self.row = train.iloc[[0]].copy(deep=True)

    def sample(self, n_rows: int, *, seed: int) -> pd.DataFrame:
        if n_rows < 1:
            raise ValueError("n_rows must be positive")
        return pd.concat([self.row] * n_rows, ignore_index=True)

    def environment_info(self):
        return {"implementation": "fake", "training": "none"}


@dataclass(frozen=True)
class FakeMetric:
    id: str = "test.contract"
    version: str = "test-only-1"
    domain: MetricDomain = "fidelity"
    direction: MetricDirection = "diagnostic"
    required_inputs: frozenset[str] = frozenset({"schema", "synthetic"})
    required_capabilities: frozenset[ExtensionRequirement] = frozenset()

    def compute(self, context: EvaluationContext, *, seed: int) -> MetricResult:
        return MetricResult(
            run_id="test-only",
            dataset_id="toy",
            metric_id=self.id,
            metric_domain=self.domain,
            metric_version=self.version,
            value=None,
            direction=self.direction,
            unit="test-only",
            status="not_applicable",
            reference_split="none",
            method_config_hash=config_hash({}),
            notes="fake plugin: interface validation only; no algorithm implemented",
        )
