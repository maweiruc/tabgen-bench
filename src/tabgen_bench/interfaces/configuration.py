"""Strict external workflow configuration and separate frozen plans."""

import csv
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from tabgen_bench.core.errors import ConfigurationError


class Storage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    root: str = "runs"
    cache_root: str = "cache"


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_path: str = Field(alias="schema")  # A complete, standalone DatasetSpec.
    synthetic: str
    synthetic_label: str = Field(default="external", min_length=1)
    train: str | None = None
    validation: str | None = None
    test: str | None = None
    reference: str | None = None
    evaluation_spec: str
    dependency_lock: str | None = None
    seed: int = Field(default=37, ge=0, le=2**32 - 1)
    split_seed: int = 20261025
    expected_n_rows: int | None = Field(default=None, ge=0)
    output: str


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plan_config: str
    project_root: str
    data_root: str
    quality_metadata: str | None = None
    dependency_lock: str | None = None
    storage: Storage
    local_datasets: list[dict[str, str]] = Field(default_factory=list)


def read_yaml(path):
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ConfigurationError("configuration must be a YAML object")
    return document


def relative(base, path):
    return (Path(base) / path).resolve()


def read_table(path):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            header = next(csv.reader(stream), [])
        if len(header) != len(set(header)):
            raise ConfigurationError("CSV duplicate column names; refusing implicit renaming")
        # Only empty fields encode missingness. Literal nan / unknown are preserved.
        return pd.read_csv(path, keep_default_na=False, na_values=[""])
    raise ConfigurationError("input must be CSV or Parquet")
