"""Explicit, local registry. Loading a registry never downloads data."""

import json
from pathlib import Path

import yaml

from tabgen_bench.core.hashing import config_hash
from tabgen_bench.core.schema import DatasetSchema, DatasetSource, DatasetSpec, TargetSpec

PINNED_SOURCES = {"adult": (1590, 2), "bank_marketing": (1461, 1), "credit_g": (31, 1)}


class DatasetRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def config(self, dataset_id: str) -> dict:
        if dataset_id not in PINNED_SOURCES:
            raise KeyError(f"unknown dataset: {dataset_id}")
        config = yaml.safe_load((self.root / "configs/datasets" / f"{dataset_id}.yaml").read_text())
        source = config["source"]
        if (source["data_id"], source["version"]) != PINNED_SOURCES[dataset_id]:
            raise ValueError("OpenML ID/version differs from the frozen V1.1 source")
        if config["id"] != dataset_id:
            raise ValueError("dataset identity mismatch")
        return config

    def spec(self, dataset_id: str) -> DatasetSpec:
        config = self.config(dataset_id)
        source = config["source"]
        schema = DatasetSchema.model_validate(
            yaml.safe_load((self.root / config["schema"]["definition_path"]).read_text())
        )
        return DatasetSpec(
            id=dataset_id,
            version=str(source["version"]),
            source=DatasetSource(
                provider="openml",
                source_id=str(source["data_id"]),
                version=str(source["version"]),
                upstream_url=source["upstream_url"],
                citation=source["citation"],
                license_note=source["license_note"],
            ),
            dataset_schema=schema,
            canonicalization_version=config["canonicalization_version"],
            expected_rows=config["expected_rows"],
            target=TargetSpec(**config["target"]),
        )

    def lock(self, dataset_id: str) -> dict:
        lock = json.loads((self.root / "datasets.lock.json").read_text())
        if lock["lock_version"] != "1.0.0":
            raise ValueError("unsupported dataset lock version")
        entry = lock["datasets"][dataset_id]
        spec, config = self.spec(dataset_id), self.config(dataset_id)
        expected = {
            "config_hash": config_hash(config),
            "schema_hash": spec.dataset_schema.fingerprint(),
            "generation_schema_hash": spec.dataset_schema.generation_view().fingerprint(),
            "validity_rules_hash": spec.dataset_schema.validity_rules().fingerprint(),
            "canonicalization_version": spec.canonicalization_version,
        }
        for key, value in expected.items():
            if entry[key] != value:
                raise ValueError(f"dataset lock mismatch: {key}")
        return entry
