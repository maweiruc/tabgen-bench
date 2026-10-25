"""Dataset and generator composition for experiment execution."""

import json
from pathlib import Path

import yaml

from tabgen_bench.core.config import FrozenConfig
from tabgen_bench.core.hashing import config_hash
from tabgen_bench.datasets.openml import load_dataset
from tabgen_bench.datasets.provenance import load_public_metadata
from tabgen_bench.datasets.registry import DatasetRegistry
from tabgen_bench.datasets.splits import DatasetSplit
from tabgen_bench.experiments.contracts import (
    DatasetBundle,
    EvaluationSpec,
    GeneratorSpec,
    MetricSpec,
)
from tabgen_bench.experiments.manifest import environment_info
from tabgen_bench.experiments.planner import Planner
from tabgen_bench.generators.bootstrap import BootstrapGenerator
from tabgen_bench.generators.independent_marginal import IndependentMarginalGenerator
from tabgen_bench.generators.sdv_adapter import CTGANGenerator, GaussianCopulaGenerator

from .dependencies import dependency_fingerprint


class LocalDatasets:
    def __init__(self, *, registry_root, data_root, quality_metadata_path=None):
        self.registry = DatasetRegistry(registry_root)
        self.data_root = Path(data_root)
        self.provenance = (
            load_public_metadata("data-quality.json")
            if quality_metadata_path is None
            else json.loads(Path(quality_metadata_path).read_text())
        )
        self.loaded = {}

    def get(self, dataset_id):
        if dataset_id not in self.loaded:
            spec, lock = self.registry.spec(dataset_id), self.registry.lock(dataset_id)
            frame = load_dataset(self.registry, dataset_id, self.data_root)
            split = DatasetSplit.load(
                self.data_root / "splits" / f"{dataset_id}.json", expected=lock["split"]
            )
            self.loaded[dataset_id] = DatasetBundle(
                spec,
                frame,
                split,
                provenance=FrozenConfig.from_mapping(
                    {"source_lock": lock, "row_overlap": self.provenance[dataset_id]}
                ),
            )
        return self.loaded[dataset_id]


def generator_factory(spec):
    classes = {
        "bootstrap": BootstrapGenerator,
        "independent_marginal": IndependentMarginalGenerator,
        "gaussian_copula": GaussianCopulaGenerator,
        "ctgan": CTGANGenerator,
    }
    return classes[spec.id](**spec.params.to_dict())


def read_evaluation_spec(path):
    config = yaml.safe_load(Path(path).read_text())
    config["metrics"] = tuple(
        MetricSpec(**{**m, "params": FrozenConfig.from_mapping(m.get("params", {}))})
        for m in config["metrics"]
    )
    return EvaluationSpec.model_validate(config)


def plan_from_file(config_path, *, project_root, datasets, dependency_lock_path=None):
    root = Path(project_root)
    config = yaml.safe_load(Path(config_path).read_text())
    suite = read_evaluation_spec(root / config["evaluation_spec"])
    if config["experiment"]["suite"] != suite.id:
        raise ValueError("experiment/suite identity mismatch")
    if suite.id == "standard_v1":
        lock = json.loads((root / "configs/suites/protocols.lock.json").read_text())
        if (
            config_hash(config) != lock["standard_experiment_hash"]
            or suite.fingerprint() != lock["standard_evaluation_hash"]
        ):
            raise ValueError("frozen standard_v1 changed; declare a new protocol version")
    inputs = [datasets.get(name) for name in config["datasets"]]
    if any(b.split.seed != config["experiment"]["split_seed"] for b in inputs):
        raise ValueError("experiment attempts to replace an existing frozen split")
    if config["sampling"]["size"] != "train_size":
        raise ValueError("Experiment YAML requires train_size sampling")
    generators = [
        GeneratorSpec(
            id=g["id"],
            version=g["version"],
            adapter_version=g["adapter_version"],
            params=FrozenConfig.from_mapping(g["params"]),
        )
        for g in config["generators"]
    ]
    planner = Planner(
        dependency_lock_hash=dependency_fingerprint(dependency_lock_path),
        runtime_fingerprint=environment_info().fingerprint(),
    )
    return planner.expand(
        inputs, generators, suite, replicate_seeds=tuple(config["experiment"]["replicate_seeds"])
    )
