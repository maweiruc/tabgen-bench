import copy
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from tabgen_bench.core.schema import (
    ColumnSchema,
    DatasetSchema,
    DatasetSource,
    DatasetSpec,
    TargetSpec,
)
from tabgen_bench.datasets.canonicalize import canonicalize, normalized_content_hash, parquet_bytes
from tabgen_bench.datasets.openml import sha256, verify_source
from tabgen_bench.datasets.registry import PINNED_SOURCES, DatasetRegistry
from tabgen_bench.datasets.splits import DatasetSplit, split_dataset
from tabgen_bench.generators.sdv_adapter import CTGANGenerator, GaussianCopulaGenerator

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = DatasetRegistry(ROOT)


@pytest.mark.parametrize("dataset_id,pair", list(PINNED_SOURCES.items()))
def test_openml_id_version_pinned(dataset_id, pair):
    config, spec, lock = (
        REGISTRY.config(dataset_id),
        REGISTRY.spec(dataset_id),
        REGISTRY.lock(dataset_id),
    )
    assert (config["source"]["data_id"], config["source"]["version"]) == pair
    assert (lock["source"]["data_id"], lock["source"]["version"]) == pair
    assert spec.expected_rows == lock["rows"]
    assert spec.target.column == config["target"]["column"]


def test_registry_rejects_changed_id_version(tmp_path):
    folder = tmp_path / "configs/datasets"
    folder.mkdir(parents=True)
    config = REGISTRY.config("adult")
    config["source"]["version"] = 3
    (folder / "adult.yaml").write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="ID/version"):
        DatasetRegistry(tmp_path).config("adult")
    with pytest.raises(KeyError):
        REGISTRY.config("latest_adult")


@pytest.mark.parametrize("dataset_id", PINNED_SOURCES)
def test_schema_constraints_source_and_hash(dataset_id):
    schema = REGISTRY.spec(dataset_id).dataset_schema
    lock = REGISTRY.lock(dataset_id)
    assert schema.fingerprint() == lock["schema_hash"]
    assert schema.generation_view().fingerprint() == lock["generation_schema_hash"]
    assert schema.validity_rules().fingerprint() == lock["validity_rules_hash"]
    for column in schema.columns:
        assert "OpenML" in column.constraint_source and "UCI" in column.constraint_source
        assert column.minimum is None and column.maximum is None
        if column.logical_type == "categorical":
            assert column.allowed_values
        assert column.rule_version == "1.0.0"
    projected = schema.generation_view().model_dump_json()
    for field in ("minimum", "maximum", "allowed_values", "constraint_source", "rule_version"):
        assert field not in projected


def test_schema_mutation_invalidates_lock(tmp_path):
    import shutil

    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    shutil.copy(ROOT / "datasets.lock.json", tmp_path / "datasets.lock.json")
    path = tmp_path / "configs/datasets/schemas/adult_v1.yaml"
    schema = yaml.safe_load(path.read_text())
    schema["columns"][0]["minimum"] = 25
    path.write_text(yaml.safe_dump(schema))
    with pytest.raises(ValueError, match="schema_hash"):
        DatasetRegistry(tmp_path).lock("adult")


def toy_spec():
    return DatasetSpec(
        id="toy",
        version="1",
        source=DatasetSource(provider="test", source_id="1", version="1"),
        canonicalization_version="v1",
        expected_rows=4,
        dataset_schema=DatasetSchema(
            columns=(
                ColumnSchema(name="a", role="feature", logical_type="integer", nullable=True),
                ColumnSchema(
                    name="y",
                    role="target",
                    logical_type="categorical",
                    nullable=False,
                    allowed_values=("no", "yes"),
                ),
            )
        ),
        target=TargetSpec(column="y", positive_label="yes", negative_label="no"),
    )


def test_canonicalization_is_explicit_and_preserves_na():
    config = {
        "mapping": {
            "features": {"V1": "a"},
            "raw_target": "Class",
            "values": {"y": {"1": "no", "2": "yes"}},
        }
    }
    data = pd.DataFrame({"V1": [1.0, None, 2.0, 3.0]})
    target = pd.Series(["1", "2", "1", "2"], name="Class")
    original = data.copy(deep=True)
    frame = canonicalize(data, target, config, toy_spec())
    assert str(frame.a.dtype) == "Int64" and pd.isna(frame.a.iloc[1])
    assert list(frame.y) == ["no", "yes", "no", "yes"]
    pd.testing.assert_frame_equal(data, original)
    for changed, y in (
        (data.rename(columns={"V1": "a"}), target),
        (data.astype("string"), target),
        (data.assign(V1=[1.2, 2.0, 3.0, 4.0]), target),
        (data, target.rename("y")),
        (data, target.replace("1", "3")),
    ):
        with pytest.raises(ValueError):
            canonicalize(changed, y, config, toy_spec())


def test_canonical_hash_stable_and_sensitive():
    frame = pd.DataFrame({"n": pd.Series([1, pd.NA], dtype="Int64"), "s": ["nan", None]})
    assert normalized_content_hash(frame) == normalized_content_hash(frame.copy())
    assert sha256(parquet_bytes(frame)) == sha256(parquet_bytes(frame.copy()))
    assert normalized_content_hash(frame) != normalized_content_hash(frame.iloc[::-1])
    assert normalized_content_hash(frame) != normalized_content_hash(frame.replace("nan", None))


def test_source_checksum_and_identity_errors():
    lock = REGISTRY.lock("adult")
    source = lock["source"]
    metadata = {
        "id": source["data_id"],
        "version": source["version"],
        "file_id": source["file_id"],
        "url": source["raw_url"],
        "default_target_attribute": source["raw_target"],
        "md5_checksum": source["raw_md5"],
        "licence": "Public",
    }
    with pytest.raises(ValueError, match="MD5"):
        verify_source(REGISTRY.config("adult"), metadata, b"tampered", lock)
    metadata["version"] = 3
    with pytest.raises(ValueError, match="ID/version/target"):
        verify_source(REGISTRY.config("adult"), metadata, b"tampered", lock)


def test_split_uses_positions_and_checks_content(tmp_path):
    frame = pd.DataFrame({"x": range(100), "y": ["no", "yes"] * 50}, index=[5] * 100)
    digest = normalized_content_hash(frame)
    split = split_dataset(frame, "y", content_hash=digest)
    assert [len(v) for v in split.frames(frame)] == [60, 20, 20]
    assert split == split_dataset(frame, "y", content_hash=digest)
    assert split != split_dataset(frame, "y", content_hash=digest, seed=23)
    with pytest.raises(ValueError, match="content"):
        split.frames(frame.iloc[::-1])
    split.save(tmp_path / "split.json")
    assert json.loads((tmp_path / "split.json").read_text()) == split.to_dict()
    assert DatasetSplit.load(tmp_path / "split.json") == split
    wrong_expected = {k: v for k, v in split.to_dict().items() if k != "positions"}
    wrong_expected["seed"] = 23
    with pytest.raises(ValueError, match="frozen dataset lock"):
        DatasetSplit.load(tmp_path / "split.json", expected=wrong_expected)
    with pytest.raises(FileExistsError):
        split.save(tmp_path / "split.json")
    changed = split.to_dict()
    changed["position_hashes"]["test"] = "0" * 64
    (tmp_path / "tampered.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="hash mismatch"):
        DatasetSplit.load(tmp_path / "tampered.json")
    kwargs = {
        name: getattr(split, name)
        for name in ("train", "validation", "test", "seed", "n_rows", "content_hash")
    }
    kwargs["test"] = split.validation
    with pytest.raises(ValueError, match="disjoint"):
        DatasetSplit(**kwargs)


@pytest.mark.parametrize("generator", [GaussianCopulaGenerator, CTGANGenerator])
def test_sdv_configuration_and_projection_without_import(generator):
    from tabgen_bench.core.errors import ConfigurationError

    with pytest.raises(ConfigurationError):
        generator(unsupported=True)
    with pytest.raises(ConfigurationError):
        generator(cpu_threads=0)
    g = generator()
    with pytest.raises(TypeError, match="rule-free"):
        g.fit(pd.DataFrame({"a": [1]}), toy_spec().dataset_schema, seed=11)
    with pytest.raises(RuntimeError):
        g.sample(2, seed=11)
    assert g.environment_info()["train_rows"] is None


def test_ctgan_cpu_policy():
    from tabgen_bench.core.errors import ConfigurationError

    with pytest.raises(ConfigurationError, match="CPU"):
        CTGANGenerator(enable_gpu=True)
    with pytest.raises(ConfigurationError, match="unsupported"):
        CTGANGenerator(cuda=False)


def test_config_hash_changes_on_mapping_change():
    from tabgen_bench.core.hashing import config_hash

    config = REGISTRY.config("bank_marketing")
    changed = copy.deepcopy(config)
    changed["mapping"]["values"]["y"]["1"] = "yes"
    assert config_hash(config) != config_hash(changed)


def test_download_transient_retry_is_bounded(monkeypatch):
    from io import BytesIO
    from urllib.error import HTTPError

    from tabgen_bench.datasets import openml

    calls, sleeps = [], []

    def transient(*args, **kwargs):
        calls.append(args[0])
        if len(calls) < 3:
            raise HTTPError(args[0], 503, "temporary", {}, None)
        return BytesIO(b"real-network-is-not-used-by-this-unit-test")

    monkeypatch.setattr(openml, "urlopen", transient)
    monkeypatch.setattr(openml.time, "sleep", sleeps.append)
    assert openml.read_url("https://example.invalid/source")
    assert len(calls) == 3 and sleeps == [1, 2]

    def permanent(*args, **kwargs):
        raise HTTPError(args[0], 404, "missing", {}, None)

    monkeypatch.setattr(openml, "urlopen", permanent)
    with pytest.raises(HTTPError):
        openml.read_url("https://example.invalid/source")
    assert sleeps == [1, 2]
