"""Explicit OpenML fetch and strict local content verification; never fetch on import."""

import hashlib
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd
import pyarrow
import sklearn
from sklearn.datasets import fetch_openml

from tabgen_bench.core.hashing import config_hash

from .canonicalize import canonicalize, normalized_content_hash, parquet_bytes
from .registry import DatasetRegistry


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_url(url: str, *, timeout: int = 90) -> bytes:
    """Bounded retry for transient public-source failures, never for integrity mismatches."""
    for attempt in range(3):
        try:
            with urlopen(url, timeout=timeout) as response:
                return response.read()
        except HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def source_identity(metadata: dict) -> dict:
    info = metadata["data_set_description"] if "data_set_description" in metadata else metadata
    return {
        "data_id": int(info["id"]),
        "version": int(info["version"]),
        "file_id": str(info["file_id"]),
        "raw_url": info["url"],
        "raw_target": info["default_target_attribute"],
        "raw_md5": info["md5_checksum"],
        "openml_license": info["licence"],
    }


def verify_source(config: dict, metadata: dict, raw: bytes, lock: dict | None = None):
    source = source_identity(metadata)
    if (source["data_id"], source["version"], source["raw_target"]) != (
        config["source"]["data_id"],
        config["source"]["version"],
        config["mapping"]["raw_target"],
    ):
        raise ValueError("OpenML source ID/version/target mismatch")
    if hashlib.md5(raw, usedforsecurity=False).hexdigest() != source["raw_md5"]:
        raise ValueError("OpenML raw MD5 mismatch")
    if lock is not None:
        if source != lock["source"] or sha256(raw) != lock["raw_sha256"]:
            raise ValueError("OpenML source/raw SHA256 differs from frozen lock")
    return source


def make_lock_entry(
    registry: DatasetRegistry, dataset_id: str, metadata: dict, raw: bytes, frame: pd.DataFrame
) -> dict:
    """For the explicit first-freeze script; does not write or update any lock."""
    config, spec = registry.config(dataset_id), registry.spec(dataset_id)
    source = verify_source(config, metadata, raw)
    return {
        "source": source,
        "raw_sha256": sha256(raw),
        "raw_bytes": len(raw),
        "canonicalization_version": spec.canonicalization_version,
        "config_hash": config_hash(config),
        "schema_hash": spec.dataset_schema.fingerprint(),
        "generation_schema_hash": spec.dataset_schema.generation_view().fingerprint(),
        "validity_rules_hash": spec.dataset_schema.validity_rules().fingerprint(),
        "constraint_rules_version": spec.dataset_schema.constraint_rules_version,
        "normalized_parquet_sha256": sha256(parquet_bytes(frame)),
        "normalized_content_sha256": normalized_content_hash(frame),
        "rows": len(frame),
        "columns": list(frame.columns),
        "target_counts": {
            str(k): int(v) for k, v in frame[spec.target.column].value_counts().items()
        },
        "missing_counts": {k: int(v) for k, v in frame.isna().sum().items()},
        "writer": {
            "pyarrow": pyarrow.__version__,
            "pandas": pd.__version__,
            "sklearn_parser": sklearn.__version__,
            "parquet_profile": "uncompressed_v1",
        },
    }


def _write_or_verify(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"existing artifact differs; refusing to overwrite: {path}")
    else:
        with path.open("xb") as output:
            output.write(data)


def fetch_dataset(registry: DatasetRegistry, dataset_id: str, data_root: str | Path):
    """Network action: fetch exactly the locked object, verify, persist, and return a copy."""
    config, lock = registry.config(dataset_id), registry.lock(dataset_id)
    spec = registry.spec(dataset_id)
    data_root = Path(data_root)
    data_id = config["source"]["data_id"]
    metadata_url = f"https://www.openml.org/api/v1/json/data/{data_id}"
    metadata_bytes = read_url(metadata_url)
    metadata = json.loads(metadata_bytes)
    if source_identity(metadata) != lock["source"]:
        raise ValueError("OpenML metadata differs from the locked source")
    raw = read_url(lock["source"]["raw_url"], timeout=180)
    verify_source(config, metadata, raw, lock)
    # This is deliberately the public API required by the protocol, using .data and .target.
    bunch = fetch_openml(
        data_id=data_id, as_frame=True, parser="pandas", data_home=data_root / "openml/sklearn"
    )
    if source_identity(bunch.details) != lock["source"]:
        raise ValueError("fetch_openml metadata differs from the locked raw source")
    frame = canonicalize(bunch.data, bunch.target, config, spec)
    payload = parquet_bytes(frame)
    if sha256(payload) != lock["normalized_parquet_sha256"]:
        raise ValueError("canonical Parquet SHA256 mismatch (including pinned writer profile)")
    if normalized_content_hash(frame) != lock["normalized_content_sha256"]:
        raise ValueError("normalized semantic content SHA256 mismatch")
    folder = data_root / "openml/raw" / str(data_id)
    _write_or_verify(folder / "source.arff", raw)
    # Metadata contains mutable administrative fields; preserve the first exact response.
    if not (folder / "metadata.json").exists():
        _write_or_verify(folder / "metadata.json", metadata_bytes)
    _write_or_verify(data_root / "canonical" / f"{dataset_id}.parquet", payload)
    return frame.copy(deep=True)


def load_dataset(registry: DatasetRegistry, dataset_id: str, data_root: str | Path):
    """Offline-only verified load. Missing artifacts are errors, never implicit downloads."""
    lock = registry.lock(dataset_id)
    data_root = Path(data_root)
    folder = data_root / "openml/raw" / str(lock["source"]["data_id"])
    verify_source(
        registry.config(dataset_id),
        json.loads((folder / "metadata.json").read_text()),
        (folder / "source.arff").read_bytes(),
        lock,
    )
    path = data_root / "canonical" / f"{dataset_id}.parquet"
    if sha256(path.read_bytes()) != lock["normalized_parquet_sha256"]:
        raise ValueError("canonical Parquet SHA256 mismatch")
    frame = pd.read_parquet(path)
    if normalized_content_hash(frame) != lock["normalized_content_sha256"]:
        raise ValueError("normalized semantic content SHA256 mismatch")
    return frame
