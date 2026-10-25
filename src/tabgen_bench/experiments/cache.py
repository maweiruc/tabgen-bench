"""Immutable stage entries. Corruption is an error, never an implicit cache miss."""

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from tabgen_bench.core.manifest import ArtifactInfo

from .artifacts import (
    ArtifactConflictError,
    IntegrityError,
    atomic_write,
    digest,
    file_lock,
    json_bytes,
    safe_path,
)

STAGES = {"data", "split", "fit", "sample", "metric", "report_input"}


@dataclass(frozen=True)
class CacheEntry:
    root: Path
    stage: str
    key: str
    artifacts: tuple[ArtifactInfo, ...]
    metadata: dict

    def read(self, name):
        artifact = next(a for a in self.artifacts if a.path == name)
        payload = safe_path(self.root, name).read_bytes()
        if digest(payload) != artifact.sha256 or len(payload) != artifact.size_bytes:
            raise IntegrityError(f"cache artifact mismatch: {self.stage}/{self.key}/{name}")
        return payload


class Cache:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, stage, key):
        if stage not in STAGES or not re.fullmatch("[0-9a-f]{64}", key):
            raise ValueError("invalid cache stage/key")
        return safe_path(self.root, f"{stage}/{key}")

    def get(self, stage, key):
        folder = self.directory(stage, key)
        if not folder.exists():
            return None
        try:
            payload = (folder / "entry.json").read_bytes()
            if (folder / "entry.sha256").read_text().strip() != digest(payload):
                raise IntegrityError("cache receipt hash mismatch")
            record = json.loads(payload)
            if (record["stage"], record["key"], record["cache_format_version"]) != (
                stage,
                key,
                "1.0.0",
            ):
                raise IntegrityError("cache entry identity/version mismatch")
            entry = CacheEntry(
                folder,
                stage,
                key,
                tuple(ArtifactInfo.model_validate(a) for a in record["artifacts"]),
                record["metadata"],
            )
            if len({a.path for a in entry.artifacts}) != len(entry.artifacts):
                raise IntegrityError("duplicate cache artifact")
            for artifact in entry.artifacts:
                entry.read(artifact.path)
            return entry
        except (OSError, ValueError, KeyError, StopIteration) as error:
            raise IntegrityError(f"corrupt cache entry: {stage}/{key}") from error

    def publish(self, stage, key, payloads, *, metadata=None):
        destination = self.directory(stage, key)
        lock = self.root / ".locks" / f"{stage}-{key}.lock"
        with file_lock(lock):
            existing = self.get(stage, key)
            if existing is not None:
                if {a.path for a in existing.artifacts} != set(payloads) or any(
                    existing.read(name) != value for name, value in payloads.items()
                ):
                    raise ArtifactConflictError(
                        "different output already published for this cache key"
                    )
                return existing
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=destination.parent))
            try:
                artifacts = []
                for name, payload in payloads.items():
                    if name in {"entry.json", "entry.sha256"}:
                        raise ValueError("reserved cache receipt filename")
                    atomic_write(safe_path(temporary, name), payload)
                    artifacts.append(
                        ArtifactInfo(path=name, sha256=digest(payload), size_bytes=len(payload))
                    )
                receipt = json_bytes(
                    {
                        "cache_format_version": "1.0.0",
                        "stage": stage,
                        "key": key,
                        "artifacts": [a.model_dump() for a in artifacts],
                        "metadata": metadata or {},
                    }
                )
                atomic_write(temporary / "entry.json", receipt)
                atomic_write(temporary / "entry.sha256", (digest(receipt) + "\n").encode())
                os.rename(temporary, destination)
                directory = os.open(destination.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            return self.get(stage, key)
