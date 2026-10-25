"""Local atomic, exclusive publication and cryptographic inventory checks."""

import fcntl
import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

from tabgen_bench.core.hashing import canonical_json
from tabgen_bench.core.manifest import ArtifactInfo


class IntegrityError(RuntimeError):
    pass


class ArtifactConflictError(RuntimeError):
    pass


def digest(payload: bytes):
    return hashlib.sha256(payload).hexdigest()


def safe_path(root: Path, relative: str):
    path = Path(relative)
    if path.is_absolute() or not path.parts or any(p in {".", ".."} for p in path.parts):
        raise IntegrityError("artifact path must be a relative path within its store")
    root = root.resolve()
    target = root / path
    if not target.resolve().is_relative_to(root):
        raise IntegrityError("artifact symlink escapes store")
    for parent in (target, *target.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise IntegrityError("artifact symlinks are not allowed")
    return target


def atomic_write(path: Path, payload: bytes):
    """fsync then exclusive hard-link publication: existing paths are never replaced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ArtifactConflictError(f"refusing to overwrite {path}") from error
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


def json_bytes(value):
    return (canonical_json(value) + "\n").encode()


@contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class ArtifactStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, run_id):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", run_id):
            raise ValueError("invalid run ID")
        return safe_path(self.root, run_id)

    def create(self, run_id):
        self.directory(run_id).mkdir(exist_ok=False)

    def write(self, run_id, relative, payload):
        if (self.directory(run_id) / "manifest.json").exists():
            raise ArtifactConflictError("terminal run is sealed; create a new run record")
        atomic_write(safe_path(self.directory(run_id), relative), payload)
        return ArtifactInfo(path=relative, sha256=digest(payload), size_bytes=len(payload))

    def read(self, run_id, artifact: ArtifactInfo):
        path = safe_path(self.directory(run_id), artifact.path)
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise IntegrityError(f"missing artifact: {artifact.path}") from error
        if digest(payload) != artifact.sha256 or len(payload) != artifact.size_bytes:
            raise IntegrityError(f"artifact hash/size mismatch: {artifact.path}")
        return payload

    def verify(self, manifest):
        for artifact in manifest.artifacts:
            self.read(manifest.run_id, artifact)

    def read_json(self, run_id, artifact):
        return json.loads(self.read(run_id, artifact))
