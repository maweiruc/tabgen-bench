import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from tabgen_bench.experiments.artifacts import (
    ArtifactConflictError,
    IntegrityError,
    atomic_write,
    safe_path,
)
from tabgen_bench.experiments.cache import Cache


def test_atomic_exclusive_publication_preserves_existing_bytes(tmp_path):
    path = tmp_path / "artifact"
    atomic_write(path, b"first")
    with pytest.raises(ArtifactConflictError):
        atomic_write(path, b"second")
    assert path.read_bytes() == b"first" and not list(tmp_path.glob(".pending-*"))


def test_concurrent_publish_does_not_overwrite(tmp_path):
    cache = Cache(tmp_path / "cache")
    with ThreadPoolExecutor(max_workers=4) as pool:
        entries = list(
            pool.map(lambda _: cache.publish("sample", "1" * 64, {"x": b"same"}), range(4))
        )
    assert all(e.read("x") == b"same" for e in entries)
    with pytest.raises(ArtifactConflictError):
        cache.publish("sample", "1" * 64, {"x": b"changed"})


def test_cache_receipt_corruption_is_not_a_miss(tmp_path):
    cache = Cache(tmp_path)
    entry = cache.publish("data", "2" * 64, {"x": b"data"})
    path = entry.root / "entry.json"
    record = json.loads(path.read_text())
    record["metadata"] = {"altered": True}
    path.write_text(json.dumps(record))
    with pytest.raises(IntegrityError, match="receipt"):
        cache.get("data", "2" * 64)


def test_unpublished_temp_directory_is_not_cache_hit(tmp_path):
    cache = Cache(tmp_path)
    pending = tmp_path / "sample/.pending-process-killed"
    pending.mkdir(parents=True)
    (pending / "unfinished").write_bytes(b"incomplete")
    assert cache.get("sample", "3" * 64) is None


@pytest.mark.parametrize("name", ["../outside", "/absolute", "a/../../outside"])
def test_artifact_traversal_rejected(tmp_path, name):
    with pytest.raises(IntegrityError):
        safe_path(tmp_path, name)


def test_artifact_symlinks_rejected(tmp_path):
    (tmp_path / "link").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(IntegrityError):
        safe_path(tmp_path, "link/file")
