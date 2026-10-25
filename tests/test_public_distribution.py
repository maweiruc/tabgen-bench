"""Public installation, resource and dependency identity contracts."""

import hashlib
import json
import re
import tomllib
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace

import pytest

from tabgen_bench.datasets.provenance import PUBLIC_METADATA_SHA256, load_public_metadata
from tabgen_bench.interfaces import dependencies
from tabgen_bench.interfaces.configuration import EvaluationConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", PUBLIC_METADATA_SHA256)
def test_public_provenance_is_bundled_and_immutable(name):
    payload = files("tabgen_bench.datasets").joinpath("resources", name).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == PUBLIC_METADATA_SHA256[name]
    assert load_public_metadata(name) == json.loads(payload)


def test_dependency_identity_is_order_independent_and_version_sensitive(monkeypatch):
    a = SimpleNamespace(metadata={"Name": "Example_Package"}, version="1.0")
    b = SimpleNamespace(metadata={"Name": "Another-Package"}, version="2.0")
    monkeypatch.setattr(dependencies, "distributions", lambda: [a, b])
    first = dependencies.dependency_fingerprint()
    monkeypatch.setattr(dependencies, "distributions", lambda: [b, a])
    assert dependencies.dependency_fingerprint() == first
    a.version = "1.1"
    assert dependencies.dependency_fingerprint() != first


def test_explicit_dependency_snapshot_keeps_byte_hash_contract(tmp_path):
    path = tmp_path / "dependencies.txt"
    path.write_bytes(b"example==1.0\n")
    assert (
        dependencies.dependency_fingerprint(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    with pytest.raises(FileNotFoundError):
        dependencies.dependency_fingerprint(tmp_path / "missing.txt")


def test_dependency_snapshot_is_optional():
    assert EvaluationConfig.model_fields["dependency_lock"].default is None


def test_package_license_and_optional_dependencies_are_explicit():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["version"] == "0.3.0" and project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE"]
    assert project["optional-dependencies"]["sdv"] == ["sdv==1.38.5"]
    assert all(not d.startswith(("sdv", "torch")) for d in project["dependencies"])


def test_public_document_links_resolve():
    for path in (ROOT / "README.md", ROOT / "docs/protocol.md"):
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if not target.startswith(("https://", "http://", "#")):
                assert (path.parent / target.split("#", 1)[0]).is_file(), (path, target)
