"""All tests run offline. Only explicitly marked real SDV tests may import optional ML."""

import importlib.abc
import socket
import sys

import pytest

from tabgen_bench.core.schema import ColumnSchema, DatasetSchema


class BlockOptionalML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {"sdv", "torch"}:
            raise AssertionError(f"optional ML import forbidden in core tests: {fullname}")
        return None


@pytest.fixture(autouse=True)
def offline_without_optional_ml(monkeypatch, request):
    def blocked(*args, **kwargs):
        raise AssertionError("network access forbidden in tests")

    for name in ("connect", "connect_ex", "sendto", "sendmsg"):
        if hasattr(socket.socket, name):
            monkeypatch.setattr(socket.socket, name, blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    finder = BlockOptionalML()
    optional_allowed = request.node.get_closest_marker("sdv") is not None
    if not optional_allowed:
        sys.meta_path.insert(0, finder)
    yield
    if not optional_allowed:
        sys.meta_path.remove(finder)


@pytest.fixture
def schema():
    return DatasetSchema(
        columns=(
            ColumnSchema(
                name="age",
                role="feature",
                logical_type="integer",
                nullable=False,
                minimum=0,
                maximum=120,
                constraint_source="toy_dictionary",
                rule_version="1",
            ),
            ColumnSchema(
                name="group",
                role="feature",
                logical_type="categorical",
                nullable=True,
                allowed_values=("A", "B", "unknown"),
                constraint_source="toy_dictionary",
                rule_version="1",
            ),
            ColumnSchema(name="label", role="target", logical_type="boolean", nullable=False),
        )
    )
