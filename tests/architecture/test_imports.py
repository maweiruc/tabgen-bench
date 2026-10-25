import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.architecture.import_policy import ALLOWED, MODULES, check_source, check_tree

ROOT = Path(__file__).resolve().parents[2]


def test_architecture_import_dag():
    root = ROOT / "src" / "tabgen_bench"
    assert {p.name for p in root.iterdir() if p.is_dir() and p.name != "__pycache__"} == MODULES
    assert {p.stem for p in root.glob("*.py")} == {"__init__"}
    errors, edges = check_tree(root)
    assert not errors, "\n".join(errors)

    def visit(node, visiting):
        assert node not in visiting, f"dependency cycle at {node}"
        for source, target in edges:
            if source == node:
                visit(target, visiting | {node})

    for module in MODULES:
        visit(module, set())


FORBIDDEN = [
    (source, target)
    for source in sorted(MODULES)
    for target in sorted(MODULES)
    if source != target and target not in ALLOWED[source]
]


@pytest.mark.parametrize("source,target", FORBIDDEN)
def test_import_checker_rejects_each_forbidden_edge(source, target):
    errors, _ = check_source(f"import tabgen_bench.{target}", f"tabgen_bench.{source}.example")
    assert errors


@pytest.mark.parametrize(
    "source",
    [
        "from tabgen_bench.evaluation import engine",
        "from tabgen_bench import evaluation as evaluator",
        "from ..evaluation import engine",
        "from .. import evaluation",
        "if True:\n    import tabgen_bench.evaluation.engine as e",
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from .. import evaluation",
        "from ..evaluation import *",
        "import importlib\nimportlib.import_module('tabgen_bench.evaluation')",
        "from importlib import import_module as load\nload('tabgen_bench.evaluation')",
        "__import__('tabgen_bench.evaluation')",
        "import importlib as il\nil.import_module(module_name)",
    ],
)
def test_import_checker_catches_alternative_import_forms(source):
    errors, _ = check_source(source, "tabgen_bench.metrics.example")
    assert errors


@pytest.mark.parametrize(
    "source,target", [(s, t) for s in sorted(MODULES) for t in sorted(ALLOWED[s] | {s})]
)
def test_import_checker_accepts_permitted_edges(source, target):
    errors, _ = check_source(
        f"from tabgen_bench.{target} import types", f"tabgen_bench.{source}.example"
    )
    assert not errors


@pytest.mark.parametrize(
    "source", ["import sdv", "from torch import nn", "from tests import fakes"]
)
def test_core_optional_and_test_imports_forbidden(source):
    errors, _ = check_source(source, "tabgen_bench.core.example")
    assert errors


def test_package_init_import_boundaries():
    errors, _ = check_source("from .. import metrics", "tabgen_bench.core", is_package=True)
    assert errors
    errors, _ = check_source("from . import core", "tabgen_bench", is_package=True)
    assert errors  # root must remain light; avoid transitive DAG bypasses


def test_core_import_without_sdv():
    result = subprocess.run(
        [
            os.environ.get("TABGENBENCH_CORE_PYTHON", sys.executable),
            "-I",
            str(ROOT / "tests" / "import_smoke.py"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"sdv_installed": false' in result.stdout
    assert '"torch_installed": false' in result.stdout


def test_no_network_for_core_tests():
    import socket

    with pytest.raises(AssertionError, match="network"):
        socket.create_connection(("127.0.0.1", 9))


def test_engine_has_no_algorithm_import_or_domain_dispatch():
    import ast

    tree = ast.parse((ROOT / "src/tabgen_bench/evaluation/engine.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module.startswith("tabgen_bench.core") or node.module in {
                "datetime",
                "uuid",
            }
        if isinstance(node, ast.Compare):
            assert not any(
                isinstance(term, ast.Constant)
                and term.value
                in {"validity", "fidelity", "privacy", "utility", "statistical", "diversity"}
                for term in node.comparators
            )
