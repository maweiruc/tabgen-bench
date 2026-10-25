import ast
from pathlib import Path

from tests.architecture.import_policy import check_source

ROOT = Path(__file__).resolve().parents[2]


def test_reporting_only_imports_core_and_never_runs_evaluation():
    for path in (ROOT / "src/tabgen_bench/reporting").glob("*.py"):
        errors, _ = check_source(path.read_text(), f"tabgen_bench.reporting.{path.stem}")
        assert not errors
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {
                    "fit",
                    "sample",
                    "evaluate",
                    "compute",
                    "read_parquet",
                    "read_csv",
                }
