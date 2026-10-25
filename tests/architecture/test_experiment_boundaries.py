import ast
from pathlib import Path

from tests.architecture.import_policy import check_source

ROOT = Path(__file__).resolve().parents[2]


def test_runner_has_no_algorithm_or_metric_domain_dispatch():
    path = ROOT / "src/tabgen_bench/experiments/runner.py"
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = getattr(node, "module", "") or ""
            assert not module.startswith(("tabgen_bench.metrics", "sdv", "torch"))
        if isinstance(node, ast.Compare):
            assert not any(
                isinstance(t, ast.Constant)
                and t.value
                in {
                    "statistical",
                    "privacy",
                    "fidelity",
                    "validity",
                    "utility",
                    "diversity",
                    "bootstrap",
                    "independent_marginal",
                    "gaussian_copula",
                    "ctgan",
                }
                for t in node.comparators
            )


def test_experiments_cannot_import_reporting_or_metrics():
    for module in ("reporting", "metrics"):
        errors, _ = check_source(
            f"from tabgen_bench.{module} import algorithm", "tabgen_bench.experiments.runner"
        )
        assert errors


def test_no_model_pickle_or_private_sdv_rng_in_experiments():
    for file in (ROOT / "src/tabgen_bench/experiments").glob("*.py"):
        text = file.read_text()
        assert "import pickle" not in text and "import torch" not in text
        assert "_set_random_state" not in text and "._model" not in text
