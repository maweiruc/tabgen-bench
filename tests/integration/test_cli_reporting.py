import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tabgen_bench.interfaces import api
from tabgen_bench.interfaces.cli import main
from tabgen_bench.interfaces.report_service import read_record
from tests.support_workflows import ROOT, eval_config


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--help"],
        ["--version"],
        ["evaluate", "--help"],
        ["run", "--help"],
        ["compare", "--help"],
        ["report", "--help"],
        ["datasets", "fetch", "--help"],
        ["datasets", "verify", "--help"],
        ["unknown"],
        ["evaluate"],
        ["run"],
        ["compare"],
        ["report"],
    ],
)
def test_cli_help_exit_codes(arguments, capsys):
    expected = (
        2 if arguments in [["unknown"], ["evaluate"], ["run"], ["compare"], ["report"]] else 0
    )
    try:
        code = main(arguments)
    except SystemExit as error:
        code = error.code
    assert code == expected
    captured = capsys.readouterr()
    assert captured.out or captured.err


def test_evaluate_only_without_sdv(tmp_path):
    python = os.environ.get("TABGENBENCH_CORE_PYTHON", sys.executable)
    config = eval_config(tmp_path)
    code = """
import importlib.util, socket, sys
assert importlib.util.find_spec('sdv') is None
assert importlib.util.find_spec('torch') is None
def blocked(*a, **k): raise AssertionError('network forbidden')
socket.socket.connect=blocked
socket.create_connection=blocked
socket.getaddrinfo=blocked
from tabgen_bench.interfaces.cli import main
raise SystemExit(main(['evaluate','--config',sys.argv[1]]))
"""
    completed = subprocess.run(
        [python, "-I", "-c", code, str(config)], text=True, capture_output=True, timeout=60
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["metric_status_counts"] == {"ok": 17}
    assert Path(result["report"]["report"]).is_file()


def test_python_api_equals_cli_metrics(tmp_path, capsys):
    one = api.evaluate(eval_config(tmp_path, name="python-api"), create_report=False)
    assert main(["evaluate", "--config", str(eval_config(tmp_path, name="cli-api"))]) == 0
    two = json.loads(capsys.readouterr().out)
    left, right = read_record(one["run"]), read_record(two["run"])
    for a, b in zip(left.metrics, right.metrics, strict=True):
        for field in (
            "metric_id",
            "value",
            "status",
            "direction",
            "unit",
            "details",
            "method_config_hash",
            "references",
        ):
            assert getattr(a, field) == getattr(b, field)
    folder = Path(two["report"]["output"])
    csv_rows = list(csv.DictReader((folder / "metrics.csv").open()))
    json_rows = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
    for record, csv_row, json_row in zip(right.metrics, csv_rows, json_rows, strict=True):
        assert json_row == record.model_dump(mode="json")
        assert float(csv_row["value"]) == record.value


def test_dry_run_no_training_sampling_or_run_records(tmp_path, monkeypatch):
    from tabgen_bench.generators.bootstrap import BootstrapGenerator

    def forbidden(*a, **k):
        raise AssertionError("dry run attempted execution")

    monkeypatch.setattr(BootstrapGenerator, "fit", forbidden)
    monkeypatch.setattr(BootstrapGenerator, "sample", forbidden)
    result = api.run(
        ROOT / "configs/examples/toy_run.yaml",
        dry_run=True,
        run_root=tmp_path / "runs",
        cache_root=tmp_path / "cache",
    )
    assert result["n_planned"] == 2 and result["n_attempted"] == 0
    assert not list(tmp_path.iterdir())


def test_run_service_cli_real_diagnostic_generators(tmp_path, capsys):
    arguments = [
        "run",
        "--config",
        str(ROOT / "configs/examples/toy_run.yaml"),
        "--run-root",
        str(tmp_path / "runs"),
        "--cache-root",
        str(tmp_path / "cache"),
    ]
    assert main(arguments) == 0
    result = json.loads(capsys.readouterr().out)
    assert (result["n_planned"], result["n_attempted"], result["n_completed"]) == (2, 2, 2)
    assert all(len(read_record(r["run"]).metrics) == 17 for r in result["runs"])
    report = api.compare([r["run"] for r in result["runs"]], output=tmp_path / "comparison")
    assert len(report["groups"]) == 2


@pytest.mark.parametrize(
    "error,exit_code",
    [
        (ValueError("bad config"), 2),
        (FileExistsError("already exists"), 4),
        (RuntimeError("unexpected"), 1),
        (KeyboardInterrupt(), 130),
    ],
)
def test_cli_traceable_errors(error, exit_code, tmp_path, monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(api, "evaluate", fail)
    assert main(["evaluate", "--config", str(tmp_path / "missing.yaml")]) == exit_code
    diagnostic = json.loads(capsys.readouterr().err)
    assert diagnostic["trace_id"] and diagnostic["exit_code"] == exit_code
    assert diagnostic["command"] == "evaluate"


def test_cli_partial_failure_exit_three(tmp_path, monkeypatch, capsys):
    from tabgen_bench.metrics.registry import PluginRegistry
    from tabgen_bench.metrics.validity.schema_checks import plugins
    from tests.support_runner import FakeMetric

    calls = {"fail_metric": True}
    fake = FakeMetric(
        id="fake.future",
        domain="statistical",
        direction="diagnostic",
        unit="test_only",
        required_inputs=frozenset({"synthetic"}),
        calls=calls,
    )
    registry = PluginRegistry((*plugins(), fake))
    suite = {
        "id": "explicit_test_suite",
        "version": "1.0.0",
        "metrics": [
            {"id": p.id, "version": p.version, "domain": p.domain, "params": p.params.to_dict()}
            for p in (*plugins(), fake)
        ],
    }
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(yaml.safe_dump(suite))
    path = eval_config(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    cfg["evaluation_spec"] = str(suite_path)
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(api, "build_metric_registry", lambda: registry)
    assert main(["evaluate", "--config", str(path)]) == 3
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "partial_failure"
    html = Path(result["report"]["report"]).read_text()
    assert "test-only future plugin failure" in html and "failed" in html


def test_dataset_errors_not_silently_excluded(tmp_path, capsys):
    assert main(["datasets", "verify", "adult", "unknown", "--data-root", str(tmp_path)]) == 3
    result = json.loads(capsys.readouterr().out)
    assert len(result["datasets"]) == 2
    assert all(r["status"] == "failed" and r["error_message"] for r in result["datasets"])


def test_external_infinity_reaches_validity_without_repair(tmp_path):
    import pandas as pd

    path = eval_config(tmp_path, name="nonfinite")
    cfg = yaml.safe_load(path.read_text())
    frame = pd.read_csv(cfg["synthetic"])
    frame["x"] = frame["x"].astype(float)
    frame.loc[0, "x"] = float("inf")
    synthetic = tmp_path / "nonfinite.csv"
    frame.to_csv(synthetic, index=False)
    cfg["synthetic"] = str(synthetic)
    path.write_text(yaml.safe_dump(cfg))
    result = api.evaluate(path)
    record = read_record(result["run"])
    assert record.metrics[0].value == 0
    assert record.audit["synthetic_multiset_sha256"] is None
    assert result["report"]["groups"][0]["n_unique_outputs"] is None
    assert "unknown, not zero" in Path(result["report"]["report"]).read_text()
    assert "inf" in synthetic.read_text()


def test_reference_only_api_keeps_na_metrics(tmp_path):
    result = api.evaluate(eval_config(tmp_path, reference_only=True), create_report=False)
    records = read_record(result["run"]).metrics
    assert len(records) == 17
    assert any(r.status == "not_applicable" and r.value is None for r in records)


def test_csv_duplicate_headers_explicitly_rejected(tmp_path):
    from tabgen_bench.core.errors import ConfigurationError
    from tabgen_bench.interfaces.configuration import read_table

    path = tmp_path / "duplicated.csv"
    path.write_text("x,x,label\n1,2,True\n")
    with pytest.raises(ConfigurationError, match="implicit renaming"):
        read_table(path)


def test_evaluation_does_not_overwrite_or_create_invalid_config_run(tmp_path):
    path = eval_config(tmp_path)
    result = api.evaluate(path, create_report=False)
    before = (Path(result["run"]) / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        api.evaluate(path, create_report=False)
    assert before == (Path(result["run"]) / "manifest.json").read_bytes()
    cfg = yaml.safe_load(path.read_text())
    cfg["output"] = str(tmp_path / "bad-lock")
    cfg["dependency_lock"] = str(tmp_path / "missing.lock")
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(FileNotFoundError):
        api.evaluate(path, create_report=False)
    assert not (tmp_path / "bad-lock").exists()


def test_external_shared_real_file_rejected(tmp_path):
    from tabgen_bench.core.errors import ConfigurationError

    path = eval_config(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    cfg["validation"] = cfg["test"]
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ConfigurationError, match="same real file"):
        api.evaluate(path, create_report=False)
    assert not Path(cfg["output"]).exists()


def test_external_identical_content_not_assumed_independent(tmp_path):
    path = eval_config(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    copied = tmp_path / "copied-validation.csv"
    copied.write_bytes(Path(cfg["test"]).read_bytes())
    cfg["validation"] = str(copied)
    path.write_text(yaml.safe_dump(cfg))
    result = api.evaluate(path, create_report=False)
    record = read_record(result["run"])
    calibration = next(m for m in record.metrics if m.metric_id == "joint.c2st_real_vs_real_auc")
    assert calibration.status == "not_applicable"
    assert calibration.notes == "independent_real_splits_required"
    assert record.metadata["provenance"]["identical_real_file_contents"]
