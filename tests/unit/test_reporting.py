import csv
import json
from dataclasses import replace
from html.parser import HTMLParser
from pathlib import Path

import pytest

from tabgen_bench.core.config import FrozenConfig
from tabgen_bench.interfaces.api import report
from tabgen_bench.interfaces.report_service import read_record
from tabgen_bench.reporting.comparison import summarize
from tabgen_bench.reporting.html import render
from tabgen_bench.reporting.plots import plotting_environment
from tabgen_bench.reporting.records import IncompatibleRunsError
from tests.support_runner import for_seed, setup
from tests.support_workflows import evaluated


class Parsed(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.tags, self.metrics, self.images = [], {}, []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append(tag)
        if tag == "tr" and "data-metric" in attrs:
            self.metrics[(attrs["data-run"], attrs["data-metric"])] = attrs["data-value"]
        if tag == "img":
            self.images.append(attrs["src"])
        assert not any(k.lower().startswith("on") for k in attrs)


@pytest.fixture(scope="module")
def valid_run(tmp_path_factory):
    return evaluated(tmp_path_factory.mktemp("m4-valid"))


@pytest.fixture(scope="module")
def invalid_run(tmp_path_factory):
    return evaluated(tmp_path_factory.mktemp("m4-invalid"), invalid=True)


def rendered(record):
    with plotting_environment():
        return render([record], summarize([record]))


def test_html_validity_reason_breakdown(invalid_run):
    record = read_record(invalid_run)
    text = rendered(record)
    assert "out_of_range" in text and "artificial_example_dictionary" in text
    assert "Validity reason breakdown" in text and "blocked_by_validity" in text
    results = {r.metric_id: r for r in record.metrics}
    assert results["validity.invalid_cell_rate"].value == 1 / (96 * 3)
    assert results["validity.invalid_row_rate"].value == 1 / 96
    parsed = Parsed(text)
    assert float(parsed.metrics[(record.manifest.run_id, "validity.invalid_cell_rate")]) == 1 / 288


def test_html_no_fake_future_metric_scores(valid_run):
    text = rendered(read_record(valid_run))
    parsed = Parsed(text)
    assert "Diversity" in text and "Statistical" in text and "Formal Privacy Attacks" in text
    assert text.count("Not included in this suite") == 2
    assert all(
        not metric.startswith(("statistical.", "diversity.", "privacy.attack"))
        for _, metric in parsed.metrics
    )
    assert "not a privacy guarantee" in text


def test_report_generation_offline(valid_run, tmp_path):
    output = report(valid_run, output=tmp_path / "offline")
    folder = Path(output["output"])
    parsed = Parsed((folder / "report.html").read_text())
    assert "real_validation/validation" in (folder / "report.html").read_text()
    assert "real_test/test" in (folder / "report.html").read_text()
    assert parsed.images and all(src.startswith("data:image/png;base64,") for src in parsed.images)
    assert not {"script", "link", "iframe", "object", "form"} & set(parsed.tags)
    assert {
        "manifest.json",
        "metrics.csv",
        "metrics.jsonl",
        "summary.csv",
        "aggregation.json",
        "report.html",
        "report_receipt.json",
    } <= {p.name for p in folder.iterdir()}
    record = read_record(valid_run)
    csv_rows = list(csv.DictReader((folder / "metrics.csv").open()))
    json_rows = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
    for r, csv_row, json_row in zip(record.metrics, csv_rows, json_rows, strict=True):
        assert float(parsed.metrics[(r.run_id, r.metric_id)]) == r.value
        assert float(csv_row["value"]) == r.value == json_row["value"]
        assert json.loads(csv_row["details"]) == r.details.to_dict()
    assert (folder / "manifest.json").read_bytes() == (valid_run / "manifest.json").read_bytes()


def test_report_does_not_run_evaluation(valid_run, tmp_path, monkeypatch):
    import pandas as pd

    from tabgen_bench.evaluation.engine import EvaluationEngine
    from tabgen_bench.generators.bootstrap import BootstrapGenerator
    from tabgen_bench.metrics.base import Metric

    def forbidden(*a, **k):
        raise AssertionError("report attempted data parsing / computation")

    for owner, name in (
        (EvaluationEngine, "evaluate"),
        (BootstrapGenerator, "fit"),
        (BootstrapGenerator, "sample"),
        (Metric, "compute"),
        (pd, "read_parquet"),
        (pd, "read_csv"),
    ):
        monkeypatch.setattr(owner, name, forbidden)
    before = {p.relative_to(valid_run): p.read_bytes() for p in valid_run.rglob("*") if p.is_file()}
    report(valid_run, output=tmp_path / "redrawn")
    after = {p.relative_to(valid_run): p.read_bytes() for p in valid_run.rglob("*") if p.is_file()}
    assert before == after
    with pytest.raises(ValueError, match="immutable"):
        report(valid_run, output=valid_run / "new-report")


def test_html_escaping(valid_run):
    record = read_record(valid_run)
    attack = "\"><script>alert('x')</script><img src=x onerror=alert(1)> & $bad_{"
    manifest = record.manifest.model_copy(
        update={
            "generator_params": FrozenConfig.from_mapping({"user": attack}),
            "error_message": attack,
        }
    )
    metrics = tuple(r.model_copy(update={"notes": attack}) for r in record.metrics)
    record = replace(record, manifest=manifest, metrics=metrics)
    text = rendered(record)
    assert "<script>" not in text and "<img src=x" not in text
    assert "&lt;script&gt;" in text and "&amp;" in text
    assert "script" not in Parsed(text).tags


@pytest.mark.parametrize(
    "field",
    ["validity_rules", "suite_id", "schema", "split", "data", "metric_versions", "evaluation_spec"],
)
def test_compare_rejects_incompatible_rules_or_suites(valid_run, field):
    record = read_record(valid_run)
    second = replace(record, manifest=record.manifest.model_copy(update={"run_id": "different"}))
    if field in {"suite_id", "metric_versions"}:
        value = "other_suite" if field == "suite_id" else ()
        second = replace(second, manifest=second.manifest.model_copy(update={field: value}))
    elif field == "evaluation_spec":
        second = replace(second, metadata={**second.metadata, "evaluation": {"version": "changed"}})
    else:
        hash_field = "schema_hash" if field == "schema" else field
        hashes = second.manifest.hashes.model_copy(update={hash_field: "f" * 64})
        second = replace(second, manifest=second.manifest.model_copy(update={"hashes": hashes}))
    with pytest.raises(IncompatibleRunsError, match=field):
        summarize([record, second])


def test_failed_na_warning_not_excluded(invalid_run):
    record = read_record(invalid_run)
    values = list(record.metrics)
    values[-1] = values[-1].model_copy(
        update={
            "status": "failed",
            "error_type": "ExplicitError",
            "error_message": "failure retained",
            "value": None,
        }
    )
    values[0] = values[0].model_copy(update={"status": "warning", "notes": "warning retained"})
    record = replace(record, metrics=tuple(values))
    summary = summarize([record])
    text = rendered(record)
    parsed = Parsed(text)
    assert len(parsed.metrics) == 17
    assert "not_applicable" in text and "failure retained" in text and "warning retained" in text
    assert any(m["status_counts"].get("failed") for m in summary["groups"][0]["metrics"])
    assert any(m["status_counts"].get("not_applicable") for m in summary["groups"][0]["metrics"])


def test_html_duplicate_audit_and_retry_counts(tmp_path):
    runner, cell, counts = setup(tmp_path, counts={"fit": 0, "sample": 0, "ignore_seed": True})
    manifests = [runner.run(for_seed(cell, seed)) for seed in (11, 23, 37, 53, 71)]
    replay = runner.run(cell)
    records = [read_record(runner.store.directory(m.run_id)) for m in [*manifests, replay]]
    group = summarize(records)["groups"][0]
    assert [group[k] for k in ("n_planned", "n_attempted", "n_completed", "n_unique_outputs")] == [
        5,
        6,
        6,
        1,
    ]
    assert group["n_independent_replicates"] is None
    with plotting_environment():
        text = render(records, summarize(records))
    assert "Repeated synthetic outputs detected" in text and "unconfirmed (null)" in text
    assert "n_independent_replicates" in text and "n_unique_outputs" in text
    assert all(m["std"] is None for m in group["metrics"])


@pytest.mark.parametrize("domain", ["statistical", "privacy"])
def test_html_future_plugin_uses_generic_result_interface(tmp_path, domain):
    from tests.support_runner import registry_and_suite

    registry, suite = registry_and_suite(domain=domain)
    runner, cell, _ = setup(tmp_path, registry=registry, suite=suite)
    manifest = runner.run(cell)
    text = rendered(read_record(runner.store.directory(manifest.run_id)))
    assert "fake.future" in text and domain.title() in text
    assert "data-metric='fake.future'" in text


def test_failed_run_missing_metrics_visible(tmp_path):
    runner, cell, _ = setup(tmp_path, counts={"fit": 0, "sample": 0, "fail_fit": True})
    manifest = runner.run(cell)
    record = read_record(runner.store.directory(manifest.run_id))
    assert manifest.status == "failed" and not record.metrics
    group = summarize([record])["groups"][0]
    assert group["n_completed"] == 0 and group["n_attempted"] == 1
    assert all(m["status_counts"] == {"missing": 1} for m in group["metrics"])
    text = rendered(record)
    assert "test-only fit failure" in text and "missing" in text
    assert "Included in this suite; no persisted MetricResult" in text


def test_report_input_mismatch_rejected(valid_run, tmp_path):
    import shutil

    from tabgen_bench.experiments.artifacts import IntegrityError

    target = tmp_path / valid_run.name
    shutil.copytree(valid_run, target)
    path = target / "report_input.json"
    data = json.loads(path.read_text())
    data["metric_results_sha"] = "0" * 64
    path.write_text(json.dumps(data))
    with pytest.raises(IntegrityError, match="hash/size"):
        report(target, output=tmp_path / "bad-report")
    assert not (tmp_path / "bad-report").exists()


def test_html_adult_overlap_metadata_retained(tmp_path):
    from tabgen_bench.datasets.provenance import load_public_metadata

    provenance = {"row_overlap": load_public_metadata("data-quality.json")["adult"]}
    runner, cell, _ = setup(tmp_path)
    manifest = runner.run(cell)
    record = read_record(runner.store.directory(manifest.run_id))
    record = replace(record, metadata={**record.metadata, "provenance": provenance})
    text = rendered(record)
    assert "cross-split" in text and "Frozen row positions retained" in text
    assert "52" in text and "11" in text and "10" in text
    assert provenance["row_overlap"] == load_public_metadata("data-quality.json")["adult"]


def test_external_input_identity_and_labels_not_silently_merged(valid_run):
    record = read_record(valid_run)
    second = replace(
        record,
        manifest=record.manifest.model_copy(update={"run_id": "external-two"}),
        audit={
            **record.audit,
            "synthetic_content_sha256": "b" * 64,
            "synthetic_multiset_sha256": "c" * 64,
        },
    )
    group = summarize([record, second])["groups"][0]
    assert group["n_planned"] == group["n_unique_outputs"] == 2
    second = replace(second, metadata={**second.metadata, "synthetic_label": "user-model"})
    assert {g["generator"]["id"] for g in summarize([record, second])["groups"]} == {
        "external",
        "user-model",
    }
