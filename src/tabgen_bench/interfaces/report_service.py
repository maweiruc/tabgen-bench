"""Composition of verified stored results with the algorithm-free report renderer."""

import csv
import json
from io import StringIO
from pathlib import Path
from uuid import uuid4

from tabgen_bench.core.hashing import canonical_json
from tabgen_bench.core.results import MetricResult
from tabgen_bench.experiments.artifacts import (
    ArtifactStore,
    IntegrityError,
    atomic_write,
    digest,
    json_bytes,
    safe_path,
)
from tabgen_bench.experiments.contracts import CellPlan
from tabgen_bench.experiments.manifest import load_manifest
from tabgen_bench.experiments.runner import semantic_results
from tabgen_bench.reporting.comparison import summarize
from tabgen_bench.reporting.html import TEMPLATE_VERSION, render
from tabgen_bench.reporting.plots import plotting_environment
from tabgen_bench.reporting.records import ReportRecord


def read_record(run):
    folder = Path(run).resolve(strict=True)
    store = ArtifactStore(folder.parent)
    manifest = load_manifest(store, folder.name)
    inventory = {a.path: a for a in manifest.artifacts}

    def read(name, fallback=None):
        item = inventory.get(name)
        return fallback if item is None else store.read(manifest.run_id, item)

    results = tuple(
        MetricResult.model_validate_json(line)
        for line in read("metrics.jsonl", b"").splitlines()
        if line
    )
    ids = [r.metric_id for r in results]
    versions = {v.name: v.version for v in manifest.metric_versions}
    if len(ids) != len(set(ids)) or any(
        (r.run_id, r.dataset_id, r.generator_id, r.metric_version)
        != (manifest.run_id, manifest.dataset_id, manifest.generator_id, versions.get(r.metric_id))
        for r in results
    ):
        raise IntegrityError("MetricResult identity/version differs from manifest")
    if manifest.generator_id is not None and any(
        r.replicate_seed != manifest.seeds.replicate_seed for r in results
    ):
        raise IntegrityError("MetricResult replicate seed differs from manifest")
    if manifest.status in {"completed", "partial_failure"} and set(ids) != set(versions):
        raise IntegrityError("terminal successful evaluation has missing MetricResult records")
    resolved = manifest.resolved_config.to_dict()
    if "cell" in resolved:
        cell = CellPlan.model_validate(resolved["cell"])
        metadata = {
            "dataset": cell.dataset.spec.model_dump(mode="json"),
            "split_counts": cell.dataset.split_counts.to_dict(),
            "provenance": cell.dataset.provenance.to_dict(),
            "evaluation": cell.evaluation.model_dump(mode="json"),
            "real_inputs": cell.dataset.references.to_dict(),
        }
    else:
        metadata = resolved["report_metadata"]
    report_input = json.loads(read("report_input.json", b"{}"))
    if report_input:
        expected = digest(json_bytes(semantic_results(results)))
        if report_input.get("metric_results_sha") != expected:
            raise IntegrityError("report input metric hash differs from MetricResult records")
        if report_input.get("metrics") != semantic_results(results):
            raise IntegrityError("report input contains divergent metric records")
    return ReportRecord(
        manifest,
        results,
        digest(json_bytes(manifest.model_dump(mode="json"))),
        metadata,
        report_input,
        json.loads(read("randomness.json", b"{}")),
    )


def csv_bytes(rows, fields):
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def export_report(records, output):
    aggregation = summarize(records)  # Compatibility is checked before creating any output.
    folder = Path(output).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    inventory = []

    def write(name, payload):
        atomic_write(safe_path(folder, name), payload)
        inventory.append({"path": name, "sha256": digest(payload), "size_bytes": len(payload)})

    metrics = [r.model_dump(mode="json") for record in records for r in record.metrics]
    write("metrics.jsonl", b"".join(json_bytes(row) for row in metrics))
    fields = list(MetricResult.model_fields)
    flat = [
        {k: canonical_json(v) if isinstance(v, (dict, list)) else v for k, v in row.items()}
        for row in metrics
    ]
    write("metrics.csv", csv_bytes(flat, fields))
    summaries = [
        {
            "generator_id": g["generator"]["id"],
            "group_hash": g["group_hash"],
            **m,
            **{k: g[k] for k in ("n_planned", "n_attempted", "n_completed", "n_unique_outputs")},
        }
        for g in aggregation["groups"]
        for m in g["metrics"]
    ]
    summary_fields = list(summaries[0]) if summaries else ["generator_id", "metric_id", "mean"]
    write(
        "summary.csv",
        csv_bytes(
            [
                {k: canonical_json(v) if isinstance(v, (dict, list)) else v for k, v in row.items()}
                for row in summaries
            ],
            summary_fields,
        ),
    )
    write("aggregation.json", json_bytes(aggregation))
    # Source manifests stay unchanged; report files have a separate receipt.
    for record in records:
        write(
            f"sources/{record.manifest.run_id}/manifest.json",
            json_bytes(record.manifest.model_dump(mode="json")),
        )
    if len(records) == 1:
        write("manifest.json", json_bytes(records[0].manifest.model_dump(mode="json")))
    else:
        write(
            "manifest.json",
            json_bytes(
                {
                    "comparison_manifest_version": "1.0.0",
                    "sources": [
                        {"run_id": r.manifest.run_id, "manifest_sha256": r.manifest_sha256}
                        for r in records
                    ],
                }
            ),
        )
    with plotting_environment():
        write("report.html", render(records, aggregation).encode("utf-8"))
    receipt = {
        "report_version": "1.0.0",
        "template_version": TEMPLATE_VERSION,
        "sources": [
            {"run_id": r.manifest.run_id, "manifest_sha256": r.manifest_sha256} for r in records
        ],
        "artifacts": inventory,
    }
    atomic_write(folder / "report_receipt.json", json_bytes(receipt))
    return {
        "output": str(folder),
        "report": str(folder / "report.html"),
        "n_metrics": len(metrics),
        "groups": aggregation["groups"],
    }


def compare(runs, *, output):
    paths = [Path(run).resolve(strict=True) for run in runs]
    destination = Path(output).resolve()
    if any(destination.is_relative_to(path) for path in paths):
        raise ValueError("report output cannot be inside an immutable source run")
    return export_report([read_record(path) for path in paths], destination)


def report(run, *, output=None):
    path = Path(run).resolve(strict=True)
    destination = (
        path.parent / "reports" / f"{path.name}-{uuid4()}" if output is None else Path(output)
    )
    return compare([path], output=destination)
