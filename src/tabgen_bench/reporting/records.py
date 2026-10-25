"""Normalized, serialized report inputs. No data tables or execution dependencies."""

from dataclasses import dataclass, field

from tabgen_bench.core.manifest import RunManifest
from tabgen_bench.core.results import MetricResult


@dataclass(frozen=True)
class ReportRecord:
    manifest: RunManifest
    metrics: tuple[MetricResult, ...]
    manifest_sha256: str
    metadata: dict = field(default_factory=dict)
    report_input: dict = field(default_factory=dict)
    audit: dict = field(default_factory=dict)


class IncompatibleRunsError(ValueError):
    code = "incompatible_runs"


def compatibility(record):
    m = record.manifest
    return {
        "dataset_id": m.dataset_id,
        "dataset_version": m.dataset_version,
        "source": m.source.model_dump(mode="json"),
        "data": m.hashes.data,
        "split": m.hashes.split,
        "schema": m.hashes.schema_hash,
        "generation_schema": m.hashes.generation_schema_view,
        "validity_rules": m.hashes.validity_rules,
        "validity_rules_version": m.hashes.validity_rules_version,
        "suite_id": m.suite_id,
        "suite_version": m.suite_version,
        "metric_versions": sorted((v.name, v.version) for v in m.metric_versions),
        "evaluation_spec": record.metadata["evaluation"],
        "real_inputs": record.metadata["real_inputs"],
        "n_synthetic": m.n_synthetic,
        "dependency_lock": m.hashes.dependency_lock,
    }


def require_compatible(records):
    if not records:
        raise ValueError("at least one run is required")
    identities = [r.manifest.run_id for r in records]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate run IDs are not additional attempts")
    expected = compatibility(records[0])
    errors = []
    signatures = {}
    for record in records:
        actual = compatibility(record)
        differing = [name for name in expected if expected[name] != actual[name]]
        if differing:
            errors.append(f"{record.manifest.run_id}: {', '.join(differing)}")
        for metric in record.metrics:
            signature = (
                metric.metric_version,
                metric.metric_domain,
                metric.direction,
                metric.unit,
                metric.method_config_hash,
            )
            if metric.metric_id in signatures and signatures[metric.metric_id] != signature:
                errors.append(f"{record.manifest.run_id}: metric signature {metric.metric_id}")
            signatures[metric.metric_id] = signature
    if errors:
        raise IncompatibleRunsError("incompatible runs; " + "; ".join(errors))
    return expected
