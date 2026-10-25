import pytest

from tabgen_bench.core.manifest import RunManifest
from tabgen_bench.experiments.aggregation import aggregate
from tabgen_bench.experiments.manifest import load_manifest
from tests.support_runner import changed, setup


def test_core_manifest_format_still_roundtrips_without_migration(tmp_path):
    runner, cell, _ = setup(tmp_path)
    manifest = runner.run(cell)
    assert manifest.manifest_schema_version == "1.0.0"
    assert RunManifest.model_validate_json(manifest.model_dump_json()) == manifest
    assert load_manifest(runner.store, manifest.run_id) == manifest


def test_frozen_metric_spec_requires_matching_actual_plugin(tmp_path):
    runner, cell, _ = setup(tmp_path)
    metric = cell.evaluation.metrics[-1].model_copy(update={"version": "future_not_installed"})
    suite = cell.evaluation.model_copy(update={"metrics": (*cell.evaluation.metrics[:-1], metric)})
    result = runner.run(changed(cell, evaluation=suite))
    assert result.status == "failed" and result.error_type == "ValueError"
    assert "MetricSpec" in result.error_message


def test_partial_retry_retains_failure_and_reuses_successes(tmp_path):
    runner, cell, counts = setup(tmp_path)
    counts["fail_metric"] = True
    failed = runner.run(cell)
    assert failed.status == "partial_failure"
    counts["fail_metric"] = False
    complete = runner.resume(failed.run_id)
    assert complete.status == "completed" and counts["fit"] == 1 and counts["metric"] == 2
    group = aggregate(runner.store, [failed.run_id, complete.run_id])["groups"][0]
    assert group["logical_cell_count"] == 1 and group["attempt_count"] == 2
    metric = next(m for m in group["metrics"] if m["metric_id"] == "fake.future")
    assert metric["status_counts"] == {"ok": 1}
    assert metric["attempt_status_counts"] == {"failed": 1, "ok": 1}


def test_invalid_generated_values_are_not_repaired(tmp_path):
    from tabgen_bench.core.config import FrozenConfig

    runner, cell, _ = setup(tmp_path)
    generator = cell.generator.model_copy(
        update={"params": FrozenConfig.from_mapping({"offset": -1000})}
    )
    result = runner.run(changed(cell, generator=generator))
    assert result.status == "completed"
    import json

    rows = [
        json.loads(line)
        for line in (runner.store.directory(result.run_id) / "metrics.jsonl")
        .read_text()
        .splitlines()
    ]
    assert rows[0]["value"] == 0 and rows[1]["value"] > 0
    assert rows[-1]["status"] == "not_applicable" and rows[-1]["notes"] == "blocked_by_validity"


def test_resume_rejects_corrupted_checkpoint_artifact(tmp_path):
    from tabgen_bench.experiments.artifacts import IntegrityError

    runner, cell, _ = setup(tmp_path)

    def interrupt(stage, run_id):
        if stage == "sample":
            raise KeyboardInterrupt()

    runner.after_stage = interrupt
    with pytest.raises(KeyboardInterrupt):
        runner.run(cell, run_id="interrupted-corrupted")
    folder = runner.store.directory("interrupted-corrupted")
    (folder / "splits.json").write_bytes(b"broken")
    with pytest.raises(IntegrityError):
        runner.resume("interrupted-corrupted")


def test_warning_without_numeric_value_is_not_fabricated_in_aggregation(tmp_path):
    from dataclasses import dataclass

    from tabgen_bench.metrics.registry import PluginRegistry
    from tests.support_runner import FakeMetric

    runner, cell, _ = setup(tmp_path)
    original = runner.registry.get("fake.future")

    @dataclass(frozen=True)
    class WarningMetric(FakeMetric):
        def compute(self, context, *, seed):
            result = super().compute(context, seed=seed)
            return result.model_copy(
                update={
                    "value": None,
                    "status": "warning",
                    "notes": "test-only unavailable warning",
                }
            )

    warning = WarningMetric(
        id=original.id,
        domain=original.domain,
        version=original.version,
        direction=original.direction,
        unit=original.unit,
        required_inputs=original.required_inputs,
    )
    runner.registry = PluginRegistry(
        [runner.registry.get(m.id) for m in cell.evaluation.metrics[:-1]] + [warning]
    )
    result = runner.run(cell)
    metric = aggregate(runner.store, [result.run_id])["groups"][0]["metrics"][-1]
    assert metric["status_counts"] == {"warning": 1} and metric["mean"] is None
