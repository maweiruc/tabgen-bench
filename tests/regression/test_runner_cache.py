import json
from pathlib import Path

import pytest

from tabgen_bench.core.config import FrozenConfig
from tabgen_bench.experiments.aggregation import aggregate
from tabgen_bench.experiments.artifacts import ArtifactConflictError, IntegrityError
from tabgen_bench.experiments.manifest import load_manifest
from tests.support_runner import bundle, changed, for_seed, registry_and_suite, setup


def stage(manifest, name):
    return next(c for c in manifest.cache if c.stage == name)


def test_same_config_hits_sample_and_metric_cache(tmp_path):
    runner, cell, counts = setup(tmp_path)
    first = runner.run(cell)
    assert first.status == "completed"
    before = dict(counts)
    second = runner.run(cell)
    assert first.run_id != second.run_id
    assert second.status == "completed" and counts == before
    assert all(c.hit for c in second.cache)
    assert stage(first, "sample").key == stage(second, "sample").key
    metric_entry = runner.cache.get("metric", stage(second, "metric:fake.future").key)
    assert metric_entry.metadata["origin_run_id"] == first.run_id
    assert (
        metric_entry.metadata["computed_started_at"] <= metric_entry.metadata["computed_ended_at"]
    )
    assert runner.store.directory(first.run_id).joinpath("manifest.json").exists()


def test_validity_rule_change_invalidates_only_metrics_when_not_consumed_by_generator(tmp_path):
    runner, cell, counts = setup(tmp_path)
    first = runner.run(cell)
    updated = bundle(maximum=2000)
    runner.datasets.bundles["toy"] = updated
    second = runner.run(changed(cell, dataset=updated.describe()))
    assert second.status == "completed" and counts["fit"] == counts["sample"] == 1
    assert stage(second, "sample").hit
    assert not any(c.hit for c in second.cache if c.stage.startswith("metric:"))
    assert first.hashes.validity_rules != second.hashes.validity_rules
    assert first.hashes.generation_schema_view == second.hashes.generation_schema_view


def test_rule_change_invalidates_fit_when_consumed_by_generator(tmp_path):
    from tabgen_bench.experiments.contracts import GeneratorSpec

    runner, cell, counts = setup(
        tmp_path, generator=GeneratorSpec(id="bootstrap", consumes_validity_rules=True)
    )
    first = runner.run(cell)
    updated = bundle(maximum=2000)
    runner.datasets.bundles["toy"] = updated
    second = runner.run(changed(cell, dataset=updated.describe()))
    assert first.status == second.status == "completed" and counts["fit"] == 2
    assert not stage(second, "fit").hit and not stage(second, "sample").hit
    assert counts["consumed_rules"] == updated.spec.dataset_schema.validity_rules().fingerprint()


def test_metric_version_change_reuses_samples(tmp_path):
    runner, cell, counts = setup(tmp_path)
    runner.run(cell)
    registry, suite = registry_and_suite(version="2.0.0", calls=counts)
    runner.registry = registry
    second = runner.run(changed(cell, evaluation=suite))
    assert second.status == "completed" and counts["fit"] == 1
    assert stage(second, "sample").hit
    assert not stage(second, "metric:fake.future").hit
    assert stage(second, "metric:validity.schema_pass").hit


def test_generator_param_change_invalidates_samples(tmp_path):
    runner, cell, counts = setup(tmp_path)
    runner.run(cell)
    g = cell.generator.model_copy(update={"params": FrozenConfig.from_mapping({"offset": 1})})
    second = runner.run(changed(cell, generator=g))
    assert second.status == "completed" and counts["fit"] == counts["sample"] == 2
    assert not stage(second, "sample").hit


def test_split_seed_change_invalidates_downstream(tmp_path):
    from tabgen_bench.experiments.planner import seed_spec

    runner, cell, counts = setup(tmp_path)
    runner.run(cell)
    updated = bundle(split_seed=23)
    runner.datasets.bundles["toy"] = updated
    second = runner.run(changed(cell, dataset=updated.describe(), seeds=seed_spec(11, 23)))
    assert second.status == "completed" and counts["fit"] == 2
    assert stage(second, "data").hit and not stage(second, "split").hit
    assert not stage(second, "fit").hit and not stage(second, "sample").hit


def test_corrupted_artifact_rejected(tmp_path):
    runner, cell, _ = setup(tmp_path)
    first = runner.run(cell)
    sample = stage(first, "sample")
    path = runner.cache.directory("sample", sample.key) / "synthetic.parquet"
    path.write_bytes(path.read_bytes() + b"corrupt")
    second = runner.run(cell)
    assert second.status == "failed" and second.error_type == "IntegrityError"
    assert path.read_bytes().endswith(b"corrupt")
    run_path = runner.store.directory(first.run_id) / "synthetic.parquet"
    run_path.write_bytes(b"broken")
    with pytest.raises(IntegrityError):
        load_manifest(runner.store, first.run_id)


def test_resume_after_interruption(tmp_path):
    runner, cell, counts = setup(tmp_path)

    def interrupt(stage_name, run_id):
        if stage_name == "sample":
            raise KeyboardInterrupt("test-only interruption")

    runner.after_stage = interrupt
    with pytest.raises(KeyboardInterrupt):
        runner.run(cell, run_id="interrupted")
    old = (runner.store.directory("interrupted") / "manifest.json").read_bytes()
    assert load_manifest(runner.store, "interrupted").status == "failed"
    runner.after_stage = None
    resumed = runner.resume("interrupted")
    assert resumed.status == "completed" and resumed.run_id != "interrupted"
    assert stage(resumed, "sample").hit and counts["fit"] == counts["sample"] == 1
    assert old == (runner.store.directory("interrupted") / "manifest.json").read_bytes()
    execution = json.loads((runner.store.directory(resumed.run_id) / "execution.json").read_text())
    assert execution["resume_of"] == "interrupted"


def test_failure_visible_in_aggregation(tmp_path):
    runner, cell, counts = setup(tmp_path)
    counts["fail_metric"] = True
    partial = runner.run(cell)
    assert partial.status == "partial_failure"
    counts["fail_fit"] = True
    failed = runner.run(for_seed(cell, 23))
    assert failed.status == "failed"
    group = aggregate(runner.store, [partial.run_id, failed.run_id])["groups"][0]
    assert group["attempt_status_counts"] == {"partial_failure": 1, "failed": 1}
    fake = next(m for m in group["metrics"] if m["metric_id"] == "fake.future")
    assert fake["status_counts"] == {"failed": 1, "missing": 1}
    assert fake["mean"] is None


def test_immutable_run_manifest(tmp_path):
    runner, cell, _ = setup(tmp_path)
    first = runner.run(cell, run_id="frozen")
    path = runner.store.directory(first.run_id) / "manifest.json"
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        runner.run(cell, run_id="frozen")
    with pytest.raises(ArtifactConflictError):
        runner.store.write(first.run_id, "manifest.json", b"overwrite")
    with pytest.raises(ArtifactConflictError, match="sealed"):
        runner.store.write(first.run_id, "new-untracked-artifact.json", b"append")
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="completed"):
        runner.resume(first.run_id)


def test_missing_sample_refits_without_loading_pickle(tmp_path):
    runner, cell, counts = setup(tmp_path)
    first = runner.run(cell)
    seeds = cell.seeds.model_copy(update={"sample_seed": 123})
    second = runner.run(changed(cell, seeds=seeds))
    assert first.status == second.status == "completed"
    assert stage(second, "fit").hit and counts["fit"] == 2
    log = json.loads((runner.store.directory(second.run_id) / "fit_execution.json").read_text())
    assert log["execution"] == "refit_metadata_only_no_safe_model_reload"
    assert not list(Path(tmp_path).rglob("*.pkl"))


def test_report_template_change_reuses_metrics(tmp_path):
    runner, cell, counts = setup(tmp_path)
    runner.run(cell)
    before = dict(counts)
    result = runner.run(changed(cell, report_template_version="report_input_v2"))
    assert result.status == "completed" and before == counts
    assert stage(result, "sample").hit
    assert all(c.hit for c in result.cache if c.stage.startswith("metric:"))
    assert not stage(result, "report_input").hit


@pytest.mark.parametrize("field", ["dependency_lock_hash", "runtime_fingerprint"])
def test_dependency_or_runtime_change_invalidates_fit(tmp_path, field):
    runner, cell, counts = setup(tmp_path)
    runner.run(cell)
    result = runner.run(changed(cell, **{field: "9" * 64}))
    assert result.status == "completed" and counts["fit"] == 2
    assert not stage(result, "fit").hit


def test_future_domain_switch_without_runner_change(tmp_path):
    runner, cell, counts = setup(tmp_path)
    runner.run(cell)
    runner.registry, suite = registry_and_suite(domain="privacy", calls=counts)
    result = runner.run(changed(cell, evaluation=suite))
    assert result.status == "completed" and counts["fit"] == 1
    assert not stage(result, "metric:fake.future").hit


@pytest.mark.parametrize("target", ["terminal", "snapshot"])
def test_manifest_corruption_rejected(tmp_path, target):
    runner, cell, _ = setup(tmp_path)
    run = runner.run(cell)
    folder = runner.store.directory(run.run_id)
    path = (
        folder / "manifest.json"
        if target == "terminal"
        else sorted((folder / "manifests").glob("*.json"))[-1]
    )
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(IntegrityError, match="manifest"):
        load_manifest(runner.store, run.run_id)
