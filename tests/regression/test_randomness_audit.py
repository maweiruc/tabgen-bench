from tabgen_bench.experiments.aggregation import aggregate
from tabgen_bench.experiments.randomness import multiset_hash
from tests.support_runner import changed, for_seed, setup


def test_duplicate_synthetic_across_seeds_detected(tmp_path):
    runner, cell, counts = setup(tmp_path)
    counts["ignore_seed"] = True
    runs = [runner.run(for_seed(cell, seed)) for seed in (11, 23, 37)]
    assert all(r.status == "completed" for r in runs)
    group = aggregate(runner.store, [r.run_id for r in runs])["groups"][0]
    assert group["n_unique_synthetic_datasets"] == 1
    assert group["duplicate_outputs"][0]["replicate_seeds"] == [11, 23, 37]
    assert group["duplicate_outputs"][0]["duplicate_across_seeds"]


def test_sample_seed_not_assumed_independent(tmp_path):
    runner, cell, counts = setup(tmp_path)
    counts["ignore_seed"] = True
    other = changed(
        cell, seeds=cell.seeds.model_copy(update={"sample_seed": 123, "replicate_seed": 23})
    )
    runs = [runner.run(c) for c in (cell, other)]
    group = aggregate(runner.store, [r.run_id for r in runs])["groups"][0]
    assert not group["sample_seed_independence_assumed"]
    assert group["n_independent_replicates"] is None
    assert group["duplicate_outputs"][0]["duplicate_across_seeds"]


def test_duplicate_replicates_not_counted_as_independent(tmp_path):
    runner, cell, counts = setup(tmp_path)
    counts["ignore_seed"] = True
    runs = [runner.run(for_seed(cell, seed)) for seed in (11, 23, 37, 53, 71)]
    group = aggregate(runner.store, [r.run_id for r in runs])["groups"][0]
    assert group["logical_cell_count"] == group["cell_status_counts"]["completed"] == 5
    metric = next(m for m in group["metrics"] if m["metric_id"] == "fake.future")
    assert metric["status_counts"] == {"ok": 5}
    assert metric["n_unique_output_units"] == 1 and metric["std"] is None
    assert metric["n_independent_replicates"] is None
    assert metric["mean"] == 1


def test_replay_and_resume_do_not_add_replicates(tmp_path):
    runner, cell, _ = setup(tmp_path)
    runs = [runner.run(cell) for _ in range(2)]
    group = aggregate(runner.store, [r.run_id for r in runs])["groups"][0]
    assert group["attempt_count"] == 2 and group["logical_cell_count"] == 1
    assert group["n_unique_synthetic_datasets"] == 1


def test_multiset_hash_detects_permuted_duplicates_and_preserves_multiplicity():
    import pandas as pd

    frame = pd.DataFrame({"x": [1, 2, 1], "group": [None, "nan", None]})
    assert multiset_hash(frame) == multiset_hash(frame.iloc[::-1])
    assert multiset_hash(frame) != multiset_hash(frame.iloc[:2])
    assert multiset_hash(frame) != multiset_hash(frame.fillna("nan"))


def test_duplicate_audit_respects_numeric_semantics_without_coercing_categories():
    import pandas as pd

    from tabgen_bench.core.schema import ColumnSchema, DatasetSchema

    numeric = DatasetSchema(
        columns=(ColumnSchema(name="x", role="feature", logical_type="numeric", nullable=False),)
    )
    category = DatasetSchema(
        columns=(
            ColumnSchema(name="x", role="feature", logical_type="categorical", nullable=False),
        )
    )
    ints, floats = pd.DataFrame({"x": [1, 2]}), pd.DataFrame({"x": [1.0, 2.0]})
    assert multiset_hash(ints, numeric) == multiset_hash(floats, numeric)
    assert multiset_hash(ints, category) != multiset_hash(floats, category)
