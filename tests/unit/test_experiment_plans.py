from pathlib import Path

import pytest

from tabgen_bench.core.config import FrozenConfig
from tabgen_bench.core.hashing import config_hash
from tabgen_bench.experiments.contracts import GeneratorSpec, StageNode
from tabgen_bench.experiments.keys import metric_key, report_key
from tabgen_bench.experiments.planner import Planner, derive_seed, make_cell, seed_spec
from tabgen_bench.interfaces.experiments import read_evaluation_spec
from tests.support_runner import bundle, changed, registry_and_suite, setup

ROOT = Path(__file__).resolve().parents[2]


def test_plan_expands_expected_60_cells():
    datasets = [bundle(name) for name in ("adult", "bank_marketing", "credit_g")]
    generators = [
        GeneratorSpec(id=name)
        for name in ("bootstrap", "independent_marginal", "gaussian_copula", "ctgan")
    ]
    _, suite = registry_and_suite()
    planner = Planner(dependency_lock_hash="1" * 64, runtime_fingerprint="2" * 64)
    cells = planner.expand(datasets, generators, suite)
    assert len(cells) == len({c.fingerprint() for c in cells}) == 60
    for name in ("adult", "bank_marketing", "credit_g"):
        selected = [c for c in cells if c.dataset.spec.id == name]
        assert len({c.dataset.split_hash for c in selected}) == 1
        assert {c.seeds.split_seed for c in selected} == {20261025}
        assert {c.seeds.replicate_seed for c in selected} == {11, 23, 37, 53, 71}
        assert all(c.n_synthetic == 72 for c in selected)
    for seed in (11, 23, 37, 53, 71):
        selected = [c for c in cells if c.seeds.replicate_seed == seed]
        assert len({c.seeds.evaluation_seed for c in selected}) == 1
    assert len({derive_seed(p, 11) for p in ("fit", "sample", "evaluation")}) == 3


def test_seed_policy_is_frozen_and_alias_is_explicit():
    expected = int(
        config_hash({"policy": "seed_v1", "purpose": "fit", "replicate_seed": 11})[:8], 16
    )
    assert derive_seed("fit", 11) == expected
    seed = seed_spec(11, 20261025)
    assert seed.reference_sampling_seed == seed.evaluation_seed


def test_typed_dag_rejects_cycle_and_duplicate_nodes(tmp_path):
    _, cell, _ = setup(tmp_path)
    assert cell.nodes[-1].stage == "report_input" and cell.nodes[-1].key is None
    for nodes in (
        (StageNode(id="a", stage="data", key="1" * 64, depends_on=("a",)),),
        (cell.nodes[0], cell.nodes[0]),
    ):
        with pytest.raises(ValueError, match="DAG"):
            type(cell).model_validate({**cell.model_dump(), "nodes": nodes})


def test_generation_schema_change_invalidates_fit(tmp_path):
    _, cell, _ = setup(tmp_path)
    schema = cell.dataset.spec.dataset_schema
    updated_schema = schema.model_copy(
        update={
            "columns": (
                schema.columns[0].model_copy(update={"logical_type": "numeric"}),
                *schema.columns[1:],
            )
        }
    )
    spec = cell.dataset.spec.model_copy(update={"dataset_schema": updated_schema})
    updated = cell.dataset.model_copy(update={"spec": spec})
    new = changed(cell, dataset=updated)
    assert new.nodes[0].key == cell.nodes[0].key
    assert new.nodes[1].key == cell.nodes[1].key
    assert new.nodes[2].key != cell.nodes[2].key
    assert new.nodes[3].key != cell.nodes[3].key


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", "other"),
        ("version", "2.0.0"),
        ("domain", "privacy"),
        ("params", FrozenConfig.from_mapping({"threshold": 2})),
        ("preprocessing_version", "2.0.0"),
    ],
)
def test_every_metric_identity_component_invalidates_key(tmp_path, field, value):
    _, cell, _ = setup(tmp_path)
    metric = cell.evaluation.metrics[-1]
    modified = metric.model_copy(update={field: value})
    sample = cell.nodes[3].key
    assert metric_key(cell, metric, sample) != metric_key(cell, modified, sample)


@pytest.mark.parametrize("field", ["references", "extensions_hash", "content_hash"])
def test_reference_and_extension_hashes_affect_downstream(tmp_path, field):
    _, cell, _ = setup(tmp_path)
    old = metric_key(cell, cell.evaluation.metrics[-1], cell.nodes[3].key)
    value = (
        FrozenConfig.from_mapping({"real_test": "9" * 64}) if field == "references" else "9" * 64
    )
    dataset = cell.dataset.model_copy(update={field: value})
    new = changed(cell, dataset=dataset)
    # content_hash alone is an audit field; canonical_bytes_sha256 drives the data key.
    if field == "content_hash":
        new = changed(cell, dataset=dataset.model_copy(update={"canonical_bytes_sha256": "8" * 64}))
    assert old != metric_key(new, new.evaluation.metrics[-1], new.nodes[3].key)


def test_generator_versions_and_canonicalization_version_change_keys(tmp_path):
    _, cell, _ = setup(tmp_path)
    for field in ("version", "adapter_version"):
        new = changed(cell, generator=cell.generator.model_copy(update={field: "2.0.0"}))
        assert new.nodes[2].key != cell.nodes[2].key
    spec = cell.dataset.spec.model_copy(update={"canonicalization_version": "toy_v2"})
    new = changed(cell, dataset=cell.dataset.model_copy(update={"spec": spec}))
    assert all(new.nodes[i].key != cell.nodes[i].key for i in range(4))


def test_sampling_size_changes_sample_without_fit(tmp_path):
    _, cell, _ = setup(tmp_path)
    new = changed(cell, n_synthetic=5)
    assert new.nodes[2].key == cell.nodes[2].key and new.nodes[3].key != cell.nodes[3].key
    assert report_key("0" * 64, "v1") != report_key("0" * 64, "v2")


def test_rejects_unrecorded_sampler_parameters_and_seed_protocol_change(tmp_path):
    _, cell, _ = setup(tmp_path)
    args = dict(
        dependency_lock_hash=cell.dependency_lock_hash, runtime_fingerprint=cell.runtime_fingerprint
    )
    with pytest.raises(ValueError, match="sampling"):
        make_cell(
            cell.dataset,
            cell.generator,
            cell.evaluation,
            cell.seeds,
            sampling_config=FrozenConfig.from_mapping({"temperature": 1}),
            **args,
        )
    with pytest.raises(ValueError, match="alias"):
        changed(cell, seeds=cell.seeds.model_copy(update={"reference_sampling_seed": 99}))
    with pytest.raises(ValueError, match="frozen split"):
        changed(cell, seeds=cell.seeds.model_copy(update={"split_seed": 99}))


def test_frozen_standard_protocol_lock_matches_actual_spec():
    import json

    import yaml

    lock = json.loads((ROOT / "configs/suites/protocols.lock.json").read_text())
    config = yaml.safe_load((ROOT / "configs/experiments/standard_v1.yaml").read_text())
    assert config_hash(config) == lock["standard_experiment_hash"]
    assert (
        read_evaluation_spec(ROOT / "configs/suites/standard_v1.yaml").fingerprint()
        == lock["standard_evaluation_hash"]
    )


def test_plan_rejects_duplicate_replicate_seeds():
    planner = Planner(dependency_lock_hash="1" * 64, runtime_fingerprint="2" * 64)
    _, suite = registry_and_suite()
    with pytest.raises(ValueError, match="unique"):
        planner.expand([bundle()], [GeneratorSpec(id="bootstrap")], suite, replicate_seeds=(11, 11))
