"""Deterministic typed cell expansion, with no training, downloads, or metric dispatch."""

from itertools import product

from tabgen_bench.core.config import FrozenConfig
from tabgen_bench.core.hashing import config_hash
from tabgen_bench.core.manifest import SeedSpec

from .contracts import CellPlan, StageNode
from .keys import generation_keys, metric_key

REPLICATE_SEEDS = (11, 23, 37, 53, 71)


def derive_seed(purpose: str, replicate_seed: int):
    return int(
        config_hash({"policy": "seed_v1", "purpose": purpose, "replicate_seed": replicate_seed})[
            :8
        ],
        16,
    )


def seed_spec(replicate_seed, split_seed):
    evaluation = derive_seed("evaluation", replicate_seed)
    return SeedSpec(
        replicate_seed=replicate_seed,
        split_seed=split_seed,
        fit_seed=derive_seed("fit", replicate_seed),
        sample_seed=derive_seed("sample", replicate_seed),
        evaluation_seed=evaluation,
        reference_sampling_seed=evaluation,
    )


def make_cell(
    dataset,
    generator,
    evaluation,
    seeds,
    *,
    dependency_lock_hash,
    runtime_fingerprint,
    n_synthetic=None,
    sampling_config=None,
    report_template_version="report_input_v1",
):
    if seeds.split_seed != dataset.split_seed:
        raise ValueError("requested split seed differs from the supplied frozen split")
    if seeds.reference_sampling_seed != seeds.evaluation_seed:
        raise ValueError("V1 metrics require the reference/evaluation seed alias")
    n_rows = dataset.split_counts.to_dict()["train"] if n_synthetic is None else n_synthetic
    sampling = sampling_config or FrozenConfig()
    if sampling.to_dict():
        raise ValueError("GeneratorAdapter v1 has no extra sampling parameters")
    keys = generation_keys(
        dataset, generator, seeds, dependency_lock_hash, n_rows, sampling, runtime_fingerprint
    )
    nodes = tuple(
        StageNode(id=s, stage=s, key=keys[s], depends_on=deps)
        for s, deps in (
            ("data", ()),
            ("split", ("data",)),
            ("fit", ("split",)),
            ("sample", ("fit",)),
        )
    )
    cell = CellPlan(
        dataset=dataset,
        generator=generator,
        evaluation=evaluation,
        seeds=seeds,
        dependency_lock_hash=dependency_lock_hash,
        runtime_fingerprint=runtime_fingerprint,
        n_synthetic=n_rows,
        sampling_config=sampling,
        report_template_version=report_template_version,
        nodes=nodes,
    )
    metrics = tuple(
        StageNode(
            id="metric:" + m.id,
            stage="metric",
            key=metric_key(cell, m, keys["sample"]),
            depends_on=("sample",),
        )
        for m in evaluation.metrics
    )
    # The report key is content driven and therefore resolved after metrics finish.
    report = StageNode(
        id="report_input", stage="report_input", key=None, depends_on=tuple(n.id for n in metrics)
    )
    return cell.model_copy(update={"nodes": (*nodes, *metrics, report)})


class Planner:
    def __init__(self, *, dependency_lock_hash, runtime_fingerprint):
        self.dependency_lock_hash = dependency_lock_hash
        self.runtime_fingerprint = runtime_fingerprint

    def expand(self, datasets, generators, evaluation, *, replicate_seeds=REPLICATE_SEEDS):
        if len(set(replicate_seeds)) != len(replicate_seeds):
            raise ValueError("replicate seeds must be unique")
        descriptions = tuple(bundle.describe() for bundle in datasets)
        if len({d.spec.id for d in descriptions}) != len(descriptions):
            raise ValueError("duplicate dataset ID")
        if len({g.id for g in generators}) != len(generators):
            raise ValueError("duplicate generator ID")
        return tuple(
            make_cell(
                d,
                g,
                evaluation,
                seed_spec(s, d.split_seed),
                dependency_lock_hash=self.dependency_lock_hash,
                runtime_fingerprint=self.runtime_fingerprint,
            )
            for d, g, s in product(descriptions, generators, replicate_seeds)
        )
