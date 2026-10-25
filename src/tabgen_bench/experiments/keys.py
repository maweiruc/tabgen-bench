"""Versioned pure cache keys; no IO, metric formula, or generator implementation."""

from tabgen_bench.core.context import InputIdentity
from tabgen_bench.core.hashing import config_hash

KEY_VERSION = "1.0.0"


def key(stage, **inputs):
    return config_hash({"cache_key_version": KEY_VERSION, "stage": stage, **inputs})


def generation_keys(
    dataset, generator, seeds, dependency_lock_hash, n_rows, sampling_config, runtime_fingerprint
):
    data = key(
        "data",
        canonical_bytes=dataset.canonical_bytes_sha256,
        canonicalization_version=dataset.spec.canonicalization_version,
    )
    split = key(
        "split",
        data=data,
        split=dataset.split_hash,
        seed=seeds.split_seed,
        splitter_version=dataset.splitter_version,
        config={"fractions": [0.6, 0.2, 0.2]},
    )
    fit = key(
        "fit",
        split=split,
        generation_schema=dataset.spec.dataset_schema.generation_view().fingerprint(),
        generator_id=generator.id,
        generator_version=generator.version,
        adapter_version=generator.adapter_version,
        hyperparams=generator.params.to_dict(),
        fit_seed=seeds.fit_seed,
        dependency_lock=dependency_lock_hash,
        runtime=runtime_fingerprint,
        consumed_generation_constraints_hash=(
            dataset.spec.dataset_schema.validity_rules().fingerprint()
            if generator.consumes_validity_rules
            else None
        ),
    )
    sample = key(
        "sample",
        fit=fit,
        sample_seed=seeds.sample_seed,
        n_rows=n_rows,
        sampling_config=sampling_config.to_dict(),
    )
    return {"data": data, "split": split, "fit": fit, "sample": sample}


def metric_key(cell, metric, sample_key):
    schema = cell.dataset.spec.dataset_schema
    return key(
        "metric",
        sample_key=sample_key,
        real_reference_hashes=cell.dataset.references.to_dict(),
        real_reference_identities=[i.model_dump(mode="json") for i in real_input_identities(cell)],
        source=cell.dataset.spec.source.model_dump(mode="json"),
        target=None if cell.dataset.spec.target is None else cell.dataset.spec.target.model_dump(),
        generation_schema_view_hash=schema.generation_view().fingerprint(),
        validity_rules_hash=schema.validity_rules().fingerprint(),
        metric_id=metric.id,
        metric_version=metric.version,
        metric_domain=metric.domain,
        metric_params=metric.params.to_dict(),
        eval_seed=cell.seeds.evaluation_seed,
        reference_sampling_seed=cell.seeds.reference_sampling_seed,
        preprocessing_version=metric.preprocessing_version,
        extension_fingerprint=cell.dataset.extensions_hash,
        policy=cell.evaluation.policy.fingerprint(),
        suite_id=cell.evaluation.id,
        suite_version=cell.evaluation.version,
        dependency_lock=cell.dependency_lock_hash,
        runtime=cell.runtime_fingerprint,
    )


def real_input_identities(cell):
    split_key = next(n.key for n in cell.nodes if n.id == "split")
    source = cell.dataset.spec.source
    return tuple(
        InputIdentity(
            input_name=name,
            source_id=f"{source.provider}:{source.source_id}",
            split_id=f"{split_key}:{name}",
            reference_split=name.removeprefix("real_"),
            content_hash=content,
        )
        for name, content in cell.dataset.references.to_dict().items()
    )


def report_key(metric_results_sha, report_template_version):
    return key(
        "report_input",
        metric_results_sha=metric_results_sha,
        report_template_version=report_template_version,
    )
