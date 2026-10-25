"""External Evaluation Only execution and immutable persistence, shared by CLI/API."""

from collections import Counter
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tabgen_bench.core.config import FrozenConfig
from tabgen_bench.core.context import EvaluationContext, InputIdentity
from tabgen_bench.core.errors import ConfigurationError
from tabgen_bench.core.hashing import canonical_json, config_hash
from tabgen_bench.core.manifest import ContractHashes, RunManifest, SeedSpec, VersionInfo
from tabgen_bench.core.schema import DatasetSpec
from tabgen_bench.evaluation.engine import EvaluationEngine
from tabgen_bench.experiments.artifacts import ArtifactStore, atomic_write, digest, json_bytes
from tabgen_bench.experiments.manifest import environment_info, git_info, utc_now
from tabgen_bench.experiments.randomness import multiset_hash
from tabgen_bench.experiments.runner import semantic_results

from .configuration import EvaluationConfig, read_table, read_yaml, relative
from .dependencies import dependency_fingerprint
from .experiments import read_evaluation_spec


def table_summary(frame, spec):
    """Execution-time aggregate counts for report input; never serialize full records."""
    missing = {
        str(name): int(frame.iloc[:, i].isna().sum()) for i, name in enumerate(frame.columns)
    }
    distributions = {}
    # Bad structure remains untouched; Validity records it. Summaries must not block evaluation.
    if list(frame.columns) == [c.name for c in spec.dataset_schema.columns]:
        for column in spec.dataset_schema.columns:
            values = frame[column.name].dropna()
            if column.logical_type in {"numeric", "integer"}:
                values = [
                    v
                    for v in values
                    if isinstance(v, (int, float, np.number))
                    and not isinstance(v, (bool, np.bool_))
                    and np.isfinite(v)
                ]
                if values:
                    counts, edges = np.histogram(values, bins=10)
                    distributions[column.name] = {
                        f"[{edges[i]}, {edges[i + 1]}{']' if i == 9 else ')'}": int(n)
                        for i, n in enumerate(counts)
                    }
            else:
                counts = Counter(
                    canonical_json(v.item() if isinstance(v, np.generic) else v) for v in values
                )
                top = counts.most_common(20)
                distributions[column.name] = dict(top)
                rest = sum(counts.values()) - sum(n for _, n in top)
                if rest:
                    distributions[column.name]["OTHER (display only)"] = rest
    target = (
        {}
        if spec.target is None or spec.target.column not in frame.columns
        else dict(
            Counter(
                canonical_json(v.item() if isinstance(v, np.generic) else v)
                for v in frame[spec.target.column].dropna()
            )
        )
    )
    return {
        "n_rows": len(frame),
        "missing_counts": missing,
        "target_counts": target,
        "distributions": distributions,
    }


def report_summary(frame, spec):
    try:
        return table_summary(frame, spec)
    except (TypeError, ValueError, OverflowError) as error:
        return {
            "n_rows": len(frame),
            "missing_counts": {},
            "target_counts": {},
            "distributions": {},
            "warning": f"summary unavailable: {type(error).__name__}",
        }


def evaluate(config, *, metric_registry=None, command=(), create_report=True):
    from .api import build_metric_registry

    path = Path(config).resolve(strict=True)
    cfg = EvaluationConfig.model_validate(read_yaml(path))
    base = path.parent
    schema_path = relative(base, cfg.schema_path)
    spec = DatasetSpec.model_validate(read_yaml(schema_path))
    suite = read_evaluation_spec(relative(base, cfg.evaluation_spec))
    registry = build_metric_registry() if metric_registry is None else metric_registry
    for metric in suite.metrics:
        plugin = registry.get(metric.id)
        if (plugin.version, plugin.domain, getattr(plugin, "params", FrozenConfig())) != (
            metric.version,
            metric.domain,
            metric.params,
        ) or metric.preprocessing_version != "1.0.0":
            raise ValueError(f"registry differs from EvaluationSpec: {metric.id}")
    tables, identities, paths, hashes = {}, [], {}, {}
    for alias, name in (
        ("train", "real_train"),
        ("validation", "real_validation"),
        ("test", "real_test"),
        ("reference", "real_reference"),
        ("synthetic", "synthetic"),
    ):
        filename = getattr(cfg, alias)
        if filename is None:
            continue
        table_path = relative(base, filename)
        tables[name] = read_table(table_path)
        # External byte identities can represent illegal scalars (Inf etc.) without repairing them.
        hashes[name] = digest(table_path.read_bytes())
        paths[name] = {"path": str(table_path), "file_sha256": hashes[name]}
        identities.append(
            InputIdentity(
                input_name=name,
                source_id=f"{spec.source.provider}:{spec.source.source_id}",
                split_id=config_hash({"external_file_content": hashes[name]}),
                reference_split=alias,
                content_hash=hashes[name],
            )
        )
    real_paths = [value["path"] for name, value in paths.items() if name != "synthetic"]
    if len(real_paths) != len(set(real_paths)):
        raise ConfigurationError(
            "same real file assigned to multiple splits; independence not established"
        )
    context = EvaluationContext(
        schema=spec.dataset_schema,
        target=spec.target,
        **tables,
        identities=tuple(identities),
        expected_n_rows=cfg.expected_n_rows,
    )
    real_hashes = {k: v for k, v in hashes.items() if k != "synthetic"}
    metadata = {
        "synthetic_label": cfg.synthetic_label,
        "dataset": spec.model_dump(mode="json"),
        "evaluation": suite.model_dump(mode="json"),
        "split_counts": {k: len(v) for k, v in tables.items() if k != "synthetic"},
        "real_inputs": real_hashes,
        "provenance": {
            "split_definition": "user-supplied role tables; subject independence not asserted",
            "identical_real_file_contents": len(real_hashes) != len(set(real_hashes.values())),
        },
    }
    resolved = FrozenConfig.from_mapping(
        {
            "workflow": "evaluation_only",
            "report_metadata": metadata,
            "config": cfg.model_dump(mode="json", by_alias=True),
            "input_files": paths,
        }
    )
    output = relative(base, cfg.output)
    # Exact output directory is supplied by the caller; a second invocation never overwrites it.
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation run: {output}")
    created = utc_now()
    commit, dirty = git_info(base)
    manifest = RunManifest(
        run_id=output.name,
        status="pending",
        created_at=created,
        started_at=None,
        command=tuple(command),
        git_commit=commit,
        git_dirty=dirty,
        environment=environment_info(),
        dataset_id=spec.id,
        dataset_version=spec.version,
        source=spec.source,
        hashes=ContractHashes(
            data=config_hash(real_hashes),
            split=config_hash({"roles": real_hashes, "split_seed": cfg.split_seed}),
            schema_hash=spec.dataset_schema.fingerprint(),
            generation_schema_view=spec.dataset_schema.generation_view().fingerprint(),
            validity_rules=spec.dataset_schema.validity_rules().fingerprint(),
            validity_rules_version=spec.dataset_schema.constraint_rules_version,
            dependency_lock=dependency_fingerprint(
                relative(base, cfg.dependency_lock) if cfg.dependency_lock else None
            ),
        ),
        seeds=SeedSpec(
            replicate_seed=cfg.seed,
            split_seed=cfg.split_seed,
            fit_seed=cfg.seed,
            sample_seed=cfg.seed,
            evaluation_seed=cfg.seed,
            reference_sampling_seed=cfg.seed,
        ),
        generator_id=None,
        generator_version=None,
        suite_id=suite.id,
        suite_version=suite.version,
        metric_versions=tuple(VersionInfo(name=m.id, version=m.version) for m in suite.metrics),
        resolved_config=resolved,
        config_hash=resolved.fingerprint(),
        n_synthetic=len(tables["synthetic"]),
    )
    store = ArtifactStore(output.parent)
    store.create(output.name)
    artifacts = []

    def write(name, payload):
        artifacts.append(store.write(output.name, name, payload))

    def snapshot(number):
        payload = json_bytes(manifest.model_dump(mode="json"))
        atomic_write(output / f"manifests/{number:06d}.json", payload)
        atomic_write(output / f"manifests/{number:06d}.sha256", (digest(payload) + "\n").encode())
        return payload

    snapshot(0)
    manifest = manifest.model_copy(update={"status": "running", "started_at": utc_now()})
    snapshot(1)
    error = None
    results = ()
    try:
        write("config.resolved.yaml", yaml.safe_dump(resolved.to_dict(), sort_keys=True).encode())
        results = EvaluationEngine(
            registry, policy=suite.policy, run_id=output.name, dataset_id=spec.id
        ).evaluate(context, metric_ids=tuple(m.id for m in suite.metrics), seed=cfg.seed)
        write("metrics.jsonl", b"".join(json_bytes(r.model_dump(mode="json")) for r in results))
        rows = []
        for r in results:
            row = r.model_dump(mode="json")
            row["details_json"] = canonical_json(row.pop("details"))
            row["references_json"] = canonical_json(row.pop("references"))
            rows.append(row)
        buffer = BytesIO()
        pd.DataFrame(rows).to_parquet(buffer, index=False)
        write("metrics.parquet", buffer.getvalue())
        semantic = semantic_results(results)
        write(
            "report_input.json",
            json_bytes(
                {
                    "report_input_version": "1.0.0",
                    "template_version": "report_input_v1",
                    "metric_results_sha": digest(json_bytes(semantic)),
                    "metrics": semantic,
                    "independence_assumed": False,
                    "table_summaries": {
                        name: report_summary(frame, spec) for name, frame in tables.items()
                    },
                }
            ),
        )
        try:
            multiset = multiset_hash(tables["synthetic"], spec.dataset_schema)
            audit_note = None
        except (ValueError, TypeError):
            multiset = None
            audit_note = "multiset audit unavailable for unsupported/illegal scalars; no repair"
        write(
            "randomness.json",
            json_bytes(
                {
                    "audit_version": "1.0.0",
                    "synthetic_content_sha256": hashes["synthetic"],
                    "synthetic_multiset_sha256": multiset,
                    "notes": audit_note,
                    "monte_carlo_independence": "not_established",
                    "seed_note": (
                        "external input; fit/sample seeds unused; not generator seed claims"
                    ),
                }
            ),
        )
    except BaseException as exc:
        error = exc
    write(
        "logs.jsonl",
        json_bytes(
            {
                "stage": "evaluation",
                "run_id": output.name,
                "error_type": None if error is None else type(error).__name__,
                "error_message": None if error is None else str(error),
            }
        ),
    )
    manifest = RunManifest.model_validate(
        {
            **manifest.model_dump(),
            "artifacts": artifacts,
            "ended_at": utc_now(),
            "status": "failed"
            if error
            else ("partial_failure" if any(r.status == "failed" for r in results) else "completed"),
            "error_type": None if error is None else type(error).__name__,
            "error_message": None if error is None else str(error) or repr(error),
            "metric_success_fraction": (
                sum(r.status in {"ok", "warning"} for r in results) / len(results)
                if results
                else None
            ),
        }
    )
    atomic_write(output / "manifest.json", snapshot(2))
    if error is not None and not isinstance(error, Exception):
        raise error
    response = {
        "run": str(output),
        "run_id": output.name,
        "status": manifest.status,
        "metric_status_counts": dict(Counter(r.status for r in results)),
        "error_type": manifest.error_type,
        "error_message": manifest.error_message,
    }
    if create_report:
        from .report_service import report

        response["report"] = report(output)
    return response
