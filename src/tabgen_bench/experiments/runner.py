"""Sequential stage orchestration through injected ports; no generator or metric algorithms."""

import json
from io import BytesIO
from uuid import uuid4

import pandas as pd

from tabgen_bench.core.context import EvaluationContext, InputIdentity
from tabgen_bench.core.hashing import config_hash
from tabgen_bench.datasets.canonicalize import normalized_content_hash
from tabgen_bench.evaluation.engine import EvaluationEngine

from .artifacts import IntegrityError, digest, json_bytes
from .contracts import CellPlan
from .keys import real_input_identities, report_key
from .manifest import ManifestJournal, load_manifest, utc_now, validate_checkpoints
from .metric_cache import evaluate_cached
from .randomness import audit_sample


def table_bytes(frame):
    output = BytesIO()
    frame.to_parquet(output, index=False)
    return output.getvalue()


def semantic_results(results):
    return [
        {
            **r.model_dump(mode="json"),
            "run_id": "unbound",
            "replicate_seed": None,
            "started_at": None,
            "ended_at": None,
        }
        for r in results
    ]


class Runner:
    def __init__(
        self,
        *,
        datasets,
        generators,
        metric_registry,
        cache,
        store,
        engine_factory=EvaluationEngine,
        command=(),
        git_root=".",
        after_stage=None,
    ):
        self.datasets, self.generators, self.registry = datasets, generators, metric_registry
        self.cache, self.store, self.engine_factory = cache, store, engine_factory
        self.command, self.git_root, self.after_stage = command, git_root, after_stage

    def resume(self, run_id):
        previous = load_manifest(self.store, run_id)
        if previous.status == "completed":
            raise ValueError("completed run is immutable; run the same plan to create a new record")
        validate_checkpoints(self.store, previous, self.cache)
        cell = CellPlan.model_validate(previous.resolved_config.to_dict()["cell"])
        return self.run(cell, resume_of=run_id)

    def run(self, cell: CellPlan, *, run_id=None, resume_of=None):
        journal = ManifestJournal(
            self.store,
            run_id or str(uuid4()),
            cell,
            command=self.command,
            git_root=self.git_root,
            resume_of=resume_of,
        )
        nodes = {n.id: n for n in cell.nodes}
        active = "data"
        last_boundary = utc_now().isoformat()

        def recorded(stage, key, hit, **kwargs):
            nonlocal last_boundary
            metadata = kwargs.setdefault("metadata", {})
            metadata.setdefault("started_at", last_boundary)
            journal.stage(stage, key, hit, **kwargs)
            last_boundary = utc_now().isoformat()
            if self.after_stage is not None:
                self.after_stage(stage, journal.run_id)

        def cached_stage(name, payloads, metadata):
            entry = self.cache.get(name, nodes[name].key)
            hit = entry is not None
            if not hit:
                entry = self.cache.publish(
                    name,
                    nodes[name].key,
                    payloads(),
                    metadata={**metadata, "origin_run_id": journal.run_id},
                )
            return entry, hit

        try:
            bundle = self.datasets.get(cell.dataset.spec.id)
            if bundle.describe() != cell.dataset:
                raise IntegrityError("dataset/schema/split/provenance differs from planned input")
            frame = bundle.frame
            started = utc_now().isoformat()
            data, hit = cached_stage(
                "data",
                lambda: {"data.parquet": table_bytes(frame)},
                {"content_hash": cell.dataset.content_hash},
            )
            frame = pd.read_parquet(BytesIO(data.read("data.parquet")))
            if normalized_content_hash(frame) != cell.dataset.content_hash:
                raise IntegrityError("cached dataset semantic hash mismatch")
            recorded("data", data.key, hit, entry=data, metadata={"started_at": started})
            active = "split"
            split, hit = cached_stage(
                "split",
                lambda: {"splits.json": json_bytes(bundle.split.to_dict())},
                {"split_hash": cell.dataset.split_hash},
            )
            if json.loads(split.read("splits.json")) != bundle.split.to_dict():
                raise IntegrityError("cached split differs from frozen row positions")
            journal.write("splits.json", split.read("splits.json"))
            recorded("split", split.key, hit, entry=split)
            train, validation, test = bundle.split.frames(frame)

            active = "sample"
            sample_entry = self.cache.get("sample", nodes["sample"].key)
            active = "fit"
            fit_entry = self.cache.get("fit", nodes["fit"].key)
            fit_hit = fit_entry is not None
            fitted = None
            started = utc_now().isoformat()
            if sample_entry is None:
                fitted = self.generators(cell.generator)
                if (fitted.id, fitted.version) != (cell.generator.id, cell.generator.version):
                    raise ValueError("generator factory differs from frozen GeneratorSpec")
                view = cell.dataset.spec.dataset_schema.generation_view()
                if cell.generator.consumes_validity_rules:
                    consume = getattr(fitted, "fit_with_constraints", None)
                    if consume is None:
                        raise ValueError(
                            "declared constraint consumer has no explicit fitting port"
                        )
                    consume(
                        train.copy(deep=True),
                        view,
                        cell.dataset.spec.dataset_schema.validity_rules(),
                        seed=cell.seeds.fit_seed,
                    )
                else:
                    fitted.fit(train.copy(deep=True), view, seed=cell.seeds.fit_seed)
                environment = dict(fitted.environment_info())
                fit_metadata = {
                    "origin_run_id": journal.run_id,
                    "fit_started_at": started,
                    "fit_ended_at": utc_now().isoformat(),
                    "fit_seed": cell.seeds.fit_seed,
                    "environment": environment,
                    "train_rows": len(train),
                    "generation_schema_hash": view.fingerprint(),
                    "consumed_validity_rules_hash": (
                        cell.dataset.spec.dataset_schema.validity_rules().fingerprint()
                        if cell.generator.consumes_validity_rules
                        else None
                    ),
                }
                fit_execution = "refit_metadata_only_no_safe_model_reload" if fit_hit else "fit"
            else:
                fit_metadata = json.loads(sample_entry.read("generation.json"))["fit_metadata"]
                fit_execution = "skipped_sample_cache_hit"
            if fit_entry is None:
                fit_entry = self.cache.publish(
                    "fit",
                    nodes["fit"].key,
                    {"fit.json": json_bytes(fit_metadata)},
                    metadata={
                        "model_serialized": False,
                        "origin_run_id": fit_metadata["origin_run_id"],
                    },
                )
            recorded(
                "fit",
                fit_entry.key,
                fit_hit,
                entry=fit_entry,
                metadata={
                    "started_at": started,
                    "execution": fit_execution,
                    "model_cache_hit": False,
                    "model_serialized": False,
                },
            )
            journal.write(
                "fit_execution.json",
                json_bytes(
                    {
                        "execution": fit_execution,
                        "fit_metadata": fit_metadata,
                        "fit_metadata_cache_hit": fit_hit,
                    }
                ),
            )

            active = "sample"
            sample_hit = sample_entry is not None
            started = utc_now().isoformat()
            if not sample_hit:
                synthetic = fitted.sample(cell.n_synthetic, seed=cell.seeds.sample_seed)
                if not isinstance(synthetic, pd.DataFrame):
                    raise TypeError("generator must return a DataFrame")
                sample_entry = self.cache.publish(
                    "sample",
                    nodes["sample"].key,
                    {
                        "synthetic.parquet": table_bytes(synthetic),
                        "generation.json": json_bytes(
                            {
                                "fit_metadata": fit_metadata,
                                "environment": dict(fitted.environment_info()),
                                "content_hash": normalized_content_hash(synthetic),
                                "origin_run_id": journal.run_id,
                                "sample_started_at": started,
                                "sample_ended_at": utc_now().isoformat(),
                            }
                        ),
                    },
                    metadata={"origin_run_id": journal.run_id},
                )
            generation = json.loads(sample_entry.read("generation.json"))
            model_environment = generation["environment"]
            journal.current = journal.current.model_copy(
                update={
                    "environment": journal.current.environment.model_copy(
                        update={
                            "actual_device": model_environment.get(
                                "device", "unspecified_by_adapter"
                            ),
                            "gpu": model_environment.get("gpu"),
                            "reproducibility_level": model_environment.get(
                                "reproducibility_level",
                                journal.current.environment.reproducibility_level,
                            ),
                        }
                    )
                }
            )
            synthetic = pd.read_parquet(BytesIO(sample_entry.read("synthetic.parquet")))
            if normalized_content_hash(synthetic) != generation["content_hash"]:
                raise IntegrityError("synthetic semantic hash mismatch")
            journal.write("synthetic.parquet", sample_entry.read("synthetic.parquet"))
            audit = audit_sample(synthetic, cell, generation["environment"])
            journal.write("randomness.json", json_bytes(audit))
            journal.current = journal.current.model_copy(update={"n_synthetic": len(synthetic)})
            recorded(
                "sample",
                sample_entry.key,
                sample_hit,
                entry=sample_entry,
                metadata={"started_at": started, **audit},
            )

            active = "metric"
            tables = {
                "real_train": train,
                "real_validation": validation,
                "real_test": test,
                "synthetic": synthetic,
            }
            identities = (
                *real_input_identities(cell),
                InputIdentity(
                    input_name="synthetic",
                    source_id=cell.generator.id,
                    split_id=nodes["sample"].key,
                    reference_split="synthetic",
                    content_hash=normalized_content_hash(synthetic),
                ),
            )
            context = EvaluationContext(
                schema=cell.dataset.spec.dataset_schema,
                target=cell.dataset.spec.target,
                expected_n_rows=cell.n_synthetic,
                **tables,
                identities=identities,
                extensions=bundle.extensions,
            )
            results = evaluate_cached(
                self.registry,
                context,
                cell,
                self.cache,
                recorded,
                journal.run_id,
                self.engine_factory,
            )
            journal.write(
                "metrics.jsonl", b"".join(json_bytes(r.model_dump(mode="json")) for r in results)
            )
            rows = []
            for result in results:
                row = result.model_dump(mode="json")
                row["details_json"] = json_bytes(row.pop("details")).decode().strip()
                row["references_json"] = json_bytes(row.pop("references")).decode().strip()
                rows.append(row)
            journal.write("metrics.parquet", table_bytes(pd.DataFrame(rows)))

            active = "report_input"
            semantic = semantic_results(results)
            result_hash = digest(json_bytes(semantic))
            report_cache_key = report_key(result_hash, cell.report_template_version)
            report = self.cache.get("report_input", report_cache_key)
            hit = report is not None
            if report is None:
                report = self.cache.publish(
                    "report_input",
                    report_cache_key,
                    {
                        "report_input.json": json_bytes(
                            {
                                "report_input_version": "1.0.0",
                                "template_version": cell.report_template_version,
                                "metric_results_sha": result_hash,
                                "metrics": semantic,
                                "independence_assumed": False,
                            }
                        )
                    },
                )
            journal.write("report_input.json", report.read("report_input.json"))
            recorded("report_input", report_cache_key, hit, entry=report)
            fraction = sum(r.status in {"ok", "warning"} for r in results) / len(results)
            return journal.finish(
                "partial_failure" if any(r.status == "failed" for r in results) else "completed",
                metric_success_fraction=fraction,
            )
        except BaseException as error:
            node = nodes.get(active)
            failed_key = (
                node.key
                if node is not None and node.key is not None
                else config_hash({"stage": active, "cell": cell.fingerprint()})
            )
            journal.stage(
                active,
                failed_key,
                False,
                metadata={
                    "started_at": last_boundary,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error_message": str(error) or repr(error),
                },
            )
            failed = journal.finish("failed", error=error)
            if not isinstance(error, Exception):
                raise
            return failed
