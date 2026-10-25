"""Experiment execution service over stable plans and ports."""

from collections import Counter
from pathlib import Path
from uuid import uuid4

from tabgen_bench.core.config import FrozenConfig
from tabgen_bench.core.schema import DatasetSpec
from tabgen_bench.datasets.splits import DatasetSplit
from tabgen_bench.experiments.artifacts import ArtifactStore, atomic_write, json_bytes
from tabgen_bench.experiments.cache import Cache
from tabgen_bench.experiments.contracts import DatasetBundle
from tabgen_bench.experiments.runner import Runner

from .configuration import ExecutionConfig, Storage, read_table, read_yaml, relative
from .experiments import LocalDatasets, generator_factory, plan_from_file


class FileDatasets:
    """Explicit external frozen inputs, including the shipped artificial CLI example."""

    def __init__(self, definitions, base):
        self.loaded = {}
        for entry in definitions:
            if set(entry) != {"schema", "data", "split"}:
                raise ValueError("local dataset requires exactly schema/data/split")
            spec = DatasetSpec.model_validate(read_yaml(relative(base, entry["schema"])))
            if spec.source.provider == "openml":
                raise ValueError("pinned OpenML data must use LocalDatasets and its content lock")
            if spec.id in self.loaded:
                raise ValueError("duplicate local dataset ID")
            self.loaded[spec.id] = DatasetBundle(
                spec,
                read_table(relative(base, entry["data"])),
                DatasetSplit.load(relative(base, entry["split"])),
                provenance=FrozenConfig.from_mapping({"external_frozen_positions": True}),
            )

    def get(self, name):
        return self.loaded[name]


def run(
    config,
    *,
    dry_run=False,
    project_root=None,
    data_root=None,
    run_root=None,
    cache_root=None,
    datasets=None,
    generators=None,
    metric_registry=None,
    command=(),
    create_reports=True,
):
    from .api import build_metric_registry

    path = Path(config).resolve(strict=True)
    document = read_yaml(path)
    local = []
    if "plan_config" in document:
        cfg = ExecutionConfig.model_validate(document)
        base = path.parent
        project = relative(base, cfg.project_root)
        plan = relative(base, cfg.plan_config)
        data = relative(base, cfg.data_root)
        quality = relative(base, cfg.quality_metadata) if cfg.quality_metadata else None
        dependency_lock = relative(base, cfg.dependency_lock) if cfg.dependency_lock else None
        runs = relative(base, cfg.storage.root)
        cache_path = relative(base, cfg.storage.cache_root)
        local = cfg.local_datasets
    else:
        project = Path(project_root or Path.cwd()).resolve()
        plan, data = path, project / "data"
        quality = None
        dependency_lock = None
        runs, cache_path = project / "runs", project / "cache"
        if "storage" in document:
            storage_cfg = Storage.model_validate(document["storage"])
            runs = relative(path.parent, storage_cfg.root)
            cache_path = relative(path.parent, storage_cfg.cache_root)
    data = Path(data_root).resolve() if data_root is not None else data
    runs = Path(run_root).resolve() if run_root is not None else runs
    cache_path = Path(cache_root).resolve() if cache_root is not None else cache_path
    if project_root is not None:
        project = Path(project_root).resolve()
    provider = datasets
    if provider is None:
        provider = (
            FileDatasets(local, path.parent)
            if local
            else LocalDatasets(registry_root=project, data_root=data, quality_metadata_path=quality)
        )
    cells = plan_from_file(
        plan, project_root=project, datasets=provider, dependency_lock_path=dependency_lock
    )
    storage = {
        "project_root": str(project),
        "data_root": str(data),
        "run_root": str(runs),
        "cache_root": str(cache_path),
        "plan_config": str(plan),
        "dependency_lock": str(dependency_lock) if dependency_lock is not None else None,
    }
    if dry_run:
        # Cache constructor creates directories. Do not construct it when no cache exists.
        cache = Cache(cache_path) if cache_path.is_dir() else None
        return {
            "dry_run": True,
            "n_planned": len(cells),
            "n_attempted": 0,
            "storage": storage,
            "cells": [
                {
                    "dataset_id": c.dataset.spec.id,
                    "generator_id": c.generator.id,
                    "replicate_seed": c.seeds.replicate_seed,
                    "split_seed": c.seeds.split_seed,
                    "n_synthetic": c.n_synthetic,
                    "cache": [
                        {
                            "stage": n.stage,
                            "key": n.key,
                            "hit": (
                                cache.get(n.stage, n.key) is not None
                                if cache is not None and n.key is not None
                                else False
                            ),
                        }
                        for n in c.nodes
                    ],
                }
                for c in cells
            ],
        }
    registry = build_metric_registry() if metric_registry is None else metric_registry
    store = ArtifactStore(runs)
    cache = Cache(cache_path)
    batch = runs / "batches" / str(uuid4())
    batch.mkdir(parents=True, exist_ok=False)
    atomic_write(
        batch / "plan.json",
        json_bytes(
            {
                "batch_version": "1.0.0",
                "n_planned": len(cells),
                "storage": storage,
                "cells": [c.model_dump(mode="json") for c in cells],
            }
        ),
    )
    runner = Runner(
        datasets=provider,
        generators=generators or generator_factory,
        metric_registry=registry,
        cache=cache,
        store=store,
        command=command,
        git_root=project,
    )
    attempted = []
    for index, cell in enumerate(cells):
        manifest = runner.run(cell)
        result = {
            "run": str(store.directory(manifest.run_id)),
            "run_id": manifest.run_id,
            "dataset_id": manifest.dataset_id,
            "generator_id": manifest.generator_id,
            "replicate_seed": manifest.seeds.replicate_seed,
            "status": manifest.status,
            "error_type": manifest.error_type,
            "error_message": manifest.error_message,
        }
        attempted.append(result)
        atomic_write(batch / f"attempts/{index:06d}.json", json_bytes(result))
        if create_reports:
            from .report_service import report

            result["report"] = report(result["run"])
    summary = {
        "dry_run": False,
        "batch": str(batch),
        "n_planned": len(cells),
        "n_attempted": len(attempted),
        "n_completed": sum(r["status"] == "completed" for r in attempted),
        "status_counts": dict(Counter(r["status"] for r in attempted)),
        "runs": attempted,
        "storage": storage,
    }
    atomic_write(batch / "summary.json", json_bytes(summary))
    return summary
