"""Descriptive, duplicate-aware aggregation of verified persisted results; no metric algorithms."""

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from tabgen_bench.core.hashing import config_hash
from tabgen_bench.core.results import MetricResult

from .artifacts import IntegrityError, atomic_write, digest, json_bytes
from .contracts import CellPlan
from .manifest import load_manifest


def _artifact(store, manifest, name):
    found = next((a for a in manifest.artifacts if a.path == name), None)
    return None if found is None else store.read(manifest.run_id, found)


def aggregate(store, run_ids):
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("duplicate run IDs are not additional attempts")
    groups = defaultdict(list)
    for run_id in run_ids:
        manifest = load_manifest(store, run_id)
        cell = CellPlan.model_validate(manifest.resolved_config.to_dict()["cell"])
        compatibility = {
            "dataset_id": manifest.dataset_id,
            "dataset_version": manifest.dataset_version,
            "hashes": manifest.hashes.model_dump(mode="json"),
            "source": manifest.source.model_dump(mode="json"),
            "generator": cell.generator.model_dump(mode="json"),
            "evaluation": cell.evaluation.model_dump(mode="json"),
            "runtime_fingerprint": cell.runtime_fingerprint,
            "n_synthetic": cell.n_synthetic,
            "sampling": cell.sampling_config.to_dict(),
        }
        group = config_hash(compatibility)
        # Rendering or storage choices do not create additional scientific replicates.
        scientific_cell = config_hash({"group": group, "seeds": manifest.seeds.model_dump()})
        raw = _artifact(store, manifest, "metrics.jsonl")
        metrics = (
            []
            if raw is None
            else [MetricResult.model_validate_json(line) for line in raw.splitlines() if line]
        )
        if any(
            (r.run_id, r.dataset_id, r.generator_id, r.replicate_seed)
            != (run_id, manifest.dataset_id, manifest.generator_id, manifest.seeds.replicate_seed)
            for r in metrics
        ):
            raise IntegrityError("persisted metric identity differs from run manifest")
        raw_audit = _artifact(store, manifest, "randomness.json")
        audit = None if raw_audit is None else json.loads(raw_audit)
        groups[group].append(
            {
                "manifest": manifest,
                "cell": cell,
                "scientific_cell": scientific_cell,
                "metrics": metrics,
                "audit": audit,
                "compatibility": compatibility,
            }
        )
    output = []
    for group, attempts in sorted(groups.items()):
        latest = {}
        for attempt in sorted(
            attempts, key=lambda a: (a["manifest"].created_at, a["manifest"].run_id)
        ):
            latest[attempt["scientific_cell"]] = attempt
        records = list(latest.values())
        units = defaultdict(list)
        for record in records:
            if record["audit"] is not None:
                units[record["audit"]["synthetic_multiset_sha256"]].append(record)
        duplicates = []
        for content, members in sorted(units.items()):
            if len(members) > 1:
                duplicates.append(
                    {
                        "synthetic_multiset_sha256": content,
                        "run_ids": sorted(r["manifest"].run_id for r in members),
                        "replicate_seeds": sorted(
                            {r["manifest"].seeds.replicate_seed for r in members}
                        ),
                        "seed_tuples": [r["manifest"].seeds.model_dump() for r in members],
                        "duplicate_across_seeds": len(
                            {
                                (r["manifest"].seeds.fit_seed, r["manifest"].seeds.sample_seed)
                                for r in members
                            }
                        )
                        > 1,
                    }
                )
        metric_rows = []
        for spec in attempts[0]["cell"].evaluation.metrics:
            status = Counter()
            attempt_status = Counter()
            for record in attempts:
                matching = [m for m in record["metrics"] if m.metric_id == spec.id]
                attempt_status[matching[0].status if matching else "missing"] += 1
            for record in records:
                matching = [m for m in record["metrics"] if m.metric_id == spec.id]
                status[matching[0].status if matching else "missing"] += 1
            values = []
            for members in units.values():
                usable = [
                    m.value
                    for r in members
                    for m in r["metrics"]
                    if m.metric_id == spec.id
                    and m.status in {"ok", "warning"}
                    and m.value is not None
                ]
                if usable:
                    # One unit per output dataset, with evaluation variation kept within it.
                    values.append(statistics.fmean(usable))
            metric_rows.append(
                {
                    "metric_id": spec.id,
                    "metric_domain": spec.domain,
                    "metric_version": spec.version,
                    "status_counts": dict(status),
                    "attempt_status_counts": dict(attempt_status),
                    "n_unique_output_units": len(values),
                    "mean": statistics.fmean(values) if values else None,
                    "std": statistics.stdev(values) if len(values) > 1 else None,
                    "summary_kind": "descriptive_equal_weight_per_unique_synthetic_dataset",
                    "monte_carlo_independence_assumed": False,
                    "n_independent_replicates": None,
                }
            )
        output.append(
            {
                "group_hash": group,
                "compatibility": attempts[0]["compatibility"],
                "attempt_count": len(attempts),
                "logical_cell_count": len(records),
                "attempt_status_counts": dict(Counter(a["manifest"].status for a in attempts)),
                "cell_status_counts": dict(Counter(a["manifest"].status for a in records)),
                "run_ids": sorted(a["manifest"].run_id for a in attempts),
                "latest_cell_run_ids": sorted(r["manifest"].run_id for r in records),
                "n_unique_synthetic_datasets": len(units),
                "duplicate_outputs": duplicates,
                "sample_seed_independence_assumed": False,
                "n_independent_replicates": None,
                "real_split_replicates": 1,
                "metrics": metric_rows,
                "dataset_provenance": attempts[0]["cell"].dataset.provenance.to_dict(),
            }
        )
    return {
        "aggregation_version": "1.0.0",
        "independence_policy_version": "1.0.0",
        "groups": output,
    }


def write_aggregation(store, run_ids, output):
    result = aggregate(store, run_ids)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    payload = json_bytes(result)
    atomic_write(root / "aggregation.json", payload)
    atomic_write(root / "aggregation.sha256", (digest(payload) + "\n").encode())
    return result
