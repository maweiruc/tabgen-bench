"""Descriptive comparison of persisted MetricResults, with explicit duplicate units."""

import statistics
from collections import Counter, defaultdict

from tabgen_bench.core.hashing import config_hash

from .records import require_compatible


def summarize(records):
    """No evaluation: only summarize numbers already in standard records."""
    compatible = require_compatible(records)
    groups = defaultdict(list)
    for record in records:
        m = record.manifest
        generator = {
            "id": m.generator_id or record.metadata.get("synthetic_label", "external"),
            "version": m.generator_version,
            "params": m.generator_params.to_dict(),
        }
        groups[config_hash(generator)].append((generator, record))
    output = []
    for group_hash, members in sorted(groups.items()):
        generator = members[0][0]
        attempts = [r for _, r in members]
        latest = {}
        for r in sorted(attempts, key=lambda r: (r.manifest.created_at, r.manifest.run_id)):
            # Evaluation-only files with identical seeds and inputs are one logical cell too.
            identity = config_hash(
                {
                    "seeds": r.manifest.seeds.model_dump(),
                    "group": group_hash,
                    "external_input": (
                        r.audit.get("synthetic_content_sha256")
                        if r.manifest.generator_id is None
                        else None
                    ),
                }
            )
            latest[identity] = r
        cells = list(latest.values())
        units = defaultdict(list)
        unaudited_outputs = 0
        for r in cells:
            content = r.audit.get("synthetic_multiset_sha256")
            if content is not None:
                units[content].append(r)
            elif r.audit:
                unaudited_outputs += 1
        metric_ids = [v.name for v in attempts[0].manifest.metric_versions]
        summaries = []
        for metric_id in metric_ids:
            all_results = [m for r in attempts for m in r.metrics if m.metric_id == metric_id]
            first = all_results[0] if all_results else None
            values = []
            for unit in units.values():
                numeric = [
                    m.value
                    for r in unit
                    for m in r.metrics
                    if m.metric_id == metric_id
                    and m.status in {"ok", "warning"}
                    and m.value is not None
                ]
                if numeric:
                    values.append(statistics.fmean(numeric))
            statuses = Counter(m.status for m in all_results)
            if len(all_results) < len(attempts):
                statuses["missing"] = len(attempts) - len(all_results)
            summaries.append(
                {
                    "metric_id": metric_id,
                    "metric_domain": None if first is None else first.metric_domain,
                    "metric_version": None if first is None else first.metric_version,
                    "direction": None if first is None else first.direction,
                    "unit": None if first is None else first.unit,
                    "mean": statistics.fmean(values) if values else None,
                    "std": statistics.stdev(values) if len(values) > 1 else None,
                    "n_unique_output_units": len(values),
                    "status_counts": dict(statuses),
                    "n_independent_replicates": None,
                    "summary_kind": "descriptive_equal_weight_per_unique_synthetic_dataset",
                }
            )
        duplicates = [
            {
                "synthetic_multiset_sha256": content,
                "run_ids": [r.manifest.run_id for r in unit],
                "replicate_seeds": [r.manifest.seeds.replicate_seed for r in unit],
            }
            for content, unit in sorted(units.items())
            if len(unit) > 1
        ]
        output.append(
            {
                "group_hash": group_hash,
                "generator": generator,
                "n_planned": len(cells),
                "n_attempted": len(attempts),
                "n_completed": sum(r.manifest.status == "completed" for r in attempts),
                "n_unique_outputs": None if unaudited_outputs else len(units),
                "n_outputs_without_multiset_audit": unaudited_outputs,
                "n_independent_replicates": None,
                "planning_scope": (
                    "logical cells covered by selected runs; not the full experiment plan"
                ),
                "attempt_status_counts": dict(Counter(r.manifest.status for r in attempts)),
                "cell_status_counts": dict(Counter(r.manifest.status for r in cells)),
                "run_ids": [r.manifest.run_id for r in attempts],
                "latest_cell_run_ids": [r.manifest.run_id for r in cells],
                "duplicate_outputs": duplicates,
                "metrics": summaries,
                "real_split_replicates": 1,
                "monte_carlo_independence_assumed": False,
            }
        )
    return {"comparison_version": "1.0.0", "compatibility": compatible, "groups": output}
