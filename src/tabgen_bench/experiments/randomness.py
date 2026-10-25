"""Exact output audits. Different seeds/hashes alone do not establish statistical independence."""

import hashlib

import numpy as np
import pandas as pd

from tabgen_bench.core.hashing import canonical_json
from tabgen_bench.datasets.canonicalize import normalized_content_hash


def multiset_hash(frame, schema=None):
    rows = []
    logical = None if schema is None else [c.logical_type for c in schema.columns]
    for row in frame.itertuples(index=False, name=None):
        values = [None if pd.isna(v) else v.item() if isinstance(v, np.generic) else v for v in row]
        if logical is not None and len(logical) == len(values):
            values = [
                int(value)
                if kind in {"numeric", "integer"}
                and isinstance(value, float)
                and value.is_integer()
                else value
                for kind, value in zip(logical, values, strict=True)
            ]
        rows.append(canonical_json(values))
    data = canonical_json(list(frame.columns)) + "\n" + "\n".join(sorted(rows)) + "\n"
    return hashlib.sha256(data.encode()).hexdigest()


def audit_sample(frame, cell, environment):
    return {
        "audit_version": "1.0.0",
        "seeds": cell.seeds.model_dump(),
        "synthetic_content_sha256": normalized_content_hash(frame),
        "synthetic_multiset_sha256": multiset_hash(frame, cell.dataset.spec.dataset_schema),
        "rows": len(frame),
        "environment": environment,
        "sample_seed_independence_assumed": False,
        "monte_carlo_independence": "not_established",
        "real_split_replicates": 1,
        "reference_sampling_seed_semantics": "alias_of_evaluation_seed_for_M1_metrics",
    }
