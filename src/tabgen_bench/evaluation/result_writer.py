"""Lossless standardized result exports."""

from pathlib import Path

import pandas as pd

from tabgen_bench.core.hashing import canonical_json
from tabgen_bench.core.results import MetricResult


def write_results(
    results: tuple[MetricResult, ...], *, jsonl_path: Path, parquet_path: Path
) -> None:
    if jsonl_path == parquet_path or jsonl_path.exists() or parquet_path.exists():
        raise FileExistsError("result outputs must be new, distinct paths")
    rows = [result.model_dump(mode="json") for result in results]
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("x") as stream:
        for row in rows:
            stream.write(canonical_json(row) + "\n")
    flat = [
        {
            **{key: value for key, value in row.items() if key not in {"details", "references"}},
            "details_json": canonical_json(row["details"]),
            "references_json": canonical_json(row["references"]),
        }
        for row in rows
    ]
    pd.DataFrame(flat).to_parquet(parquet_path, index=False)
