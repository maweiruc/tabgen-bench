import json

import pandas as pd
import pytest

from tabgen_bench.core.results import MetricResult
from tabgen_bench.evaluation.result_writer import write_results
from tests.unit.test_results import record


def test_jsonl_parquet_preserve_failed_na_diagnostics_and_identity(tmp_path):
    results = (
        record(details={"column_reasons": [{"column": "age", "count": 1}]}),
        record(value=None, status="failed", error_type="test", error_message="failure"),
        record(value=None, status="not_applicable", notes="missing_input"),
    )
    a, b = tmp_path / "metrics.jsonl", tmp_path / "metrics.parquet"
    write_results(results, jsonl_path=a, parquet_path=b)
    loaded = tuple(MetricResult.model_validate_json(line) for line in a.read_text().splitlines())
    assert loaded == results
    parquet = pd.read_parquet(b)
    assert list(parquet.status) == ["ok", "failed", "not_applicable"]
    assert json.loads(parquet.iloc[0].details_json)["column_reasons"][0]["count"] == 1
    with pytest.raises(FileExistsError):
        write_results(results, jsonl_path=a, parquet_path=b)
