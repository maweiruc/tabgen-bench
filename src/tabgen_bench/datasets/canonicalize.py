"""Versioned source mapping, without fitting rules on validation/test observations."""

import hashlib
from io import BytesIO

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tabgen_bench.core.hashing import canonical_json
from tabgen_bench.core.schema import DatasetSpec


def canonicalize(data: pd.DataFrame, target: pd.Series, config: dict, spec: DatasetSpec):
    mapping = config["mapping"]
    if list(data.columns) != list(mapping["features"]):
        raise ValueError("OpenML feature width/order/names differ from the frozen mapping")
    if target.name != mapping["raw_target"] or not data.index.equals(target.index):
        raise ValueError("OpenML target identity/alignment mismatch")
    if len(data) != spec.expected_rows:
        raise ValueError("OpenML row count mismatch")
    if target.isna().any():
        raise ValueError("missing target is not supported")
    frame = data.copy(deep=True)
    frame.columns = list(mapping["features"].values())
    frame[spec.target.column] = target.copy()
    frame = frame.reset_index(drop=True)
    for column in spec.dataset_schema.columns:
        series = frame[column.name]
        values = mapping.get("values", {}).get(column.name)
        if values is not None:
            # Codes are explicitly declared nominal strings, never arbitrary numeric coercion.
            keys = series.astype("string")
            if not set(keys.dropna().unique()) <= set(values):
                raise ValueError(f"unrecognized source category code: {column.name}")
            series = keys.map(values)
        if column.logical_type == "integer":
            if not pd.api.types.is_numeric_dtype(series):
                raise ValueError(f"expected numeric source storage: {column.name}")
            numbers = series.dropna().to_numpy(dtype=float)
            if not np.isfinite(numbers).all() or not (numbers == np.floor(numbers)).all():
                raise ValueError(f"nonintegral source value: {column.name}")
            series = series.astype("Int64")
        elif column.logical_type == "categorical":
            series = series.astype("string")
        else:
            raise ValueError(f"unsupported registry logical type: {column.logical_type}")
        if not column.nullable and series.isna().any():
            raise ValueError(f"unexpected missing source value: {column.name}")
        if column.allowed_values is not None:
            if not set(series.dropna().unique()) <= set(column.allowed_values):
                raise ValueError(f"value outside declared source dictionary: {column.name}")
        frame[column.name] = series
    if list(frame.columns) != [c.name for c in spec.dataset_schema.columns]:
        raise ValueError("canonical column order differs from declared schema")
    if set(frame[spec.target.column].unique()) != {
        spec.target.positive_label,
        spec.target.negative_label,
    }:
        raise ValueError("target labels differ from the declared binary encoding")
    return frame


def parquet_bytes(frame: pd.DataFrame) -> bytes:
    # No pandas metadata/index or machine-dependent paths/timestamps in the content lock.
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(None)
    output = BytesIO()
    pq.write_table(
        table,
        output,
        version="2.6",
        compression="NONE",
        use_dictionary=False,
        write_statistics=False,
        row_group_size=65536,
        data_page_version="1.0",
    )
    return output.getvalue()


def normalized_content_hash(frame: pd.DataFrame) -> str:
    """Semantic rows hash is independent of Parquet writer versions and category dictionaries."""
    digest = hashlib.sha256()
    digest.update((canonical_json(list(frame.columns)) + "\n").encode())
    for row in frame.itertuples(index=False, name=None):
        values = [None if pd.isna(v) else v.item() if isinstance(v, np.generic) else v for v in row]
        digest.update((canonical_json(values) + "\n").encode())
    return digest.hexdigest()
