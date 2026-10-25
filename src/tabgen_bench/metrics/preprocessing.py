"""Train-fitted sklearn preprocessing; source frames are never changed."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tabgen_bench.core.schema import DatasetSchema

from .data import missing, numeric, token


def model_frame(frame: pd.DataFrame, schema: DatasetSchema, columns: tuple[str, ...]):
    result = pd.DataFrame(index=frame.index)
    for column in schema.columns:
        if column.name not in columns:
            continue
        values = frame[column.name]
        if column.logical_type in {"numeric", "integer"}:
            converted = []
            for value in values:
                if missing(value):
                    converted.append(np.nan)
                elif numeric(value):
                    converted.append(float(value))
                else:
                    raise ValueError(f"non-numeric model input: {column.name}")
            result[column.name] = converted
        else:
            result[column.name] = [token(v) for v in values]
    return result


def preprocessor(schema: DatasetSchema, columns: tuple[str, ...]) -> ColumnTransformer:
    numbers = [
        c.name
        for c in schema.columns
        if c.name in columns and c.logical_type in {"numeric", "integer"}
    ]
    categories = [c.name for c in schema.columns if c.name in columns and c.name not in numbers]
    transformers = []
    if numbers:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median", add_indicator=True, keep_empty_features=True
                            ),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                numbers,
            )
        )
    if categories:
        transformers.append(
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categories)
        )
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)
