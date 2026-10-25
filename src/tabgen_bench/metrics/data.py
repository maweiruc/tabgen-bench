"""Scalar semantics shared by metrics; transformations never repair source tables."""

from numbers import Real

import numpy as np
import pandas as pd

from tabgen_bench.core.context import EvaluationContext
from tabgen_bench.core.hashing import canonical_json

from .base import NotApplicable


def missing(value) -> bool:
    result = pd.isna(value)
    return isinstance(result, (bool, np.bool_)) and bool(result)


def numeric(value) -> bool:
    return (
        isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and np.isfinite(value)
    )


def supported_scalar(value) -> bool:
    return missing(value) or isinstance(value, (str, bool, np.bool_)) or numeric(value)


def token(value) -> str:
    if missing(value):
        return canonical_json(["missing"])
    if isinstance(value, str):
        return canonical_json(["str", value])
    if isinstance(value, (bool, np.bool_)):
        return canonical_json(["bool", bool(value)])
    if numeric(value):
        normalized = int(value) if float(value).is_integer() else float(value)
        return canonical_json(["number", normalized])
    raise ValueError(f"unsupported scalar: {type(value).__name__}")


def numeric_values(series: pd.Series) -> np.ndarray:
    values = []
    for value in series:
        if not missing(value):
            if not numeric(value):
                raise ValueError("numeric metric input contains a non-finite or non-numeric value")
            values.append(float(value))
    return np.asarray(values, dtype=float)


def reference_table(context: EvaluationContext):
    for name in ("real_test", "real_reference"):
        table = context.table(name)
        if table is not None:
            if not len(table):
                raise NotApplicable("empty_reference", (name,))
            if list(table.columns) != [c.name for c in context.schema.columns]:
                raise NotApplicable("reference_schema_mismatch", (name,))
            return name, table
    raise NotApplicable("missing_input: real_test or real_reference")
