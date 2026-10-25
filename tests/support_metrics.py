"""Local artificial evaluation fixtures; never fetch real data."""

from hashlib import sha256

import numpy as np
import pandas as pd

from tabgen_bench.core.context import EvaluationContext, InputIdentity
from tabgen_bench.core.schema import ColumnSchema, DatasetSchema, TargetSpec


def context(schema, synthetic=None, *, target=None, expected_n_rows=None, extensions=(), **real):
    tables = {k: v for k, v in {"synthetic": synthetic, **real}.items() if v is not None}
    identities = tuple(
        InputIdentity(
            input_name=name,
            source_id="toy-local",
            split_id=name,
            reference_split={
                "real_train": "train",
                "real_test": "test",
                "real_validation": "validation",
                "real_reference": "reference",
                "synthetic": "synthetic",
            }[name],
            content_hash=sha256(frame.to_csv(index=False).encode()).hexdigest(),
        )
        for name, frame in tables.items()
    )
    return EvaluationContext(
        schema=schema,
        identities=identities,
        target=target,
        expected_n_rows=expected_n_rows,
        extensions=extensions,
        **tables,
    )


def binary_fixture():
    schema = DatasetSchema(
        columns=(
            ColumnSchema(name="x", role="feature", logical_type="numeric", nullable=True),
            ColumnSchema(name="group", role="feature", logical_type="categorical", nullable=True),
            ColumnSchema(name="label", role="target", logical_type="boolean", nullable=False),
        )
    )
    target = TargetSpec(column="label", positive_label=True, negative_label=False)

    def table(n, start):
        x = np.linspace(-3, 3, n)
        return pd.DataFrame(
            {"x": x, "group": [f"g{i % 3}" for i in range(n)], "label": x > 0},
            index=np.arange(start, start + n),
        )

    return schema, target, table(80, 0), table(40, 80), table(40, 120)
