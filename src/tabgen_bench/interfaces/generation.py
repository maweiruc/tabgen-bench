"""Train-only composition helper: pass only a train snapshot and rule-free projection."""

import pandas as pd

from tabgen_bench.core.contracts import GeneratorAdapter
from tabgen_bench.core.schema import DatasetSchema
from tabgen_bench.datasets.splits import DatasetSplit


def fit_on_train(
    adapter: GeneratorAdapter,
    frame: pd.DataFrame,
    schema: DatasetSchema,
    split: DatasetSplit,
    *,
    seed: int,
) -> GeneratorAdapter:
    train, _, _ = split.frames(frame)
    adapter.fit(train, schema.generation_view(), seed=seed)
    return adapter
