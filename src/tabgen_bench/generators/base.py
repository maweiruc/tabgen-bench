"""Train-only storage shared by the two diagnostic adapters."""

import platform

import numpy as np
import pandas as pd

from tabgen_bench.core.schema import GenerationSchemaView


class DiagnosticGenerator:
    version = "1.0.0"

    def fit(self, train: pd.DataFrame, schema: GenerationSchemaView, *, seed: int) -> None:
        if type(schema) is not GenerationSchemaView:
            raise TypeError("fit requires a rule-free GenerationSchemaView")
        if not len(train) or list(train.columns) != [c.name for c in schema.columns]:
            raise ValueError("train must be nonempty and match generation schema columns/order")
        self._train = train.copy(deep=True)
        self._schema = schema
        self._fit_seed = seed

    def _check_sample(self, n_rows: int) -> None:
        if not hasattr(self, "_train"):
            raise RuntimeError("generator is not fitted")
        if type(n_rows) is not int or n_rows < 1:
            raise ValueError("n_rows must be a positive integer")

    def environment_info(self):
        return {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "fit_seed": getattr(self, "_fit_seed", None),
            "diagnostic_only": True,
        }
