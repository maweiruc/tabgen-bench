"""Uniform independent column resampling, intentionally destroying dependence."""

import numpy as np
import pandas as pd

from .base import DiagnosticGenerator


class IndependentMarginalGenerator(DiagnosticGenerator):
    id = "independent_marginal"

    def sample(self, n_rows: int, *, seed: int) -> pd.DataFrame:
        self._check_sample(n_rows)
        rng = np.random.default_rng(seed)
        columns = [
            self._train.iloc[rng.integers(len(self._train), size=n_rows), j].reset_index(drop=True)
            for j in range(self._train.shape[1])
        ]
        return pd.concat(columns, axis=1).copy(deep=True)
