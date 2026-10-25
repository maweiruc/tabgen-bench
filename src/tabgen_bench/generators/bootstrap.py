"""Uniform train-row resampling: a memorization diagnostic, not privacy protection."""

import numpy as np
import pandas as pd

from .base import DiagnosticGenerator


class BootstrapGenerator(DiagnosticGenerator):
    id = "bootstrap"

    def sample(self, n_rows: int, *, seed: int) -> pd.DataFrame:
        self._check_sample(n_rows)
        positions = np.random.default_rng(seed).integers(len(self._train), size=n_rows)
        return self._train.iloc[positions].reset_index(drop=True).copy(deep=True)
