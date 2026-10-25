"""KS statistics, categorical total variation and column missingness MAE."""

from collections import Counter
from math import fsum

import numpy as np
from scipy.stats import ks_2samp

from tabgen_bench.core.context import EvaluationContext

from ..base import Measurement, Metric, NotApplicable
from ..data import missing, numeric_values, reference_table, token


def total_variation(left, right) -> float:
    a, b = Counter(left), Counter(right)
    if not a or not b:
        raise ValueError("TV requires nonempty samples")
    return 0.5 * fsum(
        abs(a[k] / len(left) - b[k] / len(right)) for k in sorted(a.keys() | b.keys())
    )


class MarginalMetric(Metric):
    def calculate(self, context: EvaluationContext, *, seed: int) -> Measurement:
        name, real = reference_table(context)
        synthetic = context.synthetic
        if synthetic is None or not len(synthetic):
            raise NotApplicable("empty_or_missing_synthetic", (name,))
        scores, columns, skipped = [], {}, []
        for column in context.schema.columns:
            a, b = real[column.name], synthetic[column.name]
            if self.id == "marginal.ks_mean":
                if column.logical_type not in {"numeric", "integer"}:
                    continue
                x, y = numeric_values(a), numeric_values(b)
                if not len(x) or not len(y):
                    skipped.append(column.name)
                    continue
                score = float(ks_2samp(x, y).statistic)
            elif self.id == "marginal.tv_mean":
                if column.logical_type not in {"categorical", "boolean"}:
                    continue
                score = total_variation([token(v) for v in a], [token(v) for v in b])
            else:
                score = abs(
                    sum(missing(v) for v in a) / len(a) - sum(missing(v) for v in b) / len(b)
                )
            scores.append(score)
            columns[column.name] = score
        if not scores:
            raise NotApplicable("no_applicable_columns", (name,))
        return Measurement(
            float(np.mean(scores)),
            (name,),
            len(real),
            {"per_column": columns, "skipped_empty_numeric_columns": skipped},
        )


def plugins() -> tuple[MarginalMetric, ...]:
    return tuple(
        MarginalMetric(
            id=f"marginal.{suffix}",
            domain="fidelity",
            direction="minimize",
            unit="distance",
            required_inputs=frozenset({"schema", "synthetic"}),
            optional_inputs=frozenset({"real_test", "real_reference"}),
        )
        for suffix in ("ks_mean", "tv_mean", "missingness_mae")
    )
