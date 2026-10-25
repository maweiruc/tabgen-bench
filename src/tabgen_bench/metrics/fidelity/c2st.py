"""Held-out C2ST, with an independently scheduled real-vs-real calibration plugin."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from tabgen_bench.core.config import FrozenConfig
from tabgen_bench.core.context import EvaluationContext

from ..base import Measurement, Metric, NotApplicable
from ..data import reference_table
from ..preprocessing import model_frame, preprocessor


def held_out_auc(real, other, schema, seed):
    n = min(len(real), len(other))
    if n < 4:
        raise NotApplicable("insufficient_rows_for_stratified_c2st")
    expected = [c.name for c in schema.columns]
    if list(real.columns) != expected or list(other.columns) != expected:
        raise NotApplicable("c2st_input_schema_mismatch")
    rng = np.random.default_rng(seed)
    a, b = rng.choice(len(real), n, replace=False), rng.choice(len(other), n, replace=False)
    pooled = pd.concat([real.iloc[a], other.iloc[b]], ignore_index=True)
    y = np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)]
    fit, evaluate = train_test_split(np.arange(2 * n), test_size=0.3, stratify=y, random_state=seed)
    columns = tuple(expected)
    x = model_frame(pooled, schema, columns)
    model = Pipeline(
        [
            ("preprocessor", preprocessor(schema, columns)),
            ("classifier", RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)),
        ]
    )
    model.fit(x.iloc[fit], y[fit])
    probability = model.predict_proba(x.iloc[evaluate])[:, 1]
    auc = float(roc_auc_score(y[evaluate], probability))
    return auc, {
        "n_per_class": n,
        "real_positions": a.tolist(),
        "other_positions": b.tolist(),
        "discriminator_fit_positions": fit.tolist(),
        "discriminator_eval_positions": evaluate.tolist(),
        "auc_source": "held_out",
        "fit_fraction": 0.7,
        "preprocessor_fit_rows": len(fit),
        "n_estimators": 200,
        "n_jobs": 1,
    }


class C2ST(Metric):
    def calculate(self, context: EvaluationContext, *, seed: int) -> Measurement:
        if self.id == "joint.c2st_real_vs_real_auc":
            names = ("real_validation", "real_test")
            identities = {i.input_name: i for i in context.identities}
            a, b = (identities[name] for name in names)
            if a.source_id != b.source_id or a.split_id == b.split_id:
                raise NotApplicable("independent_real_splits_required", names)
            real, other = context.real_validation, context.real_test
        else:
            name, real = reference_table(context)
            names, other = (name,), context.synthetic
        try:
            auc, audit = held_out_auc(real, other, context.schema, seed)
        except NotApplicable as exc:
            raise NotApplicable(str(exc), names) from exc
        return Measurement(auc, names, min(len(real), len(other)), audit)


def plugins() -> tuple[C2ST, ...]:
    params = FrozenConfig.from_mapping(
        {
            "n_estimators": 200,
            "n_jobs": 1,
            "test_size": 0.3,
            "stratify": True,
            "sample": "equal_without_replacement",
        }
    )
    return (
        C2ST(
            id="joint.c2st_auc",
            domain="fidelity",
            direction="target",
            unit="auc",
            required_inputs=frozenset({"schema", "synthetic"}),
            optional_inputs=frozenset({"real_test", "real_reference"}),
            params=params,
        ),
        C2ST(
            id="joint.c2st_real_vs_real_auc",
            domain="fidelity",
            direction="target",
            unit="auc",
            required_inputs=frozenset({"schema", "real_validation", "real_test"}),
            params=params,
        ),
    )
