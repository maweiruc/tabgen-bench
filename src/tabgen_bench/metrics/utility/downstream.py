"""Same train-fitted predictor for TRTR and TSTR; target is never a feature."""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from tabgen_bench.core.config import FrozenConfig
from tabgen_bench.core.context import EvaluationContext

from ..base import Measurement, Metric, NotApplicable
from ..data import token
from ..preprocessing import model_frame, preprocessor


def target_values(frame, target):
    mapping = {token(target.negative_label): 0, token(target.positive_label): 1}
    try:
        y = np.asarray([mapping[token(v)] for v in frame[target.column]], dtype=int)
    except KeyError as exc:
        raise NotApplicable("missing_or_unknown_target_label") from exc
    if len(np.unique(y)) != 2:
        raise NotApplicable("binary_classes_required")
    return y


def prediction_auc(train, test, context, seed):
    if len(train) < 4 or len(test) < 2:
        raise NotApplicable("insufficient_utility_rows")
    expected = [c.name for c in context.schema.columns]
    if list(train.columns) != expected or list(test.columns) != expected:
        raise NotApplicable("utility_input_schema_mismatch")
    target = context.target
    columns = tuple(c.name for c in context.schema.columns if c.name != target.column)
    if not columns:
        raise NotApplicable("no_features")
    y_train, y_test = target_values(train, target), target_values(test, target)
    x_train, x_test = (model_frame(table, context.schema, columns) for table in (train, test))
    model = Pipeline(
        [
            ("preprocessor", preprocessor(context.schema, columns)),
            ("classifier", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed)),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x_train, y_train)
    value = float(roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]))
    unseen = {}
    for column in context.schema.columns:
        if column.name in columns and column.logical_type in {"categorical", "boolean"}:
            known = {token(value) for value in train[column.name]}
            unseen[column.name] = sum(token(value) not in known for value in test[column.name])
    return value, {
        "features": list(columns),
        "target": target.column,
        "train_rows": len(train),
        "test_rows": len(test),
        "preprocessor_fit_rows": len(train),
        "unseen_test_categories": unseen,
        "predictor": "LogisticRegression",
        "max_iter": 1000,
        "solver": "lbfgs",
    }


class UtilityMetric(Metric):
    def calculate(self, context: EvaluationContext, *, seed: int) -> Measurement:
        test = context.real_test
        real_score = syn_score = None
        audit = {}
        names = ["real_test"]
        try:
            if self.id != "utility.tstr_auc":
                real_score, audit["trtr"] = prediction_auc(context.real_train, test, context, seed)
                names.insert(0, "real_train")
            if self.id != "utility.trtr_auc":
                syn_score, audit["tstr"] = prediction_auc(context.synthetic, test, context, seed)
        except NotApplicable as exc:
            raise NotApplicable(str(exc), tuple(names)) from exc
        value = (
            real_score
            if self.id == "utility.trtr_auc"
            else syn_score
            if self.id == "utility.tstr_auc"
            else syn_score - real_score
        )
        audit.update(trtr_auc=real_score, tstr_auc=syn_score)
        return Measurement(value, tuple(names), len(test), audit)


def plugins() -> tuple[UtilityMetric, ...]:
    return tuple(
        UtilityMetric(
            id=f"utility.{suffix}",
            domain="utility",
            direction=direction,
            unit="auc",
            required_inputs=frozenset({"schema", "target", "real_test"} | inputs),
            params=FrozenConfig.from_mapping(
                {
                    "predictor": "LogisticRegression",
                    "max_iter": 1000,
                    "solver": "lbfgs",
                    "handle_unknown": "ignore",
                    "numeric_imputer": "median_with_indicator",
                }
            ),
        )
        for suffix, direction, inputs in (
            ("trtr_auc", "maximize", {"real_train"}),
            ("tstr_auc", "maximize", {"synthetic"}),
            ("tstr_minus_trtr_auc", "target", {"real_train", "synthetic"}),
        )
    )
