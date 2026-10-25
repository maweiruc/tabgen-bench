import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from tabgen_bench.interfaces.api import evaluate_only
from tabgen_bench.metrics.fidelity import c2st
from tabgen_bench.metrics.preprocessing import model_frame
from tabgen_bench.metrics.utility import downstream
from tests.support_metrics import binary_fixture, context


def test_c2st_held_out_only(monkeypatch):
    schema, _, train, test, _ = binary_fixture()
    synthetic = train.iloc[:40].copy()
    test = test.copy()
    test["group"] = [f"unique_real_{i}" for i in range(len(test))]
    synthetic["group"] = [f"unique_synthetic_{i}" for i in range(len(synthetic))]
    captured = {}
    original = c2st.preprocessor

    def construct(*args):
        captured["preprocessor"] = original(*args)
        return captured["preprocessor"]

    monkeypatch.setattr(c2st, "preprocessor", construct)
    original_auc = c2st.roc_auc_score

    def capture_auc(y, probability):
        captured["y"], captured["probability"] = y, probability
        return original_auc(y, probability)

    monkeypatch.setattr(c2st, "roc_auc_score", capture_auc)
    result = evaluate_only(
        context(schema, synthetic, real_test=test), seed=23, metric_ids=("joint.c2st_auc",)
    )[-1]
    assert result.status == "ok"
    audit = result.details.to_dict()
    fit, heldout = audit["discriminator_fit_positions"], audit["discriminator_eval_positions"]
    assert set(fit).isdisjoint(heldout) and len(fit) + len(heldout) == 80
    assert len(captured["probability"]) == len(heldout)
    assert result.value == roc_auc_score(captured["y"], captured["probability"])
    pooled = pd.concat(
        [test.iloc[audit["real_positions"]], synthetic.iloc[audit["other_positions"]]],
        ignore_index=True,
    )
    transformed = model_frame(pooled, schema, tuple(c.name for c in schema.columns))
    encoder = captured["preprocessor"].named_transformers_["categorical"]
    assert set(encoder.categories_[0]) == set(transformed.iloc[fit]["group"])
    assert set(encoder.categories_[0]).isdisjoint(set(transformed.iloc[heldout]["group"]))
    assert audit["auc_source"] == "held_out" and audit["preprocessor_fit_rows"] == len(fit)


def test_c2st_does_not_flip_auc_below_half(monkeypatch):
    schema, _, train, test, _ = binary_fixture()
    monkeypatch.setattr(c2st, "roc_auc_score", lambda y, p: 0.2)
    result = evaluate_only(
        context(schema, train, real_test=test), seed=11, metric_ids=("joint.c2st_auc",)
    )[-1]
    assert result.value == 0.2


def test_real_vs_real_is_independent_and_available_with_invalid_synthetic():
    schema, target, train, test, validation = binary_fixture()
    invalid = train.drop(columns="group")
    results = {
        r.metric_id: r
        for r in evaluate_only(
            context(
                schema,
                invalid,
                target=target,
                real_train=train,
                real_test=test,
                real_validation=validation,
            ),
            seed=11,
        )
    }
    assert results["joint.c2st_auc"].notes == "blocked_by_validity"
    calibration = results["joint.c2st_real_vs_real_auc"]
    assert calibration.status == "ok" and calibration.reference_split == "multiple"
    assert results["utility.trtr_auc"].status == "ok"
    assert results["utility.tstr_auc"].notes == "blocked_by_validity"


def test_tstr_no_target_leakage(monkeypatch):
    schema, target, train, test, _ = binary_fixture()
    synthetic = train.iloc[::-1].copy()
    captured = []
    original = downstream.preprocessor

    def construct(schema, columns):
        assert target.column not in columns
        processor = original(schema, columns)
        captured.append(processor)
        return processor

    monkeypatch.setattr(downstream, "preprocessor", construct)
    result = evaluate_only(
        context(schema, synthetic, target=target, real_test=test),
        seed=11,
        metric_ids=("utility.tstr_auc",),
    )[-1]
    assert result.status == "ok"
    audit = result.details.to_dict()["tstr"]
    assert audit["features"] == ["x", "group"] and audit["preprocessor_fit_rows"] == len(synthetic)
    imputer = captured[0].named_transformers_["numeric"].named_steps["imputer"]
    assert imputer.statistics_[0] == np.median(synthetic.x)


def test_tstr_trtr_same_predictor_and_signed_gap():
    schema, target, train, test, _ = binary_fixture()
    synthetic = train.copy()
    synthetic["label"] = ~synthetic.label
    ids = ("utility.trtr_auc", "utility.tstr_auc", "utility.tstr_minus_trtr_auc")
    results = {
        r.metric_id: r
        for r in evaluate_only(
            context(schema, synthetic, target=target, real_train=train, real_test=test),
            metric_ids=ids,
            seed=11,
        )
    }
    real, syn, gap = (results[metric_id] for metric_id in ids)
    assert all(r.status == "ok" for r in (real, syn, gap))
    assert gap.value == pytest.approx(syn.value - real.value) and gap.value < 0
    assert real.details.to_dict()["trtr"]["predictor"] == syn.details.to_dict()["tstr"]["predictor"]


@pytest.mark.parametrize(
    "kind",
    ["single_class_train", "single_class_test", "unknown_label", "missing_label", "tiny_train"],
)
def test_utility_inapplicable_inputs_have_no_score(kind):
    schema, target, train, test, _ = binary_fixture()
    if kind == "single_class_train":
        train["label"] = True
    elif kind == "single_class_test":
        test["label"] = True
    elif kind == "unknown_label":
        train["label"] = "unknown"
    elif kind == "missing_label":
        train = train.astype({"label": object})
        train.loc[0, "label"] = None
    else:
        train = train.iloc[:2]
    result = evaluate_only(
        context(schema, target=target, real_train=train, real_test=test),
        seed=11,
        metric_ids=("utility.trtr_auc",),
    )[-1]
    assert result.status == "not_applicable" and result.value is None


def test_unseen_test_category_does_not_refit_or_switch_model():
    schema, target, train, test, _ = binary_fixture()
    test["group"] = "unseen"
    result = evaluate_only(
        context(schema, target=target, real_train=train, real_test=test),
        seed=11,
        metric_ids=("utility.trtr_auc",),
    )[-1]
    assert result.status == "ok"
    assert result.details.to_dict()["trtr"]["unseen_test_categories"]["group"] == len(test)
