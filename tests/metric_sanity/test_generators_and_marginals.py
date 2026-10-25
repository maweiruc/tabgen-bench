import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from tabgen_bench.core.contracts import GeneratorAdapter
from tabgen_bench.core.schema import ColumnSchema, DatasetSchema
from tabgen_bench.generators.bootstrap import BootstrapGenerator
from tabgen_bench.generators.independent_marginal import IndependentMarginalGenerator
from tabgen_bench.interfaces.api import evaluate_only
from tests.fakes import toy_table
from tests.support_metrics import context


def metric_value(schema, real, synthetic, metric_id):
    results = evaluate_only(
        context(schema, synthetic, real_reference=real), metric_ids=(metric_id,), seed=11
    )
    return results[-1]


def test_bootstrap_rows_subset_of_train(schema):
    frame = toy_table()
    frame.loc[0, "group"] = None
    generator = BootstrapGenerator()
    assert isinstance(generator, GeneratorAdapter)
    generator.fit(frame, schema.generation_view(), seed=11)
    sampled = generator.sample(41, seed=23)
    assert sampled.shape == (41, 3) and sampled.dtypes.equals(frame.dtypes)
    for _, row in sampled.iterrows():
        assert any(row.equals(source) for _, source in frame.iterrows())
    assert_frame_equal(sampled, generator.sample(41, seed=23))


def test_bootstrap_exact_copy_rate_one(schema):
    frame = toy_table()
    generator = BootstrapGenerator()
    generator.fit(frame, schema.generation_view(), seed=11)
    sampled = generator.sample(101, seed=23)
    result = evaluate_only(
        context(schema, sampled, real_train=frame),
        seed=11,
        metric_ids=("proximity.exact_train_copy_rate",),
    )[-1]
    assert result.status == "ok" and result.value == 1


def test_independent_marginal_breaks_dependency():
    schema = DatasetSchema(
        columns=tuple(
            ColumnSchema(name=n, role="feature", logical_type="categorical", nullable=False)
            for n in ("a", "b")
        )
    )
    train = pd.DataFrame({"a": ["A", "B"] * 100, "b": ["A", "B"] * 100})
    generator = IndependentMarginalGenerator()
    generator.fit(train, schema.generation_view(), seed=11)
    sampled = generator.sample(10000, seed=23)
    assert 0.4 < (sampled.a != sampled.b).mean() < 0.6
    assert abs((sampled.a == "A").mean() - 0.5) < 0.04
    assert abs((sampled.b == "A").mean() - 0.5) < 0.04
    assert_frame_equal(sampled, generator.sample(10000, seed=23))


def test_independent_marginal_keeps_marginals_breaks_dependency():
    test_independent_marginal_breaks_dependency()


@pytest.mark.parametrize("cls", [BootstrapGenerator, IndependentMarginalGenerator])
def test_generator_no_global_rng_or_holdout_and_rejects_invalid_request(schema, cls):
    generator = cls()
    with pytest.raises(RuntimeError):
        generator.sample(1, seed=1)
    with pytest.raises(TypeError):
        generator.fit(toy_table(), schema, seed=1)
    generator.fit(toy_table(), schema.generation_view(), seed=11)
    np.random.seed(99)
    before = np.random.get_state()
    generator.sample(3, seed=5)
    after = np.random.get_state()
    assert np.array_equal(before[1], after[1])
    for n in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            generator.sample(n, seed=1)
    with pytest.raises(TypeError):
        generator.fit(toy_table(), schema.generation_view(), real_test=toy_table(), seed=1)


def test_ks_identical_zero(schema):
    assert metric_value(schema, toy_table(), toy_table(), "marginal.ks_mean").value == 0


def test_tv_disjoint_categories_one():
    schema = DatasetSchema(
        columns=(ColumnSchema(name="c", role="feature", logical_type="categorical", nullable=True),)
    )
    assert (
        metric_value(
            schema,
            pd.DataFrame({"c": ["A", "A"]}),
            pd.DataFrame({"c": ["B", "B"]}),
            "marginal.tv_mean",
        ).value
        == 1
    )
    assert (
        metric_value(
            schema, pd.DataFrame({"c": [None]}), pd.DataFrame({"c": ["nan"]}), "marginal.tv_mean"
        ).value
        == 1
    )


def test_missingness_mae_includes_target_and_does_not_treat_nan_string_as_missing():
    schema = DatasetSchema(
        columns=(ColumnSchema(name="c", role="feature", logical_type="categorical", nullable=True),)
    )
    result = metric_value(
        schema,
        pd.DataFrame({"c": [None, None]}),
        pd.DataFrame({"c": ["nan", None]}),
        "marginal.missingness_mae",
    )
    assert result.value == 0.5 and result.reference_split == "reference"


def test_no_applicable_columns_is_na():
    schema = DatasetSchema(
        columns=(ColumnSchema(name="c", role="feature", logical_type="categorical", nullable=True),)
    )
    result = metric_value(
        schema, pd.DataFrame({"c": ["A"]}), pd.DataFrame({"c": ["A"]}), "marginal.ks_mean"
    )
    assert result.status == "not_applicable" and result.value is None


def test_tv_independent_of_sample_and_category_iteration_order():
    from tabgen_bench.metrics.fidelity.marginal import total_variation

    left = ["a"] * 17 + ["b"] * 11 + ["c"] * 7 + ["d"] * 5
    right = ["d"] * 13 + ["c"] * 9 + ["b"] * 6 + ["a"] * 3
    assert total_variation(left, right) == total_variation(left[::-1], right[::-1])
