from pathlib import Path

import pandas as pd
import pytest

from tabgen_bench.core.contracts import GeneratorAdapter
from tabgen_bench.core.schema import GenerationSchemaView
from tabgen_bench.datasets.canonicalize import normalized_content_hash
from tabgen_bench.datasets.registry import DatasetRegistry
from tabgen_bench.datasets.splits import split_dataset
from tabgen_bench.generators.bootstrap import BootstrapGenerator
from tabgen_bench.generators.independent_marginal import IndependentMarginalGenerator
from tabgen_bench.generators.sdv_adapter import CTGANGenerator, GaussianCopulaGenerator
from tabgen_bench.interfaces.generation import fit_on_train


@pytest.mark.parametrize(
    "adapter",
    [
        BootstrapGenerator(),
        IndependentMarginalGenerator(),
        GaussianCopulaGenerator(),
        CTGANGenerator(),
    ],
)
def test_existing_generator_protocol_unchanged(adapter):
    assert isinstance(adapter, GeneratorAdapter)


def test_only_train_given_to_generator():
    from tests.unit.test_datasets import toy_spec

    schema = toy_spec().dataset_schema
    frame = pd.DataFrame({"a": range(100), "y": ["no", "yes"] * 50})
    snapshot = frame.copy(deep=True)
    split = split_dataset(frame, "y", content_hash=normalized_content_hash(frame))

    class Capture:
        def fit(self, train, view, *, seed):
            assert type(view) is GenerationSchemaView
            assert not hasattr(view, "validity_rules")
            assert seed == 11
            assert tuple(train.a) == split.train
            assert not set(train.a) & set(split.validation + split.test)
            train.iloc[0, 0] = -1000  # Adapter cannot mutate the source dataset.

    fit_on_train(Capture(), frame, schema, split, seed=11)
    pd.testing.assert_frame_equal(frame, snapshot)


def test_evaluation_rule_change_does_not_change_generator_metadata_view():
    registry = DatasetRegistry(Path(__file__).resolve().parents[2])
    schema = registry.spec("adult").dataset_schema
    changed = schema.model_copy(
        update={
            "columns": (
                schema.columns[0].model_copy(update={"minimum": 123, "rule_version": "test"}),
                *schema.columns[1:],
            )
        }
    )
    assert changed.validity_rules().fingerprint() != schema.validity_rules().fingerprint()
    assert changed.generation_view() == schema.generation_view()
