import json

import pytest
from pydantic import ValidationError

from tabgen_bench.core.schema import (
    ColumnSchema,
    DatasetSchema,
    DatasetSource,
    DatasetSpec,
    GenerationSchemaView,
    TargetSpec,
    ValidityRuleSet,
)


def test_schema_roundtrip(schema):
    encoded = schema.model_dump_json()
    assert DatasetSchema.model_validate_json(encoded) == schema
    assert json.loads(encoded)["columns"][0]["maximum"] == 120
    assert GenerationSchemaView.model_validate_json(schema.generation_view().model_dump_json()) == (
        schema.generation_view()
    )
    assert ValidityRuleSet.model_validate_json(schema.validity_rules().model_dump_json()) == (
        schema.validity_rules()
    )


def test_schema_constraint_version_in_hash(schema):
    data = schema.model_dump()
    data["constraint_rules_version"] = "2"
    revised = DatasetSchema.model_validate(data)
    assert revised.fingerprint() != schema.fingerprint()
    assert revised.validity_rules().fingerprint() != schema.validity_rules().fingerprint()
    assert revised.generation_view().fingerprint() == schema.generation_view().fingerprint()


@pytest.mark.parametrize(
    "field,value", [("maximum", 130), ("constraint_source", "user_rule"), ("rule_version", "2")]
)
def test_evaluation_only_rule_hash_is_separate(schema, field, value):
    data = schema.model_dump()
    data["columns"][0][field] = value
    revised = DatasetSchema.model_validate(data)
    assert revised.validity_rules().fingerprint() != schema.validity_rules().fingerprint()
    assert revised.generation_view() == schema.generation_view()


def test_generation_schema_view_excludes_validity_rules(schema):
    view = schema.generation_view().model_dump_json()
    for field in (
        "minimum",
        "maximum",
        "allowed_values",
        "constraint_source",
        "rule_version",
        "constraint_rules_version",
        "toy_dictionary",
    ):
        assert field not in view
    assert schema.validity_rules().columns[0].maximum == 120
    with pytest.raises(ValidationError):
        schema.generation_view().columns[0].name = "changed"


def test_generation_view_rejects_schema_subclass_rule_leak(schema):
    # Pydantic otherwise accepts ColumnSchema subclasses in base-typed fields.
    with pytest.raises(ValidationError, match="rule-free"):
        GenerationSchemaView(columns=schema.columns)


def test_logical_type_change_affects_generation_hash(schema):
    data = schema.model_dump()
    data["columns"][0]["logical_type"] = "numeric"
    revised = DatasetSchema.model_validate(data)
    assert revised.generation_view().fingerprint() != schema.generation_view().fingerprint()


def test_undeclared_bounds_remain_absent():
    column = ColumnSchema(name="age", role="feature", logical_type="integer", nullable=False)
    assert column.minimum is None and column.maximum is None and column.allowed_values is None


@pytest.mark.parametrize(
    "patch",
    [
        {"minimum": 121, "maximum": 120},
        {"logical_type": "datetime"},
        {"logical_type": "id"},
        {"logical_type": "pii"},
        {"allowed_values": ()},
        {"allowed_values": ("1", "2")},
        {"minimum": float("inf")},
        {"allowed_values": (1, 1)},
        {"allowed_values": (1.5,)},
    ],
)
def test_invalid_schema_declarations_rejected(patch):
    with pytest.raises(ValidationError):
        ColumnSchema(
            **{
                "name": "x",
                "role": "feature",
                "logical_type": "integer",
                "nullable": False,
                **patch,
            }
        )


def test_duplicate_columns_rejected(schema):
    with pytest.raises(ValidationError):
        DatasetSchema(columns=(schema.columns[0], schema.columns[0]))


def test_validity_rules_validate_deserialized_declarations(schema):
    data = schema.validity_rules().model_dump()
    data["columns"][0]["minimum"] = 999
    with pytest.raises(ValidationError):
        ValidityRuleSet.model_validate(data)
    data = schema.validity_rules().model_dump()
    data["columns"] = (data["columns"][0], data["columns"][0])
    with pytest.raises(ValidationError):
        ValidityRuleSet.model_validate(data)


def test_schema_is_immutable(schema):
    with pytest.raises(ValidationError):
        schema.columns[0].maximum = 999
    with pytest.raises(ValidationError):
        schema.constraint_rules_version = "2"


def test_dataset_spec_roundtrip(schema):
    spec = DatasetSpec(
        id="toy",
        version="1",
        dataset_schema=schema,
        source=DatasetSource(provider="local", source_id="toy.csv", version="1"),
        target=TargetSpec(column="label", positive_label=True, negative_label=False),
        canonicalization_version="1",
        expected_rows=3,
    )
    assert DatasetSpec.model_validate_json(spec.model_dump_json()) == spec
    data = spec.model_dump()
    data["target"]["column"] = "missing"
    with pytest.raises(ValidationError):
        DatasetSpec.model_validate(data)
