"""Strict structure and declared column rules. No repair, reordering, or filtering."""

from collections import Counter
from numbers import Real

import numpy as np

from tabgen_bench.core.context import EvaluationContext

from ..base import Measurement, Metric
from ..data import missing, numeric, supported_scalar, token


def inspect_validity(context: EvaluationContext) -> dict:
    frame = context.synthetic
    if frame is None:
        raise ValueError("synthetic input was not supplied")
    expected = [c.name for c in context.schema.columns]
    actual = list(frame.columns)
    structural = []
    for code, columns in (
        ("duplicate_column", [str(c) for c, count in Counter(actual).items() if count > 1]),
        ("missing_column", [c for c in expected if c not in actual]),
        ("extra_column", [str(c) for c in actual if c not in expected]),
    ):
        if columns:
            structural.append({"reason_code": code, "columns": columns})
    if actual != expected:
        structural.append(
            {
                "reason_code": "order_mismatch",
                "expected": expected,
                "actual": [str(c) for c in actual],
            }
        )
    if not len(frame):
        structural.append({"reason_code": "empty_table", "columns": []})
    if context.expected_n_rows is not None and len(frame) != context.expected_n_rows:
        structural.append(
            {
                "reason_code": "unexpected_row_count",
                "actual": len(frame),
                "expected": context.expected_n_rows,
            }
        )
    unsupported = [
        str(frame.columns[j])
        for j in range(frame.shape[1])
        if any(not supported_scalar(v) and not isinstance(v, Real) for v in frame.iloc[:, j])
    ]
    if unsupported:
        structural.append({"reason_code": "unsupported_storage", "columns": unsupported})
    report = {
        "schema_pass": 0,
        "structural_errors": structural,
        "column_reasons": [],
        "n_rows": len(frame),
        "expected_columns": expected,
        "invalid_cell_count": None,
        "invalid_row_count": None,
        "checked_cell_count": None,
        "invalid_cell_rate": None,
        "invalid_row_rate": None,
        "validity_rules_hash": context.schema.validity_rules().fingerprint(),
    }
    if structural:
        return report
    invalid = np.zeros(frame.shape, dtype=bool)
    for j, rule in enumerate(context.schema.validity_rules().columns):
        counts = Counter()
        allowed = None if rule.allowed_values is None else {token(v) for v in rule.allowed_values}
        for row, value in enumerate(frame.iloc[:, j]):
            reasons = set()
            if missing(value):
                if not rule.nullable:
                    reasons.add("null_not_allowed")
            else:
                is_number = numeric(value)
                if rule.logical_type in {"numeric", "integer"}:
                    if not is_number:
                        reasons.add("type_mismatch")
                    else:
                        if rule.logical_type == "integer" and not float(value).is_integer():
                            reasons.add("non_integer")
                        if (rule.minimum is not None and value < rule.minimum) or (
                            rule.maximum is not None and value > rule.maximum
                        ):
                            reasons.add("out_of_range")
                elif rule.logical_type == "boolean" and not isinstance(value, (bool, np.bool_)):
                    reasons.add("type_mismatch")
                if not supported_scalar(value):
                    reasons.add("type_mismatch")
                elif allowed is not None and token(value) not in allowed:
                    reasons.add("unknown_category")
            invalid[row, j] = bool(reasons)
            counts.update(reasons)
        report["column_reasons"].extend(
            {
                "column": rule.name,
                "reason_code": reason,
                "count": count,
                "rate": count / len(frame),
                "rule_version": rule.rule_version,
                "constraint_source": rule.constraint_source,
            }
            for reason, count in sorted(counts.items())
        )
    cells, rows = int(invalid.sum()), int(invalid.any(axis=1).sum())
    report.update(
        schema_pass=int(cells == 0),
        invalid_cell_count=cells,
        invalid_row_count=rows,
        checked_cell_count=frame.size,
        invalid_cell_rate=cells / frame.size,
        invalid_row_rate=rows / len(frame),
    )
    return report


class ValidityMetric(Metric):
    def calculate(self, context: EvaluationContext, *, seed: int) -> Measurement:
        report = inspect_validity(context)
        value = report[self.id.split(".")[1]]
        if value is None:
            return Measurement(
                None,
                details=report,
                status="not_applicable",
                notes="structural_error: semantic rate is not computable",
            )
        return Measurement(value, details=report)


def plugins() -> tuple[ValidityMetric, ...]:
    return tuple(
        ValidityMetric(
            id=f"validity.{name}",
            domain="validity",
            direction=direction,
            unit="fraction",
            required_inputs=frozenset({"schema", "synthetic"}),
            optional_inputs=frozenset({"expected_n_rows"}),
        )
        for name, direction in (
            ("schema_pass", "target"),
            ("invalid_cell_rate", "minimize"),
            ("invalid_row_rate", "minimize"),
        )
    )
