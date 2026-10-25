"""Pairwise TV with fixed training-derived bins/top categories and bounded pair count."""

from collections import Counter
from itertools import combinations

import numpy as np

from tabgen_bench.core.config import FrozenConfig
from tabgen_bench.core.context import EvaluationContext

from ..base import Measurement, Metric, NotApplicable
from ..data import missing, numeric, numeric_values, reference_table, token
from .marginal import total_variation


def encode_column(basis, values, logical_type):
    if logical_type in {"numeric", "integer"}:
        observed = numeric_values(basis)
        edges = np.unique(np.quantile(observed, np.linspace(0, 1, 11))) if len(observed) else []

        def encode(value):
            if missing(value):
                return "NA"
            if not numeric(value):
                raise ValueError("pairwise numeric input is not numeric")
            if not len(edges):
                return "UNOBSERVED_NUMERIC"
            if value < edges[0]:
                return "UNDERFLOW"
            if value > edges[-1]:
                return "OVERFLOW"
            return f"BIN_{np.searchsorted(edges[1:-1], value, side='right')}"

        return [encode(v) for v in values], {"quantile_edges": list(map(float, edges))}
    frequencies = Counter(token(v) for v in basis if not missing(v))
    top = [k for k, _ in sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))[:20]]
    return ["NA" if missing(v) else token(v) if token(v) in top else "OTHER" for v in values], {
        "top_categories": top,
    }


class PairwiseTV(Metric):
    def calculate(self, context: EvaluationContext, *, seed: int) -> Measurement:
        reference_name, real = reference_table(context)
        basis_name, basis = "real_train", context.real_train
        if basis is None:
            basis_name, basis = "real_reference", context.real_reference
            identity = next((i for i in context.identities if i.input_name == basis_name), None)
            if basis is None or identity.reference_split != "reference":
                raise NotApplicable(
                    "missing_input: train or ordinary reference for bins", (reference_name,)
                )
        if not len(basis):
            raise NotApplicable("empty_bin_fit_reference", (reference_name, basis_name))
        if list(basis.columns) != [c.name for c in context.schema.columns]:
            raise NotApplicable("bin_fit_schema_mismatch", (reference_name, basis_name))
        names = [c.name for c in context.schema.columns]
        pairs = list(combinations(names, 2))[:45]
        if not pairs:
            raise NotApplicable("fewer_than_two_columns", (reference_name, basis_name))
        encoded_real, encoded_syn, bins = {}, {}, {}
        synthetic = context.synthetic
        for column in context.schema.columns:
            encoded_real[column.name], bins[column.name] = encode_column(
                basis[column.name], real[column.name], column.logical_type
            )
            encoded_syn[column.name], _ = encode_column(
                basis[column.name], synthetic[column.name], column.logical_type
            )
        scores = []
        for a, b in pairs:
            score = total_variation(
                list(zip(encoded_real[a], encoded_real[b], strict=True)),
                list(zip(encoded_syn[a], encoded_syn[b], strict=True)),
            )
            scores.append({"columns": [a, b], "tv": score})
        return Measurement(
            float(np.mean([p["tv"] for p in scores])),
            tuple(dict.fromkeys((reference_name, basis_name))),
            len(real),
            {
                "pairs": scores,
                "pair_cap": 45,
                "pair_selection": "schema_order_first",
                "bin_fit_input": basis_name,
                "bins": bins,
            },
        )


def plugin() -> PairwiseTV:
    return PairwiseTV(
        id="dependency.pairwise_tv_mean",
        domain="fidelity",
        direction="minimize",
        unit="distance",
        required_inputs=frozenset({"schema", "synthetic"}),
        optional_inputs=frozenset({"real_train", "real_test", "real_reference"}),
        params=FrozenConfig.from_mapping({"max_bins": 10, "top_categories": 20, "pair_cap": 45}),
    )
