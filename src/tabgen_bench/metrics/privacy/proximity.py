"""Column-mean nearest-neighbor distance and full-train exact-copy diagnostics."""

from hashlib import sha256

import numpy as np

from tabgen_bench.core.config import FrozenConfig
from tabgen_bench.core.context import EvaluationContext
from tabgen_bench.core.hashing import canonical_json

from ..base import Measurement, Metric, NotApplicable
from ..data import missing, numeric, numeric_values, token

DISCLAIMER = "Proximity diagnostic only; neither a privacy guarantee nor a membership attack."


def row_hashes(frame):
    return [
        sha256(canonical_json([token(v) for v in row]).encode()).hexdigest()
        for row in frame.itertuples(index=False, name=None)
    ]


def distance_block(left, right, columns, ranges):
    """At most rows_left × rows_right, never a rows × references × columns tensor."""
    # Numeric storage is checked before conversion; strings cannot become numbers.
    distance = np.zeros((len(left), len(right)), dtype=float)
    for column in columns:
        a, b = list(left[column.name]), list(right[column.name])
        ma, mb = np.asarray([missing(v) for v in a]), np.asarray([missing(v) for v in b])
        if column.logical_type in {"numeric", "integer"}:
            for value in a + b:
                if not missing(value) and not numeric(value):
                    raise ValueError("DCR numeric input is not finite numeric")
            x = np.asarray([np.nan if missing(v) else float(v) for v in a])
            y = np.asarray([np.nan if missing(v) else float(v) for v in b])
            span = ranges[column.name]
            contribution = (
                abs(x[:, None] - y[None, :]) / span
                if span > 0
                else (x[:, None] != y[None, :]).astype(float)
            )
        else:
            x, y = np.asarray([token(v) for v in a]), np.asarray([token(v) for v in b])
            contribution = (x[:, None] != y[None, :]).astype(float)
        contribution[ma[:, None] | mb[None, :]] = 1
        contribution[ma[:, None] & mb[None, :]] = 0
        distance += contribution
    return distance / len(columns)


def nearest_distances(synthetic, reference, columns, ranges):
    nearest = np.full(len(synthetic), np.inf)
    for start in range(0, len(synthetic), 64):
        block = synthetic.iloc[start : start + 64]
        minima = np.full(len(block), np.inf)
        for j in range(0, len(reference), 128):
            distances = distance_block(block, reference.iloc[j : j + 128], columns, ranges)
            minima = np.minimum(minima, distances.min(axis=1))
        nearest[start : start + len(block)] = minima
    return nearest


class ProximityMetric(Metric):
    def calculate(self, context: EvaluationContext, *, seed: int) -> Measurement:
        train, synthetic = context.real_train, context.synthetic
        expected = [c.name for c in context.schema.columns]
        if list(train.columns) != expected or not len(train) or not len(synthetic):
            raise NotApplicable("empty_or_schema_mismatched_proximity_input", ("real_train",))
        if self.id == "proximity.exact_train_copy_rate":
            hashes = set(row_hashes(train))
            hits = sum(h in hashes for h in row_hashes(synthetic))
            return Measurement(
                hits / len(synthetic),
                ("real_train",),
                len(train),
                {
                    "disclaimer": DISCLAIMER,
                    "n_train_full": len(train),
                    "n_checked": len(synthetic),
                    "exact_copies": hits,
                },
            )
        test = context.real_test
        if not len(test) or list(test.columns) != expected:
            raise NotApplicable("empty_or_schema_mismatched_holdout", ("real_train", "real_test"))
        n_ref, n_syn = min(len(train), len(test), 1024), min(len(synthetic), 512)
        rng = np.random.default_rng(seed)
        train_pos = rng.choice(len(train), n_ref, replace=False)
        test_pos = rng.choice(len(test), n_ref, replace=False)
        syn_pos = rng.choice(len(synthetic), n_syn, replace=False)
        ranges = {}
        for column in context.schema.columns:
            if column.logical_type in {"numeric", "integer"}:
                values = numeric_values(train[column.name])
                ranges[column.name] = float(np.ptp(values)) if len(values) else 0.0
        sampled = synthetic.iloc[syn_pos]
        d_train = nearest_distances(sampled, train.iloc[train_pos], context.schema.columns, ranges)
        d_test = nearest_distances(sampled, test.iloc[test_pos], context.schema.columns, ranges)
        values = {
            "proximity.dcr_train_median": float(np.median(d_train)),
            "proximity.dcr_holdout_median": float(np.median(d_test)),
            "proximity.train_closer_rate": float(np.mean(d_train < d_test)),
            "proximity.dcr_tie_rate": float(np.mean(d_train == d_test)),
        }
        return Measurement(
            values[self.id],
            ("real_train", "real_test"),
            n_ref,
            {
                "disclaimer": DISCLAIMER,
                "n_ref_each": n_ref,
                "n_synthetic_used": n_syn,
                "reference_sampling_seed": seed,
                "train_positions": train_pos.tolist(),
                "holdout_positions": test_pos.tolist(),
                "synthetic_positions": syn_pos.tolist(),
                "numeric_ranges_fit_on": "real_train",
                "numeric_ranges": ranges,
                "tie_rate": values["proximity.dcr_tie_rate"],
                "train_median": values["proximity.dcr_train_median"],
                "holdout_median": values["proximity.dcr_holdout_median"],
                "block_shape_limit": [64, 128],
            },
        )


def plugins() -> tuple[ProximityMetric, ...]:
    return tuple(
        ProximityMetric(
            id=f"proximity.{suffix}",
            domain="privacy",
            direction="diagnostic",
            unit=unit,
            required_inputs=frozenset({"schema", "real_train", "synthetic"} | extra),
            params=FrozenConfig.from_mapping(
                {
                    "max_reference_rows": 1024,
                    "max_synthetic_rows": 512,
                    "distance": "column_mean_train_range_scaled",
                    "constant_distance": "equal_zero_unequal_one",
                }
            )
            if suffix != "exact_train_copy_rate"
            else FrozenConfig.from_mapping({"full_train": True}),
        )
        for suffix, unit, extra in (
            ("dcr_train_median", "distance", {"real_test"}),
            ("dcr_holdout_median", "distance", {"real_test"}),
            ("train_closer_rate", "fraction", {"real_test"}),
            ("dcr_tie_rate", "fraction", {"real_test"}),
            ("exact_train_copy_rate", "fraction", set()),
        )
    )
