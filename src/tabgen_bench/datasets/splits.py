"""Frozen V1 row-position split; no generators or evaluation algorithms."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tabgen_bench.core.hashing import canonical_json, config_hash

from .canonicalize import normalized_content_hash

SPLIT_SEED = 20261025
SPLIT_VERSION = "stratified_60_20_20_v1"


@dataclass(frozen=True)
class DatasetSplit:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    seed: int
    n_rows: int
    content_hash: str

    def __post_init__(self):
        parts = (self.train, self.validation, self.test)
        if any(
            type(part) is not tuple or not part or any(type(i) is not int for i in part)
            for part in parts
        ):
            raise ValueError("splits require nonempty integer row positions")
        if type(self.n_rows) is not int or self.n_rows < 1:
            raise ValueError("n_rows must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        flattened = [i for part in parts for i in part]
        if len(flattened) != self.n_rows or set(flattened) != set(range(self.n_rows)):
            raise ValueError("split positions must be disjoint and cover every row exactly once")

    def to_dict(self):
        positions = {name: list(getattr(self, name)) for name in ("train", "validation", "test")}
        return {
            "split_version": SPLIT_VERSION,
            "seed": self.seed,
            "n_rows": self.n_rows,
            "content_hash": self.content_hash,
            "positions": positions,
            "position_hashes": {k: config_hash(v) for k, v in positions.items()},
        }

    def save(self, path: str | Path):
        # Exclusive creation prevents replacement of a user's frozen split.
        with Path(path).open("x") as output:
            output.write(canonical_json(self.to_dict()) + "\n")

    @classmethod
    def load(cls, path: str | Path, *, expected: dict | None = None):
        record = json.loads(Path(path).read_text())
        if record["split_version"] != SPLIT_VERSION:
            raise ValueError("unsupported split version")
        if set(record["positions"]) != {"train", "validation", "test"}:
            raise ValueError("split must contain exactly train/validation/test")
        split = cls(
            **{k: tuple(v) for k, v in record["positions"].items()},
            seed=record["seed"],
            n_rows=record["n_rows"],
            content_hash=record["content_hash"],
        )
        if split.to_dict() != record:
            raise ValueError("split positions/hash mismatch")
        if expected is not None:
            summary = {k: v for k, v in record.items() if k != "positions"}
            summary["counts"] = {k: len(v) for k, v in record["positions"].items()}
            if summary != expected:
                raise ValueError("split differs from the frozen dataset lock")
        return split

    def frames(self, frame: pd.DataFrame):
        if len(frame) != self.n_rows:
            raise ValueError("split/frame length mismatch")
        if normalized_content_hash(frame) != self.content_hash:
            raise ValueError("split/frame content or row order mismatch")
        return tuple(
            frame.iloc[list(getattr(self, name))].copy(deep=True)
            for name in ("train", "validation", "test")
        )


def split_dataset(
    frame: pd.DataFrame, target: str, *, content_hash: str, seed: int = SPLIT_SEED
) -> DatasetSplit:
    if frame[target].isna().any() or frame[target].nunique() != 2:
        raise ValueError("V1 split requires a nonmissing binary target")
    positions = np.arange(len(frame))
    train, holdout = train_test_split(
        positions, test_size=0.4, random_state=seed, stratify=frame[target].to_numpy()
    )
    validation, test = train_test_split(
        holdout, test_size=0.5, random_state=seed, stratify=frame.iloc[holdout][target].to_numpy()
    )
    return DatasetSplit(
        *(tuple(int(i) for i in sorted(part)) for part in (train, validation, test)),
        seed=seed,
        n_rows=len(frame),
        content_hash=content_hash,
    )
