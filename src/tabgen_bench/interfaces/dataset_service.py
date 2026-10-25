"""Explicit fetch and offline verification over the frozen registry."""

from pathlib import Path

from tabgen_bench.datasets.openml import fetch_dataset, load_dataset
from tabgen_bench.datasets.registry import PINNED_SOURCES, DatasetRegistry
from tabgen_bench.datasets.splits import DatasetSplit, split_dataset
from tabgen_bench.experiments.artifacts import atomic_write, json_bytes


def datasets_action(action, dataset_ids=(), *, project_root=None, data_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    data = Path(data_root).resolve() if data_root is not None else root / "data"
    registry = DatasetRegistry(root)
    names = tuple(dataset_ids) or tuple(PINNED_SOURCES)
    results = []
    for name in names:
        try:
            spec, lock = registry.spec(name), registry.lock(name)
            frame = (
                fetch_dataset(registry, name, data)
                if action == "fetch"
                else load_dataset(registry, name, data)
            )
            split_path = data / "splits" / f"{name}.json"
            if action == "fetch" and not split_path.exists():
                frozen = split_dataset(
                    frame,
                    spec.target.column,
                    content_hash=lock["normalized_content_sha256"],
                    seed=lock["split"]["seed"],
                )
                info = frozen.to_dict()
                expected = {k: v for k, v in info.items() if k != "positions"}
                expected["counts"] = {k: len(v) for k, v in info["positions"].items()}
                if expected != lock["split"]:
                    raise ValueError("reconstructed split differs from frozen lock")
                atomic_write(split_path, json_bytes(info))
            split = DatasetSplit.load(split_path, expected=lock["split"])
            split.frames(frame)
            results.append(
                {
                    "dataset_id": name,
                    "status": "ok",
                    "rows": len(frame),
                    "content_sha256": lock["normalized_content_sha256"],
                    "split_counts": lock["split"]["counts"],
                }
            )
        except Exception as error:
            results.append(
                {
                    "dataset_id": name,
                    "status": "failed",
                    "stage": f"datasets.{action}",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
    return {
        "action": action,
        "project_root": str(root),
        "data_root": str(data),
        "datasets": results,
    }


def datasets_fetch(dataset_ids=(), **kwargs):
    return datasets_action("fetch", dataset_ids, **kwargs)


def datasets_verify(dataset_ids=(), **kwargs):
    return datasets_action("verify", dataset_ids, **kwargs)
