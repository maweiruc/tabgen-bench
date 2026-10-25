"""Fixtures use the shipped artificial data and evaluation service."""

from pathlib import Path

import yaml

from tabgen_bench.interfaces.api import evaluate

ROOT = Path(__file__).resolve().parents[1]


def eval_config(tmp_path, *, name="evaluated", invalid=False, reference_only=False):
    document = yaml.safe_load((ROOT / "configs/examples/evaluate_only.yaml").read_text())
    base = ROOT / "configs/examples"
    for key in (
        "schema",
        "synthetic",
        "train",
        "validation",
        "test",
        "evaluation_spec",
    ):
        document[key] = str((base / document[key]).resolve())
    document["output"] = str(tmp_path / name)
    if invalid:
        import pandas as pd

        data = pd.read_csv(document["synthetic"])
        data.loc[0, "x"] = -12
        document["synthetic"] = str(tmp_path / f"{name}-invalid.csv")
        data.to_csv(document["synthetic"], index=False)
    if reference_only:
        document["reference"] = document.pop("test")
        document.pop("train")
        document.pop("validation")
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(document))
    return path


def evaluated(tmp_path, **kwargs):
    config = eval_config(tmp_path, **kwargs)
    result = evaluate(config, create_report=False)
    assert result["status"] == "completed"
    return Path(result["run"])
