"""Immutable, row-free public dataset provenance bundled with the package."""

import hashlib
import json
from importlib.resources import files

PUBLIC_METADATA_SHA256 = {
    "data-quality.json": "576c3807dbe85559d73ea5bfcb76cba410457097567bcf414588f3d9962825fa",
    "bank-mapping-verification.json": (
        "c45455f5b513b04fa821cc03adbb67eb25eb10c2b6db3ff71c757540b670a172"
    ),
}


def load_public_metadata(name):
    expected = PUBLIC_METADATA_SHA256[name]
    payload = files("tabgen_bench.datasets").joinpath("resources", name).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"packaged public metadata SHA256 mismatch: {name}")
    return json.loads(payload)
