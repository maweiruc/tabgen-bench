"""Dependency identities without requiring a project-specific installer or lock file."""

import hashlib
import platform
import re
from importlib.metadata import distributions
from pathlib import Path

from tabgen_bench.core.hashing import config_hash


def dependency_fingerprint(path=None):
    """Preserve explicit file identities, otherwise hash the installed environment."""
    if path is not None:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    packages = sorted(
        (re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower(), dist.version)
        for dist in distributions()
    )
    return config_hash(
        {
            "dependency_identity_version": "installed_distributions_v1",
            "python": platform.python_version(),
            "packages": packages,
        }
    )
