"""Optional pinned SDV boundary. No SDK import or metadata detection at module import."""

import platform
import random
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pandas as pd

from tabgen_bench.core.errors import ConfigurationError, OptionalDependencyMissingError
from tabgen_bench.core.schema import GenerationSchemaView

SDV_VERSION = "1.38.5"
SDK_SAMPLING_SEED = 73251


def _sdk():
    try:
        import torch
        from sdv.metadata import Metadata
        from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
    except ModuleNotFoundError as error:
        raise OptionalDependencyMissingError(
            "SDV adapter requires the optional 'tabgen-bench[sdv]' dependency set"
        ) from error
    if version("sdv") != SDV_VERSION:
        raise ConfigurationError(f"SDV {SDV_VERSION} is required by adapter 1.0.0")
    return torch, Metadata, GaussianCopulaSynthesizer, CTGANSynthesizer


def generation_metadata(schema: GenerationSchemaView):
    if type(schema) is not GenerationSchemaView:
        raise TypeError("SDV metadata requires a rule-free GenerationSchemaView")
    _, Metadata, _, _ = _sdk()
    columns = {}
    for column in schema.columns:
        definition = {
            "sdtype": {
                "integer": "numerical",
                "numeric": "numerical",
                "categorical": "categorical",
                "boolean": "boolean",
            }[column.logical_type]
        }
        if column.logical_type in {"integer", "numeric"}:
            definition["computer_representation"] = (
                "Int64" if column.logical_type == "integer" else "Float64"
            )
        columns[column.name] = definition
    metadata = Metadata.load_from_dict(
        {
            "METADATA_SPEC_VERSION": "V1",
            "tables": {"table": {"columns": columns}},
            "relationships": [],
        }
    )
    metadata.validate()
    return metadata


@contextmanager
def _seeded(seed: int, torch, threads: int):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state, old_threads = torch.get_rng_state(), torch.get_num_threads()
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(threads)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        torch.set_num_threads(old_threads)


class SDVGenerator:
    version = "1.0.0"
    id = ""
    allowed_parameters = frozenset()

    def __init__(self, *, cpu_threads: int = 1, **parameters):
        if type(cpu_threads) is not int or cpu_threads < 1:
            raise ConfigurationError("cpu_threads must be a positive integer")
        unknown = set(parameters) - self.allowed_parameters
        if unknown:
            raise ConfigurationError(f"unsupported {self.id} parameters: {sorted(unknown)}")
        if parameters.get("enable_gpu", False) is not False:
            raise ConfigurationError("This adapter supports explicitly CPU execution only")
        self._parameters = {"enforce_min_max_values": False, "enforce_rounding": True, **parameters}
        self._threads = cpu_threads

    def fit(self, train: pd.DataFrame, schema: GenerationSchemaView, *, seed: int) -> None:
        # A failed refit must not leave a previous model available for sampling.
        self._synthesizer = None
        self._fit_seed = None
        self._sample_seed = None
        self._train_rows = None
        self._metadata = None
        if type(schema) is not GenerationSchemaView:
            raise TypeError("fit requires a rule-free GenerationSchemaView")
        if not len(train) or list(train.columns) != [c.name for c in schema.columns]:
            raise ValueError("nonempty train must match generation schema columns/order")
        torch, _, gaussian, ctgan = _sdk()
        metadata = generation_metadata(schema)
        params = dict(self._parameters)
        cls = gaussian
        if self.id == "ctgan":
            cls = ctgan
            params.setdefault("epochs", 100)
            params["enable_gpu"] = False
        with _seeded(seed, torch, self._threads):
            synthesizer = cls(metadata, **params)
            synthesizer.fit(data=train.copy(deep=True))
        self._synthesizer = synthesizer
        self._schema = schema
        self._metadata = metadata.to_dict()
        self._fit_seed = seed
        self._train_rows = len(train)
        self._sample_seed = None

    def sample(self, n_rows: int, *, seed: int) -> pd.DataFrame:
        if getattr(self, "_synthesizer", None) is None:
            raise RuntimeError("generator is not fitted")
        if type(n_rows) is not int or n_rows < 1:
            raise ValueError("n_rows must be a positive integer")
        torch, _, _, _ = _sdk()
        with _seeded(seed, torch, self._threads):
            self._synthesizer.reset_sampling()
            frame = self._synthesizer.sample(num_rows=n_rows)
        self._sample_seed = seed
        if len(frame) != n_rows or list(frame.columns) != [c.name for c in self._schema.columns]:
            raise ValueError("SDK output shape/order mismatch; adapter will not repair it")
        return frame.copy(deep=True)

    def environment_info(self):
        installed = {}
        for package in ("sdv", "torch", "ctgan", "copulas", "rdt", "numpy", "pandas"):
            try:
                installed[package] = version(package)
            except PackageNotFoundError:
                installed[package] = None
        synth = getattr(self, "_synthesizer", None)
        return {
            "adapter_id": self.id,
            "adapter_version": self.version,
            "python": platform.python_version(),
            "packages": installed,
            "device": "cpu",
            "cpu_threads": self._threads,
            "parameters": synth.get_parameters() if synth is not None else self._parameters,
            "fit_seed": getattr(self, "_fit_seed", None),
            "sample_requested_seed": getattr(self, "_sample_seed", None),
            "sdk_sampling_seed": SDK_SAMPLING_SEED,
            "sample_seed_semantics": "ambient RNG only; SDK reset uses fixed stream 73251",
            "reproducibility_level": "same-environment CPU; no cross-platform bitwise guarantee",
            "train_rows": getattr(self, "_train_rows", None),
            "generation_schema_hash": self._schema.fingerprint() if synth is not None else None,
            "metadata": getattr(self, "_metadata", None),
        }


class GaussianCopulaGenerator(SDVGenerator):
    id = "gaussian_copula"
    allowed_parameters = frozenset(
        {
            "enforce_min_max_values",
            "enforce_rounding",
            "locales",
            "numerical_distributions",
            "default_distribution",
        }
    )


class CTGANGenerator(SDVGenerator):
    id = "ctgan"
    allowed_parameters = frozenset(
        {
            "enforce_min_max_values",
            "enforce_rounding",
            "locales",
            "embedding_dim",
            "generator_dim",
            "discriminator_dim",
            "generator_lr",
            "generator_decay",
            "discriminator_lr",
            "discriminator_decay",
            "batch_size",
            "discriminator_steps",
            "log_frequency",
            "verbose",
            "epochs",
            "pac",
            "enable_gpu",
        }
    )
