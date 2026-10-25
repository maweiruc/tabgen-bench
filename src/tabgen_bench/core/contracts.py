"""Shared plugin and injection protocols; never import an implementation."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import pandas as pd

if TYPE_CHECKING:
    from .context import EvaluationContext, ExtensionRequirement
    from .results import MetricResult
    from .schema import GenerationSchemaView

MetricDomain = Literal["validity", "fidelity", "diversity", "utility", "privacy", "statistical"]
MetricDirection = Literal["minimize", "maximize", "target", "diagnostic"]
METRIC_DOMAINS: frozenset[str] = frozenset(
    {"validity", "fidelity", "diversity", "utility", "privacy", "statistical"}
)
METRIC_DIRECTIONS: frozenset[str] = frozenset({"minimize", "maximize", "target", "diagnostic"})
TABLE_INPUTS: frozenset[str] = frozenset(
    {"real_reference", "real_train", "real_validation", "real_test", "synthetic"}
)
CONTEXT_INPUTS: frozenset[str] = TABLE_INPUTS | {"schema", "target", "expected_n_rows"}


def declared_domain_matches(metric_id: str, domain: str) -> bool:
    """Preserve legacy IDs without restricting independently named future plugins."""
    known = {
        "validity": "validity",
        "marginal": "fidelity",
        "dependency": "fidelity",
        "joint": "fidelity",
        "utility": "utility",
        "proximity": "privacy",
    }
    return known.get(metric_id.split(".", 1)[0], domain) == domain


@runtime_checkable
class GeneratorAdapter(Protocol):
    id: str
    version: str

    def fit(self, train: pd.DataFrame, schema: "GenerationSchemaView", *, seed: int) -> None: ...

    def sample(self, n_rows: int, *, seed: int) -> pd.DataFrame: ...

    def environment_info(self) -> Mapping[str, Any]: ...


@runtime_checkable
class MetricPlugin(Protocol):
    id: str
    version: str
    domain: MetricDomain
    direction: MetricDirection
    required_inputs: frozenset[str]
    required_capabilities: frozenset["ExtensionRequirement"]

    def compute(self, context: "EvaluationContext", *, seed: int) -> "MetricResult": ...


@runtime_checkable
class MetricRegistry(Protocol):
    def get(self, metric_id: str) -> MetricPlugin: ...


@runtime_checkable
class OptionalInputPlugin(Protocol):
    """Backward-compatible opt-in access; absence never means missing required input."""

    optional_inputs: frozenset[str]


class EvaluationEngine(Protocol):
    """Shared constructor/evaluation contract for the injected implementation."""

    def __init__(self, metric_registry: MetricRegistry) -> None: ...

    def evaluate(
        self, context: "EvaluationContext", *, metric_ids: tuple[str, ...], seed: int
    ) -> tuple["MetricResult", ...]: ...
