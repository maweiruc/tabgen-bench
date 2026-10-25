"""Public Python API and composition root. No hidden registry or implicit downloads."""

from tabgen_bench.core.context import EvaluationContext
from tabgen_bench.core.contracts import MetricRegistry
from tabgen_bench.core.results import MetricResult
from tabgen_bench.evaluation.engine import EvaluationEngine
from tabgen_bench.metrics.fidelity import c2st, dependency, marginal
from tabgen_bench.metrics.privacy import proximity
from tabgen_bench.metrics.registry import PluginRegistry
from tabgen_bench.metrics.utility import downstream
from tabgen_bench.metrics.validity import schema_checks

# These public services are also used by the CLI. The context-based API remains available.
from .dataset_service import datasets_fetch as datasets_fetch
from .dataset_service import datasets_verify as datasets_verify
from .evaluate_service import evaluate as evaluate
from .report_service import compare as compare
from .report_service import report as report
from .run_service import run as run

M1_METRIC_IDS = (
    "validity.schema_pass",
    "validity.invalid_cell_rate",
    "validity.invalid_row_rate",
    "marginal.ks_mean",
    "marginal.tv_mean",
    "marginal.missingness_mae",
    "dependency.pairwise_tv_mean",
    "joint.c2st_auc",
    "joint.c2st_real_vs_real_auc",
    "utility.trtr_auc",
    "utility.tstr_auc",
    "utility.tstr_minus_trtr_auc",
    "proximity.dcr_train_median",
    "proximity.dcr_holdout_median",
    "proximity.train_closer_rate",
    "proximity.dcr_tie_rate",
    "proximity.exact_train_copy_rate",
)


def build_metric_registry() -> PluginRegistry:
    return PluginRegistry(
        (
            *schema_checks.plugins(),
            *marginal.plugins(),
            dependency.plugin(),
            *c2st.plugins(),
            *downstream.plugins(),
            *proximity.plugins(),
        )
    )


def evaluate_only(
    context: EvaluationContext,
    *,
    seed: int,
    metric_ids: tuple[str, ...] | None = None,
    metric_registry: MetricRegistry | None = None,
    run_id: str | None = None,
    dataset_id: str = "external",
) -> tuple[MetricResult, ...]:
    registry = build_metric_registry() if metric_registry is None else metric_registry
    engine = EvaluationEngine(registry, run_id=run_id, dataset_id=dataset_id)
    return engine.evaluate(
        context, metric_ids=M1_METRIC_IDS if metric_ids is None else metric_ids, seed=seed
    )
