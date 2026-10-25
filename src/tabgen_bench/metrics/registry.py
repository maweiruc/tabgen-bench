"""Explicit instance registry, with no built-ins or import-time discovery."""

from collections.abc import Iterable

from tabgen_bench.core.context import ExtensionRequirement
from tabgen_bench.core.contracts import (
    CONTEXT_INPUTS,
    METRIC_DIRECTIONS,
    METRIC_DOMAINS,
    MetricPlugin,
    OptionalInputPlugin,
    declared_domain_matches,
)
from tabgen_bench.core.errors import DuplicateMetricError, InvalidPluginError, UnknownMetricError


class PluginRegistry:
    """Implements Core.MetricRegistry for all six peer domains uniformly."""

    def __init__(self, plugins: Iterable[MetricPlugin] = ()) -> None:
        self._plugins: dict[str, MetricPlugin] = {}
        for plugin in plugins:
            self.register(plugin)

    def register(self, plugin: MetricPlugin) -> None:
        if not isinstance(plugin, MetricPlugin):
            raise InvalidPluginError("plugin does not satisfy MetricPlugin")
        if not all(isinstance(v, str) and v.strip() for v in (plugin.id, plugin.version)):
            raise InvalidPluginError("plugin ID and version must be nonempty strings")
        if plugin.domain not in METRIC_DOMAINS or plugin.direction not in METRIC_DIRECTIONS:
            raise InvalidPluginError("invalid domain or direction")
        if not declared_domain_matches(plugin.id, plugin.domain):
            raise InvalidPluginError("domain conflicts with a stable metric ID")
        if not isinstance(plugin.required_inputs, frozenset):
            raise InvalidPluginError("required_inputs must be a frozenset")
        if not plugin.required_inputs <= CONTEXT_INPUTS:
            raise InvalidPluginError(
                "unknown table input; use a versioned capability for extensions"
            )
        if isinstance(plugin, OptionalInputPlugin):
            if not isinstance(plugin.optional_inputs, frozenset) or not (
                plugin.optional_inputs <= CONTEXT_INPUTS
            ):
                raise InvalidPluginError("invalid optional_inputs declaration")
        if not isinstance(plugin.required_capabilities, frozenset) or not all(
            isinstance(c, ExtensionRequirement) for c in plugin.required_capabilities
        ):
            raise InvalidPluginError("required_capabilities must contain ExtensionRequirement")
        if not callable(plugin.compute):
            raise InvalidPluginError("compute must be callable")
        if plugin.id in self._plugins:
            raise DuplicateMetricError(plugin.id)
        self._plugins[plugin.id] = plugin

    def get(self, metric_id: str) -> MetricPlugin:
        try:
            return self._plugins[metric_id]
        except KeyError as exc:
            raise UnknownMetricError(metric_id) from exc
