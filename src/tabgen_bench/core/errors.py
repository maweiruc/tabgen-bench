"""Machine-readable error identities; no execution or fallback behavior."""


class TabGenBenchError(Exception):
    code = "tabgen_bench_error"


class UnsupportedSchemaError(TabGenBenchError):
    code = "unsupported_schema"


class ConfigurationError(TabGenBenchError):
    code = "invalid_configuration"


class OptionalDependencyMissingError(TabGenBenchError):
    code = "optional_dependency_missing"


class DuplicateMetricError(TabGenBenchError):
    code = "duplicate_metric"


class UnknownMetricError(TabGenBenchError):
    code = "unknown_metric"


class InvalidPluginError(TabGenBenchError):
    code = "invalid_plugin"


class MissingCapabilityError(TabGenBenchError):
    code = "missing_capability"
