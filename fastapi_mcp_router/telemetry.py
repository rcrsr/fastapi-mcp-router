"""OTel (OpenTelemetry) wrapper functions for optional tracing and metrics.

Returns None for all calls when opentelemetry-api is not installed or when
telemetry is explicitly disabled via enable=False.
"""

try:
    from opentelemetry import trace as _otel_trace  # type: ignore[import-untyped]
except ImportError:
    _otel_trace = None

try:
    from opentelemetry import metrics as _otel_metrics  # type: ignore[import-untyped]
except ImportError:
    _otel_metrics = None


def _get_otel_provider(enable: bool, module: object, factory: str) -> object:
    """Return an OTel provider instance, or None when unavailable.

    Args:
        enable: When False, returns None immediately.
        module: The imported OTel module, or None if not installed.
        factory: Name of the factory method to call on the module.

    Returns:
        The OTel provider instance, or None if disabled or unavailable.
    """
    if not enable:
        return None
    if module is None:
        return None
    return getattr(module, factory)("fastapi-mcp-router")


def get_tracer(enable: bool) -> object:
    """Return an OpenTelemetry tracer, or None when tracing is unavailable.

    Args:
        enable: When False, tracing is disabled and None is returned.

    Returns:
        An OpenTelemetry Tracer instance, or None if enable is False or
        opentelemetry-api is not installed.

    Example:
        >>> tracer = get_tracer(enable=True)
        >>> if tracer is not None:
        ...     with tracer.start_as_current_span("my-span"):
        ...         pass
    """
    return _get_otel_provider(enable, _otel_trace, "get_tracer")


def get_meter(enable: bool) -> object:
    """Return an OpenTelemetry meter, or None when metrics are unavailable.

    Args:
        enable: When False, metrics are disabled and None is returned.

    Returns:
        An OpenTelemetry Meter instance, or None if enable is False or
        opentelemetry-api is not installed.

    Example:
        >>> meter = get_meter(enable=True)
        >>> if meter is not None:
        ...     counter = meter.create_counter("requests")
    """
    return _get_otel_provider(enable, _otel_metrics, "get_meter")
