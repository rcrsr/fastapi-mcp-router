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
    if not enable:
        return None
    if _otel_trace is None:
        return None
    return _otel_trace.get_tracer("fastapi-mcp-router")


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
    if not enable:
        return None
    if _otel_metrics is None:
        return None
    return _otel_metrics.get_meter("fastapi-mcp-router")
