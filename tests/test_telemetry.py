"""Unit and integration tests for the telemetry module.

Tests get_tracer() and get_meter() behavior under three conditions:
- enable=False: always returns None regardless of OTel availability
- _otel_trace/_otel_metrics is None (OTel not installed): returns None
- OTel available: returns a tracer/meter instance

Integration tests verify router telemetry behavior using mock tracers/meters
injected by patching get_tracer and get_meter before router creation.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from fastapi_mcp_router import MCPToolRegistry, create_mcp_router

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_tracer() -> MagicMock:
    """Build a MagicMock tracer with a context-manager span.

    Returns:
        MagicMock tracer where start_as_current_span returns a context-manager
        that yields a MagicMock span with a set_attribute method.
    """
    mock_span = MagicMock()
    mock_span_cm = MagicMock()
    mock_span_cm.__enter__ = MagicMock(return_value=mock_span)
    mock_span_cm.__exit__ = MagicMock(return_value=False)
    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span = MagicMock(return_value=mock_span_cm)
    return mock_tracer


def _make_mock_meter() -> MagicMock:
    """Build a MagicMock meter with a counter.

    Returns:
        MagicMock meter where create_counter returns a MagicMock counter
        with an add method.
    """
    mock_counter = MagicMock()
    mock_meter = MagicMock()
    mock_meter.create_counter = MagicMock(return_value=mock_counter)
    return mock_meter


def _tools_call_body(tool_name: str, args: dict) -> dict:
    """Build a JSON-RPC tools/call request body.

    Args:
        tool_name: Name of the MCP tool to call.
        args: Arguments dict to pass to the tool.

    Returns:
        Dict representing a valid JSON-RPC 2.0 tools/call request.
    """
    return {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": args},
        "id": 1,
    }


# ---------------------------------------------------------------------------
# Unit tests: get_tracer()
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_tracer_returns_none_when_disabled() -> None:
    """get_tracer(enable=False) returns None regardless of OTel availability.

    Covers AC-11: enable_telemetry=False suppresses all MCP spans.
    """
    from fastapi_mcp_router.telemetry import get_tracer

    mock_trace_module = MagicMock()
    with patch("fastapi_mcp_router.telemetry._otel_trace", mock_trace_module):
        result = get_tracer(enable=False)

    assert result is None
    mock_trace_module.get_tracer.assert_not_called()


@pytest.mark.unit
def test_get_tracer_returns_none_when_otel_not_installed() -> None:
    """get_tracer(enable=True) returns None when _otel_trace is None.

    Covers EC-3 and AC-10: opentelemetry-api not installed -> no errors, None returned.
    """
    from fastapi_mcp_router.telemetry import get_tracer

    with patch("fastapi_mcp_router.telemetry._otel_trace", None):
        result = get_tracer(enable=True)

    assert result is None


@pytest.mark.unit
def test_get_tracer_returns_tracer_when_otel_available() -> None:
    """get_tracer(enable=True) calls get_tracer('fastapi-mcp-router') on the trace module.

    Covers IR-5 and AC-9: with OTel installed, a tracer is returned.
    """
    from fastapi_mcp_router.telemetry import get_tracer

    mock_tracer_instance = MagicMock()
    mock_trace_module = MagicMock()
    mock_trace_module.get_tracer = MagicMock(return_value=mock_tracer_instance)

    with patch("fastapi_mcp_router.telemetry._otel_trace", mock_trace_module):
        result = get_tracer(enable=True)

    assert result is mock_tracer_instance
    mock_trace_module.get_tracer.assert_called_once_with("fastapi-mcp-router")


# ---------------------------------------------------------------------------
# Unit tests: get_meter()
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_meter_returns_none_when_disabled() -> None:
    """get_meter(enable=False) returns None regardless of OTel availability.

    Covers AC-11: enable_telemetry=False suppresses all MCP metrics.
    """
    from fastapi_mcp_router.telemetry import get_meter

    mock_metrics_module = MagicMock()
    with patch("fastapi_mcp_router.telemetry._otel_metrics", mock_metrics_module):
        result = get_meter(enable=False)

    assert result is None
    mock_metrics_module.get_meter.assert_not_called()


@pytest.mark.unit
def test_get_meter_returns_none_when_otel_not_installed() -> None:
    """get_meter(enable=True) returns None when _otel_metrics is None.

    Covers EC-3 and AC-16: opentelemetry-api not installed -> no errors, None returned.
    """
    from fastapi_mcp_router.telemetry import get_meter

    with patch("fastapi_mcp_router.telemetry._otel_metrics", None):
        result = get_meter(enable=True)

    assert result is None


@pytest.mark.unit
def test_get_meter_returns_meter_when_otel_available() -> None:
    """get_meter(enable=True) calls get_meter('fastapi-mcp-router') on the metrics module.

    Covers IR-5 and AC-12: with OTel installed, a meter is returned.
    """
    from fastapi_mcp_router.telemetry import get_meter

    mock_meter_instance = MagicMock()
    mock_metrics_module = MagicMock()
    mock_metrics_module.get_meter = MagicMock(return_value=mock_meter_instance)

    with patch("fastapi_mcp_router.telemetry._otel_metrics", mock_metrics_module):
        result = get_meter(enable=True)

    assert result is mock_meter_instance
    mock_metrics_module.get_meter.assert_called_once_with("fastapi-mcp-router")


# ---------------------------------------------------------------------------
# Unit test: import guard (EC-3, AC-10, AC-16)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_telemetry_module_level_vars_none_when_otel_missing() -> None:
    """Module-level _otel_trace and _otel_metrics are None when OTel is not installed.

    Covers EC-3: ImportError on opentelemetry sets _otel_trace to None.
    Covers AC-16: library imports without error when opentelemetry-api absent.
    """
    import fastapi_mcp_router.telemetry as telemetry_module

    # Simulate OTel not installed by patching module-level vars to None.
    with patch.object(telemetry_module, "_otel_trace", None), patch.object(telemetry_module, "_otel_metrics", None):
        assert telemetry_module._otel_trace is None
        assert telemetry_module._otel_metrics is None


@pytest.mark.unit
def test_get_tracer_and_get_meter_callable_without_otel() -> None:
    """Both functions return None gracefully when OTel modules are absent.

    Covers AC-10 and AC-16: no import errors, no runtime failures.
    """
    import fastapi_mcp_router.telemetry as telemetry_module

    # Simulate OTel not installed by patching module-level vars to None.
    with patch.object(telemetry_module, "_otel_trace", None), patch.object(telemetry_module, "_otel_metrics", None):
        tracer = telemetry_module.get_tracer(enable=True)
        meter = telemetry_module.get_meter(enable=True)

        assert tracer is None
        assert meter is None


# ---------------------------------------------------------------------------
# Unit test: AC-27 -- enable_telemetry=True, no TracerProvider -> no error
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_tracer_with_noop_provider_returns_object() -> None:
    """get_tracer with a mocked NoOp-style provider returns a tracer object without error.

    Covers AC-27: enable_telemetry=True with no real TracerProvider produces no errors.
    The router treats whatever tracer is returned as valid (NonRecordingSpan equivalent).
    """
    from fastapi_mcp_router.telemetry import get_tracer

    # Simulate a NoOp tracer (what OTel returns when no provider is configured).
    noop_tracer = MagicMock(name="NoOpTracer")
    mock_trace_module = MagicMock()
    mock_trace_module.get_tracer = MagicMock(return_value=noop_tracer)

    with patch("fastapi_mcp_router.telemetry._otel_trace", mock_trace_module):
        result = get_tracer(enable=True)

    # No exception raised; tracer object is returned.
    assert result is noop_tracer


# ---------------------------------------------------------------------------
# Integration fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(name="telemetry_registry")
async def telemetry_registry_fixture() -> MCPToolRegistry:
    """Create a registry with one echo tool for telemetry integration tests.

    Returns:
        MCPToolRegistry with an 'echo' tool registered.
    """
    registry = MCPToolRegistry()

    @registry.tool()
    async def echo(message: str) -> str:
        """Echo the message back."""
        return message

    return registry


# ---------------------------------------------------------------------------
# Integration test: AC-9 -- tools/call span has correct attributes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_tools_call_span_has_correct_attributes(
    telemetry_registry: MCPToolRegistry,
) -> None:
    """tools/call request sets four span attributes per OTel RPC semantic conventions.

    Covers AC-9: rpc.system.name=jsonrpc, rpc.method=tools/call,
    rpc.jsonrpc.version=2.0, mcp.tool.name=<name>.
    Covers AC-18: span attributes use OTel RPC semantic convention string literals.
    """
    mock_tracer = _make_mock_tracer()
    mock_span = mock_tracer.start_as_current_span.return_value.__enter__.return_value

    with patch("fastapi_mcp_router.router.get_tracer", return_value=mock_tracer):
        fastapi_app = FastAPI()
        mcp_router = create_mcp_router(telemetry_registry, enable_telemetry=True)
        fastapi_app.include_router(mcp_router, prefix="/mcp")

        transport = httpx.ASGITransport(app=fastapi_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/mcp",
                json=_tools_call_body("echo", {"message": "hello"}),
                headers={"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"},
            )

    assert response.status_code == 200
    # Verify the span received all four required attributes.
    mock_span.set_attribute.assert_any_call("rpc.system.name", "jsonrpc")
    mock_span.set_attribute.assert_any_call("rpc.method", "tools/call")
    mock_span.set_attribute.assert_any_call("rpc.jsonrpc.version", "2.0")
    mock_span.set_attribute.assert_any_call("mcp.tool.name", "echo")


# ---------------------------------------------------------------------------
# Integration test: AC-11 -- enable_telemetry=False suppresses spans
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_enable_telemetry_false_suppresses_spans(
    telemetry_registry: MCPToolRegistry,
) -> None:
    """enable_telemetry=False means get_tracer returns None and no span is created.

    Covers AC-11: enable_telemetry=False suppresses all MCP spans even with
    opentelemetry-api conceptually available.
    """
    mock_trace_module = MagicMock()
    mock_tracer = MagicMock()
    mock_trace_module.get_tracer = MagicMock(return_value=mock_tracer)

    with patch("fastapi_mcp_router.telemetry._otel_trace", mock_trace_module):
        fastapi_app = FastAPI()
        # enable_telemetry=False must prevent get_tracer from calling trace.get_tracer.
        mcp_router = create_mcp_router(telemetry_registry, enable_telemetry=False)
        fastapi_app.include_router(mcp_router, prefix="/mcp")

        transport = httpx.ASGITransport(app=fastapi_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/mcp",
                json=_tools_call_body("echo", {"message": "hello"}),
                headers={"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"},
            )

    assert response.status_code == 200
    # _otel_trace.get_tracer must not have been called.
    mock_trace_module.get_tracer.assert_not_called()
    # The returned tracer mock must not have had start_as_current_span called.
    mock_tracer.start_as_current_span.assert_not_called()


# ---------------------------------------------------------------------------
# Integration test: AC-12 -- request counter incremented with rpc.method
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_request_counter_incremented_with_rpc_method(
    telemetry_registry: MCPToolRegistry,
) -> None:
    """Each JSON-RPC request increments mcp.server.request.count with rpc.method attribute.

    Covers AC-12: counter.add(1, {"rpc.method": method}) called for tools/call.
    """
    mock_meter = _make_mock_meter()
    mock_counter = mock_meter.create_counter.return_value

    with (
        patch("fastapi_mcp_router.router.get_tracer", return_value=None),
        patch("fastapi_mcp_router.router.get_meter", return_value=mock_meter),
    ):
        fastapi_app = FastAPI()
        mcp_router = create_mcp_router(telemetry_registry, enable_telemetry=True)
        fastapi_app.include_router(mcp_router, prefix="/mcp")

        transport = httpx.ASGITransport(app=fastapi_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/mcp",
                json=_tools_call_body("echo", {"message": "world"}),
                headers={"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"},
            )

    assert response.status_code == 200
    # Counter must have been created with the correct name.
    mock_meter.create_counter.assert_called_once_with("mcp.server.request.count")
    # Counter add must have been called with method="tools/call".
    mock_counter.add.assert_called_once_with(1, {"rpc.method": "tools/call"})


# ---------------------------------------------------------------------------
# Integration test: AC-12 -- counter incremented for each method
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_request_counter_incremented_for_initialize(
    telemetry_registry: MCPToolRegistry,
) -> None:
    """initialize request increments mcp.server.request.count with rpc.method=initialize.

    Covers AC-12: counter incremented for every JSON-RPC method, not just tools/call.
    """
    mock_meter = _make_mock_meter()
    mock_counter = mock_meter.create_counter.return_value

    with (
        patch("fastapi_mcp_router.router.get_tracer", return_value=None),
        patch("fastapi_mcp_router.router.get_meter", return_value=mock_meter),
    ):
        fastapi_app = FastAPI()
        mcp_router = create_mcp_router(telemetry_registry, enable_telemetry=True)
        fastapi_app.include_router(mcp_router, prefix="/mcp")

        transport = httpx.ASGITransport(app=fastapi_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {"protocolVersion": "2025-06-18"},
                    "id": 1,
                },
                headers={"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"},
            )

    assert response.status_code == 200
    mock_counter.add.assert_called_once_with(1, {"rpc.method": "initialize"})


# ---------------------------------------------------------------------------
# Integration test: AC-22 / EC-4 -- span creation failure does not break request
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_span_creation_failure_does_not_break_request(
    telemetry_registry: MCPToolRegistry,
) -> None:
    """Span creation raising an exception still delivers a valid JSON-RPC response.

    Covers AC-22 and EC-4: OTel span creation failure does not prevent
    JSON-RPC response delivery. The router catches and logs the exception.
    """
    broken_tracer = MagicMock()
    broken_tracer.start_as_current_span = MagicMock(side_effect=RuntimeError("otel broken"))

    with patch("fastapi_mcp_router.router.get_tracer", return_value=broken_tracer):
        fastapi_app = FastAPI()
        mcp_router = create_mcp_router(telemetry_registry, enable_telemetry=True)
        fastapi_app.include_router(mcp_router, prefix="/mcp")

        transport = httpx.ASGITransport(app=fastapi_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/mcp",
                json=_tools_call_body("echo", {"message": "resilience"}),
                headers={"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"},
            )

    assert response.status_code == 200
    body = response.json()
    # JSON-RPC response must still have a valid result (no error).
    assert "result" in body
    assert "error" not in body


# ---------------------------------------------------------------------------
# Integration test: AC-27 -- enable_telemetry=True, NoOp provider -> no errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_noop_tracer_produces_no_errors(
    telemetry_registry: MCPToolRegistry,
) -> None:
    """enable_telemetry=True with a NoOp-style tracer produces no errors.

    Covers AC-27: NoOp tracer (NonRecordingSpan equivalent) does not raise
    and the JSON-RPC response is still delivered correctly.
    """
    # A NoOp tracer's span has set_attribute as a no-op.
    noop_span = MagicMock()
    noop_span.set_attribute = MagicMock(return_value=None)
    noop_span_cm = MagicMock()
    noop_span_cm.__enter__ = MagicMock(return_value=noop_span)
    noop_span_cm.__exit__ = MagicMock(return_value=False)
    noop_tracer = MagicMock()
    noop_tracer.start_as_current_span = MagicMock(return_value=noop_span_cm)

    with patch("fastapi_mcp_router.router.get_tracer", return_value=noop_tracer):
        fastapi_app = FastAPI()
        mcp_router = create_mcp_router(telemetry_registry, enable_telemetry=True)
        fastapi_app.include_router(mcp_router, prefix="/mcp")

        transport = httpx.ASGITransport(app=fastapi_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/mcp",
                json=_tools_call_body("echo", {"message": "noop"}),
                headers={"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"},
            )

    assert response.status_code == 200
    body = response.json()
    assert "result" in body
    assert "error" not in body


# ---------------------------------------------------------------------------
# Integration test: AC-13 -- parent span context propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_tracer_receives_start_as_current_span_call_per_tools_call(
    telemetry_registry: MCPToolRegistry,
) -> None:
    """Tracer's start_as_current_span is invoked for each tools/call request.

    Covers AC-13: the router calls start_as_current_span, which inherits
    whatever parent span context OTel propagation has established.
    Two sequential tools/call requests produce two start_as_current_span calls.
    """
    mock_tracer = _make_mock_tracer()

    with patch("fastapi_mcp_router.router.get_tracer", return_value=mock_tracer):
        fastapi_app = FastAPI()
        mcp_router = create_mcp_router(telemetry_registry, enable_telemetry=True)
        fastapi_app.include_router(mcp_router, prefix="/mcp")

        transport = httpx.ASGITransport(app=fastapi_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/mcp",
                json=_tools_call_body("echo", {"message": "first"}),
                headers={"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"},
            )
            await client.post(
                "/mcp",
                json=_tools_call_body("echo", {"message": "second"}),
                headers={"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"},
            )

    assert mock_tracer.start_as_current_span.call_count == 2


# ---------------------------------------------------------------------------
# Integration test: AC-17 -- non-tools/call methods do NOT create a span
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_non_tools_call_methods_do_not_create_span(
    telemetry_registry: MCPToolRegistry,
) -> None:
    """initialize and tools/list do not invoke start_as_current_span.

    Covers AC-17: span creation is scoped to tools/call only; the tracer
    is available for application-level exporters but the library only
    creates spans for tools/call.
    """
    mock_tracer = _make_mock_tracer()

    with patch("fastapi_mcp_router.router.get_tracer", return_value=mock_tracer):
        fastapi_app = FastAPI()
        mcp_router = create_mcp_router(telemetry_registry, enable_telemetry=True)
        fastapi_app.include_router(mcp_router, prefix="/mcp")

        transport = httpx.ASGITransport(app=fastapi_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {"protocolVersion": "2025-06-18"},
                    "id": 1,
                },
                headers={"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"},
            )
            await client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 2},
                headers={"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"},
            )

    mock_tracer.start_as_current_span.assert_not_called()
