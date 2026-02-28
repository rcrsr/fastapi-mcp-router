"""Tests for MCP logging/setLevel method and MCPLoggingHandler behaviour.

Covers AC-57, AC-58, AC-59, EC-22, EC-23.

AC-57: logging/setLevel sets minimum level per session
AC-58: Log notifications filtered by level, delivered via SSE
AC-59: Default log level is info
EC-22: Invalid log level → MCPError -32602
EC-23: Logging without stateful mode → MCPError -32601
"""

import httpx
import pytest
from fastapi import FastAPI

from fastapi_mcp_router import InMemorySessionStore, MCPToolRegistry, create_mcp_router
from fastapi_mcp_router.session import MCPLoggingHandler

# ---------------------------------------------------------------------------
# Shared constants and helpers
# ---------------------------------------------------------------------------

_HEADERS = {"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"}


def _make_stateful_app(store: InMemorySessionStore) -> FastAPI:
    """Create a minimal stateful FastAPI app backed by InMemorySessionStore.

    Args:
        store: InMemorySessionStore instance to use for session tracking.

    Returns:
        FastAPI app with stateful MCP router mounted at /mcp.
    """

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        """Accept requests with X-API-Key: test-key."""
        return api_key == "test-key"

    registry = MCPToolRegistry()
    router = create_mcp_router(registry, session_store=store, auth_validator=auth_validator)
    app = FastAPI()
    app.include_router(router, prefix="/mcp")
    return app


def _make_stateless_app() -> FastAPI:
    """Create a minimal stateless FastAPI app with no session_store.

    Returns:
        FastAPI app with stateless MCP router mounted at /mcp.
    """
    registry = MCPToolRegistry()
    router = create_mcp_router(registry)
    app = FastAPI()
    app.include_router(router, prefix="/mcp")
    return app


def _rpc(method: str, params: dict | None = None, rpc_id: int = 1) -> dict:
    """Build a JSON-RPC 2.0 request body.

    Args:
        method: JSON-RPC method name.
        params: Optional params dict.
        rpc_id: JSON-RPC request id, defaults to 1.

    Returns:
        JSON-RPC 2.0 formatted request dict.
    """
    body: dict = {"jsonrpc": "2.0", "id": rpc_id, "method": method}
    if params is not None:
        body["params"] = params
    return body


async def _initialize_session(client: httpx.AsyncClient) -> str:
    """Send initialize request and return the Mcp-Session-Id header value.

    Args:
        client: Configured httpx.AsyncClient for the stateful app.

    Returns:
        Session ID string from the Mcp-Session-Id response header.
    """
    resp = await client.post(
        "/mcp",
        json=_rpc("initialize", {"protocolVersion": "2025-06-18", "clientInfo": {}, "capabilities": {}}),
        headers=_HEADERS,
    )
    assert resp.status_code == 200, f"initialize failed: {resp.text}"
    session_id = resp.headers.get("mcp-session-id")
    assert session_id, "initialize response missing Mcp-Session-Id header"
    return session_id


# ---------------------------------------------------------------------------
# AC-57: logging/setLevel sets minimum level per session
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_logging_setLevel_success() -> None:
    """AC-57: logging/setLevel returns {} and does not error when level is valid."""
    store = InMemorySessionStore()
    app = _make_stateful_app(store)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _initialize_session(client)

        resp = await client.post(
            "/mcp",
            json=_rpc("logging/setLevel", {"level": "debug"}),
            headers={**_HEADERS, "Mcp-Session-Id": session_id},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "error" not in body, f"Unexpected error: {body}"
    assert "result" in body, f"Expected result, got: {body}"
    assert body["result"] == {}


# ---------------------------------------------------------------------------
# AC-59: Default log level is info
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_logging_default_level_info() -> None:
    """AC-59: Default minimum level is info; info-level messages enqueue, debug does not."""
    store = InMemorySessionStore()
    session = await store.create("2025-06-18", {}, {})
    session_id = session.session_id

    handler = MCPLoggingHandler(session_store=store)

    # info == default minimum: must enqueue
    await handler.log_message(session_id, "info", "myapp", "info message")
    # debug < info: must not enqueue
    await handler.log_message(session_id, "debug", "myapp", "debug message")

    messages = await store.dequeue_messages(session_id)
    assert len(messages) == 1, f"Expected 1 message, got {len(messages)}: {messages}"
    assert messages[0]["params"]["level"] == "info"


# ---------------------------------------------------------------------------
# AC-58: Log notifications filtered by level
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_logging_level_filtering() -> None:
    """AC-58: Messages below the configured minimum level are not enqueued."""
    store = InMemorySessionStore()
    session = await store.create("2025-06-18", {}, {})
    session_id = session.session_id

    handler = MCPLoggingHandler(session_store=store)
    handler.set_level(session_id, "warning")

    # below minimum: must not enqueue
    await handler.log_message(session_id, "info", "myapp", "info message")
    # equal to minimum: must enqueue
    await handler.log_message(session_id, "warning", "myapp", "warning message")
    # above minimum: must enqueue
    await handler.log_message(session_id, "error", "myapp", "error message")

    messages = await store.dequeue_messages(session_id)
    assert len(messages) == 2, f"Expected 2 messages, got {len(messages)}: {messages}"
    levels = [m["params"]["level"] for m in messages]
    assert "warning" in levels
    assert "error" in levels
    assert "info" not in levels


# ---------------------------------------------------------------------------
# AC-58: notifications/message format verification
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_logging_notification_format() -> None:
    """AC-58: Enqueued log message uses notifications/message JSON-RPC format."""
    store = InMemorySessionStore()
    session = await store.create("2025-06-18", {}, {})
    session_id = session.session_id

    handler = MCPLoggingHandler(session_store=store)
    await handler.log_message(session_id, "error", "audit", {"code": 42})

    messages = await store.dequeue_messages(session_id)
    assert len(messages) == 1, f"Expected 1 message, got {len(messages)}"
    msg = messages[0]

    assert msg["jsonrpc"] == "2.0"
    assert msg["method"] == "notifications/message"
    assert "params" in msg
    params = msg["params"]
    assert params["level"] == "error"
    assert params["logger"] == "audit"
    assert params["data"] == {"code": 42}


# ---------------------------------------------------------------------------
# EC-22: Invalid log level → MCPError -32602
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_logging_invalid_level_returns_32602() -> None:
    """EC-22: logging/setLevel with an unrecognised level returns MCPError -32602."""
    store = InMemorySessionStore()
    app = _make_stateful_app(store)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _initialize_session(client)

        resp = await client.post(
            "/mcp",
            json=_rpc("logging/setLevel", {"level": "invalid"}),
            headers={**_HEADERS, "Mcp-Session-Id": session_id},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "error" in body, f"Expected error, got: {body}"
    assert body["error"]["code"] == -32602


# ---------------------------------------------------------------------------
# EC-23: Logging without stateful mode → MCPError -32601
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_logging_requires_stateful() -> None:
    """EC-23: logging/setLevel on a stateless router returns MCPError -32601."""
    app = _make_stateless_app()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=_rpc("logging/setLevel", {"level": "debug"}),
            headers=_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "error" in body, f"Expected error, got: {body}"
    assert body["error"]["code"] == -32601
