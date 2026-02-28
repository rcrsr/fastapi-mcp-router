"""Tests for progress injection, delivery, and error handling.

Covers AC-41, AC-42, AC-43, AC-44, AC-45/EC-18, AC-50, AC-51.

AC-41: ProgressCallback injectable into tool handler signatures
AC-42: Stateful: progress emitted as notifications/progress via SSE
AC-43: Stateless: progress silently dropped (no-op, tool continues normally)
AC-44: Tools without progress parameter unaffected
AC-45/EC-18: Progress callback errors logged; tool continues
AC-50: notifications/cancelled stores cancellation token
AC-51: Tools can check is_cancelled() between iterations via progress callback
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI

from fastapi_mcp_router import InMemorySessionStore, MCPToolRegistry, ProgressCallback, create_mcp_router

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HEADERS = {"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"}


def _make_stateless_app(registry: MCPToolRegistry) -> FastAPI:
    """Create a minimal stateless FastAPI app with API-key auth."""

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        """Accept requests with X-API-Key: test-key."""
        return api_key == "test-key"

    app = FastAPI()
    router = create_mcp_router(registry, auth_validator=auth_validator)
    app.include_router(router, prefix="/mcp")
    return app


def _make_stateful_app(
    registry: MCPToolRegistry,
    store: InMemorySessionStore,
) -> FastAPI:
    """Create a stateful FastAPI app backed by InMemorySessionStore."""

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        """Accept requests with X-API-Key: test-key."""
        return api_key == "test-key"

    app = FastAPI()
    router = create_mcp_router(
        registry,
        session_store=store,
        stateful=True,
        auth_validator=auth_validator,
    )
    app.include_router(router, prefix="/mcp")
    return app


async def _initialize_session(client: httpx.AsyncClient) -> str:
    """Send initialize request and return the Mcp-Session-Id header value.

    Args:
        client: Configured httpx.AsyncClient for the stateful app.

    Returns:
        Session ID string from the Mcp-Session-Id response header.
    """
    resp = await client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "method": "initialize", "id": 1, "params": {}},
        headers=_HEADERS,
    )
    assert resp.status_code == 200, f"initialize failed: {resp.text}"
    session_id = resp.headers.get("mcp-session-id")
    assert session_id, "initialize response missing Mcp-Session-Id header"
    return session_id


# ---------------------------------------------------------------------------
# AC-41: ProgressCallback injected into tool signature
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_progress_callback_injected_as_callable() -> None:
    """AC-41: registry.call_tool() injects a callable for progress: ProgressCallback param."""
    registry = MCPToolRegistry()
    received: list[object] = []

    @registry.tool()
    async def progress_tool(progress: ProgressCallback) -> dict:
        """Tool that captures the injected progress callable."""
        received.append(progress)
        return {"ok": True}

    await registry.call_tool("progress_tool", {})

    assert len(received) == 1
    cb = received[0]
    assert callable(cb), "progress must be callable"
    # No-op: calling it must not raise
    progress_cb: ProgressCallback = cb  # type: ignore[assignment]
    await progress_cb(1, 10, "step")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_progress_callback_not_in_schema() -> None:
    """AC-41: progress: ProgressCallback is excluded from the tool's input schema."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def schema_tool(value: str, progress: ProgressCallback) -> dict:
        """Tool with both a user param and a progress param."""
        return {"value": value}

    tools = registry.list_tools()
    assert len(tools) == 1
    schema_props = tools[0]["inputSchema"]["properties"]  # type: ignore[index]
    assert "value" in schema_props, "user param must appear in schema"
    assert "progress" not in schema_props, "ProgressCallback must be excluded from schema"


# ---------------------------------------------------------------------------
# AC-42: Stateful progress emitted as notifications/progress via SSE
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stateful_progress_enqueued_as_notification() -> None:
    """AC-42: Tool calling progress in stateful mode enqueues notifications/progress."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def reporting_tool(progress: ProgressCallback) -> dict:
        """Tool that emits a progress notification."""
        await progress(1, 10, "step 1")
        return {"done": True}

    store = InMemorySessionStore()
    app = _make_stateful_app(registry, store)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _initialize_session(client)

        resp = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "reporting_tool",
                    "arguments": {},
                    "_meta": {"progressToken": "tok-1"},
                },
            },
            headers={**_HEADERS, "Mcp-Session-Id": session_id},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body, f"Expected result, got: {body}"

    messages = await store.dequeue_messages(session_id)
    assert len(messages) == 1
    notification = messages[0]
    assert notification["jsonrpc"] == "2.0"
    assert notification["method"] == "notifications/progress"
    params = notification["params"]
    assert params["progressToken"] == "tok-1"
    assert params["progress"] == 1
    assert params["total"] == 10
    assert params["message"] == "step 1"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stateful_progress_uses_rpc_id_when_no_token() -> None:
    """AC-42: When _meta.progressToken absent, request id is used as progressToken."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def reporting_tool_no_token(progress: ProgressCallback) -> dict:
        """Tool that emits a progress notification without an explicit token."""
        await progress(2, 5, None)
        return {"done": True}

    store = InMemorySessionStore()
    app = _make_stateful_app(registry, store)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _initialize_session(client)

        resp = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 42,
                "method": "tools/call",
                "params": {"name": "reporting_tool_no_token", "arguments": {}},
            },
            headers={**_HEADERS, "Mcp-Session-Id": session_id},
        )

    assert resp.status_code == 200
    messages = await store.dequeue_messages(session_id)
    assert len(messages) == 1
    params = messages[0]["params"]
    assert params["progressToken"] == "42"
    assert "message" not in params, "message key must be omitted when None"


# ---------------------------------------------------------------------------
# AC-43: Stateless progress silently dropped
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stateless_progress_dropped_tool_continues() -> None:
    """AC-43: Progress calls in stateless mode are no-ops; tool returns its result."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def stateless_progress_tool(progress: ProgressCallback) -> dict:
        """Tool that calls progress multiple times in stateless mode."""
        await progress(1, 3, "first")
        await progress(2, 3, "second")
        await progress(3, 3, "third")
        return {"result": "complete"}

    app = _make_stateless_app(registry)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "stateless_progress_tool", "arguments": {}},
            },
            headers=_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body, f"Expected result key, got: {body}"
    assert body["result"].get("isError") is not True
    content_text = body["result"]["content"][0]["text"]
    content = json.loads(content_text)
    assert content == {"result": "complete"}


# ---------------------------------------------------------------------------
# AC-44: Tools without progress parameter unaffected
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_without_progress_stateless_unaffected() -> None:
    """AC-44: Tool with no progress parameter works normally in stateless mode."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def no_progress_tool(value: str) -> dict:
        """Tool with no progress parameter."""
        return {"echo": value}

    app = _make_stateless_app(registry)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "no_progress_tool", "arguments": {"value": "hello"}},
            },
            headers=_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body
    assert body["result"].get("isError") is not True
    content = json.loads(body["result"]["content"][0]["text"])
    assert content == {"echo": "hello"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_without_progress_stateful_unaffected() -> None:
    """AC-44: Tool with no progress parameter works normally in stateful mode."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def no_progress_stateful(value: str) -> dict:
        """Tool with no progress parameter in stateful mode."""
        return {"echo": value}

    store = InMemorySessionStore()
    app = _make_stateful_app(registry, store)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _initialize_session(client)

        resp = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "no_progress_stateful", "arguments": {"value": "world"}},
            },
            headers={**_HEADERS, "Mcp-Session-Id": session_id},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body
    assert body["result"].get("isError") is not True
    content = json.loads(body["result"]["content"][0]["text"])
    assert content == {"echo": "world"}


# ---------------------------------------------------------------------------
# AC-45 / EC-18: Progress callback error logged; tool continues
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_progress_error_does_not_break_tool() -> None:
    """AC-45/EC-18: enqueue_message raising RuntimeError is swallowed; tool returns result."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def error_progress_tool(progress: ProgressCallback) -> dict:
        """Tool that calls progress; enqueue raises RuntimeError."""
        await progress(1, 2, "step")
        return {"finished": True}

    store = InMemorySessionStore()
    app = _make_stateful_app(registry, store)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _initialize_session(client)

        with patch.object(store, "enqueue_message", new_callable=AsyncMock) as mock_enqueue:
            mock_enqueue.side_effect = RuntimeError("storage failure")

            resp = await client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": "error_progress_tool", "arguments": {}},
                },
                headers={**_HEADERS, "Mcp-Session-Id": session_id},
            )

    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body, f"Expected result, got: {body}"
    assert body["result"].get("isError") is not True
    content = json.loads(body["result"]["content"][0]["text"])
    assert content == {"finished": True}


# ---------------------------------------------------------------------------
# AC-50: notifications/cancelled stores cancellation token
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cancelled_notification_stores_token() -> None:
    """AC-50: notifications/cancelled marks the request ID as cancelled in the tracker.

    Verifies the token is stored by calling a progress tool with the same
    progressToken and confirming the callback raises asyncio.CancelledError.
    """
    registry = MCPToolRegistry()
    cancelled_error_raised: list[bool] = []

    @registry.tool()
    async def verify_cancel_tool(progress: ProgressCallback) -> dict:
        """Tool that detects CancelledError from the progress callback."""
        try:
            await progress(1, 10, "check")
            cancelled_error_raised.append(False)
        except asyncio.CancelledError:
            cancelled_error_raised.append(True)
        return {"checked": True}

    store = InMemorySessionStore()
    app = _make_stateful_app(registry, store)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _initialize_session(client)

        # Send the cancellation notification for "req-99"
        cancel_resp = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "notifications/cancelled",
                "params": {"requestId": "req-99"},
            },
            headers={**_HEADERS, "Mcp-Session-Id": session_id},
        )
        assert cancel_resp.status_code == 202

        # Verify the token was stored: progress callback must raise CancelledError
        # when called with progressToken matching the cancelled requestId.
        resp = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": "tool-call-99",
                "method": "tools/call",
                "params": {
                    "name": "verify_cancel_tool",
                    "arguments": {},
                    "_meta": {"progressToken": "req-99"},
                },
            },
            headers={**_HEADERS, "Mcp-Session-Id": session_id},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body, f"Expected result, got: {body}"
    assert cancelled_error_raised == [True], "Token not stored: progress callback did not raise CancelledError"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancelled_notification_stateless_returns_202() -> None:
    """AC-50: notifications/cancelled in stateless mode returns 202 without error."""
    registry = MCPToolRegistry()
    app = _make_stateless_app(registry)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        cancel_resp = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "notifications/cancelled",
                "params": {"requestId": "req-1"},
            },
            headers=_HEADERS,
        )

    assert cancel_resp.status_code == 202


# ---------------------------------------------------------------------------
# AC-51: Tools can check is_cancelled() via progress callback
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_progress_callback_raises_cancelled_error_when_token_cancelled() -> None:
    """AC-51: progress callback raises asyncio.CancelledError when requestId is cancelled."""
    registry = MCPToolRegistry()
    cancelled_error_raised: list[bool] = []

    @registry.tool()
    async def cancellable_tool(progress: ProgressCallback) -> dict:
        """Tool that catches CancelledError from progress callback."""
        try:
            await progress(1, 10, "check")
            cancelled_error_raised.append(False)
        except asyncio.CancelledError:
            cancelled_error_raised.append(True)
        return {"checked": True}

    store = InMemorySessionStore()
    app = _make_stateful_app(registry, store)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _initialize_session(client)

        # Send cancellation for the progress token we will use
        await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "notifications/cancelled",
                "params": {"requestId": "tok-cancel"},
            },
            headers={**_HEADERS, "Mcp-Session-Id": session_id},
        )

        # Call the tool with the same progress token
        resp = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "cancellable_tool",
                    "arguments": {},
                    "_meta": {"progressToken": "tok-cancel"},
                },
            },
            headers={**_HEADERS, "Mcp-Session-Id": session_id},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body, f"Expected result, got: {body}"
    # Tool must have received CancelledError from the progress callback
    assert cancelled_error_raised == [True], "Expected CancelledError to be raised in progress callback"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_progress_callback_no_cancelled_error_when_not_cancelled() -> None:
    """AC-51: progress callback does not raise when request is not cancelled."""
    registry = MCPToolRegistry()
    cancelled_error_raised: list[bool] = []

    @registry.tool()
    async def non_cancelled_tool(progress: ProgressCallback) -> dict:
        """Tool that checks if progress raises unexpectedly."""
        try:
            await progress(1, 10, "step")
            cancelled_error_raised.append(False)
        except asyncio.CancelledError:
            cancelled_error_raised.append(True)
        return {"checked": True}

    store = InMemorySessionStore()
    app = _make_stateful_app(registry, store)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _initialize_session(client)

        resp = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {
                    "name": "non_cancelled_tool",
                    "arguments": {},
                    "_meta": {"progressToken": "tok-active"},
                },
            },
            headers={**_HEADERS, "Mcp-Session-Id": session_id},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body
    assert cancelled_error_raised == [False], "CancelledError must not be raised when not cancelled"
