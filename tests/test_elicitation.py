"""Tests for ElicitationManager: request enqueue, response correlation, timeout, and mode checks.

Covers AC-63, AC-64, AC-65, AC-66, AC-86, EC-24, EC-25.

AC-63: elicitation/create sends request to client via SSE
AC-64: Elicitation client response includes action + optional content
AC-65: Elicitation response validated against requestedSchema
AC-66: Elicitation requires stateful mode
AC-86: Elicitation without stateful mode returns MCPError -32601
EC-24: Elicitation without stateful mode → MCPError -32601
EC-25: Elicitation timeout → MCPError -32603
"""

import asyncio
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI

from fastapi_mcp_router import InMemorySessionStore, MCPError, MCPToolRegistry, create_mcp_router
from fastapi_mcp_router.router import ElicitationManager

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
    """Create a stateful FastAPI app with elicitation support."""

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
# AC-66 / AC-86 / EC-24: elicitation/create without stateful mode → -32601
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_elicitation_create_stateless_returns_32601() -> None:
    """AC-66/AC-86/EC-24: POST elicitation/create on stateless router returns MCPError -32601."""
    registry = MCPToolRegistry()
    app = _make_stateless_app(registry)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "elicitation/create",
                "params": {
                    "message": "Please confirm",
                    "requestedSchema": {"type": "object", "properties": {}},
                },
            },
            headers=_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "error" in body, f"Expected error, got: {body}"
    assert body["error"]["code"] == -32601


@pytest.mark.integration
@pytest.mark.asyncio
async def test_elicitation_create_stateless_error_message() -> None:
    """AC-86: Stateless elicitation error message references stateful mode requirement."""
    registry = MCPToolRegistry()
    app = _make_stateless_app(registry)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "elicitation/create",
                "params": {"message": "hi", "requestedSchema": {}},
            },
            headers=_HEADERS,
        )

    body = resp.json()
    assert "stateful" in body["error"]["message"].lower()


# ---------------------------------------------------------------------------
# AC-63 / AC-64: elicitation/create enqueues SSE request; client accepts
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_elicitation_create_enqueues_request() -> None:
    """AC-63: ElicitationManager.create() enqueues an elicitation/create JSON-RPC request."""
    store = InMemorySessionStore()
    session = await store.create(
        protocol_version="2025-06-18",
        client_info={},
        capabilities={},
    )
    manager = ElicitationManager(session_store=store)

    async def respond_after_dequeue() -> None:
        """Dequeue the elicitation request and resolve the future."""
        await asyncio.sleep(0.01)
        msgs = await store.dequeue_messages(session.session_id)
        assert len(msgs) == 1, f"Expected 1 queued message, got {len(msgs)}"
        assert msgs[0]["method"] == "elicitation/create"
        request_id = msgs[0]["id"]
        manager.handle_response(
            request_id,
            {"action": "accept", "content": {"confirmed": True}},
        )

    task = asyncio.create_task(respond_after_dequeue())
    await manager.create(
        session_id=session.session_id,
        message="Please confirm your choice",
        requested_schema={"type": "object", "properties": {"confirmed": {"type": "boolean"}}},
    )
    await task


@pytest.mark.unit
@pytest.mark.asyncio
async def test_elicitation_create_request_params_structure() -> None:
    """AC-63: Enqueued request contains jsonrpc, method, id, and params fields."""
    store = InMemorySessionStore()
    session = await store.create(
        protocol_version="2025-06-18",
        client_info={},
        capabilities={},
    )
    manager = ElicitationManager(session_store=store)

    async def respond() -> None:
        """Resolve the pending future so create() does not block."""
        await asyncio.sleep(0.01)
        msgs = await store.dequeue_messages(session.session_id)
        assert len(msgs) == 1
        req = msgs[0]
        assert req.get("jsonrpc") == "2.0"
        assert req.get("method") == "elicitation/create"
        assert "id" in req
        assert "params" in req
        assert req["params"]["message"] == "Confirm?"
        assert "requestedSchema" in req["params"]
        request_id = req["id"]
        manager.handle_response(request_id, {"action": "decline"})

    task = asyncio.create_task(respond())
    await manager.create(
        session_id=session.session_id,
        message="Confirm?",
        requested_schema={"type": "object"},
    )
    await task


@pytest.mark.unit
@pytest.mark.asyncio
async def test_elicitation_response_accept_with_content() -> None:
    """AC-64: Elicitation client response includes action='accept' and content dict."""
    store = InMemorySessionStore()
    session = await store.create(
        protocol_version="2025-06-18",
        client_info={},
        capabilities={},
    )
    manager = ElicitationManager(session_store=store)
    expected_content = {"name": "Alice", "age": 30}

    async def respond_accept() -> None:
        """Resolve future with accept action and content."""
        await asyncio.sleep(0.01)
        msgs = await store.dequeue_messages(session.session_id)
        request_id = msgs[0]["id"]
        manager.handle_response(
            request_id,
            {"action": "accept", "content": expected_content},
        )

    task = asyncio.create_task(respond_accept())
    result = await manager.create(
        session_id=session.session_id,
        message="Enter your details",
        requested_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
        },
    )
    await task

    assert result["action"] == "accept"
    assert result["content"] == expected_content


@pytest.mark.unit
@pytest.mark.asyncio
async def test_elicitation_response_decline_no_content() -> None:
    """AC-64: Elicitation client response includes action='decline' with no content."""
    store = InMemorySessionStore()
    session = await store.create(
        protocol_version="2025-06-18",
        client_info={},
        capabilities={},
    )
    manager = ElicitationManager(session_store=store)

    async def respond_decline() -> None:
        """Resolve future with decline action."""
        await asyncio.sleep(0.01)
        msgs = await store.dequeue_messages(session.session_id)
        request_id = msgs[0]["id"]
        manager.handle_response(request_id, {"action": "decline"})

    task = asyncio.create_task(respond_decline())
    result = await manager.create(
        session_id=session.session_id,
        message="Do you accept?",
        requested_schema={"type": "object"},
    )
    await task

    assert result["action"] == "decline"
    assert result["content"] is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_elicitation_handle_response_unknown_id_is_silent() -> None:
    """AC-64: handle_response() with unknown request_id does not raise."""
    store = InMemorySessionStore()
    manager = ElicitationManager(session_store=store)
    # Must not raise for an unknown or stale request ID
    manager.handle_response("unknown-id", {"action": "accept", "content": None})


# ---------------------------------------------------------------------------
# AC-65: Elicitation response validated against requestedSchema
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_elicitation_schema_validation_rejects_wrong_type() -> None:
    """AC-65: content that violates requestedSchema (wrong type) raises MCPError -32602."""
    store = InMemorySessionStore()
    session = await store.create(
        protocol_version="2025-06-18",
        client_info={},
        capabilities={},
    )
    manager = ElicitationManager(session_store=store)

    async def respond_with_invalid_content() -> None:
        """Resolve future with accept action but content of the wrong type (string not object)."""
        await asyncio.sleep(0.01)
        msgs = await store.dequeue_messages(session.session_id)
        request_id = msgs[0]["id"]
        # Schema requires object, but we send a string — validation must reject it.
        manager.handle_response(request_id, {"action": "accept", "content": "not-an-object"})

    task = asyncio.create_task(respond_with_invalid_content())
    with pytest.raises(MCPError) as exc_info:
        await manager.create(
            session_id=session.session_id,
            message="Provide object",
            requested_schema={"type": "object", "properties": {}},
        )
    await task

    assert exc_info.value.code == -32602


@pytest.mark.unit
@pytest.mark.asyncio
async def test_elicitation_schema_validation_rejects_missing_required_field() -> None:
    """AC-65: content missing a required field raises MCPError -32602."""
    store = InMemorySessionStore()
    session = await store.create(
        protocol_version="2025-06-18",
        client_info={},
        capabilities={},
    )
    manager = ElicitationManager(session_store=store)

    async def respond_missing_field() -> None:
        """Resolve future with accept action but content missing required field."""
        await asyncio.sleep(0.01)
        msgs = await store.dequeue_messages(session.session_id)
        request_id = msgs[0]["id"]
        # Schema requires 'email' field, but content only has 'name'.
        manager.handle_response(request_id, {"action": "accept", "content": {"name": "Bob"}})

    task = asyncio.create_task(respond_missing_field())
    with pytest.raises(MCPError) as exc_info:
        await manager.create(
            session_id=session.session_id,
            message="Provide details",
            requested_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}, "email": {"type": "string"}},
                "required": ["name", "email"],
            },
        )
    await task

    assert exc_info.value.code == -32602
    assert "email" in exc_info.value.message


@pytest.mark.unit
@pytest.mark.asyncio
async def test_elicitation_schema_validation_skipped_for_decline() -> None:
    """AC-65: Schema validation is skipped for action='decline' (no content to validate)."""
    store = InMemorySessionStore()
    session = await store.create(
        protocol_version="2025-06-18",
        client_info={},
        capabilities={},
    )
    manager = ElicitationManager(session_store=store)

    async def respond_decline() -> None:
        """Resolve future with decline action — validation must not run."""
        await asyncio.sleep(0.01)
        msgs = await store.dequeue_messages(session.session_id)
        request_id = msgs[0]["id"]
        manager.handle_response(request_id, {"action": "decline"})

    task = asyncio.create_task(respond_decline())
    # Must not raise even though schema has required fields.
    result = await manager.create(
        session_id=session.session_id,
        message="Provide email",
        requested_schema={
            "type": "object",
            "properties": {"email": {"type": "string"}},
            "required": ["email"],
        },
    )
    await task

    assert result["action"] == "decline"


# ---------------------------------------------------------------------------
# EC-25: Elicitation timeout → MCPError -32603
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_elicitation_timeout_raises_mcp_error() -> None:
    """EC-25: create() raises MCPError(-32603) when client does not respond within timeout."""
    store = InMemorySessionStore()
    session = await store.create(
        protocol_version="2025-06-18",
        client_info={},
        capabilities={},
    )
    manager = ElicitationManager(session_store=store)

    with patch("fastapi_mcp_router.router._ELICITATION_TIMEOUT", 0.001), pytest.raises(MCPError) as exc_info:
        await manager.create(
            session_id=session.session_id,
            message="Will you respond?",
            requested_schema={"type": "object"},
        )

    assert exc_info.value.code == -32603


@pytest.mark.unit
@pytest.mark.asyncio
async def test_elicitation_timeout_error_message_contains_timed_out() -> None:
    """EC-25: Timeout MCPError message indicates that the request timed out."""
    store = InMemorySessionStore()
    session = await store.create(
        protocol_version="2025-06-18",
        client_info={},
        capabilities={},
    )
    manager = ElicitationManager(session_store=store)

    with patch("fastapi_mcp_router.router._ELICITATION_TIMEOUT", 0.001), pytest.raises(MCPError) as exc_info:
        await manager.create(
            session_id=session.session_id,
            message="Timeout test",
            requested_schema={},
        )

    assert "timed out" in exc_info.value.message.lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_elicitation_timeout_cleans_up_pending() -> None:
    """EC-25: After timeout, the request_id is removed from _pending."""
    store = InMemorySessionStore()
    session = await store.create(
        protocol_version="2025-06-18",
        client_info={},
        capabilities={},
    )
    manager = ElicitationManager(session_store=store)

    with patch("fastapi_mcp_router.router._ELICITATION_TIMEOUT", 0.001), pytest.raises(MCPError):
        await manager.create(
            session_id=session.session_id,
            message="Cleanup test",
            requested_schema={},
        )

    assert len(manager._pending) == 0, "_pending must be empty after timeout"


# ---------------------------------------------------------------------------
# Integration: full HTTP roundtrip for elicitation/create
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_elicitation_full_roundtrip_via_http() -> None:
    """AC-63/AC-64: Full HTTP roundtrip — POST elicitation/create; client response resolves it."""
    store = InMemorySessionStore()
    registry = MCPToolRegistry()

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        """Accept requests with X-API-Key: test-key."""
        return api_key == "test-key"

    app = _make_stateful_app(registry, store)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _initialize_session(client)

        async def send_client_response() -> None:
            """Wait for elicitation/create to be enqueued, then POST the client response."""
            await asyncio.sleep(0.02)
            msgs = await store.dequeue_messages(session_id)
            elicit_req = next(
                (m for m in msgs if m.get("method") == "elicitation/create"),
                None,
            )
            if elicit_req is None:
                return
            await client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "result": {"action": "accept", "content": {"ok": True}},
                    "id": elicit_req["id"],
                },
                headers={**_HEADERS, "Mcp-Session-Id": session_id},
            )

        response_task = asyncio.create_task(send_client_response())
        elicit_resp = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "elicitation/create",
                "params": {
                    "message": "Please confirm",
                    "requestedSchema": {
                        "type": "object",
                        "properties": {"ok": {"type": "boolean"}},
                    },
                },
            },
            headers={**_HEADERS, "Mcp-Session-Id": session_id},
        )
        await response_task

    assert elicit_resp.status_code == 200
    body = elicit_resp.json()
    assert "result" in body, f"Expected result, got: {body}"
    assert body["result"]["action"] == "accept"
    assert body["result"]["content"] == {"ok": True}
