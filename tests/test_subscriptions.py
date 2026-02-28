"""Tests for resources/subscribe and resources/unsubscribe MCP methods.

Covers AC-46, AC-47, AC-48, AC-49, AC-78, AC-79, AC-90, EC-15, EC-16, EC-17.

AC-46: resources/subscribe tracks URI in session subscriptions set
AC-47: resources/unsubscribe removes URI from session subscriptions set
AC-48: Subscription requires stateful mode → MCPError -32601 if not stateful
AC-49: Max 100 subscriptions per session enforced
AC-78: resources/subscribe without stateful mode returns MCPError -32601
AC-79: Subscriptions exceeding 100 per session return MCPError -32602
AC-90: Session at 100 subscriptions: 101st returns MCPError -32602
EC-15: Subscribe without stateful mode → MCPError -32601 (method not found)
EC-16: Missing uri in subscribe params → MCPError -32602 (invalid params)
EC-17: Subscriptions exceed 100 per session → MCPError -32602
"""

import httpx
import pytest
from fastapi import FastAPI

from fastapi_mcp_router import InMemorySessionStore, MCPToolRegistry, create_mcp_router

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
# AC-46: resources/subscribe tracks URI in session subscriptions set
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_subscribe_tracks_uri_in_session() -> None:
    """AC-46: resources/subscribe stores the URI in the session subscriptions set."""
    store = InMemorySessionStore()
    app = _make_stateful_app(store)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _initialize_session(client)

        resp = await client.post(
            "/mcp",
            json=_rpc("resources/subscribe", {"uri": "file:///doc.txt"}),
            headers={**_HEADERS, "Mcp-Session-Id": session_id},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body, f"Expected result, got: {body}"
    assert body["result"] == {}

    session = await store.get(session_id)
    assert session is not None
    assert "file:///doc.txt" in session.subscriptions


# ---------------------------------------------------------------------------
# AC-47: resources/unsubscribe removes URI from session subscriptions set
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_unsubscribe_removes_uri_from_session() -> None:
    """AC-47: resources/unsubscribe removes the URI from the session subscriptions set."""
    store = InMemorySessionStore()
    app = _make_stateful_app(store)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _initialize_session(client)
        session_headers = {**_HEADERS, "Mcp-Session-Id": session_id}

        sub_resp = await client.post(
            "/mcp",
            json=_rpc("resources/subscribe", {"uri": "file:///doc.txt"}),
            headers=session_headers,
        )
        assert sub_resp.status_code == 200
        assert "result" in sub_resp.json()

        unsub_resp = await client.post(
            "/mcp",
            json=_rpc("resources/unsubscribe", {"uri": "file:///doc.txt"}),
            headers=session_headers,
        )

    assert unsub_resp.status_code == 200
    body = unsub_resp.json()
    assert "result" in body, f"Expected result, got: {body}"
    assert body["result"] == {}

    session = await store.get(session_id)
    assert session is not None
    assert "file:///doc.txt" not in session.subscriptions


# ---------------------------------------------------------------------------
# AC-46 + idempotency: Duplicate subscription is idempotent
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_duplicate_subscribe_is_idempotent() -> None:
    """AC-46 idempotency: Subscribing the same URI twice produces no error."""
    store = InMemorySessionStore()
    app = _make_stateful_app(store)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _initialize_session(client)
        session_headers = {**_HEADERS, "Mcp-Session-Id": session_id}

        first_resp = await client.post(
            "/mcp",
            json=_rpc("resources/subscribe", {"uri": "file:///doc.txt"}, rpc_id=2),
            headers=session_headers,
        )
        assert first_resp.status_code == 200
        assert "result" in first_resp.json()

        second_resp = await client.post(
            "/mcp",
            json=_rpc("resources/subscribe", {"uri": "file:///doc.txt"}, rpc_id=3),
            headers=session_headers,
        )

    assert second_resp.status_code == 200
    body = second_resp.json()
    assert "result" in body, f"Expected result on second subscribe, got: {body}"
    assert body["result"] == {}

    session = await store.get(session_id)
    assert session is not None
    uri_count = sum(1 for uri in session.subscriptions if uri == "file:///doc.txt")
    assert uri_count == 1, f"URI must appear exactly once in subscriptions set, found {uri_count}"


# ---------------------------------------------------------------------------
# AC-47 + idempotency: Unsubscribe non-subscribed URI is a no-op
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_unsubscribe_nonsubscribed_uri_is_noop() -> None:
    """AC-47 idempotency: Unsubscribing a URI not subscribed returns success with no error."""
    store = InMemorySessionStore()
    app = _make_stateful_app(store)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _initialize_session(client)

        resp = await client.post(
            "/mcp",
            json=_rpc("resources/unsubscribe", {"uri": "file:///doc.txt"}),
            headers={**_HEADERS, "Mcp-Session-Id": session_id},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body, f"Expected result on no-op unsubscribe, got: {body}"
    assert body["result"] == {}


# ---------------------------------------------------------------------------
# AC-48/AC-78/EC-15: Subscribe without stateful mode → MCPError -32601
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_subscribe_without_stateful_returns_32601() -> None:
    """AC-48/AC-78/EC-15: resources/subscribe on a stateless router returns MCPError -32601."""
    app = _make_stateless_app()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=_rpc("resources/subscribe", {"uri": "file:///doc.txt"}),
            headers=_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "error" in body, f"Expected error, got: {body}"
    assert body["error"]["code"] == -32601


# ---------------------------------------------------------------------------
# AC-48/AC-78/EC-15: Unsubscribe without stateful mode → MCPError -32601
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_unsubscribe_without_stateful_returns_32601() -> None:
    """AC-48/AC-78/EC-15: resources/unsubscribe on a stateless router returns MCPError -32601."""
    app = _make_stateless_app()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=_rpc("resources/unsubscribe", {"uri": "file:///doc.txt"}),
            headers=_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "error" in body, f"Expected error, got: {body}"
    assert body["error"]["code"] == -32601


# ---------------------------------------------------------------------------
# EC-16: Missing uri in subscribe params → MCPError -32602
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_subscribe_missing_uri_returns_32602() -> None:
    """EC-16: resources/subscribe with empty params (missing uri) returns MCPError -32602."""
    store = InMemorySessionStore()
    app = _make_stateful_app(store)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _initialize_session(client)

        resp = await client.post(
            "/mcp",
            json=_rpc("resources/subscribe", {}),
            headers={**_HEADERS, "Mcp-Session-Id": session_id},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "error" in body, f"Expected error, got: {body}"
    assert body["error"]["code"] == -32602


# ---------------------------------------------------------------------------
# EC-16: Missing uri in unsubscribe params → MCPError -32602
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_unsubscribe_missing_uri_returns_32602() -> None:
    """EC-16: resources/unsubscribe with empty params (missing uri) returns MCPError -32602."""
    store = InMemorySessionStore()
    app = _make_stateful_app(store)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _initialize_session(client)

        resp = await client.post(
            "/mcp",
            json=_rpc("resources/unsubscribe", {}),
            headers={**_HEADERS, "Mcp-Session-Id": session_id},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "error" in body, f"Expected error, got: {body}"
    assert body["error"]["code"] == -32602


# ---------------------------------------------------------------------------
# AC-49/AC-79/AC-90/EC-17: Max 100 subscriptions per session enforced
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_subscription_limit_100_enforced() -> None:
    """AC-49/AC-79/AC-90/EC-17: The 101st subscription returns MCPError -32602."""
    store = InMemorySessionStore()
    app = _make_stateful_app(store)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _initialize_session(client)
        session_headers = {**_HEADERS, "Mcp-Session-Id": session_id}

        for i in range(100):
            resp = await client.post(
                "/mcp",
                json=_rpc("resources/subscribe", {"uri": f"file:///doc{i}.txt"}, rpc_id=i + 2),
                headers=session_headers,
            )
            assert resp.status_code == 200, f"subscribe {i} failed: {resp.text}"
            body = resp.json()
            assert "result" in body, f"subscribe {i} returned error: {body}"

        over_limit_resp = await client.post(
            "/mcp",
            json=_rpc("resources/subscribe", {"uri": "file:///doc100.txt"}, rpc_id=102),
            headers=session_headers,
        )

    assert over_limit_resp.status_code == 200
    body = over_limit_resp.json()
    assert "error" in body, f"Expected error on 101st subscription, got: {body}"
    assert body["error"]["code"] == -32602

    session = await store.get(session_id)
    assert session is not None
    assert len(session.subscriptions) == 100, f"Expected 100 subscriptions, got {len(session.subscriptions)}"
