"""Tests for SessionStore ABC, InMemorySessionStore, DELETE endpoint, and router validation.

Covers AC-16, AC-17, AC-18, AC-19, AC-20, AC-21, AC-88, AC-97, AC-98, EC-11, EC-13.
"""

import asyncio
import inspect
from datetime import UTC, datetime
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI

from fastapi_mcp_router import InMemorySessionStore, MCPToolRegistry, SessionStore, create_mcp_router
from fastapi_mcp_router.types import McpSessionData

# ---------------------------------------------------------------------------
# AC-16: SessionStore ABC has 6 abstract methods
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_session_store_abc_has_six_abstract_methods() -> None:
    """AC-16: SessionStore defines exactly 6 abstract methods."""
    abstract_methods = {
        name for name, method in inspect.getmembers(SessionStore) if getattr(method, "__isabstractmethod__", False)
    }
    assert len(abstract_methods) == 6


# ---------------------------------------------------------------------------
# AC-17: InMemorySessionStore with configurable TTL
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_in_memory_session_store_default_ttl() -> None:
    """AC-17: InMemorySessionStore defaults to ttl_seconds=3600."""
    store = InMemorySessionStore()
    assert store.ttl_seconds == 3600


@pytest.mark.unit
def test_in_memory_session_store_custom_ttl() -> None:
    """AC-17: InMemorySessionStore accepts a custom ttl_seconds value."""
    store = InMemorySessionStore(ttl_seconds=60)
    assert store.ttl_seconds == 60


# ---------------------------------------------------------------------------
# AC-18: DELETE returns 204 on valid session
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_valid_session_returns_204() -> None:
    """AC-18: DELETE /mcp with a valid Mcp-Session-Id returns 204 No Content."""
    registry = MCPToolRegistry()
    store = InMemorySessionStore()

    @registry.tool()
    async def dummy() -> dict:
        """Dummy tool for test setup."""
        return {}

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        """Accept requests with X-API-Key: test-key."""
        return api_key == "test-key"

    router = create_mcp_router(registry, session_store=store, auth_validator=auth_validator)
    app = FastAPI()
    app.include_router(router, prefix="/mcp")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        init_resp = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "clientInfo": {},
                    "capabilities": {},
                },
            },
            headers={"X-API-Key": "test-key"},
        )
        session_id = init_resp.headers.get("Mcp-Session-Id")
        assert session_id is not None, "initialize must return Mcp-Session-Id header"

        del_resp = await client.delete("/mcp", headers={"Mcp-Session-Id": session_id})
        assert del_resp.status_code == 204


# ---------------------------------------------------------------------------
# AC-19: DELETE returns 404 on unknown session
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_unknown_session_returns_404() -> None:
    """AC-19: DELETE /mcp with a non-existent session ID returns 404."""
    registry = MCPToolRegistry()
    store = InMemorySessionStore()

    router = create_mcp_router(registry, session_store=store)
    app = FastAPI()
    app.include_router(router, prefix="/mcp")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.delete("/mcp", headers={"Mcp-Session-Id": "nonexistent-session-id"})
        assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_without_session_store_returns_404() -> None:
    """AC-19: DELETE /mcp without session_store configured returns 404."""
    registry = MCPToolRegistry()

    router = create_mcp_router(registry)
    app = FastAPI()
    app.include_router(router, prefix="/mcp")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.delete("/mcp", headers={"Mcp-Session-Id": "any-id"})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# AC-20: Callback-based session API still works
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_callback_based_session_api_still_works() -> None:
    """AC-20: session_getter/session_creator callbacks work independently of session_store."""
    registry = MCPToolRegistry()
    session_store_dict: dict[str, McpSessionData] = {}

    @registry.tool()
    async def echo(value: str) -> str:
        """Echo the input value."""
        return value

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        """Accept Bearer tokens only."""
        return bearer_token is not None

    async def session_getter(session_id: str) -> McpSessionData | None:
        """Return session from dict or None.

        Args:
            session_id: Session identifier to look up.

        Returns:
            McpSessionData if found, None otherwise.
        """
        return session_store_dict.get(session_id)

    async def session_creator(oauth_client_id: UUID | None, connection_id: UUID | None) -> str:
        """Create a session and store it.

        Args:
            oauth_client_id: OAuth client UUID or None.
            connection_id: Connection UUID or None.

        Returns:
            New session ID string.
        """
        sid = str(uuid4())
        session_store_dict[sid] = McpSessionData(
            session_id=sid,
            oauth_client_id=oauth_client_id,
            connection_id=connection_id,
            last_event_id=0,
            created_at=datetime.now(UTC),
        )
        return sid

    from fastapi import Request

    router = create_mcp_router(
        registry,
        auth_validator=auth_validator,
        session_getter=session_getter,
        session_creator=session_creator,
    )
    app = FastAPI()

    @app.middleware("http")
    async def set_bearer_state(request: Request, call_next):
        """Set oauth_client_id and connection_id for Bearer token requests."""
        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            request.state.oauth_client_id = uuid4()
            request.state.connection_id = uuid4()
        return await call_next(request)

    app.include_router(router, prefix="/mcp")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        init_resp = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "clientInfo": {},
                    "capabilities": {},
                },
            },
            headers={"Authorization": "Bearer test-token"},
        )
        assert init_resp.status_code == 200
        session_id = init_resp.headers.get("Mcp-Session-Id")
        assert session_id is not None

        call_resp = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "echo", "arguments": {"value": "hello"}},
            },
            headers={
                "Authorization": "Bearer test-token",
                "Mcp-Session-Id": session_id,
            },
        )
        assert call_resp.status_code == 200
        body = call_resp.json()
        assert body["result"]["content"][0]["text"] == "hello"


# ---------------------------------------------------------------------------
# AC-21 / AC-89: Message queue bounded at 1000
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_message_queue_bounded_at_1000() -> None:
    """AC-21/AC-89: Enqueuing 1001 messages drops the 1001st; queue holds exactly 1000."""
    store = InMemorySessionStore()
    session = await store.create("2025-06-18", {}, {})

    for i in range(1001):
        await store.enqueue_message(session.session_id, {"i": i})

    msgs = await store.dequeue_messages(session.session_id)
    assert len(msgs) == 1000


# ---------------------------------------------------------------------------
# AC-88: dequeue_messages returns [] for empty queue
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dequeue_empty_queue_returns_empty_list() -> None:
    """AC-88: dequeue_messages returns an empty list when no messages are queued."""
    store = InMemorySessionStore()
    session = await store.create("2025-06-18", {}, {})

    msgs = await store.dequeue_messages(session.session_id)
    assert msgs == []


# ---------------------------------------------------------------------------
# AC-97: Expired session → get() returns None
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_expired_session_returns_none() -> None:
    """AC-97: After TTL elapses, get() returns None for the expired session."""
    store = InMemorySessionStore(ttl_seconds=1)
    session = await store.create("2025-06-18", {}, {})

    await asyncio.sleep(1.1)

    result = await store.get(session.session_id)
    assert result is None


# ---------------------------------------------------------------------------
# AC-98: Concurrent enqueue_message - no race condition
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_concurrent_enqueue_no_race_condition() -> None:
    """AC-98: 50 concurrent enqueue_message calls all succeed without data loss."""
    store = InMemorySessionStore()
    session = await store.create("2025-06-18", {}, {})

    tasks = [store.enqueue_message(session.session_id, {"i": i}) for i in range(50)]
    await asyncio.gather(*tasks)

    msgs = await store.dequeue_messages(session.session_id)
    assert len(msgs) == 50


# ---------------------------------------------------------------------------
# EC-11: session_store + session_getter → ValueError
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_session_store_and_getter_raises_value_error() -> None:
    """EC-11: create_mcp_router raises ValueError when both session_store and session_getter are provided."""
    registry = MCPToolRegistry()

    async def session_getter(session_id: str) -> None:
        """Stub session getter."""
        return None

    with pytest.raises(ValueError):
        create_mcp_router(
            registry,
            session_store=InMemorySessionStore(),
            session_getter=session_getter,
        )


# ---------------------------------------------------------------------------
# EC-13: stateful=True without session_store → ValueError
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stateful_without_session_store_raises_value_error() -> None:
    """EC-13: create_mcp_router raises ValueError when stateful=True but session_store is None."""
    registry = MCPToolRegistry()

    with pytest.raises(ValueError):
        create_mcp_router(registry, stateful=True)
