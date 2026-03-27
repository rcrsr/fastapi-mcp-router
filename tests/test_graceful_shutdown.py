"""Graceful shutdown tests for SSE streams.

Tests verify that setting shutdown_event causes SSE generators to exit
and yield a server-shutdown comment.
"""

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from datetime import UTC, datetime
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI, Request

from fastapi_mcp_router import MCPRouter, MCPToolRegistry, create_mcp_router
from fastapi_mcp_router.types import EventSubscriber, McpSessionData
from tests.conftest import SseCapture

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_app(
    shutdown_event: asyncio.Event,
    event_subscriber: EventSubscriber | None = None,
) -> tuple[FastAPI, dict[str, McpSessionData]]:
    """Build a FastAPI app wired to a caller-owned shutdown_event.

    Args:
        shutdown_event: Event passed to create_mcp_router; caller sets it to
            trigger graceful shutdown.
        event_subscriber: Optional EventSubscriber passed through to the router.

    Returns:
        Tuple of (FastAPI app, session_store dict).
    """
    session_store: dict[str, McpSessionData] = {}

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        """Accept all Bearer-token requests.

        Args:
            api_key: API key header value, or None.
            bearer_token: Bearer token value, or None.

        Returns:
            True when a Bearer token is present.
        """
        return bearer_token is not None

    async def session_getter(session_id: str) -> McpSessionData | None:
        """Return session data or None when absent from store.

        Args:
            session_id: Session identifier to look up.

        Returns:
            McpSessionData if found, None otherwise.
        """
        return session_store.get(session_id)

    async def session_creator(oauth_client_id: UUID | None, connection_id: UUID | None) -> str:
        """Generate a UUID v4 session ID and store McpSessionData.

        Args:
            oauth_client_id: UUID of the OAuth client, or None.
            connection_id: UUID of the connection, or None.

        Returns:
            The new session ID string.
        """
        session_id = str(uuid4())
        session_store[session_id] = McpSessionData(
            session_id=session_id,
            oauth_client_id=oauth_client_id,
            connection_id=connection_id,
            last_event_id=0,
            created_at=datetime.now(UTC),
        )
        return session_id

    fastapi_app = FastAPI()

    @fastapi_app.middleware("http")
    async def set_bearer_state(request: Request, call_next: Callable[..., Awaitable[object]]) -> object:
        """Attach oauth_client_id and connection_id to request.state for Bearer tokens.

        Args:
            request: Incoming HTTP request.
            call_next: Next ASGI handler.

        Returns:
            HTTP response from the next handler.
        """
        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            request.state.oauth_client_id = uuid4()
            request.state.connection_id = uuid4()
        return await call_next(request)

    mcp_router = create_mcp_router(
        MCPToolRegistry(),
        auth_validator=auth_validator,
        session_getter=session_getter,
        session_creator=session_creator,
        event_subscriber=event_subscriber,
        legacy_sse=True,
        shutdown_event=shutdown_event,
    )
    fastapi_app.include_router(mcp_router, prefix="/mcp")

    return fastapi_app, session_store


async def _create_session(client: httpx.AsyncClient) -> str:
    """Send an initialize POST and return the Mcp-Session-Id header value.

    Args:
        client: Configured AsyncClient for the stateful app.

    Returns:
        Session ID string from the Mcp-Session-Id response header.
    """
    response = await client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
            "id": 1,
        },
        headers={
            "MCP-Protocol-Version": "2025-06-18",
            "Authorization": "Bearer test-token",
        },
    )
    assert response.status_code == 200, f"initialize failed: {response.text}"
    return response.headers["Mcp-Session-Id"]


# ---------------------------------------------------------------------------
# Test 1: shutdown_event stops event_stream (with event_subscriber)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_shutdown_event_stops_event_stream() -> None:
    """Setting shutdown_event terminates the SSE stream and yields server-shutdown.

    Creates a router with a blocking event_subscriber (never yields) and a
    caller-owned shutdown_event. After the stream starts, sets the event and
    verifies the response completes with a server-shutdown comment.
    """
    shutdown_event = asyncio.Event()

    async def blocking_subscriber(
        session_id: str,
        last_event_id: int | None,
    ) -> AsyncGenerator[tuple[int, dict], None]:
        """Block forever without yielding events.

        Args:
            session_id: MCP session ID.
            last_event_id: Last-Event-ID value from header, or None.
        """
        await asyncio.Event().wait()
        yield  # pragma: no cover  — makes this an async generator

    fastapi_app, _ = _build_app(shutdown_event, event_subscriber=blocking_subscriber)
    transport = httpx.ASGITransport(app=fastapi_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _create_session(client)

    capture = SseCapture(fastapi_app)
    sse_transport = httpx.ASGITransport(app=capture)

    async with httpx.AsyncClient(transport=sse_transport, base_url="http://test") as sse_client:
        sse_task = asyncio.create_task(
            sse_client.get(
                "/mcp",
                headers={
                    "Authorization": "Bearer test-token",
                    "Mcp-Session-Id": session_id,
                },
            )
        )

        # Wait for headers before signalling shutdown.
        await asyncio.wait_for(capture.headers_received.wait(), timeout=5.0)

        # Trigger graceful shutdown.
        shutdown_event.set()

        # Stream must terminate without an external cancel.
        await asyncio.wait_for(sse_task, timeout=5.0)

    assert capture.status_code == 200
    full_content = "".join(capture.chunks)
    assert ": server-shutdown" in full_content, f"Expected server-shutdown comment in SSE stream, got: {full_content!r}"


# ---------------------------------------------------------------------------
# Test 2: shutdown_event stops keepalive-only stream (event_subscriber=None)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_shutdown_event_stops_keepalive_only_stream() -> None:
    """Setting shutdown_event terminates a keepalive-only SSE stream.

    Creates a router with no event_subscriber so the generator runs the
    keepalive-only branch. After the stream starts, sets the shutdown_event
    and verifies the response completes with a server-shutdown comment.
    """
    shutdown_event = asyncio.Event()

    fastapi_app, _ = _build_app(shutdown_event, event_subscriber=None)
    transport = httpx.ASGITransport(app=fastapi_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _create_session(client)

    capture = SseCapture(fastapi_app)
    sse_transport = httpx.ASGITransport(app=capture)

    async with httpx.AsyncClient(transport=sse_transport, base_url="http://test") as sse_client:
        sse_task = asyncio.create_task(
            sse_client.get(
                "/mcp",
                headers={
                    "Authorization": "Bearer test-token",
                    "Mcp-Session-Id": session_id,
                },
            )
        )

        # Wait for headers before signalling shutdown.
        await asyncio.wait_for(capture.headers_received.wait(), timeout=5.0)

        # Trigger graceful shutdown; the asyncio.wait race exits promptly.
        shutdown_event.set()

        # Stream must terminate without an external cancel.
        await asyncio.wait_for(sse_task, timeout=5.0)

    assert capture.status_code == 200
    full_content = "".join(capture.chunks)
    assert ": server-shutdown" in full_content, (
        f"Expected server-shutdown comment in keepalive-only SSE stream, got: {full_content!r}"
    )


# ---------------------------------------------------------------------------
# Test 3: MCPRouter.shutdown() sets _shutdown_event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.unit
async def test_mcp_router_shutdown_method() -> None:
    """MCPRouter.shutdown() sets _shutdown_event to signal active streams.

    Verifies that _shutdown_event exists and is not set at construction time,
    and that calling shutdown() sets it.
    """
    mcp = MCPRouter()

    assert hasattr(mcp, "_shutdown_event"), "MCPRouter must have a _shutdown_event attribute"
    assert not mcp._shutdown_event.is_set(), "_shutdown_event must not be set at construction"

    await mcp.shutdown()

    assert mcp._shutdown_event.is_set(), "_shutdown_event must be set after shutdown()"


# ---------------------------------------------------------------------------
# Test 4: create_mcp_router() backward compat — shutdown_event defaults to None
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_shutdown_event_default_none() -> None:
    """create_mcp_router() works without shutdown_event (backward compatibility).

    Constructs a router without providing shutdown_event and verifies that
    routes are registered, confirming the parameter defaults to None.
    """
    router = create_mcp_router(MCPToolRegistry(), legacy_sse=True)

    assert len(router.routes) > 0, "Router must register at least one route"
