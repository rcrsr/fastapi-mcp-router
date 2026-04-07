"""Integration tests for SSE streaming endpoint in stateful MCP router.

Tests the GET /mcp endpoint that handles SSE streaming for OAuth (Bearer token)
connections. Covers session creation, session resumption, error cases (expired
session, API key in stateful mode), Last-Event-ID handling, client disconnect,
keepalive comment delivery, and session_getter failure after session_creator.

Each test that uses session_client receives a stateful AsyncClient and the
underlying FastAPI app. The local fixture shadows the conftest.py fixture
because pytest-asyncio strict mode rejects async fixtures decorated with
plain @pytest.fixture.

SSE streaming uses an ASGI middleware (SseCapture) that intercepts
http.response.start messages and signals an asyncio.Event. Tests await
that event to confirm status code and headers, then cancel the background
task that drives the infinite streaming generator.
"""

import asyncio
import contextlib
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import UTC, datetime
from unittest.mock import patch
from uuid import UUID, uuid4

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI, Request

from fastapi_mcp_router import MCPToolRegistry, create_mcp_router
from fastapi_mcp_router.exceptions import MCPError
from fastapi_mcp_router.session import InMemorySessionStore
from fastapi_mcp_router.types import McpSessionData
from tests.conftest import SseCapture

# ---------------------------------------------------------------------------
# Local session_client fixture (shadows conftest.py for this file)
# ---------------------------------------------------------------------------


@dataclass
class StatefulSession:
    """Bundle of stateful app and AsyncClient for SSE tests.

    Attributes:
        app: FastAPI app with session callbacks configured.
        client: httpx.AsyncClient backed by ASGITransport for the app.
    """

    app: FastAPI
    client: httpx.AsyncClient


@pytest_asyncio.fixture(name="session_client")
async def session_client_fixture() -> AsyncGenerator[StatefulSession]:
    """Create a StatefulSession with Bearer auth and in-memory session store.

    Mirrors the session_client fixture in conftest.py but uses
    @pytest_asyncio.fixture so pytest-asyncio strict mode accepts it.
    Returns StatefulSession instead of AsyncClient so tests can wrap
    the app in SseCapture for streaming assertions.

    Yields:
        StatefulSession with app and AsyncClient configured.
    """
    session_store: dict[str, McpSessionData] = {}

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        """Accept all Bearer token requests."""
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
    async def set_bearer_state(request: Request, call_next):
        """Set oauth_client_id and connection_id on request.state for Bearer tokens."""
        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            request.state.oauth_client_id = uuid4()
            request.state.connection_id = uuid4()
        response = await call_next(request)
        return response

    mcp_router = create_mcp_router(
        MCPToolRegistry(),
        auth_validator=auth_validator,
        session_getter=session_getter,
        session_creator=session_creator,
        legacy_sse=True,
    )
    fastapi_app.include_router(mcp_router, prefix="/mcp")

    transport = httpx.ASGITransport(app=fastapi_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield StatefulSession(app=fastapi_app, client=async_client)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _create_session(client: httpx.AsyncClient) -> str:
    """Send an initialize POST request and return the Mcp-Session-Id header value.

    Args:
        client: Configured AsyncClient for the stateful app.

    Returns:
        The session ID string from the Mcp-Session-Id response header.
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


async def _run_sse_capture(
    app: FastAPI,
    headers: dict[str, str],
    *,
    settle_seconds: float = 0.05,
) -> SseCapture:
    """Run a GET /mcp request against app wrapped in SseCapture, cancel after headers arrive.

    Creates a fresh SseCapture wrapping app, runs the GET in a background task,
    awaits the headers_received event, waits settle_seconds for body chunks,
    then cancels the task.

    Args:
        app: FastAPI app to wrap in SseCapture.
        headers: HTTP headers to send with the GET request.
        settle_seconds: Extra seconds to wait after headers arrive (allows chunks to land).

    Returns:
        SseCapture instance populated with status_code, headers, and chunks.
    """
    capture = SseCapture(app)
    transport = httpx.ASGITransport(app=capture)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        task = asyncio.create_task(client.get("/mcp", headers=headers))

        # capture.status_code remains None on timeout; test assertion will fail clearly
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(capture.headers_received.wait()), timeout=5.0)

        if settle_seconds > 0:
            await asyncio.sleep(settle_seconds)

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    return capture


# ---------------------------------------------------------------------------
# IR-6: session resume returns HTTP 200
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sse_session_resume_returns_200(session_client: StatefulSession) -> None:
    """GET with valid Mcp-Session-Id header on an active session returns HTTP 200.

    Args:
        session_client: Stateful fixture bundle with app and AsyncClient.
    """
    session_id = await _create_session(session_client.client)

    capture = await _run_sse_capture(
        session_client.app,
        headers={
            "Authorization": "Bearer test-token",
            "Mcp-Session-Id": session_id,
        },
    )

    assert capture.status_code == 200


# ---------------------------------------------------------------------------
# IR-7 (EC-1): expired session returns HTTP 410
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sse_expired_session_returns_410(session_client: StatefulSession) -> None:
    """GET with nonexistent Mcp-Session-Id returns HTTP 410 with session_expired error.

    Args:
        session_client: Stateful fixture bundle with app and AsyncClient.
    """
    response = await session_client.client.get(
        "/mcp",
        headers={
            "Authorization": "Bearer test-token",
            "Mcp-Session-Id": "nonexistent-session-id-xyz",
        },
    )
    assert response.status_code == 410
    data = response.json()
    assert data["error"] == "session_expired"


# ---------------------------------------------------------------------------
# IR-8 (EC-2): API key in stateful mode returns HTTP 405
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sse_api_key_returns_405_in_stateful_mode() -> None:
    """GET with X-API-Key (no Bearer token) in stateful mode returns HTTP 405.

    The router returns 405 when session callbacks are configured but the
    connection is not an OAuth (Bearer token) connection.
    """
    session_store: dict[str, McpSessionData] = {}

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        """Accept both API key and Bearer token connections."""
        return api_key == "test-key" or bearer_token is not None

    async def session_getter(session_id: str) -> McpSessionData | None:
        """Return session from in-memory store."""
        return session_store.get(session_id)

    async def session_creator(oauth_client_id: UUID | None, connection_id: UUID | None) -> str:
        """Create and store a new session."""
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
    mcp_router = create_mcp_router(
        MCPToolRegistry(),
        auth_validator=auth_validator,
        session_getter=session_getter,
        session_creator=session_creator,
        legacy_sse=True,
    )
    fastapi_app.include_router(mcp_router, prefix="/mcp")

    transport = httpx.ASGITransport(app=fastapi_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/mcp",
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 405
    data = response.json()
    assert data["error"] == "method_not_allowed"


# ---------------------------------------------------------------------------
# IR-9: Last-Event-ID header parsed, stream establishes (HTTP 200)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sse_last_event_id_header_parsed(session_client: StatefulSession) -> None:
    """GET with Last-Event-ID: 42 establishes the SSE stream with HTTP 200.

    Args:
        session_client: Stateful fixture bundle with app and AsyncClient.
    """
    session_id = await _create_session(session_client.client)

    capture = await _run_sse_capture(
        session_client.app,
        headers={
            "Authorization": "Bearer test-token",
            "Mcp-Session-Id": session_id,
            "Last-Event-ID": "42",
        },
    )

    assert capture.status_code == 200


# ---------------------------------------------------------------------------
# IR-10 (AC-10): invalid Last-Event-ID falls back to session default, no 400
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sse_invalid_last_event_id_uses_session_default(session_client: StatefulSession) -> None:
    """GET with Last-Event-ID: not-a-number does not return 400; falls back gracefully.

    The router logs a warning and falls back to session.last_event_id instead
    of raising an error when the header value cannot be parsed as an integer.

    Args:
        session_client: Stateful fixture bundle with app and AsyncClient.
    """
    session_id = await _create_session(session_client.client)

    capture = await _run_sse_capture(
        session_client.app,
        headers={
            "Authorization": "Bearer test-token",
            "Mcp-Session-Id": session_id,
            "Last-Event-ID": "not-a-number",
        },
    )

    assert capture.status_code != 400
    assert capture.status_code == 200


# ---------------------------------------------------------------------------
# IR-11 (AC-8): client disconnect terminates SSE generator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sse_client_disconnect_terminates_generator(session_client: StatefulSession) -> None:
    """Cancelling an SSE streaming task fires CancelledError and leaves no leaked coroutines.

    The SSE event_stream generator in router.py catches CancelledError and
    re-raises it. This test cancels the consuming task after headers arrive,
    verifies cancellation completes, and confirms no asyncio.sleep coroutine
    from the SSE generator remains pending afterward.

    Args:
        session_client: Stateful fixture bundle with app and AsyncClient.
    """
    session_id = await _create_session(session_client.client)

    # settle_seconds=0.1 gives the generator time to enter asyncio.sleep(30)
    capture = await _run_sse_capture(
        session_client.app,
        headers={
            "Authorization": "Bearer test-token",
            "Mcp-Session-Id": session_id,
        },
        settle_seconds=0.1,
    )

    assert capture.status_code == 200

    # Verify no lingering asyncio.sleep coroutines from the SSE generator
    pending = [t for t in asyncio.all_tasks() if not t.done()]
    for t in pending:
        coro = t.get_coro()
        coro_name = getattr(coro, "__qualname__", getattr(coro, "__name__", ""))
        assert "sleep" not in coro_name, f"Leaked pending sleep coroutine: {coro_name}"


# ---------------------------------------------------------------------------
# IR-12: new session created from Bearer token
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sse_new_session_created_from_bearer_token(session_client: StatefulSession) -> None:
    """GET with Bearer token and valid connection_id creates a new session.

    The fixture middleware sets oauth_client_id and connection_id on request.state
    for Bearer token requests, enabling session_creator to associate the session.

    Args:
        session_client: Stateful fixture bundle with app and AsyncClient.
    """
    capture = await _run_sse_capture(
        session_client.app,
        headers={"Authorization": "Bearer test-token"},
    )

    assert capture.status_code == 200
    assert "mcp-session-id" in capture.headers


# ---------------------------------------------------------------------------
# IR-13: SSE response includes Mcp-Session-Id header
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sse_response_includes_mcp_session_id_header(session_client: StatefulSession) -> None:
    """SSE streaming response contains Mcp-Session-Id header matching the requested session.

    Args:
        session_client: Stateful fixture bundle with app and AsyncClient.
    """
    session_id = await _create_session(session_client.client)

    capture = await _run_sse_capture(
        session_client.app,
        headers={
            "Authorization": "Bearer test-token",
            "Mcp-Session-Id": session_id,
        },
    )

    assert capture.status_code == 200
    assert "mcp-session-id" in capture.headers
    assert capture.headers["mcp-session-id"] == session_id


# ---------------------------------------------------------------------------
# IR-14: keepalive comment sent in SSE stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sse_keepalive_comment_sent(session_client: StatefulSession) -> None:
    """SSE stream emits the initial establishment comment before the keepalive loop.

    The event_stream generator in router.py yields ': SSE stream established\\n\\n'
    immediately on connection. This test reads chunks via SseCapture and confirms
    the SSE comment prefix ': ' appears in the received content.

    Args:
        session_client: Stateful fixture bundle with app and AsyncClient.
    """
    session_id = await _create_session(session_client.client)

    # settle_seconds=0.1 allows the first body chunk to arrive after response.start
    capture = await _run_sse_capture(
        session_client.app,
        headers={
            "Authorization": "Bearer test-token",
            "Mcp-Session-Id": session_id,
        },
        settle_seconds=0.1,
    )

    assert capture.status_code == 200
    assert len(capture.chunks) > 0, "No chunks received from SSE stream"
    full_content = "".join(capture.chunks)
    assert ": " in full_content, f"Expected SSE comment in stream, got: {full_content!r}"


# ---------------------------------------------------------------------------
# EC-5 (SSE path): session_getter returns None after session_creator succeeds
# Covers router.py lines 374-375
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sse_session_getter_returns_none_after_create_returns_500() -> None:
    """GET with Bearer token returns HTTP 500 when session_getter returns None after session_creator.

    session_creator stores a session ID, but session_getter always returns None.
    The router calls session_getter to verify the newly created session (line 372)
    and hits the error branch at lines 374-375 when it returns None.
    """

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        """Accept all Bearer token requests."""
        return bearer_token is not None

    async def broken_session_getter(session_id: str) -> McpSessionData | None:
        """Always return None, simulating a retrieval failure after creation.

        Args:
            session_id: Session identifier (ignored).

        Returns:
            Always None.
        """
        return None

    async def session_creator(oauth_client_id: UUID | None, connection_id: UUID | None) -> str:
        """Return a session ID without storing anything, so getter always misses.

        Args:
            oauth_client_id: UUID of the OAuth client, or None.
            connection_id: UUID of the connection, or None.

        Returns:
            A UUID string that broken_session_getter will not find.
        """
        return str(uuid4())

    fastapi_app = FastAPI()

    @fastapi_app.middleware("http")
    async def set_bearer_state(request: Request, call_next):
        """Set oauth_client_id and connection_id on request.state for Bearer tokens."""
        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            request.state.oauth_client_id = uuid4()
            request.state.connection_id = uuid4()
        response = await call_next(request)
        return response

    mcp_router = create_mcp_router(
        MCPToolRegistry(),
        auth_validator=auth_validator,
        session_getter=broken_session_getter,
        session_creator=session_creator,
        legacy_sse=True,
    )
    fastapi_app.include_router(mcp_router, prefix="/mcp")

    transport = httpx.ASGITransport(app=fastapi_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/mcp",
            headers={"Authorization": "Bearer test-token"},
        )

    assert response.status_code == 500
    data = response.json()
    assert data["error"] == "internal_error"


# ---------------------------------------------------------------------------
# Line 420: keepalive yield after asyncio.sleep(30) fires
# Covers router.py line 420
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sse_keepalive_yield_fires_with_patched_sleep(session_client: StatefulSession) -> None:
    """SSE stream emits keepalive comment when the 30-second sleep is patched to return immediately.

    Patches asyncio.sleep selectively: skips the 30-second keepalive wait but
    preserves the short settle sleep in _run_sse_capture so the task is not
    cancelled before the generator yields the keepalive comment.

    Args:
        session_client: Stateful fixture bundle with app and AsyncClient.
    """
    session_id = await _create_session(session_client.client)

    real_sleep = asyncio.sleep

    async def fast_sleep(delay: float) -> None:
        if delay >= 10:
            return  # skip long keepalive waits
        await real_sleep(delay)

    with patch("asyncio.sleep", new=fast_sleep):
        capture = await _run_sse_capture(
            session_client.app,
            headers={
                "Authorization": "Bearer test-token",
                "Mcp-Session-Id": session_id,
            },
            settle_seconds=0.1,
        )

    assert capture.status_code == 200
    assert len(capture.chunks) > 0, "No chunks received from SSE stream"
    full_content = "".join(capture.chunks)
    assert "keepalive" in full_content, f"Expected keepalive in stream, got: {full_content!r}"


# ---------------------------------------------------------------------------
# Helpers for session_store-based resilience tests
# ---------------------------------------------------------------------------


def _build_store_app() -> tuple[FastAPI, InMemorySessionStore]:
    """Build a FastAPI app using InMemorySessionStore for the session_store code path.

    Returns:
        Tuple of (FastAPI app, InMemorySessionStore instance).
    """
    registry = MCPToolRegistry()
    store = InMemorySessionStore()

    @registry.tool()
    async def dummy() -> dict:
        """Dummy tool for test setup."""
        return {}

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        return bearer_token is not None

    router = create_mcp_router(
        registry,
        session_store=store,
        auth_validator=auth_validator,
        legacy_sse=True,
    )
    app = FastAPI()
    app.include_router(router, prefix="/mcp")
    return app, store


async def _init_store_session(client: httpx.AsyncClient) -> str:
    """Send initialize POST and return session ID for a session_store app.

    Args:
        client: AsyncClient for the store-based app.

    Returns:
        Session ID from Mcp-Session-Id header.
    """
    resp = await client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "clientInfo": {}, "capabilities": {}},
        },
        headers={"Authorization": "Bearer test-token"},
    )
    assert resp.status_code == 200, f"initialize failed: {resp.text}"
    return resp.headers["Mcp-Session-Id"]


# ---------------------------------------------------------------------------
# Resilience: transient MCPError from dequeue_messages does not kill SSE stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sse_stream_survives_transient_dequeue_failure() -> None:
    """SSE stream continues after a transient MCPError from dequeue_messages."""
    app, _store = _build_store_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _init_store_session(client)

    original_dequeue = InMemorySessionStore.dequeue_messages
    call_count = 0

    async def failing_then_ok(self: InMemorySessionStore, sid: str) -> list[dict]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise MCPError(code=-32603, message="transient")
        return await original_dequeue(self, sid)

    with patch.object(InMemorySessionStore, "dequeue_messages", failing_then_ok):
        capture = await _run_sse_capture(
            app,
            headers={
                "Authorization": "Bearer test-token",
                "Mcp-Session-Id": session_id,
            },
            settle_seconds=1.5,
        )

    assert capture.status_code == 200
    body = "".join(capture.chunks)
    assert "SSE stream established" in body
    assert "event: error" not in body


# ---------------------------------------------------------------------------
# Resilience: non-MCPError from dequeue_messages yields event: error payload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sse_stream_yields_error_event_on_fatal_failure() -> None:
    """SSE stream yields event: error with JSON-RPC payload on non-MCPError exception."""
    app, _store = _build_store_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _init_store_session(client)

    async def fatal_dequeue(self: InMemorySessionStore, sid: str) -> list[dict]:
        raise RuntimeError("unexpected failure")

    with patch.object(InMemorySessionStore, "dequeue_messages", fatal_dequeue):
        capture = await _run_sse_capture(
            app,
            headers={
                "Authorization": "Bearer test-token",
                "Mcp-Session-Id": session_id,
            },
            settle_seconds=1.5,
        )

    assert capture.status_code == 200
    body = "".join(capture.chunks)
    assert "event: error" in body
    for line in body.split("\n"):
        if line.startswith("data: ") and "-32603" in line:
            payload = json.loads(line[len("data: ") :])
            assert payload["jsonrpc"] == "2.0"
            assert payload["error"]["code"] == -32603
            break
    else:
        pytest.fail("No JSON-RPC error data line found in SSE stream")
