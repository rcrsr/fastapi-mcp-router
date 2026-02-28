"""SSE event delivery tests for the MCP router.

Tests the event_subscriber integration in the SSE GET /mcp endpoint.
Covers AC-1 through AC-6, AC-87, AC-95, AC-96, and EC-11.

Each test verifies one acceptance criterion. Integration tests use
SseCapture middleware (ASGI wrapping) to intercept streaming responses
without blocking on the infinite keepalive loop.
"""

import asyncio
import contextlib
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch
from uuid import UUID, uuid4

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI, Request

from fastapi_mcp_router import MCPToolRegistry, create_mcp_router
from fastapi_mcp_router.types import EventSubscriber, McpSessionData

# ---------------------------------------------------------------------------
# ASGI capture middleware (same pattern as test_sse_streaming.py)
# ---------------------------------------------------------------------------


class SseCapture:
    """ASGI middleware that captures SSE response headers and body chunks.

    Intercepts http.response.start to record the status code and headers,
    then signals headers_received so tests can inspect them before the
    streaming task is cancelled.

    Attributes:
        app: Inner ASGI application to wrap.
        status_code: HTTP status code from http.response.start, or None.
        headers: Response headers dict (lower-case keys), empty until set.
        chunks: Decoded body chunks received before cancellation.
        headers_received: Event that fires when http.response.start arrives.
    """

    def __init__(self, app: Any) -> None:
        self.app = app
        self.status_code: int | None = None
        self.headers: dict[str, str] = {}
        self.chunks: list[str] = []
        self.headers_received: asyncio.Event = asyncio.Event()

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """Wrap the inner app, capturing response start and body messages.

        Args:
            scope: ASGI connection scope.
            receive: ASGI receive callable.
            send: ASGI send callable.
        """

        async def capturing_send(message: Any) -> None:
            if message["type"] == "http.response.start":
                self.status_code = message["status"]
                self.headers = {k.decode(): v.decode() for k, v in message.get("headers", [])}
                self.headers_received.set()
            elif message["type"] == "http.response.body":
                body = message.get("body", b"")
                if body:
                    self.chunks.append(body.decode())
            await send(message)

        await self.app(scope, receive, capturing_send)


# ---------------------------------------------------------------------------
# Stateful fixture bundle
# ---------------------------------------------------------------------------


@dataclass
class StatefulApp:
    """Bundle of stateful FastAPI app and AsyncClient for SSE event tests.

    Attributes:
        app: FastAPI app configured with session callbacks.
        client: httpx.AsyncClient backed by ASGITransport.
        session_store: In-memory session store backing the app.
    """

    app: FastAPI
    client: httpx.AsyncClient
    session_store: dict[str, McpSessionData]


def _build_stateful_app(
    event_subscriber: EventSubscriber | None = None,
) -> tuple[FastAPI, dict[str, McpSessionData]]:
    """Construct a stateful FastAPI app with optional event_subscriber.

    Args:
        event_subscriber: Optional subscriber to pass to create_mcp_router.

    Returns:
        Tuple of (FastAPI app, session_store dict).
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
    async def set_bearer_state(request: Request, call_next: Any) -> Any:
        """Set oauth_client_id and connection_id on request.state for Bearer tokens."""
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
    )
    fastapi_app.include_router(mcp_router, prefix="/mcp")

    return fastapi_app, session_store


@pytest_asyncio.fixture(name="stateful_app")
async def stateful_app_fixture() -> AsyncGenerator[StatefulApp]:
    """Create a StatefulApp with Bearer auth and no event_subscriber.

    Yields:
        StatefulApp with app, client, and session_store configured.
    """
    fastapi_app, session_store = _build_stateful_app(event_subscriber=None)
    transport = httpx.ASGITransport(app=fastapi_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield StatefulApp(app=fastapi_app, client=async_client, session_store=session_store)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
        settle_seconds: Extra seconds to wait after headers arrive.

    Returns:
        SseCapture instance populated with status_code, headers, and chunks.
    """
    capture = SseCapture(app)
    transport = httpx.ASGITransport(app=capture)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        task = asyncio.create_task(client.get("/mcp", headers=headers))

        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(capture.headers_received.wait()), timeout=5.0)

        if settle_seconds > 0:
            await asyncio.sleep(settle_seconds)

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    return capture


# ---------------------------------------------------------------------------
# AC-1: event_subscriber events delivered in SSE stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sse_delivers_subscriber_events() -> None:
    """SSE stream delivers events emitted by event_subscriber.

    Creates an event_subscriber that yields one event, verifies the SSE
    stream body contains the event data payload.
    """
    received_calls: list[tuple[str, int | None]] = []

    async def event_subscriber(
        session_id: str,
        last_event_id: int | None,
    ) -> AsyncGenerator[tuple[int, dict]]:
        """Yield one event then stop.

        Args:
            session_id: MCP session ID.
            last_event_id: Last-Event-ID value from header, or None.
        """
        received_calls.append((session_id, last_event_id))
        yield (1, {"jsonrpc": "2.0", "method": "notifications/tools/list_changed", "params": {}})

    fastapi_app, _ = _build_stateful_app(event_subscriber=event_subscriber)
    transport = httpx.ASGITransport(app=fastapi_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _create_session(client)

    capture = await _run_sse_capture(
        fastapi_app,
        headers={
            "Authorization": "Bearer test-token",
            "Mcp-Session-Id": session_id,
        },
        settle_seconds=0.1,
    )

    assert capture.status_code == 200
    full_content = "".join(capture.chunks)
    assert "notifications/tools/list_changed" in full_content, (
        f"Event payload not found in SSE stream: {full_content!r}"
    )


# ---------------------------------------------------------------------------
# AC-2: event formatted as id: N\nevent: message\ndata: {...}\n\n
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sse_event_format() -> None:
    """SSE events have the format: id: N\\nevent: message\\ndata: {json}\\n\\n.

    Verifies that the event_id, "event: message" label, and JSON data lines
    appear in the correct SSE format in the stream body.
    """
    payload = {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"value": 42}}

    async def event_subscriber(
        session_id: str,
        last_event_id: int | None,
    ) -> AsyncGenerator[tuple[int, dict]]:
        """Yield one event with id=7.

        Args:
            session_id: MCP session ID.
            last_event_id: Last-Event-ID value from header, or None.
        """
        yield (7, payload)

    fastapi_app, _ = _build_stateful_app(event_subscriber=event_subscriber)
    transport = httpx.ASGITransport(app=fastapi_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _create_session(client)

    capture = await _run_sse_capture(
        fastapi_app,
        headers={
            "Authorization": "Bearer test-token",
            "Mcp-Session-Id": session_id,
        },
        settle_seconds=0.1,
    )

    assert capture.status_code == 200
    full_content = "".join(capture.chunks)
    expected_fragment = f"id: 7\nevent: message\ndata: {json.dumps(payload)}\n\n"
    assert expected_fragment in full_content, (
        f"Expected SSE event format not found in stream.\n"
        f"Expected fragment: {expected_fragment!r}\n"
        f"Actual content: {full_content!r}"
    )


# ---------------------------------------------------------------------------
# AC-3: Last-Event-ID header value passed to subscriber on reconnect
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sse_last_event_id_passed_to_subscriber() -> None:
    """Last-Event-ID header value is forwarded to event_subscriber as last_event_id.

    Sends Last-Event-ID: 5 and confirms the subscriber receives int 5.
    """
    captured_last_event_id: list[int | None] = []

    async def event_subscriber(
        session_id: str,
        last_event_id: int | None,
    ) -> AsyncGenerator[tuple[int, dict]]:
        """Record last_event_id and yield nothing.

        Args:
            session_id: MCP session ID.
            last_event_id: Last-Event-ID value from header, or None.
        """
        captured_last_event_id.append(last_event_id)
        return
        yield  # make this an async generator

    fastapi_app, _ = _build_stateful_app(event_subscriber=event_subscriber)
    transport = httpx.ASGITransport(app=fastapi_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _create_session(client)

    await _run_sse_capture(
        fastapi_app,
        headers={
            "Authorization": "Bearer test-token",
            "Mcp-Session-Id": session_id,
            "Last-Event-ID": "5",
        },
        settle_seconds=0.1,
    )

    assert len(captured_last_event_id) >= 1, "event_subscriber was not called"
    assert captured_last_event_id[0] == 5, f"Expected last_event_id=5, got {captured_last_event_id[0]!r}"


# ---------------------------------------------------------------------------
# AC-4: keepalive fires every 30s between events (mock asyncio.sleep)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sse_keepalive_at_30s_interval() -> None:
    """Keepalive comment appears in stream when asyncio.wait_for timeout fires.

    The router uses asyncio.wait_for(gen.__anext__(), timeout=30) to wait for
    the next event. When the timeout fires (TimeoutError), it yields a keepalive.
    This test patches asyncio.wait_for in the router module so the 30-second
    timeout raises TimeoutError immediately, triggering the keepalive branch.
    """

    async def event_subscriber(
        session_id: str,
        last_event_id: int | None,
    ) -> AsyncGenerator[tuple[int, dict]]:
        """Yield no events — forces wait_for timeout to fire.

        Args:
            session_id: MCP session ID.
            last_event_id: Last-Event-ID value from header, or None.
        """
        # Never yields; wait_for will hit its timeout on __anext__.
        await asyncio.Event().wait()
        yield  # make this an async generator

    fastapi_app, _ = _build_stateful_app(event_subscriber=event_subscriber)
    transport = httpx.ASGITransport(app=fastapi_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _create_session(client)

    real_wait_for = asyncio.wait_for

    async def instant_timeout(coro: Any, timeout: float, **kwargs: Any) -> Any:
        """Raise TimeoutError immediately for 30-second waits; delegate otherwise.

        Args:
            coro: Coroutine or future to wrap.
            timeout: Timeout in seconds.
            **kwargs: Additional keyword arguments passed to real wait_for.
        """
        if timeout >= 10:
            # Cancel the wrapped coroutine to avoid resource leaks before raising.
            task = asyncio.ensure_future(coro)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            raise TimeoutError
        return await real_wait_for(coro, timeout=timeout, **kwargs)

    with patch("fastapi_mcp_router.router.asyncio.wait_for", new=instant_timeout):
        capture = await _run_sse_capture(
            fastapi_app,
            headers={
                "Authorization": "Bearer test-token",
                "Mcp-Session-Id": session_id,
            },
            settle_seconds=0.1,
        )

    assert capture.status_code == 200
    full_content = "".join(capture.chunks)
    assert "keepalive" in full_content, f"Expected keepalive comment in SSE stream, got: {full_content!r}"


# ---------------------------------------------------------------------------
# AC-5: client disconnect handled with logging, no server crash
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.unit
async def test_sse_client_disconnect_logged(caplog: Any) -> None:
    """SSE stream logs cancellation and does not crash when client disconnects.

    Cancels the streaming task after headers arrive. Verifies the router
    logs the cancellation event and does not raise an unhandled exception.
    """
    import logging

    fastapi_app, _ = _build_stateful_app(event_subscriber=None)
    transport = httpx.ASGITransport(app=fastapi_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _create_session(client)

    with caplog.at_level(logging.INFO, logger="fastapi_mcp_router.router"):
        capture = await _run_sse_capture(
            fastapi_app,
            headers={
                "Authorization": "Bearer test-token",
                "Mcp-Session-Id": session_id,
            },
            settle_seconds=0.1,
        )

    # Stream must have started successfully before disconnect
    assert capture.status_code == 200

    # Router must log cancellation without crashing
    cancel_logged = any(
        "cancelled" in record.message.lower() or "SSE stream" in record.message for record in caplog.records
    )
    assert cancel_logged, f"Expected cancellation log entry, got records: {[r.message for r in caplog.records]}"


# ---------------------------------------------------------------------------
# AC-6: event_subscriber=None → keepalive-only stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sse_keepalive_only_when_no_subscriber() -> None:
    """SSE stream sends only keepalives when event_subscriber is None.

    Patches asyncio.sleep to return immediately, then verifies the stream
    contains a keepalive and no application event lines.
    """
    fastapi_app, _ = _build_stateful_app(event_subscriber=None)
    transport = httpx.ASGITransport(app=fastapi_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _create_session(client)

    real_sleep = asyncio.sleep

    async def fast_sleep(delay: float, *args: Any, **kwargs: Any) -> None:
        if delay >= 10:
            return
        await real_sleep(delay)

    with patch("asyncio.sleep", new=fast_sleep):
        capture = await _run_sse_capture(
            fastapi_app,
            headers={
                "Authorization": "Bearer test-token",
                "Mcp-Session-Id": session_id,
            },
            settle_seconds=0.1,
        )

    assert capture.status_code == 200
    full_content = "".join(capture.chunks)
    assert "keepalive" in full_content, f"Expected keepalive-only stream, got: {full_content!r}"
    assert "event: message" not in full_content, (
        f"Expected no application events, but found 'event: message' in: {full_content!r}"
    )


# ---------------------------------------------------------------------------
# AC-87: empty generator closes stream cleanly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sse_empty_generator_closes_cleanly() -> None:
    """SSE stream closes without error when event_subscriber yields nothing.

    An async generator that returns immediately (StopAsyncIteration) must
    cause the stream to close cleanly. The response must be HTTP 200.
    """

    async def empty_subscriber(
        session_id: str,
        last_event_id: int | None,
    ) -> AsyncGenerator[tuple[int, dict]]:
        """Yield nothing; stop immediately.

        Args:
            session_id: MCP session ID.
            last_event_id: Last-Event-ID value from header, or None.
        """
        return
        yield  # make this an async generator

    fastapi_app, _ = _build_stateful_app(event_subscriber=empty_subscriber)
    transport = httpx.ASGITransport(app=fastapi_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _create_session(client)

    capture = await _run_sse_capture(
        fastapi_app,
        headers={
            "Authorization": "Bearer test-token",
            "Mcp-Session-Id": session_id,
        },
        settle_seconds=0.1,
    )

    # Stream must start successfully — router must not crash on StopAsyncIteration
    assert capture.status_code == 200


# ---------------------------------------------------------------------------
# AC-95: Last-Event-ID: 0 → subscriber receives int 0 (not None)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sse_last_event_id_zero_passed_as_int() -> None:
    """Last-Event-ID: 0 is forwarded as integer 0, not None.

    Verifies the router parses the string "0" to int 0 and passes it to
    event_subscriber. The subscriber must NOT receive None.
    """
    captured: list[int | None] = []

    async def event_subscriber(
        session_id: str,
        last_event_id: int | None,
    ) -> AsyncGenerator[tuple[int, dict]]:
        """Record last_event_id then stop.

        Args:
            session_id: MCP session ID.
            last_event_id: Last-Event-ID value from header, or None.
        """
        captured.append(last_event_id)
        return
        yield  # make this an async generator

    fastapi_app, _ = _build_stateful_app(event_subscriber=event_subscriber)
    transport = httpx.ASGITransport(app=fastapi_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _create_session(client)

    await _run_sse_capture(
        fastapi_app,
        headers={
            "Authorization": "Bearer test-token",
            "Mcp-Session-Id": session_id,
            "Last-Event-ID": "0",
        },
        settle_seconds=0.1,
    )

    assert len(captured) >= 1, "event_subscriber was not called"
    assert captured[0] == 0, f"Expected last_event_id=0 (int), got {captured[0]!r}"
    assert captured[0] is not None, "Expected int 0, got None"


# ---------------------------------------------------------------------------
# AC-96: Last-Event-ID absent → subscriber receives None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sse_last_event_id_absent_passed_as_none() -> None:
    """Absent Last-Event-ID header causes subscriber to receive None.

    When no Last-Event-ID header is present, event_subscriber's
    last_event_id parameter must be None.
    """
    captured: list[int | None] = []

    async def event_subscriber(
        session_id: str,
        last_event_id: int | None,
    ) -> AsyncGenerator[tuple[int, dict]]:
        """Record last_event_id then stop.

        Args:
            session_id: MCP session ID.
            last_event_id: Last-Event-ID value from header, or None.
        """
        captured.append(last_event_id)
        return
        yield  # make this an async generator

    fastapi_app, _ = _build_stateful_app(event_subscriber=event_subscriber)
    transport = httpx.ASGITransport(app=fastapi_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _create_session(client)

    await _run_sse_capture(
        fastapi_app,
        headers={
            "Authorization": "Bearer test-token",
            "Mcp-Session-Id": session_id,
            # No Last-Event-ID header
        },
        settle_seconds=0.1,
    )

    assert len(captured) >= 1, "event_subscriber was not called"
    assert captured[0] is None, f"Expected last_event_id=None, got {captured[0]!r}"


# ---------------------------------------------------------------------------
# EC-11: session_store + session_getter → ValueError at creation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_create_mcp_router_raises_when_both_session_store_and_getter() -> None:
    """create_mcp_router raises ValueError when both session_store and session_getter are provided.

    session_store and session_getter are mutually exclusive. Passing both
    must raise ValueError at router construction time, before any request
    is handled.
    """

    async def dummy_session_getter(session_id: str) -> McpSessionData | None:
        """Stub session getter that always returns None.

        Args:
            session_id: Session identifier (unused).

        Returns:
            None always.
        """
        return None

    with pytest.raises(ValueError, match="mutually exclusive"):
        create_mcp_router(
            MCPToolRegistry(),
            session_getter=dummy_session_getter,
            session_store=object(),
        )
