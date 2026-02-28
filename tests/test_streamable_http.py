"""Integration tests for Streamable HTTP (POST endpoint SSE responses).

Tests the POST /mcp endpoint's ability to return StreamingResponse when the
client sends Accept: text/event-stream and a session_store is configured (IR-3).

Covers:
- AC-1: POST with Accept: text/event-stream + method=initialize returns
  StreamingResponse with Content-Type: text/event-stream and Mcp-Session-Id header.
- AC-2: POST with Accept: application/json returns JSONResponse (existing behavior).
- AC-3: Initialize stores clientInfo from params; session_store.get() reflects it.
- AC-4: Initialize stores capabilities from params; session_store.get() reflects it.
- AC-5: legacy_sse=True registers GET endpoint; client receives SSE stream via GET.
- AC-6/AC-23: legacy_sse=False (default) omits GET endpoint; GET returns 405.
- AC-7: legacy_sse=True serves both GET SSE and POST Streamable HTTP simultaneously.
- AC-8: Client with protocolVersion: "2025-03-26" receives "2025-03-26" in response.

SSE response reading strategy:
  httpx.AsyncClient with ASGITransport accumulates SSE body automatically.
  The body contains one or more 'data: ...' lines terminated by a blank line.
  Tests decode response.text to verify the SSE payload.
"""

import asyncio
import json
from collections.abc import Awaitable, Callable

import httpx
import pytest
from fastapi import FastAPI

from fastapi_mcp_router import InMemorySessionStore, MCPToolRegistry, create_mcp_router

# ---------------------------------------------------------------------------
# ASGI capture middleware (replicated from test_sse_streaming.py pattern)
# ---------------------------------------------------------------------------


class SseCapture:
    """ASGI middleware that captures SSE response headers and body chunks.

    Intercepts http.response.start to record the status code and headers,
    then signals headers_received so tests can inspect them before the
    streaming body completes.

    Attributes:
        app: Inner ASGI application to wrap.
        status_code: HTTP status code from http.response.start, or None.
        headers: Response headers dict (lower-case keys), empty until set.
        chunks: Decoded body chunks received.
        headers_received: Event that fires when http.response.start arrives.
    """

    def __init__(self, app: Callable[..., Awaitable[None]]) -> None:
        self.app = app
        self.status_code: int | None = None
        self.headers: dict[str, str] = {}
        self.chunks: list[str] = []
        self.headers_received: asyncio.Event = asyncio.Event()

    async def __call__(
        self,
        scope: object,
        receive: object,
        send: Callable[..., Awaitable[None]],
    ) -> None:
        """Wrap the inner app, capturing response start and body messages.

        Args:
            scope: ASGI connection scope.
            receive: ASGI receive callable.
            send: ASGI send callable.
        """

        async def capturing_send(message: dict[str, object]) -> None:
            if message["type"] == "http.response.start":
                status = message["status"]
                assert isinstance(status, int)
                self.status_code = status
                raw_headers = message.get("headers", [])
                assert isinstance(raw_headers, list)
                self.headers = {
                    k.decode(): v.decode()
                    for k, v in raw_headers  # type: ignore[misc]
                }
                self.headers_received.set()
            elif message["type"] == "http.response.body":
                body = message.get("body", b"")
                assert isinstance(body, bytes)
                if body:
                    self.chunks.append(body.decode())
            await send(message)

        await self.app(scope, receive, capturing_send)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_stateful_app(store: InMemorySessionStore) -> FastAPI:
    """Create a FastAPI app configured with the given session_store.

    Args:
        store: InMemorySessionStore instance to pass to create_mcp_router.

    Returns:
        FastAPI app with POST /mcp endpoint and session_store enabled.
    """
    registry = MCPToolRegistry()

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        """Accept any request with X-API-Key or Bearer token."""
        return api_key is not None or bearer_token is not None

    router = create_mcp_router(registry, session_store=store, auth_validator=auth_validator)
    app = FastAPI()
    app.include_router(router, prefix="/mcp")
    return app


async def _post_initialize(
    client: httpx.AsyncClient,
    extra_headers: dict[str, str] | None = None,
    params: dict[str, object] | None = None,
) -> httpx.Response:
    """POST initialize to /mcp and return the raw response.

    Args:
        client: Configured AsyncClient for the app.
        extra_headers: Additional request headers to merge in.
        params: JSON-RPC params dict (merged into the request body).

    Returns:
        httpx.Response from the POST.
    """
    headers = {"X-API-Key": "test-key", "MCP-Protocol-Version": "2025-06-18"}
    if extra_headers:
        headers.update(extra_headers)

    body: dict[str, object] = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2025-06-18"},
    }
    if params:
        existing_params = body["params"]
        assert isinstance(existing_params, dict)
        existing_params.update(params)

    return await client.post("/mcp", json=body, headers=headers)


# ---------------------------------------------------------------------------
# AC-1: POST with Accept: text/event-stream returns StreamingResponse
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_post_sse_accept_returns_streaming_response_with_headers() -> None:
    """AC-1: POST initialize with Accept: text/event-stream returns StreamingResponse.

    Verifies that the response Content-Type is text/event-stream and that
    the Mcp-Session-Id header is set when session_store is configured.
    """
    store = InMemorySessionStore()
    app = _build_stateful_app(store)

    capture = SseCapture(app)
    transport = httpx.ASGITransport(app=capture)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await _post_initialize(
            client,
            extra_headers={"Accept": "text/event-stream"},
        )

    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    assert "text/event-stream" in content_type, f"Expected text/event-stream, got: {content_type!r}"
    assert "mcp-session-id" in response.headers, "Expected Mcp-Session-Id header in response"
    session_id = response.headers["mcp-session-id"]
    assert session_id, "Mcp-Session-Id must be non-empty"


# ---------------------------------------------------------------------------
# AC-1 (body): SSE response body contains a valid JSON-RPC data event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_post_sse_body_contains_jsonrpc_data_event() -> None:
    """AC-1: SSE response body contains a 'data: ...' line with the JSON-RPC result.

    The router wraps the JSON-RPC initialize result in a single SSE data event
    formatted as 'data: <json>\\n\\n'.
    """
    store = InMemorySessionStore()
    app = _build_stateful_app(store)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await _post_initialize(
            client,
            extra_headers={"Accept": "text/event-stream"},
        )

    assert response.status_code == 200
    body_text = response.text
    assert "data: " in body_text, f"Expected SSE 'data: ' prefix in body, got: {body_text!r}"

    # Extract the JSON payload from the first data line
    for line in body_text.splitlines():
        if line.startswith("data: "):
            payload = json.loads(line[len("data: ") :])
            assert payload.get("jsonrpc") == "2.0"
            assert "result" in payload
            break
    else:
        pytest.fail(f"No 'data: ' line found in SSE body: {body_text!r}")


# ---------------------------------------------------------------------------
# AC-2: POST with Accept: application/json returns JSONResponse unchanged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_post_json_accept_returns_json_response() -> None:
    """AC-2: POST initialize with Accept: application/json returns JSONResponse.

    Verifies the existing behavior is unaffected: a standard JSON-RPC response
    with Content-Type application/json and Mcp-Session-Id header is returned.
    """
    store = InMemorySessionStore()
    app = _build_stateful_app(store)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await _post_initialize(
            client,
            extra_headers={"Accept": "application/json"},
        )

    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    assert "application/json" in content_type, f"Expected application/json, got: {content_type!r}"
    assert "text/event-stream" not in content_type

    data = response.json()
    assert data.get("jsonrpc") == "2.0"
    assert "result" in data


# ---------------------------------------------------------------------------
# AC-3: Initialize stores clientInfo from params
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_initialize_stores_client_info_in_session() -> None:
    """AC-3: Initialize stores clientInfo from params; session_store.get() has client_info.

    Sends initialize with clientInfo: {name: 'test-client', version: '1.0'} and
    verifies the created session reflects that value.
    """
    store = InMemorySessionStore()
    app = _build_stateful_app(store)

    client_info = {"name": "test-client", "version": "1.0"}

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await _post_initialize(
            client,
            params={"clientInfo": client_info},
        )

    assert response.status_code == 200
    session_id = response.headers.get("mcp-session-id")
    assert session_id, "initialize must return Mcp-Session-Id header"

    session = await store.get(session_id)
    assert session is not None, f"Session {session_id!r} not found in store after initialize"
    assert session.client_info == client_info, f"Expected client_info={client_info!r}, got {session.client_info!r}"


# ---------------------------------------------------------------------------
# AC-4: Initialize stores capabilities from params
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_initialize_stores_capabilities_in_session() -> None:
    """AC-4: Initialize stores capabilities from params; session_store.get() has capabilities.

    Sends initialize with capabilities: {roots: {listChanged: True}} and verifies
    the created session reflects that value.
    """
    store = InMemorySessionStore()
    app = _build_stateful_app(store)

    capabilities = {"roots": {"listChanged": True}}

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await _post_initialize(
            client,
            params={"capabilities": capabilities},
        )

    assert response.status_code == 200
    session_id = response.headers.get("mcp-session-id")
    assert session_id, "initialize must return Mcp-Session-Id header"

    session = await store.get(session_id)
    assert session is not None, f"Session {session_id!r} not found in store after initialize"
    assert session.capabilities == capabilities, f"Expected capabilities={capabilities!r}, got {session.capabilities!r}"


# ---------------------------------------------------------------------------
# AC-5: legacy_sse=True registers GET endpoint; client receives SSE stream
# ---------------------------------------------------------------------------


def _build_legacy_sse_app() -> FastAPI:
    """Create a FastAPI app with legacy_sse=True and session callbacks.

    Args: (none)

    Returns:
        FastAPI app with GET /mcp SSE endpoint and POST /mcp endpoint.
    """
    from datetime import UTC, datetime
    from uuid import uuid4

    from fastapi_mcp_router.types import McpSessionData

    session_store_dict: dict[str, McpSessionData] = {}

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        """Accept all Bearer token requests."""
        return bearer_token is not None

    async def session_getter(session_id: str) -> McpSessionData | None:
        """Return session data from in-memory store.

        Args:
            session_id: Session identifier to look up.

        Returns:
            McpSessionData if found, None otherwise.
        """
        return session_store_dict.get(session_id)

    async def session_creator(oauth_client_id: object, connection_id: object) -> str:
        """Generate a session ID and store McpSessionData.

        Args:
            oauth_client_id: UUID of the OAuth client, or None.
            connection_id: UUID of the connection, or None.

        Returns:
            The new session ID string.
        """
        session_id = str(uuid4())
        session_store_dict[session_id] = McpSessionData(
            session_id=session_id,
            oauth_client_id=None,
            connection_id=None,
            last_event_id=0,
            created_at=datetime.now(UTC),
        )
        return session_id

    from fastapi import Request

    fastapi_app = FastAPI()

    @fastapi_app.middleware("http")
    async def set_bearer_state(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Set request.state fields so session_creator can run for Bearer tokens."""
        from uuid import uuid4 as _uuid4

        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            request.state.oauth_client_id = _uuid4()
            request.state.connection_id = _uuid4()
        return await call_next(request)

    mcp_router = create_mcp_router(
        MCPToolRegistry(),
        auth_validator=auth_validator,
        session_getter=session_getter,
        session_creator=session_creator,
        legacy_sse=True,
    )
    fastapi_app.include_router(mcp_router, prefix="/mcp")
    return fastapi_app


@pytest.mark.asyncio
@pytest.mark.integration
async def test_legacy_sse_true_get_returns_200_sse_stream() -> None:
    """AC-5: legacy_sse=True registers GET endpoint; GET returns 200 with SSE stream.

    Sends a GET /mcp with a Bearer token (which triggers session_creator) and
    verifies the response status is 200 with Content-Type text/event-stream.
    """
    import asyncio
    import contextlib

    fastapi_app = _build_legacy_sse_app()
    capture = SseCapture(fastapi_app)
    transport = httpx.ASGITransport(app=capture)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        task = asyncio.create_task(client.get("/mcp", headers={"Authorization": "Bearer test-token"}))

        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(capture.headers_received.wait()), timeout=5.0)

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    assert capture.status_code == 200, f"Expected 200 from GET /mcp with legacy_sse=True, got {capture.status_code}"
    content_type = capture.headers.get("content-type", "")
    assert "text/event-stream" in content_type, f"Expected text/event-stream, got: {content_type!r}"


# ---------------------------------------------------------------------------
# AC-6 / AC-23: legacy_sse=False (default) omits GET endpoint; GET returns 405
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_legacy_sse_false_get_returns_405() -> None:
    """AC-6/AC-23: legacy_sse=False (default) omits GET endpoint; GET returns 405.

    Creates a router without legacy_sse=True and verifies that GET /mcp returns
    HTTP 405. This covers both AC-6 (default is False) and AC-23 (explicit False).
    """
    store = InMemorySessionStore()
    app = _build_stateful_app(store)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/mcp",
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 405, f"Expected 405 when legacy_sse=False, got {response.status_code}"


# ---------------------------------------------------------------------------
# AC-7: legacy_sse=True serves both GET SSE and POST Streamable HTTP
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_legacy_sse_true_serves_both_get_and_post() -> None:
    """AC-7: legacy_sse=True serves both GET SSE and POST Streamable HTTP simultaneously.

    Verifies that with legacy_sse=True:
    - GET /mcp with Bearer token returns 200 (SSE endpoint is registered).
    - POST /mcp with Accept: text/event-stream returns 200 (POST endpoint works).
    """
    import asyncio
    import contextlib

    fastapi_app = _build_legacy_sse_app()

    # Verify GET returns 200
    capture = SseCapture(fastapi_app)
    transport_get = httpx.ASGITransport(app=capture)
    async with httpx.AsyncClient(transport=transport_get, base_url="http://test") as client:
        task = asyncio.create_task(client.get("/mcp", headers={"Authorization": "Bearer test-token"}))
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(capture.headers_received.wait()), timeout=5.0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    assert capture.status_code == 200, f"GET /mcp expected 200, got {capture.status_code}"

    # Verify POST returns 200 with SSE accept header
    transport_post = httpx.ASGITransport(app=fastapi_app)
    async with httpx.AsyncClient(transport=transport_post, base_url="http://test") as client:
        response = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-06-18"},
            },
            headers={
                "Authorization": "Bearer test-token",
                "MCP-Protocol-Version": "2025-06-18",
                "Accept": "text/event-stream",
            },
        )

    assert response.status_code == 200, f"POST /mcp expected 200, got {response.status_code}"
    data = response.json()
    assert data.get("jsonrpc") == "2.0", f"Expected JSON-RPC 2.0 response from POST, got: {data!r}"
    assert "result" in data, f"Expected 'result' key in POST response: {data!r}"


# ---------------------------------------------------------------------------
# AC-8: protocolVersion: "2025-03-26" in request reflected in server response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_protocol_version_2025_03_26_reflected_in_response() -> None:
    """AC-8: Client sending protocolVersion: "2025-03-26" receives "2025-03-26" in response.

    Sends initialize with protocolVersion "2025-03-26" in both the params and the
    MCP-Protocol-Version header, then verifies the JSON-RPC result contains the
    same version string.
    """
    registry = MCPToolRegistry()
    mcp_router = create_mcp_router(registry)
    app = FastAPI()
    app.include_router(mcp_router, prefix="/mcp")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-03-26"},
            },
            headers={"MCP-Protocol-Version": "2025-03-26", "X-API-Key": "test-key"},
        )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data.get("jsonrpc") == "2.0"
    assert "result" in data, f"Expected 'result' key in response: {data!r}"
    result = data["result"]
    assert result.get("protocolVersion") == "2025-03-26", (
        f"Expected protocolVersion '2025-03-26' in result, got: {result.get('protocolVersion')!r}"
    )


# ---------------------------------------------------------------------------
# AC-19 / EC-2: POST with Accept: text/event-stream but session_store=None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_post_sse_accept_stateless_returns_json_response() -> None:
    """AC-19/EC-2: POST Accept: text/event-stream with session_store=None returns JSONResponse.

    When no session_store is configured, the router cannot create sessions.
    SSE requests fall through to JSONResponse (EC-2 fallback).
    """
    registry = MCPToolRegistry()
    router = create_mcp_router(registry)
    app = FastAPI()
    app.include_router(router, prefix="/mcp")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-06-18"},
            },
            headers={
                "Accept": "text/event-stream",
                "MCP-Protocol-Version": "2025-06-18",
                "X-API-Key": "test-key",
            },
        )

    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    assert "application/json" in content_type, (
        f"Expected application/json fallback for stateless SSE request, got: {content_type!r}"
    )
    assert "text/event-stream" not in content_type, (
        f"Did not expect text/event-stream when session_store=None, got: {content_type!r}"
    )
    data = response.json()
    assert data.get("jsonrpc") == "2.0"
    assert "result" in data


# ---------------------------------------------------------------------------
# AC-20 / EC-5: Initialize without clientInfo defaults to client_info={}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_initialize_without_client_info_stores_empty_dict() -> None:
    """AC-20/EC-5: Initialize request without clientInfo creates session with client_info={}.

    When params contains no clientInfo key, the router defaults to {} and stores
    that value in the session. session_store.get() must reflect client_info={}.
    """
    store = InMemorySessionStore()
    app = _build_stateful_app(store)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await _post_initialize(client)

    assert response.status_code == 200
    session_id = response.headers.get("mcp-session-id")
    assert session_id, "initialize must return Mcp-Session-Id header"

    session = await store.get(session_id)
    assert session is not None, f"Session {session_id!r} not found in store"
    assert session.client_info == {}, f"Expected client_info={{}} when clientInfo omitted, got {session.client_info!r}"


# ---------------------------------------------------------------------------
# AC-21 / EC-6: Initialize without capabilities defaults to capabilities={}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_initialize_without_capabilities_stores_empty_dict() -> None:
    """AC-21/EC-6: Initialize request without capabilities creates session with capabilities={}.

    When params contains no capabilities key, the router defaults to {} and stores
    that value in the session. session_store.get() must reflect capabilities={}.
    """
    store = InMemorySessionStore()
    app = _build_stateful_app(store)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await _post_initialize(client)

    assert response.status_code == 200
    session_id = response.headers.get("mcp-session-id")
    assert session_id, "initialize must return Mcp-Session-Id header"

    session = await store.get(session_id)
    assert session is not None, f"Session {session_id!r} not found in store"
    assert session.capabilities == {}, (
        f"Expected capabilities={{}} when capabilities omitted, got {session.capabilities!r}"
    )


# ---------------------------------------------------------------------------
# AC-24: Accept: application/json, text/event-stream chooses SSE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_post_combined_accept_header_chooses_sse() -> None:
    """AC-24: POST with Accept: application/json, text/event-stream returns SSE response.

    Per MCP spec: when Accept includes text/event-stream, the server SHOULD use SSE.
    The router checks "text/event-stream" in accept, so the combined header triggers SSE.
    """
    store = InMemorySessionStore()
    app = _build_stateful_app(store)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await _post_initialize(
            client,
            extra_headers={"Accept": "application/json, text/event-stream"},
        )

    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    assert "text/event-stream" in content_type, (
        f"Expected text/event-stream when Accept includes text/event-stream, got: {content_type!r}"
    )


# ---------------------------------------------------------------------------
# AC-25 / EC-1: Accept with unknown media type defaults to JSON response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_post_unknown_accept_header_defaults_to_json() -> None:
    """AC-25/EC-1: POST with Accept: text/xml falls back to JSONResponse.

    When the Accept header contains no recognized media types, the router
    treats the request as application/json (EC-1 fallback).
    """
    store = InMemorySessionStore()
    app = _build_stateful_app(store)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await _post_initialize(
            client,
            extra_headers={"Accept": "text/xml"},
        )

    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    assert "application/json" in content_type, (
        f"Expected application/json fallback for Accept: text/xml, got: {content_type!r}"
    )
    assert "text/event-stream" not in content_type
    data = response.json()
    assert data.get("jsonrpc") == "2.0"
    assert "result" in data


# ---------------------------------------------------------------------------
# AC-26: Concurrent POST SSE responses with distinct session IDs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_concurrent_post_sse_requests_get_distinct_session_ids() -> None:
    """AC-26: Two concurrent POST SSE requests return distinct Mcp-Session-Id headers.

    Uses asyncio.gather() to send two initialize requests simultaneously.
    Each request must produce a unique session ID, confirming independent sessions.
    """
    store = InMemorySessionStore()
    app = _build_stateful_app(store)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response_a, response_b = await asyncio.gather(
            _post_initialize(client, extra_headers={"Accept": "text/event-stream"}),
            _post_initialize(client, extra_headers={"Accept": "text/event-stream"}),
        )

    assert response_a.status_code == 200, f"Request A failed with {response_a.status_code}"
    assert response_b.status_code == 200, f"Request B failed with {response_b.status_code}"

    session_id_a = response_a.headers.get("mcp-session-id")
    session_id_b = response_b.headers.get("mcp-session-id")

    assert session_id_a, "Request A must return Mcp-Session-Id header"
    assert session_id_b, "Request B must return Mcp-Session-Id header"
    assert session_id_a != session_id_b, (
        f"Concurrent requests must produce distinct session IDs, both got: {session_id_a!r}"
    )


# ---------------------------------------------------------------------------
# AC-14: Full MCP 2025-06-18 Streamable HTTP lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_mcp_2025_06_18_full_lifecycle_initialize_sse_then_tools_call() -> None:
    """AC-14: MCP 2025-06-18 client initializes via SSE POST then calls a tool via POST.

    Step 1: POST initialize with Accept: text/event-stream and
    MCP-Protocol-Version: 2025-06-18. Verify the response is SSE and
    extract the Mcp-Session-Id header.

    Step 2: POST tools/call with the session ID from step 1. Verify the
    JSON-RPC result contains the expected tool output.

    This test verifies the full Streamable HTTP round-trip defined by
    NFR-HTTP-1.
    """
    registry = MCPToolRegistry()

    @registry.tool()
    async def greet(name: str) -> str:
        """Return a greeting for the given name.

        Args:
            name: Name to greet.

        Returns:
            Greeting string.
        """
        return f"Hello, {name}!"

    store = InMemorySessionStore()

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        """Accept any request with X-API-Key header."""
        return api_key is not None

    router = create_mcp_router(registry, session_store=store, auth_validator=auth_validator)
    app = FastAPI()
    app.include_router(router, prefix="/mcp")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Step 1: POST initialize with Accept: text/event-stream
        init_response = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "clientInfo": {"name": "mcp-test-client", "version": "1.0"},
                    "capabilities": {},
                },
            },
            headers={
                "X-API-Key": "test-key",
                "MCP-Protocol-Version": "2025-06-18",
                "Accept": "text/event-stream",
            },
        )

        assert init_response.status_code == 200, (
            f"initialize expected 200, got {init_response.status_code}: {init_response.text}"
        )
        content_type = init_response.headers.get("content-type", "")
        assert "text/event-stream" in content_type, (
            f"Expected SSE response to initialize, got content-type: {content_type!r}"
        )
        session_id = init_response.headers.get("mcp-session-id")
        assert session_id, "Initialize SSE response must include Mcp-Session-Id header"

        # Step 2: POST tools/call with the session ID from initialize
        call_response = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "greet",
                    "arguments": {"name": "World"},
                },
            },
            headers={
                "X-API-Key": "test-key",
                "MCP-Protocol-Version": "2025-06-18",
                "Mcp-Session-Id": session_id,
            },
        )

    assert call_response.status_code == 200, (
        f"tools/call expected 200, got {call_response.status_code}: {call_response.text}"
    )
    body = call_response.json()
    assert body.get("jsonrpc") == "2.0", f"Expected jsonrpc 2.0, got: {body!r}"
    assert body.get("id") == 2, f"Expected id 2, got: {body!r}"
    assert "result" in body, f"Expected 'result' in tools/call response, got: {body!r}"
    result = body["result"]
    content = result.get("content", [])
    assert len(content) == 1, f"Expected 1 content item, got: {content!r}"
    assert content[0]["type"] == "text", f"Expected text content, got: {content[0]!r}"
    assert content[0]["text"] == "Hello, World!", f"Expected 'Hello, World!', got: {content[0]['text']!r}"
