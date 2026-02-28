"""Tests for OAuth session management in MCP router.

Tests the session creation, validation, and lifecycle for OAuth (Bearer token)
connections, ensuring the initialize method creates sessions and subsequent
requests validate them properly.
"""

from datetime import datetime
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from fastapi_mcp_router import MCPToolRegistry, create_mcp_router
from fastapi_mcp_router.types import McpSessionData

# Test fixtures


@pytest.fixture(name="registry")
def registry_fixture() -> MCPToolRegistry:
    """Create tool registry with test tool."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def ping() -> str:
        """Simple ping tool."""
        return "pong"

    return registry


@pytest.fixture(name="session_store")
def session_store_fixture() -> dict[str, McpSessionData]:
    """Create in-memory session store for testing."""
    return {}


@pytest.fixture(name="connection_id")
def connection_id_fixture() -> UUID:
    """Create test connection ID."""
    return uuid4()


# Helper functions


def make_jsonrpc_request(
    method: str,
    params: dict[str, object] | None = None,
    request_id: int | str | None = 1,
) -> dict[str, object]:
    """Create JSON-RPC 2.0 request body."""
    request: dict[str, object] = {
        "jsonrpc": "2.0",
        "method": method,
    }
    if params is not None:
        request["params"] = params
    if request_id is not None:
        request["id"] = request_id
    return request


# Session management tests


@pytest.mark.integration
def test_oauth_initialize_creates_session_and_returns_header(
    registry: MCPToolRegistry,
    session_store: dict[str, McpSessionData],
    connection_id: UUID,
):
    """Test OAuth initialize request creates session and returns Mcp-Session-Id header."""

    async def session_getter(session_id: str) -> McpSessionData | None:
        """Get session from store."""
        return session_store.get(session_id)

    async def session_creator(oauth_client_id: UUID | None, conn_id: UUID | None) -> str:
        """Create new session."""
        session_id = f"session_{uuid4()}"
        session_store[session_id] = McpSessionData(
            session_id=session_id,
            oauth_client_id=oauth_client_id,
            connection_id=conn_id,
            last_event_id=0,
            created_at=datetime.now(),
        )
        return session_id

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        """Validate bearer token and set connection_id in request.state."""
        return bearer_token == "valid-token"

    # Need to set connection_id in request.state for session creation
    app = FastAPI()

    # Create middleware to set connection_id after auth
    @app.middleware("http")
    async def set_connection_id(request: Request, call_next):
        """Set connection_id in request.state for OAuth connections."""
        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            request.state.connection_id = connection_id
        response = await call_next(request)
        return response

    mcp_router = create_mcp_router(
        registry,
        auth_validator=auth_validator,
        session_getter=session_getter,
        session_creator=session_creator,
    )
    app.include_router(mcp_router, prefix="/mcp")
    client = TestClient(app)

    request = make_jsonrpc_request(
        method="initialize",
        params={"protocolVersion": "2025-06-18"},
    )

    response = client.post(
        "/mcp",
        json=request,
        headers={
            "MCP-Protocol-Version": "2025-06-18",
            "Authorization": "Bearer valid-token",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert data["result"]["protocolVersion"] == "2025-06-18"

    # Should return Mcp-Session-Id header
    assert "Mcp-Session-Id" in response.headers
    session_id = response.headers["Mcp-Session-Id"]
    assert session_id.startswith("session_")

    # Session should exist in store
    assert session_id in session_store
    assert session_store[session_id].connection_id == connection_id


@pytest.mark.integration
def test_oauth_initialize_without_connection_id_returns_401(
    registry: MCPToolRegistry,
    session_store: dict[str, McpSessionData],
):
    """Test OAuth initialize returns 401 when auth_validator doesn't set connection_id."""

    async def session_getter(session_id: str) -> McpSessionData | None:
        """Get session from store."""
        return session_store.get(session_id)

    async def session_creator(oauth_client_id: UUID | None, conn_id: UUID | None) -> str:
        """Create new session."""
        session_id = f"session_{uuid4()}"
        session_store[session_id] = McpSessionData(
            session_id=session_id,
            oauth_client_id=oauth_client_id,
            connection_id=conn_id,
            last_event_id=0,
            created_at=datetime.now(),
        )
        return session_id

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        """Validate bearer token but DON'T set connection_id."""
        return bearer_token == "valid-token"

    app = FastAPI()
    mcp_router = create_mcp_router(
        registry,
        auth_validator=auth_validator,
        session_getter=session_getter,
        session_creator=session_creator,
    )
    app.include_router(mcp_router, prefix="/mcp")
    client = TestClient(app)

    request = make_jsonrpc_request(
        method="initialize",
        params={"protocolVersion": "2025-06-18"},
    )

    response = client.post(
        "/mcp",
        json=request,
        headers={
            "MCP-Protocol-Version": "2025-06-18",
            "Authorization": "Bearer valid-token",
        },
    )

    assert response.status_code == 401
    data = response.json()
    assert "error" in data
    assert data["error"] == "unauthorized"
    assert "identity" in data["error_description"].lower()
    # WWW-Authenticate is absent when oauth_resource_metadata is not configured (AC-9)
    assert "WWW-Authenticate" not in response.headers


@pytest.mark.integration
def test_oauth_tools_call_requires_session_id(
    registry: MCPToolRegistry,
    session_store: dict[str, McpSessionData],
    connection_id: UUID,
):
    """Test OAuth tools/call requires Mcp-Session-Id header."""

    async def session_getter(session_id: str) -> McpSessionData | None:
        """Get session from store."""
        return session_store.get(session_id)

    async def session_creator(oauth_client_id: UUID | None, conn_id: UUID | None) -> str:
        """Create new session."""
        session_id = f"session_{uuid4()}"
        session_store[session_id] = McpSessionData(
            session_id=session_id,
            oauth_client_id=oauth_client_id,
            connection_id=conn_id,
            last_event_id=0,
            created_at=datetime.now(),
        )
        return session_id

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        """Validate bearer token."""
        return bearer_token == "valid-token"

    app = FastAPI()
    mcp_router = create_mcp_router(
        registry,
        auth_validator=auth_validator,
        session_getter=session_getter,
        session_creator=session_creator,
    )
    app.include_router(mcp_router, prefix="/mcp")
    client = TestClient(app)

    request = make_jsonrpc_request(
        method="tools/call",
        params={"name": "ping", "arguments": {}},
    )

    response = client.post(
        "/mcp",
        json=request,
        headers={
            "MCP-Protocol-Version": "2025-06-18",
            "Authorization": "Bearer valid-token",
            # Missing Mcp-Session-Id header
        },
    )

    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"] == "missing_session_id"
    assert "Mcp-Session-Id" in data["error_description"]


@pytest.mark.integration
def test_oauth_tools_call_with_valid_session_succeeds(
    registry: MCPToolRegistry,
    session_store: dict[str, McpSessionData],
    connection_id: UUID,
):
    """Test OAuth tools/call succeeds with valid session ID."""
    # Pre-create session
    session_id = f"session_{uuid4()}"
    session_store[session_id] = McpSessionData(
        session_id=session_id,
        oauth_client_id=None,
        connection_id=connection_id,
        last_event_id=0,
        created_at=datetime.now(),
    )

    async def session_getter(sid: str) -> McpSessionData | None:
        """Get session from store."""
        return session_store.get(sid)

    async def session_creator(oauth_client_id: UUID | None, conn_id: UUID | None) -> str:
        """Create new session."""
        new_sid = f"session_{uuid4()}"
        session_store[new_sid] = McpSessionData(
            session_id=new_sid,
            oauth_client_id=oauth_client_id,
            connection_id=conn_id,
            last_event_id=0,
            created_at=datetime.now(),
        )
        return new_sid

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        """Validate bearer token."""
        return bearer_token == "valid-token"

    app = FastAPI()
    mcp_router = create_mcp_router(
        registry,
        auth_validator=auth_validator,
        session_getter=session_getter,
        session_creator=session_creator,
    )
    app.include_router(mcp_router, prefix="/mcp")
    client = TestClient(app)

    request = make_jsonrpc_request(
        method="tools/call",
        params={"name": "ping", "arguments": {}},
    )

    response = client.post(
        "/mcp",
        json=request,
        headers={
            "MCP-Protocol-Version": "2025-06-18",
            "Authorization": "Bearer valid-token",
            "Mcp-Session-Id": session_id,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert data["result"]["content"][0]["text"] == "pong"


@pytest.mark.integration
def test_oauth_tools_call_with_expired_session_returns_410(
    registry: MCPToolRegistry,
    session_store: dict[str, McpSessionData],
):
    """Test OAuth tools/call with expired/nonexistent session returns 410."""

    async def session_getter(session_id: str) -> McpSessionData | None:
        """Get session from store - simulate expired session."""
        return None  # Session not found/expired

    async def session_creator(oauth_client_id: UUID | None, conn_id: UUID | None) -> str:
        """Create new session."""
        session_id = f"session_{uuid4()}"
        session_store[session_id] = McpSessionData(
            session_id=session_id,
            oauth_client_id=oauth_client_id,
            connection_id=conn_id,
            last_event_id=0,
            created_at=datetime.now(),
        )
        return session_id

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        """Validate bearer token."""
        return bearer_token == "valid-token"

    app = FastAPI()
    mcp_router = create_mcp_router(
        registry,
        auth_validator=auth_validator,
        session_getter=session_getter,
        session_creator=session_creator,
    )
    app.include_router(mcp_router, prefix="/mcp")
    client = TestClient(app)

    request = make_jsonrpc_request(
        method="tools/call",
        params={"name": "ping", "arguments": {}},
    )

    response = client.post(
        "/mcp",
        json=request,
        headers={
            "MCP-Protocol-Version": "2025-06-18",
            "Authorization": "Bearer valid-token",
            "Mcp-Session-Id": "expired_session_123",
        },
    )

    assert response.status_code == 410
    data = response.json()
    assert "error" in data
    assert data["error"] == "session_expired"
    assert "expired_session_123" in data["error_description"]


@pytest.mark.integration
def test_api_key_connections_bypass_session_validation(
    registry: MCPToolRegistry,
    session_store: dict[str, McpSessionData],
):
    """Test API key connections remain stateless and don't require sessions."""

    async def session_getter(session_id: str) -> McpSessionData | None:
        """Get session from store."""
        return session_store.get(session_id)

    async def session_creator(oauth_client_id: UUID | None, conn_id: UUID | None) -> str:
        """Create new session."""
        session_id = f"session_{uuid4()}"
        session_store[session_id] = McpSessionData(
            session_id=session_id,
            oauth_client_id=oauth_client_id,
            connection_id=conn_id,
            last_event_id=0,
            created_at=datetime.now(),
        )
        return session_id

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        """Validate credentials."""
        if api_key:
            return api_key == "valid-key"
        if bearer_token:
            return bearer_token == "valid-token"
        return False

    app = FastAPI()
    mcp_router = create_mcp_router(
        registry,
        auth_validator=auth_validator,
        session_getter=session_getter,
        session_creator=session_creator,
    )
    app.include_router(mcp_router, prefix="/mcp")
    client = TestClient(app)

    # API key connections should work without session management
    request = make_jsonrpc_request(
        method="tools/call",
        params={"name": "ping", "arguments": {}},
    )

    response = client.post(
        "/mcp",
        json=request,
        headers={
            "MCP-Protocol-Version": "2025-06-18",
            "X-API-Key": "valid-key",
            # No Mcp-Session-Id header - API key connections are stateless
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert data["result"]["content"][0]["text"] == "pong"


@pytest.mark.integration
def test_oauth_initialize_does_not_require_session_id_header(
    registry: MCPToolRegistry,
    session_store: dict[str, McpSessionData],
    connection_id: UUID,
):
    """Test initialize method does NOT require Mcp-Session-Id header for OAuth."""

    async def session_getter(session_id: str) -> McpSessionData | None:
        """Get session from store."""
        return session_store.get(session_id)

    async def session_creator(oauth_client_id: UUID | None, conn_id: UUID | None) -> str:
        """Create new session."""
        session_id = f"session_{uuid4()}"
        session_store[session_id] = McpSessionData(
            session_id=session_id,
            oauth_client_id=oauth_client_id,
            connection_id=conn_id,
            last_event_id=0,
            created_at=datetime.now(),
        )
        return session_id

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        """Validate bearer token."""
        return bearer_token == "valid-token"

    app = FastAPI()

    @app.middleware("http")
    async def set_connection_id(request: Request, call_next):
        """Set connection_id in request.state for OAuth connections."""
        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            request.state.connection_id = connection_id
        response = await call_next(request)
        return response

    mcp_router = create_mcp_router(
        registry,
        auth_validator=auth_validator,
        session_getter=session_getter,
        session_creator=session_creator,
    )
    app.include_router(mcp_router, prefix="/mcp")
    client = TestClient(app)

    request = make_jsonrpc_request(
        method="initialize",
        params={"protocolVersion": "2025-06-18"},
    )

    # Initialize should NOT require Mcp-Session-Id header
    response = client.post(
        "/mcp",
        json=request,
        headers={
            "MCP-Protocol-Version": "2025-06-18",
            "Authorization": "Bearer valid-token",
            # No Mcp-Session-Id header - initialize creates the session
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "result" in data

    # Should create and return session ID
    assert "Mcp-Session-Id" in response.headers
    session_id = response.headers["Mcp-Session-Id"]
    assert session_id in session_store


@pytest.mark.integration
def test_oauth_ping_requires_session_id(
    registry: MCPToolRegistry,
    session_store: dict[str, McpSessionData],
    connection_id: UUID,
):
    """Test OAuth ping method requires Mcp-Session-Id header."""
    # Pre-create session
    session_id = f"session_{uuid4()}"
    session_store[session_id] = McpSessionData(
        session_id=session_id,
        oauth_client_id=None,
        connection_id=connection_id,
        last_event_id=0,
        created_at=datetime.now(),
    )

    async def session_getter(sid: str) -> McpSessionData | None:
        """Get session from store."""
        return session_store.get(sid)

    async def session_creator(oauth_client_id: UUID | None, conn_id: UUID | None) -> str:
        """Create new session."""
        new_sid = f"session_{uuid4()}"
        session_store[new_sid] = McpSessionData(
            session_id=new_sid,
            oauth_client_id=oauth_client_id,
            connection_id=conn_id,
            last_event_id=0,
            created_at=datetime.now(),
        )
        return new_sid

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        """Validate bearer token."""
        return bearer_token == "valid-token"

    app = FastAPI()
    mcp_router = create_mcp_router(
        registry,
        auth_validator=auth_validator,
        session_getter=session_getter,
        session_creator=session_creator,
    )
    app.include_router(mcp_router, prefix="/mcp")
    client = TestClient(app)

    # Test ping without session ID fails
    request = make_jsonrpc_request(method="ping")
    response = client.post(
        "/mcp",
        json=request,
        headers={
            "MCP-Protocol-Version": "2025-06-18",
            "Authorization": "Bearer valid-token",
        },
    )

    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "missing_session_id"

    # Test ping with valid session ID succeeds
    response = client.post(
        "/mcp",
        json=request,
        headers={
            "MCP-Protocol-Version": "2025-06-18",
            "Authorization": "Bearer valid-token",
            "Mcp-Session-Id": session_id,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert data["result"] == {}


@pytest.mark.integration
def test_oauth_tools_list_requires_session_id(
    registry: MCPToolRegistry,
    session_store: dict[str, McpSessionData],
    connection_id: UUID,
):
    """Test OAuth tools/list method requires Mcp-Session-Id header."""
    # Pre-create session
    session_id = f"session_{uuid4()}"
    session_store[session_id] = McpSessionData(
        session_id=session_id,
        oauth_client_id=None,
        connection_id=connection_id,
        last_event_id=0,
        created_at=datetime.now(),
    )

    async def session_getter(sid: str) -> McpSessionData | None:
        """Get session from store."""
        return session_store.get(sid)

    async def session_creator(oauth_client_id: UUID | None, conn_id: UUID | None) -> str:
        """Create new session."""
        new_sid = f"session_{uuid4()}"
        session_store[new_sid] = McpSessionData(
            session_id=new_sid,
            oauth_client_id=oauth_client_id,
            connection_id=conn_id,
            last_event_id=0,
            created_at=datetime.now(),
        )
        return new_sid

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        """Validate bearer token."""
        return bearer_token == "valid-token"

    app = FastAPI()
    mcp_router = create_mcp_router(
        registry,
        auth_validator=auth_validator,
        session_getter=session_getter,
        session_creator=session_creator,
    )
    app.include_router(mcp_router, prefix="/mcp")
    client = TestClient(app)

    # Test tools/list without session ID fails
    request = make_jsonrpc_request(method="tools/list")
    response = client.post(
        "/mcp",
        json=request,
        headers={
            "MCP-Protocol-Version": "2025-06-18",
            "Authorization": "Bearer valid-token",
        },
    )

    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "missing_session_id"

    # Test tools/list with valid session ID succeeds
    response = client.post(
        "/mcp",
        json=request,
        headers={
            "MCP-Protocol-Version": "2025-06-18",
            "Authorization": "Bearer valid-token",
            "Mcp-Session-Id": session_id,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "tools" in data["result"]


@pytest.mark.integration
def test_oauth_sse_without_connection_id_returns_401(
    registry: MCPToolRegistry,
    session_store: dict[str, McpSessionData],
):
    """Test OAuth SSE (GET) returns 401 when auth_validator doesn't set connection_id."""

    async def session_getter(session_id: str) -> McpSessionData | None:
        """Get session from store."""
        return session_store.get(session_id)

    async def session_creator(oauth_client_id: UUID | None, conn_id: UUID | None) -> str:
        """Create new session."""
        session_id = f"session_{uuid4()}"
        session_store[session_id] = McpSessionData(
            session_id=session_id,
            oauth_client_id=oauth_client_id,
            connection_id=conn_id,
            last_event_id=0,
            created_at=datetime.now(),
        )
        return session_id

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        """Validate bearer token but DON'T set connection_id."""
        return bearer_token == "valid-token"

    app = FastAPI()
    mcp_router = create_mcp_router(
        registry,
        auth_validator=auth_validator,
        session_getter=session_getter,
        session_creator=session_creator,
        legacy_sse=True,
    )
    app.include_router(mcp_router, prefix="/mcp")
    client = TestClient(app)

    # GET request for SSE without connection_id set
    response = client.get(
        "/mcp",
        headers={
            "Authorization": "Bearer valid-token",
        },
    )

    assert response.status_code == 401
    data = response.json()
    assert "error" in data
    assert data["error"] == "unauthorized"
    assert "identity" in data["error_description"].lower()
    # WWW-Authenticate is absent when oauth_resource_metadata is not configured (AC-9)
    assert "WWW-Authenticate" not in response.headers


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sse_stream_handles_client_disconnect():
    """Test SSE stream generator handles CancelledError gracefully.

    Verifies that an async generator with CancelledError handling (like
    the SSE event_stream in router.py) properly catches and re-raises
    CancelledError when the client disconnects.
    """
    import asyncio

    cleanup_called = False

    async def event_stream_simulation():
        """Simulate the SSE event_stream generator from router.py."""
        nonlocal cleanup_called
        yield ": SSE stream established\n\n"

        try:
            while True:
                await asyncio.sleep(30)
                yield ": keepalive\n\n"
        except asyncio.CancelledError:
            # This matches the pattern in router.py
            cleanup_called = True
            raise

    # Create generator and read first value
    gen = event_stream_simulation()
    first_value = await anext(gen)
    assert first_value == ": SSE stream established\n\n"

    # Create a task that consumes the generator
    async def consume_generator():
        async for _ in gen:
            pass

    task = asyncio.create_task(consume_generator())

    # Give the task a moment to start waiting on asyncio.sleep
    await asyncio.sleep(0.01)

    # Cancel the task (simulating client disconnect)
    task.cancel()

    # Wait for cancellation to complete
    with pytest.raises(asyncio.CancelledError):
        await task

    # Verify cleanup was called
    assert cleanup_called, "CancelledError handler should have been called"
