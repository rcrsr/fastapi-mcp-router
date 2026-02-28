"""Backwards compatibility tests for fastapi-mcp-router public API.

Verifies that MCPToolRegistry + create_mcp_router() API is unchanged after
the MCPRouter refactor. Covers AC-39: all 16 exports remain in __all__,
existing tool call flows work, and conftest session_client fixture pattern
functions correctly.
"""

from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from uuid import UUID, uuid4

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI, Request

import fastapi_mcp_router
from fastapi_mcp_router import MCPToolRegistry, create_mcp_router
from fastapi_mcp_router.types import McpSessionData

# ---------------------------------------------------------------------------
# AC-39: all expected exports present in __all__
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_all_exports_present_in_dunder_all():
    """Test all 16 expected public names exist in fastapi_mcp_router.__all__.

    AC-39: no symbols removed or renamed by MCPRouter refactor.
    """
    expected = {
        "EventSubscriber",
        "InMemorySessionStore",
        "MCPError",
        "MCPRouter",
        "MCPToolRegistry",
        "PromptRegistry",
        "ResourceRegistry",
        "ServerIcon",
        "ServerInfo",
        "SessionStore",
        "TextContent",
        "ToolError",
        "ToolFilter",
        "ToolResponse",
        "create_mcp_router",
        "create_prm_router",
    }

    actual = set(fastapi_mcp_router.__all__)

    missing = expected - actual
    assert not missing, f"Missing from __all__: {missing}"


@pytest.mark.unit
def test_all_exports_importable_from_top_level():
    """Test every name in __all__ is importable from the top-level package.

    AC-39: each symbol resolves without AttributeError, proving the export
    is backed by a real object and not a stub entry in __all__.
    """
    for name in fastapi_mcp_router.__all__:
        assert hasattr(fastapi_mcp_router, name), f"{name!r} in __all__ but not importable"


# ---------------------------------------------------------------------------
# AC-39: MCPToolRegistry + create_mcp_router() end-to-end tool call
# ---------------------------------------------------------------------------


@pytest.fixture(name="compat_registry")
def compat_registry_fixture() -> MCPToolRegistry:
    """Create a registry with one echo tool for backwards-compat tests.

    Returns:
        MCPToolRegistry with 'echo' tool registered.
    """
    registry = MCPToolRegistry()

    @registry.tool()
    async def echo(message: str) -> str:
        """Echo the message back."""
        return message

    return registry


@pytest_asyncio.fixture(name="compat_client")
async def compat_client_fixture(
    compat_registry: MCPToolRegistry,
) -> AsyncGenerator[httpx.AsyncClient]:
    """Create an AsyncClient for the stateless MCPToolRegistry + create_mcp_router() app.

    Replicates the conftest 'client' fixture pattern using the legacy API
    to confirm it still works after the MCPRouter refactor.

    Args:
        compat_registry: Registry with echo tool provided by compat_registry_fixture.

    Yields:
        httpx.AsyncClient configured with ASGITransport and base_url http://test.
    """
    fastapi_app = FastAPI()
    mcp_router = create_mcp_router(compat_registry)
    fastapi_app.include_router(mcp_router, prefix="/mcp")
    transport = httpx.ASGITransport(app=fastapi_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcptoolregistry_create_mcp_router_tool_call_succeeds(
    compat_client: httpx.AsyncClient,
) -> None:
    """Test MCPToolRegistry + create_mcp_router() tool call returns expected result.

    AC-39: the full legacy path (register tool, build router, call via HTTP)
    works identically after the MCPRouter refactor.
    """
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "echo",
            "arguments": {"message": "hello backwards compat"},
        },
    }

    response = await compat_client.post(
        "/mcp",
        json=payload,
        headers={
            "Content-Type": "application/json",
            "MCP-Protocol-Version": "2025-06-18",
            "X-API-Key": "test-key",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["jsonrpc"] == "2.0"
    assert body["id"] == 1
    result = body["result"]
    content = result["content"]
    assert len(content) == 1
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "hello backwards compat"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcptoolregistry_create_mcp_router_tools_list_unchanged(
    compat_client: httpx.AsyncClient,
) -> None:
    """Test tools/list returns the registered echo tool via the legacy API.

    AC-39: tools/list response format is unchanged after refactor.
    """
    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {},
    }

    response = await compat_client.post(
        "/mcp",
        json=payload,
        headers={
            "Content-Type": "application/json",
            "MCP-Protocol-Version": "2025-06-18",
            "X-API-Key": "test-key",
        },
    )

    assert response.status_code == 200
    body = response.json()
    tools = body["result"]["tools"]
    names = [t["name"] for t in tools]
    assert "echo" in names


# ---------------------------------------------------------------------------
# AC-39: conftest session_client fixture pattern works post-refactor
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(name="session_client")
async def session_client_fixture() -> AsyncGenerator[httpx.AsyncClient]:
    """Replicate the conftest session_client fixture using the legacy create_mcp_router() API.

    AC-39: existing test patterns that create session_client fixtures via
    MCPToolRegistry + create_mcp_router() still function after the refactor.
    The Bearer auth + session callbacks path exercises the stateful mode.

    Yields:
        httpx.AsyncClient configured with ASGITransport and base_url http://test.
    """
    registry = MCPToolRegistry()

    @registry.tool()
    async def ping_tool() -> str:
        """Return pong."""
        return "pong"

    session_store: dict[str, McpSessionData] = {}

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        """Accept all Bearer token requests."""
        return bearer_token is not None

    async def session_getter(session_id: str) -> McpSessionData | None:
        """Return session data or None when absent.

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
        registry,
        auth_validator=auth_validator,
        session_getter=session_getter,
        session_creator=session_creator,
    )
    fastapi_app.include_router(mcp_router, prefix="/mcp")

    transport = httpx.ASGITransport(app=fastapi_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client


@pytest.mark.integration
@pytest.mark.asyncio
async def test_session_client_fixture_pattern_tool_call_succeeds(
    session_client: httpx.AsyncClient,
) -> None:
    """Test the conftest session_client fixture pattern works post-refactor.

    AC-39: existing integration tests that inject session_client and POST
    tools/call with Bearer auth still receive a successful JSON-RPC response.
    The stateful flow requires initialize first to obtain a session ID, then
    subsequent requests include the Mcp-Session-Id header.
    """
    bearer_headers = {
        "Content-Type": "application/json",
        "MCP-Protocol-Version": "2025-06-18",
        "Authorization": "Bearer test-token",
    }

    # Step 1: initialize — creates a session and returns Mcp-Session-Id header.
    init_response = await session_client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"},
            },
        },
        headers=bearer_headers,
    )
    assert init_response.status_code == 200
    session_id = init_response.headers.get("mcp-session-id")
    assert session_id is not None, "initialize must return Mcp-Session-Id header"

    # Step 2: tools/call with the session ID from initialize.
    call_response = await session_client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "ping_tool",
                "arguments": {},
            },
        },
        headers={**bearer_headers, "Mcp-Session-Id": session_id},
    )

    assert call_response.status_code == 200
    body = call_response.json()
    assert body["jsonrpc"] == "2.0"
    assert body["id"] == 2
    result = body["result"]
    content = result["content"]
    assert len(content) == 1
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "pong"
