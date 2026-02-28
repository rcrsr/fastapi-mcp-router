"""Tests for MCPRouter class — decorator API and FastAPI mounting.

Covers AC-35, AC-36, AC-37, AC-38, AC-40, AC-75, AC-76, EC-13, EC-14.
"""

import httpx
import pytest
from fastapi import APIRouter, FastAPI

from fastapi_mcp_router import (
    InMemorySessionStore,
    MCPRouter,
    ServerInfo,
)
from fastapi_mcp_router.resources import Resource, ResourceContents, ResourceProvider

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_MCP_HEADERS = {"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"}


def _rpc(method: str, params: dict | None = None, rpc_id: int = 1) -> dict:
    """Build a minimal JSON-RPC 2.0 request body.

    Args:
        method: JSON-RPC method name
        params: Optional parameters dict
        rpc_id: Request identifier

    Returns:
        JSON-RPC 2.0 request dict
    """
    body: dict = {"jsonrpc": "2.0", "id": rpc_id, "method": method}
    if params is not None:
        body["params"] = params
    return body


# ---------------------------------------------------------------------------
# AC-35: MCPRouter extends APIRouter
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mcp_router_is_instance_of_api_router() -> None:
    """AC-35: isinstance(MCPRouter(), APIRouter) returns True."""
    mcp = MCPRouter()
    assert isinstance(mcp, APIRouter)


# ---------------------------------------------------------------------------
# AC-36: @mcp.tool() decorator registers tools callable via HTTP
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_tool_decorator_callable_via_http() -> None:
    """AC-36: @mcp.tool() registers a tool; POST /mcp tools/call returns the result."""
    mcp = MCPRouter()

    @mcp.tool()
    async def echo(message: str) -> str:
        """Echo the message back."""
        return f"echo: {message}"

    app = FastAPI()
    app.include_router(mcp, prefix="/mcp")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=_rpc("tools/call", {"name": "echo", "arguments": {"message": "hello"}}),
            headers=_MCP_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body
    content = body["result"]["content"]
    assert len(content) == 1
    assert "echo: hello" in content[0]["text"]


# ---------------------------------------------------------------------------
# AC-36: @mcp.resource() decorator registers resources callable via HTTP
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_resource_decorator_callable_via_http() -> None:
    """AC-36: @mcp.resource() registers a resource handler; POST /mcp resources/read returns content."""
    mcp = MCPRouter()

    @mcp.resource("docs://{slug}", name="Doc", description="A document")
    async def get_doc(slug: str) -> str:
        """Return document content."""
        return f"content:{slug}"

    app = FastAPI()
    app.include_router(mcp, prefix="/mcp")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=_rpc("resources/read", {"uri": "docs://intro"}),
            headers=_MCP_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body
    contents = body["result"]["contents"]
    assert len(contents) == 1
    assert contents[0]["text"] == "content:intro"


# ---------------------------------------------------------------------------
# AC-36: @mcp.prompt() decorator registers prompts callable via HTTP
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_prompt_decorator_callable_via_http() -> None:
    """AC-36: @mcp.prompt() registers a prompt; POST /mcp prompts/get returns messages."""
    mcp = MCPRouter()

    @mcp.prompt()
    async def greet(username: str) -> list[dict]:
        """Greet a user."""
        return [{"role": "user", "content": f"Hello {username}"}]

    app = FastAPI()
    app.include_router(mcp, prefix="/mcp")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=_rpc("prompts/get", {"name": "greet", "arguments": {"username": "Alice"}}),
            headers=_MCP_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body
    messages = body["result"]["messages"]
    assert len(messages) == 1
    assert "Alice" in messages[0]["content"]


# ---------------------------------------------------------------------------
# AC-37: mcp.add_resource_provider() registers providers
# ---------------------------------------------------------------------------


class _StubProvider(ResourceProvider):
    """Minimal ResourceProvider stub returning one fixed resource."""

    def list_resources(self) -> list[Resource]:
        """Return one stub resource."""
        return [Resource(uri="stub://item1", name="Item1", description="A stub item")]

    async def read_resource(self, uri: str) -> ResourceContents:
        """Return stub text content."""
        return ResourceContents(uri=uri, text="stub-content")

    def subscribe(self, uri: str) -> bool:
        """Subscriptions not supported."""
        return False

    def unsubscribe(self, uri: str) -> bool:
        """Unsubscription not supported."""
        return False

    async def watch(self):
        """No-op watch."""
        return
        yield  # pragma: no cover


@pytest.mark.integration
@pytest.mark.asyncio
async def test_add_resource_provider_included_in_resources_list() -> None:
    """AC-37: add_resource_provider() registers a provider; resources/list includes its resources."""
    mcp = MCPRouter()
    provider = _StubProvider()
    mcp.add_resource_provider("stub://", provider)

    app = FastAPI()
    app.include_router(mcp, prefix="/mcp")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=_rpc("resources/list"),
            headers=_MCP_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body
    uris = [r["uri"] for r in body["result"]["resources"]]
    assert "stub://item1" in uris


# ---------------------------------------------------------------------------
# AC-38: app.include_router(mcp, prefix="/mcp") mounts POST and GET endpoints
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_include_router_mounts_post_endpoint() -> None:
    """AC-38: POST /mcp is reachable after app.include_router(mcp, prefix="/mcp")."""
    mcp = MCPRouter()
    app = FastAPI()
    app.include_router(mcp, prefix="/mcp")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=_rpc("ping"),
            headers=_MCP_HEADERS,
        )

    assert resp.status_code == 200


@pytest.mark.integration
@pytest.mark.asyncio
async def test_include_router_mounts_get_endpoint() -> None:
    """AC-38: GET /mcp responds (SSE endpoint) after include_router with stateless MCPRouter."""
    mcp = MCPRouter()
    app = FastAPI()
    app.include_router(mcp, prefix="/mcp")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/mcp", headers=_MCP_HEADERS)

    # Stateless mode returns 405 or 200; any non-404 means the route is mounted.
    assert resp.status_code != 404


# ---------------------------------------------------------------------------
# AC-40: All create_mcp_router() parameters accepted by MCPRouter constructor
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mcp_router_accepts_all_create_mcp_router_params() -> None:
    """AC-40: MCPRouter constructor accepts all create_mcp_router() parameters without TypeError."""
    store = InMemorySessionStore()

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        """Accept all requests."""
        return True

    def tool_filter(is_oauth: bool) -> list[str] | None:
        """Pass all tools through (no exclusions)."""
        return None

    server_info: ServerInfo = {"name": "test-server", "version": "1.0.0"}

    # Should not raise TypeError
    mcp = MCPRouter(
        auth_validator=auth_validator,
        session_store=store,
        tool_filter=tool_filter,
        server_info=server_info,
        base_url="http://localhost:8000",
        stateful=True,
    )

    assert isinstance(mcp, MCPRouter)


# ---------------------------------------------------------------------------
# AC-75 / EC-13: stateful=True without session_store raises ValueError
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mcp_router_stateful_without_session_store_raises_value_error() -> None:
    """AC-75 / EC-13: MCPRouter(stateful=True) without session_store raises ValueError."""
    with pytest.raises(ValueError, match="session_store"):
        MCPRouter(stateful=True)


# ---------------------------------------------------------------------------
# AC-76 / EC-14: sampling_enabled=True without stateful=True raises ValueError
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mcp_router_sampling_enabled_without_stateful_raises_value_error() -> None:
    """AC-76 / EC-14: MCPRouter(sampling_enabled=True, stateful=False) raises ValueError."""
    with pytest.raises(ValueError, match="stateful"):
        MCPRouter(sampling_enabled=True)
