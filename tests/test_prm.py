"""Integration and unit tests for OAuth Protected Resource Metadata (PRM) endpoint.

Tests AC-7, AC-8, AC-9, AC-10, AC-73, AC-74, EC-11, EC-12.
Covers PRM endpoint correctness, WWW-Authenticate header injection, and
ValueError guards on create_prm_router() and create_mcp_router().
"""

import httpx
import pytest
from fastapi import FastAPI

from fastapi_mcp_router import MCPToolRegistry, create_mcp_router, create_prm_router

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(name="registry")
def registry_fixture() -> MCPToolRegistry:
    """Create tool registry with a single test tool.

    Returns:
        MCPToolRegistry with a ping tool registered.
    """
    reg = MCPToolRegistry()

    @reg.tool()
    async def ping() -> str:
        """Simple ping tool."""
        return "pong"

    return reg


# ---------------------------------------------------------------------------
# AC-7 / AC-8: PRM endpoint returns valid RFC 9728 JSON matching configured metadata
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_prm_endpoint_returns_rfc9728_json() -> None:
    """AC-7: GET /.well-known/oauth-protected-resource returns 200 with JSON body."""
    metadata: dict[str, object] = {
        "resource": "https://api.example.io/mcp",
        "authorization_servers": ["https://auth.example.io"],
    }
    app = FastAPI()
    prm_router = create_prm_router(metadata)
    app.include_router(prm_router)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/.well-known/oauth-protected-resource")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert "resource" in body
    assert "authorization_servers" in body


@pytest.mark.integration
@pytest.mark.asyncio
async def test_prm_fields_match_configured_metadata() -> None:
    """AC-8: PRM response fields exactly match the oauth_resource_metadata dict."""
    metadata: dict[str, object] = {
        "resource": "https://api.example.io/mcp",
        "authorization_servers": ["https://auth.example.io"],
        "scopes_supported": ["mcp:read", "mcp:evaluate"],
    }
    app = FastAPI()
    prm_router = create_prm_router(metadata)
    app.include_router(prm_router)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/.well-known/oauth-protected-resource")

    assert response.status_code == 200
    body = response.json()
    assert body["resource"] == "https://api.example.io/mcp"
    assert body["authorization_servers"] == ["https://auth.example.io"]
    assert body["scopes_supported"] == ["mcp:read", "mcp:evaluate"]


# ---------------------------------------------------------------------------
# AC-9: 401 responses include WWW-Authenticate with PRM URL
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_401_includes_www_authenticate_with_prm_url(registry: MCPToolRegistry) -> None:
    """AC-9: Unauthenticated POST /mcp returns 401 with WWW-Authenticate pointing to PRM URL."""
    metadata: dict[str, object] = {
        "resource": "https://api.example.io/mcp",
        "authorization_servers": ["https://auth.example.io"],
    }

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        """Reject all requests to force 401."""
        return False

    app = FastAPI()
    mcp_router = create_mcp_router(
        registry,
        auth_validator=auth_validator,
        base_url="https://api.example.io",
        oauth_resource_metadata=metadata,
    )
    app.include_router(mcp_router, prefix="/mcp")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "ping", "id": 1},
            headers={"MCP-Protocol-Version": "2025-06-18"},
        )

    assert response.status_code == 401
    www_auth = response.headers.get("www-authenticate", "")
    assert "Bearer" in www_auth
    assert "resource_metadata=" in www_auth
    assert "/.well-known/oauth-protected-resource" in www_auth


# ---------------------------------------------------------------------------
# AC-10: No oauth_resource_metadata → no PRM endpoint registered
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_no_prm_endpoint_when_not_configured(registry: MCPToolRegistry) -> None:
    """AC-10: Without oauth_resource_metadata, PRM endpoint does not exist (returns 404)."""
    app = FastAPI()
    mcp_router = create_mcp_router(registry)
    app.include_router(mcp_router, prefix="/mcp")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/.well-known/oauth-protected-resource")

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# AC-73 / EC-12: Missing required keys raise ValueError
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_missing_resource_raises_value_error() -> None:
    """AC-73 / EC-12: create_prm_router raises ValueError when 'resource' key is absent."""
    with pytest.raises(ValueError, match="resource"):
        create_prm_router({"authorization_servers": ["https://auth.example.io"]})


@pytest.mark.unit
def test_missing_authorization_servers_raises_value_error() -> None:
    """EC-12: create_prm_router raises ValueError when 'authorization_servers' key is absent."""
    with pytest.raises(ValueError, match="authorization_servers"):
        create_prm_router({"resource": "https://api.example.io/mcp"})


# ---------------------------------------------------------------------------
# AC-74 / EC-11: Both session_store and session_getter raises ValueError
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_both_session_store_and_getter_raises_value_error() -> None:
    """AC-74 / EC-11: create_mcp_router raises ValueError when both session_store and session_getter are given."""
    registry = MCPToolRegistry()

    async def session_getter(session_id: str) -> None:
        """Stub session getter."""
        return None

    with pytest.raises(ValueError, match="session_store"):
        create_mcp_router(
            registry,
            session_store=object(),
            session_getter=session_getter,
        )
