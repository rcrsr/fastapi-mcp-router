"""Tests for RootsManager and the roots/list MCP method.

Covers AC-56: roots/list returns all registered roots.
"""

import httpx
import pytest
from fastapi import FastAPI

from fastapi_mcp_router import MCPToolRegistry, create_mcp_router
from fastapi_mcp_router.session import RootsManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MCP_HEADERS = {"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"}


def _build_app(roots_manager: RootsManager) -> FastAPI:
    """Create a minimal stateless FastAPI app with the given RootsManager.

    Args:
        roots_manager: Pre-populated RootsManager to inject into the router.

    Returns:
        FastAPI app with MCP router mounted at /mcp.
    """
    registry = MCPToolRegistry()
    app = FastAPI()
    router = create_mcp_router(registry, roots_manager=roots_manager)
    app.include_router(router, prefix="/mcp")
    return app


def _rpc(method: str, params: dict | None = None, rpc_id: int = 1) -> dict:
    """Build a minimal JSON-RPC 2.0 request body.

    Args:
        method: JSON-RPC method name.
        params: Optional parameters dict.
        rpc_id: Request identifier.

    Returns:
        JSON-RPC 2.0 request dict.
    """
    body: dict = {"jsonrpc": "2.0", "id": rpc_id, "method": method}
    if params is not None:
        body["params"] = params
    return body


# ---------------------------------------------------------------------------
# Unit tests for RootsManager
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_roots_manager_empty_on_init() -> None:
    """RootsManager starts with no roots registered."""
    manager = RootsManager()

    assert manager.list_roots() == []


@pytest.mark.unit
def test_roots_manager_add_root_with_name() -> None:
    """add_root stores uri and name when both provided."""
    manager = RootsManager()

    manager.add_root(uri="file:///workspace", name="Workspace")
    roots = manager.list_roots()

    assert len(roots) == 1
    assert roots[0]["uri"] == "file:///workspace"
    assert roots[0]["name"] == "Workspace"


@pytest.mark.unit
def test_roots_manager_add_root_without_name() -> None:
    """add_root stores only uri when name is omitted."""
    manager = RootsManager()

    manager.add_root(uri="file:///tmp")
    roots = manager.list_roots()

    assert len(roots) == 1
    assert roots[0]["uri"] == "file:///tmp"
    assert "name" not in roots[0]


@pytest.mark.unit
def test_roots_manager_list_roots_returns_all() -> None:
    """list_roots returns every registered root in insertion order."""
    manager = RootsManager()

    manager.add_root(uri="file:///a", name="Alpha")
    manager.add_root(uri="file:///b", name="Beta")
    manager.add_root(uri="file:///c")

    roots = manager.list_roots()

    assert len(roots) == 3
    assert roots[0]["uri"] == "file:///a"
    assert roots[1]["uri"] == "file:///b"
    assert roots[2]["uri"] == "file:///c"


@pytest.mark.unit
def test_roots_manager_list_roots_returns_copy() -> None:
    """list_roots returns a copy; mutating the result does not affect the manager."""
    manager = RootsManager()
    manager.add_root(uri="file:///x")

    result = manager.list_roots()
    result.clear()

    assert len(manager.list_roots()) == 1


# ---------------------------------------------------------------------------
# AC-56: roots/list returns all registered roots
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_roots_list_empty() -> None:
    """AC-56: roots/list returns empty list when no roots are registered."""
    manager = RootsManager()
    app = _build_app(manager)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=_rpc("roots/list"),
            headers=_MCP_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body
    assert body["result"]["roots"] == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_roots_list_with_roots() -> None:
    """AC-56: roots/list returns all registered roots."""
    manager = RootsManager()
    manager.add_root(uri="file:///workspace", name="Workspace")
    manager.add_root(uri="file:///data", name="Data")

    app = _build_app(manager)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=_rpc("roots/list"),
            headers=_MCP_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body
    roots = body["result"]["roots"]
    assert len(roots) == 2
    uris = [r["uri"] for r in roots]
    assert "file:///workspace" in uris
    assert "file:///data" in uris


@pytest.mark.integration
@pytest.mark.asyncio
async def test_roots_list_uri_and_name() -> None:
    """AC-56: roots/list includes uri and name fields when both registered."""
    manager = RootsManager()
    manager.add_root(uri="file:///project", name="My Project")

    app = _build_app(manager)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=_rpc("roots/list"),
            headers=_MCP_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    root = body["result"]["roots"][0]
    assert root["uri"] == "file:///project"
    assert root["name"] == "My Project"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_roots_list_uri_only() -> None:
    """AC-56: roots/list includes root with name field null when no name registered."""
    manager = RootsManager()
    manager.add_root(uri="file:///no-name")

    app = _build_app(manager)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=_rpc("roots/list"),
            headers=_MCP_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    roots = body["result"]["roots"]
    assert len(roots) == 1
    root = roots[0]
    assert root["uri"] == "file:///no-name"
    # name field is present but null per router dispatch: r.get("name") returns None
    assert "name" in root
    assert root["name"] is None
