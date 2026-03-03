"""Integration tests for stateful POST generator collection constraint.

Covers AC-7, AC-8, AC-9, EC-4 from the Phase 2 specification (IR-3).

In stateful POST mode (session_store provided + Mcp-Session-Id header),
generator tools must have all yielded items collected into the POST response
body as a JSON array. Non-generator tools follow the existing code path.
"""

import json
from collections.abc import AsyncGenerator

import httpx
import pytest
from fastapi import FastAPI

from fastapi_mcp_router import InMemorySessionStore, MCPToolRegistry, create_mcp_router

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HEADERS = {"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"}


def _make_stateful_app(registry: MCPToolRegistry) -> FastAPI:
    """Create a minimal stateful FastAPI app backed by InMemorySessionStore.

    Args:
        registry: Tool registry containing registered tool handlers.

    Returns:
        FastAPI application with MCP router mounted at /mcp.
    """

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        """Accept requests with X-API-Key: test-key."""
        return api_key == "test-key"

    store = InMemorySessionStore()
    app = FastAPI()
    router = create_mcp_router(
        registry,
        session_store=store,
        stateful=True,
        auth_validator=auth_validator,
    )
    app.include_router(router, prefix="/mcp")
    return app


async def _initialize_session(client: httpx.AsyncClient) -> str:
    """Send MCP initialize and return the Mcp-Session-Id header value.

    Args:
        client: AsyncClient configured for the stateful app.

    Returns:
        Session ID string from the Mcp-Session-Id response header.
    """
    resp = await client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "method": "initialize", "id": 1, "params": {}},
        headers=_HEADERS,
    )
    assert resp.status_code == 200, f"initialize failed: {resp.text}"
    session_id = resp.headers.get("mcp-session-id")
    assert session_id, "initialize response missing Mcp-Session-Id header"
    return session_id


async def _call_tool(
    client: httpx.AsyncClient,
    session_id: str,
    tool_name: str,
    arguments: dict,
) -> dict:
    """Send a tools/call request in stateful mode and return the parsed JSON body.

    Args:
        client: AsyncClient configured for the stateful app.
        session_id: Active session identifier from initialize.
        tool_name: Name of the tool to invoke.
        arguments: Tool argument dict.

    Returns:
        Parsed JSON response body as a dict.
    """
    resp = await client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        },
        headers={**_HEADERS, "Mcp-Session-Id": session_id},
    )
    assert resp.status_code == 200, f"tools/call failed: {resp.text}"
    return resp.json()


# ---------------------------------------------------------------------------
# AC-7: generator tool stateful POST 3 items → 3-item list
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stateful_post_generator_three_items_collected() -> None:
    """AC-7: Generator yielding 3 TextContent dicts in stateful POST → 3-item list in response."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def three_items() -> AsyncGenerator[dict]:
        """Yield three content dicts."""
        yield {"type": "text", "text": "first"}
        yield {"type": "text", "text": "second"}
        yield {"type": "text", "text": "third"}

    app = _make_stateful_app(registry)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _initialize_session(client)
        body = await _call_tool(client, session_id, "three_items", {})

    assert "result" in body, f"Expected result key, got: {body}"
    content = body["result"]["content"]
    assert len(content) == 3
    assert content[0] == {"type": "text", "text": "first"}
    assert content[1] == {"type": "text", "text": "second"}
    assert content[2] == {"type": "text", "text": "third"}


# ---------------------------------------------------------------------------
# AC-8: generator tool stateful POST 0 items → empty list
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stateful_post_generator_zero_items_returns_empty_list() -> None:
    """AC-8: Generator yielding 0 items in stateful POST → response body content == []."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def empty_gen() -> AsyncGenerator[dict]:
        """Yield nothing."""
        return
        yield  # make this an async generator  # noqa: unreachable

    app = _make_stateful_app(registry)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _initialize_session(client)
        body = await _call_tool(client, session_id, "empty_gen", {})

    assert "result" in body, f"Expected result key, got: {body}"
    result = body["result"]
    assert "isError" not in result, f"Unexpected isError in result: {result}"
    assert result["content"] == []


# ---------------------------------------------------------------------------
# AC-9: plain-value tool stateful POST → unchanged behavior
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stateful_post_plain_value_tool_unchanged() -> None:
    """AC-9: Non-generator tool in stateful POST returns existing text-content format."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def plain_tool(value: str) -> dict:
        """Return a simple dict."""
        return {"echo": value}

    app = _make_stateful_app(registry)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _initialize_session(client)
        body = await _call_tool(client, session_id, "plain_tool", {"value": "hello"})

    assert "result" in body, f"Expected result key, got: {body}"
    result = body["result"]
    assert "isError" not in result, f"Unexpected isError in result: {result}"
    content = result["content"]
    assert len(content) == 1
    assert content[0]["type"] == "text"
    assert json.loads(content[0]["text"]) == {"echo": "hello"}


# ---------------------------------------------------------------------------
# EC-4: generator raises mid-iteration → isError: true in POST response
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stateful_post_generator_raises_mid_iteration_returns_is_error() -> None:
    """EC-4: Generator raising mid-iteration in stateful POST → isError: true response."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def crashing_gen() -> AsyncGenerator[dict]:
        """Yield one item then raise."""
        yield {"partial": True}
        raise ValueError("mid-iteration crash")

    app = _make_stateful_app(registry)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _initialize_session(client)
        body = await _call_tool(client, session_id, "crashing_gen", {})

    assert "result" in body, f"Expected result key, got: {body}"
    result = body["result"]
    assert result.get("isError") is True, f"Expected isError: true, got: {result}"
    error_text = result["content"][0]["text"]
    assert "mid-iteration crash" in error_text
