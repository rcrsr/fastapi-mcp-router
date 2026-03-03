"""Tests for async generator tool support (streaming tool results).

Covers AC-11, AC-12, AC-13, AC-14, AC-15, AC-85, AC-91, EC-9, EC-10, IC-11.

Generator tools return AsyncGenerator[dict, None]. In stateless mode the
registry collects all yielded dicts into a single JSON-RPC response. In
stateful mode the router drains the generator in a background task and
enqueues each dict to the session store for delivery via SSE.
"""

import json
from collections.abc import AsyncGenerator

import httpx
import pytest
from fastapi import FastAPI

from fastapi_mcp_router import InMemorySessionStore, MCPToolRegistry, create_mcp_router
from fastapi_mcp_router.registry import ToolDefinition

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_API_KEY_HEADERS = {"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"}


def _make_stateless_app(registry: MCPToolRegistry) -> FastAPI:
    """Create a minimal stateless FastAPI app with no auth."""
    app = FastAPI()
    router = create_mcp_router(registry)
    app.include_router(router, prefix="/mcp")
    return app


def _make_stateful_app(
    registry: MCPToolRegistry,
    store: InMemorySessionStore,
) -> FastAPI:
    """Create a stateful FastAPI app backed by InMemorySessionStore with API-key auth."""

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        """Accept requests with X-API-Key: test-key."""
        return api_key == "test-key"

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
    """Send initialize request and return the Mcp-Session-Id header value.

    Args:
        client: Configured httpx.AsyncClient for the stateful app.

    Returns:
        Session ID string from the Mcp-Session-Id response header.
    """
    resp = await client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "method": "initialize", "id": 1, "params": {}},
        headers=_API_KEY_HEADERS,
    )
    assert resp.status_code == 200, f"initialize failed: {resp.text}"
    session_id = resp.headers.get("mcp-session-id")
    assert session_id, "initialize response missing Mcp-Session-Id header"
    return session_id


# ---------------------------------------------------------------------------
# AC-11: AsyncGenerator[dict, None] accepted as tool return type
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_async_generator_tool_registered() -> None:
    """AC-11: Registry accepts AsyncGenerator[dict, None] return type without error."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def streaming_tool() -> AsyncGenerator[dict]:
        """Yield two result dicts."""
        yield {"chunk": 1}
        yield {"chunk": 2}

    tools = registry.list_tools()
    assert len(tools) == 1
    assert tools[0]["name"] == "streaming_tool"

    # Internal ToolDefinition marks is_generator=True
    tool_def: ToolDefinition = registry._tools["streaming_tool"]
    assert tool_def.is_generator is True


# ---------------------------------------------------------------------------
# AC-13: Stateless mode collects all dicts into single JSON-RPC response
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stateless_generator_tool_collects_all_dicts() -> None:
    """AC-13: Stateless POST tools/call returns all yielded dicts in a single response."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def multi_yield() -> AsyncGenerator[dict]:
        """Yield three result dicts."""
        yield {"step": 1, "value": "a"}
        yield {"step": 2, "value": "b"}
        yield {"step": 3, "value": "c"}

    app = _make_stateless_app(registry)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "multi_yield", "arguments": {}},
            },
            headers=_API_KEY_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body, f"Expected result key, got: {body}"
    content = body["result"]["content"]
    # Collected list is serialised as JSON in a single text content item
    collected = json.loads(content[0]["text"])
    assert collected == [
        {"step": 1, "value": "a"},
        {"step": 2, "value": "b"},
        {"step": 3, "value": "c"},
    ]


# ---------------------------------------------------------------------------
# AC-14: dict-returning tools unaffected
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dict_returning_tool_unchanged() -> None:
    """AC-14: Plain dict-returning tool returns its dict unchanged alongside generator tool."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def plain_tool() -> dict:
        """Return a simple dict."""
        return {"status": "ok"}

    @registry.tool()
    async def gen_tool() -> AsyncGenerator[dict]:
        """Yield a single dict."""
        yield {"streamed": True}

    app = _make_stateless_app(registry)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp_plain = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "plain_tool", "arguments": {}},
            },
            headers=_API_KEY_HEADERS,
        )
        resp_gen = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "gen_tool", "arguments": {}},
            },
            headers=_API_KEY_HEADERS,
        )

    # Plain dict tool
    assert resp_plain.status_code == 200
    plain_body = resp_plain.json()
    assert "result" in plain_body
    plain_text = plain_body["result"]["content"][0]["text"]
    assert json.loads(plain_text) == {"status": "ok"}

    # Generator tool still works
    assert resp_gen.status_code == 200
    gen_body = resp_gen.json()
    assert "result" in gen_body
    gen_collected = json.loads(gen_body["result"]["content"][0]["text"])
    assert gen_collected == [{"streamed": True}]


# ---------------------------------------------------------------------------
# AC-91: Empty generator yields empty content list
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_empty_generator_returns_empty_content() -> None:
    """AC-91: Generator that yields nothing produces content list with empty JSON array."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def empty_gen() -> AsyncGenerator[dict]:
        """Yield nothing."""
        return
        yield  # make it an async generator

    app = _make_stateless_app(registry)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "empty_gen", "arguments": {}},
            },
            headers=_API_KEY_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body
    assert "isError" not in body.get("result", {})
    collected = json.loads(body["result"]["content"][0]["text"])
    assert collected == []


# ---------------------------------------------------------------------------
# AC-15 / AC-85: Generator exception → ToolError (isError: true)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generator_exception_produces_tool_error() -> None:
    """AC-15/AC-85: Generator that raises ValueError produces isError: true in response."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def failing_gen() -> AsyncGenerator[dict]:
        """Raise mid-stream."""
        yield {"ok": True}
        raise ValueError("mid-stream failure")

    app = _make_stateless_app(registry)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "failing_gen", "arguments": {}},
            },
            headers=_API_KEY_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body
    assert body["result"].get("isError") is True
    error_text = body["result"]["content"][0]["text"]
    assert "mid-stream failure" in error_text


# ---------------------------------------------------------------------------
# EC-9: Generator yields non-dict → ToolError (isError: true)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generator_non_dict_yield_produces_tool_error() -> None:
    """EC-9: Generator yielding a non-dict value produces isError: true."""
    registry = MCPToolRegistry()

    # Register with a broad type hint to bypass annotation check;
    # the registry enforces non-dict at runtime during collection.
    @registry.tool(input_schema={"type": "object", "properties": {}, "required": []})
    async def bad_yield_gen() -> AsyncGenerator[dict]:  # type: ignore[return]
        """Yield a string instead of a dict."""
        yield "not a dict"  # type: ignore[misc]

    app = _make_stateless_app(registry)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "bad_yield_gen", "arguments": {}},
            },
            headers=_API_KEY_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body
    assert body["result"].get("isError") is True
    error_text = body["result"]["content"][0]["text"]
    assert "non-dict" in error_text.lower() or "str" in error_text.lower()


# ---------------------------------------------------------------------------
# EC-10: Generator raises exception → ToolError (isError: true)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generator_raises_produces_tool_error() -> None:
    """EC-10: Generator that raises RuntimeError produces isError: true via stateless path."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def error_gen() -> AsyncGenerator[dict]:
        """Raise immediately without yielding."""
        raise RuntimeError("generator error")
        yield  # make it an async generator

    app = _make_stateless_app(registry)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "error_gen", "arguments": {}},
            },
            headers=_API_KEY_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body
    assert body["result"].get("isError") is True
    error_text = body["result"]["content"][0]["text"]
    assert "generator error" in error_text


# ---------------------------------------------------------------------------
# AC-12: Stateful mode — dicts enqueued to session store
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stateful_generator_collects_dicts_in_response() -> None:
    """IR-3: Stateful POST tools/call collects all yielded dicts into the POST response body."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def stream_data() -> AsyncGenerator[dict]:
        """Yield two event dicts."""
        yield {"event": "start", "index": 0}
        yield {"event": "end", "index": 1}

    store = InMemorySessionStore()
    app = _make_stateful_app(registry, store)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Initialize session to get session ID
        session_id = await _initialize_session(client)

        # Call the generator tool with the active session
        resp = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "stream_data", "arguments": {}},
            },
            headers={**_API_KEY_HEADERS, "Mcp-Session-Id": session_id},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body
    # IR-3: POST response contains all yielded dicts as a list
    content = body["result"]["content"]
    assert len(content) == 2
    assert content[0] == {"event": "start", "index": 0}
    assert content[1] == {"event": "end", "index": 1}


# ---------------------------------------------------------------------------
# AC-12 supplemental: stateful generator exception enqueues isError message
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stateful_generator_exception_returns_is_error_response() -> None:
    """EC-4: Stateful generator raising mid-iteration returns isError: true in POST response."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def failing_stream() -> AsyncGenerator[dict]:
        """Yield one dict then raise."""
        yield {"partial": True}
        raise RuntimeError("stream crashed")

    store = InMemorySessionStore()
    app = _make_stateful_app(registry, store)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _initialize_session(client)

        resp = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "failing_stream", "arguments": {}},
            },
            headers={**_API_KEY_HEADERS, "Mcp-Session-Id": session_id},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body
    result = body["result"]
    # EC-4: generator raised mid-iteration — POST response contains isError: true
    assert result.get("isError") is True
    assert "stream crashed" in result["content"][0]["text"]
