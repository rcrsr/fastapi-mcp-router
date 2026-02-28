"""Tests for tool output schema support (IC-42).

Covers AC-67 through AC-70:
- AC-67: output_schema parameter accepted by @tool() decorator
- AC-68: outputSchema included in tools/list response
- AC-69: structuredContent returned when outputSchema present
- AC-70: text content returned (no structuredContent) when no output_schema
"""

from collections.abc import AsyncGenerator

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from fastapi_mcp_router import MCPToolRegistry, create_mcp_router
from fastapi_mcp_router.registry import ToolDefinition  # internal symbol — submodule import per §LIB.2.2

_HEADERS = {"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"}

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(name="output_schema_client")
async def output_schema_client_fixture() -> AsyncGenerator[httpx.AsyncClient]:
    """Create an AsyncClient for an app with both a schema tool and a plain tool.

    Yields:
        httpx.AsyncClient configured with ASGITransport and base_url http://test.
    """
    registry = MCPToolRegistry()

    @registry.tool(output_schema={"type": "object", "properties": {"score": {"type": "number"}}})
    async def analyze(text: str) -> dict:
        """Analyze text and return a score."""
        return {"score": 0.95}

    @registry.tool()
    async def plain(message: str) -> str:
        """Echo back a message."""
        return message

    fastapi_app = FastAPI()
    mcp_router = create_mcp_router(registry)
    fastapi_app.include_router(mcp_router, prefix="/mcp")

    transport = httpx.ASGITransport(app=fastapi_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client


# ---------------------------------------------------------------------------
# AC-67: @tool(output_schema={...}) stores schema in ToolDefinition
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_output_schema_stored_in_tool_definition() -> None:
    """AC-67: output_schema parameter stored in ToolDefinition without error.

    Verifies the decorator accepts output_schema and the ToolDefinition
    reflects the value exactly as passed.
    """
    registry = MCPToolRegistry()
    schema = {"type": "object", "properties": {"score": {"type": "number"}}}

    @registry.tool(output_schema=schema)
    async def analyze(text: str) -> dict:
        """Analyze text."""
        return {"score": 0.9}

    # Access internal ToolDefinition (internal symbol — submodule import per §LIB.2.2)
    tool_def: ToolDefinition = registry._tools["analyze"]
    assert tool_def.output_schema == schema


@pytest.mark.unit
def test_no_output_schema_stores_none() -> None:
    """AC-67 complement: omitting output_schema leaves ToolDefinition.output_schema as None."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def plain(message: str) -> str:
        """Plain tool."""
        return message

    tool_def: ToolDefinition = registry._tools["plain"]
    assert tool_def.output_schema is None


# ---------------------------------------------------------------------------
# AC-68: outputSchema included in tools/list response
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tools_list_includes_output_schema(
    output_schema_client: httpx.AsyncClient,
) -> None:
    """AC-68: tools/list includes outputSchema key for tools that set output_schema."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {},
    }

    response = await output_schema_client.post(
        "/mcp",
        json=payload,
        headers=_HEADERS,
    )

    assert response.status_code == 200
    body = response.json()
    tools = body["result"]["tools"]
    analyze_tool = next(t for t in tools if t["name"] == "analyze")
    assert "outputSchema" in analyze_tool
    assert analyze_tool["outputSchema"] == {
        "type": "object",
        "properties": {"score": {"type": "number"}},
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tools_list_omits_output_schema_when_not_set(
    output_schema_client: httpx.AsyncClient,
) -> None:
    """AC-68 complement: tools/list omits outputSchema key for tools without output_schema."""
    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {},
    }

    response = await output_schema_client.post(
        "/mcp",
        json=payload,
        headers=_HEADERS,
    )

    assert response.status_code == 200
    body = response.json()
    tools = body["result"]["tools"]
    plain_tool = next(t for t in tools if t["name"] == "plain")
    assert "outputSchema" not in plain_tool


# ---------------------------------------------------------------------------
# AC-69: structuredContent returned when outputSchema present
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tools_call_returns_structured_content_when_output_schema_set(
    output_schema_client: httpx.AsyncClient,
) -> None:
    """AC-69: tools/call returns structuredContent at top level when output_schema present."""
    payload = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "analyze",
            "arguments": {"text": "hello world"},
        },
    }

    response = await output_schema_client.post(
        "/mcp",
        json=payload,
        headers=_HEADERS,
    )

    assert response.status_code == 200
    body = response.json()
    result = body["result"]
    assert "structuredContent" in result
    assert result["structuredContent"] == {"score": 0.95}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tools_call_structured_content_includes_backward_compat_text(
    output_schema_client: httpx.AsyncClient,
) -> None:
    """AC-69: tools/call includes backward-compatible text content alongside structuredContent."""
    payload = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "analyze",
            "arguments": {"text": "hello world"},
        },
    }

    response = await output_schema_client.post(
        "/mcp",
        json=payload,
        headers=_HEADERS,
    )

    assert response.status_code == 200
    body = response.json()
    result = body["result"]
    assert "content" in result
    assert len(result["content"]) == 1
    assert result["content"][0]["type"] == "text"
    assert result["content"][0]["text"] == '{"score": 0.95}'


# ---------------------------------------------------------------------------
# AC-70: text content returned (no structuredContent) when no output_schema
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tools_call_returns_text_content_when_no_output_schema(
    output_schema_client: httpx.AsyncClient,
) -> None:
    """AC-70: tools/call returns text content without structuredContent when no output_schema."""
    payload = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "plain",
            "arguments": {"message": "hello"},
        },
    }

    response = await output_schema_client.post(
        "/mcp",
        json=payload,
        headers=_HEADERS,
    )

    assert response.status_code == 200
    body = response.json()
    result = body["result"]
    assert "structuredContent" not in result
    assert "content" in result
    assert len(result["content"]) == 1
    assert result["content"][0]["type"] == "text"
    assert result["content"][0]["text"] == "hello"
