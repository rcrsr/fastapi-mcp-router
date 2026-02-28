"""Tests for tool annotations support in fastapi-mcp-router.

Tests the annotations parameter in the tool decorator and its inclusion
in the list_tools() response.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fastapi_mcp_router import MCPToolRegistry, create_mcp_router


@pytest.fixture(name="registry")
def registry_fixture() -> MCPToolRegistry:
    """Create tool registry with annotated and non-annotated tools."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def simple_tool(value: str) -> str:
        """Simple tool without annotations."""
        return value.upper()

    @registry.tool(annotations={"readOnlyHint": True})
    async def readonly_tool(id: str) -> dict[str, str]:
        """Tool with readOnlyHint annotation."""
        return {"id": id, "name": "Example"}

    @registry.tool(annotations={"readOnlyHint": False, "custom": "value"})
    async def write_tool(data: dict[str, str]) -> dict[str, str]:
        """Tool with multiple annotations."""
        return data

    return registry


@pytest.fixture(name="app")
def app_fixture(registry: MCPToolRegistry) -> FastAPI:
    """Create test FastAPI app with MCP router."""
    app = FastAPI()
    mcp_router = create_mcp_router(registry)
    app.include_router(mcp_router, prefix="/mcp")
    return app


@pytest.fixture(name="client")
def client_fixture(app: FastAPI) -> TestClient:
    """Create test client for FastAPI app."""
    return TestClient(app)


@pytest.mark.integration
def test_tool_without_annotations_has_no_annotations_field(client: TestClient):
    """Test that tools without annotations don't include annotations field."""
    request = {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "id": 1,
    }

    response = client.post(
        "/mcp",
        json=request,
        headers={"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"},
    )

    assert response.status_code == 200
    data = response.json()
    tools = data["result"]["tools"]

    # Find simple_tool
    simple_tool = next(tool for tool in tools if tool["name"] == "simple_tool")
    assert "annotations" not in simple_tool


@pytest.mark.integration
def test_tool_with_annotations_includes_annotations_field(client: TestClient):
    """Test that tools with annotations include annotations field."""
    request = {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "id": 1,
    }

    response = client.post(
        "/mcp",
        json=request,
        headers={"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"},
    )

    assert response.status_code == 200
    data = response.json()
    tools = data["result"]["tools"]

    # Find readonly_tool
    readonly_tool = next(tool for tool in tools if tool["name"] == "readonly_tool")
    assert "annotations" in readonly_tool
    assert readonly_tool["annotations"]["readOnlyHint"] is True


@pytest.mark.integration
def test_annotations_with_multiple_fields(client: TestClient):
    """Test that tools can have multiple annotation fields."""
    request = {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "id": 1,
    }

    response = client.post(
        "/mcp",
        json=request,
        headers={"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"},
    )

    assert response.status_code == 200
    data = response.json()
    tools = data["result"]["tools"]

    # Find write_tool
    write_tool = next(tool for tool in tools if tool["name"] == "write_tool")
    assert "annotations" in write_tool
    assert write_tool["annotations"]["readOnlyHint"] is False
    assert write_tool["annotations"]["custom"] == "value"


@pytest.mark.unit
def test_list_tools_direct_call_includes_annotations():
    """Test that list_tools() method directly includes annotations."""
    registry = MCPToolRegistry()

    @registry.tool(annotations={"readOnlyHint": True})
    async def test_tool(value: str) -> str:
        """Test tool."""
        return value

    tools = registry.list_tools()
    assert len(tools) == 1
    assert tools[0]["name"] == "test_tool"
    assert "annotations" in tools[0]
    assert tools[0]["annotations"]["readOnlyHint"] is True


@pytest.mark.unit
def test_list_tools_without_annotations_excludes_field():
    """Test that list_tools() excludes annotations field when None."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def test_tool(value: str) -> str:
        """Test tool."""
        return value

    tools = registry.list_tools()
    assert len(tools) == 1
    assert tools[0]["name"] == "test_tool"
    assert "annotations" not in tools[0]


@pytest.mark.integration
def test_annotations_do_not_affect_tool_execution(client: TestClient):
    """Test that annotations don't affect tool execution."""
    request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 1,
        "params": {
            "name": "readonly_tool",
            "arguments": {"id": "test-123"},
        },
    }

    response = client.post(
        "/mcp",
        json=request,
        headers={"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "isError" not in data["result"]

    import json

    result_text = data["result"]["content"][0]["text"]
    result_data = json.loads(result_text)
    assert result_data["id"] == "test-123"
    assert result_data["name"] == "Example"
