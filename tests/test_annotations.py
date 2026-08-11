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


@pytest.mark.integration
def test_tool_annotations_model_dump_used_as_annotations_argument():
    """Integration: a ToolAnnotations instance's model_dump feeds tool(annotations=...) over HTTP.

    Exercises the intended usage path for the typed ToolAnnotations model: build one,
    dump it to a dict, and confirm the resulting tools/list wire shape matches what raw
    dict-based annotations already produce (see test_tool_with_annotations_includes_annotations_field).
    registry.tool() also validates the dict against ToolAnnotations at registration
    time (see test_tool_annotations_invalid_hint_type_rejected for the negative path).
    """
    from fastapi_mcp_router import ToolAnnotations

    registry = MCPToolRegistry()

    hints = ToolAnnotations(readOnlyHint=True, title="Typed Tool")

    @registry.tool(annotations=hints.model_dump(exclude_none=True))
    async def typed_annotations_tool(value: str) -> str:
        """Tool annotated via the typed ToolAnnotations model."""
        return value

    app = FastAPI()
    app.include_router(create_mcp_router(registry), prefix="/mcp")
    client = TestClient(app)

    request = {"jsonrpc": "2.0", "method": "tools/list", "id": 1}
    response = client.post(
        "/mcp",
        json=request,
        headers={"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"},
    )

    assert response.status_code == 200
    tools = response.json()["result"]["tools"]
    tool = next(t for t in tools if t["name"] == "typed_annotations_tool")
    assert tool["annotations"] == {"readOnlyHint": True, "title": "Typed Tool"}


@pytest.mark.unit
def test_tool_annotations_invalid_hint_type_rejected():
    """A known ToolAnnotations field with the wrong type is rejected at registration.

    registry.tool() validates annotations via ToolAnnotations.model_validate(),
    so a known hint field (readOnlyHint) that isn't a bool raises a pydantic
    ValidationError instead of silently reaching the wire.
    """
    from pydantic import ValidationError

    registry = MCPToolRegistry()

    with pytest.raises(ValidationError):

        @registry.tool(annotations={"readOnlyHint": {"nested": "object"}})
        async def invalid_annotations_tool(value: str) -> str:
            """Tool with an invalid annotation value."""
            return value
