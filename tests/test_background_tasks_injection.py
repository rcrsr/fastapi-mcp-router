"""Tests for FastAPI BackgroundTasks parameter injection in MCP tools."""

import pytest
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.testclient import TestClient

from fastapi_mcp_router import MCPToolRegistry, create_mcp_router


@pytest.fixture
def app() -> FastAPI:
    """Create FastAPI app with MCP router for testing."""
    return FastAPI()


@pytest.fixture
def registry() -> MCPToolRegistry:
    """Create fresh tool registry for each test."""
    return MCPToolRegistry()


@pytest.fixture
def client(app: FastAPI, registry: MCPToolRegistry) -> TestClient:
    """Create test client with MCP router."""
    mcp_router = create_mcp_router(registry)
    app.include_router(mcp_router, prefix="/mcp")
    return TestClient(app)


@pytest.mark.integration
def test_tool_with_background_tasks_direct_type(registry: MCPToolRegistry, client: TestClient):
    """Test that tools with BackgroundTasks parameter receive the object (direct type)."""
    tasks_executed = []

    def background_task(task_id: str) -> None:
        """Sample background task."""
        tasks_executed.append(task_id)

    @registry.tool(
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        }
    )
    async def tool_with_bg_direct(message: str, background_tasks: BackgroundTasks) -> dict[str, str]:
        """Tool with direct BackgroundTasks type annotation."""
        background_tasks.add_task(background_task, "task-1")
        return {"message": message, "scheduled": True}

    response = client.post(
        "/mcp",
        headers={"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"},
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "tool_with_bg_direct",
                "arguments": {"message": "hello"},
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "result" in data

    import json

    result = json.loads(data["result"]["content"][0]["text"])
    assert result["message"] == "hello"
    assert result["scheduled"] is True
    assert "task-1" in tasks_executed


@pytest.mark.integration
def test_tool_with_background_tasks_union_type(registry: MCPToolRegistry, client: TestClient):
    """Test that tools with BackgroundTasks | None parameter receive the object (union type)."""
    tasks_executed = []

    def background_task(task_id: str) -> None:
        """Sample background task."""
        tasks_executed.append(task_id)

    @registry.tool(
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        }
    )
    async def tool_with_bg_union(message: str, background_tasks: BackgroundTasks | None = None) -> dict[str, object]:
        """Tool with BackgroundTasks | None type annotation."""
        scheduled = False
        if background_tasks is not None:
            background_tasks.add_task(background_task, "task-2")
            scheduled = True
        return {"message": message, "scheduled": scheduled}

    response = client.post(
        "/mcp",
        headers={"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"},
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "tool_with_bg_union",
                "arguments": {"message": "hello"},
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "result" in data

    import json

    result = json.loads(data["result"]["content"][0]["text"])
    assert result["message"] == "hello"
    assert result["scheduled"] is True
    assert "task-2" in tasks_executed


@pytest.mark.integration
def test_tool_with_request_and_background_tasks_union(registry: MCPToolRegistry, client: TestClient):
    """Test that Request and BackgroundTasks | None both work together."""
    tasks_executed = []

    def background_task(user: str, action: str) -> None:
        """Sample background task."""
        tasks_executed.append(f"{user}:{action}")

    @registry.tool(
        input_schema={
            "type": "object",
            "properties": {"action": {"type": "string"}},
            "required": ["action"],
        }
    )
    async def tool_with_both(
        request: Request,
        action: str,
        background_tasks: BackgroundTasks | None = None,
    ) -> dict[str, object]:
        """Tool with both Request and BackgroundTasks | None."""
        user = request.headers.get("x-user", "anonymous")
        scheduled = False
        if background_tasks is not None:
            background_tasks.add_task(background_task, user, action)
            scheduled = True
        return {"user": user, "action": action, "scheduled": scheduled}

    response = client.post(
        "/mcp",
        headers={
            "MCP-Protocol-Version": "2025-06-18",
            "X-User": "alice",
            "X-API-Key": "test-key",
        },
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "tool_with_both",
                "arguments": {"action": "create"},
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "result" in data

    import json

    result = json.loads(data["result"]["content"][0]["text"])
    assert result["user"] == "alice"
    assert result["action"] == "create"
    assert result["scheduled"] is True
    assert "alice:create" in tasks_executed


@pytest.mark.integration
def test_tool_without_background_tasks_works(registry: MCPToolRegistry, client: TestClient):
    """Test that tools without BackgroundTasks parameter work normally."""

    @registry.tool(
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        }
    )
    async def simple_tool(value: str) -> dict[str, str]:
        """Simple tool without BackgroundTasks parameter."""
        return {"result": value.upper()}

    response = client.post(
        "/mcp",
        headers={"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"},
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "simple_tool", "arguments": {"value": "test"}},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "result" in data

    import json

    result = json.loads(data["result"]["content"][0]["text"])
    assert result["result"] == "TEST"


@pytest.mark.integration
def test_tool_background_tasks_optional_when_none(registry: MCPToolRegistry, client: TestClient):
    """Test that BackgroundTasks | None handles None gracefully."""

    @registry.tool(
        input_schema={
            "type": "object",
            "properties": {"data": {"type": "string"}},
            "required": ["data"],
        }
    )
    async def tool_bg_optional(data: str, background_tasks: BackgroundTasks | None = None) -> dict[str, object]:
        """Tool with optional BackgroundTasks."""
        has_tasks = background_tasks is not None
        return {"data": data, "has_background_tasks": has_tasks}

    response = client.post(
        "/mcp",
        headers={"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"},
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "tool_bg_optional",
                "arguments": {"data": "test-data"},
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "result" in data

    import json

    result = json.loads(data["result"]["content"][0]["text"])
    assert result["data"] == "test-data"
    # BackgroundTasks should be injected by the router
    assert result["has_background_tasks"] is True


@pytest.mark.integration
def test_tool_request_union_type(registry: MCPToolRegistry, client: TestClient):
    """Test that Request | None union type also works (regression test)."""

    @registry.tool(
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        }
    )
    async def tool_request_union(value: str, request: Request | None = None) -> dict[str, object]:
        """Tool with Request | None type annotation."""
        has_request = request is not None
        api_key = None
        if request is not None:
            api_key = request.headers.get("x-api-key")
        return {"value": value, "has_request": has_request, "api_key": api_key}

    response = client.post(
        "/mcp",
        headers={
            "MCP-Protocol-Version": "2025-06-18",
            "X-API-Key": "secret-123",
        },
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "tool_request_union",
                "arguments": {"value": "test"},
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "result" in data

    import json

    result = json.loads(data["result"]["content"][0]["text"])
    assert result["value"] == "test"
    assert result["has_request"] is True
    assert result["api_key"] == "secret-123"
