"""Tests for FastAPI Request and Depends parameter injection in MCP tools."""

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from fastapi_mcp_router import MCPToolRegistry, create_mcp_router


@pytest.fixture
def app() -> FastAPI:
    """Create FastAPI app with MCP router for testing."""
    app_instance = FastAPI()
    return app_instance


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
def test_tool_without_request_parameter_works(registry: MCPToolRegistry, client: TestClient):
    """Test that tools without Request parameter work normally."""

    @registry.tool(
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        }
    )
    async def simple_tool(value: str) -> dict[str, str]:
        """Simple tool without Request parameter."""
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
    assert data["result"]["content"][0]["text"] == '{"result": "TEST"}'


@pytest.mark.integration
def test_tool_with_request_parameter_receives_request(registry: MCPToolRegistry, client: TestClient):
    """Test that tools with Request parameter receive the Request object."""

    @registry.tool(
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        }
    )
    async def tool_with_request(request: Request, message: str) -> dict[str, str]:
        """Tool that requires Request parameter."""
        api_key = request.headers.get("x-api-key", "no-key")
        return {"message": message, "api_key": api_key}

    response = client.post(
        "/mcp",
        headers={
            "MCP-Protocol-Version": "2025-06-18",
            "X-API-Key": "test-key-123",
        },
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "tool_with_request",
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
    assert result["api_key"] == "test-key-123"


@pytest.mark.integration
def test_tool_with_request_and_optional_params(registry: MCPToolRegistry, client: TestClient):
    """Test that Request injection works with optional parameters."""

    @registry.tool(
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
    )
    async def tool_with_optional(request: Request, name: str, count: int | None = None) -> dict[str, object]:
        """Tool with Request and optional parameters."""
        user_agent = request.headers.get("user-agent", "unknown")
        return {"name": name, "count": count, "user_agent": user_agent}

    response = client.post(
        "/mcp",
        headers={
            "MCP-Protocol-Version": "2025-06-18",
            "User-Agent": "test-client/1.0",
            "X-API-Key": "test-key",
        },
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "tool_with_optional", "arguments": {"name": "Alice"}},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "result" in data

    import json

    result = json.loads(data["result"]["content"][0]["text"])
    assert result["name"] == "Alice"
    assert result["count"] is None
    assert result["user_agent"] == "test-client/1.0"


@pytest.mark.integration
def test_tool_with_request_in_middle_of_params(registry: MCPToolRegistry, client: TestClient):
    """Test that Request injection works when Request is not the first parameter."""

    @registry.tool(
        input_schema={
            "type": "object",
            "properties": {
                "first": {"type": "string"},
                "second": {"type": "string"},
            },
            "required": ["first", "second"],
        }
    )
    async def tool_request_middle(first: str, request: Request, second: str) -> dict[str, str]:
        """Tool with Request parameter in the middle."""
        path = request.url.path
        return {"first": first, "second": second, "path": path}

    response = client.post(
        "/mcp",
        headers={"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"},
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "tool_request_middle",
                "arguments": {"first": "A", "second": "B"},
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "result" in data

    import json

    result = json.loads(data["result"]["content"][0]["text"])
    assert result["first"] == "A"
    assert result["second"] == "B"
    assert result["path"] == "/mcp"


@pytest.mark.integration
def test_multiple_tools_with_and_without_request(registry: MCPToolRegistry, client: TestClient):
    """Test that multiple tools with different signatures work correctly."""

    @registry.tool(
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        }
    )
    async def no_request(value: int) -> dict[str, int]:
        """Tool without Request."""
        return {"doubled": value * 2}

    @registry.tool(
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        }
    )
    async def with_request(request: Request, value: int) -> dict[str, object]:
        """Tool with Request."""
        method = request.method
        return {"tripled": value * 3, "method": method}

    # Test tool without Request
    response1 = client.post(
        "/mcp",
        headers={"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"},
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "no_request", "arguments": {"value": 5}},
        },
    )

    assert response1.status_code == 200
    data1 = response1.json()
    import json

    result1 = json.loads(data1["result"]["content"][0]["text"])
    assert result1["doubled"] == 10

    # Test tool with Request
    response2 = client.post(
        "/mcp",
        headers={"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"},
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "with_request", "arguments": {"value": 5}},
        },
    )

    assert response2.status_code == 200
    data2 = response2.json()
    result2 = json.loads(data2["result"]["content"][0]["text"])
    assert result2["tripled"] == 15
    assert result2["method"] == "POST"


@pytest.mark.integration
def test_tool_with_depends_resolves_dependency(registry: MCPToolRegistry, client: TestClient):
    """Test that tools with Depends() parameter resolve the dependency at runtime."""

    # Define a sync dependency function
    def get_user_from_header(request: Request) -> str:
        """Extract user from X-User header."""
        return request.headers.get("x-user", "anonymous")

    @registry.tool(
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        }
    )
    async def tool_with_depends(
        request: Request, message: str, user: str = Depends(get_user_from_header)
    ) -> dict[str, str]:
        """Tool that uses Depends() for authentication."""
        return {"message": message, "user": user}

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
                "name": "tool_with_depends",
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
    assert result["user"] == "alice"


@pytest.mark.integration
def test_tool_with_async_depends_resolves_dependency(registry: MCPToolRegistry, client: TestClient):
    """Test that tools with async Depends() parameter resolve the dependency."""

    # Define an async dependency function
    async def get_connection_async(request: Request) -> dict[str, object]:
        """Async dependency that returns connection info."""
        api_key = request.headers.get("x-api-key", "none")
        return {"api_key": api_key, "authenticated": api_key != "none"}

    @registry.tool(
        input_schema={
            "type": "object",
            "properties": {"data": {"type": "string"}},
            "required": ["data"],
        }
    )
    async def tool_with_async_depends(
        request: Request,
        data: str,
        connection: dict[str, object] = Depends(get_connection_async),
    ) -> dict[str, object]:
        """Tool that uses async Depends()."""
        return {"data": data, "connection": connection}

    response = client.post(
        "/mcp",
        headers={
            "MCP-Protocol-Version": "2025-06-18",
            "X-API-Key": "secret-key-123",
        },
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "tool_with_async_depends",
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
    assert result["connection"]["api_key"] == "secret-key-123"
    assert result["connection"]["authenticated"] is True


@pytest.mark.integration
def test_tool_depends_without_request_in_dependency(registry: MCPToolRegistry, client: TestClient):
    """Test Depends() with a dependency that doesn't need Request."""

    def get_config() -> dict[str, str]:
        """Simple dependency without Request."""
        return {"version": "1.0", "env": "test"}

    @registry.tool(
        input_schema={
            "type": "object",
            "properties": {"action": {"type": "string"}},
            "required": ["action"],
        }
    )
    async def tool_simple_depends(
        request: Request, action: str, config: dict[str, str] = Depends(get_config)
    ) -> dict[str, object]:
        """Tool with simple Depends()."""
        return {"action": action, "config": config}

    response = client.post(
        "/mcp",
        headers={"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"},
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "tool_simple_depends",
                "arguments": {"action": "process"},
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "result" in data

    import json

    result = json.loads(data["result"]["content"][0]["text"])
    assert result["action"] == "process"
    assert result["config"]["version"] == "1.0"
    assert result["config"]["env"] == "test"
