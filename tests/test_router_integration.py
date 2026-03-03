"""Integration tests for fastapi-mcp-router router module.

Tests the complete MCP router request/response cycle using FastAPI TestClient.
Covers initialize, notifications/initialized, tools/list, tools/call, ping methods,
header validation, and JSON-RPC error handling.
"""

from unittest.mock import patch
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from fastapi_mcp_router import (
    MCPError,
    MCPToolRegistry,
    ToolError,
    create_mcp_router,
)
from fastapi_mcp_router.types import McpSessionData, ServerInfo

# Test fixtures


@pytest.fixture(name="registry")
def registry_fixture() -> MCPToolRegistry:
    """Create tool registry with test tools."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def add(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    @registry.tool()
    async def echo(message: str) -> str:
        """Echo back a message."""
        return message

    @registry.tool()
    async def get_dict(key: str, value: str) -> dict[str, str]:
        """Return a dictionary with key and value."""
        return {key: value}

    @registry.tool()
    async def error_tool(should_error: bool = True) -> str:
        """Tool that raises ToolError when should_error is True."""
        if should_error:
            raise ToolError("This is a tool error")
        return "success"

    @registry.tool()
    async def mcp_error_tool(should_error: bool = True) -> str:
        """Tool that raises MCPError when should_error is True."""
        if should_error:
            raise MCPError(code=-32000, message="This is an MCP error")
        return "success"

    return registry


@pytest.fixture(name="app")
def app_fixture(registry: MCPToolRegistry) -> FastAPI:
    """Create test FastAPI app with MCP router."""
    app = FastAPI()
    mcp_router = create_mcp_router(registry, legacy_sse=True)
    app.include_router(mcp_router, prefix="/mcp")
    return app


@pytest.fixture(name="client")
def client_fixture(app: FastAPI) -> TestClient:
    """Create test client for FastAPI app."""
    return TestClient(app)


# Helper functions


def make_jsonrpc_request(
    method: str,
    params: dict[str, object] | None = None,
    request_id: int | str | None = 1,
) -> dict[str, object]:
    """Create JSON-RPC 2.0 request body.

    Args:
        method: JSON-RPC method name
        params: Optional parameters dict
        request_id: Optional request ID (None for notifications)

    Returns:
        JSON-RPC request dict
    """
    request: dict[str, object] = {
        "jsonrpc": "2.0",
        "method": method,
    }
    if params is not None:
        request["params"] = params
    if request_id is not None:
        request["id"] = request_id
    return request


def post_mcp(
    client: TestClient,
    body: dict[str, object],
    protocol_version: str = "2025-06-18",
    auth_header: str | None = "test-api-key",
) -> httpx.Response:
    """Post MCP request with protocol header and authentication.

    Args:
        client: TestClient instance
        body: JSON-RPC request body
        protocol_version: MCP protocol version header value
        auth_header: Authentication header value (X-API-Key) or None to omit

    Returns:
        Response object from client
    """
    headers = {"MCP-Protocol-Version": protocol_version}
    if auth_header is not None:
        headers["X-API-Key"] = auth_header
    return client.post("/mcp", json=body, headers=headers)


# Initialize method tests


@pytest.mark.integration
def test_initialize_successful(client: TestClient):
    """Test successful initialization with correct headers."""
    request = make_jsonrpc_request(
        method="initialize",
        params={"protocolVersion": "2025-06-18"},
        request_id=1,
    )

    response = post_mcp(client, request)

    assert response.status_code == 200
    data = response.json()
    assert data["jsonrpc"] == "2.0"
    assert data["id"] == 1
    assert "result" in data
    assert "error" not in data


@pytest.mark.integration
def test_initialize_protocol_version(client: TestClient):
    """Test response contains protocolVersion 2025-06-18."""
    request = make_jsonrpc_request(
        method="initialize",
        params={"protocolVersion": "2025-06-18"},
    )

    response = post_mcp(client, request)

    assert response.status_code == 200
    data = response.json()
    result = data["result"]
    assert result["protocolVersion"] == "2025-06-18"


@pytest.mark.integration
def test_initialize_capabilities_with_tools(client: TestClient):
    """Test response contains capabilities with tools."""
    request = make_jsonrpc_request(method="initialize", params={})

    response = post_mcp(client, request)

    assert response.status_code == 200
    data = response.json()
    result = data["result"]
    assert "capabilities" in result
    assert "tools" in result["capabilities"]
    assert isinstance(result["capabilities"]["tools"], dict)


@pytest.mark.integration
def test_initialize_server_info(client: TestClient):
    """Test response contains serverInfo with name and version."""
    request = make_jsonrpc_request(method="initialize", params={})

    response = post_mcp(client, request)

    assert response.status_code == 200
    data = response.json()
    result = data["result"]
    assert "serverInfo" in result
    assert "name" in result["serverInfo"]
    assert "version" in result["serverInfo"]
    assert result["serverInfo"]["name"] == "fastapi-mcp-router"
    assert result["serverInfo"]["version"] == "0.1.0"


@pytest.mark.integration
def test_initialize_custom_server_info(registry: MCPToolRegistry):
    """Test custom server_info merges with defaults and returns icons."""
    app = FastAPI()
    custom_info: ServerInfo = {
        "name": "my-server",
        "version": "2.0.0",
        "title": "My Custom Server",
        "description": "A test server",
        "websiteUrl": "https://example.com",
        "icons": [{"src": "https://example.com/icon.svg", "mimeType": "image/svg+xml"}],
    }
    mcp_router = create_mcp_router(registry, server_info=custom_info)
    app.include_router(mcp_router, prefix="/mcp")
    client = TestClient(app)

    request = make_jsonrpc_request(method="initialize", params={})
    response = post_mcp(client, request)

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["serverInfo"]["name"] == "my-server"
    assert result["serverInfo"]["version"] == "2.0.0"
    assert result["serverInfo"]["title"] == "My Custom Server"
    assert result["serverInfo"]["icons"][0]["src"] == "https://example.com/icon.svg"


@pytest.mark.integration
def test_initialize_partial_server_info_merges_defaults(registry: MCPToolRegistry):
    """Test partial server_info preserves default name/version when not overridden."""
    app = FastAPI()
    partial_info: ServerInfo = {"title": "Just a Title"}  # Only override title
    mcp_router = create_mcp_router(registry, server_info=partial_info)
    app.include_router(mcp_router, prefix="/mcp")
    client = TestClient(app)

    request = make_jsonrpc_request(method="initialize", params={})
    response = post_mcp(client, request)

    result = response.json()["result"]
    # Defaults preserved
    assert result["serverInfo"]["name"] == "fastapi-mcp-router"
    assert result["serverInfo"]["version"] == "0.1.0"
    # Custom field added
    assert result["serverInfo"]["title"] == "Just a Title"


@pytest.mark.integration
def test_initialize_no_session_id_stateless(client: TestClient):
    """Test response does NOT contain sessionId in stateless mode."""
    request = make_jsonrpc_request(method="initialize", params={})

    response = post_mcp(client, request)

    assert response.status_code == 200
    data = response.json()
    result = data["result"]
    assert "sessionId" not in result


# Notifications/initialized tests


@pytest.mark.integration
def test_initialized_notification_accepted(client: TestClient):
    """Test initialized notification returns 202 Accepted."""
    request = make_jsonrpc_request(
        method="notifications/initialized",
        request_id=None,  # No id for notification
    )

    response = post_mcp(client, request)

    assert response.status_code == 202


@pytest.mark.integration
def test_initialized_notification_no_response_body(client: TestClient):
    """Test initialized notification has empty response body."""
    request = make_jsonrpc_request(
        method="notifications/initialized",
        request_id=None,
    )

    response = post_mcp(client, request)

    assert response.status_code == 202
    assert response.json() == {}


@pytest.mark.integration
def test_initialized_notification_without_id(client: TestClient):
    """Test initialized notification with no id field."""
    # Manually create request without id field
    request = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
    }

    response = post_mcp(client, request)

    assert response.status_code == 202
    assert response.json() == {}


# Tools/list method tests


@pytest.mark.integration
def test_tools_list_success(client: TestClient):
    """Test listing tools from registry with correct headers."""
    request = make_jsonrpc_request(method="tools/list")

    response = post_mcp(client, request)

    assert response.status_code == 200
    data = response.json()
    assert data["jsonrpc"] == "2.0"
    assert "result" in data
    assert "error" not in data


@pytest.mark.integration
def test_tools_list_contains_tools_array(client: TestClient):
    """Test response contains tools array."""
    request = make_jsonrpc_request(method="tools/list")

    response = post_mcp(client, request)

    assert response.status_code == 200
    data = response.json()
    result = data["result"]
    assert "tools" in result
    assert isinstance(result["tools"], list)


@pytest.mark.integration
def test_tools_list_correct_format(client: TestClient):
    """Test tools have correct format with name, description, inputSchema."""
    request = make_jsonrpc_request(method="tools/list")

    response = post_mcp(client, request)

    assert response.status_code == 200
    data = response.json()
    tools = data["result"]["tools"]

    # Check we have at least one tool
    assert len(tools) > 0

    # Check each tool has required fields
    for tool in tools:
        assert "name" in tool
        assert "description" in tool
        assert "inputSchema" in tool
        assert isinstance(tool["name"], str)
        assert isinstance(tool["description"], str)
        assert isinstance(tool["inputSchema"], dict)


@pytest.mark.integration
def test_tools_list_multiple_registered_tools(client: TestClient):
    """Test multiple registered tools are all listed."""
    request = make_jsonrpc_request(method="tools/list")

    response = post_mcp(client, request)

    assert response.status_code == 200
    data = response.json()
    tools = data["result"]["tools"]

    # We registered add, echo, get_dict, error_tool, mcp_error_tool
    assert len(tools) == 5

    tool_names = [tool["name"] for tool in tools]
    assert "add" in tool_names
    assert "echo" in tool_names
    assert "get_dict" in tool_names
    assert "error_tool" in tool_names
    assert "mcp_error_tool" in tool_names


# Tools/call method tests


@pytest.mark.integration
def test_tools_call_with_correct_arguments(client: TestClient):
    """Test calling tool with correct arguments."""
    request = make_jsonrpc_request(
        method="tools/call",
        params={"name": "add", "arguments": {"a": 5, "b": 3}},
    )

    response = post_mcp(client, request)

    assert response.status_code == 200
    data = response.json()
    assert data["jsonrpc"] == "2.0"
    assert "result" in data
    assert "error" not in data


@pytest.mark.integration
def test_tools_call_response_contains_content_array(client: TestClient):
    """Test response contains content array with text type."""
    request = make_jsonrpc_request(
        method="tools/call",
        params={"name": "add", "arguments": {"a": 2, "b": 3}},
    )

    response = post_mcp(client, request)

    assert response.status_code == 200
    data = response.json()
    result = data["result"]
    assert "content" in result
    assert isinstance(result["content"], list)
    assert len(result["content"]) == 1
    assert result["content"][0]["type"] == "text"
    assert "text" in result["content"][0]


@pytest.mark.integration
def test_tools_call_dict_result_as_json_string(client: TestClient):
    """Test tool returns dict result as JSON string."""
    request = make_jsonrpc_request(
        method="tools/call",
        params={"name": "get_dict", "arguments": {"key": "foo", "value": "bar"}},
    )

    response = post_mcp(client, request)

    assert response.status_code == 200
    data = response.json()
    result = data["result"]
    text = result["content"][0]["text"]

    # Dict should be serialized to JSON
    import json

    parsed = json.loads(text)
    assert parsed == {"foo": "bar"}


@pytest.mark.integration
def test_tools_call_str_result_as_is(client: TestClient):
    """Test tool returns str result as-is."""
    request = make_jsonrpc_request(
        method="tools/call",
        params={"name": "echo", "arguments": {"message": "Hello World"}},
    )

    response = post_mcp(client, request)

    assert response.status_code == 200
    data = response.json()
    result = data["result"]
    text = result["content"][0]["text"]

    # String should be returned as-is
    assert text == "Hello World"


@pytest.mark.integration
def test_tools_call_nonexistent_tool_returns_error(client: TestClient):
    """Test calling non-existent tool returns error -32601."""
    request = make_jsonrpc_request(
        method="tools/call",
        params={"name": "nonexistent_tool", "arguments": {}},
    )

    response = post_mcp(client, request)

    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert "result" not in data
    assert data["error"]["code"] == -32601
    assert "not found" in data["error"]["message"].lower()


@pytest.mark.integration
def test_tools_call_missing_arguments_returns_error(client: TestClient):
    """Test calling tool with missing arguments returns error -32602."""
    request = make_jsonrpc_request(
        method="tools/call",
        params={"name": "add", "arguments": {"a": 5}},  # Missing 'b'
    )

    response = post_mcp(client, request)

    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == -32602


@pytest.mark.integration
def test_tools_call_tool_error_returns_is_error_true(client: TestClient):
    """Test tool that raises ToolError returns isError: true response."""
    request = make_jsonrpc_request(
        method="tools/call",
        params={"name": "error_tool", "arguments": {"should_error": True}},
    )

    response = post_mcp(client, request)

    assert response.status_code == 200
    data = response.json()
    # ToolError is converted to isError: true response by router
    assert "result" in data
    assert "error" not in data
    assert data["result"]["isError"] is True
    assert len(data["result"]["content"]) == 1
    assert data["result"]["content"][0]["type"] == "text"
    assert "This is a tool error" in data["result"]["content"][0]["text"]


@pytest.mark.integration
def test_tools_call_mcp_error_returns_json_rpc_error(client: TestClient):
    """Test tool that raises MCPError returns JSON-RPC error."""
    request = make_jsonrpc_request(
        method="tools/call",
        params={"name": "mcp_error_tool", "arguments": {"should_error": True}},
    )

    response = post_mcp(client, request)

    assert response.status_code == 200
    data = response.json()
    # MCPError returns error (not result)
    assert "error" in data
    assert "result" not in data
    assert data["error"]["code"] == -32000
    assert "This is an MCP error" in data["error"]["message"]


@pytest.mark.integration
def test_tools_call_missing_name_parameter(client: TestClient):
    """Test calling tools/call without name parameter returns error -32602."""
    request = make_jsonrpc_request(
        method="tools/call",
        params={"arguments": {"a": 1, "b": 2}},  # Missing 'name'
    )

    response = post_mcp(client, request)

    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == -32602
    assert "name" in data["error"]["message"].lower()


# Ping method tests


@pytest.mark.integration
def test_ping_returns_empty_result(client: TestClient):
    """Test ping returns empty result {}."""
    request = make_jsonrpc_request(method="ping")

    response = post_mcp(client, request)

    assert response.status_code == 200
    data = response.json()
    assert data["jsonrpc"] == "2.0"
    assert "result" in data
    assert data["result"] == {}


@pytest.mark.integration
def test_ping_with_correct_headers(client: TestClient):
    """Test ping with correct headers."""
    request = make_jsonrpc_request(method="ping")

    response = post_mcp(client, request, protocol_version="2025-06-18")

    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "error" not in data


# Header validation tests


@pytest.mark.integration
def test_missing_protocol_version_header_defaults_to_2025_03_26(client: TestClient):
    """Test missing MCP-Protocol-Version header defaults to 2025-03-26 for backwards compatibility."""
    request = make_jsonrpc_request(method="ping")

    # Post without MCP-Protocol-Version header (but with auth)
    response = client.post("/mcp", json=request, headers={"X-API-Key": "test-key"})

    # Should succeed with default version 2025-03-26
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert data["result"] == {}


@pytest.mark.integration
def test_unsupported_protocol_version_returns_400(client: TestClient):
    """Test unsupported protocol version returns 400."""
    request = make_jsonrpc_request(method="ping")

    response = post_mcp(client, request, protocol_version="1.0.0")

    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert "unsupported" in data["error"].lower() or "supported" in data["error"].lower()


@pytest.mark.integration
def test_correct_protocol_version_works(client: TestClient):
    """Test correct protocol version 2025-06-18 works."""
    request = make_jsonrpc_request(method="ping")

    response = post_mcp(client, request, protocol_version="2025-06-18")

    assert response.status_code == 200
    data = response.json()
    assert "result" in data


# JSON-RPC validation tests


@pytest.mark.integration
def test_invalid_jsonrpc_version_returns_error(client: TestClient):
    """Test invalid jsonrpc version returns error -32600."""
    request = {
        "jsonrpc": "1.0",  # Invalid version
        "id": 1,
        "method": "ping",
    }

    response = post_mcp(client, request)

    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == -32600


@pytest.mark.integration
def test_missing_method_returns_error(client: TestClient):
    """Test missing method returns error."""
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        # Missing method field
    }

    response = post_mcp(client, request)

    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == -32601


# NOTE: Test removed - invalid JSON body currently causes UnboundLocalError
# in router.py line 206 because 'body' variable is not assigned when JSON
# parsing fails. This is a bug that should be fixed in the router implementation.
# The router should initialize body = None before the try block.


@pytest.mark.integration
def test_method_not_found_returns_error_32601(client: TestClient):
    """Test unknown method returns error -32601."""
    request = make_jsonrpc_request(method="unknown/method")

    response = post_mcp(client, request)

    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == -32601
    assert "not found" in data["error"]["message"].lower()


# Response format validation tests


@pytest.mark.integration
def test_successful_response_has_correct_jsonrpc_format(client: TestClient):
    """Test successful responses have correct JSON-RPC 2.0 format."""
    request = make_jsonrpc_request(method="ping", request_id=123)

    response = post_mcp(client, request)

    assert response.status_code == 200
    data = response.json()
    assert "jsonrpc" in data
    assert data["jsonrpc"] == "2.0"
    assert "id" in data
    assert data["id"] == 123
    assert "result" in data
    assert "error" not in data


@pytest.mark.integration
def test_error_response_has_correct_jsonrpc_format(client: TestClient):
    """Test error responses have correct JSON-RPC 2.0 format."""
    request = make_jsonrpc_request(method="unknown/method", request_id=456)

    response = post_mcp(client, request)

    assert response.status_code == 200
    data = response.json()
    assert "jsonrpc" in data
    assert data["jsonrpc"] == "2.0"
    assert "id" in data
    assert data["id"] == 456
    assert "error" in data
    assert "result" not in data
    assert "code" in data["error"]
    assert "message" in data["error"]


# Edge cases and integration scenarios


@pytest.mark.integration
def test_multiple_sequential_requests(client: TestClient):
    """Test multiple sequential requests work correctly."""
    # Initialize
    init_request = make_jsonrpc_request(
        method="initialize",
        params={"protocolVersion": "2025-06-18"},
        request_id=1,
    )
    response1 = post_mcp(client, init_request)
    assert response1.status_code == 200

    # List tools
    list_request = make_jsonrpc_request(method="tools/list", request_id=2)
    response2 = post_mcp(client, list_request)
    assert response2.status_code == 200

    # Call tool
    call_request = make_jsonrpc_request(
        method="tools/call",
        params={"name": "add", "arguments": {"a": 10, "b": 20}},
        request_id=3,
    )
    response3 = post_mcp(client, call_request)
    assert response3.status_code == 200
    result = response3.json()["result"]
    assert result["content"][0]["text"] == "30"


@pytest.mark.integration
def test_tool_with_no_parameters(client: TestClient, registry: MCPToolRegistry):
    """Test calling tool with no parameters."""

    @registry.tool()
    async def no_params() -> str:
        """Tool with no parameters."""
        return "success"

    # Re-create router with updated registry
    app = FastAPI()
    mcp_router = create_mcp_router(registry)
    app.include_router(mcp_router, prefix="/mcp")
    client = TestClient(app)

    request = make_jsonrpc_request(
        method="tools/call",
        params={"name": "no_params", "arguments": {}},
    )

    response = post_mcp(client, request)

    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert data["result"]["content"][0]["text"] == "success"


@pytest.mark.integration
def test_tool_call_with_numeric_result(client: TestClient):
    """Test tool call that returns numeric result is converted to string."""
    request = make_jsonrpc_request(
        method="tools/call",
        params={"name": "add", "arguments": {"a": 100, "b": 200}},
    )

    response = post_mcp(client, request)

    assert response.status_code == 200
    data = response.json()
    result = data["result"]
    # Numeric result should be serialized
    assert result["content"][0]["text"] == "300"


@pytest.mark.integration
def test_request_id_preserved_in_response(client: TestClient):
    """Test request ID is preserved in response."""
    request = make_jsonrpc_request(method="ping", request_id="custom-id-123")

    response = post_mcp(client, request)

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "custom-id-123"


@pytest.mark.integration
def test_params_field_optional_for_methods(client: TestClient):
    """Test params field is optional when method doesn't need parameters."""
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "ping",
        # No params field
    }

    response = post_mcp(client, request)

    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert data["result"] == {}


@pytest.mark.integration
def test_invalid_json_returns_parse_error(client: TestClient):
    """Test malformed JSON returns parse error -32700."""
    # Post invalid JSON directly
    response = client.post(
        "/mcp",
        content=b"{invalid json}",  # Malformed JSON
        headers={
            "Content-Type": "application/json",
            "MCP-Protocol-Version": "2025-06-18",
            "X-API-Key": "test-key",
        },
    )

    assert response.status_code == 200  # JSON-RPC errors return 200
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == -32700  # Parse error
    assert "parse" in data["error"]["message"].lower()
    assert data.get("id") is None  # No request_id available


@pytest.mark.integration
def test_empty_request_body_returns_parse_error(client: TestClient):
    """Test empty request body returns parse error."""
    response = client.post(
        "/mcp",
        content=b"",  # Empty body
        headers={
            "Content-Type": "application/json",
            "MCP-Protocol-Version": "2025-06-18",
            "X-API-Key": "test-key",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == -32700
    assert data.get("id") is None


# POST endpoint authentication tests


@pytest.mark.integration
def test_post_endpoint_requires_authentication(client: TestClient):
    """Test POST /mcp returns 401 when no authentication headers provided."""
    request = make_jsonrpc_request(method="ping")

    # Post without any authentication headers
    response = client.post(
        "/mcp",
        json=request,
        headers={"MCP-Protocol-Version": "2025-06-18"},
    )

    assert response.status_code == 401
    data = response.json()
    assert "error" in data
    assert "Authentication required" in data["error"]
    # WWW-Authenticate: Bearer realm="mcp" is always present on 401 (EC-1)
    assert response.headers["WWW-Authenticate"] == 'Bearer realm="mcp"'


@pytest.mark.integration
def test_post_endpoint_with_api_key_succeeds(client: TestClient):
    """Test POST /mcp returns 200 when X-API-Key header provided."""
    request = make_jsonrpc_request(method="ping")

    response = post_mcp(client, request, auth_header="test-api-key")

    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert data["result"] == {}


@pytest.mark.integration
def test_post_endpoint_with_bearer_token_succeeds(client: TestClient):
    """Test POST /mcp returns 200 when Authorization Bearer header provided."""
    request = make_jsonrpc_request(method="ping")

    response = client.post(
        "/mcp",
        json=request,
        headers={
            "MCP-Protocol-Version": "2025-06-18",
            "Authorization": "Bearer test-token",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert data["result"] == {}


@pytest.mark.integration
def test_post_endpoint_with_invalid_auth_header_returns_401(client: TestClient):
    """Test POST /mcp returns 401 when Authorization header lacks Bearer prefix."""
    request = make_jsonrpc_request(method="ping")

    response = client.post(
        "/mcp",
        json=request,
        headers={
            "MCP-Protocol-Version": "2025-06-18",
            "Authorization": "Basic credentials",
        },
    )

    assert response.status_code == 401
    data = response.json()
    assert "error" in data
    assert "Authentication required" in data["error"]


@pytest.mark.integration
def test_post_endpoint_initialize_requires_authentication(client: TestClient):
    """Test POST /mcp initialize method requires authentication."""
    request = make_jsonrpc_request(
        method="initialize",
        params={"protocolVersion": "2025-06-18"},
    )

    # Post without authentication
    response = client.post(
        "/mcp",
        json=request,
        headers={"MCP-Protocol-Version": "2025-06-18"},
    )

    assert response.status_code == 401
    data = response.json()
    assert "error" in data
    assert "Authentication required" in data["error"]


@pytest.mark.integration
def test_post_endpoint_tools_call_requires_authentication(client: TestClient):
    """Test POST /mcp tools/call method requires authentication."""
    request = make_jsonrpc_request(
        method="tools/call",
        params={"name": "add", "arguments": {"a": 1, "b": 2}},
    )

    # Post without authentication
    response = client.post(
        "/mcp",
        json=request,
        headers={"MCP-Protocol-Version": "2025-06-18"},
    )

    assert response.status_code == 401
    data = response.json()
    assert "error" in data
    assert "Authentication required" in data["error"]


# GET endpoint authentication tests


@pytest.mark.integration
def test_get_endpoint_requires_authentication(client: TestClient):
    """Test GET /mcp returns 401 when no authentication headers provided."""
    response = client.get("/mcp")

    assert response.status_code == 401
    data = response.json()
    assert "error" in data
    assert "Authentication required" in data["error"]
    # WWW-Authenticate: Bearer realm="mcp" is always present on 401 (EC-1)
    assert response.headers["WWW-Authenticate"] == 'Bearer realm="mcp"'


@pytest.mark.integration
def test_get_endpoint_with_api_key_succeeds(client: TestClient):
    """Test GET /mcp returns 200 when X-API-Key header provided."""
    response = client.get("/mcp", headers={"X-API-Key": "test-api-key"})

    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "MCP Server"
    assert "supportedVersions" in data


@pytest.mark.integration
def test_get_endpoint_with_bearer_token_succeeds(client: TestClient):
    """Test GET /mcp returns 200 when Authorization Bearer header provided."""
    response = client.get("/mcp", headers={"Authorization": "Bearer test-token"})

    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "MCP Server"
    assert "supportedVersions" in data


@pytest.mark.integration
def test_get_endpoint_with_invalid_auth_header_returns_401(client: TestClient):
    """Test GET /mcp returns 401 when Authorization header lacks Bearer prefix."""
    response = client.get("/mcp", headers={"Authorization": "Basic credentials"})

    assert response.status_code == 401
    data = response.json()
    assert "error" in data
    assert "Authentication required" in data["error"]


@pytest.mark.integration
def test_get_endpoint_sse_request_requires_authentication(client: TestClient):
    """Test GET /mcp with SSE Accept header still requires authentication."""
    response = client.get("/mcp", headers={"Accept": "text/event-stream"})

    assert response.status_code == 401
    data = response.json()
    assert "error" in data
    assert "Authentication required" in data["error"]


@pytest.mark.integration
def test_get_endpoint_sse_with_auth_returns_server_info(client: TestClient):
    """Test authenticated SSE request returns server info in stateless mode."""
    response = client.get(
        "/mcp",
        headers={"Accept": "text/event-stream", "X-API-Key": "test-key"},
    )

    # In stateless mode (no session callbacks), returns server info as JSON
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "MCP Server"
    assert data["transport"] == "HTTP (stateless)"
    assert "supportedVersions" in data


# Error handler tests (IR-15 through IR-20)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_client_disconnect_during_body_read_returns_gracefully(
    auth_client: httpx.AsyncClient,
) -> None:
    """Test ClientDisconnect during request.body() read returns HTTP 499 with empty body."""
    from unittest.mock import AsyncMock, patch

    from starlette.requests import ClientDisconnect

    with patch(
        "starlette.requests.Request.body",
        new_callable=AsyncMock,
        side_effect=ClientDisconnect(),
    ):
        response = await auth_client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
            headers={"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"},
        )

    assert response.status_code == 499
    assert response.json() == {}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_tool_raises_unexpected_exception_returns_32603(
    registry: MCPToolRegistry,
) -> None:
    """Test tool raising RuntimeError returns JSON-RPC error code -32603."""

    @registry.tool()
    async def boom() -> str:
        """Tool that always raises RuntimeError."""
        raise RuntimeError("unexpected failure")

    fastapi_app = FastAPI()

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        return api_key == "test-key"

    mcp_router = create_mcp_router(registry, auth_validator=auth_validator)
    fastapi_app.include_router(mcp_router, prefix="/mcp")

    transport = httpx.ASGITransport(app=fastapi_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as local_client:
        response = await local_client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "boom", "arguments": {}}},
            headers={"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == -32603


@pytest.mark.asyncio
@pytest.mark.integration
async def test_session_creation_failure_returns_500() -> None:
    """Test session_creator returning None causes HTTP 500 with internal_error JSON."""
    registry = MCPToolRegistry()

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        return bearer_token is not None

    async def session_getter(session_id: str | None) -> McpSessionData | None:
        return None

    async def broken_session_creator(oauth_client_id: UUID | None, connection_id: UUID | None) -> str | None:
        return None

    fastapi_app = FastAPI()

    @fastapi_app.middleware("http")
    async def set_bearer_state(request: Request, call_next):
        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            request.state.oauth_client_id = uuid4()
            request.state.connection_id = uuid4()
        response = await call_next(request)
        return response

    mcp_router = create_mcp_router(
        registry,
        auth_validator=auth_validator,
        session_getter=session_getter,
        session_creator=broken_session_creator,
    )
    fastapi_app.include_router(mcp_router, prefix="/mcp")

    transport = httpx.ASGITransport(app=fastapi_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as local_client:
        response = await local_client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-06-18"},
            },
            headers={
                "MCP-Protocol-Version": "2025-06-18",
                "Authorization": "Bearer test-token",
            },
        )

    assert response.status_code == 500
    data = response.json()
    assert data["error"] == "internal_error"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_notifications_cancelled_returns_202(
    auth_client: httpx.AsyncClient,
) -> None:
    """Test notifications/cancelled returns HTTP 202 with empty body."""
    response = await auth_client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
            "params": {"requestId": 42},
        },
        headers={"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"},
    )

    assert response.status_code == 202
    assert response.json() == {}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_tools_call_with_array_arguments_returns_32602(
    registry: MCPToolRegistry,
) -> None:
    """Test tools/call with arguments as list returns JSON-RPC error -32602."""

    @registry.tool()
    async def noop(x: int) -> int:
        """No-op tool."""
        return x

    fastapi_app = FastAPI()

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        return api_key == "test-key"

    mcp_router = create_mcp_router(registry, auth_validator=auth_validator)
    fastapi_app.include_router(mcp_router, prefix="/mcp")

    transport = httpx.ASGITransport(app=fastapi_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as local_client:
        response = await local_client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "noop", "arguments": [1, 2]},
            },
            headers={"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == -32602


@pytest.mark.asyncio
@pytest.mark.integration
async def test_tools_call_with_string_arguments_returns_32602(
    registry: MCPToolRegistry,
) -> None:
    """Test tools/call with arguments as string returns JSON-RPC error -32602."""

    @registry.tool()
    async def noop2(x: int) -> int:
        """No-op tool for string args test."""
        return x

    fastapi_app = FastAPI()

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        return api_key == "test-key"

    mcp_router = create_mcp_router(registry, auth_validator=auth_validator)
    fastapi_app.include_router(mcp_router, prefix="/mcp")

    transport = httpx.ASGITransport(app=fastapi_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as local_client:
        response = await local_client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "noop2", "arguments": "invalid"},
            },
            headers={"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == -32602


@pytest.mark.asyncio
@pytest.mark.integration
async def test_unexpected_value_error_in_handler_returns_32603() -> None:
    """Unexpected ValueError inside the POST handler returns JSON-RPC -32603.

    Covers router.py lines 792-793 (except ValueError catch block).
    Patches handle_tools_list to raise ValueError to simulate an unexpected
    error that escapes the normal MCPError wrapping.
    """
    registry = MCPToolRegistry()
    app = FastAPI()

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        return api_key == "test-key"

    mcp_router = create_mcp_router(registry, auth_validator=auth_validator)
    app.include_router(mcp_router, prefix="/mcp")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        with patch("fastapi_mcp_router.router.handle_tools_list", side_effect=ValueError("unexpected value error")):
            response = await client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                headers={"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"},
            )

    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == -32603


@pytest.mark.asyncio
@pytest.mark.integration
async def test_unexpected_runtime_error_in_handler_returns_32603() -> None:
    """Unexpected RuntimeError inside the POST handler returns JSON-RPC -32603.

    Covers router.py lines 811-814 (except Exception catch block).
    Patches handle_tools_list to raise RuntimeError to simulate an unexpected
    error that is not a ValueError, ClientDisconnect, or MCPError.
    """
    registry = MCPToolRegistry()
    app = FastAPI()

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        return api_key == "test-key"

    mcp_router = create_mcp_router(registry, auth_validator=auth_validator)
    app.include_router(mcp_router, prefix="/mcp")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        with patch("fastapi_mcp_router.router.handle_tools_list", side_effect=RuntimeError("unexpected router error")):
            response = await client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                headers={"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"},
            )

    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == -32603
