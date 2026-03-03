"""Integration tests for fastapi-mcp-router security features.

Tests authentication validation, rate limiting, and host header injection
prevention features added to the MCP router.
"""

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from fastapi_mcp_router import MCPToolRegistry, create_mcp_router

# Test fixtures


@pytest.fixture(name="registry")
def registry_fixture() -> MCPToolRegistry:
    """Create tool registry with test tool."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def ping() -> str:
        """Simple ping tool."""
        return "pong"

    return registry


# Helper functions


def make_jsonrpc_request(
    method: str,
    params: dict[str, object] | None = None,
    request_id: int | str | None = 1,
) -> dict[str, object]:
    """Create JSON-RPC 2.0 request body."""
    request: dict[str, object] = {
        "jsonrpc": "2.0",
        "method": method,
    }
    if params is not None:
        request["params"] = params
    if request_id is not None:
        request["id"] = request_id
    return request


# Authentication validator tests


@pytest.mark.integration
def test_auth_validator_rejects_invalid_api_key(registry: MCPToolRegistry):
    """Test auth_validator rejects requests with invalid API key."""

    async def validate_auth(api_key: str | None, bearer_token: str | None) -> bool:
        """Validate that API key is 'valid-key'."""
        if api_key:
            return api_key == "valid-key"
        if bearer_token:
            return bearer_token == "valid-token"
        return False

    app = FastAPI()
    mcp_router = create_mcp_router(registry, auth_validator=validate_auth)
    app.include_router(mcp_router, prefix="/mcp")
    client = TestClient(app)

    request = make_jsonrpc_request(method="ping")
    response = client.post(
        "/mcp",
        json=request,
        headers={
            "MCP-Protocol-Version": "2025-06-18",
            "X-API-Key": "invalid-key",
        },
    )

    assert response.status_code == 401
    data = response.json()
    assert "error" in data
    assert "Authentication required" in data["error"]


@pytest.mark.integration
def test_auth_validator_accepts_valid_api_key(registry: MCPToolRegistry):
    """Test auth_validator accepts requests with valid API key."""

    async def validate_auth(api_key: str | None, bearer_token: str | None) -> bool:
        """Validate that API key is 'valid-key'."""
        if api_key:
            return api_key == "valid-key"
        if bearer_token:
            return bearer_token == "valid-token"
        return False

    app = FastAPI()
    mcp_router = create_mcp_router(registry, auth_validator=validate_auth)
    app.include_router(mcp_router, prefix="/mcp")
    client = TestClient(app)

    request = make_jsonrpc_request(method="ping")
    response = client.post(
        "/mcp",
        json=request,
        headers={
            "MCP-Protocol-Version": "2025-06-18",
            "X-API-Key": "valid-key",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "result" in data


@pytest.mark.integration
def test_auth_validator_rejects_invalid_bearer_token(registry: MCPToolRegistry):
    """Test auth_validator rejects requests with invalid bearer token."""

    async def validate_auth(api_key: str | None, bearer_token: str | None) -> bool:
        """Validate that bearer token is 'valid-token'."""
        if api_key:
            return api_key == "valid-key"
        if bearer_token:
            return bearer_token == "valid-token"
        return False

    app = FastAPI()
    mcp_router = create_mcp_router(registry, auth_validator=validate_auth)
    app.include_router(mcp_router, prefix="/mcp")
    client = TestClient(app)

    request = make_jsonrpc_request(method="ping")
    response = client.post(
        "/mcp",
        json=request,
        headers={
            "MCP-Protocol-Version": "2025-06-18",
            "Authorization": "Bearer invalid-token",
        },
    )

    assert response.status_code == 401
    data = response.json()
    assert "error" in data
    assert "Authentication required" in data["error"]


@pytest.mark.integration
def test_auth_validator_accepts_valid_bearer_token(registry: MCPToolRegistry):
    """Test auth_validator accepts requests with valid bearer token."""

    async def validate_auth(api_key: str | None, bearer_token: str | None) -> bool:
        """Validate that bearer token is 'valid-token'."""
        if api_key:
            return api_key == "valid-key"
        if bearer_token:
            return bearer_token == "valid-token"
        return False

    app = FastAPI()
    mcp_router = create_mcp_router(registry, auth_validator=validate_auth)
    app.include_router(mcp_router, prefix="/mcp")
    client = TestClient(app)

    request = make_jsonrpc_request(method="ping")
    response = client.post(
        "/mcp",
        json=request,
        headers={
            "MCP-Protocol-Version": "2025-06-18",
            "Authorization": "Bearer valid-token",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "result" in data


@pytest.mark.integration
def test_auth_validator_works_for_get_endpoint(registry: MCPToolRegistry):
    """Test auth_validator is applied to GET endpoint."""

    async def validate_auth(api_key: str | None, bearer_token: str | None) -> bool:
        """Validate credentials."""
        if api_key:
            return api_key == "valid-key"
        return False

    app = FastAPI()
    mcp_router = create_mcp_router(registry, auth_validator=validate_auth, legacy_sse=True)
    app.include_router(mcp_router, prefix="/mcp")
    client = TestClient(app)

    # Invalid key should fail
    response = client.get("/mcp", headers={"X-API-Key": "invalid-key"})
    assert response.status_code == 401

    # Valid key should succeed
    response = client.get("/mcp", headers={"X-API-Key": "valid-key"})
    assert response.status_code == 200


# Rate limiting tests


@pytest.mark.integration
def test_rate_limit_dependency_blocks_requests(registry: MCPToolRegistry):
    """Test rate_limit_dependency blocks requests when limit exceeded."""

    async def rate_limit():
        """Rate limit that always raises."""
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    app = FastAPI()
    mcp_router = create_mcp_router(registry, rate_limit_dependency=rate_limit)
    app.include_router(mcp_router, prefix="/mcp")
    client = TestClient(app)

    request = make_jsonrpc_request(method="ping")
    response = client.post(
        "/mcp",
        json=request,
        headers={
            "MCP-Protocol-Version": "2025-06-18",
            "X-API-Key": "test-key",
        },
    )

    assert response.status_code == 429
    data = response.json()
    assert "detail" in data
    assert "Rate limit exceeded" in data["detail"]


@pytest.mark.integration
def test_rate_limit_dependency_allows_requests(registry: MCPToolRegistry):
    """Test rate_limit_dependency allows requests when under limit."""

    async def rate_limit():
        """Rate limit that always allows."""
        pass

    app = FastAPI()
    mcp_router = create_mcp_router(registry, rate_limit_dependency=rate_limit)
    app.include_router(mcp_router, prefix="/mcp")
    client = TestClient(app)

    request = make_jsonrpc_request(method="ping")
    response = client.post(
        "/mcp",
        json=request,
        headers={
            "MCP-Protocol-Version": "2025-06-18",
            "X-API-Key": "test-key",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "result" in data


@pytest.mark.integration
def test_rate_limit_dependency_works_for_get_endpoint(registry: MCPToolRegistry):
    """Test rate_limit_dependency is applied to GET endpoint."""

    async def rate_limit():
        """Rate limit that always raises."""
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    app = FastAPI()
    mcp_router = create_mcp_router(registry, rate_limit_dependency=rate_limit, legacy_sse=True)
    app.include_router(mcp_router, prefix="/mcp")
    client = TestClient(app)

    response = client.get("/mcp", headers={"X-API-Key": "test-key"})

    assert response.status_code == 429
    data = response.json()
    assert "detail" in data
    assert "Rate limit exceeded" in data["detail"]


# Host header injection prevention tests


@pytest.mark.integration
def test_base_url_uses_configured_parameter(registry: MCPToolRegistry):
    """Test base URL uses configured base_url parameter in WWW-Authenticate header."""
    prm = {"resource": "https://api.example.com/mcp", "authorization_servers": ["https://auth.example.com"]}
    app = FastAPI()
    mcp_router = create_mcp_router(
        registry,
        base_url="https://api.example.com",
        oauth_resource_metadata=prm,
        legacy_sse=True,
    )
    app.include_router(mcp_router, prefix="/mcp")
    client = TestClient(app)

    # Request without auth to get WWW-Authenticate header
    response = client.get("/mcp")

    assert response.status_code == 401
    www_auth = response.headers["WWW-Authenticate"]
    # Should use configured base_url, not request host
    assert "https://api.example.com/.well-known/oauth-protected-resource" in www_auth
    assert "testserver" not in www_auth  # TestClient uses testserver as host


@pytest.mark.integration
def test_base_url_falls_back_to_request_when_not_configured(registry: MCPToolRegistry):
    """Test base URL falls back to request.base_url when base_url not configured."""
    prm = {"resource": "http://testserver/mcp", "authorization_servers": ["https://auth.example.com"]}
    app = FastAPI()
    mcp_router = create_mcp_router(registry, oauth_resource_metadata=prm, legacy_sse=True)  # No base_url parameter
    app.include_router(mcp_router, prefix="/mcp")
    client = TestClient(app)

    # Request without auth to get WWW-Authenticate header
    response = client.get("/mcp")

    assert response.status_code == 401
    www_auth = response.headers["WWW-Authenticate"]
    # Should fall back to request.base_url
    assert "http://testserver/.well-known/oauth-protected-resource" in www_auth


@pytest.mark.integration
def test_base_url_strips_trailing_slash(registry: MCPToolRegistry):
    """Test base URL strips trailing slash from configured parameter."""
    prm = {"resource": "https://api.example.com/mcp", "authorization_servers": ["https://auth.example.com"]}
    app = FastAPI()
    mcp_router = create_mcp_router(
        registry,
        base_url="https://api.example.com/",
        oauth_resource_metadata=prm,
        legacy_sse=True,
    )
    app.include_router(mcp_router, prefix="/mcp")
    client = TestClient(app)

    # Request without auth to get WWW-Authenticate header
    response = client.get("/mcp")

    assert response.status_code == 401
    www_auth = response.headers["WWW-Authenticate"]
    # Should not have double slash
    assert "https://api.example.com/.well-known/oauth-protected-resource" in www_auth
    assert "https://api.example.com//.well-known" not in www_auth


# Combined security features tests


@pytest.mark.integration
def test_auth_validator_and_rate_limit_work_together(registry: MCPToolRegistry):
    """Test auth_validator and rate_limit_dependency work together."""
    request_count = 0

    async def validate_auth(api_key: str | None, bearer_token: str | None) -> bool:
        """Validate credentials."""
        return api_key == "valid-key"

    async def rate_limit():
        """Rate limit that blocks after first request."""
        nonlocal request_count
        request_count += 1
        if request_count > 1:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

    app = FastAPI()
    mcp_router = create_mcp_router(
        registry,
        rate_limit_dependency=rate_limit,
        auth_validator=validate_auth,
    )
    app.include_router(mcp_router, prefix="/mcp")
    client = TestClient(app)

    request = make_jsonrpc_request(method="ping")
    headers = {
        "MCP-Protocol-Version": "2025-06-18",
        "X-API-Key": "valid-key",
    }

    # First request should succeed (auth valid, under rate limit)
    response = client.post("/mcp", json=request, headers=headers)
    assert response.status_code == 200

    # Second request should fail with rate limit (auth happens after rate limit in dependency chain)
    response = client.post("/mcp", json=request, headers=headers)
    assert response.status_code == 429


# Auth validator falsy/truthy return value tests


@pytest.mark.integration
@pytest.mark.parametrize(
    "falsy_value",
    [None, False, 0, "", []],
    ids=["None", "False", "0", "empty_str", "empty_list"],
)
def test_auth_validator_falsy_return_yields_401(registry: MCPToolRegistry, falsy_value: object):
    """EC-1: Validator returning any falsy value returns 401 with WWW-Authenticate header.

    AC-1: None → 401 + WWW-Authenticate.
    AC-2: False → 401 + WWW-Authenticate.
    EC-1: All falsy values (None, False, 0, "", []) trigger 401.
    """

    async def validate_auth(api_key: str | None, bearer_token: str | None) -> object:
        """Return the parametrized falsy value unconditionally."""
        return falsy_value

    app = FastAPI()
    mcp_router = create_mcp_router(registry, auth_validator=validate_auth)
    app.include_router(mcp_router, prefix="/mcp")
    client = TestClient(app)

    body = make_jsonrpc_request(method="ping")
    response = client.post(
        "/mcp",
        json=body,
        headers={
            "MCP-Protocol-Version": "2025-06-18",
            "X-API-Key": "any-key",
        },
    )

    assert response.status_code == 401
    assert "WWW-Authenticate" in response.headers
    assert 'Bearer realm="mcp"' in response.headers["WWW-Authenticate"]


@pytest.mark.integration
def test_auth_validator_returns_none_yields_401_with_www_authenticate(registry: MCPToolRegistry):
    """AC-1: Validator returning None → 401 with WWW-Authenticate header."""

    async def validate_auth(api_key: str | None, bearer_token: str | None) -> None:
        """Return None unconditionally."""
        return None

    app = FastAPI()
    mcp_router = create_mcp_router(registry, auth_validator=validate_auth)
    app.include_router(mcp_router, prefix="/mcp")
    client = TestClient(app)

    body = make_jsonrpc_request(method="ping")
    response = client.post(
        "/mcp",
        json=body,
        headers={
            "MCP-Protocol-Version": "2025-06-18",
            "Authorization": "Bearer some-token",
        },
    )

    assert response.status_code == 401
    assert "WWW-Authenticate" in response.headers


@pytest.mark.integration
def test_auth_validator_returns_false_yields_401_with_www_authenticate(registry: MCPToolRegistry):
    """AC-2: Validator returning False → 401 with WWW-Authenticate header."""

    async def validate_auth(api_key: str | None, bearer_token: str | None) -> bool:
        """Return False unconditionally."""
        return False

    app = FastAPI()
    mcp_router = create_mcp_router(registry, auth_validator=validate_auth)
    app.include_router(mcp_router, prefix="/mcp")
    client = TestClient(app)

    body = make_jsonrpc_request(method="ping")
    response = client.post(
        "/mcp",
        json=body,
        headers={
            "MCP-Protocol-Version": "2025-06-18",
            "Authorization": "Bearer some-token",
        },
    )

    assert response.status_code == 401
    assert "WWW-Authenticate" in response.headers


@pytest.mark.integration
def test_auth_validator_returns_dict_stores_auth_context():
    """AC-3: Validator returning dict → request.state.auth_context == returned dict.

    Uses a tool that reads request.state.auth_context and stores it in a shared
    dict. The tool provides an explicit input_schema to bypass Pydantic schema
    generation for the Request-typed parameter.
    """
    from fastapi import Request

    expected_context = {"user_id": "user-42", "scopes": ["read", "write"]}
    captured: dict[str, object] = {}

    registry = MCPToolRegistry()

    @registry.tool(input_schema={"type": "object", "properties": {}, "required": []})
    async def get_auth_context(request: Request) -> str:
        """Read auth_context from request.state for assertion."""
        ctx = getattr(request.state, "auth_context", None)
        captured["auth_context"] = ctx
        return "ok"

    async def validate_auth(api_key: str | None, bearer_token: str | None) -> dict[str, object]:
        """Return a dict as auth context."""
        return expected_context

    app = FastAPI()
    mcp_router = create_mcp_router(registry, auth_validator=validate_auth)
    app.include_router(mcp_router, prefix="/mcp")
    client = TestClient(app)

    body = make_jsonrpc_request(
        method="tools/call",
        params={"name": "get_auth_context", "arguments": {}},
    )
    response = client.post(
        "/mcp",
        json=body,
        headers={
            "MCP-Protocol-Version": "2025-06-18",
            "Authorization": "Bearer valid-token",
        },
    )

    assert response.status_code == 200
    assert captured.get("auth_context") == expected_context


@pytest.mark.integration
def test_auth_validator_returns_true_stores_auth_context_and_proceeds():
    """AC-4: Validator returning True → request proceeds and auth_context == True.

    Provides an explicit input_schema to bypass Pydantic schema generation
    for the Request-typed parameter.
    """
    from fastapi import Request

    captured: dict[str, object] = {}

    registry = MCPToolRegistry()

    @registry.tool(input_schema={"type": "object", "properties": {}, "required": []})
    async def get_auth_context(request: Request) -> str:
        """Capture auth_context from request.state."""
        ctx = getattr(request.state, "auth_context", None)
        captured["auth_context"] = ctx
        return "ok"

    async def validate_auth(api_key: str | None, bearer_token: str | None) -> bool:
        """Return True unconditionally."""
        return True

    app = FastAPI()
    mcp_router = create_mcp_router(registry, auth_validator=validate_auth)
    app.include_router(mcp_router, prefix="/mcp")
    client = TestClient(app)

    body = make_jsonrpc_request(
        method="tools/call",
        params={"name": "get_auth_context", "arguments": {}},
    )
    response = client.post(
        "/mcp",
        json=body,
        headers={
            "MCP-Protocol-Version": "2025-06-18",
            "X-API-Key": "any-key",
        },
    )

    assert response.status_code == 200
    assert captured.get("auth_context") is True


@pytest.mark.integration
def test_auth_validator_raises_exception_returns_internal_error(registry: MCPToolRegistry):
    """EC-2: Validator raises exception → internal error response.

    The router's outer except-Exception handler catches errors from
    auth_validator and returns HTTP 200 with a JSON-RPC internal error
    (code -32603). FastAPI does not propagate the exception to a 500 response
    because the exception is caught within the route handler.
    """

    async def validate_auth(api_key: str | None, bearer_token: str | None) -> bool:
        """Always raise an unexpected error."""
        raise RuntimeError("auth service unavailable")

    app = FastAPI()
    mcp_router = create_mcp_router(registry, auth_validator=validate_auth)
    app.include_router(mcp_router, prefix="/mcp")
    client = TestClient(app, raise_server_exceptions=False)

    body = make_jsonrpc_request(method="ping")
    response = client.post(
        "/mcp",
        json=body,
        headers={
            "MCP-Protocol-Version": "2025-06-18",
            "X-API-Key": "any-key",
        },
    )

    # The route handler catches the exception and returns a JSON-RPC internal error.
    # HTTP status is 200 per JSON-RPC 2.0 convention (errors are protocol-level).
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == -32603


# Header sanitization tests


@pytest.mark.integration
def test_sensitive_headers_redacted_in_logs(registry: MCPToolRegistry, caplog: pytest.LogCaptureFixture):
    """Test sensitive headers are redacted in debug logs."""
    import logging

    app = FastAPI()
    mcp_router = create_mcp_router(registry)
    app.include_router(mcp_router, prefix="/mcp")
    client = TestClient(app)

    request = make_jsonrpc_request(method="ping")

    # Set logging level to DEBUG to capture header logs
    with caplog.at_level(logging.DEBUG):
        response = client.post(
            "/mcp",
            json=request,
            headers={
                "MCP-Protocol-Version": "2025-06-18",
                "X-API-Key": "secret-api-key-12345",
                "Authorization": "Bearer secret-token-67890",
                "Cookie": "session=secret-cookie",
                "X-CSRF-Token": "csrf-secret",
                "User-Agent": "TestClient",
            },
        )

    assert response.status_code == 200

    # Find the debug log with request headers
    header_logs = [record for record in caplog.records if "Request headers:" in record.message]
    assert len(header_logs) > 0, "Should have logged request headers at DEBUG level"

    header_log = header_logs[0].message

    # Sensitive headers should be redacted
    assert "secret-api-key" not in header_log
    assert "secret-token" not in header_log
    assert "secret-cookie" not in header_log
    assert "csrf-secret" not in header_log
    assert "[REDACTED]" in header_log

    # Non-sensitive headers should not be redacted
    assert "TestClient" in header_log or "user-agent" in header_log.lower()
