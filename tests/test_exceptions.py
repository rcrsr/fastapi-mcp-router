"""Unit tests for fastapi-mcp-router exceptions module.

Tests MCPError and ToolError exception classes for proper instantiation,
attribute access, inheritance, and behavior across various error scenarios.
"""

import pytest

from fastapi_mcp_router.exceptions import MCPError, ToolError

# MCPError tests


@pytest.mark.unit
def test_mcperror_basic_instantiation():
    """Test MCPError can be instantiated with code and message."""
    error = MCPError(code=-32601, message="Method not found")

    assert error.code == -32601
    assert error.message == "Method not found"
    assert error.data is None
    assert isinstance(error, Exception)


@pytest.mark.unit
def test_mcperror_with_data():
    """Test MCPError can be instantiated with code, message, and data."""
    error_data: dict[str, object] = {
        "received": ["arguments"],
        "required": ["name", "arguments"],
    }
    error = MCPError(
        code=-32602,
        message="Missing required parameter: name",
        data=error_data,
    )

    assert error.code == -32602
    assert error.message == "Missing required parameter: name"
    assert error.data is not None
    assert error.data == error_data
    assert error.data["received"] == ["arguments"]
    assert error.data["required"] == ["name", "arguments"]


@pytest.mark.unit
def test_mcperror_inherits_from_exception():
    """Test MCPError properly inherits from Exception."""
    error = MCPError(code=-32600, message="Invalid Request")

    assert isinstance(error, Exception)
    assert isinstance(error, MCPError)
    assert str(error) == "Invalid Request"


@pytest.mark.unit
def test_mcperror_attributes_accessible():
    """Test MCPError attributes are accessible after instantiation."""
    error = MCPError(
        code=-32603,
        message="Internal error",
        data={"context": "test"},
    )

    assert error.code == -32603
    assert error.message == "Internal error"
    assert error.data is not None
    assert error.data["context"] == "test"


@pytest.mark.unit
def test_mcperror_parse_error():
    """Test MCPError with parse error code (-32700)."""
    error = MCPError(code=-32700, message="Parse error: invalid JSON")

    assert error.code == -32700
    assert "Parse error" in error.message
    assert error.data is None


@pytest.mark.unit
def test_mcperror_invalid_request():
    """Test MCPError with invalid request code (-32600)."""
    error = MCPError(code=-32600, message="Invalid Request: malformed")

    assert error.code == -32600
    assert "Invalid Request" in error.message


@pytest.mark.unit
def test_mcperror_method_not_found():
    """Test MCPError with method not found code (-32601)."""
    error = MCPError(
        code=-32601,
        message="Method not found: unknown/method",
        data={"method": "unknown/method"},
    )

    assert error.code == -32601
    assert "Method not found" in error.message
    assert error.data is not None
    assert error.data["method"] == "unknown/method"


@pytest.mark.unit
def test_mcperror_invalid_params():
    """Test MCPError with invalid params code (-32602)."""
    error = MCPError(
        code=-32602,
        message="Invalid params: missing name",
        data={"missing": ["name"]},
    )

    assert error.code == -32602
    assert "Invalid params" in error.message
    assert error.data is not None
    assert error.data["missing"] == ["name"]


@pytest.mark.unit
def test_mcperror_internal_error():
    """Test MCPError with internal error code (-32603)."""
    error = MCPError(
        code=-32603,
        message="Internal error: database failure",
        data={"error": "connection timeout"},
    )

    assert error.code == -32603
    assert "Internal error" in error.message
    assert error.data is not None
    assert error.data["error"] == "connection timeout"


@pytest.mark.unit
def test_mcperror_server_defined_error_min():
    """Test MCPError with minimum server-defined error code (-32099)."""
    error = MCPError(
        code=-32099,
        message="Server error: custom error",
    )

    assert error.code == -32099
    assert -32099 >= -32099
    assert -32099 <= -32000


@pytest.mark.unit
def test_mcperror_server_defined_error_max():
    """Test MCPError with maximum server-defined error code (-32000)."""
    error = MCPError(
        code=-32000,
        message="Server error: database connection failed",
        data={"host": "db.example.com", "error": "timeout"},
    )

    assert error.code == -32000
    assert -32000 >= -32099
    assert -32000 <= -32000
    assert error.data is not None
    assert error.data["host"] == "db.example.com"


@pytest.mark.unit
def test_mcperror_server_defined_error_mid():
    """Test MCPError with mid-range server-defined error code (-32050)."""
    error = MCPError(
        code=-32050,
        message="Server error: configuration invalid",
        data={"config": "settings.json"},
    )

    assert error.code == -32050
    assert -32050 >= -32099
    assert -32050 <= -32000


@pytest.mark.unit
def test_mcperror_data_default_none():
    """Test MCPError data parameter defaults to None."""
    error = MCPError(code=-32601, message="Method not found")

    assert error.data is None


@pytest.mark.unit
def test_mcperror_complex_data_structure():
    """Test MCPError with complex nested data structure."""
    complex_data: dict[str, object] = {
        "host": "db.example.com",
        "error": "timeout",
        "retry_info": {
            "max_retries": 3,
            "backoff": [1, 2, 4],
        },
        "metadata": {
            "timestamp": "2025-11-09T12:00:00Z",
            "request_id": "req-123",
        },
    }
    error = MCPError(
        code=-32000,
        message="Database connection failed",
        data=complex_data,
    )

    assert error.data is not None
    assert error.data["host"] == "db.example.com"
    retry_info = error.data["retry_info"]
    assert isinstance(retry_info, dict)
    assert retry_info["max_retries"] == 3
    assert retry_info["backoff"] == [1, 2, 4]
    metadata = error.data["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["request_id"] == "req-123"


# ToolError tests


@pytest.mark.unit
def test_toolerror_basic_instantiation():
    """Test ToolError can be instantiated with message."""
    error = ToolError(message="File not found: /path/to/file.txt")

    assert error.message == "File not found: /path/to/file.txt"
    assert error.details is None
    assert isinstance(error, Exception)


@pytest.mark.unit
def test_toolerror_with_details():
    """Test ToolError can be instantiated with message and details."""
    error_details: dict[str, object] = {"path": "/path/to/file.txt", "exists": False}
    error = ToolError(
        message="File not found: /path/to/file.txt",
        details=error_details,
    )

    assert error.message == "File not found: /path/to/file.txt"
    assert error.details is not None
    assert error.details == error_details
    assert error.details["path"] == "/path/to/file.txt"
    assert error.details["exists"] is False


@pytest.mark.unit
def test_toolerror_inherits_from_exception():
    """Test ToolError properly inherits from Exception."""
    error = ToolError(message="Tool execution failed")

    assert isinstance(error, Exception)
    assert isinstance(error, ToolError)
    assert str(error) == "Tool execution failed"


@pytest.mark.unit
def test_toolerror_attributes_accessible():
    """Test ToolError attributes are accessible after instantiation."""
    error = ToolError(
        message="Validation failed",
        details={"field": "email", "value": "invalid"},
    )

    assert error.message == "Validation failed"
    assert error.details is not None
    assert error.details["field"] == "email"
    assert error.details["value"] == "invalid"


@pytest.mark.unit
def test_toolerror_details_default_none():
    """Test ToolError details parameter defaults to None."""
    error = ToolError(message="Resource not found")

    assert error.details is None


@pytest.mark.unit
def test_toolerror_resource_not_found_scenario():
    """Test ToolError for resource not found scenario."""
    error = ToolError(
        message="File not found: /data/report.pdf",
        details={
            "path": "/data/report.pdf",
            "exists": False,
            "parent_dir": "/data",
        },
    )

    assert "File not found" in error.message
    assert error.details is not None
    assert error.details["path"] == "/data/report.pdf"
    assert error.details["exists"] is False
    assert error.details["parent_dir"] == "/data"


@pytest.mark.unit
def test_toolerror_validation_failure_scenario():
    """Test ToolError for validation failure scenario."""
    error = ToolError(
        message="Invalid email format: not-an-email",
        details={
            "field": "email",
            "value": "not-an-email",
            "pattern": r"^[\w\.-]+@[\w\.-]+\.\w+$",
        },
    )

    assert "Invalid email format" in error.message
    assert error.details is not None
    assert error.details["field"] == "email"
    assert error.details["value"] == "not-an-email"
    assert "pattern" in error.details


@pytest.mark.unit
def test_toolerror_rate_limit_scenario():
    """Test ToolError for API rate limit scenario."""
    error = ToolError(
        message="GitHub API rate limit exceeded. Retry after 3600 seconds.",
        details={
            "service": "github",
            "limit": 5000,
            "remaining": 0,
            "reset_at": "2025-11-09T12:00:00Z",
        },
    )

    assert "rate limit exceeded" in error.message
    assert error.details is not None
    assert error.details["service"] == "github"
    assert error.details["limit"] == 5000
    assert error.details["remaining"] == 0
    assert error.details["reset_at"] == "2025-11-09T12:00:00Z"


@pytest.mark.unit
def test_toolerror_permission_denied_scenario():
    """Test ToolError for permission denied scenario."""
    error = ToolError(
        message="Access denied: user lacks write permission for repository",
        details={
            "user": "user123",
            "repository": "org/repo",
            "required_permission": "write",
            "current_permission": "read",
        },
    )

    assert "Access denied" in error.message
    assert error.details is not None
    assert error.details["user"] == "user123"
    assert error.details["repository"] == "org/repo"
    assert error.details["required_permission"] == "write"
    assert error.details["current_permission"] == "read"


@pytest.mark.unit
def test_toolerror_external_service_error_scenario():
    """Test ToolError for external service error scenario."""
    error = ToolError(
        message="External API request failed with status 503",
        details={
            "service": "payment_gateway",
            "endpoint": "https://api.payment.com/charge",
            "status_code": 503,
            "error": "Service Unavailable",
        },
    )

    assert "External API request failed" in error.message
    assert error.details is not None
    assert error.details["service"] == "payment_gateway"
    assert error.details["status_code"] == 503
    assert error.details["error"] == "Service Unavailable"


@pytest.mark.unit
def test_toolerror_complex_details_structure():
    """Test ToolError with complex nested details structure."""
    complex_details: dict[str, object] = {
        "operation": "database_query",
        "query": "SELECT * FROM users WHERE id = ?",
        "parameters": {"id": "user-123"},
        "error": {
            "type": "ConnectionError",
            "message": "Connection timeout",
            "stack_trace": ["line 1", "line 2", "line 3"],
        },
        "retry_info": {
            "attempts": 3,
            "backoff_ms": [100, 200, 400],
        },
    }
    error = ToolError(
        message="Database query failed after 3 retries",
        details=complex_details,
    )

    assert error.details is not None
    assert error.details["operation"] == "database_query"
    params = error.details["parameters"]
    assert isinstance(params, dict)
    assert params["id"] == "user-123"
    error_info = error.details["error"]
    assert isinstance(error_info, dict)
    assert error_info["type"] == "ConnectionError"
    assert error_info["stack_trace"] == ["line 1", "line 2", "line 3"]
    retry_info = error.details["retry_info"]
    assert isinstance(retry_info, dict)
    assert retry_info["attempts"] == 3


@pytest.mark.unit
def test_toolerror_empty_details():
    """Test ToolError with empty details dictionary."""
    error = ToolError(message="Error occurred", details={})

    assert error.message == "Error occurred"
    assert error.details is not None
    assert error.details == {}
    assert len(error.details) == 0


# Cross-exception tests


@pytest.mark.unit
def test_mcperror_and_toolerror_are_distinct():
    """Test MCPError and ToolError are distinct exception types."""
    mcp_error = MCPError(code=-32601, message="Method not found")
    tool_error = ToolError(message="Tool execution failed")

    assert type(mcp_error) is not type(tool_error)
    assert not isinstance(mcp_error, ToolError)
    assert not isinstance(tool_error, MCPError)
    assert isinstance(mcp_error, Exception)
    assert isinstance(tool_error, Exception)


@pytest.mark.unit
def test_mcperror_can_be_raised():
    """Test MCPError can be raised and caught."""
    with pytest.raises(MCPError) as exc_info:
        raise MCPError(code=-32601, message="Method not found")

    assert exc_info.value.code == -32601
    assert exc_info.value.message == "Method not found"


@pytest.mark.unit
def test_toolerror_can_be_raised():
    """Test ToolError can be raised and caught."""
    with pytest.raises(ToolError) as exc_info:
        raise ToolError(message="File not found")

    assert exc_info.value.message == "File not found"


@pytest.mark.unit
def test_mcperror_can_be_caught_as_exception():
    """Test MCPError can be caught as generic Exception."""
    try:
        raise MCPError(code=-32600, message="Invalid Request")
    except MCPError as e:
        assert isinstance(e, MCPError)
        assert e.code == -32600


@pytest.mark.unit
def test_toolerror_can_be_caught_as_exception():
    """Test ToolError can be caught as generic Exception."""
    try:
        raise ToolError(message="Validation failed")
    except ToolError as e:
        assert isinstance(e, ToolError)
        assert e.message == "Validation failed"
