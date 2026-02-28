"""
Unit tests for JSON-RPC 2.0 protocol response formatting.

Tests the json_rpc_response and json_rpc_error helper functions to ensure
compliance with JSON-RPC 2.0 specification.
"""

import json

import pytest

from fastapi_mcp_router.protocol import json_rpc_error, json_rpc_response

# ============================================================================
# Tests for json_rpc_response()
# ============================================================================


@pytest.mark.unit
def test_json_rpc_response_string_id():
    """Test successful response with string request ID."""
    response = json_rpc_response("req-123", {"result": "success"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"

    body = json.loads(response.body)
    assert body["jsonrpc"] == "2.0"
    assert body["id"] == "req-123"
    assert body["result"] == {"result": "success"}


@pytest.mark.unit
def test_json_rpc_response_integer_id():
    """Test successful response with integer request ID."""
    response = json_rpc_response(42, {"tools": ["tool1", "tool2"]})

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"

    body = json.loads(response.body)
    assert body["jsonrpc"] == "2.0"
    assert body["id"] == 42
    assert body["result"] == {"tools": ["tool1", "tool2"]}


@pytest.mark.unit
def test_json_rpc_response_null_id():
    """Test successful response with null request ID (notifications)."""
    response = json_rpc_response(None, {"status": "notification_sent"})

    assert response.status_code == 200

    body = json.loads(response.body)
    assert body["jsonrpc"] == "2.0"
    assert body["id"] is None
    assert body["result"] == {"status": "notification_sent"}


@pytest.mark.unit
def test_json_rpc_response_has_correct_structure():
    """Test response has correct structure with jsonrpc, id, and result fields."""
    response = json_rpc_response("test-id", {"data": "value"})

    body = json.loads(response.body)
    assert set(body.keys()) == {"jsonrpc", "id", "result"}
    assert "error" not in body


@pytest.mark.unit
def test_json_rpc_response_http_200_status():
    """Test response has HTTP 200 status code."""
    response = json_rpc_response("id", {"key": "value"})

    assert response.status_code == 200


@pytest.mark.unit
def test_json_rpc_response_content_type_json():
    """Test response content-type is application/json."""
    response = json_rpc_response("id", {"data": "test"})

    assert "application/json" in response.headers["content-type"]


@pytest.mark.unit
def test_json_rpc_response_result_field_contains_data():
    """Test result field contains provided data."""
    result_data = {
        "user_id": "123",
        "username": "testuser",
        "email": "test@example.com",
    }
    response = json_rpc_response("req-001", result_data)

    body = json.loads(response.body)
    assert body["result"] == result_data
    assert body["result"]["user_id"] == "123"
    assert body["result"]["username"] == "testuser"


@pytest.mark.unit
def test_json_rpc_response_empty_result_dict():
    """Test response with empty result dictionary."""
    response = json_rpc_response("empty", {})

    body = json.loads(response.body)
    assert body["result"] == {}
    assert isinstance(body["result"], dict)


@pytest.mark.unit
def test_json_rpc_response_complex_nested_result():
    """Test response with complex nested result dictionary."""
    complex_result = {
        "tools": [
            {
                "name": "calculator",
                "description": "Performs calculations",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string"},
                        "numbers": {"type": "array", "items": {"type": "number"}},
                    },
                },
            }
        ],
        "metadata": {
            "count": 1,
            "timestamp": "2025-11-09T12:00:00Z",
            "nested": {"level": 2, "data": [1, 2, 3]},
        },
    }
    response = json_rpc_response("complex-123", complex_result)

    body = json.loads(response.body)
    assert body["result"] == complex_result
    assert body["result"]["tools"][0]["name"] == "calculator"
    assert body["result"]["metadata"]["nested"]["level"] == 2


# ============================================================================
# Tests for json_rpc_error()
# ============================================================================


@pytest.mark.unit
def test_json_rpc_error_string_id():
    """Test error response with string request ID."""
    response = json_rpc_error("req-456", -32601, "Method not found")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"

    body = json.loads(response.body)
    assert body["jsonrpc"] == "2.0"
    assert body["id"] == "req-456"
    assert body["error"]["code"] == -32601
    assert body["error"]["message"] == "Method not found"


@pytest.mark.unit
def test_json_rpc_error_integer_id():
    """Test error response with integer request ID."""
    response = json_rpc_error(99, -32602, "Invalid params")

    assert response.status_code == 200

    body = json.loads(response.body)
    assert body["jsonrpc"] == "2.0"
    assert body["id"] == 99
    assert body["error"]["code"] == -32602


@pytest.mark.unit
def test_json_rpc_error_null_id():
    """Test error response with null request ID."""
    response = json_rpc_error(None, -32603, "Internal error")

    body = json.loads(response.body)
    assert body["id"] is None
    assert body["error"]["code"] == -32603


@pytest.mark.unit
def test_json_rpc_error_has_correct_structure():
    """Test error response has correct structure with jsonrpc, id, and error fields."""
    response = json_rpc_error("id", -32600, "Invalid Request")

    body = json.loads(response.body)
    assert set(body.keys()) == {"jsonrpc", "id", "error"}
    assert "result" not in body


@pytest.mark.unit
def test_json_rpc_error_has_code_and_message():
    """Test error object has code and message fields."""
    response = json_rpc_error("test", -32601, "Method not found")

    body = json.loads(response.body)
    assert "code" in body["error"]
    assert "message" in body["error"]
    assert body["error"]["code"] == -32601
    assert body["error"]["message"] == "Method not found"


@pytest.mark.unit
def test_json_rpc_error_http_200_status():
    """Test error response has HTTP 200 status (per JSON-RPC spec)."""
    response = json_rpc_error("id", -32603, "Internal error")

    assert response.status_code == 200


@pytest.mark.unit
def test_json_rpc_error_content_type_json():
    """Test error response content-type is application/json."""
    response = json_rpc_error("id", -32600, "Invalid Request")

    assert "application/json" in response.headers["content-type"]


@pytest.mark.unit
def test_json_rpc_error_with_data_field():
    """Test error response includes data field when provided."""
    error_data = {"expected": "string", "received": "number"}
    response = json_rpc_error("req", -32602, "Invalid params", data=error_data)

    body = json.loads(response.body)
    assert "data" in body["error"]
    assert body["error"]["data"] == error_data
    assert body["error"]["data"]["expected"] == "string"


@pytest.mark.unit
def test_json_rpc_error_without_data_field():
    """Test error response excludes data field when data is None."""
    response = json_rpc_error("req-789", -32601, "Method not found", data=None)

    body = json.loads(response.body)
    assert "data" not in body["error"]
    assert set(body["error"].keys()) == {"code", "message"}


@pytest.mark.unit
def test_json_rpc_error_parse_error_code():
    """Test error with parse error code (-32700)."""
    response = json_rpc_error(None, -32700, "Parse error")

    body = json.loads(response.body)
    assert body["error"]["code"] == -32700
    assert body["error"]["message"] == "Parse error"


@pytest.mark.unit
def test_json_rpc_error_invalid_request_code():
    """Test error with invalid request code (-32600)."""
    response = json_rpc_error("req", -32600, "Invalid Request")

    body = json.loads(response.body)
    assert body["error"]["code"] == -32600


@pytest.mark.unit
def test_json_rpc_error_method_not_found_code():
    """Test error with method not found code (-32601)."""
    response = json_rpc_error("req", -32601, "Method not found")

    body = json.loads(response.body)
    assert body["error"]["code"] == -32601


@pytest.mark.unit
def test_json_rpc_error_invalid_params_code():
    """Test error with invalid params code (-32602)."""
    response = json_rpc_error("req", -32602, "Invalid params")

    body = json.loads(response.body)
    assert body["error"]["code"] == -32602


@pytest.mark.unit
def test_json_rpc_error_internal_error_code():
    """Test error with internal error code (-32603)."""
    response = json_rpc_error("req", -32603, "Internal error")

    body = json.loads(response.body)
    assert body["error"]["code"] == -32603


@pytest.mark.unit
def test_json_rpc_error_server_error_code_range_start():
    """Test error with server-defined error code at range start (-32000)."""
    response = json_rpc_error("req", -32000, "Server error")

    body = json.loads(response.body)
    assert body["error"]["code"] == -32000


@pytest.mark.unit
def test_json_rpc_error_server_error_code_range_middle():
    """Test error with server-defined error code in range middle (-32050)."""
    response = json_rpc_error("req", -32050, "Custom server error")

    body = json.loads(response.body)
    assert body["error"]["code"] == -32050


@pytest.mark.unit
def test_json_rpc_error_server_error_code_range_end():
    """Test error with server-defined error code at range end (-32099)."""
    response = json_rpc_error("req", -32099, "Another server error")

    body = json.loads(response.body)
    assert body["error"]["code"] == -32099


@pytest.mark.unit
def test_json_rpc_error_complex_nested_data():
    """Test error with complex nested data structure."""
    complex_data = {
        "validation_errors": [
            {
                "field": "email",
                "error": "Invalid format",
                "received": "not-an-email",
            },
            {
                "field": "age",
                "error": "Out of range",
                "received": -5,
                "constraints": {"min": 0, "max": 150},
            },
        ],
        "request_info": {
            "path": "/api/users",
            "method": "POST",
            "timestamp": "2025-11-09T12:00:00Z",
        },
    }
    response = json_rpc_error("complex-error", -32602, "Validation failed", data=complex_data)

    body = json.loads(response.body)
    assert body["error"]["data"] == complex_data
    assert len(body["error"]["data"]["validation_errors"]) == 2
    assert body["error"]["data"]["validation_errors"][0]["field"] == "email"


# ============================================================================
# Tests for JSON-RPC 2.0 Specification Compliance
# ============================================================================


@pytest.mark.unit
def test_response_and_error_both_use_jsonrpc_2_0():
    """Test both response types include jsonrpc 2.0 field."""
    success_response = json_rpc_response("id", {})
    error_response = json_rpc_error("id", -32600, "Error")

    success_body = json.loads(success_response.body)
    error_body = json.loads(error_response.body)

    assert success_body["jsonrpc"] == "2.0"
    assert error_body["jsonrpc"] == "2.0"


@pytest.mark.unit
def test_response_and_error_mutually_exclusive_fields():
    """Test success has result field and error has error field, but not both."""
    success_response = json_rpc_response("id", {"data": "test"})
    error_response = json_rpc_error("id", -32600, "Error")

    success_body = json.loads(success_response.body)
    error_body = json.loads(error_response.body)

    assert "result" in success_body
    assert "error" not in success_body

    assert "error" in error_body
    assert "result" not in error_body


@pytest.mark.unit
@pytest.mark.parametrize(
    "request_id,result_data",
    [
        ("string-id", {"key": "value"}),
        (123, {"number": 456}),
        (0, {"zero": True}),
        (None, {"notification": True}),
        ("", {"empty_string_id": True}),
    ],
)
def test_json_rpc_response_various_id_types(request_id: object, result_data: dict[str, object]):
    """Test json_rpc_response with various request ID types."""
    response = json_rpc_response(request_id, result_data)

    body = json.loads(response.body)
    assert body["id"] == request_id
    assert body["result"] == result_data


@pytest.mark.unit
@pytest.mark.parametrize(
    "code,message",
    [
        (-32700, "Parse error"),
        (-32600, "Invalid Request"),
        (-32601, "Method not found"),
        (-32602, "Invalid params"),
        (-32603, "Internal error"),
        (-32000, "Server error"),
        (-32099, "Server error end"),
    ],
)
def test_json_rpc_error_standard_codes(code: int, message: str):
    """Test json_rpc_error with standard JSON-RPC error codes."""
    response = json_rpc_error("req", code, message)

    body = json.loads(response.body)
    assert body["error"]["code"] == code
    assert body["error"]["message"] == message
