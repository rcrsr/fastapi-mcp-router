"""
Unit tests for JSON-RPC 2.0 protocol response formatting.

Tests the json_rpc_response and json_rpc_error helper functions to ensure
compliance with JSON-RPC 2.0 specification.
"""

import base64
import json

import pytest

from fastapi_mcp_router.protocol import (
    decode_cursor,
    encode_cursor,
    json_rpc_error,
    json_rpc_response,
    paginate,
)

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


# ============================================================================
# Tests for encode_cursor() / decode_cursor()
# ============================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "offset",
    [0, 1, 42, 100, 999_999, 2**32, 2**63 - 1],
)
def test_decode_cursor_round_trips_encode_cursor(offset: int):
    """Test decode_cursor recovers the exact offset passed to encode_cursor."""
    cursor = encode_cursor(offset)

    assert decode_cursor(cursor) == offset


@pytest.mark.unit
def test_encode_cursor_returns_string():
    """Test encode_cursor returns a string token."""
    cursor = encode_cursor(5)

    assert isinstance(cursor, str)


@pytest.mark.unit
@pytest.mark.parametrize("offset", [0, 5, 42, 12345])
def test_encode_cursor_base64_decode_hides_offset_digits(offset: int):
    """Test base64-decoding a cursor does not expose the offset as plaintext digits."""
    cursor = encode_cursor(offset)

    decoded_bytes = base64.urlsafe_b64decode(cursor.encode("ascii"))

    assert str(offset).encode("ascii") not in decoded_bytes


@pytest.mark.unit
def test_encode_cursor_base64_decode_hides_item_keys():
    """Test base64-decoding a cursor exposes no item-related plaintext."""
    items = ["secret-item-key", "another-key"]
    _, cursor = paginate(items, None, page_size=1)

    assert cursor is not None
    decoded_bytes = base64.urlsafe_b64decode(cursor.encode("ascii"))

    assert b"secret-item-key" not in decoded_bytes
    assert b"another-key" not in decoded_bytes


@pytest.mark.unit
def test_decode_cursor_raises_on_malformed_input():
    """Test decode_cursor raises for a cursor string that is not valid base64."""
    with pytest.raises(ValueError):
        decode_cursor("not-a-valid-cursor!!!")


@pytest.mark.unit
def test_decode_cursor_error_does_not_echo_offset():
    """Test the exception raised for a malformed cursor contains no decoded offset."""
    malformed_cursor = base64.urlsafe_b64encode(b"12345").decode("ascii")

    with pytest.raises(ValueError) as exc_info:
        decode_cursor(malformed_cursor)

    assert "12345" not in str(exc_info.value)


@pytest.mark.unit
def test_encode_cursor_raises_value_error_on_negative_offset():
    """Test encode_cursor raises ValueError, not OverflowError, for a negative offset."""
    with pytest.raises(ValueError):
        encode_cursor(-1)


# ============================================================================
# Tests for paginate()
# ============================================================================


@pytest.mark.unit
def test_paginate_empty_list_returns_empty_page():
    """Test paginate on an empty list returns an empty page and no next cursor."""
    page, next_cursor = paginate([], None)

    assert page == []
    assert next_cursor is None


@pytest.mark.unit
def test_paginate_exact_page_size_has_no_next_cursor():
    """Test a list exactly page_size long yields no trailing empty page."""
    items = list(range(100))

    page, next_cursor = paginate(items, None)

    assert page == items
    assert next_cursor is None


@pytest.mark.unit
def test_paginate_multi_page_list_returns_next_cursor():
    """Test a list larger than one page returns a usable next cursor."""
    items = list(range(150))

    first_page, next_cursor = paginate(items, None)
    assert first_page == items[:100]
    assert next_cursor is not None

    second_page, final_cursor = paginate(items, next_cursor)
    assert second_page == items[100:150]
    assert final_cursor is None


@pytest.mark.unit
def test_paginate_default_page_size_is_100():
    """Test paginate defaults to a page size of 100 when unspecified."""
    items = list(range(250))

    page, _ = paginate(items, None)

    assert len(page) == 100


@pytest.mark.unit
def test_paginate_page_size_override():
    """Test a supplied page_size overrides the default of 100."""
    items = list(range(250))

    page, next_cursor = paginate(items, None, page_size=10)

    assert len(page) == 10
    assert next_cursor is not None


@pytest.mark.unit
def test_paginate_mutation_between_calls_skips_without_error():
    """Test items removed between calls cause skipped items but never raise."""
    items = list(range(10))

    first_page, next_cursor = paginate(items, None, page_size=5)
    assert first_page == [0, 1, 2, 3, 4]

    del items[0:5]

    second_page, final_cursor = paginate(items, next_cursor, page_size=5)

    assert second_page == []
    assert final_cursor is None


@pytest.mark.unit
def test_paginate_mutation_between_calls_repeats_without_error():
    """Test items inserted between calls cause repeated items but never raise."""
    items = list(range(10))

    first_page, next_cursor = paginate(items, None, page_size=5)
    assert first_page == [0, 1, 2, 3, 4]

    items.insert(0, -1)

    second_page, final_cursor = paginate(items, next_cursor, page_size=5)

    assert second_page == [4, 5, 6, 7, 8]
    assert final_cursor is not None


@pytest.mark.unit
@pytest.mark.parametrize("page_size", [0, -1, -10])
def test_paginate_raises_value_error_on_non_positive_page_size(page_size: int):
    """Test paginate rejects a zero or negative page_size instead of stalling or slicing backward."""
    with pytest.raises(ValueError):
        paginate(list(range(10)), None, page_size=page_size)
