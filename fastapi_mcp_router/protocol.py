"""
JSON-RPC 2.0 response formatting utilities for MCP protocol.

This module provides helper functions to create JSON-RPC 2.0 compliant response
messages. JSON-RPC 2.0 is a stateless, light-weight remote procedure call (RPC)
protocol that uses JSON as data format.

Key protocol rules:
- All responses include "jsonrpc": "2.0" field
- Successful responses include "result" field
- Error responses include "error" object with "code" and "message"
- Both success and error responses use HTTP 200 status code
  (errors are at JSON-RPC protocol level, not HTTP transport level)
"""

from fastapi.responses import JSONResponse


def json_rpc_response(request_id: object, result: dict[str, object]) -> JSONResponse:
    """
    Create JSON-RPC 2.0 success response.

    Formats a successful JSON-RPC response according to the JSON-RPC 2.0
    specification. The response includes the protocol version, request ID,
    and result data.

    Args:
        request_id: Request identifier from original JSON-RPC request. Can be
            string, number, or null. Used to match responses with requests.
        result: Result data to return. Must be a dictionary containing the
            method's return value.

    Returns:
        FastAPI JSONResponse with status 200 and JSON-RPC formatted body.

    Examples:
        >>> response = json_rpc_response("req-123", {"status": "success"})
        >>> # Returns JSONResponse with body:
        >>> # {
        >>> #   "jsonrpc": "2.0",
        >>> #   "id": "req-123",
        >>> #   "result": {"status": "success"}
        >>> # }

        >>> response = json_rpc_response(42, {"tools": [], "count": 0})
        >>> # Returns JSONResponse with body:
        >>> # {
        >>> #   "jsonrpc": "2.0",
        >>> #   "id": 42,
        >>> #   "result": {"tools": [], "count": 0}
        >>> # }
    """
    return JSONResponse(
        content={
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result,
        }
    )


def json_rpc_error(
    request_id: object,
    code: int,
    message: str,
    data: dict[str, object] | None = None,
) -> JSONResponse:
    """
    Create JSON-RPC 2.0 error response.

    Formats an error response according to the JSON-RPC 2.0 specification.
    The response uses HTTP 200 status code because the error is at the
    JSON-RPC protocol level, not the HTTP transport level.

    Args:
        request_id: Request identifier from original JSON-RPC request. Can be
            string, number, or null.
        code: Error code integer. Standard JSON-RPC error codes:
            -32700: Parse error (invalid JSON)
            -32600: Invalid request (malformed JSON-RPC)
            -32601: Method not found
            -32602: Invalid params
            -32603: Internal error
            -32000 to -32099: Server error (implementation defined)
        message: Human-readable error description string.
        data: Optional additional error information. Only included in response
            if not None. Can contain debug details, stack traces, etc.

    Returns:
        FastAPI JSONResponse with status 200 and JSON-RPC formatted error body.

    Examples:
        >>> response = json_rpc_error("req-123", -32601, "Method not found")
        >>> # Returns JSONResponse with status 200 and body:
        >>> # {
        >>> #   "jsonrpc": "2.0",
        >>> #   "id": "req-123",
        >>> #   "error": {
        >>> #     "code": -32601,
        >>> #     "message": "Method not found"
        >>> #   }
        >>> # }

        >>> response = json_rpc_error(
        ...     42,
        ...     -32602,
        ...     "Invalid params",
        ...     {"expected": "string", "received": "number"}
        ... )
        >>> # Returns JSONResponse with status 200 and body:
        >>> # {
        >>> #   "jsonrpc": "2.0",
        >>> #   "id": 42,
        >>> #   "error": {
        >>> #     "code": -32602,
        >>> #     "message": "Invalid params",
        >>> #     "data": {"expected": "string", "received": "number"}
        >>> #   }
        >>> # }
    """
    error = {
        "code": code,
        "message": message,
    }
    if data is not None:
        error["data"] = data

    return JSONResponse(
        status_code=200,
        content={
            "jsonrpc": "2.0",
            "id": request_id,
            "error": error,
        },
    )
