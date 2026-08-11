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

import base64

from fastapi.responses import JSONResponse

_CURSOR_BYTE_LENGTH = 8


def encode_cursor(offset: int) -> str:
    """
    Encode an integer offset as an opaque base64 pagination cursor.

    The offset is packed into fixed-width big-endian bytes before base64
    encoding, so decoding the token yields raw binary data rather than the
    ASCII digits of the offset. Clients must treat the returned string as an
    opaque token and never attempt to interpret its contents.

    Exported as part of the public API so consumer test suites can construct
    a cursor for a known offset (e.g. to assert pagination behavior) without
    depending on a prior response. It is not intended for interpreting a
    cursor a real client sent back; ``decode_cursor`` does that job and is
    kept internal (absent from ``__all__``) for that reason.

    Args:
        offset: Zero-based index into the paginated list.

    Returns:
        URL-safe base64-encoded opaque cursor string.

    Raises:
        ValueError: If offset is negative.

    Example:
        >>> encode_cursor(0)
        'AAAAAAAAAAA='
    """
    if offset < 0:
        raise ValueError("offset must be non-negative")
    offset_bytes = offset.to_bytes(_CURSOR_BYTE_LENGTH, byteorder="big", signed=False)
    return base64.urlsafe_b64encode(offset_bytes).decode("ascii")


def decode_cursor(cursor: str) -> int:
    """
    Decode an opaque pagination cursor back to an integer offset.

    Internal use only: the returned offset MUST NOT be surfaced to the
    client. Callers that receive a client-supplied cursor should catch the
    raised error and map it to a protocol-level invalid-params response
    without echoing the decoded value.

    Args:
        cursor: Opaque cursor string previously returned by encode_cursor.

    Returns:
        Zero-based offset encoded in the cursor.

    Raises:
        ValueError: If the cursor is malformed or cannot be decoded.

    Example:
        >>> decode_cursor(encode_cursor(5))
        5
    """
    try:
        offset_bytes = base64.urlsafe_b64decode(cursor.encode("ascii"))
    except ValueError as e:
        raise ValueError("Malformed pagination cursor") from e

    if len(offset_bytes) != _CURSOR_BYTE_LENGTH:
        raise ValueError("Malformed pagination cursor")

    return int.from_bytes(offset_bytes, byteorder="big", signed=False)


def paginate(
    items: list,
    cursor: str | None,
    page_size: int = 100,
) -> tuple[list, str | None]:
    """
    Return a page of items and the cursor for the next page.

    Cursors are non-stable across list mutations: if the underlying list
    changes between calls, the returned page may repeat or skip items, but
    pagination never raises because of such mutations.

    Args:
        items: Full list to paginate.
        cursor: Opaque cursor from a previous call, or None for the first
            page.
        page_size: Maximum number of items to return in this page. Defaults
            to 100; configurable per call site.

    Returns:
        Tuple of (page_slice, next_cursor). next_cursor is None when this
        page reaches the end of items.

    Raises:
        ValueError: If cursor is not None and is malformed or undecodable, or
            if page_size is not positive.

    Example:
        >>> page, next_cursor = paginate(["a", "b", "c"], None, page_size=2)
        >>> page
        ['a', 'b']
        >>> next_cursor is None
        False
    """
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    offset = decode_cursor(cursor) if cursor is not None else 0
    page = items[offset : offset + page_size]
    next_offset = offset + len(page)
    next_cursor = encode_cursor(next_offset) if next_offset < len(items) else None
    return page, next_cursor


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
    error: dict[str, int | str | dict[str, object]] = {
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
