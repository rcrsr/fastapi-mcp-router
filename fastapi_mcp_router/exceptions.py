"""Exception classes for MCP protocol and tool execution errors.

This module defines two exception types for different error scenarios:

1. MCPError - JSON-RPC protocol-level errors (unrecoverable)
   - Use for: Protocol violations, invalid requests, method not found
   - Result: JSON-RPC error response with error code and message
   - LLM cannot recover - request terminates with error

2. ToolError - Business logic errors (recoverable)
   - Use for: Tool execution failures, validation errors, resource not found
   - Result: MCP tool response with isError: true
   - LLM receives error context and can potentially recover

Guidelines:
- Raise MCPError for protocol violations (-32700 to -32603)
- Raise ToolError for business logic failures that LLM should see
- Use standard JSON-RPC error codes when applicable
"""


class MCPError(Exception):
    """MCP protocol error for JSON-RPC level failures.

    This exception represents unrecoverable protocol-level errors that violate
    the JSON-RPC specification or MCP protocol requirements. When raised, the
    error terminates request processing and returns a JSON-RPC error response.

    Standard JSON-RPC error codes:
    - -32700: Parse error (invalid JSON)
    - -32600: Invalid Request (malformed request object)
    - -32601: Method not found
    - -32602: Invalid params
    - -32603: Internal error
    - -32000 to -32099: Server-defined errors

    Attributes:
        code: JSON-RPC error code (integer)
        message: Human-readable error description
        data: Optional additional error context

    Example:
        >>> # Method not found
        >>> raise MCPError(
        ...     code=-32601,
        ...     message="Method not found: unknown/method"
        ... )

        >>> # Invalid parameters with details
        >>> raise MCPError(
        ...     code=-32602,
        ...     message="Missing required parameter: name",
        ...     data={"received": ["arguments"], "required": ["name", "arguments"]}
        ... )

        >>> # Server-specific error
        >>> raise MCPError(
        ...     code=-32000,
        ...     message="Database connection failed",
        ...     data={"host": "db.example.com", "error": "timeout"}
        ... )
    """

    def __init__(
        self,
        code: int,
        message: str,
        data: dict[str, object] | None = None,
    ) -> None:
        """Initialize MCP protocol error.

        Args:
            code: JSON-RPC error code (standard or server-defined)
            message: Human-readable error description
            data: Optional additional error context
        """
        self.code = code
        self.message = message
        self.data = data
        super().__init__(message)


class ToolError(Exception):
    """Tool execution error for business logic failures.

    This exception represents recoverable business logic errors during tool
    execution. When raised, the tool returns a result with isError: true,
    allowing the LLM to see the error context and potentially recover through
    retry, alternative approaches, or error handling logic.

    Use ToolError instead of MCPError when:
    - Resource not found (file, record, entity)
    - Business validation fails (invalid state, constraints violated)
    - External service errors (API failures, timeouts)
    - Permission denied (user lacks access)

    The LLM receives the error message and optional details, enabling it to:
    - Retry with different parameters
    - Request missing information from user
    - Try alternative tools or approaches
    - Provide meaningful error explanation to user

    Attributes:
        message: Human-readable error description for LLM
        details: Optional structured error context

    Example:
        >>> # Resource not found
        >>> raise ToolError(
        ...     message="File not found: /path/to/file.txt",
        ...     details={"path": "/path/to/file.txt", "exists": False}
        ... )

        >>> # Validation failure
        >>> raise ToolError(
        ...     message="Invalid email format: not-an-email",
        ...     details={"field": "email", "value": "not-an-email", "pattern": r"^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$"}
        ... )

        >>> # External service error
        >>> raise ToolError(
        ...     message="GitHub API rate limit exceeded. Retry after 3600 seconds.",
        ...     details={"service": "github", "limit": 5000, "remaining": 0, "reset_at": "2025-11-09T12:00:00Z"}
        ... )

        >>> # Permission denied
        >>> raise ToolError(
        ...     message="Access denied: user lacks write permission for repository",
        ...     details={"user": "user123", "repository": "org/repo", "required_permission": "write"}
        ... )
    """

    def __init__(
        self,
        message: str,
        details: dict[str, object] | None = None,
    ) -> None:
        """Initialize tool execution error.

        Args:
            message: Human-readable error description for LLM
            details: Optional structured error context
        """
        self.message = message
        self.details = details
        super().__init__(message)
