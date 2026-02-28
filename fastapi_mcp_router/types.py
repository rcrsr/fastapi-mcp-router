"""
MCP (Model Context Protocol) response format types.

This module defines Pydantic models for structuring MCP tool responses.
MCP responses consist of content items (typically text) wrapped in a response
container that indicates success or error status.

The isError field distinguishes between successful tool execution (isError=False)
and tool errors (isError=True). When isError=True, the error message is included
in the content field so LLMs can see and handle the error appropriately.

Example:
    Success response::

        response = ToolResponse(
            content=[TextContent(text="Operation completed successfully")],
            isError=False,
        )

    Error response::

        response = ToolResponse(
            content=[TextContent(text="Invalid parameter: id must be numeric")],
            isError=True,
        )
"""

from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import TypedDict
from uuid import UUID, uuid4

from pydantic import BaseModel


class ServerIcon(TypedDict, total=False):
    """
    MCP server icon metadata.

    Represents an icon that can be displayed for the MCP server in client UIs.
    Used in the serverInfo.icons array for Claude Custom Connectors.

    Attributes:
        src: URL to the icon resource (required)
        mimeType: MIME type of the icon, e.g. "image/svg+xml" (required)
        sizes: Optional list of available sizes, e.g. ["32x32", "16x16"]
    """

    src: str
    mimeType: str
    sizes: list[str]


class ServerInfo(TypedDict, total=False):
    """
    MCP server metadata for client discovery.

    Provides server identification and branding information returned in the
    MCP initialize response. Used by Claude Custom Connectors to display
    server information in the UI.

    Attributes:
        name: Server identifier, typically kebab-case (required)
        version: Semantic version string (required)
        title: Human-readable display name
        description: Server description for UI display
        icons: Array of ServerIcon objects for branding
        websiteUrl: Server website URL
    """

    name: str
    version: str
    title: str
    description: str
    icons: list[ServerIcon]
    websiteUrl: str


class TextContent(BaseModel):
    """
    Text content in MCP response.

    Represents a single text content item that can be included in an MCP
    tool response. The type field is always "text" to indicate text content.

    Attributes:
        type: Content type identifier, always "text"
        text: The actual text content

    Example::

        content = TextContent(text="Hello, world!")
        assert content.type == "text"
        assert content.text == "Hello, world!"
    """

    type: str = "text"
    text: str


class ToolResponse(BaseModel):
    """
    MCP tool response format.

    Container for MCP tool execution results. Includes a list of content items
    (typically TextContent) and an error flag to distinguish between successful
    executions and tool errors.

    When isError=True, the content field contains error messages that are visible
    to the LLM, allowing it to understand what went wrong and potentially retry
    or adjust its approach.

    Attributes:
        content: List of content items, typically TextContent instances
        isError: Whether this response represents an error (default: False)

    Example:
        Success with single content item::

            response = ToolResponse(
                content=[TextContent(text="User created successfully")],
            )

        Success with multiple content items::

            response = ToolResponse(
                content=[
                    TextContent(text="Found 3 matching records:"),
                    TextContent(text="1. Record A"),
                    TextContent(text="2. Record B"),
                    TextContent(text="3. Record C"),
                ],
            )

        Error response::

            response = ToolResponse(
                content=[
                    TextContent(text="Failed to create user: email already exists")
                ],
                isError=True,
            )
    """

    content: list[TextContent]
    isError: bool = False


EventSubscriber = Callable[
    [str, int | None],
    AsyncGenerator[tuple[int, dict]],
]
"""Type alias for SSE event subscriber callables.

A callable that, given a session ID and an optional last-event-ID,
returns an async generator yielding ``(event_id, json_rpc_notification)``
tuples.

Args:
    session_id: MCP session ID for the connected client.
    last_event_id: Value from ``Last-Event-ID`` header, or ``None``.

Returns:
    AsyncGenerator yielding ``(event_id, json_rpc_notification)`` tuples.
"""

ProgressCallback = Callable[[int, int, str | None], Awaitable[None]]
"""Type alias for tool progress reporting callables.

Injected by the registry when ``progress: ProgressCallback`` appears in a
tool handler signature, following the same pattern as ``Request`` and
``BackgroundTasks`` injection.

Args:
    current: Number of units completed so far.
    total: Total number of units to complete.
    message: Optional human-readable status message, or ``None``.

Returns:
    Awaitable that resolves once the progress notification is sent.
"""


@dataclass
class McpSessionData:
    """
    MCP session data for tracking active streaming connections.

    Tracks metadata for an active MCP session, including connection association,
    event sequence position, and session creation timestamp.

    Note:
        oauth_client_id and connection_id are stored separately to allow downstream
        code to discriminate between OAuth and API key authentication. Currently,
        only one is set per session. This design allows future flexibility - for
        example, OAuth users could choose a "default" connection, in which case
        both fields would be populated.

    Attributes:
        session_id: Unique identifier for the MCP session
        oauth_client_id: UUID of associated OAuth client (for OAuth Bearer auth), None for API key auth
        connection_id: UUID of associated connection (for API key auth), None for OAuth auth
        last_event_id: Last event sequence number delivered to this session
        created_at: Timestamp when the session was created
    """

    session_id: str
    oauth_client_id: UUID | None
    connection_id: UUID | None
    last_event_id: int
    created_at: datetime


class LogLevel(IntEnum):
    """
    MCP logging severity levels in ascending priority order.

    Integer values reflect priority so level comparisons work directly:
    ``LogLevel.debug < LogLevel.info`` evaluates to ``True``.

    Attributes:
        debug: Lowest priority; verbose diagnostic information
        info: General informational messages (default level)
        notice: Normal but significant events
        warning: Potentially harmful situations
        error: Error events that may still allow continued operation
        critical: Severe errors causing partial functionality loss
        alert: Action must be taken immediately
        emergency: Highest priority; system is unusable
    """

    debug = 0
    info = 1
    notice = 2
    warning = 3
    error = 4
    critical = 5
    alert = 6
    emergency = 7


@dataclass
class Root:
    """
    MCP root URI that the client exposes to the server.

    Roots define filesystem or resource locations the server may access.
    The server requests roots via ``roots/list`` and the client returns
    a list of ``Root`` objects.

    Attributes:
        uri: The URI identifying this root (e.g. "file:///home/user/project")
        name: Optional human-readable label for the root
    """

    uri: str
    name: str | None = None


@dataclass
class SamplingRequest:
    """
    Request sent to the client to perform LLM sampling on behalf of the server.

    Encapsulates all parameters needed for a ``sampling/createMessage`` MCP
    request. The client fulfils the request using its own LLM connection and
    returns a ``SamplingResponse``.

    Attributes:
        messages: Conversation context passed to the LLM
        request_id: UUID4 identifying this request, auto-generated
        model_preferences: Optional hints for model selection
        system_prompt: Optional system prompt prepended to the conversation
        temperature: Optional sampling temperature (0.0-1.0)
        max_tokens: Optional maximum tokens the LLM may generate
        stop_sequences: Optional list of sequences that stop generation
    """

    messages: list[dict]
    request_id: str = field(default_factory=lambda: str(uuid4()))
    model_preferences: dict | None = None
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    stop_sequences: list[str] | None = None


@dataclass
class SamplingResponse:
    """
    Response returned by the client after fulfilling a sampling request.

    Contains the LLM-generated message along with metadata about which model
    produced it and why generation stopped.

    Attributes:
        model: Identifier of the model that generated the response
        role: Conversation role of the generated message (e.g. "assistant")
        content: Generated message content as a dict (type + text or image data)
        stop_reason: Optional reason generation stopped (e.g. "end_turn", "max_tokens")
    """

    model: str
    role: str
    content: dict
    stop_reason: str | None = None


class CompletionRef(BaseModel):
    """
    Reference to a prompt or resource for argument completion.

    Identifies the prompt or resource whose argument the client is requesting
    completions for.

    Attributes:
        type: Reference type; either "ref/prompt" or "ref/resource"
        name: Name of the prompt or resource being referenced
    """

    type: str
    name: str


class CompletionArgument(BaseModel):
    """
    Argument being completed in a completion request.

    Holds the argument name and the partial value the user has typed so far.

    Attributes:
        name: Name of the argument being completed
        value: Current partial value entered by the user
    """

    name: str
    value: str


class CompletionResult(BaseModel):
    """
    Completion suggestions returned for a completion request.

    Contains the list of suggested values along with pagination metadata
    indicating whether additional results exist.

    Attributes:
        values: Suggested completion strings, max 100 items
        total: Total number of matching completions if known, or None
        hasMore: Whether more completions exist beyond those returned
    """

    values: list[str]
    total: int | None = None
    hasMore: bool = False


class ElicitationRequest(BaseModel):
    """
    Request sent to the client to elicit structured input from the user.

    Prompts the user with a human-readable message and an expected JSON Schema
    describing the structure of the requested input.

    Attributes:
        message: Human-readable prompt shown to the user
        requestedSchema: JSON Schema describing the expected user input structure
    """

    message: str
    requestedSchema: dict


class ElicitationResponse(BaseModel):
    """
    Response returned by the client after presenting an elicitation request.

    Captures both the user's action (accept, decline, or cancel) and any
    structured content the user provided when accepting.

    Attributes:
        action: User's choice; one of "accept", "decline", or "cancel"
        content: User-provided data matching requestedSchema (only when accepted), or None
    """

    action: str
    content: dict | None = None
