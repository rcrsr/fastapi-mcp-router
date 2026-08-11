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

import base64
import re
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import TypedDict
from urllib.parse import unquote
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, field_validator


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


_ALLOWED_ICON_SCHEMES = ("https://", "data:")

_ALLOWED_ICON_MIME_TYPES = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/svg+xml",
        "image/webp",
        "image/gif",
        "image/x-icon",
    }
)

# Explicit allowlist of SVG/HTML event-handler attribute names. Using an
# explicit list (rather than the open-ended "\bon\w+\s*=") avoids false
# positives on legitimate attributes/words that merely start with "on"
# (e.g. "data-once=", "data-online=", "data-only="); "\b" alone does not
# guard against those because the preceding "-" already satisfies "\b".
_EVENT_HANDLER_NAMES = (
    "onload",
    "onerror",
    "onclick",
    "onmouseover",
    "onmouseout",
    "onmousedown",
    "onmouseup",
    "onmousemove",
    "onfocus",
    "onblur",
    "onchange",
    "onsubmit",
    "onkeydown",
    "onkeypress",
    "onkeyup",
    "onactivate",
    "onbegin",
    "onend",
    "onrepeat",
    "onanimationstart",
    "onanimationend",
    "onanimationiteration",
    "ontransitionend",
    "ondrag",
    "ondrop",
    "ontoggle",
    "onwheel",
    "onscroll",
    "onresize",
    "onunload",
    "onabort",
    "oncopy",
    "oncut",
    "onpaste",
    "onpointerdown",
    "onbeforeinput",
)

# Matches "<script", "javascript:" URIs, known event-handler attributes,
# "<foreignObject" (embeds arbitrary HTML inside SVG), CSS "expression(...)"
# and "-moz-binding" (legacy IE/Firefox CSS-driven script execution), and
# XML entity/DOCTYPE declarations (billion-laughs / XXE-style expansion
# vectors). Event handlers must sit at an attribute position - immediately
# preceded by whitespace, a tag opener "<", or a quote character - not by an
# arbitrary non-word character like "-" (which would otherwise make
# "data-onload=" match as if it were the real "onload=" handler).
_EVENT_HANDLER_ALTERNATION = "|".join(_EVENT_HANDLER_NAMES)
_EXECUTABLE_CONTENT_PATTERN = re.compile(
    rf"<script|(?:^|[\s<\"'])(?:{_EVENT_HANDLER_ALTERNATION})\s*=|javascript:"
    r"|<foreignObject|expression\(|-moz-binding|<!ENTITY|<!DOCTYPE",
    re.IGNORECASE,
)

_SVG_UNSAFE_MESSAGE = "Icon src contains '<script>' or other executable content, which is rejected for security"
_SVG_UNDECODABLE_BASE64_MESSAGE = (
    "Icon src base64 payload could not be decoded under any supported base64 variant, which is rejected for security"
)


def _decode_base64_variants(payload: str) -> list[str]:
    """Decode a base64 payload trying every base64 alphabet consumers may use.

    Consumers disagree on which base64 alphabet they accept: Python's strict
    decoder rejects the URL-safe "-"/"_" alphabet, while many lenient
    JavaScript/Node base64 decoders accept it. To close that gap, this
    attempts both the standard and URL-safe alphabets (normalizing missing
    "=" padding first) and returns every variant that decodes successfully.

    Returns:
        A list of successfully decoded UTF-8 strings. Empty if the payload
        fails to decode under every known base64 variant.
    """
    padded = payload + "=" * (-len(payload) % 4)
    decoders = (
        lambda p: base64.b64decode(p, validate=False),
        base64.urlsafe_b64decode,
    )
    decoded: list[str] = []
    for decoder in decoders:
        try:
            raw_bytes = decoder(padded)
        except ValueError:
            continue
        decoded.append(raw_bytes.decode("utf-8", errors="replace"))
    return decoded


def _svg_scan_candidates(v: str) -> list[str]:
    """Build every representation of ``v`` a conformant consumer might render.

    Consumers may see the raw string, a percent-decoded ``data:`` payload
    (per RFC 2397 / WHATWG when ``;base64,`` is absent), or a base64-decoded
    payload (under either the standard or URL-safe alphabet). Scanning only
    one representation lets executable content hide in whichever form is
    not scanned; this returns every representation so a single detector can
    be run over all of them uniformly.

    Raises:
        ValueError: If ``v`` carries a ``;base64,`` payload that fails to
            decode under every known base64 variant. A legitimate encoder
            never emits an undecodable payload, so this is itself treated
            as suspicious rather than silently falling back to the raw scan.
    """
    candidates = [v]
    if not v.startswith("data:"):
        return candidates
    if ";base64," in v:
        payload = v.split(";base64,", 1)[1]
        decoded_variants = _decode_base64_variants(payload)
        if not decoded_variants:
            raise ValueError(_SVG_UNDECODABLE_BASE64_MESSAGE)
        candidates.extend(decoded_variants)
    elif "," in v:
        payload = v.split(",", 1)[1]
        candidates.append(unquote(payload))
    return candidates


class Icon(BaseModel):
    """
    Icon metadata attached to tools, resources, resource templates, and prompts.

    Distinct from ServerIcon (a TypedDict carrying server-wide branding); this
    is a validated Pydantic model used wherever the protocol requires a list
    of per-item icons.

    Executable-content filtering is a best-effort blocklist, not an
    exhaustive sanitizer: it screens out common script/event-handler
    patterns at registration time but is not a substitute for render-time
    defenses. Consumers rendering icon src values (especially inline SVG or
    "data:" payloads) should still enforce a restrictive CSP
    (``script-src 'none'``) and/or sanitize with a library such as
    DOMPurify before rendering.

    Attributes:
        src: Icon source; must use the "https://" or "data:" scheme
        mimeType: Advisory MIME type of the icon, e.g. "image/svg+xml"
        sizes: Optional list of available sizes, e.g. ["32x32", "16x16"]
        theme: Optional color theme the icon is designed for; "light" or "dark"

    Raises:
        ValueError: If src uses a scheme other than "https://" or "data:"
        ValueError: If src (or its decoded "data:" payload) contains
            "<script>", an event-handler attribute (e.g. "onload="), a
            "javascript:" URI, "<foreignObject>", a CSS "expression(...)"
            or "-moz-binding" declaration, an XML entity/DOCTYPE
            declaration, or other executable content — including such
            content hidden inside a base64-encoded "data:" payload,
            regardless of the declared media type
        ValueError: If mimeType is set to a value outside the strict
            image-type allowlist
    """

    src: str
    mimeType: str | None = None
    sizes: list[str] | None = None
    theme: str | None = None

    @field_validator("src")
    @classmethod
    def _validate_scheme(cls, v: str) -> str:
        if not v.startswith(_ALLOWED_ICON_SCHEMES):
            scheme = v.split(":", 1)[0] if ":" in v else v
            raise ValueError(f"Icon src scheme {scheme!r} is not allowed; only 'https://' and 'data:' are permitted")
        return v

    @field_validator("src")
    @classmethod
    def _validate_executable_content(cls, v: str) -> str:
        for candidate in _svg_scan_candidates(v):
            if _EXECUTABLE_CONTENT_PATTERN.search(candidate):
                raise ValueError(_SVG_UNSAFE_MESSAGE)
        return v

    @field_validator("mimeType")
    @classmethod
    def _validate_mime_type(cls, v: str | None) -> str | None:
        if v is not None and v not in _ALLOWED_ICON_MIME_TYPES:
            raise ValueError(f"Icon mimeType {v!r} is not an allowed image type")
        return v


class ImageContent(BaseModel):
    """
    Image content in MCP response.

    Represents a single image content item using a flat base64-encoded data
    string, matching the conformant MCP wire shape.

    Attributes:
        type: Content type identifier, always "image"
        data: Base64-encoded image data
        mimeType: MIME type of the image, e.g. "image/png"
    """

    type: str = "image"
    data: str
    mimeType: str


class AudioContent(BaseModel):
    """
    Audio content in MCP response.

    Represents a single audio content item using a flat base64-encoded data
    string, matching the conformant MCP wire shape.

    Attributes:
        type: Content type identifier, always "audio"
        data: Base64-encoded audio data
        mimeType: MIME type of the audio, e.g. "audio/mpeg"
    """

    type: str = "audio"
    data: str
    mimeType: str


class ResourceLinkContent(BaseModel):
    """
    Resource link content in MCP response.

    Points to a resource without embedding its full content, matching the
    conformant MCP wire shape for resource_link content blocks.

    Attributes:
        type: Content type identifier, always "resource_link"
        uri: URI identifying the linked resource
        name: Resource identifier/name
        title: Optional human-readable display title
        description: Optional resource description
        mimeType: Optional MIME type of the linked resource
        icons: Optional list of Icon objects for the resource
        size: Optional size of the resource in bytes
    """

    type: str = "resource_link"
    uri: str
    name: str
    title: str | None = None
    description: str | None = None
    mimeType: str | None = None
    icons: list[Icon] | None = None
    size: int | None = None


class ToolAnnotations(BaseModel):
    """
    Behavioral hints describing an MCP tool for client UIs.

    Annotations are untrusted hints supplied by tool authors; clients may use
    them to inform UI presentation but must not treat them as security
    guarantees. Unknown/vendor keys are preserved unchanged and pass through
    without validation.

    Attributes:
        title: Optional human-readable display title for the tool
        readOnlyHint: Optional hint that the tool does not modify state
        destructiveHint: Optional hint that the tool may perform destructive updates
        idempotentHint: Optional hint that repeated calls have no additional effect
        openWorldHint: Optional hint that the tool interacts with an open-world
            (external) environment
    """

    model_config = ConfigDict(extra="allow")

    title: str | None = None
    readOnlyHint: bool | None = None
    destructiveHint: bool | None = None
    idempotentHint: bool | None = None
    openWorldHint: bool | None = None


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
