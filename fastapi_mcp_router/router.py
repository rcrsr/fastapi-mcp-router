"""
FastAPI router factory for MCP JSON-RPC protocol.

This module provides the main router factory function that creates a configured
FastAPI APIRouter for handling MCP JSON-RPC requests over HTTP. The router
implements stateless HTTP transport with strict protocol version validation.

Key features:
- JSON-RPC 2.0 request/response handling
- MCP protocol version 2025-06-18 support
- Tool registration via MCPToolRegistry
- Comprehensive error handling for protocol and business logic errors

The router creates a single POST endpoint at route="" (empty string), designed to
be mounted with prefix="/mcp" to create the /mcp endpoint. All MCP communication
flows through this single endpoint using JSON-RPC method routing.
"""

import asyncio
import contextlib
import json
import logging
import sys
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any, cast
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.requests import ClientDisconnect
from starlette.responses import Response

from fastapi_mcp_router.exceptions import MCPError, ToolError
from fastapi_mcp_router.prompts import PromptRegistry
from fastapi_mcp_router.protocol import json_rpc_error, json_rpc_response
from fastapi_mcp_router.registry import MCPToolRegistry
from fastapi_mcp_router.resources import ResourceProvider, ResourceRegistry
from fastapi_mcp_router.session import (
    _SUBSCRIPTIONS_MAX,
    MCPLoggingHandler,
    ProgressTracker,
    RootsManager,
    SamplingManager,
    SessionStore,
)
from fastapi_mcp_router.telemetry import get_meter, get_tracer
from fastapi_mcp_router.types import EventSubscriber, McpSessionData, ServerInfo

logger = logging.getLogger(__name__)

# Type alias for authentication validator callback
AuthValidator = Callable[[str | None, str | None], Awaitable[Any]]

# Type aliases for session management callbacks
SessionGetter = Callable[[str], Awaitable[McpSessionData | None]]
SessionCreator = Callable[[UUID | None, UUID | None], Awaitable[str]]

# Type alias for tool filtering callback
# (is_oauth) -> excluded_tool_names or None
ToolFilter = Callable[[bool], list[str] | None]

# Type alias for completion handler callback
# (ref: dict, argument: dict) -> {"values": list[str], "total": int|None, "hasMore": bool}
CompletionHandler = Callable[[dict, dict], Awaitable[dict[str, object]]]

_ELICITATION_TIMEOUT = 30.0


class ElicitationManager:
    """Server-to-client elicitation request manager.

    Enqueues elicitation/create requests via a SessionStore and correlates
    client responses to waiting callers using asyncio.Future objects keyed by
    request ID. Validates accept responses against requestedSchema.

    Attributes:
        _session_store: SessionStore used to enqueue elicitation requests.
        _pending: Map of request_id to asyncio.Future awaiting client response.

    Example::

        manager = ElicitationManager(session_store=store)
        response = await manager.create(
            session_id="sess-1",
            message="Please confirm",
            requested_schema={"type": "object", "properties": {}},
        )
        assert response["action"] in ("accept", "decline", "cancel")
    """

    def __init__(self, session_store: SessionStore) -> None:
        """Initialize with a SessionStore.

        Args:
            session_store: SessionStore instance for enqueueing requests.
        """
        self._session_store = session_store
        self._pending: dict[str, asyncio.Future] = {}

    async def create(
        self,
        session_id: str,
        message: str,
        requested_schema: dict,
    ) -> dict[str, object]:
        """Send an elicitation/create request to the client and await the response.

        Generates a UUID4 request_id, enqueues the request via the SessionStore,
        registers a Future, and awaits it with a 30-second timeout. Validates
        accept responses against requestedSchema.

        Args:
            session_id: UUID4 string identifying the target session.
            message: Human-readable message to display to the user.
            requested_schema: JSON Schema dict describing the expected content.

        Returns:
            Dict with action ("accept", "decline", "cancel") and optional content.

        Raises:
            MCPError: -32603 if the client does not respond within 30 seconds (EC-25).
        """
        request_id = str(uuid4())
        notification: dict[str, object] = {
            "jsonrpc": "2.0",
            "method": "elicitation/create",
            "id": request_id,
            "params": {
                "message": message,
                "requestedSchema": requested_schema,
            },
        }
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._session_store.enqueue_message(session_id, notification)
            client_response = await asyncio.wait_for(future, timeout=_ELICITATION_TIMEOUT)
            action = client_response.get("action", "cancel")
            content = client_response.get("content")
            # Validate content against requestedSchema only for accept action (AC-65)
            if action == "accept" and content is not None:
                _validate_elicitation_content(content, requested_schema)
            return {"action": action, "content": content}
        except TimeoutError as err:
            raise MCPError(
                code=-32603,
                message=f"Elicitation request timed out after {_ELICITATION_TIMEOUT:.0f}s",
            ) from err
        finally:
            self._pending.pop(request_id, None)

    def handle_response(self, request_id: str, response: dict) -> None:
        """Correlate a client elicitation response to the waiting create call.

        Looks up the Future registered for request_id and sets its result.
        Silently ignores unknown IDs.

        Args:
            request_id: Request ID matching the original elicitation/create.
            response: Response dict from the client containing action and content.

        Returns:
            None
        """
        future = self._pending.get(request_id)
        if future is not None and not future.done():
            future.set_result(response)


def _validate_elicitation_content(content: object, schema: dict) -> None:
    """Validate elicitation content against a JSON Schema (basic type checking).

    Performs structural type validation based on the schema's ``type`` and
    ``properties`` fields. Only validates ``type`` and required field presence.
    Full JSON Schema validation requires an external library not in scope.

    Args:
        content: The content value returned by the client.
        schema: JSON Schema dict with optional ``type`` and ``properties``.

    Raises:
        MCPError: -32602 if content type or required properties do not match.
    """
    schema_type = schema.get("type")
    if schema_type == "object":
        if not isinstance(content, dict):
            raise MCPError(
                code=-32602,
                message=f"Elicitation content must be an object, got {type(content).__name__}",
            )
        required = schema.get("required", [])
        if isinstance(required, list):
            for field_name in required:
                if field_name not in content:
                    raise MCPError(
                        code=-32602,
                        message=f"Elicitation content missing required field: {field_name}",
                    )
    elif schema_type == "array":
        if not isinstance(content, list):
            raise MCPError(
                code=-32602,
                message=f"Elicitation content must be an array, got {type(content).__name__}",
            )
    elif schema_type == "string":
        if not isinstance(content, str):
            raise MCPError(
                code=-32602,
                message=f"Elicitation content must be a string, got {type(content).__name__}",
            )
    elif schema_type == "number":
        if not isinstance(content, (int, float)):
            raise MCPError(
                code=-32602,
                message=f"Elicitation content must be a number, got {type(content).__name__}",
            )
    elif schema_type == "boolean":
        if not isinstance(content, bool):
            raise MCPError(
                code=-32602,
                message=f"Elicitation content must be a boolean, got {type(content).__name__}",
            )


async def _check_authentication(
    request: Request,
    auth_validator: AuthValidator | None = None,
) -> tuple[str | None, str | None, Any]:
    """
    Check authentication headers and validate credentials.

    Extracts authentication credentials from request headers and optionally
    validates them using the provided auth_validator callback.

    Args:
        request: FastAPI Request object
        auth_validator: Optional callback to validate credentials

    Returns:
        Tuple of (api_key, bearer_token, auth_result) where:
        - api_key: Value from X-API-Key header or None
        - bearer_token: Token from Authorization Bearer header or None
        - auth_result: The raw validator return value when a validator is
          configured (truthy → authenticated; stored at
          request.state.auth_context), or a bool when no validator is
          configured (True if credentials are present, False otherwise)

    Note:
        Non-constant-time string comparison for "bearer " prefix is acceptable.
        This only leaks whether the 7-char prefix exists, not the actual token.
        Real token validation with constant-time comparison occurs in validator.
    """
    headers = request.headers
    api_key = headers.get("x-api-key")
    auth_header = headers.get("authorization", "")

    # Note: Non-constant-time string comparison for "bearer " prefix is acceptable here.
    # This only leaks whether the 7-char prefix exists, not the actual token value.
    # Real token validation with constant-time comparison occurs in auth_validator.
    has_bearer = auth_header.lower().startswith("bearer ")
    bearer_token = auth_header[7:].strip() if has_bearer else None

    # If validator provided, capture the full return value
    if auth_validator is not None:
        auth_result = await auth_validator(api_key, bearer_token)
        if auth_result:
            request.state.auth_context = auth_result
    else:
        # Default behavior: just check presence (no actual validation)
        auth_result = bool(api_key or has_bearer)

    return api_key, bearer_token, auth_result


def create_mcp_router(
    registry: MCPToolRegistry,
    rate_limit_dependency: Callable[..., Awaitable[None]] | None = None,
    auth_validator: AuthValidator | None = None,
    base_url: str | None = None,
    session_getter: SessionGetter | None = None,
    session_creator: SessionCreator | None = None,
    tool_filter: ToolFilter | None = None,
    server_info: ServerInfo | None = None,
    event_subscriber: EventSubscriber | None = None,
    oauth_resource_metadata: dict[str, object] | None = None,
    session_store: SessionStore | None = None,
    stateful: bool = False,
    resource_registry: ResourceRegistry | None = None,
    prompt_registry: PromptRegistry | None = None,
    sampling_enabled: bool = False,
    roots_manager: RootsManager | None = None,
    completion_handler: Callable[..., Awaitable[object]] | None = None,
    legacy_sse: bool = False,
    enable_telemetry: bool = True,
    shutdown_event: asyncio.Event | None = None,
) -> APIRouter:
    """
    Create FastAPI router for MCP protocol.

    Creates a configured APIRouter with a single POST endpoint for handling
    MCP JSON-RPC requests. The endpoint validates the MCP-Protocol-Version
    header, routes methods to appropriate handlers, and returns JSON-RPC
    formatted responses.

    The router is designed to be mounted with prefix="/mcp" in the main
    FastAPI application, creating the /mcp endpoint for MCP communication.

    Args:
        registry: Tool registry containing registered MCP tools and their
            handlers. Used for tool discovery (tools/list) and execution
            (tools/call) operations.
        rate_limit_dependency: Optional FastAPI dependency for rate limiting.
            If provided, will be applied to both GET and POST endpoints using
            Depends(). Should be an async callable that raises HTTPException
            when rate limit is exceeded. If None, no rate limiting is applied.
        auth_validator: Optional callback to validate authentication credentials.
            Takes (api_key, bearer_token) and returns True if valid. If None,
            only checks credential presence without validation. Should use
            constant-time comparison for tokens to prevent timing attacks.
        base_url: Optional base URL for OAuth discovery metadata. If provided,
            will be used in WWW-Authenticate header for resource_metadata URL.
            If None, falls back to request.base_url. This prevents host header
            injection attacks in production environments.
        session_getter: Optional callback to retrieve session data by session ID.
            Takes session_id (str) and returns session data dict or None if not
            found. Used for SSE streaming to restore session state.
        session_creator: Optional callback to create new session for user.
            Takes user_id (str) and returns new session_id (str). Used when
            initializing SSE streaming connection.
        tool_filter: Optional callback to filter tools based on connection type.
            Takes is_oauth (bool) and returns list of tool names to exclude,
            or None to include all tools. Used to hide tools from specific
            connection types (e.g., hide connection_info from OAuth connections).
        server_info: Optional server metadata for MCP initialize response.
            If provided, merges with default server info (name, version).
            Supports fields like title, description, icons, and websiteUrl
            for Claude Custom Connectors branding.
        event_subscriber: Optional callable that returns an async generator
            yielding (event_id, json_rpc_notification) tuples for a session.
            Takes (session_id, last_event_id) and returns AsyncGenerator. If
            None, SSE endpoint streams keepalives only (no regression).
        oauth_resource_metadata: Optional RFC 9728 Protected Resource Metadata
            dict. Must contain "resource" and "authorization_servers" keys. If
            None, no PRM endpoint is registered. Raises ValueError if provided
            without required keys.
        session_store: Optional SessionStore for stateful SSE mode. Provides
            session lifecycle (create/get/delete) and message queuing for
            streaming tool results. Cannot be combined with session_getter.
            Raises ValueError if both session_store and session_getter are
            provided.
        stateful: If True, enables stateful SSE mode. Requires session_store
            to be provided. Raises ValueError if True and session_store is None.
        roots_manager: Optional RootsManager instance for pre-populating roots
            before router creation. If None, a new empty RootsManager is created.
        completion_handler: Optional async callable for completion/complete requests.
            Takes (ref, argument) and returns a dict with a "values" key (list).
            If None, completion/complete returns -32601 Method not found.
        shutdown_event: Optional asyncio.Event that signals SSE generators to
            exit gracefully. When set, both legacy SSE generators
            (store_event_stream and event_stream) break their loops and yield
            a ``: server-shutdown`` comment before returning. If None
            (default), generators run until client disconnect. MCPRouter
            creates and manages this automatically via its shutdown() method.

    Returns:
        Configured APIRouter with POST endpoint at route "". When included
        in main app with prefix="/mcp", creates /mcp endpoint.

    Raises:
        ValueError: If both session_store and session_getter are provided.
        ValueError: If stateful=True and session_store is None.
        ValueError: If sampling_enabled=True and stateful is False.
        ValueError: If oauth_resource_metadata is missing "resource" or
            "authorization_servers" keys.

    Example:
        >>> from fastapi import FastAPI, HTTPException
        >>> from fastapi_mcp_router import MCPToolRegistry, create_mcp_router
        >>>
        >>> app = FastAPI()
        >>> registry = MCPToolRegistry()
        >>>
        >>> @registry.tool()
        >>> async def hello(name: str) -> str:
        ...     '''Say hello to someone.'''
        ...     return f"Hello, {name}!"
        >>>
        >>> # Optional: Add authentication validator
        >>> async def validate_auth(api_key: str | None, token: str | None) -> bool:
        ...     if api_key:
        ...         return api_key == "valid-key"  # Use secrets.compare_digest
        ...     if token:
        ...         return token == "valid-token"  # Use secrets.compare_digest
        ...     return False
        >>>
        >>> # Optional: Add rate limiting dependency
        >>> async def rate_limit():
        ...     # Your rate limiting logic here
        ...     pass
        >>>
        >>> mcp_router = create_mcp_router(
        ...     registry,
        ...     rate_limit_dependency=rate_limit,
        ...     auth_validator=validate_auth,
        ...     base_url="https://api.example.com",
        ... )
        >>> app.include_router(mcp_router, prefix="/mcp")
        >>>
        >>> # Creates POST /mcp endpoint for MCP JSON-RPC requests
    """
    if session_store is not None and session_getter is not None:
        raise ValueError("session_store and session_getter are mutually exclusive. Provide one or the other, not both.")

    if stateful and session_store is None:
        raise ValueError("stateful=True requires session_store to be provided.")

    if sampling_enabled and not stateful:
        raise ValueError("sampling_enabled=True requires stateful=True.")

    if oauth_resource_metadata is not None:
        missing = [key for key in ("resource", "authorization_servers") if key not in oauth_resource_metadata]
        if missing:
            raise ValueError(f"oauth_resource_metadata is missing required keys: {missing}")

    _tracer: Any = get_tracer(enable_telemetry)
    _meter: Any = get_meter(enable_telemetry)

    _request_counter: Any = None
    if _meter is not None:
        try:
            _meter_any_init = cast(Any, _meter)
            _request_counter = _meter_any_init.create_counter("mcp.server.request.count")
        except Exception as _meter_exc:
            logger.debug("OTel meter counter creation failed: %s", _meter_exc)

    router = APIRouter()

    # Create ProgressTracker for stateful mode; None in stateless mode (AC-42, AC-43)
    _progress_tracker = ProgressTracker(session_store=session_store) if session_store is not None else None

    # Create SamplingManager for stateful mode; None in stateless mode (EC-19)
    _sampling_manager = SamplingManager(session_store, sampling_enabled) if session_store is not None else None

    # RootsManager has no stateful requirement; use injected instance if provided
    _roots_manager = roots_manager if roots_manager is not None else RootsManager()

    # Create MCPLoggingHandler for stateful mode; None in stateless mode (EC-23)
    _logging_handler = MCPLoggingHandler(session_store) if session_store is not None else None

    # Create ElicitationManager for stateful mode; None in stateless mode
    _elicitation_manager = ElicitationManager(session_store=session_store) if session_store is not None else None

    def _get_base_url(request: Request) -> str:
        """
        Get base URL for OAuth discovery metadata.

        Uses configured base_url if provided, otherwise falls back to
        request.base_url for local development. This prevents host header
        injection attacks in production.

        Args:
            request: FastAPI Request object

        Returns:
            Base URL without trailing slash (e.g., "https://api.example.com")
        """
        if base_url:
            return base_url.rstrip("/")
        # Use Starlette's built-in base_url property as fallback
        return str(request.base_url).rstrip("/")

    def _sanitize_headers(headers: dict[str, str]) -> dict[str, str]:
        """
        Redact sensitive headers for safe logging.

        Args:
            headers: Request headers dictionary

        Returns:
            Dictionary with sensitive values redacted
        """
        sensitive = {"authorization", "x-api-key", "cookie", "x-csrf-token"}
        return {k: "[REDACTED]" if k.lower() in sensitive else v for k, v in headers.items()}

    # Build dependencies list for endpoints
    dependencies = []
    if rate_limit_dependency is not None:
        dependencies.append(Depends(rate_limit_dependency))

    if legacy_sse:

        @router.get("", dependencies=dependencies, response_model=None)
        async def handle_mcp_sse(
            request: Request,
            background_tasks: BackgroundTasks,
        ) -> JSONResponse | StreamingResponse:
            """
            Handle GET requests to MCP endpoint for SSE streaming.

            Per MCP spec: "The client MAY issue an HTTP GET to the MCP endpoint.
            This can be used to open an SSE stream, allowing the server to
            communicate to the client."

            When session_getter/session_creator callbacks or session_store are
            provided, this endpoint returns an SSE stream for Streamable HTTP
            transport. Otherwise, returns server information for stateless mode.

            Args:
                request: FastAPI Request object
                background_tasks: FastAPI BackgroundTasks for async operations

            Returns:
                StreamingResponse with text/event-stream when session callbacks
                or session_store provided, otherwise JSONResponse with server info

            Raises:
                401 Unauthorized when authentication fails or identity not found
                410 Gone when session expired or not found (for resume attempts)
                500 Internal Server Error when session creation fails
            """
            logger.info("Received GET request to MCP endpoint")

            # Check authentication using helper function
            _api_key, bearer_token, is_valid = await _check_authentication(request, auth_validator)

            # Require authentication: either X-API-Key or Authorization Bearer token
            if not is_valid:
                logger.warning("Unauthenticated GET request to MCP endpoint")
                response_headers: dict[str, str] = {"WWW-Authenticate": 'Bearer realm="mcp"'}
                response_content: dict[str, object] = {
                    "error": "Authentication required",
                    "error_description": "Bearer token missing or invalid. Re-authenticate.",
                }
                if oauth_resource_metadata is not None:
                    base_url_str = _get_base_url(request)
                    resource_metadata_url = f"{base_url_str}/.well-known/oauth-protected-resource"
                    response_headers["WWW-Authenticate"] = f'Bearer resource_metadata="{resource_metadata_url}"'
                    response_content["resource_metadata"] = resource_metadata_url
                return JSONResponse(
                    status_code=401,
                    headers=response_headers,
                    content=response_content,
                )

            # session_store path: require Mcp-Session-Id header, resume existing session
            if session_store is not None:
                logger.info("session_store mode - establishing SSE stream")
                session_id_hdr = request.headers.get("Mcp-Session-Id")
                if not session_id_hdr:
                    logger.warning("Missing Mcp-Session-Id header for session_store GET")
                    return JSONResponse(
                        status_code=400,
                        content={
                            "error": "missing_session_id",
                            "error_description": "Mcp-Session-Id header required. Call POST initialize first.",
                        },
                    )
                store_session = await session_store.get(session_id_hdr)
                if store_session is None:
                    logger.warning("Session not found in store: %s", session_id_hdr)
                    return JSONResponse(
                        status_code=410,
                        content={
                            "error": "session_not_found",
                            "error_description": f"Session {session_id_hdr} not found or expired",
                            "action": "reconnect",
                        },
                    )

                last_event_id_hdr = request.headers.get("Last-Event-ID")
                if last_event_id_hdr is not None:
                    try:
                        store_last_event_id: int | None = int(last_event_id_hdr)
                    except ValueError:
                        store_last_event_id = None
                else:
                    store_last_event_id = None

                store_sid = store_session.session_id

                async def store_event_stream() -> AsyncGenerator[str]:
                    """
                    Generate SSE events merging event_subscriber and dequeue_messages.

                    Polls session_store.dequeue_messages() every 1s (IC-8) and
                    delivers application events from event_subscriber. Sends a
                    keepalive comment every 30s when no events arrive.

                    Yields:
                        SSE-formatted strings: events as ``id: N\\nevent: message\\n
                        data: ...\\n\\n``, keepalives as ``: keepalive\\n\\n``.
                    """
                    logger.info("SSE stream established for session_store session: %s", store_sid)
                    yield ": SSE stream established\n\n"

                    sub_gen = event_subscriber(store_sid, store_last_event_id) if event_subscriber is not None else None
                    event_counter = 0
                    keepalive_ticks = 0  # 1 tick = 1s; keepalive every 30 ticks

                    try:
                        while shutdown_event is None or not shutdown_event.is_set():
                            # Wait up to 1s for next subscriber event
                            if sub_gen is not None:
                                try:
                                    ev_id, payload = await asyncio.wait_for(sub_gen.__anext__(), timeout=1.0)
                                    yield f"id: {ev_id}\nevent: message\ndata: {json.dumps(payload)}\n\n"
                                    keepalive_ticks = 0
                                except TimeoutError:
                                    keepalive_ticks += 1
                                except StopAsyncIteration:
                                    sub_gen = None
                                    keepalive_ticks += 1
                            else:
                                await asyncio.sleep(1)
                                keepalive_ticks += 1

                            # IC-8: poll dequeue_messages every 1s
                            messages = await session_store.dequeue_messages(store_sid)
                            for msg in messages:
                                event_counter += 1
                                yield f"id: {event_counter}\nevent: message\ndata: {json.dumps(msg)}\n\n"
                            if messages:
                                keepalive_ticks = 0

                            # Send keepalive every 30s
                            if keepalive_ticks >= 30:
                                yield ": keepalive\n\n"
                                keepalive_ticks = 0

                        # Shutdown requested — notify client and close stream.
                        yield ": server-shutdown\n\n"
                        if sub_gen is not None:
                            await sub_gen.aclose()

                    except asyncio.CancelledError:
                        logger.info("SSE stream cancelled for session_store session: %s", store_sid)
                        if sub_gen is not None:
                            await sub_gen.aclose()
                        raise

                return StreamingResponse(
                    store_event_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Mcp-Session-Id": store_sid,
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    },
                )

            # Check if callback-based session mode is active
            if session_getter is None or session_creator is None:
                logger.info("Session callbacks not provided - returning stateless mode info")
                # Return server information for stateless mode
                return JSONResponse(
                    status_code=200,
                    content={
                        "service": "MCP Server",
                        "transport": "HTTP (stateless)",
                        "supportedVersions": ["2025-06-18", "2025-03-26"],
                        "info": (
                            "This is an MCP (Model Context Protocol) server endpoint. "
                            "Send POST requests with JSON-RPC 2.0 payloads."
                        ),
                        "sse": "Not supported (stateless server)",
                        "headers": {
                            "MCP-Protocol-Version": (
                                "Optional. Defaults to 2025-03-26 if not provided. Use '2025-06-18' for latest version."
                            )
                        },
                        "methods": [
                            "initialize",
                            "tools/list",
                            "tools/call",
                            "ping",
                            "notifications/initialized",
                        ],
                    },
                )

            # API key connections are stateless - SSE only available for OAuth
            # Per MCP spec: GET must return 405 Method Not Allowed when SSE not supported
            is_oauth_connection = bearer_token is not None
            if not is_oauth_connection:
                logger.info("API key connection - SSE not available, returning 405")
                return JSONResponse(
                    status_code=405,
                    content={
                        "error": "method_not_allowed",
                        "error_description": (
                            "SSE streaming not available for API key connections. Use POST for requests."
                        ),
                    },
                )

            # OAuth connection with session callbacks - handle SSE streaming
            logger.info("OAuth connection - establishing SSE stream")

            # Extract Mcp-Session-Id header (optional)
            session_id_header = request.headers.get("Mcp-Session-Id")
            session_data: McpSessionData | None = None

            # If session ID provided, try to resume session
            if session_id_header:
                logger.info("Attempting to resume session: %s", session_id_header)
                session_data = await session_getter(session_id_header)
                if session_data is None:
                    logger.warning("Session not found or expired: %s", session_id_header)
                    return JSONResponse(
                        status_code=410,  # Gone
                        content={
                            "error": "session_expired",
                            "error_description": f"Session {session_id_header} expired or not found",
                            "action": "reconnect",
                        },
                    )
                logger.info("Session resumed: %s", session_id_header)
            else:
                # No session ID - create new session
                # Extract identifier from request.state (set by auth_validator)
                # Auth validator sets either oauth_client_id (OAuth) or connection_id (API key)
                oauth_client_id = getattr(request.state, "oauth_client_id", None)
                connection_id = getattr(request.state, "connection_id", None)
                identifier = oauth_client_id or connection_id

                if identifier is None:
                    logger.warning(
                        "Cannot create session: neither oauth_client_id nor connection_id found in request.state"
                    )
                    logger.warning(
                        "Auth validator must set request.state.oauth_client_id (OAuth) "
                        "or request.state.connection_id (API key)"
                    )
                    sse_401_headers: dict[str, str] = {}
                    if oauth_resource_metadata is not None:
                        base_url_str = _get_base_url(request)
                        resource_metadata_url = f"{base_url_str}/.well-known/oauth-protected-resource"
                        sse_401_headers["WWW-Authenticate"] = f'Bearer resource_metadata="{resource_metadata_url}"'
                    return JSONResponse(
                        status_code=401,
                        headers=sse_401_headers,
                        content={
                            "error": "unauthorized",
                            "error_description": "Authentication succeeded but no associated identity found",
                        },
                    )

                # Create new session for this identifier
                auth_type = "oauth_client" if oauth_client_id else "connection"
                logger.info("Creating new session for %s: %s", auth_type, identifier)
                session_id = await session_creator(oauth_client_id, connection_id)

                # Fetch newly created session data
                session_data = await session_getter(session_id)
                if session_data is None:
                    logger.error("Failed to retrieve newly created session: %s", session_id)
                    return JSONResponse(
                        status_code=500,
                        content={
                            "error": "internal_error",
                            "error_description": "Failed to create session",
                        },
                    )
                logger.info("New session created: %s", session_id)

            # Extract Last-Event-ID for SSE resumption.
            # Per AC-95: header "0" -> int 0 (not None).
            # Per AC-96: header absent -> None.
            last_event_id_header = request.headers.get("Last-Event-ID")
            if last_event_id_header is not None:
                try:
                    last_event_id: int | None = int(last_event_id_header)
                    logger.info("Resuming from Last-Event-ID: %d", last_event_id)
                except ValueError:
                    logger.warning("Invalid Last-Event-ID header: %s", last_event_id_header)
                    last_event_id = None
            else:
                last_event_id = None

            session_label = session_data.session_id if session_data else "new"

            # Create SSE event stream generator
            async def event_stream() -> AsyncGenerator[str]:
                """
                Generate SSE events for the client.

                Delivers application events from event_subscriber alongside
                30-second keepalive comments. The keepalive timer resets each
                time an application event is delivered.

                Yields:
                    SSE-formatted strings: application events as
                    ``id: N\\nevent: message\\ndata: ...\\n\\n``, or keepalive
                    comments as ``: keepalive\\n\\n``.
                """
                logger.info("SSE stream established for session: %s", session_label)
                yield ": SSE stream established\n\n"

                # Acquire the application-event generator if subscriber provided.
                sid = session_data.session_id if session_data else session_label
                gen = event_subscriber(sid, last_event_id) if event_subscriber is not None else None

                try:
                    if gen is None:
                        # AC-6: keepalive-only stream when no subscriber provided.
                        while shutdown_event is None or not shutdown_event.is_set():
                            if shutdown_event is not None:
                                try:
                                    await asyncio.wait_for(shutdown_event.wait(), timeout=30)
                                    # Shutdown event fired.
                                    break
                                except TimeoutError:
                                    pass
                            else:
                                await asyncio.sleep(30)
                            yield ": keepalive\n\n"
                    else:
                        # AC-1/AC-4: deliver events; keepalive fires every 30s between events.
                        # Poll every 1s so shutdown_event is checked promptly.
                        keepalive_ticks = 0
                        while shutdown_event is None or not shutdown_event.is_set():
                            try:
                                event_id, payload = await asyncio.wait_for(gen.__anext__(), timeout=1.0)
                                yield f"id: {event_id}\nevent: message\ndata: {json.dumps(payload)}\n\n"
                                keepalive_ticks = 0
                            except TimeoutError:
                                keepalive_ticks += 1
                                # AC-4: 30s elapsed without an event — send keepalive.
                                if keepalive_ticks >= 30:
                                    yield ": keepalive\n\n"
                                    keepalive_ticks = 0
                            except StopAsyncIteration:
                                # AC-87: empty or exhausted generator — close stream cleanly.
                                break

                    # Shutdown requested — notify client and close stream.
                    if shutdown_event is not None and shutdown_event.is_set():
                        yield ": server-shutdown\n\n"
                        if gen is not None:
                            await gen.aclose()
                except asyncio.CancelledError:
                    # AC-5: client disconnected — log session ID, do not crash.
                    logger.info("SSE stream cancelled for session: %s", session_label)
                    if gen is not None:
                        await gen.aclose()
                    raise

            # Return streaming response with Mcp-Session-Id header
            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={
                    "Mcp-Session-Id": session_data.session_id if session_data else "error",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

    @router.delete("", dependencies=dependencies, status_code=204)
    async def delete_mcp_session(request: Request) -> Response:
        """
        Delete an MCP session by session ID (AC-18, AC-19).

        Args:
            request: FastAPI Request object containing Mcp-Session-Id header

        Returns:
            204 No Content on successful deletion.
            404 Not Found when session_store absent or session not found.
        """
        if session_store is None:
            return JSONResponse(status_code=404, content={"error": "session_not_found"})
        session_id_hdr = request.headers.get("Mcp-Session-Id")
        if not session_id_hdr:
            return JSONResponse(status_code=404, content={"error": "session_not_found"})
        found = await session_store.get(session_id_hdr)
        if found is None:
            return JSONResponse(status_code=404, content={"error": "session_not_found"})
        await session_store.delete(session_id_hdr)
        logger.info("Session deleted: %s", session_id_hdr)
        return Response(status_code=204)

    def _parse_and_validate(
        raw_body: bytes,
    ) -> tuple[dict, str | None, object, dict] | JSONResponse:
        """Parse raw bytes into JSON-RPC body and extract fields.

        Returns (body, method, request_id, params) on success.
        Returns JSONResponse directly on any validation failure.
        Caller: if isinstance(result, JSONResponse): return result

        Args:
            raw_body: Raw request body bytes to parse.

        Returns:
            Tuple (body, method, request_id, params) on success, or
            JSONResponse on parse/validation failure.
        """
        if not raw_body or raw_body.strip() == b"":
            logger.info("Empty body received - returning server info")
            return JSONResponse(
                status_code=200,
                content={
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32700,
                        "message": "Parse error: Empty request body. Send JSON-RPC 2.0 formatted requests.",
                    },
                    "id": None,
                },
            )

        try:
            body = json.loads(raw_body)
        except ValueError as e:
            logger.error("JSON parse error: %s", e, exc_info=True)
            return json_rpc_error(
                request_id=None,
                code=-32700,
                message=f"Parse error: {e}",
            )

        logger.debug("Request body: %s", body)
        method = body.get("method")

        # Handle JSON-RPC responses from the client (sampling/elicitation correlation).
        # A response has "result" and "id" but no "method".
        if "result" in body and "id" in body and method is None:
            response_id = str(body["id"])
            response_result: dict = body["result"] if isinstance(body["result"], dict) else {}
            if _sampling_manager is not None:
                _sampling_manager.handle_response(response_id, response_result)
            if _elicitation_manager is not None:
                _elicitation_manager.handle_response(response_id, response_result)
            return JSONResponse(status_code=200, content={})

        if body.get("jsonrpc") != "2.0":
            logger.warning("Invalid JSON-RPC version: %s", body.get("jsonrpc"))
            return json_rpc_error(
                request_id=body.get("id"),
                code=-32600,
                message="Invalid JSON-RPC version",
            )

        request_id = body.get("id")
        params = body.get("params") or {}
        return (body, method, request_id, params)

    async def _validate_session_header(
        request: Request,
        method: str | None,
        lookup: Callable[[str], Awaitable[object | None]],
        context: str,
        missing_description: str,
    ) -> str | JSONResponse:
        """Validate Mcp-Session-Id header and resolve session.

        Args:
            request: FastAPI Request object.
            method: JSON-RPC method name, or None.
            lookup: Async callable to look up session data by session ID.
            context: Context label for log messages (e.g. "session_store").
            missing_description: Human-readable description when header is missing.

        Returns:
            Validated session ID string on success, JSONResponse on failure.
        """
        sid = request.headers.get("Mcp-Session-Id")
        if not sid:
            logger.warning("Missing Mcp-Session-Id header for %s (method: %s)", context, method)
            return JSONResponse(
                status_code=400,
                content={
                    "error": "missing_session_id",
                    "error_description": missing_description,
                },
            )
        logger.info("Validating session for %s: %s", context, sid)
        data = await lookup(sid)
        if data is None:
            logger.warning("Session not found or expired for %s: %s", context, sid)
            return JSONResponse(
                status_code=410,
                content={
                    "error": "session_expired",
                    "error_description": f"Session {sid} not found or expired",
                    "action": "reconnect",
                },
            )
        logger.info("Session validated for %s: %s", context, sid)
        return sid

    async def _resolve_session(
        method: str | None,
        request: Request,
        is_oauth_connection: bool,
        body: dict,
    ) -> tuple[str | None, str | None] | JSONResponse:
        """Resolve session state across 3 mutually exclusive paths.

        Returns (active_session_id, session_id_to_return) on success.
        Returns JSONResponse on auth/session failure.

        Args:
            method: JSON-RPC method name, or None.
            request: FastAPI Request object.
            is_oauth_connection: True when the connection uses Bearer token auth.
            body: Parsed JSON-RPC request body.

        Returns:
            Tuple (active_session_id, session_id_to_return) on success, or
            JSONResponse on auth/session failure.
        """
        session_id_to_return: str | None = None
        active_session_id: str | None = None

        # Log connection mode on initialize (new connection)
        sessions_enabled = session_getter is not None and session_creator is not None
        if method == "initialize":
            auth_method = "oauth" if is_oauth_connection else "api_key"
            if is_oauth_connection and sessions_enabled:
                connection_mode = "stateful"
            else:
                connection_mode = "stateless" if session_store is None else "stateful-store"
            logger.info(
                "New MCP connection: auth=%s mode=%s sessions_enabled=%s",
                auth_method,
                connection_mode,
                sessions_enabled,
            )

        # Path 1: session_store path — create session on initialize, validate on other methods
        if session_store is not None:
            if method == "initialize":
                protocol_ver_hdr = request.headers.get("MCP-Protocol-Version", "2025-03-26")
                _init_params = body.get("params", {})
                client_info = _init_params.get("clientInfo", {})
                capabilities_req = _init_params.get("capabilities", {})
                store_new = await session_store.create(
                    protocol_version=protocol_ver_hdr,
                    client_info=client_info,
                    capabilities=capabilities_req,
                )
                session_id_to_return = store_new.session_id
                active_session_id = store_new.session_id
                logger.info("session_store: new session created via initialize: %s", session_id_to_return)
            else:
                result = await _validate_session_header(
                    request,
                    method,
                    session_store.get,
                    "session_store",
                    "Mcp-Session-Id header required. Call initialize first.",
                )
                if isinstance(result, JSONResponse):
                    return result
                active_session_id = result
            return (active_session_id, session_id_to_return)

        # Path 2: callback-based OAuth path
        if session_getter is not None and session_creator is not None and is_oauth_connection:
            if method == "initialize":
                oauth_client_id = getattr(request.state, "oauth_client_id", None)
                connection_id = getattr(request.state, "connection_id", None)
                identifier = oauth_client_id or connection_id

                if identifier is None:
                    logger.warning(
                        "Cannot create session: neither oauth_client_id nor connection_id found in request.state"
                    )
                    logger.warning(
                        "Auth validator must set request.state.oauth_client_id (OAuth) "
                        "or request.state.connection_id (API key)"
                    )
                    post_init_401_headers: dict[str, str] = {}
                    if oauth_resource_metadata is not None:
                        post_init_base_url = _get_base_url(request)
                        post_init_rm_url = f"{post_init_base_url}/.well-known/oauth-protected-resource"
                        post_init_401_headers["WWW-Authenticate"] = f'Bearer resource_metadata="{post_init_rm_url}"'
                    return JSONResponse(
                        status_code=401,
                        headers=post_init_401_headers,
                        content={
                            "error": "unauthorized",
                            "error_description": "Authentication succeeded but no associated identity found",
                        },
                    )

                auth_type = "oauth_client" if oauth_client_id else "connection"
                logger.info(
                    "Creating new session for %s: %s via initialize",
                    auth_type,
                    identifier,
                )
                session_id_to_return = await session_creator(oauth_client_id, connection_id)

                session_data: McpSessionData | None = await session_getter(session_id_to_return)
                if session_data is None:
                    logger.error(
                        "Failed to retrieve newly created session: %s",
                        session_id_to_return,
                    )
                    return JSONResponse(
                        status_code=500,
                        content={
                            "error": "internal_error",
                            "error_description": "Failed to create session",
                        },
                    )
                logger.info("New session created via initialize: %s", session_id_to_return)
            else:
                result = await _validate_session_header(
                    request,
                    method,
                    session_getter,
                    "OAuth connection",
                    "Mcp-Session-Id header required for OAuth connections. Call initialize first.",
                )
                if isinstance(result, JSONResponse):
                    return result
                # active_session_id intentionally not set — callback-based sessions do not track it here

            return (active_session_id, session_id_to_return)

        # Path 3: stateless — return (None, None)
        return (None, None)

    async def _dispatch_method(
        method: str | None,
        params: dict,
        request_id: object,
        request: Request,
        background_tasks: BackgroundTasks,
        is_oauth_connection: bool,
        active_session_id: str | None,
        protocol_version: str,
    ) -> dict | JSONResponse:
        """Route method to handler, return result dict or early-exit JSONResponse.

        Notifications return JSONResponse(202) directly.
        Unknown method returns JSONResponse via json_rpc_error(-32601).
        All other methods return dict.

        Args:
            method: JSON-RPC method name, or None.
            params: Parsed params dict from JSON-RPC body.
            request_id: JSON-RPC request ID.
            request: FastAPI Request object.
            background_tasks: FastAPI BackgroundTasks for async operations.
            is_oauth_connection: True when connection uses Bearer token auth.
            active_session_id: Current session ID, or None in stateless mode.
            protocol_version: Negotiated MCP protocol version string.

        Returns:
            Result dict on success, or JSONResponse for notifications/errors.
        """
        if method == "notifications/initialized" and request_id is None:
            logger.info("Handling notifications/initialized")
            handle_initialized()
            return JSONResponse(status_code=202, content={})

        if method == "notifications/cancelled":
            request_id_to_cancel = params.get("requestId") or params.get("id") if params else None
            logger.debug(
                "Received notifications/cancelled for requestId: %s",
                request_id_to_cancel,
            )
            if _progress_tracker is not None and request_id_to_cancel is not None:
                _progress_tracker.request_cancellation(str(request_id_to_cancel))
            return JSONResponse(status_code=202, content={})

        if method == "initialize":
            logger.info(
                "Handling initialize request with protocol version: %s",
                protocol_version,
            )
            result = handle_initialize(params, protocol_version, server_info)
            capabilities = dict(cast(dict[str, object], result["capabilities"]))
            if resource_registry is not None and resource_registry.has_resources():
                capabilities["resources"] = {}
            if prompt_registry is not None and prompt_registry.has_prompts():
                capabilities["prompts"] = {}
            return {**result, "capabilities": capabilities}

        if method == "tools/list":
            logger.info("Handling tools/list request")
            excluded = tool_filter(is_oauth_connection) if tool_filter else None
            return handle_tools_list(registry, excluded_tools=excluded)

        if method == "tools/call":
            logger.info("Handling tools/call request: %s", params.get("name"))
            tool_name = params.get("name") if params else None

            # IR-6: create OTel span for tools/call when tracer is available.
            # EC-4: span creation failure must not break request handling.
            _tracer_any = cast(Any, _tracer)
            _span_cm: Any = None
            _span: Any = None
            if _tracer_any is not None:
                try:
                    _span_cm = _tracer_any.start_as_current_span(f"mcp tools/call {tool_name}")
                    _span = _span_cm.__enter__()
                    _span.set_attribute("rpc.system.name", "jsonrpc")
                    _span.set_attribute("rpc.method", "tools/call")
                    _span.set_attribute("rpc.jsonrpc.version", "2.0")
                    _span.set_attribute("mcp.tool.name", tool_name or "")
                except Exception as _span_exc:
                    logger.warning("OTel span creation failed for tools/call: %s", _span_exc)
                    _span_cm = None
                    _span = None

            # AC-42: create per-request progress callback in stateful mode
            # AC-43: pass None in stateless mode (registry injects no-op)
            try:
                if _progress_tracker is not None and active_session_id is not None:
                    _meta = params.get("_meta")
                    progress_token = _meta.get("progressToken") if isinstance(_meta, dict) else None
                    rpc_request_id = str(progress_token or request_id)
                    tracker_ref = _progress_tracker
                    captured_session_id = active_session_id
                    captured_request_id = rpc_request_id

                    async def _progress_callback(
                        current: int,
                        total: int,
                        message: str | None = None,
                    ) -> None:
                        await tracker_ref.report_progress(
                            captured_session_id,
                            captured_request_id,
                            current,
                            total,
                            message,
                        )
                        if tracker_ref.is_cancelled(captured_request_id):
                            raise asyncio.CancelledError(f"Request {captured_request_id} cancelled")

                    progress_cb: object | None = _progress_callback
                else:
                    progress_cb = None

                return await handle_tools_call(
                    registry,
                    params,
                    request,
                    background_tasks,
                    session_store=session_store,
                    session_id=active_session_id,
                    progress_callback=progress_cb,
                    sampling_manager=_sampling_manager,
                )
            except Exception as _tools_call_exc:
                if _span is not None:
                    with contextlib.suppress(Exception):
                        _span.set_attribute("error.type", type(_tools_call_exc).__name__)
                raise
            finally:
                if _span_cm is not None:
                    with contextlib.suppress(Exception):
                        _exc_info = sys.exc_info()
                        _span_cm.__exit__(_exc_info[0], _exc_info[1], _exc_info[2])

        if method == "resources/subscribe":
            logger.info("Handling resources/subscribe request")
            if session_store is None:
                raise MCPError(-32601, "resources/subscribe requires stateful mode")
            sub_uri = params.get("uri") if params else None
            if not sub_uri:
                raise MCPError(-32602, "Missing uri parameter")
            sub_session = await session_store.get(active_session_id or "")
            if sub_session is None:
                raise MCPError(-32602, "Session not found")
            if len(sub_session.subscriptions) >= _SUBSCRIPTIONS_MAX:
                raise MCPError(-32602, "Subscription limit exceeded (max 100)")
            sub_session.subscriptions.add(str(sub_uri))
            await session_store.update(sub_session)
            return {}

        if method == "resources/unsubscribe":
            logger.info("Handling resources/unsubscribe request")
            if session_store is None:
                raise MCPError(-32601, "resources/unsubscribe requires stateful mode")
            unsub_uri = params.get("uri") if params else None
            if not unsub_uri:
                raise MCPError(-32602, "Missing uri parameter")
            unsub_session = await session_store.get(active_session_id or "")
            if unsub_session is None:
                raise MCPError(-32602, "Session not found")
            unsub_session.subscriptions.discard(str(unsub_uri))
            await session_store.update(unsub_session)
            return {}

        if method == "ping":
            logger.info("Handling ping request")
            return {}

        if method == "resources/list":
            logger.info("Handling resources/list request")
            if resource_registry is None or not resource_registry.has_resources():
                raise MCPError(-32601, "Method not found: resources/list")
            resources = resource_registry.list_resources()
            templates = resource_registry.list_templates()
            return {
                "resources": [
                    {
                        "uri": r.uri,
                        "name": r.name,
                        "description": r.description,
                        "mimeType": r.mime_type,
                    }
                    for r in resources
                ],
                "resourceTemplates": [
                    {
                        "uriTemplate": t.uri_template,
                        "name": t.name,
                        "description": t.description,
                        "mimeType": t.mime_type,
                    }
                    for t in templates
                ],
            }

        if method == "resources/read":
            logger.info("Handling resources/read request")
            if resource_registry is None or not resource_registry.has_resources():
                raise MCPError(-32601, "Method not found: resources/read")
            uri = params.get("uri", "") if params else ""
            contents = await resource_registry.read_resource(str(uri), request=request)
            return {
                "contents": [
                    {
                        "uri": contents.uri,
                        "mimeType": contents.mime_type,
                        "text": contents.text,
                        "blob": contents.blob,
                    }
                ]
            }

        if method == "prompts/list":
            logger.info("Handling prompts/list request")
            if prompt_registry is None or not prompt_registry.has_prompts():
                raise MCPError(-32601, "Method not found: prompts/list")
            return {"prompts": prompt_registry.list_prompts()}

        if method == "prompts/get":
            logger.info("Handling prompts/get request")
            if prompt_registry is None or not prompt_registry.has_prompts():
                raise MCPError(-32601, "Method not found: prompts/get")
            prompt_name = params.get("name", "") if params else ""
            prompt_arguments = params.get("arguments", {}) if params else {}
            messages = await prompt_registry.get_prompt(
                str(prompt_name),
                dict(prompt_arguments) if isinstance(prompt_arguments, dict) else {},
            )
            return {"messages": messages}

        if method == "roots/list":
            logger.info("Handling roots/list request")
            roots = _roots_manager.list_roots()
            return {"roots": [{"uri": r["uri"], "name": r.get("name")} for r in roots]}

        if method == "logging/setLevel":
            logger.info("Handling logging/setLevel request")
            if _logging_handler is None:
                raise MCPError(
                    code=-32601,
                    message="logging/setLevel requires stateful mode",
                )
            level = params.get("level") if params else None
            if not level or not isinstance(level, str):
                raise MCPError(
                    code=-32602,
                    message="Missing required parameter: level",
                )
            _logging_handler.set_level(active_session_id or "", level)
            return {}

        if method == "completion/complete":
            logger.info("Handling completion/complete request")
            if completion_handler is None:
                raise MCPError(
                    code=-32601,
                    message="completion/complete not supported",
                )
            ref = params.get("ref", {}) if params else {}
            argument = params.get("argument", {}) if params else {}
            completion_result = await completion_handler(ref, argument)
            completion_dict: dict[str, object] = {}
            if isinstance(completion_result, dict):
                for k, v in completion_result.items():
                    completion_dict[str(k)] = v
            raw_values = completion_dict.get("values")
            if isinstance(raw_values, list):
                completion_dict["values"] = raw_values[:100]
            return {"completion": completion_dict}

        if method == "elicitation/create":
            logger.info("Handling elicitation/create request")
            if session_store is None or _elicitation_manager is None:
                raise MCPError(
                    code=-32601,
                    message="elicitation/create requires stateful mode",
                )
            elicit_message = params.get("message", "") if params else ""
            requested_schema = params.get("requestedSchema", {}) if params else {}
            return await _elicitation_manager.create(
                active_session_id or "",
                str(elicit_message),
                requested_schema if isinstance(requested_schema, dict) else {},
            )

        raise MCPError(-32601, f"Method not found: {method}")

    def _format_response(
        result: dict,
        request_id: object,
        request: Request,
        session_id_to_return: str | None,
    ) -> JSONResponse | StreamingResponse:
        """Wrap result dict in JSON-RPC envelope, select transport format.

        SSE when Accept contains text/event-stream AND session_store is not None.
        Adds Mcp-Session-Id header when session_id_to_return is not None.

        Args:
            result: Result dict to wrap in JSON-RPC response envelope.
            request_id: JSON-RPC request ID for the response envelope.
            request: FastAPI Request object for reading Accept header.
            session_id_to_return: Session ID to include in response header, or None.

        Returns:
            StreamingResponse for SSE clients, JSONResponse otherwise.
        """
        response = json_rpc_response(request_id, result)

        accept_header = request.headers.get("Accept", "")
        wants_sse = "text/event-stream" in accept_header
        if wants_sse and session_store is not None:
            raw_body = response.body
            result_body = bytes(raw_body).decode() if isinstance(raw_body, memoryview) else raw_body.decode()
            sse_headers: dict[str, str] = {"Cache-Control": "no-cache"}
            if session_id_to_return:
                sse_headers["Mcp-Session-Id"] = session_id_to_return
                logger.info(
                    "Including Mcp-Session-Id header in SSE response: %s",
                    session_id_to_return,
                )

            async def _single_event() -> AsyncGenerator[str]:
                yield f"data: {result_body}\n\n"

            return StreamingResponse(
                _single_event(),
                media_type="text/event-stream",
                headers=sse_headers,
            )

        if session_id_to_return:
            response.headers["Mcp-Session-Id"] = session_id_to_return
            logger.info(
                "Including Mcp-Session-Id header in response: %s",
                session_id_to_return,
            )

        return response

    def _validate_protocol_version(
        protocol_version: str | None,
        request_id: object,
    ) -> tuple[str, JSONResponse | None]:
        """Normalise and validate the MCP-Protocol-Version header value.

        Defaults a missing header to "2025-03-26" for backward compatibility.

        Args:
            protocol_version: Raw header value, or None when header is absent.
            request_id: JSON-RPC request ID used in error response.

        Returns:
            Tuple of (resolved version string, JSONResponse error or None).
            The error is a 400 response when the version is unsupported.
        """
        logger.debug("MCP-Protocol-Version header: %s", protocol_version)
        if not protocol_version:
            logger.info("Missing MCP-Protocol-Version header, assuming 2025-03-26 for backwards compatibility")
            protocol_version = "2025-03-26"
        if protocol_version not in ("2025-06-18", "2025-03-26"):
            logger.warning("Unsupported protocol version: %s", protocol_version)
            return protocol_version, JSONResponse(
                status_code=400,
                content={
                    "error": (
                        f"Unsupported protocol version: {protocol_version}. Supported versions: 2025-06-18, 2025-03-26"
                    )
                },
            )
        return protocol_version, None

    @router.post("", dependencies=dependencies, response_model=None)
    async def handle_mcp_request(
        request: Request,
        background_tasks: BackgroundTasks,
    ) -> JSONResponse | StreamingResponse:
        """
        Handle MCP JSON-RPC request.

        Processes MCP JSON-RPC requests with protocol version validation,
        method routing, and comprehensive error handling. Supports the
        following MCP methods:

        - initialize: Return server capabilities and protocol version
        - notifications/initialized: Post-initialization notification
        - tools/list: Return available tools from registry
        - tools/call: Execute tool with arguments
        - ping: Health check

        Args:
            request: FastAPI Request object containing headers and body

        Returns:
            JSONResponse with JSON-RPC formatted response or error. Success
            responses return HTTP 200 with result field. Protocol errors
            return HTTP 200 with error field (JSON-RPC convention). Header
            validation errors return HTTP 400 (transport level).

        Status Codes:
            - 200: Successful JSON-RPC response or JSON-RPC error
            - 400: Invalid request (missing session ID when required, invalid protocol version)
            - 401: Authentication required or identity not found
            - 410: Session expired or not found (stateful mode)
            - 500: Internal server error (session creation failures)

        Authentication:
            - Requires X-API-Key or Authorization: Bearer token
            - Unauthenticated requests return 401 with WWW-Authenticate header
            - Missing identity (oauth_client_id/connection_id) returns 401
            - This enables OAuth flow discovery per MCP spec

        Session Validation (when session callbacks provided):
            - OAuth connections require session management
            - "initialize" method creates new session and returns Mcp-Session-Id header
            - Other methods require Mcp-Session-Id header with valid session
            - Returns 410 Gone if session expired or not found
            - API key connections remain stateless (no session requirement)

        Protocol Version Validation:
            - MCP-Protocol-Version header REQUIRED on all requests
            - Only "2025-06-18" is accepted (strict validation)
            - Missing header returns HTTP 400 Bad Request
            - Invalid version returns HTTP 400 Bad Request
            - This enforces explicit protocol version on every request

        Error Handling:
            - MCPError: Returns JSON-RPC error response with error code
            - ToolError: Returns tool result with isError: true
            - Exception: Returns JSON-RPC error -32603 (Internal error)
            - Invalid JSON-RPC: Returns error -32600 (Invalid Request)
            - Method not found: Returns error -32601

        Example Requests:
            >>> # Initialize request
            >>> POST /mcp
            >>> Headers: {"MCP-Protocol-Version": "2025-06-18"}
            >>> Body: {
            ...     "jsonrpc": "2.0",
            ...     "id": 1,
            ...     "method": "initialize",
            ...     "params": {"protocolVersion": "2025-06-18"}
            ... }
            >>>
            >>> # Tools list request
            >>> POST /mcp
            >>> Headers: {"MCP-Protocol-Version": "2025-06-18"}
            >>> Body: {
            ...     "jsonrpc": "2.0",
            ...     "id": 2,
            ...     "method": "tools/list"
            ... }
            >>>
            >>> # Tool call request
            >>> POST /mcp
            >>> Headers: {"MCP-Protocol-Version": "2025-06-18"}
            >>> Body: {
            ...     "jsonrpc": "2.0",
            ...     "id": 3,
            ...     "method": "tools/call",
            ...     "params": {"name": "hello", "arguments": {"name": "World"}}
            ... }
        """
        body = None  # Initialize before try block to avoid UnboundLocalError in except handlers

        try:
            logger.info("Received MCP request")
            logger.debug("Request headers: %s", _sanitize_headers(dict(request.headers)))

            # 1. Auth
            _api_key, _bearer_token, is_valid = await _check_authentication(request, auth_validator)
            if not is_valid:
                logger.warning("Unauthenticated POST request to MCP endpoint")
                post_401_headers: dict[str, str] = {"WWW-Authenticate": 'Bearer realm="mcp"'}
                post_401_content: dict[str, object] = {
                    "error": "Authentication required",
                    "error_description": "Bearer token missing or invalid. Re-authenticate.",
                }
                if oauth_resource_metadata is not None:
                    post_base_url = _get_base_url(request)
                    post_rm_url = f"{post_base_url}/.well-known/oauth-protected-resource"
                    post_401_headers["WWW-Authenticate"] = f'Bearer resource_metadata="{post_rm_url}"'
                    post_401_content["resource_metadata"] = post_rm_url
                return JSONResponse(
                    status_code=401,
                    headers=post_401_headers,
                    content=post_401_content,
                )

            # 2. Parse
            raw_body = await request.body()
            parsed = _parse_and_validate(raw_body)
            if isinstance(parsed, JSONResponse):
                return parsed
            body, method, request_id, params = parsed

            # 3. Protocol version
            protocol_version, pv_err = _validate_protocol_version(
                request.headers.get("MCP-Protocol-Version"), request_id
            )
            if pv_err is not None:
                return pv_err
            # 4. OTel counter
            logger.info("Processing method: %s (id: %s)", method, request_id)
            if _request_counter is not None:
                with contextlib.suppress(Exception):  # Meter failure must not break request handling
                    _request_counter.add(1, {"rpc.method": method})

            # 5. Session
            is_oauth_connection = _bearer_token is not None
            session_result = await _resolve_session(method, request, is_oauth_connection, body)
            if isinstance(session_result, JSONResponse):
                return session_result
            active_session_id, session_id_to_return = session_result

            # 6. Dispatch
            dispatch_result = await _dispatch_method(
                method,
                params,
                request_id,
                request,
                background_tasks,
                is_oauth_connection,
                active_session_id,
                protocol_version,
            )
            if isinstance(dispatch_result, JSONResponse):
                return dispatch_result

            # 7. Format
            logger.info("Returning successful response for method: %s", method)
            return _format_response(dispatch_result, request_id, request, session_id_to_return)

        except ValueError as e:
            # ValueError should not occur here since we already parsed the body
            # This catch is for any other unexpected ValueError
            logger.error("Unexpected ValueError: %s", e, exc_info=True)
            return json_rpc_error(
                request_id=body.get("id") if body and isinstance(body, dict) else None,
                code=-32603,
                message=f"Internal error: {e}",
            )
        except ClientDisconnect:
            # Client disconnected before request completed - normal behavior
            logger.info("Client disconnected during request processing")
            return JSONResponse(status_code=499, content={})
        except MCPError as e:
            # Protocol-level errors return JSON-RPC error response
            logger.warning("MCP protocol error: %s (code: %s)", e.message, e.code)
            return json_rpc_error(
                request_id=body.get("id") if body and isinstance(body, dict) else None,
                code=e.code,
                message=e.message,
                data=e.data,
            )
        except Exception as e:
            # Unexpected errors return Internal error
            logger.error("Internal error: %s", e, exc_info=True)
            return json_rpc_error(
                request_id=body.get("id") if body and isinstance(body, dict) else None,
                code=-32603,
                message=f"Internal error: {e}",
            )

    return router


class MCPRouter(APIRouter):
    """Unified MCP router extending FastAPI APIRouter.

    Provides decorator-based tool, resource, and prompt registration alongside
    MCP JSON-RPC endpoint setup. All MCP endpoints (GET, POST, DELETE) are
    added automatically on construction.

    Use ``app.include_router(mcp, prefix="/mcp")`` to mount endpoints.

    Attributes:
        base_url: Base URL passed at construction (may be None).
        oauth_resource_metadata: RFC 9728 PRM fields passed at construction (may be None).
        _tool_registry: Internal MCPToolRegistry for tool registration
        _resource_registry: Internal ResourceRegistry for resource registration
        _prompt_registry: Internal PromptRegistry for prompt registration

    Example:
        >>> from fastapi import FastAPI
        >>> from fastapi_mcp_router import MCPRouter
        >>>
        >>> app = FastAPI()
        >>> mcp = MCPRouter()
        >>>
        >>> @mcp.tool()
        >>> async def hello(name: str) -> str:
        ...     '''Say hello.'''
        ...     return f"Hello, {name}!"
        >>>
        >>> app.include_router(mcp, prefix="/mcp")
    """

    def __init__(
        self,
        *,
        auth_validator: AuthValidator | None = None,
        session_store: SessionStore | None = None,
        session_getter: SessionGetter | None = None,
        session_creator: SessionCreator | None = None,
        event_subscriber: "EventSubscriber | None" = None,
        tool_filter: ToolFilter | None = None,
        server_info: "ServerInfo | None" = None,
        base_url: str | None = None,
        oauth_resource_metadata: dict[str, object] | None = None,
        rate_limit_dependency: Callable[..., Awaitable[None]] | None = None,
        stateful: bool = False,
        sampling_enabled: bool = False,
        legacy_sse: bool = False,
        enable_telemetry: bool = True,
    ) -> None:
        """Initialize MCPRouter with MCP configuration.

        Creates internal MCPToolRegistry, ResourceRegistry, and PromptRegistry
        instances. Adds all MCP endpoints to the router by calling the internal
        create_mcp_router() factory and including its routes.

        Args:
            auth_validator: Optional callback to validate authentication credentials.
            session_store: Optional SessionStore for stateful SSE mode. Enables
                stateful=True behaviour automatically.
            session_getter: Optional legacy callback to retrieve session data.
            session_creator: Optional legacy callback to create a new session.
            event_subscriber: Optional SSE event source callable.
            tool_filter: Optional per-connection tool filtering callback.
            server_info: Optional server metadata for MCP initialize response.
            base_url: Optional base URL to prevent host header injection.
            oauth_resource_metadata: Optional RFC 9728 PRM fields.
            rate_limit_dependency: Optional rate limiting dependency.
            stateful: If True, enables stateful mode. Requires session_store.
            sampling_enabled: If True, enables sampling. Requires stateful=True.
            legacy_sse: If True, enables legacy SSE transport compatibility mode.
            enable_telemetry: If True, enables telemetry collection (default True).

        Raises:
            ValueError: If stateful=True and session_store is None.
            ValueError: If sampling_enabled=True and stateful is False.
        """
        if stateful and session_store is None:
            raise ValueError("stateful=True requires session_store to be provided.")
        if sampling_enabled and not stateful:
            raise ValueError("sampling_enabled=True requires stateful=True.")

        super().__init__()

        self.base_url = base_url
        self.oauth_resource_metadata = oauth_resource_metadata
        self._tool_registry = MCPToolRegistry()
        self._resource_registry = ResourceRegistry()
        self._prompt_registry = PromptRegistry()
        self._shutdown_event: asyncio.Event = asyncio.Event()

        inner = create_mcp_router(
            registry=self._tool_registry,
            rate_limit_dependency=rate_limit_dependency,
            auth_validator=auth_validator,
            base_url=base_url,
            session_getter=session_getter,
            session_creator=session_creator,
            tool_filter=tool_filter,
            server_info=server_info,
            event_subscriber=event_subscriber,
            oauth_resource_metadata=oauth_resource_metadata,
            session_store=session_store,
            stateful=stateful,
            resource_registry=self._resource_registry,
            prompt_registry=self._prompt_registry,
            sampling_enabled=sampling_enabled,
            legacy_sse=legacy_sse,
            enable_telemetry=enable_telemetry,
            shutdown_event=self._shutdown_event,
        )
        # FastAPI blocks include_router when both prefix and route path are
        # empty strings. Since create_mcp_router() registers routes at ""
        # and MCPRouter has no inherent prefix, we extend routes directly.
        # The user-supplied prefix (e.g. prefix="/mcp") is applied later by
        # app.include_router(mcp, prefix="/mcp") as expected.
        self.routes.extend(inner.routes)

    async def shutdown(self) -> None:
        """Signal active SSE streams to close gracefully.

        Sets the internal shutdown event, causing all SSE generator loops to
        exit and yield a ``: server-shutdown`` comment before returning.

        Typical usage inside a FastAPI lifespan handler::

            @asynccontextmanager
            async def lifespan(app: FastAPI):
                yield
                await mcp.shutdown()
        """
        self._shutdown_event.set()

    def tool(
        self,
        name: str | None = None,
        description: str | None = None,
        input_schema: dict[str, object] | None = None,
        annotations: dict[str, object] | None = None,
    ) -> Callable:
        """Decorator to register an async function as an MCP tool.

        Delegates to the internal MCPToolRegistry.

        Args:
            name: Tool name (defaults to function name)
            description: Tool description (defaults to function docstring)
            input_schema: JSON schema for parameters (auto-generated if not provided)
            annotations: MCP annotations for tool capabilities

        Returns:
            Decorator that returns the original function unchanged

        Raises:
            TypeError: If decorated function is not async

        Example:
            >>> @mcp.tool()
            >>> async def search(query: str) -> str:
            ...     '''Search for items.'''
            ...     return f"results for {query}"
        """
        return self._tool_registry.tool(
            name=name,
            description=description,
            input_schema=input_schema,
            annotations=annotations,
        )

    def resource(
        self,
        uri: str,
        name: str | None = None,
        description: str | None = None,
        mime_type: str | None = None,
    ) -> Callable:
        """Decorator to register an async function as an MCP resource handler.

        Delegates to the internal ResourceRegistry.

        Args:
            uri: URI template string with optional {param} placeholders
            name: Resource name (defaults to function name)
            description: Resource description (defaults to function docstring)
            mime_type: Optional MIME type override

        Returns:
            Decorator that returns the original function unchanged

        Raises:
            TypeError: If the decorated function is not async

        Example:
            >>> @mcp.resource(uri="docs://{slug}", name="Document")
            >>> async def get_doc(slug: str) -> str:
            ...     '''Fetch a document by slug.'''
            ...     return f"Content of {slug}"
        """
        return self._resource_registry.resource(
            uri_template=uri,
            name=name,
            description=description,
            mime_type=mime_type,
        )

    def prompt(
        self,
        name: str | None = None,
        description: str | None = None,
        arguments: list[dict[str, object]] | None = None,
    ) -> Callable:
        """Decorator to register a function as an MCP prompt.

        Delegates to the internal PromptRegistry. The ``arguments`` parameter
        is accepted for API symmetry with the spec but argument metadata is
        auto-generated from the function signature by the PromptRegistry.

        Args:
            name: Prompt name (defaults to function name)
            description: Prompt description (defaults to function docstring)
            arguments: Reserved for future use; arguments are auto-generated
                from the function signature

        Returns:
            Decorator that returns the original function unchanged

        Example:
            >>> @mcp.prompt()
            >>> async def greet(username: str) -> list[dict]:
            ...     '''Greet a user.'''
            ...     return [{"role": "user", "content": f"Hello {username}"}]
        """
        return self._prompt_registry.prompt(
            name=name,
            description=description,
        )

    def add_resource_provider(self, uri_prefix: str, provider: "ResourceProvider") -> None:
        """Register a ResourceProvider for all URIs starting with uri_prefix.

        Delegates to the internal ResourceRegistry.

        Args:
            uri_prefix: URI prefix; all URIs starting with this value are
                dispatched to the given provider
            provider: ResourceProvider instance to handle matching URIs

        Example:
            >>> from fastapi_mcp_router.resources import FileResourceProvider
            >>> mcp.add_resource_provider("file:///data/docs",
            ...     FileResourceProvider("/data/docs"))
        """
        self._resource_registry.register_provider(uri_prefix, provider)


def create_prm_router(
    oauth_resource_metadata: dict[str, object] | None = None,
    *,
    mcp: "MCPRouter | None" = None,
) -> APIRouter:
    """
    Create a root-level FastAPI router for the OAuth PRM endpoint.

    Registers GET /.well-known/oauth-protected-resource per RFC 9728. Mount
    this router on the main FastAPI app with no prefix so the path is absolute.

    Exactly one of ``mcp`` or ``oauth_resource_metadata`` must be provided.
    When ``mcp`` is provided, the metadata is derived from the router's
    ``base_url`` and ``oauth_resource_metadata`` attributes.

    Args:
        mcp: MCPRouter instance. Derives ``resource`` from ``mcp.base_url + "/mcp"``
            and ``authorization_servers`` from ``mcp.oauth_resource_metadata``.
            Mutually exclusive with ``oauth_resource_metadata``.
        oauth_resource_metadata: RFC 9728 Protected Resource Metadata dict.
            Must contain "resource" and "authorization_servers" keys.
            Mutually exclusive with ``mcp``.

    Returns:
        APIRouter with GET /.well-known/oauth-protected-resource registered.

    Raises:
        TypeError: If both ``mcp`` and ``oauth_resource_metadata`` are provided.
        TypeError: If neither ``mcp`` nor ``oauth_resource_metadata`` is provided.
        ValueError: If the resolved metadata is missing "resource" or
            "authorization_servers" keys.

    Example:
        >>> from fastapi import FastAPI
        >>> from fastapi_mcp_router import MCPRouter, create_prm_router
        >>>
        >>> app = FastAPI()
        >>> mcp = MCPRouter(
        ...     base_url="https://api.example.io",
        ...     oauth_resource_metadata={
        ...         "resource": "https://api.example.io/mcp",
        ...         "authorization_servers": ["https://auth.example.io"],
        ...     },
        ... )
        >>> prm_router = create_prm_router(mcp=mcp)
        >>> app.include_router(prm_router)  # No prefix — path is absolute
        >>>
        >>> # Legacy call pattern still works:
        >>> prm_router = create_prm_router(oauth_resource_metadata={
        ...     "resource": "https://api.example.io/mcp",
        ...     "authorization_servers": ["https://auth.example.io"],
        ... })
        >>> app.include_router(prm_router)
    """
    if mcp is not None and oauth_resource_metadata is not None:
        raise TypeError("mcp and oauth_resource_metadata are mutually exclusive")
    if mcp is None and oauth_resource_metadata is None:
        raise TypeError("one of mcp or oauth_resource_metadata is required")

    if mcp is not None:
        resource = f"{mcp.base_url}/mcp" if mcp.base_url is not None else None
        mcp_metadata = mcp.oauth_resource_metadata or {}
        oauth_resource_metadata = dict(mcp_metadata)
        if resource is not None:
            oauth_resource_metadata["resource"] = resource

    assert oauth_resource_metadata is not None  # narrowing for type checker
    missing = [key for key in ("resource", "authorization_servers") if key not in oauth_resource_metadata]
    if missing:
        raise ValueError(f"oauth_resource_metadata is missing required keys: {missing}")

    prm_router = APIRouter()

    @prm_router.get("/.well-known/oauth-protected-resource", include_in_schema=False)
    async def get_prm() -> JSONResponse:
        """
        Return OAuth Protected Resource Metadata per RFC 9728.

        Returns:
            JSONResponse with Content-Type application/json and the
            configured resource metadata fields.
        """
        return JSONResponse(
            status_code=200,
            content=oauth_resource_metadata,
        )

    return prm_router


def handle_initialize(
    params: dict[str, object],
    protocol_version: str,
    server_info: ServerInfo | None = None,
) -> dict[str, object]:
    """
    Handle initialize request.

    Returns server capabilities and protocol version information. This is the
    first request in the MCP protocol handshake, establishing the protocol
    version and server capabilities.

    This implementation returns stateless server configuration without sessionId,
    suitable for AWS Lambda and other stateless environments. If stateful sessions
    are needed, uncomment the sessionId field and implement session management.

    Args:
        params: Initialize request parameters. Expected to contain
            protocolVersion field, though this implementation doesn't
            validate it (validation happens in router via header check).
        protocol_version: Negotiated protocol version (from header or defaulted)
        server_info: Optional custom server metadata to merge with defaults.
            Supports fields like name, version, title, description, icons,
            and websiteUrl for Claude Custom Connectors branding.

    Returns:
        Dictionary containing:
        - protocolVersion: Negotiated version (2025-06-18 or 2025-03-26)
        - capabilities: {"tools": {}} (tool support enabled)
        - serverInfo: Merged server info with defaults for name and version

        Optional fields not included in stateless mode:
        - sessionId: Would be included for stateful sessions

    Example:
        >>> result = handle_initialize({"protocolVersion": "2025-06-18"}, "2025-06-18")
        >>> result["protocolVersion"]
        '2025-06-18'
        >>> result["capabilities"]
        {'tools': {}}
        >>> result["serverInfo"]["name"]
        'fastapi-mcp-router'
        >>>
        >>> # With custom server_info
        >>> custom_info = {"name": "my-server", "title": "My Server"}
        >>> result = handle_initialize({}, "2025-06-18", custom_info)
        >>> result["serverInfo"]["name"]
        'my-server'
        >>> result["serverInfo"]["title"]
        'My Server'
    """
    default_info: dict[str, object] = {"name": "fastapi-mcp-router", "version": "0.1.0"}
    return {
        "protocolVersion": protocol_version,
        "capabilities": {"tools": {}},
        "serverInfo": {**default_info, **(server_info or {})},
        # Optional: include sessionId for stateful sessions
        # "sessionId": str(uuid.uuid4())
    }


def handle_tools_list(
    registry: MCPToolRegistry,
    excluded_tools: list[str] | None = None,
) -> dict[str, object]:
    """
    Handle tools/list request.

    Retrieves all registered tools from the registry and returns them in
    MCP format. Each tool includes name, description, and JSON schema for
    input validation. Optionally filters out tools by name.

    Args:
        registry: Tool registry containing registered MCP tools
        excluded_tools: Optional list of tool names to exclude from the response.
            If None or empty, all tools are returned.

    Returns:
        Dictionary containing:
        - tools: List of tool definitions, each with name, description, and
          inputSchema fields

    Example:
        >>> registry = MCPToolRegistry()
        >>> @registry.tool()
        >>> async def hello(name: str) -> str:
        ...     '''Say hello.'''
        ...     return f"Hello, {name}!"
        >>>
        >>> result = handle_tools_list(registry)
        >>> len(result["tools"])
        1
        >>> result["tools"][0]["name"]
        'hello'
        >>> result["tools"][0]["description"]
        'Say hello.'
        >>>
        >>> # With exclusion
        >>> result = handle_tools_list(registry, excluded_tools=["hello"])
        >>> len(result["tools"])
        0
    """
    tools = registry.list_tools()
    if excluded_tools:
        tools = [t for t in tools if t["name"] not in excluded_tools]
    return {"tools": tools}


async def handle_tools_call(
    registry: MCPToolRegistry,
    params: dict[str, object],
    request: Request,
    background_tasks: BackgroundTasks,
    session_store: SessionStore | None = None,
    session_id: str | None = None,
    progress_callback: object | None = None,
    sampling_manager: object | None = None,
) -> dict[str, object]:
    """
    Handle tools/call request.

    Executes a registered tool with the provided arguments and returns the
    result in MCP format. Handles both successful execution and tool errors.

    For stateful mode with a generator tool (is_generator=True) and an active
    session_store + session_id: the registry returns the raw AsyncGenerator.
    The router collects all yielded dicts synchronously and returns them as a
    list in the POST response body (IR-3). If the generator raises mid-iteration
    the response contains isError: true with the error message (EC-4).

    For stateless mode or non-generator tools, behaviour is unchanged.

    Tool execution errors (ToolError) return a result with isError: true,
    allowing the LLM to see the error and potentially recover. Protocol
    errors (MCPError) are raised and handled by the router.

    Args:
        registry: Tool registry containing registered MCP tools
        params: Tool call parameters containing:
            - name: Tool name to execute (required)
            - arguments: Dictionary of tool arguments (optional, defaults to {})
        request: FastAPI Request object for dependency injection
        background_tasks: FastAPI BackgroundTasks for async task scheduling
        session_store: Optional SessionStore for stateful mode detection.
            When provided with session_id, generator results are collected
            synchronously and returned in the POST response (IR-3).
        session_id: Active session ID. Required when session_store is provided
            and tool is a generator.
        progress_callback: Optional ProgressCallback injected into tools that
            declare a ``progress: ProgressCallback`` parameter. In stateful mode
            this reports progress notifications via the session SSE stream and
            raises asyncio.CancelledError when the request is cancelled (AC-42).
            Pass None in stateless mode; the registry injects a no-op (AC-43).
        sampling_manager: Optional SamplingManager injected into tools that
            declare a ``sampling_manager: SamplingManager`` parameter. Enables
            server-to-client LLM sampling requests (AC-52, AC-53).

    Returns:
        Dictionary containing:
        - content: List of content items (currently single text item)
        - isError: Optional boolean, true if tool execution failed

        Success result structure:
        {
            "content": [
                {"type": "text", "text": <result as string or JSON>}
            ]
        }

        Stateful generator collected structure (IR-3):
        {
            "content": [<dict>, ...]
        }

        Error result structure:
        {
            "content": [
                {"type": "text", "text": <error message>}
            ],
            "isError": true
        }

    Raises:
        MCPError: If tool name is missing (-32602 Invalid params)
        MCPError: Propagated from registry.call_tool for protocol errors

    Example:
        >>> registry = MCPToolRegistry()
        >>> @registry.tool()
        >>> async def add(a: int, b: int) -> int:
        ...     '''Add two numbers.'''
        ...     return a + b
        >>>
        >>> # Successful execution
        >>> result = await handle_tools_call(
        ...     registry,
        ...     {"name": "add", "arguments": {"a": 2, "b": 3}}
        ... )
        >>> result["content"][0]["text"]
        '5'
        >>> "isError" in result
        False
        >>>
        >>> # Tool error (business logic failure)
        >>> @registry.tool()
        >>> async def divide(a: int, b: int) -> float:
        ...     '''Divide two numbers.'''
        ...     if b == 0:
        ...         raise ToolError("Cannot divide by zero")
        ...     return a / b
        >>>
        >>> result = await handle_tools_call(
        ...     registry,
        ...     {"name": "divide", "arguments": {"a": 10, "b": 0}}
        ... )
        >>> result["content"][0]["text"]
        'Cannot divide by zero'
        >>> result["isError"]
        True
    """
    tool_name = params.get("name")
    arguments = params.get("arguments", {})

    if not tool_name or not isinstance(tool_name, str):
        raise MCPError(code=-32602, message="Missing required parameter: name")

    if not isinstance(arguments, dict):
        raise MCPError(code=-32602, message="Invalid arguments: must be an object")

    # Type narrowing for ty - after isinstance check, arguments is dict[str, object]
    tool_arguments = cast(dict[str, object], arguments)

    # Stateful generator mode: registry returns raw AsyncGenerator for synchronous collection
    use_stateful_streaming = session_store is not None and session_id is not None

    # Call tool - may raise MCPError or ToolError
    try:
        result = await registry.call_tool(
            tool_name,
            tool_arguments,
            request,
            background_tasks,
            stateful=use_stateful_streaming,
            progress_callback=progress_callback,
            sampling_manager=sampling_manager,
        )

        # IR-3: stateful generator — collect all yielded items into a list for the POST response
        if use_stateful_streaming and hasattr(result, "__aiter__"):
            gen_result = cast(AsyncGenerator[dict], result)
            try:
                collected = await registry._collect_generator(gen_result)
            except ToolError as gen_err:
                # EC-4: generator raised mid-iteration — return isError response
                return {
                    "content": [{"type": "text", "text": gen_err.message}],
                    "isError": True,
                }
            return {"content": collected}

        # Pass through structured content if tool returned MCP wire format directly
        if isinstance(result, dict) and "structuredContent" in result:
            wire: dict[str, object] = {str(k): v for k, v in result.items()}
            return wire  # Already in correct MCP wire format with structuredContent + content

        # Wrap successful result in MCP format
        # Convert dict/list results to JSON string, keep strings as-is
        result_text = json.dumps(result) if not isinstance(result, str) else result

        return {"content": [{"type": "text", "text": result_text}]}

    except ToolError as e:
        # Business logic errors return isError: true
        # This allows LLM to see the error and potentially recover
        return {
            "content": [{"type": "text", "text": e.message}],
            "isError": True,
        }


def handle_initialized() -> dict[str, object]:
    """
    Handle notifications/initialized notification.

    Processes the post-initialization notification sent by the client after
    receiving the initialize response. This is a JSON-RPC notification (no id
    field), so no response is expected per the JSON-RPC specification.

    In a stateless implementation, this function is called but performs no
    action. In a stateful implementation, this could be used to mark the
    session as fully initialized or trigger post-initialization setup.

    Returns:
        Empty dictionary. Notifications do not return responses per JSON-RPC
        2.0 specification. The router handles this by returning HTTP 202
        Accepted with empty body.

    Note:
        The router detects this is a notification by checking for the absence
        of the "id" field in the request body. Per JSON-RPC 2.0:
        - Request with id field = expects response
        - Request without id field = notification, no response expected

    Example:
        >>> # This is called by router when receiving notification
        >>> handle_initialized()
        {}
        >>> # Router returns HTTP 202 Accepted, no JSON-RPC response sent
    """
    # Notification acknowledged, no action needed in stateless mode
    return {}
