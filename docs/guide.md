# Guide

This guide explains the concepts, patterns, and features of fastapi-mcp-router. It assumes you have completed the [Quick Start](quickstart.md).

## Architecture

The package contains 10 modules with a strict dependency flow:

```
router.py --> registry.py --> exceptions.py
   |              |
   |--> session.py --> types.py, exceptions.py
   |--> resources.py --> exceptions.py
   |--> prompts.py --> exceptions.py
   |--> telemetry.py (leaf, zero internal imports)
   |--> protocol.py
   |--> types.py
```

All MCP communication flows through a single prefix (e.g., `/mcp`) using JSON-RPC 2.0 method routing over HTTP. The library supports two modes:

| Mode | Transport | Sessions | SSE | Lambda |
|------|-----------|----------|-----|--------|
| Stateless | HTTP POST (JSON only) | No | No | Yes |
| Stateful | Streamable HTTP POST (JSON or SSE via Accept header) | Yes | Yes | No |

With `legacy_sse=True`, a GET endpoint registers alongside POST for backward compatibility with older MCP clients.

## Two Ways to Create a Router

### MCPRouter (recommended)

`MCPRouter` extends FastAPI's `APIRouter`. It creates internal registries for tools, resources, and prompts and exposes `@mcp.tool()`, `@mcp.resource()`, and `@mcp.prompt()` decorators directly:

```python
from fastapi import FastAPI
from fastapi_mcp_router import MCPRouter

app = FastAPI()
mcp = MCPRouter()

@mcp.tool()
async def search(query: str) -> list[dict]:
    """Search items."""
    return [{"id": 1, "match": query}]

app.include_router(mcp, prefix="/mcp")
```

### create_mcp_router() factory

The factory function accepts external registry instances. Use this when you need to share registries across modules or compose from multiple sources:

```python
from fastapi_mcp_router import MCPToolRegistry, ResourceRegistry, PromptRegistry, create_mcp_router

tools = MCPToolRegistry()
resources = ResourceRegistry()
prompts = PromptRegistry()

@tools.tool()
async def ping() -> str:
    """Health check."""
    return "pong"

router = create_mcp_router(
    tools,
    resource_registry=resources,
    prompt_registry=prompts,
)
app.include_router(router, prefix="/mcp")
```

## Tools

### Registration

Decorate any async function with `@mcp.tool()` or `@registry.tool()`. The decorator:

1. Extracts the function name as the tool name (override with `name=`)
2. Uses the docstring as the description (override with `description=`)
3. Auto-generates a JSON Schema from the function signature using Pydantic `TypeAdapter`

```python
@mcp.tool()
async def create_task(
    title: str,
    priority: int = 1,
    tags: list[str] | None = None,
) -> dict:
    """Create a task with optional priority and tags."""
    return {"title": title, "priority": priority, "tags": tags or []}
```

The generated input schema:

```json
{
  "type": "object",
  "properties": {
    "title": {"type": "string"},
    "priority": {"type": "integer", "default": 1},
    "tags": {
      "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}],
      "default": null
    }
  },
  "required": ["title"]
}
```

### Supported Type Hints

Schema generation supports all standard Python types:

- `str`, `int`, `bool`, `float`
- `list[T]`, `dict[K, V]`, `tuple[T, ...]`
- `Optional[T]` / `T | None`
- `Literal["a", "b"]` (produces `enum`)
- `Union[str, int]`
- Pydantic `BaseModel` subclasses (nested objects)

### Tool Annotations

MCP 2025-06-18 tool annotations provide hints to LLM clients:

```python
@mcp.tool(annotations={"readOnlyHint": True})
async def get_status() -> dict:
    """Read-only status check."""
    return {"healthy": True}
```

### Output Schemas

Structured content enables typed tool results:

```python
@mcp.tool(output_schema={
    "type": "object",
    "properties": {"score": {"type": "number"}},
    "required": ["score"],
})
async def analyze(text: str) -> dict:
    """Analyze text and return a score."""
    return {"score": 0.95}
```

When `output_schema` is set, `tools/call` returns both `structuredContent` (the dict) and backward-compatible `content` (text). Omitting `output_schema` preserves the default text content format.

### Dependency Injection

The registry auto-injects several FastAPI and MCP types when they appear in a tool signature:

| Type | Source | Behavior |
|------|--------|----------|
| `Request` | FastAPI | Injected from the HTTP request object |
| `BackgroundTasks` | FastAPI | Injected for post-response task scheduling |
| `ProgressCallback` | MCP types | Injected for progress reporting (no-op in stateless) |
| `SamplingManager` | MCP session | Injected for server-to-client LLM sampling |
| `Depends(fn)` | FastAPI | Resolved at call time, including nested `Request` deps |

These parameters are excluded from the generated JSON Schema. The LLM client never sees them.

```python
from fastapi import Request, BackgroundTasks
from fastapi_mcp_router import ProgressCallback

@mcp.tool()
async def process(
    data: str,
    request: Request,
    tasks: BackgroundTasks,
    progress: ProgressCallback,
) -> dict:
    """Process data with full injection."""
    api_key = request.headers.get("x-api-key")
    await progress(0, 1, "Processing")
    tasks.add_task(log_usage, api_key)
    return {"result": data.upper()}
```

### Streaming Tools

Return `AsyncGenerator[dict, None]` for incremental results:

```python
from collections.abc import AsyncGenerator

@mcp.tool()
async def process_batch(items: list[str]) -> AsyncGenerator[dict, None]:
    """Process items one at a time."""
    for item in items:
        yield {"item": item, "status": "done"}
```

In stateful mode, each `yield` streams as an SSE event. In stateless mode, the generator collects all yields into a single JSON-RPC response.

Non-dict yields and generator exceptions produce a `ToolError` with `isError: true`.

### Explicit Schemas

When a tool uses FastAPI `Depends()`, auto-generation cannot introspect the dependency. Provide `input_schema` explicitly:

```python
from fastapi import Depends

@mcp.tool(input_schema={
    "type": "object",
    "properties": {"message": {"type": "string"}},
    "required": ["message"],
})
async def send(message: str, db=Depends(get_db)) -> dict:
    """Send a message."""
    await db.insert(message)
    return {"sent": True}
```

## Resources

### Decorator-Based Resources

Register resources with URI templates (RFC 6570):

```python
@mcp.resource("config://{env}/settings")
async def env_settings(env: str) -> dict:
    """Return environment configuration."""
    return {"env": env, "debug": False}
```

Parameters in `{braces}` map to function arguments. Resources without parameters are static:

```python
@mcp.resource("time://now", name="Current Time")
async def current_time() -> str:
    """Return UTC timestamp."""
    from datetime import UTC, datetime
    return datetime.now(UTC).isoformat()
```

### Provider-Based Resources

For dynamic resource collections (filesystems, databases), implement `ResourceProvider` or use the built-in `FileResourceProvider`:

```python
from fastapi_mcp_router.resources import FileResourceProvider

mcp.add_resource_provider(
    "file://docs/",
    FileResourceProvider(
        root_path="/var/data/docs",
        allowed_extensions={".txt", ".md", ".json"},
    ),
)
```

`FileResourceProvider` enforces:

- Path traversal protection (rejects `..`)
- 10 MB file size limit
- Extension allowlist (default: `.txt`, `.md`, `.json`, `.yaml`)

### Custom Providers

Implement the `ResourceProvider` ABC:

```python
from fastapi_mcp_router.resources import ResourceProvider, Resource, ResourceContents

class DatabaseProvider(ResourceProvider):
    def list_resources(self) -> list[Resource]:
        return [Resource(uri="db://users", name="Users", description="User table")]

    async def read_resource(self, uri: str) -> ResourceContents:
        data = await fetch_from_db(uri)
        return ResourceContents(uri=uri, text=json.dumps(data))

    def subscribe(self, uri: str) -> bool:
        return False  # Subscriptions not supported

    def unsubscribe(self, uri: str) -> bool:
        return False

    async def watch(self):
        raise NotImplementedError
```

### Resource Subscriptions

In stateful mode, clients subscribe to resource changes. The server notifies subscribers when resources update. Subscriptions are per-session with a maximum of 100 per session.

## Prompts

Register prompt templates with auto-generated argument metadata:

```python
@mcp.prompt()
async def summarize(text: str, max_length: int = 500) -> list[dict]:
    """Summarize text to a target length."""
    return [{"role": "user", "content": f"Summarize in {max_length} chars:\n{text}"}]
```

The registry inspects the function signature:

- Parameters without defaults become **required** arguments
- Parameters with defaults become **optional** arguments
- The `self` parameter is filtered out

Both sync and async handlers are supported. The handler must return `list[dict]` where each dict has `role` ("user" or "assistant") and `content` (str) keys.

## Sessions and Stateful Mode

### Enabling Stateful Mode

Pass a `SessionStore` and set `stateful=True`:

```python
from fastapi_mcp_router import MCPRouter, InMemorySessionStore

mcp = MCPRouter(
    session_store=InMemorySessionStore(ttl_seconds=3600),
    stateful=True,
)
```

Stateful mode adds HTTP endpoints at the mounted prefix:

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/mcp` | Streamable HTTP: JSON or SSE response based on `Accept` header |
| DELETE | `/mcp` | Session termination (204 on success) |
| GET | `/mcp` | Legacy SSE stream (only when `legacy_sse=True`) |

The `Mcp-Session-Id` header tracks sessions across requests. Clients requesting `Accept: text/event-stream` on POST receive a `StreamingResponse` with the JSON-RPC result as an SSE data event. Without that header (or with `Accept: application/json`), POST returns a standard `JSONResponse`.

### Session Stores

**InMemorySessionStore** stores sessions in a dict with asyncio.Lock protection. Suitable for single-instance deployments. Sessions expire after `ttl_seconds` of inactivity. All state is lost on restart.

**RedisSessionStore** uses Redis for multi-instance deployments:

```python
from fastapi_mcp_router.session import RedisSessionStore
import redis.asyncio as redis

client = redis.Redis(host="localhost")
store = RedisSessionStore(redis_client=client, ttl_seconds=7200)
```

Keys follow the pattern `mcp:session:{id}` (JSON) and `mcp:queue:{id}` (list). Requires the `redis` package.

**Custom SessionStore** — implement the `SessionStore` ABC with 6 async methods: `create`, `get`, `update`, `delete`, `enqueue_message`, `dequeue_messages`.

### Session Lifecycle

1. Client sends `initialize` via POST (with or without `Accept: text/event-stream`)
2. Server creates a session, returns `Mcp-Session-Id` header
3. Client sends requests via POST with `Mcp-Session-Id`
4. For SSE streaming: client POSTs with `Accept: text/event-stream` to receive SSE responses
5. With `legacy_sse=True`: client may also open GET SSE stream for server-push events
6. Client sends DELETE to terminate, or session expires after TTL

### Message Queue

Each session has a bounded message queue (1000 messages max). Enqueue silently drops messages when the queue is full. Dequeue returns all messages and clears the queue atomically.

## Progress Notifications

Add `ProgressCallback` to a tool signature for long-running operations:

```python
from fastapi_mcp_router import ProgressCallback

@mcp.tool()
async def train_model(dataset: str, progress: ProgressCallback) -> dict:
    """Train with progress reporting."""
    for i in range(100):
        await progress(i, 100, f"Processing batch {i}")
    return {"status": "complete"}
```

`ProgressCallback` takes three arguments:

| Argument | Type | Description |
|----------|------|-------------|
| `current` | `int` | Units completed |
| `total` | `int` | Total units |
| `message` | `str \| None` | Optional status text |

In stateful mode, each call emits a `notifications/progress` SSE event. In stateless mode, progress calls are silently dropped (a no-op callback is injected). Tools without `progress` in their signature are unaffected.

## SSE Event Delivery

Supply an `EventSubscriber` callback for application-driven server-to-client events (requires `legacy_sse=True`):

```python
from fastapi_mcp_router import EventSubscriber, MCPRouter, InMemorySessionStore

async def my_events(session_id: str, last_event_id: int | None):
    """Yield (event_id, json_rpc_notification) tuples."""
    event_id = (last_event_id or 0) + 1
    yield event_id, {
        "jsonrpc": "2.0",
        "method": "notifications/message",
        "params": {"level": "info", "data": "heartbeat"},
    }

mcp = MCPRouter(
    session_store=InMemorySessionStore(),
    stateful=True,
    event_subscriber=my_events,
)
```

The SSE stream merges application events with protocol messages (progress, sampling, logs). The `Last-Event-ID` header enables stream resumption after disconnection.

## Authentication

### Auth Validator

Pass an `auth_validator` callback to validate credentials on every request:

```python
import secrets
from typing import Any

VALID_KEY = "sk-production-key"

async def auth_validator(api_key: str | None, bearer_token: str | None) -> Any:
    if api_key:
        if not secrets.compare_digest(api_key, VALID_KEY):
            return False
        return True
    if bearer_token:
        claims = await validate_oauth_token(bearer_token)  # returns dict or None
        return claims  # dict stored at request.state.auth_context
    return False

mcp = MCPRouter(auth_validator=auth_validator)
```

The callback receives:

- `api_key`: value from the `X-API-Key` header, or `None`
- `bearer_token`: token from the `Authorization: Bearer <token>` header, or `None`

**Return value semantics:**

- Return a falsy value (`None`, `False`, `0`, `""`, `[]`) — the request is rejected with HTTP 401.
- Return any truthy value — the request proceeds and the value is stored at `request.state.auth_context`.
- Return `True` — backward-compatible path; `request.state.auth_context` is `True`.
- Return a dict (e.g. decoded JWT claims) — `request.state.auth_context` holds the dict, accessible in tool handlers:

```python
@registry.tool()
async def my_tool(request: Request) -> str:
    ctx = request.state.auth_context  # dict with claims, or True
    user_id = ctx.get("sub") if isinstance(ctx, dict) else None
    return f"Hello {user_id}"
```

When no `auth_validator` is configured, `request.state.auth_context` is not set.

When authentication fails, the endpoint returns HTTP 401 with a `WWW-Authenticate` header. If a PRM (Protected Resource Metadata) URL is configured, the header references that URL; otherwise the header is `WWW-Authenticate: Bearer realm="mcp"`.

### Tool Filtering

Use `ToolFilter` to hide tools based on connection type:

```python
from fastapi_mcp_router import ToolFilter

def my_filter(is_oauth: bool) -> list[str] | None:
    if is_oauth:
        return ["admin_tool", "debug_tool"]  # Exclude from OAuth
    return None  # Include all for API key

mcp = MCPRouter(tool_filter=my_filter)
```

The callback receives `is_oauth` (True when the client authenticated with a Bearer token) and returns a list of tool names to exclude, or `None` to include all.

### OAuth Protected Resource Metadata

Expose an RFC 9728 PRM endpoint for OAuth 2.1 discovery:

```python
from fastapi_mcp_router import create_prm_router

prm = create_prm_router({
    "resource": "https://api.example.com/mcp",
    "authorization_servers": ["https://auth.example.com"],
    "scopes_supported": ["mcp:read", "mcp:evaluate"],
})
app.include_router(prm)  # Mounts at /.well-known/oauth-protected-resource
```

The `base_url` parameter on `MCPRouter` controls the PRM URL in `WWW-Authenticate` headers. Set it explicitly in production to prevent host header injection.

### Rate Limiting

Pass a FastAPI dependency for rate limiting:

```python
from fastapi import HTTPException

async def rate_limit():
    if over_limit():
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

mcp = MCPRouter(rate_limit_dependency=rate_limit)
```

The dependency runs before every GET and POST request.

## Error Handling

The library defines two exception types for different error scenarios.

### MCPError — Protocol Errors

`MCPError` represents unrecoverable JSON-RPC protocol failures. The error terminates request processing and returns a JSON-RPC error response.

```python
from fastapi_mcp_router import MCPError

raise MCPError(code=-32602, message="Missing required parameter: name")
```

Standard JSON-RPC error codes:

| Code | Meaning |
|------|---------|
| -32700 | Parse error (invalid JSON) |
| -32600 | Invalid request (malformed JSON-RPC) |
| -32601 | Method not found |
| -32602 | Invalid params |
| -32603 | Internal error |
| -32000 to -32099 | Server-defined errors |

### ToolError — Business Logic Errors

`ToolError` represents recoverable business logic failures. The tool returns a result with `isError: true`, letting the LLM see the error and potentially recover.

```python
from fastapi_mcp_router import ToolError

raise ToolError(
    message="File not found: /path/to/file.txt",
    details={"path": "/path/to/file.txt"},
)
```

Use `ToolError` for: resource not found, validation failures, external API errors, permission denied.

### Error Flow

```
Tool raises ToolError  -->  isError: true response  -->  LLM sees error, can retry
Tool raises MCPError   -->  JSON-RPC error response  -->  Request terminates
Tool raises Exception  -->  Wrapped as MCPError -32603  -->  Request terminates
```

## Sampling

Server-to-client LLM sampling lets a tool ask the connected client to perform an LLM request:

```python
from fastapi_mcp_router.session import SamplingManager

@mcp.tool()
async def ask_llm(question: str, sampler: SamplingManager) -> dict:
    """Ask the client's LLM a question."""
    response = await sampler.create_message(
        messages=[{"role": "user", "content": {"type": "text", "text": question}}],
    )
    return {"answer": response.content, "model": response.model}
```

Sampling requires `stateful=True` and `sampling_enabled=True`:

```python
mcp = MCPRouter(
    session_store=InMemorySessionStore(),
    stateful=True,
    sampling_enabled=True,
)
```

## Completions

Argument completion suggestions help clients auto-complete prompt or resource arguments. Provide a `completion_handler` callback:

```python
async def complete(ref: dict, argument: dict) -> dict:
    if argument["name"] == "language":
        prefix = argument["value"]
        matches = [l for l in ["python", "rust", "go"] if l.startswith(prefix)]
        return {"values": matches, "hasMore": False}
    return {"values": []}

router = create_mcp_router(tools, completion_handler=complete)
```

## Elicitation

Structured user input requests (stateful mode only). The server sends a schema to the client, the user fills it in, and the response comes back:

```python
# Handled internally by the router via elicitation/create JSON-RPC method.
# The client receives a message + requestedSchema and responds with:
# {"action": "accept", "content": {...}} or {"action": "decline"} or {"action": "cancel"}
```

Elicitation has a 30-second timeout. The response content is validated against the requested schema.

## Logging

In stateful mode, `MCPLoggingHandler` sends log messages to connected clients via `notifications/message` SSE events. Clients set the minimum log level via the `logging/setLevel` method.

Log levels follow the MCP specification (ascending priority): `debug`, `info`, `notice`, `warning`, `error`, `critical`, `alert`, `emergency`.

## Roots

`RootsManager` tracks server operation boundaries. Clients request roots via `roots/list` and receive a list of `Root` objects (URI + optional name). Roots define filesystem or resource locations the server may access.

## Server Info

Customize server metadata returned in the `initialize` response:

```python
from fastapi_mcp_router import MCPRouter, ServerInfo, ServerIcon

mcp = MCPRouter(
    server_info=ServerInfo(
        name="my-server",
        version="1.0.0",
        title="My MCP Server",
        description="Production MCP service",
        icons=[ServerIcon(src="https://example.com/icon.svg", mimeType="image/svg+xml")],
        websiteUrl="https://example.com",
    ),
)
```

## MCP Protocol Methods

The router handles 17 MCP methods:

| Method | Type | Description |
|--------|------|-------------|
| `initialize` | Request | Protocol handshake, returns capabilities |
| `tools/list` | Request | Returns registered tools with JSON schemas |
| `tools/call` | Request | Executes a tool with arguments |
| `resources/list` | Request | Returns resources and URI templates |
| `resources/read` | Request | Reads resource content by URI |
| `resources/subscribe` | Request | Subscribes to resource changes (stateful) |
| `resources/unsubscribe` | Request | Unsubscribes from resource changes |
| `prompts/list` | Request | Returns registered prompts |
| `prompts/get` | Request | Executes prompt with validated arguments |
| `sampling/createMessage` | Request | Server-to-client LLM sampling (stateful) |
| `roots/list` | Request | Returns registered operation boundaries |
| `logging/setLevel` | Request | Sets minimum log level for the session |
| `completion/complete` | Request | Returns argument completion suggestions |
| `elicitation/create` | Request | Requests structured user input (stateful) |
| `ping` | Request | Health check (returns `{}`) |
| `notifications/initialized` | Notification | Post-init acknowledgment (returns 202) |
| `notifications/cancelled` | Notification | Request cancellation |

## OpenTelemetry

Install the optional dependency to enable tracing and metrics:

```bash
pip install fastapi-mcp-router[otel]
```

The library emits one span per `tools/call` invocation and increments a `mcp.server.request.count` counter per JSON-RPC request. Span attributes follow OTel RPC semantic conventions:

| Attribute | Value |
|-----------|-------|
| `rpc.system.name` | `jsonrpc` |
| `rpc.method` | `tools/call` |
| `rpc.jsonrpc.version` | `2.0` |
| `mcp.tool.name` | Tool name from request params |
| `error.type` | Exception class name (on error only) |

When `opentelemetry-instrumentation-fastapi` creates an HTTP span, MCP spans nest under it automatically via OTel context propagation.

Disable telemetry explicitly:

```python
mcp = MCPRouter(enable_telemetry=False)
```

Without `opentelemetry-api` installed, all tracing is a silent no-op. The library imports and runs without error.

## Protocol Details

- **Protocol versions**: 2025-06-18 (primary), 2025-03-26 (fallback)
- **Transport**: Streamable HTTP (POST with JSON or SSE response). Legacy SSE GET via `legacy_sse=True`.
- **Header**: `MCP-Protocol-Version` required on every POST request
- **Responses**: All JSON-RPC responses use HTTP 200 (errors are at protocol level)
- **Notifications**: Return HTTP 202 Accepted with no body
- **Observability**: Optional OTel spans and counters via `fastapi-mcp-router[otel]`
