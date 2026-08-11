# API Reference

Complete reference for all 25 public exports from `fastapi_mcp_router`.

## MCPRouter

```python
from fastapi_mcp_router import MCPRouter
```

`APIRouter` subclass with built-in tool, resource, and prompt registries. Creates all MCP endpoints on construction.

### Constructor

```python
MCPRouter(
    *,
    auth_validator: AuthValidator | None = None,
    session_store: SessionStore | None = None,
    session_getter: SessionGetter | None = None,
    session_creator: SessionCreator | None = None,
    event_subscriber: EventSubscriber | None = None,
    tool_filter: ToolFilter | None = None,
    server_info: ServerInfo | None = None,
    base_url: str | None = None,
    oauth_resource_metadata: dict[str, object] | None = None,
    rate_limit_dependency: Callable[..., Awaitable[None]] | None = None,
    stateful: bool = False,
    sampling_enabled: bool = False,
    legacy_sse: bool = False,
    enable_telemetry: bool = True,
)
```

All parameters are keyword-only.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `auth_validator` | `AuthValidator \| None` | `None` | Callback `(api_key, bearer_token) -> Any`. Falsy return (e.g. `None`, `False`) → 401 with `WWW-Authenticate: Bearer` header. Truthy return → stored at `request.state.auth_context`. |
| `session_store` | `SessionStore \| None` | `None` | Session persistence backend |
| `session_getter` | `SessionGetter \| None` | `None` | Legacy session retrieval callback |
| `session_creator` | `SessionCreator \| None` | `None` | Legacy session creation callback |
| `event_subscriber` | `EventSubscriber \| None` | `None` | SSE event source callback |
| `tool_filter` | `ToolFilter \| None` | `None` | Per-connection tool filtering |
| `server_info` | `ServerInfo \| None` | `None` | Server metadata for initialize |
| `base_url` | `str \| None` | `None` | Base URL for PRM discovery |
| `oauth_resource_metadata` | `dict \| None` | `None` | RFC 9728 PRM metadata |
| `rate_limit_dependency` | `Callable \| None` | `None` | FastAPI rate limit dependency |
| `stateful` | `bool` | `False` | Enable stateful mode |
| `sampling_enabled` | `bool` | `False` | Enable server-to-client sampling |
| `legacy_sse` | `bool` | `False` | Register GET endpoint for legacy SSE transport |
| `enable_telemetry` | `bool` | `True` | Emit OTel spans and counters when `opentelemetry-api` is installed |

**Raises:**
- `ValueError` if `stateful=True` and `session_store` is `None`
- `ValueError` if `sampling_enabled=True` and `stateful` is `False`

### Methods

#### tool()

```python
MCPRouter.tool(
    name: str | None = None,
    description: str | None = None,
    input_schema: dict | None = None,
    annotations: dict | None = None,
) -> Callable
```

Decorator. Registers an async function as an MCP tool. Delegates to internal `MCPToolRegistry`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `name` | Function name | Tool identifier |
| `description` | Docstring | Tool description |
| `input_schema` | Auto-generated | JSON Schema for parameters |
| `annotations` | `None` | MCP tool annotations (e.g., `{"readOnlyHint": True}`) |

**Raises:** `TypeError` if the function is not async.

#### resource()

```python
MCPRouter.resource(
    uri: str,
    name: str | None = None,
    description: str | None = None,
    mime_type: str | None = None,
    icons: list[dict] | None = None,
) -> Callable
```

Decorator. Registers an async function as an MCP resource handler. URI supports `{param}` templates (RFC 6570).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `uri` | (required) | URI or URI template string |
| `name` | Function name | Resource display name |
| `description` | Docstring | Resource description |
| `mime_type` | `None` | MIME type for the content |
| `icons` | `None` | List of icon descriptor dicts, forwarded to `ResourceRegistry.resource()` (see [ResourceRegistry](#resourceregistry) below) |

**Raises:** `TypeError` if the function is not async.

#### prompt()

```python
MCPRouter.prompt(
    name: str | None = None,
    description: str | None = None,
    arguments: list[dict] | None = None,
    icons: list[dict] | None = None,
) -> Callable
```

Decorator. Registers a sync or async function as an MCP prompt. Arguments are auto-generated from the function signature.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `name` | Function name | Prompt identifier |
| `description` | Docstring | Prompt description |
| `arguments` | `None` | Reserved for future use |
| `icons` | `None` | List of icon descriptor dicts, forwarded to `PromptRegistry.prompt()` (see [PromptRegistry](#promptregistry) below) |

#### add_resource_provider()

```python
MCPRouter.add_resource_provider(uri_prefix: str, provider: ResourceProvider) -> None
```

Registers a `ResourceProvider` for all URIs starting with `uri_prefix`.

#### shutdown()

```python
async MCPRouter.shutdown() -> None
```

Sets the router's internal shutdown event, signaling active SSE generator loops to exit gracefully (each yields a `: server-shutdown` comment before closing). Call from a FastAPI lifespan handler:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await mcp.shutdown()

app = FastAPI(lifespan=lifespan)
```

#### Notification triggers

All four methods are no-ops when no `session_store` was configured (stateless mode never enqueues notifications). They enqueue a JSON-RPC notification onto every live session's message queue (or, for `notify_resource_updated`, onto the queues of sessions subscribed to that specific URI); delivery happens the next time each session drains its queue (e.g. on the SSE stream or a subsequent `dequeue_messages` call).

```python
async MCPRouter.notify_tools_list_changed() -> None
async MCPRouter.notify_resources_list_changed() -> None
async MCPRouter.notify_prompts_list_changed() -> None
async MCPRouter.notify_resource_updated(uri: str) -> None
```

| Method | Notification method emitted | Trigger example |
|--------|------------------------------|------------------|
| `notify_tools_list_changed()` | `notifications/tools/list_changed` | After registering/removing a tool at runtime |
| `notify_resources_list_changed()` | `notifications/resources/list_changed` | After adding/removing a resource |
| `notify_prompts_list_changed()` | `notifications/prompts/list_changed` | After adding/removing a prompt |
| `notify_resource_updated(uri)` | `notifications/resources/updated` (params `{"uri": uri}`) | After the content behind a subscribed resource URI changes |

**Example:**

```python
mcp = MCPRouter(session_store=InMemorySessionStore(), stateful=True)

@mcp.tool()
async def refresh_catalog() -> dict:
    """Reload the tool catalog and tell connected clients."""
    # ... mutate registered tools/resources here ...
    await mcp.notify_tools_list_changed()
    return {"status": "refreshed"}

@mcp.resource("file:///data/report.txt")
async def report() -> str:
    return "latest report contents"

async def on_report_change() -> None:
    await mcp.notify_resource_updated("file:///data/report.txt")
```

**Raises:** `MCPError(-32603)` only if the underlying `SessionStore` lookup itself fails (e.g. an unreachable Redis backend). Failure to enqueue onto an individual session's queue is logged and does not stop delivery to the remaining sessions.

---

## create_mcp_router()

```python
from fastapi_mcp_router import create_mcp_router
```

Factory function returning a configured `APIRouter` with MCP endpoints.

```python
create_mcp_router(
    registry: MCPToolRegistry,
    rate_limit_dependency: Callable | None = None,
    auth_validator: AuthValidator | None = None,
    base_url: str | None = None,
    session_getter: SessionGetter | None = None,
    session_creator: SessionCreator | None = None,
    tool_filter: ToolFilter | None = None,
    server_info: ServerInfo | None = None,
    event_subscriber: EventSubscriber | None = None,
    oauth_resource_metadata: dict | None = None,
    session_store: SessionStore | None = None,
    stateful: bool = False,
    resource_registry: ResourceRegistry | None = None,
    prompt_registry: PromptRegistry | None = None,
    sampling_enabled: bool = False,
    completion_handler: Callable | None = None,
    legacy_sse: bool = False,
    enable_telemetry: bool = True,
    shutdown_event: asyncio.Event | None = None,
) -> APIRouter
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `registry` | `MCPToolRegistry` | (required) | Tool registry |
| `rate_limit_dependency` | `Callable \| None` | `None` | FastAPI rate limit dependency |
| `auth_validator` | `AuthValidator \| None` | `None` | Callback `(api_key, bearer_token) -> Any`. Falsy return (e.g. `None`, `False`) → 401 with `WWW-Authenticate: Bearer` header. Truthy return → stored at `request.state.auth_context`. |
| `base_url` | `str \| None` | `None` | Base URL for PRM headers |
| `session_getter` | `SessionGetter \| None` | `None` | Legacy session retrieval |
| `session_creator` | `SessionCreator \| None` | `None` | Legacy session creation |
| `tool_filter` | `ToolFilter \| None` | `None` | `(is_oauth) -> list[str] \| None` |
| `server_info` | `ServerInfo \| None` | `None` | Server metadata |
| `event_subscriber` | `EventSubscriber \| None` | `None` | SSE event source |
| `oauth_resource_metadata` | `dict \| None` | `None` | RFC 9728 PRM fields |
| `session_store` | `SessionStore \| None` | `None` | Session persistence |
| `stateful` | `bool` | `False` | Enable stateful mode |
| `resource_registry` | `ResourceRegistry \| None` | `None` | Resource registry |
| `prompt_registry` | `PromptRegistry \| None` | `None` | Prompt registry |
| `sampling_enabled` | `bool` | `False` | Enable sampling |
| `completion_handler` | `Callable \| None` | `None` | `(ref, argument) -> dict` |
| `legacy_sse` | `bool` | `False` | Register GET endpoint for legacy SSE transport |
| `enable_telemetry` | `bool` | `True` | Emit OTel spans and counters when `opentelemetry-api` is installed |
| `shutdown_event` | `asyncio.Event \| None` | `None` | Signaled to gracefully close active SSE streams (set via `MCPRouter.shutdown()`) |

**Returns:** `APIRouter` — mount with `app.include_router(router, prefix="/mcp")`.

**Raises:**
- `ValueError` if both `session_store` and `session_getter` are provided
- `ValueError` if `stateful=True` and `session_store` is `None`
- `ValueError` if `sampling_enabled=True` and `stateful` is `False`
- `ValueError` if `oauth_resource_metadata` is missing `resource` or `authorization_servers`

### Pagination

`tools/list`, `resources/list`, `resources/templates/list`, and `prompts/list` all accept an optional `cursor` param and return an opaque `nextCursor` when more items remain. Cursors are base64-encoded offsets (`fastapi_mcp_router.protocol.encode_cursor`/`paginate`) and are not stable across list mutations. Page size defaults to 100 items.

Note: on `resources/list`, only the top-level `resources` array is paginated via `cursor`/`nextCursor`; the `resourceTemplates` array in the same response is always returned in full. Call `resources/templates/list` directly if templates also need to be paginated.

**Request (first page):**

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {}
}
```

**Response (more items remain):**

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "tools": [ /* up to 100 tool definitions */ ],
    "nextCursor": "AAAAAAAAAGQ="
  }
}
```

**Request (next page, using the returned cursor):**

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list",
  "params": {"cursor": "AAAAAAAAAGQ="}
}
```

When this is the final page, `nextCursor` is omitted from the result. A malformed or undecodable `cursor` raises `MCPError(-32602, "Invalid params: malformed pagination cursor")`.

### Protocol version negotiation

Negotiation is driven **exclusively** by the `MCP-Protocol-Version` HTTP header (`_validate_protocol_version`). The router accepts a header declaring any value — it does not reject unknown or future versions outright. Instead it **clamps** the requested version to the highest version in its supported whitelist that is `<=` the requested version, using ISO-date lexicographic comparison (`_negotiate_protocol_version`, whitelist in `_SUPPORTED_PROTOCOL_VERSIONS`).

The `protocolVersion` field inside the `initialize` request params is accepted but never read — `handle_initialize` does not validate, clamp, or reflect it in the response. The response's `protocolVersion` is always the header-negotiated value from `_validate_protocol_version`, which may differ from whatever the client placed in `params.protocolVersion`.

Supported whitelist (highest to lowest): `2025-11-25`, `2025-06-18`, `2025-03-26`.

| `MCP-Protocol-Version` header | Negotiated version | Why |
|---|---|---|
| *(absent/missing)* | `2025-03-26` | Missing header defaults to `2025-03-26` before negotiation (`_validate_protocol_version`) |
| `2025-11-25` | `2025-11-25` | Exact match — highest supported version `<=` requested |
| `2026-07-28` (future/unknown) | `2025-11-25` | No exact match; clamps down to the highest supported version `<=` requested |
| `2025-06-18` | `2025-06-18` | Exact match |
| `2025-08-01` (unknown, between supported versions) | `2025-06-18` | Highest supported version `<=` requested |
| `2025-03-26` | `2025-03-26` | Exact match — lowest supported version |
| `2024-01-01` (below all supported versions) | *(rejected)* | No eligible version `<=` requested; raises `MCPError(-32602, "Unsupported protocol version: ...")` with `data.supported` listing the whitelist |

The negotiated version is never the raw client-requested string unless it happens to exactly match a supported version — the router does not echo unrecognized versions back. Content blocks gated to `2025-11-25` (`image`, `audio`, `resource_link`) are included only when the *negotiated* version is `2025-11-25` or later; on older negotiated versions those blocks are silently omitted rather than raising.

---

## create_prm_router()

```python
from fastapi_mcp_router import create_prm_router
```

Creates a root-level router for the OAuth Protected Resource Metadata endpoint (RFC 9728).

```python
def create_prm_router(
    mcp: MCPRouter | None = None,
    oauth_resource_metadata: dict[str, object] | None = None,
) -> APIRouter
```

Registers `GET /.well-known/oauth-protected-resource`. Mount with no prefix.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mcp` | `MCPRouter \| None` | `None` | Derives PRM from an `MCPRouter` instance. Sets `resource` to `mcp.base_url + "/mcp"` and `authorization_servers` from `mcp.oauth_resource_metadata`. |
| `oauth_resource_metadata` | `dict[str, object] \| None` | `None` | Explicit RFC 9728 PRM fields. Must include `resource` and `authorization_servers`. |

Exactly one of `mcp` or `oauth_resource_metadata` must be provided.

**Example — derive from MCPRouter:**

```python
mcp = MCPRouter(
    base_url="https://api.example.com",
    oauth_resource_metadata={
        "resource": "https://api.example.com/mcp",
        "authorization_servers": ["https://auth.example.com"],
    },
)
app.include_router(mcp)
app.include_router(create_prm_router(mcp=mcp))
```

**Example — explicit metadata (backward-compatible):**

```python
app.include_router(create_prm_router(oauth_resource_metadata={
    "resource": "https://api.example.com/mcp",
    "authorization_servers": ["https://auth.example.com"],
}))
```

**Raises:**
- `TypeError("mcp and oauth_resource_metadata are mutually exclusive")` if both are provided
- `TypeError("one of mcp or oauth_resource_metadata is required")` if neither is provided
- `ValueError` if `resource` or `authorization_servers` keys are missing from the resolved metadata

---

## MCPToolRegistry

```python
from fastapi_mcp_router import MCPToolRegistry
```

Standalone tool registry with decorator-based registration and auto-schema generation.

### Constructor

```python
MCPToolRegistry()
```

Creates an empty registry. No arguments.

### Methods

#### tool()

```python
MCPToolRegistry.tool(
    name: str | None = None,
    description: str | None = None,
    input_schema: dict | None = None,
    annotations: dict | None = None,
    output_schema: dict | None = None,
    title: str | None = None,
    icons: list[dict] | None = None,
) -> Callable
```

Decorator. Registers an async function as an MCP tool.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `name` | Function name | Tool identifier |
| `description` | Docstring | Tool description |
| `input_schema` | Auto-generated | JSON Schema for parameters |
| `annotations` | `None` | MCP annotations (dict, or a `ToolAnnotations` model dump) |
| `output_schema` | `None` | JSON Schema for structured results |
| `title` | `None` | Human-readable display label for the tool. Distinct from `name`; never defaults to it. Omitted from `tools/list` when not provided |
| `icons` | `None` | List of icon descriptor dicts (see `Icon`). Each entry is validated via `Icon.model_validate()` at registration time, then stored and re-emitted verbatim. Only emitted in `tools/list` for clients negotiating a protocol version of `2025-11-25` or later |

**Raises:** `TypeError` if the function is not async.

**Example — title and icons:**

```python
@registry.tool(
    title="Weather Lookup",
    icons=[{"src": "https://example.com/icon.png", "mimeType": "image/png", "sizes": ["48x48"]}],
)
async def get_weather(location: str) -> dict:
    """Get current weather for a location."""
    return {"location": location, "condition": "sunny"}
```

Note: `MCPRouter.tool()` (the decorator on the router-level facade, see [MCPRouter](#mcprouter) above) does not accept `title` or `icons`; use `MCPToolRegistry` directly (or `create_mcp_router(registry=...)`) when those fields are needed.

**Filtered parameters** (excluded from auto-generated schema):
- `self`, `cls`, `*args`, `**kwargs`
- `FastAPI.Depends()` defaults
- `Request`, `BackgroundTasks` (FastAPI types)
- `ProgressCallback`, `SamplingManager` (MCP types)

#### list_tools()

```python
MCPToolRegistry.list_tools() -> list[dict]
```

Returns tool definitions in MCP format. Each dict contains:

```python
{
    "name": str,
    "description": str,
    "inputSchema": dict,        # Always present
    "annotations": dict | None, # Present when set
    "outputSchema": dict | None # Present when set
}
```

#### shape_tool()

```python
MCPToolRegistry.shape_tool(tool: ToolDefinition, protocol_version: str | None = None) -> dict
```

Formats a single raw tool definition for the MCP wire protocol. Extracted from `list_tools()` so callers that paginate (e.g. `tools/list` in `router.py`) can shape only the page of tools being returned instead of the entire registry.

#### get_raw_tools()

```python
MCPToolRegistry.get_raw_tools() -> list[ToolDefinition]
```

Returns raw, unshaped tool definitions in registration order. Enables callers to paginate the raw registry before shaping items to the wire format.

#### call_tool()

```python
async MCPToolRegistry.call_tool(
    name: str,
    arguments: dict,
    request: object | None = None,
    background_tasks: object | None = None,
    stateful: bool = False,
    progress_callback: object | None = None,
    sampling_manager: object | None = None,
) -> object
```

Executes a registered tool with dependency injection.

| Parameter | Description |
|-----------|-------------|
| `name` | Tool name to execute |
| `arguments` | Arguments matching the input schema |
| `request` | FastAPI Request for injection |
| `background_tasks` | FastAPI BackgroundTasks for injection |
| `stateful` | Return raw AsyncGenerator for streaming tools |
| `progress_callback` | ProgressCallback for progress injection |
| `sampling_manager` | SamplingManager for sampling injection |

**Returns:** Tool result (any JSON-serializable value). Generator tools return `list[dict]` (or `AsyncGenerator` when `stateful=True`).

**Raises:**
- `MCPError(-32601)` — tool not found
- `MCPError(-32602)` — invalid arguments
- `MCPError(-32603)` — execution failure
- `ToolError` — re-raised from tool (business logic)

---

## ResourceRegistry

```python
from fastapi_mcp_router import ResourceRegistry
```

Registry for MCP resources with decorator and provider registration.

### Constructor

```python
ResourceRegistry()
```

### Methods

#### resource()

```python
ResourceRegistry.resource(
    uri_template: str,
    name: str | None = None,
    description: str | None = None,
    mime_type: str | None = None,
    icons: list[dict] | None = None,
) -> Callable
```

Decorator. Registers an async function as a resource handler. The `uri_template` supports `{param}` placeholders per RFC 6570.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `uri_template` | (required) | URI template string with optional `{param}` placeholders |
| `name` | Function name | Resource name |
| `description` | Docstring | Resource description |
| `mime_type` | `None` | Optional MIME type override |
| `icons` | `None` | List of icon descriptor dicts. Each entry is validated via `Icon.model_validate()` at registration time. Only emitted in `resources/list` and `resources/templates/list` for clients negotiating a protocol version of `2025-11-25` or later |

#### register_provider()

```python
ResourceRegistry.register_provider(uri_prefix: str, provider: ResourceProvider) -> None
```

Registers a `ResourceProvider` for all URIs matching `uri_prefix`.

#### list_resources()

```python
ResourceRegistry.list_resources() -> list[dict]
```

Returns all resources and templates in MCP format.

#### read_resource()

```python
async ResourceRegistry.read_resource(uri: str) -> ResourceContents
```

Dispatches to the matching handler or provider. Returns `ResourceContents`.

#### subscribe() / unsubscribe()

```python
ResourceRegistry.subscribe(uri: str) -> bool
ResourceRegistry.unsubscribe(uri: str) -> bool
```

Subscribe/unsubscribe from resource changes. Returns `True` if supported.

#### watch()

```python
async ResourceRegistry.watch() -> AsyncIterator
```

Aggregates change notifications from all providers.

---

## PromptRegistry

```python
from fastapi_mcp_router import PromptRegistry
```

Registry for MCP prompts with auto-generated argument metadata.

### Constructor

```python
PromptRegistry()
```

### Methods

#### prompt()

```python
PromptRegistry.prompt(
    name: str | None = None,
    description: str | None = None,
    icons: list[dict] | None = None,
) -> Callable
```

Decorator. Registers a sync or async function as a prompt. Arguments are auto-generated from the function signature.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `name` | Function name | Prompt identifier |
| `description` | Docstring | Prompt description |
| `icons` | `None` | List of icon descriptor dicts. Each entry is validated via `Icon.model_validate()` at registration time. Only emitted in `prompts/list` for clients negotiating a protocol version of `2025-11-25` or later |

#### list_prompts()

```python
PromptRegistry.list_prompts(protocol_version: str | None = None) -> list[dict]
```

Returns prompt definitions with name, description, arguments, and icons (when provided and `protocol_version` supports them).

#### shape_prompt()

```python
PromptRegistry.shape_prompt(defn: PromptDefinition, protocol_version: str | None = None) -> dict
```

Formats a single raw prompt definition for the MCP wire protocol. Extracted from `list_prompts()` so callers that paginate (e.g. `prompts/list` in `router.py`) can shape only the page of prompts being returned instead of the entire registry.

#### get_raw_prompts()

```python
PromptRegistry.get_raw_prompts() -> list[PromptDefinition]
```

Returns raw, unshaped prompt definitions in registration order. Enables callers to paginate the raw registry before shaping items to the wire format.

#### get_prompt()

```python
async PromptRegistry.get_prompt(
    name: str,
    arguments: dict | None = None,
) -> list[dict]
```

Validates required arguments, calls handler, returns message list.

**Raises:**
- `MCPError(-32602)` — prompt not found or missing required argument
- `MCPError(-32603)` — handler failure

#### has_prompts()

```python
PromptRegistry.has_prompts() -> bool
```

Returns `True` if any prompts are registered.

---

## SessionStore

```python
from fastapi_mcp_router import SessionStore
```

Abstract base class for session persistence. All methods are async.

### Abstract Methods

```python
async SessionStore.create(protocol_version: str, client_info: dict, capabilities: dict) -> Session
async SessionStore.get(session_id: str) -> Session | None
async SessionStore.update(session: Session) -> None
async SessionStore.delete(session_id: str) -> None
async SessionStore.enqueue_message(session_id: str, message: dict) -> None
async SessionStore.dequeue_messages(session_id: str) -> list[dict]
async SessionStore.list_sessions() -> list[str]
async SessionStore.find_subscribers(uri: str) -> list[str]
```

| Method | Description |
|--------|-------------|
| `create` | Creates session with UUID4 id and UTC timestamps |
| `get` | Returns session or `None` if expired/absent; updates `last_activity` |
| `update` | Persists changes to an existing session |
| `delete` | Removes a session |
| `enqueue_message` | Appends to queue (max 1000; silently drops if full) |
| `dequeue_messages` | Returns all messages and clears queue atomically |
| `list_sessions` | Returns the ids of all live (non-expired) sessions. Backs `MCPRouter.notify_tools_list_changed()` / `notify_resources_list_changed()` / `notify_prompts_list_changed()`, which enqueue a notification onto every id this returns |
| `find_subscribers` | Returns the ids of live sessions whose subscriptions set contains `uri`. Backs `MCPRouter.notify_resource_updated(uri)` |

---

## InMemorySessionStore

```python
from fastapi_mcp_router import InMemorySessionStore
```

TTL-based in-memory `SessionStore` implementation.

### Constructor

```python
InMemorySessionStore(ttl_seconds: int = 3600)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ttl_seconds` | `3600` | Seconds of inactivity before session expires |

Uses `asyncio.Lock` for concurrent safety. All state is lost on process restart.

### list_sessions() / find_subscribers()

```python
async InMemorySessionStore.list_sessions() -> list[str]
async InMemorySessionStore.find_subscribers(uri: str) -> list[str]
```

`list_sessions()` returns the ids of sessions currently present and unexpired, judged against the current time without mutating or removing expired entries (expired sessions are cleaned up on their next `get()` call). `find_subscribers(uri)` returns the same live-session filter intersected with `uri in session.subscriptions`.

**Example:**

```python
store = InMemorySessionStore()
session = await store.create("2025-11-25", {}, {})
live_ids = await store.list_sessions()
assert session.session_id in live_ids

await store.update(session)  # session.subscriptions mutated elsewhere via resources/subscribe
subscribers = await store.find_subscribers("file:///data/report.txt")
```

---

## MCPError

```python
from fastapi_mcp_router import MCPError
```

Protocol-level JSON-RPC error. Terminates request processing.

### Constructor

```python
MCPError(
    code: int,
    message: str,
    data: dict | None = None,
)
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `code` | `int` | JSON-RPC error code |
| `message` | `str` | Human-readable description |
| `data` | `dict \| None` | Additional error context |

**Standard codes:**

| Code | Meaning |
|------|---------|
| `-32700` | Parse error |
| `-32600` | Invalid request |
| `-32601` | Method not found |
| `-32602` | Invalid params |
| `-32603` | Internal error |
| `-32000` to `-32099` | Server-defined |

---

## ToolError

```python
from fastapi_mcp_router import ToolError
```

Recoverable business logic error. Returns `isError: true` response so the LLM can see and recover.

### Constructor

```python
ToolError(
    message: str,
    details: dict | None = None,
)
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `message` | `str` | Error description for LLM |
| `details` | `dict \| None` | Structured error context |

---

## TextContent

```python
from fastapi_mcp_router import TextContent
```

Pydantic model for text content in MCP responses.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `str` | `"text"` | Content type identifier |
| `text` | `str` | (required) | Text content |

---

## ImageContent

```python
from fastapi_mcp_router import ImageContent
```

Pydantic model for image content in MCP tool responses. Flat base64-encoded wire shape (never a `data:` URI).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `str` | `"image"` | Content type identifier |
| `data` | `str` | (required) | Base64-encoded image bytes |
| `mimeType` | `str` | (required) | MIME type, e.g. `"image/png"` |

Only emitted on the wire when the connection's negotiated protocol version is `2025-11-25` or later; on older negotiated versions the block is silently omitted (see [Protocol version negotiation](#protocol-version-negotiation)).

**Example — returning an image from a tool:**

```python
from fastapi_mcp_router import ImageContent

@registry.tool()
async def render_chart(data: list[float]) -> ImageContent:
    png_bytes = render_png(data)
    import base64
    return ImageContent(data=base64.b64encode(png_bytes).decode(), mimeType="image/png")
```

---

## AudioContent

```python
from fastapi_mcp_router import AudioContent
```

Pydantic model for audio content in MCP tool responses. Same flat base64-encoded wire shape as `ImageContent`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `str` | `"audio"` | Content type identifier |
| `data` | `str` | (required) | Base64-encoded audio bytes |
| `mimeType` | `str` | (required) | MIME type, e.g. `"audio/mpeg"` |

Gated the same way as `ImageContent`: omitted on the wire for connections negotiated below `2025-11-25`.

---

## ResourceLinkContent

```python
from fastapi_mcp_router import ResourceLinkContent
```

Pydantic model that points a tool result at a resource without embedding its full content.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `str` | `"resource_link"` | Content type identifier |
| `uri` | `str` | (required) | URI identifying the linked resource |
| `name` | `str` | (required) | Resource identifier/name |
| `title` | `str \| None` | `None` | Human-readable display title |
| `description` | `str \| None` | `None` | Resource description |
| `mimeType` | `str \| None` | `None` | MIME type of the linked resource |
| `icons` | `list[Icon] \| None` | `None` | Icons for the resource |
| `size` | `int \| None` | `None` | Resource size in bytes |

Gated the same way as `ImageContent`: omitted on the wire for connections negotiated below `2025-11-25`.

A tool may return a single content-block instance (`TextContent`, `ImageContent`, `AudioContent`, or `ResourceLinkContent`) or a list mixing them to emit multiple blocks in one response — the router recognizes these types directly and skips JSON/text wrapping.

---

## ToolResponse

```python
from fastapi_mcp_router import ToolResponse
```

Pydantic model for MCP tool responses.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `content` | `list[TextContent]` | (required) | Content items |
| `isError` | `bool` | `False` | Error flag |

---

## ServerInfo

```python
from fastapi_mcp_router import ServerInfo
```

`TypedDict` (total=False) for server metadata in the `initialize` response.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Server identifier (kebab-case) |
| `version` | `str` | Semantic version |
| `title` | `str` | Display name |
| `description` | `str` | Server description |
| `icons` | `list[ServerIcon]` | Branding icons |
| `websiteUrl` | `str` | Server website |

---

## ServerIcon

```python
from fastapi_mcp_router import ServerIcon
```

`TypedDict` (total=False) for server icon metadata.

| Field | Type | Description |
|-------|------|-------------|
| `src` | `str` | Icon URL |
| `mimeType` | `str` | MIME type (e.g., `image/svg+xml`) |
| `sizes` | `list[str]` | Available sizes (e.g., `["32x32"]`) |

---

## Icon

```python
from fastapi_mcp_router import Icon
```

Pydantic model for per-item icons attached to tools, resources, resource templates, and prompts. Distinct from `ServerIcon` (a `TypedDict` for server-wide branding, above) — `Icon` is used wherever the protocol requires a list of icons on an individual item (e.g. `Tool.icons`, `ResourceLinkContent.icons`). Tool icons passed to `MCPToolRegistry.tool(icons=...)` are validated via `Icon.model_validate()` at registration time (see `MCPToolRegistry.tool()` above).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `src` | `str` | (required) | Icon source; must use the `https://` or `data:` scheme |
| `mimeType` | `str \| None` | `None` | Advisory MIME type, e.g. `"image/svg+xml"`, restricted to a strict image-type allowlist |
| `sizes` | `list[str] \| None` | `None` | Available sizes, e.g. `["32x32", "16x16"]` |
| `theme` | `str \| None` | `None` | Color theme the icon is designed for, e.g. `"light"` or `"dark"` |

**Raises (on construction):**
- `ValueError` if `src` uses a scheme other than `https://` or `data:`
- `ValueError` if the icon is SVG and contains `<script>`, an event-handler attribute (e.g. `onload=`), a `javascript:` URI, or other executable content — including such content hidden inside a base64-encoded `data:` payload
- `ValueError` if `mimeType` is set to a value outside the allowed image-type list

Executable-content filtering is a best-effort blocklist, not an exhaustive sanitizer: it screens out common script/event-handler patterns at registration time but is not a substitute for render-time defenses. Consumers rendering icon `src` values (especially inline SVG or `data:` payloads) should still enforce a restrictive CSP (`script-src 'none'`) and/or sanitize with a library such as DOMPurify before rendering.

**Example:**

```python
from fastapi_mcp_router import Icon

icon = Icon(src="https://example.com/icon.png", mimeType="image/png", sizes=["48x48"])
```

Icons on tools are only emitted in `tools/list` when the connection's negotiated protocol version supports them (`2025-11-25` or later); see `MCPToolRegistry.tool()` above.

---

## ToolAnnotations

```python
from fastapi_mcp_router import ToolAnnotations
```

Pydantic model for behavioral hints describing an MCP tool for client UIs. Annotations are untrusted hints supplied by tool authors — clients may use them to inform UI presentation but must not treat them as security guarantees. Unknown/vendor keys are preserved and pass through without validation (`model_config = ConfigDict(extra="allow")`).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `title` | `str \| None` | `None` | Human-readable display title for the tool |
| `readOnlyHint` | `bool \| None` | `None` | Hint that the tool does not modify state |
| `destructiveHint` | `bool \| None` | `None` | Hint that the tool may perform destructive updates |
| `idempotentHint` | `bool \| None` | `None` | Hint that repeated calls have no additional effect |
| `openWorldHint` | `bool \| None` | `None` | Hint that the tool interacts with an open-world (external) environment |

**Example:**

```python
from fastapi_mcp_router import ToolAnnotations

annotations = ToolAnnotations(readOnlyHint=True, title="Search Catalog")

@registry.tool(annotations=annotations.model_dump(exclude_none=True))
async def search_catalog(query: str) -> dict:
    """Search the product catalog without side effects."""
    return {"results": []}
```

The `annotations` parameter on `tool()` accepts either a plain dict or a `ToolAnnotations` model dump — the registry stores whatever dict-shaped value is passed through unchanged.

---

## EventSubscriber

```python
from fastapi_mcp_router import EventSubscriber
```

Type alias for SSE event source callbacks:

```python
EventSubscriber = Callable[
    [str, int | None],           # (session_id, last_event_id)
    AsyncGenerator[tuple[int, dict]],  # yields (event_id, json_rpc_notification)
]
```

---

## ProgressCallback

```python
from fastapi_mcp_router import ProgressCallback
```

Type alias for progress reporting:

```python
ProgressCallback = Callable[[int, int, str | None], Awaitable[None]]
#                           current, total, message
```

Injected into tools when `progress: ProgressCallback` appears in the signature. In stateless mode, a no-op is injected.

---

## ToolFilter

```python
from fastapi_mcp_router import ToolFilter
```

Type alias for per-connection tool filtering:

```python
ToolFilter = Callable[[bool], list[str] | None]
#                     is_oauth -> excluded_tool_names or None
```

---

## Internal Types (not in \_\_all\_\_)

These types are used by the library but not exported. Import from submodules when needed.

### Session

```python
from fastapi_mcp_router.session import Session
```

Dataclass tracking a single MCP session.

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | `str` | UUID4 identifier |
| `created_at` | `datetime` | UTC creation time |
| `last_activity` | `datetime` | Last access time |
| `protocol_version` | `str` | Negotiated protocol version |
| `client_info` | `dict` | Client capabilities |
| `capabilities` | `dict` | Server capabilities |
| `message_queue` | `list[dict]` | Queued messages (max 1000) |
| `subscriptions` | `set[str]` | Resource URIs (max 100) |

### RedisSessionStore

```python
from fastapi_mcp_router.session import RedisSessionStore
```

Redis-backed session store for multi-instance deployments.

```python
RedisSessionStore(redis_client, ttl_seconds: int = 7200)
```

Requires `redis.asyncio` (install `redis` package). Keys: `mcp:session:{id}`, `mcp:queue:{id}`.

### list_sessions() / find_subscribers()

```python
async RedisSessionStore.list_sessions() -> list[str]
async RedisSessionStore.find_subscribers(uri: str) -> list[str]
```

`list_sessions()` enumerates the `mcp:session:*` namespace via non-blocking `SCAN` (not `KEYS`) and returns every matched id; Redis TTL-based expiration means only unexpired keys exist, so no live/expired filtering is needed. `find_subscribers(uri)` scans the same namespace, fetches and deserializes each session, and returns the ids whose `subscriptions` set contains `uri` — an O(n) operation over the live session count. Both raise `MCPError(-32603)` on Redis errors.

### SamplingManager

```python
from fastapi_mcp_router.session import SamplingManager
```

Manages server-to-client LLM sampling requests in stateful mode. Injected into tools with a `SamplingManager` parameter.

### MCPLoggingHandler

```python
from fastapi_mcp_router.session import MCPLoggingHandler
```

Sends log messages to connected clients via SSE.

### ResourceProvider

```python
from fastapi_mcp_router.resources import ResourceProvider
```

ABC for resource providers. Implement `list_resources`, `read_resource`, `subscribe`, `unsubscribe`, `watch`.

### FileResourceProvider

```python
from fastapi_mcp_router.resources import FileResourceProvider
```

Sandboxed local filesystem provider.

```python
FileResourceProvider(
    root_path: str | Path,
    allowed_extensions: set[str] = {".txt", ".md", ".json", ".yaml"},
)
```

Rejects path traversal, enforces 10 MB limit, filters by extension.

### Resource / ResourceTemplate / ResourceContents

```python
from fastapi_mcp_router.resources import Resource, ResourceTemplate, ResourceContents
```

Dataclasses for resource metadata and content. `ResourceContents` uses mutually exclusive `text` and `blob` fields.

### LogLevel

```python
from fastapi_mcp_router.types import LogLevel
```

IntEnum: `debug=0`, `info=1`, `notice=2`, `warning=3`, `error=4`, `critical=5`, `alert=6`, `emergency=7`.

### SamplingRequest / SamplingResponse

```python
from fastapi_mcp_router.types import SamplingRequest, SamplingResponse
```

Dataclasses for server-to-client LLM sampling.

### CompletionRef / CompletionArgument / CompletionResult

```python
from fastapi_mcp_router.types import CompletionRef, CompletionArgument, CompletionResult
```

Pydantic models for argument completion.

### ElicitationRequest / ElicitationResponse

```python
from fastapi_mcp_router.types import ElicitationRequest, ElicitationResponse
```

Pydantic models for structured user input.

### McpSessionData

```python
from fastapi_mcp_router.types import McpSessionData
```

Dataclass for session metadata (session_id, oauth_client_id, connection_id, last_event_id, created_at).
