# API Reference

Complete reference for all 17 public exports from `fastapi_mcp_router`.

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
) -> Callable
```

Decorator. Registers an async function as an MCP resource handler. URI supports `{param}` templates (RFC 6570).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `uri` | (required) | URI or URI template string |
| `name` | Function name | Resource display name |
| `description` | Docstring | Resource description |
| `mime_type` | `None` | MIME type for the content |

**Raises:** `TypeError` if the function is not async.

#### prompt()

```python
MCPRouter.prompt(
    name: str | None = None,
    description: str | None = None,
    arguments: list[dict] | None = None,
) -> Callable
```

Decorator. Registers a sync or async function as an MCP prompt. Arguments are auto-generated from the function signature.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `name` | Function name | Prompt identifier |
| `description` | Docstring | Prompt description |
| `arguments` | `None` | Reserved for future use |

#### add_resource_provider()

```python
MCPRouter.add_resource_provider(uri_prefix: str, provider: ResourceProvider) -> None
```

Registers a `ResourceProvider` for all URIs starting with `uri_prefix`.

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
    roots_manager: RootsManager | None = None,
    completion_handler: Callable | None = None,
    legacy_sse: bool = False,
    enable_telemetry: bool = True,
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
| `roots_manager` | `RootsManager \| None` | `None` | Operation boundaries |
| `completion_handler` | `Callable \| None` | `None` | `(ref, argument) -> dict` |
| `legacy_sse` | `bool` | `False` | Register GET endpoint for legacy SSE transport |
| `enable_telemetry` | `bool` | `True` | Emit OTel spans and counters when `opentelemetry-api` is installed |

**Returns:** `APIRouter` — mount with `app.include_router(router, prefix="/mcp")`.

**Raises:**
- `ValueError` if both `session_store` and `session_getter` are provided
- `ValueError` if `stateful=True` and `session_store` is `None`
- `ValueError` if `sampling_enabled=True` and `stateful` is `False`
- `ValueError` if `oauth_resource_metadata` is missing `resource` or `authorization_servers`

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
) -> Callable
```

Decorator. Registers an async function as an MCP tool.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `name` | Function name | Tool identifier |
| `description` | Docstring | Tool description |
| `input_schema` | Auto-generated | JSON Schema for parameters |
| `annotations` | `None` | MCP annotations |
| `output_schema` | `None` | JSON Schema for structured results |

**Raises:** `TypeError` if the function is not async.

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
) -> Callable
```

Decorator. Registers an async function as a resource handler. The `uri_template` supports `{param}` placeholders per RFC 6570.

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
) -> Callable
```

Decorator. Registers a sync or async function as a prompt. Arguments are auto-generated from the function signature.

#### list_prompts()

```python
PromptRegistry.list_prompts() -> list[dict]
```

Returns prompt definitions with name, description, and arguments.

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
```

| Method | Description |
|--------|-------------|
| `create` | Creates session with UUID4 id and UTC timestamps |
| `get` | Returns session or `None` if expired/absent; updates `last_activity` |
| `update` | Persists changes to an existing session |
| `delete` | Removes a session |
| `enqueue_message` | Appends to queue (max 1000; silently drops if full) |
| `dequeue_messages` | Returns all messages and clears queue atomically |

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

### SamplingManager

```python
from fastapi_mcp_router.session import SamplingManager
```

Manages server-to-client LLM sampling requests in stateful mode. Injected into tools with a `SamplingManager` parameter.

### RootsManager

```python
from fastapi_mcp_router.session import RootsManager
```

Tracks server operation boundary URIs.

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
