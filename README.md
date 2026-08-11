# fastapi-mcp-router

[![CI](https://github.com/rcrsr/fastapi-mcp-router/actions/workflows/ci.yml/badge.svg)](https://github.com/rcrsr/fastapi-mcp-router/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/fastapi-mcp-router)](https://pypi.org/project/fastapi-mcp-router/)
[![Python versions](https://img.shields.io/pypi/pyversions/fastapi-mcp-router)](https://pypi.org/project/fastapi-mcp-router/)
[![License](https://img.shields.io/pypi/l/fastapi-mcp-router)](https://github.com/rcrsr/fastapi-mcp-router/blob/main/LICENSE)

Add MCP to your existing FastAPI app. Register tools, resources, and prompts with decorators. Use `Depends()`, `Request`, and `BackgroundTasks` the same way you already do.

## Why fastapi-mcp-router

- **It's just an `APIRouter`.** Mount it like any other router. No separate framework, no new server process.
- **Your DI still works.** `Depends()`, `Request`, `BackgroundTasks` — same patterns, same middleware.
- **2 dependencies.** FastAPI and Pydantic. Nothing else.
- **No lock-in.** Tools are regular async functions. Call them from tests, CLI scripts, or other endpoints without MCP.
- **Lambda-ready.** Stateless mode + Mangum. No adapter layer.

**Use FastMCP instead** if you need STDIO transport, OpenAPI spec imports, or managed hosting.

## Install

```bash
pip install fastapi-mcp-router
```

## Quick Start

```python
from fastapi import FastAPI
from fastapi_mcp_router import MCPRouter

app = FastAPI()
mcp = MCPRouter()

@mcp.tool()
async def write_message(payload: str) -> dict:
    """Write coordination message."""
    return {"success": True, "message_id": "msg-123"}

@mcp.resource("project://{project_id}/config")
async def project_config(project_id: str) -> dict:
    return {"project_id": project_id, "env": "production"}

@mcp.prompt()
async def review_code(file_path: str, language: str = "python") -> list[dict]:
    return [{"role": "user", "content": f"Review {file_path} ({language})"}]

app.include_router(mcp, prefix="/mcp")
```

That's it. Your FastAPI app now speaks MCP over Streamable HTTP.

## What You Get

- **MCP `2025-11-25` (primary), with `2025-06-18` and `2025-03-26` as accepted fallbacks** — tools, resources, prompts, sampling, logging, completions, elicitation. The router negotiates protocol version per request: it clamps the client's requested version down to the highest version in that three-way whitelist that is `<=` requested, rather than echoing an unrecognized version back.
- **Pagination** — `tools/list`, `resources/list`, `resources/templates/list`, and `prompts/list` accept an opaque `cursor` and return `nextCursor` when more items remain.
- **`resources/templates/list`** — resource templates are discoverable as their own paginated method, not only nested inside `resources/list`.
- **Notifications** — `MCPRouter.notify_tools_list_changed()`, `notify_resources_list_changed()`, `notify_prompts_list_changed()`, and `notify_resource_updated(uri)` push list-changed and resource-update notifications to live sessions (stateful mode).
- **Content blocks** — tools can return `ImageContent`, `AudioContent`, and `ResourceLinkContent` alongside `TextContent`; blocks are gated to connections negotiating `2025-11-25` or later.
- **Icons** — attach `Icon` (validated, HTTPS/`data:` only, SVG-script-scanned) to tools via `title`/`icons` params on `MCPToolRegistry.tool()`, and server-wide via `ServerIcon` on `ServerInfo`.
- **Streamable HTTP** — JSON or SSE response based on `Accept` header
- **Streaming tools** — return `AsyncGenerator` for incremental results
- **Session management** — in-memory and Redis stores for stateful connections
- **Progress reporting** — inject `ProgressCallback` into tool signatures
- **Auth** — `auth_validator` callback + OAuth 2.1 PRM (RFC 9728)
- **OpenTelemetry** — opt-in spans and counters via `pip install fastapi-mcp-router[otel]`
- **Lambda-ready** — stateless mode works with Mangum, no adapter overhead

## Breaking Change in 0.4.0: Roots Removed

The MCP spec defines Roots as a **client→server** capability (the client tells the server which directories/URIs it may operate within); previous versions of this library inverted that relationship by having the server host its own root list, which was non-conformant. Roots is also deprecated outright going forward in the spec.

As of 0.4.0:

- The `roots_manager` constructor parameter of `create_mcp_router()` has been removed. If you passed `roots_manager=...`, remove it — there is no replacement parameter.
- The `roots/list` JSON-RPC method now returns `-32601` (`Method not found`) for any caller.
- `Root` and `RootsManager` were never part of the public API (`fastapi_mcp_router.__all__`), so no import statement needs to change — only the `roots_manager` constructor argument and any direct calls to `roots/list` are affected.

If your application relied on server-hosted roots, there is no drop-in replacement in this library; implement root-boundary enforcement in your own tool handlers instead.

## Documentation

- [Quick Start](docs/quickstart.md) — installation, first tool, stateful mode, auth, Lambda
- [Guide](docs/guide.md) — resources, prompts, streaming, sessions, telemetry
- [API Reference](docs/reference.md) — all exports, types, and configuration options

## License

MIT
