# fastapi-mcp-router

Add MCP to your existing FastAPI app. Register tools, resources, and prompts with decorators. Use `Depends()`, `Request`, and `BackgroundTasks` the same way you already do.

## Why fastapi-mcp-router

- **It's just an `APIRouter`.** Mount it like any other router. No separate framework, no new server process.
- **Your DI still works.** `Depends()`, `Request`, `BackgroundTasks` — same patterns, same middleware.
- **2 dependencies.** FastAPI and Pydantic. Nothing else.
- **No lock-in.** Tools are regular async functions. Call them from tests, CLI scripts, or other endpoints without MCP.
- **Lambda-ready.** Stateless mode + Mangum. No adapter layer.

**Use FastMCP instead** if you need STDIO transport, OpenAPI spec imports, managed hosting, or Python 3.10-3.12 support.

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

- **Full MCP 2025-06-18 spec** — tools, resources, prompts, sampling, logging, completions, elicitation
- **Streamable HTTP** — JSON or SSE response based on `Accept` header
- **Streaming tools** — return `AsyncGenerator` for incremental results
- **Session management** — in-memory and Redis stores for stateful connections
- **Progress reporting** — inject `ProgressCallback` into tool signatures
- **Auth** — `auth_validator` callback + OAuth 2.1 PRM (RFC 9728)
- **OpenTelemetry** — opt-in spans and counters via `pip install fastapi-mcp-router[otel]`
- **Lambda-ready** — stateless mode works with Mangum, no adapter overhead

## Documentation

- [Quick Start](docs/quickstart.md) — installation, first tool, stateful mode, auth, Lambda
- [Guide](docs/guide.md) — resources, prompts, streaming, sessions, telemetry
- [API Reference](docs/reference.md) — all exports, types, and configuration options

## License

MIT
