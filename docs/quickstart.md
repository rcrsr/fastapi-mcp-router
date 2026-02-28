# Quick Start

Install the package, register a tool, and serve it over HTTP in under 5 minutes.

## Prerequisites

- Python 3.13+
- FastAPI 0.134.0+
- An ASGI server (uvicorn recommended)

## Install

```bash
pip install fastapi-mcp-router uvicorn

# With OpenTelemetry support
pip install fastapi-mcp-router[otel] uvicorn
```

## Minimal Server

Create `server.py`:

```python
from fastapi import FastAPI
from fastapi_mcp_router import MCPRouter

app = FastAPI()
mcp = MCPRouter()

@mcp.tool()
async def greet(name: str) -> str:
    """Greet a user by name."""
    return f"Hello, {name}!"

app.include_router(mcp, prefix="/mcp")
```

Start the server:

```bash
uvicorn server:app --reload
```

## Test with MCP Inspector

```bash
npx @modelcontextprotocol/inspector
```

Set transport to **Streamable HTTP** and URL to `http://localhost:8000/mcp`.

## Test with curl

List available tools:

```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "MCP-Protocol-Version: 2025-06-18" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
      "protocolVersion": "2025-06-18",
      "clientInfo": {"name": "curl-test", "version": "1.0"},
      "capabilities": {}
    }
  }'
```

Call the tool:

```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "MCP-Protocol-Version: 2025-06-18" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
      "name": "greet",
      "arguments": {"name": "World"}
    }
  }'
```

## Add a Resource

```python
@mcp.resource("config://{key}")
async def get_config(key: str) -> str:
    """Read a configuration value."""
    store = {"debug": "true", "version": "1.0"}
    return store.get(key, "unknown")
```

## Add a Prompt

```python
@mcp.prompt()
async def review_code(language: str, style: str = "concise") -> list[dict]:
    """Generate a code review prompt."""
    return [{"role": "user", "content": f"Review this {language} code. Be {style}."}]
```

## Enable Stateful Mode

Stateful mode adds Streamable HTTP, session management, and progress notifications:

```python
from fastapi_mcp_router import MCPRouter, InMemorySessionStore

mcp = MCPRouter(
    session_store=InMemorySessionStore(ttl_seconds=3600),
    stateful=True,
)
```

POST requests with `Accept: text/event-stream` return SSE responses. Without that header, POST returns JSON (same as stateless mode). To also register a legacy GET SSE endpoint, set `legacy_sse=True`.

## Deploy to AWS Lambda

Wrap the app with Mangum for Lambda (stateless mode only):

```python
from mangum import Mangum
handler = Mangum(app, lifespan="off")
```

## Next Steps

- [Guide](guide.md) covers tools, resources, prompts, sessions, streaming, auth, and error handling in depth.
- [Reference](reference.md) documents every public class, function, and type alias.
