# Example MCP Server

Minimal FastAPI app that registers tools, resources, and prompts via `MCPRouter`. Use it to validate the library with MCP Inspector or `curl`.

## What the server registers

| Type     | Name           | Description                              |
|----------|----------------|------------------------------------------|
| Tool     | `echo`         | Returns the input message                |
| Tool     | `countdown`    | Counts down N seconds with progress      |
| Resource | `time://now`   | Current UTC timestamp                    |
| Resource | `config://{key}` | Demo config lookup (debug, version, region) |
| Prompt   | `code_review`  | Code review prompt (language + optional style) |

## Start the server

```bash
source .venv/bin/activate
uvicorn examples.server:app --reload
```

The MCP endpoint is at `http://localhost:8000/mcp`.

## Connect MCP Inspector

```bash
npx @modelcontextprotocol/inspector
```

In the Inspector UI:

1. Set transport to **Streamable HTTP**
2. Set URL to `http://localhost:8000/mcp`
3. Click **Connect**

## CLI validation

Initialize a session:

```bash
curl -s -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "MCP-Protocol-Version: 2025-06-18" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
      "protocolVersion": "2025-06-18",
      "capabilities": {},
      "clientInfo": {"name": "curl", "version": "1.0"}
    }
  }'
```

Save the `Mcp-Session-Id` header from the response, then use it for subsequent requests:

```bash
SESSION="<paste session id here>"

# List tools
curl -s -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "MCP-Protocol-Version: 2025-06-18" \
  -H "Mcp-Session-Id: $SESSION" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'

# Call echo tool
curl -s -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "MCP-Protocol-Version: 2025-06-18" \
  -H "Mcp-Session-Id: $SESSION" \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"echo","arguments":{"message":"hello"}}}'

# List resources
curl -s -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "MCP-Protocol-Version: 2025-06-18" \
  -H "Mcp-Session-Id: $SESSION" \
  -d '{"jsonrpc":"2.0","id":4,"method":"resources/list","params":{}}'

# Read a resource
curl -s -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "MCP-Protocol-Version: 2025-06-18" \
  -H "Mcp-Session-Id: $SESSION" \
  -d '{"jsonrpc":"2.0","id":5,"method":"resources/read","params":{"uri":"time://now"}}'

# List prompts
curl -s -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "MCP-Protocol-Version: 2025-06-18" \
  -H "Mcp-Session-Id: $SESSION" \
  -d '{"jsonrpc":"2.0","id":6,"method":"prompts/list","params":{}}'

# Get a prompt
curl -s -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "MCP-Protocol-Version: 2025-06-18" \
  -H "Mcp-Session-Id: $SESSION" \
  -d '{"jsonrpc":"2.0","id":7,"method":"prompts/get","params":{"name":"code_review","arguments":{"language":"python"}}}'
```
