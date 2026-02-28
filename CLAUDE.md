# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

fastapi-mcp-router is a lightweight FastAPI integration for the Model Context Protocol (MCP). It exposes MCP tools, resources, and prompts through FastAPI endpoints using decorator-based registration and JSON-RPC 2.0 transport. Supports stateless HTTP (Lambda-compatible) and Streamable HTTP with SSE streaming.

## Commands

```bash
# Run all tests (includes coverage, enforces 80% minimum)
uv run pytest

# Run a single test
uv run pytest tests/test_registry.py::test_register_tool_default_name -v

# Run tests by marker
uv run pytest -m unit
uv run pytest -m integration

# Run tests without coverage (faster iteration)
uv run pytest --no-cov

# Build package
uv run python -m build
```

No separate lint or format commands are configured. No Makefile exists. The venv is managed by `uv` — use `uv run` to execute commands.

## Architecture

The package has 10 modules across 5,909 LOC with this dependency flow:

```
router.py → registry.py → exceptions.py
    ↓            ↓
protocol.py   types.py
    ↓
telemetry.py (leaf, zero internal imports)
    ↓
session.py, resources.py, prompts.py
```

**router.py** — `create_mcp_router()` factory and `MCPRouter` class. Handles 17 MCP methods. Streamable HTTP POST returns `JSONResponse` or `StreamingResponse` based on `Accept` header. `legacy_sse=True` registers GET endpoint for backward compatibility. OTel instrumentation via `telemetry.py`.

**registry.py** — `MCPToolRegistry` stores tool definitions registered via `@registry.tool()` decorator. Auto-generates JSON schemas from function signatures using Pydantic `TypeAdapter`. Filters out FastAPI `Depends()`, `Request`, and `BackgroundTasks` parameters from schemas.

**telemetry.py** — Leaf module with zero internal imports. Wraps optional `opentelemetry-api` behind try/except ImportError. Exports `get_tracer()` and `get_meter()`.

**protocol.py** — Pure functions for formatting JSON-RPC 2.0 responses and errors.

**types.py** — Pydantic models: `TextContent`, `ToolResponse`, `ServerInfo`, `ServerIcon`, `McpSessionData`.

**exceptions.py** — `MCPError` (protocol-level, JSON-RPC error codes -32700 to -32603) vs `ToolError` (business logic, returns `isError: true` so LLM can recover).

**session.py** — `SessionStore` ABC, `InMemorySessionStore`, `RedisSessionStore`, `SamplingManager`, `RootsManager`.

**resources.py** — `ResourceRegistry`, `ResourceProvider` ABC, `FileResourceProvider`.

**prompts.py** — `PromptRegistry` with auto-generated argument metadata.

## Key Patterns

- Streamable HTTP: POST with `Accept: text/event-stream` returns `StreamingResponse`; without returns `JSONResponse`.
- `legacy_sse=True` registers SSE GET endpoint alongside Streamable HTTP POST. Default is `False`.
- All JSON-RPC responses use HTTP 200. Errors are at the protocol level, not HTTP status codes.
- Tool handlers can be sync or async. Registry detects and handles both.
- `ToolFilter` callback enables per-connection tool filtering (e.g., OAuth vs API key scopes).
- Auth validation uses `auth_validator` callback passed to `create_mcp_router()`.
- Protocol versions supported: `2025-06-18` (primary), `2025-03-26` (fallback).
- Public API is defined in `__init__.py` via `__all__` (17 exports).
- OTel tracing is optional: `enable_telemetry=True` (default) emits spans when `opentelemetry-api` is installed. Install via `pip install fastapi-mcp-router[otel]`.

## Testing

- 544 tests across 33 files using pytest + pytest-asyncio.
- Integration tests use `httpx.AsyncClient` with FastAPI `TestClient`.
- Tests use `@pytest.mark.unit` and `@pytest.mark.integration` markers.
- Async test functions use explicit `@pytest.mark.asyncio` decorator (`asyncio_mode = "auto"` is NOT configured).
- Key test files: `test_streamable_http.py` (Streamable HTTP transport), `test_telemetry.py` (OTel), `test_sse_streaming.py` (legacy SSE).
