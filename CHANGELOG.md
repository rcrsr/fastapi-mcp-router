# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- SSE stream no longer terminates on transient Redis failures; `dequeue_messages()` errors are caught and retried on the next polling tick
- SSE stream yields `event: error` with JSON-RPC error payload before closing on unrecoverable failures, replacing silent connection drops
- `RedisSessionStore.dequeue_messages()` retries once with 0.5s backoff before raising `MCPError(-32603)`, reducing empty-poll windows on Upstash idle disconnects

## [0.3.0] - 2026-03-26

### Added

- `MCPRouter.shutdown()` method signals active SSE streams to close gracefully
- `shutdown_event` parameter on `create_mcp_router()` for direct factory usage
- SSE generators yield `: server-shutdown` comment before closing on shutdown
- Graceful shutdown test suite (4 tests covering both SSE code paths)

## [0.2.1] - 2026-03-03

### Changed

- Reduce cyclomatic complexity in router and registry modules, consolidate duplicated logic, and remove dead code paths across source and test suite

## [0.2.0] - 2026-03-02

### Changed

- `AuthValidator` now accepts `Any` return type; falsy values (None, False, 0, empty string/list) trigger HTTP 401 with `WWW-Authenticate: Bearer realm="mcp"`; truthy values stored at `request.state.auth_context` for handler access
- Resource handlers now support FastAPI `Depends()` parameters with dependency resolution and schema filtering; mirrors tool registry pattern for auth and DB injection in resource providers
- Stateful POST path collects async generator tool results into JSON array instead of background drainage; enables streaming tools over stateful connections
- `asyncio.iscoroutinefunction` replaced with `inspect.iscoroutinefunction` in registry.py and resources.py to eliminate Python 3.14 deprecation warnings
- `create_prm_router()` accepts keyword-only `mcp` parameter (MCPRouter instance) that derives resource URL and authorization_servers automatically; existing `oauth_resource_metadata` dict path preserved for backward compatibility
- `AuthValidator` type alias added to public `__init__.py` exports
- Test coverage expanded to 576 tests (32 new) with 90.66% coverage and 0 deprecation warnings

## [0.1.0] - 2026-03-01

Initial release.

### Added

- `MCPRouter` — `APIRouter` subclass with `@mcp.tool()`, `@mcp.resource()`, `@mcp.prompt()` decorators
- `create_mcp_router()` factory for external registry composition
- Full MCP 2025-06-18 protocol coverage (17 methods)
- Streamable HTTP transport (JSON and SSE responses via `Accept` header)
- Legacy SSE compatibility via `legacy_sse=True`
- `MCPToolRegistry` with auto-generated JSON schemas from function signatures
- FastAPI `Depends()`, `Request`, and `BackgroundTasks` injection in tool handlers
- `ToolFilter` callback for per-connection tool filtering
- `ResourceRegistry` with decorator and provider patterns
- `FileResourceProvider` with path traversal protection and 10 MB size limit
- `PromptRegistry` with auto-generated argument metadata
- Streaming tools via `AsyncGenerator` return type
- `ProgressCallback` injection for long-running tools
- `SessionStore` ABC with `InMemorySessionStore` and `RedisSessionStore`
- `SamplingManager` and `RootsManager` for server-to-client requests
- Resource subscriptions with per-session URI change tracking
- `auth_validator` callback for API key and Bearer token authentication
- `create_prm_router()` for OAuth 2.1 Protected Resource Metadata (RFC 9728)
- `MCPError` (protocol-level) and `ToolError` (LLM-visible) error separation
- Optional OpenTelemetry spans and counters via `fastapi-mcp-router[otel]`
- Stateless mode for AWS Lambda deployment via Mangum
- Documentation: quick start, narrative guide, API reference
