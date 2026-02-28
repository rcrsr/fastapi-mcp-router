# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
