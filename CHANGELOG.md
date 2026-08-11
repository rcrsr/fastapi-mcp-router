# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **MCP 2025-11-25 support:** Protocol revision `2025-11-25` is supported additively alongside `2025-06-18` and `2025-03-26`; clients negotiating an older revision see payloads unchanged from 0.3.1. ([#12](https://github.com/rcrsr/fastapi-mcp-router/pull/12))
- **Version-negotiation clamping:** An unrecognized `protocolVersion` is now clamped down to the newest supported revision instead of rejected. An unclampable version returns JSON-RPC `-32602` with `data.supported` and `data.requested` at HTTP 200, replacing the former HTTP 400 plain-error body. ([#12](https://github.com/rcrsr/fastapi-mcp-router/pull/12))
- **Cursor-based pagination:** `tools/list`, `resources/list`, `prompts/list`, and `resources/templates/list` accept an opaque `cursor` and return `nextCursor`, defaulting to 100 items per page. `encode_cursor` and `paginate` are exported for custom list handlers. ([#12](https://github.com/rcrsr/fastapi-mcp-router/pull/12))
- **`resources/templates/list` method:** URI templates are now served from their own paginated method, in addition to the existing inline `resourceTemplates` bundle on `resources/list`. ([#12](https://github.com/rcrsr/fastapi-mcp-router/pull/12))
- **Expanded `initialize` capabilities:** The capability block now advertises `tools.listChanged`, `resources.listChanged`, `resources.subscribe`, `prompts.listChanged`, and `logging` when a session store is configured and the corresponding registry is non-empty. Stateless servers keep the bare `0.3.1` capability shape. ([#12](https://github.com/rcrsr/fastapi-mcp-router/pull/12))
- **List-changed notifications:** `notify_tools_list_changed()`, `notify_resources_list_changed()`, `notify_prompts_list_changed()`, and `notify_resource_updated()` on `MCPRouter` fan out over SSE to subscribed sessions. `SessionStore` gains `list_sessions()` and `find_subscribers()` to support this. ([#12](https://github.com/rcrsr/fastapi-mcp-router/pull/12))
- **Typed content blocks:** `ImageContent`, `AudioContent`, and `ResourceLinkContent` join `TextContent` as exported tool-response types, emitted only to clients on `2025-11-25` or newer. ([#12](https://github.com/rcrsr/fastapi-mcp-router/pull/12))
- **Tool `title` and `ToolAnnotations`:** `@registry.tool()` accepts a human-readable `title` and typed annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`). ([#12](https://github.com/rcrsr/fastapi-mcp-router/pull/12))
- **`Icon` on all four carriers:** The `@tool`, `@resource`, and `@prompt` decorators accept `icons=[{"src": ..., "mimeType": ...}]`, and resource templates carry them too. Icons are validated at registration time against an HTTPS/`data:` scheme allowlist and an image MIME allowlist, and SVG sources are scanned for executable content across raw, percent-encoded, and base64 representations. This is a best-effort blocklist, not a sanitizer — apply CSP `script-src 'none'` or DOMPurify when rendering icons in a browser. ([#12](https://github.com/rcrsr/fastapi-mcp-router/pull/12))

### Changed

- **BREAKING:** `SessionStore` ABC adds two new required abstract methods, `list_sessions()` and `find_subscribers()`; downstream custom `SessionStore` implementations must implement both to remain instantiable. ([#12](https://github.com/rcrsr/fastapi-mcp-router/pull/12))
- **BREAKING:** Server-hosted `roots/list` handling is removed. The `roots_manager` parameter and its `RootsManager()` default are gone from `create_mcp_router()`, and `RootsManager` (`session.py`) and `Root` (`types.py`) no longer exist. A `roots/list` request now falls through to the generic unknown-method path and returns a JSON-RPC `-32601` error. Neither symbol was previously exported in `__all__`, so this affects only callers that passed `roots_manager=` directly.

  This removes functionality that was never spec-conformant: the MCP specification defines `roots/list` as a request the *server* sends to the *client* to discover the client's workspace roots — a client capability, not an inbound method a server handles. This library's server-hosted implementation inverted that direction. Roots itself is deprecated as of the `2026-07-28` MCP specification revision, with implementations encouraged to migrate toward passing directories or files via tool parameters, resource URIs, or server configuration instead. No server-side replacement is offered here — that is intentional, since a conformant client-hosted roots capability is outside this server library's scope.

  This ships as a `0.4.0` minor release rather than a major version because the project is still in SemVer's `0.x` initial-development phase, where the public API is explicitly unstable and a disclosed breaking change may ship in a minor release. ([#12](https://github.com/rcrsr/fastapi-mcp-router/pull/12))

## [0.3.1] - 2026-04-07

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
