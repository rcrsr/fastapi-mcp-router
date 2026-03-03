"""
Shared test fixtures and helper classes for fastapi-mcp-router test suite.

Provides five pytest fixtures used across integration test files:
- registry: empty MCPToolRegistry per test
- app: stateless FastAPI app with MCP router mounted at /mcp
- client: AsyncClient for the stateless app (no auth)
- auth_client: AsyncClient for app with X-API-Key auth validator (stateless)
- session_client: AsyncClient for app with Bearer auth + session callbacks (stateful)

Also provides one shared ASGI helper class:
- SseCapture: ASGI middleware that captures SSE response headers and body chunks
"""

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from datetime import UTC, datetime
from uuid import UUID, uuid4

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI, Request

from fastapi_mcp_router import MCPToolRegistry, create_mcp_router
from fastapi_mcp_router.types import McpSessionData


@pytest.fixture(name="registry")
def registry_fixture() -> MCPToolRegistry:
    """Create an empty tool registry scoped to one test.

    Returns:
        A new MCPToolRegistry with no tools registered.
    """
    return MCPToolRegistry()


@pytest.fixture(name="app")
def app_fixture(registry: MCPToolRegistry) -> FastAPI:
    """Create a stateless FastAPI app with MCP router mounted at /mcp.

    Args:
        registry: Empty MCPToolRegistry provided by the registry fixture.

    Returns:
        FastAPI app with MCP router included at prefix /mcp.
    """
    fastapi_app = FastAPI()
    mcp_router = create_mcp_router(registry)
    fastapi_app.include_router(mcp_router, prefix="/mcp")
    return fastapi_app


@pytest_asyncio.fixture(name="client")
async def client_fixture(app: FastAPI) -> AsyncGenerator[httpx.AsyncClient]:
    """Create an AsyncClient for the stateless app with no auth headers.

    Args:
        app: FastAPI app provided by the app fixture.

    Yields:
        httpx.AsyncClient configured with ASGITransport and base_url http://test.
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client


@pytest_asyncio.fixture(name="auth_client")
async def auth_client_fixture(registry: MCPToolRegistry) -> AsyncGenerator[httpx.AsyncClient]:
    """Create an AsyncClient for an app with X-API-Key auth validation (stateless).

    The app accepts requests with X-API-Key: test-key. No session callbacks
    are provided, keeping the app in stateless mode.

    Args:
        registry: Empty MCPToolRegistry provided by the registry fixture.

    Yields:
        httpx.AsyncClient configured with ASGITransport and base_url http://test.
    """

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        """Accept requests with X-API-Key: test-key."""
        return api_key == "test-key"

    fastapi_app = FastAPI()
    mcp_router = create_mcp_router(registry, auth_validator=auth_validator)
    fastapi_app.include_router(mcp_router, prefix="/mcp")

    transport = httpx.ASGITransport(app=fastapi_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client


@pytest_asyncio.fixture(name="session_client")
async def session_client_fixture(
    registry: MCPToolRegistry,
) -> AsyncGenerator[httpx.AsyncClient]:
    """Create an AsyncClient for an app with Bearer auth and session callbacks (stateful).

    The app accepts Bearer tokens and maintains sessions in an in-memory dict.
    Middleware sets request.state.oauth_client_id and request.state.connection_id
    for Bearer token requests so the session_creator can associate them.

    Args:
        registry: Empty MCPToolRegistry provided by the registry fixture.

    Yields:
        httpx.AsyncClient configured with ASGITransport and base_url http://test.
    """
    session_store: dict[str, McpSessionData] = {}

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        """Accept all Bearer token requests."""
        return bearer_token is not None

    async def session_getter(session_id: str) -> McpSessionData | None:
        """Return session data or None when absent from store.

        Args:
            session_id: Session identifier to look up.

        Returns:
            McpSessionData if found, None otherwise.
        """
        return session_store.get(session_id)

    async def session_creator(oauth_client_id: UUID | None, connection_id: UUID | None) -> str:
        """Generate a UUID v4 session ID and store McpSessionData.

        Args:
            oauth_client_id: UUID of the OAuth client, or None.
            connection_id: UUID of the connection, or None.

        Returns:
            The new session ID string.
        """
        session_id = str(uuid4())
        session_store[session_id] = McpSessionData(
            session_id=session_id,
            oauth_client_id=oauth_client_id,
            connection_id=connection_id,
            last_event_id=0,
            created_at=datetime.now(UTC),
        )
        return session_id

    fastapi_app = FastAPI()

    @fastapi_app.middleware("http")
    async def set_bearer_state(request: Request, call_next):
        """Set oauth_client_id and connection_id on request.state for Bearer tokens."""
        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            request.state.oauth_client_id = uuid4()
            request.state.connection_id = uuid4()
        response = await call_next(request)
        return response

    mcp_router = create_mcp_router(
        registry,
        auth_validator=auth_validator,
        session_getter=session_getter,
        session_creator=session_creator,
    )
    fastapi_app.include_router(mcp_router, prefix="/mcp")

    transport = httpx.ASGITransport(app=fastapi_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client


# ---------------------------------------------------------------------------
# Shared ASGI helper class
# ---------------------------------------------------------------------------


class SseCapture:
    """ASGI middleware that captures SSE response headers and body chunks.

    Intercepts http.response.start to record the status code and headers,
    then signals headers_received so tests can inspect them before the
    streaming body completes.

    Attributes:
        app: Inner ASGI application to wrap.
        status_code: HTTP status code from http.response.start, or None.
        headers: Response headers dict (lower-case keys), empty until set.
        chunks: Decoded body chunks received.
        headers_received: Event that fires when http.response.start arrives.
    """

    def __init__(self, app: Callable[..., Awaitable[None]]) -> None:
        self.app = app
        self.status_code: int | None = None
        self.headers: dict[str, str] = {}
        self.chunks: list[str] = []
        self.headers_received: asyncio.Event = asyncio.Event()

    async def __call__(
        self,
        scope: object,
        receive: object,
        send: Callable[..., Awaitable[None]],
    ) -> None:
        """Wrap the inner app, capturing response start and body messages.

        Args:
            scope: ASGI connection scope.
            receive: ASGI receive callable.
            send: ASGI send callable.
        """

        async def capturing_send(message: dict[str, object]) -> None:
            if message["type"] == "http.response.start":
                status = message["status"]
                assert isinstance(status, int)
                self.status_code = status
                raw_headers = message.get("headers", [])
                assert isinstance(raw_headers, list)
                self.headers = {
                    k.decode(): v.decode()
                    for k, v in raw_headers  # type: ignore[misc]
                }
                self.headers_received.set()
            elif message["type"] == "http.response.body":
                body = message.get("body", b"")
                assert isinstance(body, bytes)
                if body:
                    self.chunks.append(body.decode())
            await send(message)

        await self.app(scope, receive, capturing_send)
