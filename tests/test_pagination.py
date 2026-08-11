"""Integration tests for cursor-based pagination on tools/list, resources/list,
prompts/list, and resources/templates/list.

Covers:
- A list method with no cursor includes nextCursor when items remain, and
  omits the key entirely once the final page is reached.
- A valid cursor returns the next batch after the cursor position, with no
  overlap against the first page, across all four paginated list methods.
- An invalid/expired cursor returns -32602 without leaking the decoded
  offset, across all four paginated list methods.
- Default page size is 100 for all four list methods; a smaller per-call
  page_size (verified at the handle_tools_list unit level) is honored.
- Under Accept: text/event-stream, each item streams as its own JSON-RPC
  partial result, and the final frame carries the authoritative page plus
  nextCursor/completion, across more than one list method.
- title/icons are absent keys (never null) on tools/list when unset.
- icons are emitted on tools/list only when the negotiated protocol version
  is 2025-11-25 or later; uniformly omitted below that gate.
"""

import base64
import json
from collections.abc import Callable

import httpx
import pytest
from fastapi import FastAPI

from fastapi_mcp_router import InMemorySessionStore, MCPToolRegistry, create_mcp_router
from fastapi_mcp_router.prompts import PromptRegistry
from fastapi_mcp_router.resources import ResourceRegistry
from fastapi_mcp_router.router import handle_tools_list
from tests.conftest import SseCapture

_ENTRY_COUNT = 105


def _make_tool(index: int) -> Callable:
    async def handler() -> str:
        """Test tool handler."""
        return str(index)

    return handler


def _build_many_entries_registries() -> tuple[MCPToolRegistry, ResourceRegistry, PromptRegistry]:
    """Build registries each populated with more than one page of entries.

    Returns:
        Tuple of (tool_registry, resource_registry, prompt_registry), each
        holding _ENTRY_COUNT registered entries.
    """
    registry = MCPToolRegistry()
    for i in range(_ENTRY_COUNT):
        registry.tool(name=f"tool_{i:03d}")(_make_tool(i))

    resource_registry = ResourceRegistry()
    for i in range(_ENTRY_COUNT):

        async def resource_handler() -> str:
            """Test resource handler."""
            return "content"

        resource_registry.resource(uri_template=f"item://{i:03d}", name=f"resource_{i:03d}")(resource_handler)

    prompt_registry = PromptRegistry()
    for i in range(_ENTRY_COUNT):

        async def prompt_handler() -> list[dict]:
            """Test prompt handler."""
            return [{"role": "user", "content": "hi"}]

        prompt_registry.prompt(name=f"prompt_{i:03d}")(prompt_handler)

    return registry, resource_registry, prompt_registry


def _build_stateless_app() -> FastAPI:
    """Build a stateless app (no auth, no session_store) with many entries.

    Returns:
        FastAPI app with MCP router mounted at /mcp.
    """
    registry, resource_registry, prompt_registry = _build_many_entries_registries()
    router = create_mcp_router(
        registry,
        resource_registry=resource_registry,
        prompt_registry=prompt_registry,
    )
    app = FastAPI()
    app.include_router(router, prefix="/mcp")
    return app


def _build_stateful_app(store: InMemorySessionStore) -> FastAPI:
    """Build a stateful app (session_store configured) with many entries.

    Args:
        store: InMemorySessionStore instance to pass to create_mcp_router.

    Returns:
        FastAPI app with MCP router mounted at /mcp.
    """
    registry, resource_registry, prompt_registry = _build_many_entries_registries()

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        return api_key is not None or bearer_token is not None

    router = create_mcp_router(
        registry,
        resource_registry=resource_registry,
        prompt_registry=prompt_registry,
        session_store=store,
        auth_validator=auth_validator,
    )
    app = FastAPI()
    app.include_router(router, prefix="/mcp")
    return app


def _rpc_body(method: str, cursor: str | None = None) -> dict[str, object]:
    """Build a JSON-RPC request body for a list method, optionally with a cursor.

    Args:
        method: JSON-RPC method name.
        cursor: Optional pagination cursor to include in params.

    Returns:
        JSON-RPC 2.0 request body dict.
    """
    body: dict[str, object] = {"jsonrpc": "2.0", "id": 1, "method": method}
    if cursor is not None:
        body["params"] = {"cursor": cursor}
    return body


async def _initialize(client: httpx.AsyncClient, extra_headers: dict[str, str] | None = None) -> str:
    """Send an initialize request and return the assigned Mcp-Session-Id.

    Args:
        client: Configured AsyncClient.
        extra_headers: Additional headers merged into the request.

    Returns:
        The Mcp-Session-Id header value from the initialize response.
    """
    headers = {"X-API-Key": "test-key", "MCP-Protocol-Version": "2025-06-18"}
    if extra_headers:
        headers.update(extra_headers)
    response = await client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        },
        headers=headers,
    )
    return response.headers["mcp-session-id"]


# ---------------------------------------------------------------------------
# nextCursor presence and omission
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.parametrize(
    ("method", "result_key"),
    [
        ("tools/list", "tools"),
        ("resources/list", "resources"),
        ("prompts/list", "prompts"),
    ],
)
async def test_list_method_first_page_includes_next_cursor(method: str, result_key: str) -> None:
    """First page of >100 items includes nextCursor and defaults to 100 items."""
    app = _build_stateless_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers={"X-API-Key": "test-key"}
    ) as client:
        response = await client.post("/mcp", json=_rpc_body(method))

    result = response.json()["result"]
    assert len(result[result_key]) == 100
    assert "nextCursor" in result


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.parametrize(
    ("method", "result_key"),
    [
        ("tools/list", "tools"),
        ("resources/list", "resources"),
        ("prompts/list", "prompts"),
    ],
)
async def test_list_method_final_page_omits_next_cursor_key(method: str, result_key: str) -> None:
    """Final page's result dict has no 'nextCursor' key at all (never null)."""
    app = _build_stateless_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers={"X-API-Key": "test-key"}
    ) as client:
        first = await client.post("/mcp", json=_rpc_body(method))
        first_cursor = first.json()["result"]["nextCursor"]
        second = await client.post("/mcp", json=_rpc_body(method, cursor=first_cursor))

    second_result = second.json()["result"]
    assert len(second_result[result_key]) == _ENTRY_COUNT - 100
    assert "nextCursor" not in second_result


# ---------------------------------------------------------------------------
# Valid cursor advances without overlap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_valid_cursor_returns_next_batch_without_overlap() -> None:
    """A valid cursor returns the next batch after the cursor position."""
    app = _build_stateless_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers={"X-API-Key": "test-key"}
    ) as client:
        first = await client.post("/mcp", json=_rpc_body("tools/list"))
        first_names = {t["name"] for t in first.json()["result"]["tools"]}
        next_cursor = first.json()["result"]["nextCursor"]

        second = await client.post("/mcp", json=_rpc_body("tools/list", cursor=next_cursor))
        second_names = {t["name"] for t in second.json()["result"]["tools"]}

    assert first_names.isdisjoint(second_names)
    assert len(second_names) == _ENTRY_COUNT - 100


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.parametrize(
    ("method", "result_key"),
    [
        ("resources/list", "resources"),
        ("prompts/list", "prompts"),
        ("resources/templates/list", "resourceTemplates"),
    ],
)
async def test_valid_cursor_returns_next_batch_without_overlap_for_other_list_methods(
    method: str, result_key: str
) -> None:
    """A valid cursor advances resources/list, prompts/list, and templates/list too."""
    app = _build_stateless_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers={"X-API-Key": "test-key"}
    ) as client:
        first = await client.post("/mcp", json=_rpc_body(method))
        first_result = first.json()["result"]
        first_names = {item.get("name") or item.get("uri") for item in first_result[result_key]}
        next_cursor = first_result["nextCursor"]

        second = await client.post("/mcp", json=_rpc_body(method, cursor=next_cursor))
        second_result = second.json()["result"]
        second_names = {item.get("name") or item.get("uri") for item in second_result[result_key]}

    assert first_names.isdisjoint(second_names)
    assert len(second_names) == _ENTRY_COUNT - 100


# ---------------------------------------------------------------------------
# Invalid cursor -> -32602, no leaked offset
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_malformed_cursor_returns_invalid_params_without_leaking_offset() -> None:
    """A garbage cursor returns -32602 with a message naming an invalid cursor."""
    app = _build_stateless_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers={"X-API-Key": "test-key"}
    ) as client:
        response = await client.post("/mcp", json=_rpc_body("tools/list", cursor="not-a-valid-cursor!!"))

    error = response.json()["error"]
    assert error["code"] == -32602
    assert "cursor" in error["message"].lower()
    # No decoded offset digits leaked into the message.
    assert not any(str(n) in error["message"] for n in range(_ENTRY_COUNT))


@pytest.mark.asyncio
@pytest.mark.integration
async def test_wrong_length_cursor_returns_invalid_params() -> None:
    """A structurally-valid-base64 but wrong-length cursor is rejected as malformed."""
    bad_cursor = base64.urlsafe_b64encode(b"short").decode("ascii")
    app = _build_stateless_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers={"X-API-Key": "test-key"}
    ) as client:
        response = await client.post("/mcp", json=_rpc_body("resources/list", cursor=bad_cursor))

    error = response.json()["error"]
    assert error["code"] == -32602


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.parametrize(
    "method",
    ["tools/list", "resources/list", "prompts/list", "resources/templates/list"],
)
async def test_malformed_cursor_returns_invalid_params_on_every_list_method(method: str) -> None:
    """Every paginated list method rejects a garbage cursor with -32602 and no offset leak."""
    app = _build_stateless_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers={"X-API-Key": "test-key"}
    ) as client:
        response = await client.post("/mcp", json=_rpc_body(method, cursor="not-a-valid-cursor!!"))

    error = response.json()["error"]
    assert error["code"] == -32602
    assert "cursor" in error["message"].lower()
    assert not any(str(n) in error["message"] for n in range(_ENTRY_COUNT))


# ---------------------------------------------------------------------------
# Default page size 100 / per-call override
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_handle_tools_list_default_page_size_is_100() -> None:
    """handle_tools_list defaults to a page size of 100 when unspecified."""
    registry = MCPToolRegistry()
    for i in range(_ENTRY_COUNT):
        registry.tool(name=f"tool_{i:03d}")(_make_tool(i))

    result = handle_tools_list(registry)
    tools = result["tools"]
    assert isinstance(tools, list)

    assert len(tools) == 100
    assert "nextCursor" in result


@pytest.mark.unit
def test_handle_tools_list_page_size_is_configurable() -> None:
    """handle_tools_list honors a caller-supplied page_size override."""
    registry = MCPToolRegistry()
    for i in range(_ENTRY_COUNT):
        registry.tool(name=f"tool_{i:03d}")(_make_tool(i))

    result = handle_tools_list(registry, page_size=10)
    tools = result["tools"]
    assert isinstance(tools, list)

    assert len(tools) == 10
    assert "nextCursor" in result


# ---------------------------------------------------------------------------
# SSE partial-result streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sse_streams_each_item_as_partial_result_with_final_frame_carrying_next_cursor() -> None:
    """Under Accept: text/event-stream, each item streams as a partial result
    and the final frame carries the authoritative page plus nextCursor."""
    store = InMemorySessionStore()
    app = _build_stateful_app(store)

    capture = SseCapture(app)
    transport = httpx.ASGITransport(app=capture)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _initialize(client, extra_headers={"Accept": "application/json"})

        response = await client.post(
            "/mcp",
            json=_rpc_body("tools/list"),
            headers={
                "X-API-Key": "test-key",
                "MCP-Protocol-Version": "2025-06-18",
                "Mcp-Session-Id": session_id,
                "Accept": "text/event-stream",
            },
        )

    assert "text/event-stream" in response.headers.get("content-type", "")

    frames = [chunk for chunk in "".join(capture.chunks).split("\n\n") if chunk.strip()]
    data_frames = [f for f in frames if "data:" in f]
    assert len(data_frames) == 101  # 100 partial-result frames + 1 final frame

    payloads = [json.loads(f.split("data: ", 1)[1]) for f in data_frames]

    for partial in payloads[:-1]:
        assert len(partial["result"]["tools"]) == 1

    final = payloads[-1]
    assert len(final["result"]["tools"]) == 100
    assert "nextCursor" in final["result"]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sse_final_page_frame_omits_next_cursor_key() -> None:
    """The final SSE frame for the last page carries no nextCursor key."""
    store = InMemorySessionStore()
    app = _build_stateful_app(store)

    capture = SseCapture(app)
    transport = httpx.ASGITransport(app=capture)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _initialize(client, extra_headers={"Accept": "application/json"})

        first = await client.post(
            "/mcp",
            json=_rpc_body("tools/list"),
            headers={
                "X-API-Key": "test-key",
                "MCP-Protocol-Version": "2025-06-18",
                "Mcp-Session-Id": session_id,
            },
        )
        next_cursor = first.json()["result"]["nextCursor"]

        response = await client.post(
            "/mcp",
            json=_rpc_body("tools/list", cursor=next_cursor),
            headers={
                "X-API-Key": "test-key",
                "MCP-Protocol-Version": "2025-06-18",
                "Mcp-Session-Id": session_id,
                "Accept": "text/event-stream",
            },
        )

    assert "text/event-stream" in response.headers.get("content-type", "")

    frames = [chunk for chunk in "".join(capture.chunks).split("\n\n") if chunk.strip()]
    data_frames = [f for f in frames if "data:" in f]
    final_payload = json.loads(data_frames[-1].split("data: ", 1)[1])

    assert "nextCursor" not in final_payload["result"]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sse_streams_resources_list_partial_results_with_final_next_cursor() -> None:
    """SSE partial-result streaming applies to resources/list too, not just tools/list."""
    store = InMemorySessionStore()
    app = _build_stateful_app(store)

    capture = SseCapture(app)
    transport = httpx.ASGITransport(app=capture)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _initialize(client, extra_headers={"Accept": "application/json"})

        response = await client.post(
            "/mcp",
            json=_rpc_body("resources/list"),
            headers={
                "X-API-Key": "test-key",
                "MCP-Protocol-Version": "2025-06-18",
                "Mcp-Session-Id": session_id,
                "Accept": "text/event-stream",
            },
        )

    assert "text/event-stream" in response.headers.get("content-type", "")

    frames = [chunk for chunk in "".join(capture.chunks).split("\n\n") if chunk.strip()]
    data_frames = [f for f in frames if "data:" in f]
    assert len(data_frames) == 101  # 100 partial-result frames + 1 final frame

    payloads = [json.loads(f.split("data: ", 1)[1]) for f in data_frames]

    for partial in payloads[:-1]:
        assert len(partial["result"]["resources"]) == 1

    final = payloads[-1]
    assert len(final["result"]["resources"]) == 100
    assert "nextCursor" in final["result"]


# ---------------------------------------------------------------------------
# title/icons omission on tools/list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_tools_list_omits_title_and_icons_when_unset() -> None:
    """A tool registered without title/icons has neither key on tools/list (never null)."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def plain_tool() -> str:
        """A tool with no title or icons."""
        return "ok"

    router = create_mcp_router(registry)
    app = FastAPI()
    app.include_router(router, prefix="/mcp")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers={"X-API-Key": "test-key"}
    ) as client:
        response = await client.post("/mcp", json=_rpc_body("tools/list"))

    tool = response.json()["result"]["tools"][0]
    assert "title" not in tool
    assert "icons" not in tool


@pytest.mark.asyncio
@pytest.mark.integration
async def test_tools_list_includes_icons_when_negotiated_version_supports_them() -> None:
    """A tool registered with icons emits them on tools/list when the client negotiates 2025-11-25."""
    registry = MCPToolRegistry()

    @registry.tool(icons=[{"src": "https://example.com/icon.png"}])
    async def icon_tool() -> str:
        """A tool with an icon."""
        return "ok"

    router = create_mcp_router(registry)
    app = FastAPI()
    app.include_router(router, prefix="/mcp")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"X-API-Key": "test-key", "MCP-Protocol-Version": "2025-11-25"},
    ) as client:
        response = await client.post("/mcp", json=_rpc_body("tools/list"))

    tool = response.json()["result"]["tools"][0]
    assert tool["icons"] == [{"src": "https://example.com/icon.png"}]


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.parametrize("protocol_version", ["2025-06-18", "2025-03-26"])
async def test_tools_list_omits_icons_for_versions_below_2025_11_25(protocol_version: str) -> None:
    """A tool with icons has the key uniformly omitted on tools/list below the 2025-11-25 gate."""
    registry = MCPToolRegistry()

    @registry.tool(icons=[{"src": "https://example.com/icon.png"}])
    async def icon_tool() -> str:
        """A tool with an icon."""
        return "ok"

    router = create_mcp_router(registry)
    app = FastAPI()
    app.include_router(router, prefix="/mcp")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"X-API-Key": "test-key", "MCP-Protocol-Version": protocol_version},
    ) as client:
        response = await client.post("/mcp", json=_rpc_body("tools/list"))

    body = response.json()
    assert "error" not in body
    assert "icons" not in body["result"]["tools"][0]
