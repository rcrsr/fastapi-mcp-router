"""Tests for completion/complete MCP method dispatch.

Covers AC-60, AC-61, AC-62, AC-92, EC-26:
AC-60: completion/complete returns suggestion list (max 100)
AC-61: Completions work for both prompt and resource ref types
AC-62: Empty result when no completions available
AC-92: completion/complete with no matches returns values=[], hasMore=False
EC-26: No completion handler registered -> MCPError -32601
"""

import httpx
import pytest
from fastapi import FastAPI

from fastapi_mcp_router import MCPToolRegistry, create_mcp_router

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HEADERS = {"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(completion_handler=None) -> FastAPI:
    """Create a minimal FastAPI app with optional completion_handler.

    Args:
        completion_handler: Async callable for completion/complete requests.
            If None, the router has no completion support.

    Returns:
        FastAPI app with MCP router mounted at /mcp.
    """
    registry = MCPToolRegistry()
    app = FastAPI()
    mcp_router = create_mcp_router(registry, completion_handler=completion_handler)
    app.include_router(mcp_router, prefix="/mcp")
    return app


def _completion_request(
    ref: dict[str, object],
    argument: dict[str, object],
    request_id: int = 1,
) -> dict[str, object]:
    """Build a JSON-RPC 2.0 completion/complete request body.

    Args:
        ref: Reference dict with "type" and "name" keys.
        argument: Argument dict with "name" and "value" keys.
        request_id: JSON-RPC request identifier.

    Returns:
        JSON-RPC 2.0 request dict for completion/complete.
    """
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "completion/complete",
        "params": {"ref": ref, "argument": argument},
    }


# ---------------------------------------------------------------------------
# AC-60: completion/complete returns suggestion list (max 100)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_completion_complete_returns_values() -> None:
    """AC-60: completion/complete returns the values list from the handler."""
    expected_values = ["apple", "banana", "cherry", "date", "elderberry"]

    async def handler(ref: dict, argument: dict) -> dict:
        """Return 5 fixed completion values."""
        return {"values": expected_values, "total": 5, "hasMore": False}

    app = _make_app(completion_handler=handler)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=_completion_request(
                ref={"type": "ref/prompt", "name": "my_prompt"},
                argument={"name": "fruit", "value": "a"},
            ),
            headers=_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body, f"Expected result, got: {body}"
    completion = body["result"]["completion"]
    assert completion["values"] == expected_values
    assert len(completion["values"]) == 5


@pytest.mark.integration
@pytest.mark.asyncio
async def test_completion_complete_caps_values_at_100() -> None:
    """AC-60: Handler returning 150 values is capped to exactly 100."""
    all_values = [f"item_{i}" for i in range(150)]

    async def handler(ref: dict, argument: dict) -> dict:
        """Return 150 completion values."""
        return {"values": all_values, "total": 150, "hasMore": True}

    app = _make_app(completion_handler=handler)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=_completion_request(
                ref={"type": "ref/prompt", "name": "my_prompt"},
                argument={"name": "item", "value": "item_"},
            ),
            headers=_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body, f"Expected result, got: {body}"
    completion = body["result"]["completion"]
    assert len(completion["values"]) == 100
    assert completion["values"] == all_values[:100]


# ---------------------------------------------------------------------------
# AC-61: Completions work for both prompt and resource ref types
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_completion_complete_with_prompt_ref() -> None:
    """AC-61: completion/complete works with ref.type = ref/prompt."""
    received_refs: list[dict] = []

    async def handler(ref: dict, argument: dict) -> dict:
        """Capture ref and return fixed values."""
        received_refs.append(ref)
        return {"values": ["opt1", "opt2"], "total": 2, "hasMore": False}

    app = _make_app(completion_handler=handler)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=_completion_request(
                ref={"type": "ref/prompt", "name": "summarize"},
                argument={"name": "style", "value": "b"},
            ),
            headers=_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body, f"Expected result, got: {body}"
    assert body["result"]["completion"]["values"] == ["opt1", "opt2"]
    assert len(received_refs) == 1
    assert received_refs[0]["type"] == "ref/prompt"
    assert received_refs[0]["name"] == "summarize"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_completion_complete_with_resource_ref() -> None:
    """AC-61: completion/complete works with ref.type = ref/resource."""
    received_refs: list[dict] = []

    async def handler(ref: dict, argument: dict) -> dict:
        """Capture ref and return fixed values."""
        received_refs.append(ref)
        return {"values": ["res_a", "res_b"], "total": 2, "hasMore": False}

    app = _make_app(completion_handler=handler)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=_completion_request(
                ref={"type": "ref/resource", "uri": "file:///data/"},
                argument={"name": "path", "value": "file:///"},
            ),
            headers=_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body, f"Expected result, got: {body}"
    assert body["result"]["completion"]["values"] == ["res_a", "res_b"]
    assert len(received_refs) == 1
    assert received_refs[0]["type"] == "ref/resource"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_completion_complete_handler_receives_correct_ref_and_argument() -> None:
    """AC-61: Handler receives the exact ref and argument sent in the request."""
    captured: list[tuple[dict, dict]] = []

    async def handler(ref: dict, argument: dict) -> dict:
        """Capture ref and argument for assertion."""
        captured.append((ref, argument))
        return {"values": [], "total": 0, "hasMore": False}

    app = _make_app(completion_handler=handler)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/mcp",
            json=_completion_request(
                ref={"type": "ref/prompt", "name": "code_review"},
                argument={"name": "language", "value": "py"},
            ),
            headers=_HEADERS,
        )

    assert len(captured) == 1
    received_ref, received_argument = captured[0]
    assert received_ref == {"type": "ref/prompt", "name": "code_review"}
    assert received_argument == {"name": "language", "value": "py"}


# ---------------------------------------------------------------------------
# AC-62 / AC-92: Empty result when no completions available
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_completion_complete_empty_values() -> None:
    """AC-62/AC-92: Handler returning empty values yields values=[], hasMore=False."""

    async def handler(ref: dict, argument: dict) -> dict:
        """Return empty completion result."""
        return {"values": [], "total": 0, "hasMore": False}

    app = _make_app(completion_handler=handler)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=_completion_request(
                ref={"type": "ref/prompt", "name": "my_prompt"},
                argument={"name": "q", "value": "zzz"},
            ),
            headers=_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body, f"Expected result, got: {body}"
    completion = body["result"]["completion"]
    assert completion["values"] == []
    assert completion["hasMore"] is False


# ---------------------------------------------------------------------------
# EC-26: No completion handler registered -> MCPError -32601
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_completion_complete_no_handler_returns_32601() -> None:
    """EC-26: completion/complete with no handler registered returns JSON-RPC -32601."""
    app = _make_app(completion_handler=None)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=_completion_request(
                ref={"type": "ref/prompt", "name": "any"},
                argument={"name": "q", "value": "x"},
            ),
            headers=_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "error" in body, f"Expected error, got: {body}"
    assert body["error"]["code"] == -32601
