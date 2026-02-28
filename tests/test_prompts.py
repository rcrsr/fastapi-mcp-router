"""Tests for PromptRegistry and prompts MCP protocol methods.

Covers AC-29 through AC-83, EC-5 through EC-8, and IC-18:
- @prompt() decorator registers handler with auto-generated arguments
- prompts/list returns name, description, arguments
- prompts/get calls handler with validated args and returns messages
- initialize includes prompts capability when prompts registered
- No prompts -> no prompts capability; prompt methods return -32601
- Missing required argument -> MCPError -32602
- Prompt not found -> MCPError -32602
- Handler raises -> MCPError -32603
- No prompts registered, method called -> MCPError -32601
"""

import httpx
import pytest
from fastapi import FastAPI

from fastapi_mcp_router import MCPToolRegistry, PromptRegistry, create_mcp_router
from fastapi_mcp_router.exceptions import MCPError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_jsonrpc(
    method: str,
    params: dict[str, object] | None = None,
    request_id: int = 1,
) -> dict[str, object]:
    """Build a minimal JSON-RPC 2.0 request body.

    Args:
        method: JSON-RPC method name
        params: Optional parameters dict
        request_id: Request identifier

    Returns:
        JSON-RPC 2.0 request dict
    """
    body: dict[str, object] = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        body["params"] = params
    return body


def build_app_with_prompts(prompt_registry: PromptRegistry) -> FastAPI:
    """Create a FastAPI app with MCP router and the given prompt registry.

    Args:
        prompt_registry: PromptRegistry to attach to the MCP router

    Returns:
        FastAPI app with MCP router mounted at /mcp
    """
    registry = MCPToolRegistry()
    app = FastAPI()
    mcp_router = create_mcp_router(registry, prompt_registry=prompt_registry)
    app.include_router(mcp_router, prefix="/mcp")
    return app


def build_app_no_prompts() -> FastAPI:
    """Create a FastAPI app with MCP router and no prompt registry.

    Returns:
        FastAPI app with MCP router mounted at /mcp, no prompt support
    """
    registry = MCPToolRegistry()
    app = FastAPI()
    mcp_router = create_mcp_router(registry)
    app.include_router(mcp_router, prefix="/mcp")
    return app


MCP_HEADERS = {"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"}


# ---------------------------------------------------------------------------
# AC-29: @prompt() decorator registers handler with auto-generated arguments
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_prompt_decorator_registers_handler_with_auto_arguments() -> None:
    """AC-29: @prompt() registers handler; arguments derived from signature."""
    registry = PromptRegistry()

    @registry.prompt()
    async def greet_user(username: str, lang: str = "en") -> list[dict]:
        """Greet a user."""
        return [{"role": "user", "content": f"Hello {username}"}]

    assert registry.has_prompts()
    prompts = registry.list_prompts()
    assert len(prompts) == 1
    assert prompts[0]["name"] == "greet_user"

    args: list[dict[str, object]] = list(prompts[0]["arguments"])  # type: ignore[arg-type]
    req_args = [a for a in args if a["required"] is True]
    opt_args = [a for a in args if a["required"] is not True]

    assert len(req_args) == 1
    assert req_args[0]["name"] == "username"
    assert len(opt_args) == 1
    assert opt_args[0]["name"] == "lang"


@pytest.mark.unit
def test_prompt_decorator_with_custom_name_and_description() -> None:
    """AC-29: name and description override decorator params."""
    registry = PromptRegistry()

    @registry.prompt(name="custom_name", description="Custom description")
    async def my_prompt(topic: str) -> list[dict]:
        """Original docstring."""
        return [{"role": "user", "content": topic}]

    prompts = registry.list_prompts()
    assert len(prompts) == 1
    assert prompts[0]["name"] == "custom_name"
    assert prompts[0]["description"] == "Custom description"


@pytest.mark.unit
def test_prompt_decorator_uses_docstring_as_default_description() -> None:
    """AC-29: When no description provided, docstring is used."""
    registry = PromptRegistry()

    @registry.prompt()
    async def my_prompt(topic: str) -> list[dict]:
        """Summarize a topic."""
        return [{"role": "user", "content": topic}]

    prompts = registry.list_prompts()
    assert prompts[0]["description"] == "Summarize a topic."


# ---------------------------------------------------------------------------
# AC-30: prompts/list returns name, description, arguments
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_prompts_list_returns_name_description_arguments() -> None:
    """AC-30: prompts/list response includes name, description, arguments fields."""
    prompt_registry = PromptRegistry()

    @prompt_registry.prompt()
    async def greet_user(username: str, lang: str = "en") -> list[dict]:
        """Greet a user."""
        return [{"role": "user", "content": f"Hello {username}"}]

    app = build_app_with_prompts(prompt_registry)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=make_jsonrpc("prompts/list"),
            headers=MCP_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body
    prompts = body["result"]["prompts"]
    assert len(prompts) == 1
    p = prompts[0]
    assert p["name"] == "greet_user"
    assert "description" in p
    assert "arguments" in p


# ---------------------------------------------------------------------------
# AC-31: prompts/get calls handler with validated args, returns messages
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_prompts_get_calls_handler_and_returns_messages() -> None:
    """AC-31: prompts/get returns messages list with role and content."""
    prompt_registry = PromptRegistry()

    @prompt_registry.prompt()
    async def greet_user(username: str) -> list[dict]:
        """Greet a user."""
        return [{"role": "user", "content": f"Hello {username}"}]

    app = build_app_with_prompts(prompt_registry)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=make_jsonrpc(
                "prompts/get",
                params={"name": "greet_user", "arguments": {"username": "Alice"}},
            ),
            headers=MCP_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body
    messages = body["result"]["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "Alice" in messages[0]["content"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_prompts_get_sync_handler_works() -> None:
    """AC-31: Sync handler works the same as async handler."""
    prompt_registry = PromptRegistry()

    @prompt_registry.prompt()
    def sync_prompt(topic: str) -> list[dict]:
        """A sync prompt handler."""
        return [{"role": "assistant", "content": f"Topic: {topic}"}]

    app = build_app_with_prompts(prompt_registry)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=make_jsonrpc(
                "prompts/get",
                params={"name": "sync_prompt", "arguments": {"topic": "AI"}},
            ),
            headers=MCP_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    messages = body["result"]["messages"]
    assert messages[0]["role"] == "assistant"
    assert "AI" in messages[0]["content"]


# ---------------------------------------------------------------------------
# AC-32: initialize includes prompts capability when prompts registered
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_initialize_includes_prompts_capability_when_registry_has_prompts() -> None:
    """AC-32: initialize result.capabilities.prompts exists when prompts registered."""
    prompt_registry = PromptRegistry()

    @prompt_registry.prompt()
    async def example_prompt(text: str) -> list[dict]:
        """Example prompt."""
        return [{"role": "user", "content": text}]

    app = build_app_with_prompts(prompt_registry)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=make_jsonrpc(
                "initialize",
                params={"protocolVersion": "2025-06-18", "clientInfo": {}, "capabilities": {}},
            ),
            headers=MCP_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    capabilities = body["result"]["capabilities"]
    assert "prompts" in capabilities


# ---------------------------------------------------------------------------
# AC-33: No prompts -> no prompts capability in initialize
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_initialize_no_prompts_capability_when_no_registry() -> None:
    """AC-33: initialize result.capabilities has no prompts key when no registry."""
    app = build_app_no_prompts()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=make_jsonrpc(
                "initialize",
                params={"protocolVersion": "2025-06-18", "clientInfo": {}, "capabilities": {}},
            ),
            headers=MCP_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    capabilities = body["result"]["capabilities"]
    assert "prompts" not in capabilities


@pytest.mark.integration
@pytest.mark.asyncio
async def test_initialize_no_prompts_capability_when_empty_registry() -> None:
    """AC-33: initialize result.capabilities has no prompts key when registry is empty."""
    empty_registry = PromptRegistry()
    app = build_app_with_prompts(empty_registry)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=make_jsonrpc(
                "initialize",
                params={"protocolVersion": "2025-06-18", "clientInfo": {}, "capabilities": {}},
            ),
            headers=MCP_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    capabilities = body["result"]["capabilities"]
    assert "prompts" not in capabilities


# ---------------------------------------------------------------------------
# AC-33 continued: prompts methods return -32601 when no registry
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_prompts_list_returns_32601_when_no_registry() -> None:
    """AC-33 / EC-8: prompts/list returns -32601 when no prompt registry provided."""
    app = build_app_no_prompts()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=make_jsonrpc("prompts/list"),
            headers=MCP_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["error"]["code"] == -32601


@pytest.mark.integration
@pytest.mark.asyncio
async def test_prompts_get_returns_32601_when_no_registry() -> None:
    """AC-33 / EC-8: prompts/get returns -32601 when no prompt registry provided."""
    app = build_app_no_prompts()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=make_jsonrpc(
                "prompts/get",
                params={"name": "any_prompt", "arguments": {}},
            ),
            headers=MCP_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["error"]["code"] == -32601


# ---------------------------------------------------------------------------
# AC-34 / AC-83: Missing required argument -> -32602
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_prompt_missing_required_arg_raises_mcp_error_32602() -> None:
    """AC-34 / AC-83 / EC-6: Missing required argument raises MCPError -32602."""
    registry = PromptRegistry()

    @registry.prompt()
    async def greet_user(username: str) -> list[dict]:
        """Greet a user."""
        return [{"role": "user", "content": f"Hello {username}"}]

    with pytest.raises(MCPError) as exc_info:
        await registry.get_prompt("greet_user", {})

    assert exc_info.value.code == -32602
    assert "username" in exc_info.value.message


@pytest.mark.integration
@pytest.mark.asyncio
async def test_prompts_get_missing_required_arg_returns_32602() -> None:
    """AC-34 / AC-83 / EC-6: prompts/get returns -32602 when required arg is absent."""
    prompt_registry = PromptRegistry()

    @prompt_registry.prompt()
    async def greet_user(username: str) -> list[dict]:
        """Greet a user."""
        return [{"role": "user", "content": f"Hello {username}"}]

    app = build_app_with_prompts(prompt_registry)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=make_jsonrpc(
                "prompts/get",
                params={"name": "greet_user", "arguments": {}},
            ),
            headers=MCP_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["error"]["code"] == -32602


# ---------------------------------------------------------------------------
# EC-5: Prompt not found -> -32602
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_prompt_not_found_raises_mcp_error_32602() -> None:
    """EC-5: get_prompt with unknown name raises MCPError -32602."""
    registry = PromptRegistry()

    with pytest.raises(MCPError) as exc_info:
        await registry.get_prompt("nonexistent", {})

    assert exc_info.value.code == -32602
    assert "nonexistent" in exc_info.value.message


@pytest.mark.integration
@pytest.mark.asyncio
async def test_prompts_get_unknown_prompt_returns_32602() -> None:
    """EC-5: prompts/get with unknown name returns -32602."""
    prompt_registry = PromptRegistry()

    @prompt_registry.prompt()
    async def known_prompt(text: str) -> list[dict]:
        """A known prompt."""
        return [{"role": "user", "content": text}]

    app = build_app_with_prompts(prompt_registry)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=make_jsonrpc(
                "prompts/get",
                params={"name": "nonexistent_prompt", "arguments": {}},
            ),
            headers=MCP_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["error"]["code"] == -32602


# ---------------------------------------------------------------------------
# EC-7: Handler raises -> -32603
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_prompt_handler_raises_wraps_as_mcp_error_32603() -> None:
    """EC-7: When handler raises a non-MCPError exception, MCPError -32603 is raised."""
    registry = PromptRegistry()

    @registry.prompt()
    async def error_prompt(arg: str) -> list[dict]:
        """A prompt that always fails."""
        raise ValueError("internal failure")

    with pytest.raises(MCPError) as exc_info:
        await registry.get_prompt("error_prompt", {"arg": "val"})

    assert exc_info.value.code == -32603


@pytest.mark.integration
@pytest.mark.asyncio
async def test_prompts_get_handler_exception_returns_32603() -> None:
    """EC-7: prompts/get returns -32603 when the handler raises an exception."""
    prompt_registry = PromptRegistry()

    @prompt_registry.prompt()
    async def error_prompt(arg: str) -> list[dict]:
        """A prompt that always fails."""
        raise ValueError("internal failure")

    app = build_app_with_prompts(prompt_registry)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=make_jsonrpc(
                "prompts/get",
                params={"name": "error_prompt", "arguments": {"arg": "val"}},
            ),
            headers=MCP_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["error"]["code"] == -32603


# ---------------------------------------------------------------------------
# EC-8: No prompts registered, prompts/list called -> -32601
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_has_prompts_returns_false_for_empty_registry() -> None:
    """EC-8: PromptRegistry.has_prompts() returns False when no prompts registered."""
    registry = PromptRegistry()
    assert registry.has_prompts() is False


@pytest.mark.unit
def test_has_prompts_returns_true_after_registration() -> None:
    """EC-8: PromptRegistry.has_prompts() returns True after registration."""
    registry = PromptRegistry()

    @registry.prompt()
    async def example(text: str) -> list[dict]:
        """Example."""
        return []

    assert registry.has_prompts() is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_prompts_list_returns_32601_when_registry_is_empty() -> None:
    """EC-8: prompts/list returns -32601 when the prompt registry has no prompts."""
    empty_registry = PromptRegistry()
    app = build_app_with_prompts(empty_registry)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=make_jsonrpc("prompts/list"),
            headers=MCP_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["error"]["code"] == -32601
