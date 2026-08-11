"""Tests for the `initialize` capability block in `fastapi_mcp_router/router.py`.

Covers the conditional capability shapes for `tools`, `resources`, `prompts`,
and `logging` based on the presence of a `session_store`, resource registry,
and prompt registry.
"""

import httpx
import pytest
from fastapi import FastAPI

from fastapi_mcp_router import (
    InMemorySessionStore,
    MCPToolRegistry,
    PromptRegistry,
    ResourceRegistry,
    create_mcp_router,
)

_MCP_HEADERS = {"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"}


def _rpc(method: str, params: dict | None = None, rpc_id: int = 1) -> dict:
    """Build a JSON-RPC 2.0 request body."""
    body: dict = {"jsonrpc": "2.0", "id": rpc_id, "method": method}
    if params is not None:
        body["params"] = params
    return body


def _make_app(
    *,
    with_session_store: bool = False,
    resource_registry: ResourceRegistry | None = None,
    prompt_registry: PromptRegistry | None = None,
) -> FastAPI:
    """Create a minimal FastAPI app with the MCP router mounted at /mcp."""
    tool_registry = MCPToolRegistry()
    session_store = InMemorySessionStore() if with_session_store else None
    app = FastAPI()
    router = create_mcp_router(
        tool_registry,
        session_store=session_store,
        resource_registry=resource_registry,
        prompt_registry=prompt_registry,
    )
    app.include_router(router, prefix="/mcp")
    return app


async def _post_initialize(app: FastAPI) -> dict:
    """POST an initialize request and return the parsed capabilities dict."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=_rpc(
                "initialize",
                {"protocolVersion": "2025-06-18", "clientInfo": {}, "capabilities": {}},
            ),
            headers=_MCP_HEADERS,
        )
    assert resp.status_code == 200
    body = resp.json()
    return body["result"]["capabilities"]


def _resource_registry_with_one_resource() -> ResourceRegistry:
    registry = ResourceRegistry()

    @registry.resource("cfg://{key}", name="Config", description="Config entry")
    async def get_cfg(key: str) -> str:
        """Return config value."""
        return f"cfg:{key}"

    return registry


def _prompt_registry_with_one_prompt() -> PromptRegistry:
    registry = PromptRegistry()

    @registry.prompt()
    async def validate_model(project_id: str) -> list[dict]:
        """Validate a project model."""
        return [{"role": "user", "content": f"Validate project {project_id}"}]

    return registry


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tools_capability_advertises_list_changed_with_session_store() -> None:
    """Tool registry + session store: tools.listChanged is true."""
    app = _make_app(with_session_store=True)

    capabilities = await _post_initialize(app)

    assert capabilities["tools"] == {"listChanged": True}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resources_capability_advertises_subscribe_and_list_changed_with_session_store() -> None:
    """Resource registry + session store: resources.subscribe and listChanged both true."""
    app = _make_app(with_session_store=True, resource_registry=_resource_registry_with_one_resource())

    capabilities = await _post_initialize(app)

    assert capabilities["resources"] == {"subscribe": True, "listChanged": True}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_prompts_capability_advertises_list_changed_with_session_store() -> None:
    """Prompt registry + session store: prompts.listChanged is true."""
    app = _make_app(with_session_store=True, prompt_registry=_prompt_registry_with_one_prompt())

    capabilities = await _post_initialize(app)

    assert capabilities["prompts"] == {"listChanged": True}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_logging_capability_present_when_session_store_configured() -> None:
    """Session store present: logging key present and equals {}."""
    app = _make_app(with_session_store=True)

    capabilities = await _post_initialize(app)

    assert capabilities["logging"] == {}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stateless_capabilities_omit_subscribe_list_changed_and_logging() -> None:
    """No session store: no subscribe/listChanged/logging anywhere; tools == {}."""
    app = _make_app(
        with_session_store=False,
        resource_registry=_resource_registry_with_one_resource(),
        prompt_registry=_prompt_registry_with_one_prompt(),
    )

    capabilities = await _post_initialize(app)

    assert capabilities["tools"] == {}
    assert capabilities["resources"] == {}
    assert capabilities["prompts"] == {}
    assert "logging" not in capabilities
    serialized = str(capabilities)
    assert "listChanged" not in serialized
    assert "subscribe" not in serialized


@pytest.mark.integration
@pytest.mark.asyncio
async def test_no_resource_or_prompt_registry_omits_those_keys_entirely() -> None:
    """No resource registry and no prompt registry: resources/prompts keys absent."""
    app = _make_app(with_session_store=True)

    capabilities = await _post_initialize(app)

    assert "resources" not in capabilities
    assert "prompts" not in capabilities
    assert capabilities["tools"] == {"listChanged": True}
    assert capabilities["logging"] == {}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_no_tasks_capability_advertised() -> None:
    """No `tasks` capability key is ever advertised."""
    app = _make_app(
        with_session_store=True,
        resource_registry=_resource_registry_with_one_resource(),
        prompt_registry=_prompt_registry_with_one_prompt(),
    )

    capabilities = await _post_initialize(app)

    assert "tasks" not in capabilities
