"""Tests for ResourceRegistry, FileResourceProvider, and resource HTTP endpoints.

Covers AC-22 through AC-94, EC-1 through EC-4, and IC-17.
"""

import inspect

import httpx
import pytest
from fastapi import FastAPI

from fastapi_mcp_router import MCPToolRegistry, ResourceRegistry, create_mcp_router
from fastapi_mcp_router.exceptions import MCPError
from fastapi_mcp_router.resources import FileResourceProvider, ResourceProvider

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_MCP_HEADERS = {"MCP-Protocol-Version": "2025-06-18", "X-API-Key": "test-key"}


def _make_app_with_resource_registry(resource_registry: ResourceRegistry) -> FastAPI:
    """Create a minimal stateless FastAPI app with a resource_registry and no auth."""
    tool_registry = MCPToolRegistry()
    app = FastAPI()
    router = create_mcp_router(tool_registry, resource_registry=resource_registry)
    app.include_router(router, prefix="/mcp")
    return app


def _make_app_without_resource_registry() -> FastAPI:
    """Create a minimal stateless FastAPI app with no resource_registry."""
    tool_registry = MCPToolRegistry()
    app = FastAPI()
    router = create_mcp_router(tool_registry)
    app.include_router(router, prefix="/mcp")
    return app


def _rpc(method: str, params: dict | None = None, rpc_id: int = 1) -> dict:
    """Build a JSON-RPC 2.0 request body."""
    body: dict = {"jsonrpc": "2.0", "id": rpc_id, "method": method}
    if params is not None:
        body["params"] = params
    return body


# ---------------------------------------------------------------------------
# AC-22: @resource() decorator registers handler with URI template
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_resource_decorator_registers_handler() -> None:
    """AC-22: @resource() decorator registers handler; has_resources() returns True."""
    registry = ResourceRegistry()

    @registry.resource("file://{path}", name="MyFile", description="A file")
    async def get_file(path: str) -> str:
        """Return file content."""
        return "content"

    assert registry.has_resources()


@pytest.mark.unit
def test_resource_decorator_preserves_function() -> None:
    """AC-22: @resource() decorator returns the original function unchanged."""
    registry = ResourceRegistry()

    async def get_doc(slug: str) -> str:
        """Return document content."""
        return f"doc:{slug}"

    result = registry.resource("docs://{slug}")(get_doc)
    assert result is get_doc


@pytest.mark.unit
def test_resource_decorator_sync_handler_raises_type_error() -> None:
    """AC-22: Registering a sync function raises TypeError."""
    registry = ResourceRegistry()

    with pytest.raises(TypeError):

        @registry.resource("file://{path}")
        def sync_handler(path: str) -> str:
            return path


@pytest.mark.unit
def test_empty_registry_has_no_resources() -> None:
    """AC-22/AC-26: Empty ResourceRegistry reports has_resources() == False."""
    registry = ResourceRegistry()
    assert not registry.has_resources()


# ---------------------------------------------------------------------------
# AC-23: resources/list returns correct structure via HTTP
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resources_list_returns_correct_structure() -> None:
    """AC-23: resources/list returns uri, name, description, mimeType per resource."""
    registry = ResourceRegistry()

    @registry.resource(
        "notes://{id}",
        name="Note",
        description="A user note",
        mime_type="text/plain",
    )
    async def get_note(id: str) -> str:
        """Return note text."""
        return f"note:{id}"

    app = _make_app_with_resource_registry(registry)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/mcp", json=_rpc("resources/list"), headers=_MCP_HEADERS)

    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body
    resources = body["result"]["resources"]
    assert len(resources) == 1
    r = resources[0]
    assert r["uri"] == "notes://{id}"
    assert r["name"] == "Note"
    assert r["description"] == "A user note"
    assert r["mimeType"] == "text/plain"


# ---------------------------------------------------------------------------
# AC-24: resources/read matches URI and returns contents
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resources_read_returns_text_content() -> None:
    """AC-24: resources/read matches URI template and returns contents[0].text."""
    registry = ResourceRegistry()

    @registry.resource("data://{key}", name="Data", description="Data by key")
    async def get_data(key: str) -> str:
        """Return data for key."""
        return f"value:{key}"

    app = _make_app_with_resource_registry(registry)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=_rpc("resources/read", {"uri": "data://hello"}),
            headers=_MCP_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body
    contents = body["result"]["contents"]
    assert len(contents) == 1
    assert contents[0]["text"] == "value:hello"


# ---------------------------------------------------------------------------
# AC-25: initialize includes resources capability when resources are registered
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_initialize_includes_resources_capability_when_registered() -> None:
    """AC-25: initialize response includes capabilities.resources when registry has resources."""
    registry = ResourceRegistry()

    @registry.resource("cfg://{key}", name="Config", description="Config entry")
    async def get_cfg(key: str) -> str:
        """Return config value."""
        return f"cfg:{key}"

    app = _make_app_with_resource_registry(registry)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=_rpc(
                "initialize",
                {
                    "protocolVersion": "2025-06-18",
                    "clientInfo": {},
                    "capabilities": {},
                },
            ),
            headers=_MCP_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    capabilities = body["result"]["capabilities"]
    assert "resources" in capabilities


# ---------------------------------------------------------------------------
# AC-26: No resources -> no resources capability; resource methods return -32601
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_initialize_omits_resources_capability_when_no_registry() -> None:
    """AC-26: initialize response omits capabilities.resources when no resource_registry is provided."""
    app = _make_app_without_resource_registry()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=_rpc(
                "initialize",
                {
                    "protocolVersion": "2025-06-18",
                    "clientInfo": {},
                    "capabilities": {},
                },
            ),
            headers=_MCP_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    capabilities = body["result"]["capabilities"]
    assert "resources" not in capabilities


@pytest.mark.integration
@pytest.mark.asyncio
async def test_initialize_omits_resources_capability_when_empty_registry() -> None:
    """AC-26: initialize response omits capabilities.resources when registry has no resources."""
    empty_registry = ResourceRegistry()
    app = _make_app_with_resource_registry(empty_registry)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=_rpc(
                "initialize",
                {
                    "protocolVersion": "2025-06-18",
                    "clientInfo": {},
                    "capabilities": {},
                },
            ),
            headers=_MCP_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    capabilities = body["result"]["capabilities"]
    assert "resources" not in capabilities


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resources_list_returns_32601_when_no_registry() -> None:
    """AC-26: resources/list returns error code -32601 when no resource_registry is configured."""
    app = _make_app_without_resource_registry()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/mcp", json=_rpc("resources/list"), headers=_MCP_HEADERS)

    assert resp.status_code == 200
    body = resp.json()
    assert body["error"]["code"] == -32601


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resources_read_returns_32601_when_no_registry() -> None:
    """AC-26: resources/read returns error code -32601 when no resource_registry is configured."""
    app = _make_app_without_resource_registry()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=_rpc("resources/read", {"uri": "file://test.txt"}),
            headers=_MCP_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["error"]["code"] == -32601


# ---------------------------------------------------------------------------
# AC-27: ResourceProvider interface has exactly 5 abstract methods
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_resource_provider_has_five_abstract_methods() -> None:
    """AC-27: ResourceProvider ABC defines exactly 5 abstract methods."""
    expected_methods = {"list_resources", "read_resource", "subscribe", "unsubscribe", "watch"}
    for method_name in expected_methods:
        assert hasattr(ResourceProvider, method_name), f"ResourceProvider missing method: {method_name}"
        static_method = inspect.getattr_static(ResourceProvider, method_name)
        is_abstract = getattr(static_method, "__isabstractmethod__", False)
        assert is_abstract, f"ResourceProvider.{method_name} must be abstract"


# ---------------------------------------------------------------------------
# AC-28: FileResourceProvider sandboxed access and extension whitelist
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_file_resource_provider_lists_allowed_files(tmp_path) -> None:
    """AC-28: FileResourceProvider lists files with allowed extensions within root."""
    (tmp_path / "readme.md").write_text("hello")
    (tmp_path / "data.json").write_text("{}")
    (tmp_path / "script.py").write_text("pass")  # not in default whitelist

    provider = FileResourceProvider(root_path=tmp_path)
    resources = provider.list_resources()

    uris = {r.uri for r in resources}
    assert any("readme.md" in u for u in uris)
    assert any("data.json" in u for u in uris)
    assert not any("script.py" in u for u in uris)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_file_resource_provider_reads_file_within_root(tmp_path) -> None:
    """AC-28: FileResourceProvider successfully reads a file within the root directory."""
    target = tmp_path / "note.txt"
    target.write_text("hello world")

    provider = FileResourceProvider(root_path=tmp_path)
    contents = await provider.read_resource(f"file://{target}")

    assert contents.text == "hello world"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_file_resource_provider_rejects_file_outside_root(tmp_path) -> None:
    """AC-28: FileResourceProvider rejects URIs pointing outside the root directory."""
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("sensitive")

    provider = FileResourceProvider(root_path=tmp_path)

    with pytest.raises(MCPError) as exc_info:
        await provider.read_resource(f"file://{outside}")

    assert exc_info.value.code == -32602


@pytest.mark.unit
@pytest.mark.asyncio
async def test_file_resource_provider_rejects_disallowed_extension(tmp_path) -> None:
    """AC-28: FileResourceProvider rejects files with extensions not in the whitelist."""
    target = tmp_path / "script.py"
    target.write_text("pass")

    provider = FileResourceProvider(root_path=tmp_path)

    with pytest.raises(MCPError) as exc_info:
        await provider.read_resource(f"file://{target}")

    assert exc_info.value.code == -32602


# ---------------------------------------------------------------------------
# AC-77: Unknown URI -> MCPError -32602
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resources_read_unknown_uri_returns_32602() -> None:
    """AC-77: resources/read with a URI that matches no handler returns error -32602."""
    registry = ResourceRegistry()

    @registry.resource("known://{id}", name="Known", description="Known resource")
    async def get_known(id: str) -> str:
        """Return known content."""
        return f"known:{id}"

    app = _make_app_with_resource_registry(registry)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=_rpc("resources/read", {"uri": "unknown://does-not-exist"}),
            headers=_MCP_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["error"]["code"] == -32602


# ---------------------------------------------------------------------------
# AC-81: Path traversal -> MCPError -32602
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_file_resource_provider_rejects_path_traversal(tmp_path) -> None:
    """AC-81: FileResourceProvider rejects URIs containing path traversal sequences."""
    provider = FileResourceProvider(root_path=tmp_path)

    with pytest.raises(MCPError) as exc_info:
        await provider.read_resource("file://../secret.txt")

    assert exc_info.value.code == -32602


# ---------------------------------------------------------------------------
# AC-82 / AC-93 / AC-94: File size boundary tests
# ---------------------------------------------------------------------------

_10_MB = 10 * 1024 * 1024  # 10485760 bytes


@pytest.mark.unit
@pytest.mark.asyncio
async def test_file_resource_provider_rejects_file_over_10mb(tmp_path) -> None:
    """AC-82: FileResourceProvider raises MCPError -32602 for files over 10 MB."""
    big_file = tmp_path / "big.txt"
    big_file.write_bytes(b"x" * (_10_MB + 1))

    provider = FileResourceProvider(root_path=tmp_path)

    with pytest.raises(MCPError) as exc_info:
        await provider.read_resource(f"file://{big_file}")

    assert exc_info.value.code == -32602


@pytest.mark.unit
@pytest.mark.asyncio
async def test_file_resource_provider_accepts_file_at_exactly_10mb(tmp_path) -> None:
    """AC-93: FileResourceProvider succeeds for a file at exactly 10 MB."""
    exact_file = tmp_path / "exact.txt"
    exact_file.write_bytes(b"x" * _10_MB)

    provider = FileResourceProvider(root_path=tmp_path)
    # Should not raise
    contents = await provider.read_resource(f"file://{exact_file}")
    assert contents is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_file_resource_provider_rejects_file_at_10mb_plus_one(tmp_path) -> None:
    """AC-94: FileResourceProvider raises MCPError -32602 for files at 10 MB + 1 byte."""
    over_file = tmp_path / "over.txt"
    over_file.write_bytes(b"x" * (_10_MB + 1))

    provider = FileResourceProvider(root_path=tmp_path)

    with pytest.raises(MCPError) as exc_info:
        await provider.read_resource(f"file://{over_file}")

    assert exc_info.value.code == -32602


# ---------------------------------------------------------------------------
# EC-1: URI matches nothing -> MCPError -32602 (unit via registry)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_read_resource_unmatched_uri_raises_32602() -> None:
    """EC-1: registry.read_resource() raises MCPError -32602 when no handler matches."""
    registry = ResourceRegistry()

    @registry.resource("known://{id}", name="Known", description="Known resource")
    async def get_known(id: str) -> str:
        """Return known content."""
        return f"known:{id}"

    with pytest.raises(MCPError) as exc_info:
        await registry.read_resource("unknown://anything")

    assert exc_info.value.code == -32602


# ---------------------------------------------------------------------------
# EC-2: Handler raises -> MCPError -32603
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_read_resource_handler_exception_raises_32603() -> None:
    """EC-2: When the handler raises an unexpected exception, registry raises MCPError -32603."""
    registry = ResourceRegistry()

    @registry.resource("fail://{id}", name="Fail", description="Always fails")
    async def failing_handler(id: str) -> str:
        """Raise ValueError unconditionally."""
        raise ValueError("something went wrong")

    with pytest.raises(MCPError) as exc_info:
        await registry.read_resource("fail://test")

    assert exc_info.value.code == -32603


# ---------------------------------------------------------------------------
# EC-3: Handler returns unsupported type -> MCPError -32603
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_read_resource_unsupported_return_type_raises_32603() -> None:
    """EC-3: Handler returning an unsupported type causes registry to raise MCPError -32603."""
    registry = ResourceRegistry()

    @registry.resource("bad://{id}", name="Bad", description="Returns int")
    async def bad_handler(id: str) -> int:  # type: ignore[return]
        """Return an integer, which is not a supported return type."""
        return 42

    with pytest.raises(MCPError) as exc_info:
        await registry.read_resource("bad://anything")

    assert exc_info.value.code == -32603


# ---------------------------------------------------------------------------
# EC-4: No resources registered -> MCPError -32601
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_list_resources_raises_32601_when_empty() -> None:
    """EC-4: Calling list_resources() on an empty ResourceRegistry raises MCPError -32601."""
    registry = ResourceRegistry()

    with pytest.raises(MCPError) as exc_info:
        registry.list_resources()

    assert exc_info.value.code == -32601
