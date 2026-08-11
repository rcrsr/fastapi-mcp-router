"""Tests for ResourceRegistry, FileResourceProvider, and resource HTTP endpoints."""

import inspect

import httpx
import pytest
from fastapi import Depends, FastAPI

from fastapi_mcp_router import MCPToolRegistry, ResourceRegistry, create_mcp_router
from fastapi_mcp_router.exceptions import MCPError
from fastapi_mcp_router.resources import (
    FileResourceProvider,
    Resource,
    ResourceContents,
    ResourceProvider,
    ResourceTemplate,
)
from fastapi_mcp_router.router import _shape_resource, _shape_resource_template
from fastapi_mcp_router.types import Icon

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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resources_list_omits_mime_type_when_unset() -> None:
    """A resource registered without mime_type omits the mimeType key on the wire."""
    registry = ResourceRegistry()

    @registry.resource(
        "notes://{id}",
        name="Note",
        description="A user note",
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
    resources = body["result"]["resources"]
    assert len(resources) == 1
    assert "mimeType" not in resources[0]


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
# AC-27: ResourceProvider interface has exactly 4 abstract methods; watch is optional
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_resource_provider_has_four_abstract_methods() -> None:
    """AC-27: ResourceProvider ABC defines exactly 4 abstract methods; watch is optional."""
    required_abstract = {"list_resources", "read_resource", "subscribe", "unsubscribe"}
    for method_name in required_abstract:
        assert hasattr(ResourceProvider, method_name), f"ResourceProvider missing method: {method_name}"
        static_method = inspect.getattr_static(ResourceProvider, method_name)
        is_abstract = getattr(static_method, "__isabstractmethod__", False)
        assert is_abstract, f"ResourceProvider.{method_name} must be abstract"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_resource_provider_watch_default_is_empty_async_generator() -> None:
    """AC-27: ResourceProvider.watch() default implementation yields nothing."""

    class MinimalProvider(ResourceProvider):
        def list_resources(self) -> list:
            return []

        async def read_resource(self, uri: str) -> ResourceContents:
            return ResourceContents(uri=uri, text="")

        def subscribe(self, uri: str) -> bool:
            return False

        def unsubscribe(self, uri: str) -> bool:
            return False

    provider = MinimalProvider()
    results = [r async for r in provider.watch()]
    assert results == []


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


# ---------------------------------------------------------------------------
# AC-5: Resource handler with Depends() -> dependency injected, schema excludes param
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_resource_depends_dependency_is_injected() -> None:
    """AC-5: Depends() param is resolved and injected into the handler at call time."""
    registry = ResourceRegistry()

    def get_db() -> str:
        """Return a fake database connection string."""
        return "db://test"

    @registry.resource("item://{id}", name="Item", description="Item by id")
    async def get_item(id: str, db: str = Depends(get_db)) -> str:
        """Return item using injected db connection."""
        return f"{db}:{id}"

    contents = await registry.read_resource("item://42")

    assert contents.text == "db://test:42"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_resource_depends_async_dependency_is_injected() -> None:
    """AC-5: Async Depends() is awaited and injected into the handler."""
    registry = ResourceRegistry()

    async def get_service() -> str:
        """Async dependency returning a service name."""
        return "svc://live"

    @registry.resource("svc://{name}", name="Service", description="Service by name")
    async def get_svc(name: str, service: str = Depends(get_service)) -> str:
        """Return service info."""
        return f"{service}/{name}"

    contents = await registry.read_resource("svc://payments")

    assert contents.text == "svc://live/payments"


@pytest.mark.unit
def test_resource_depends_param_excluded_from_list_resources_metadata() -> None:
    """AC-5: list_resources() metadata does not expose the Depends() parameter name."""
    registry = ResourceRegistry()

    def get_db() -> str:
        """Return a fake database handle."""
        return "db://test"

    @registry.resource("order://{id}", name="Order", description="Order by id")
    async def get_order(id: str, db: str = Depends(get_db)) -> str:
        """Return order content."""
        return f"order:{id}"

    resources = registry.list_resources()

    assert len(resources) == 1
    r = resources[0]
    # Resource metadata only contains protocol fields — no db param leaks through
    assert r.uri == "order://{id}"
    assert r.name == "Order"
    assert r.description == "Order by id"
    assert not hasattr(r, "db")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resource_depends_injected_via_http_endpoint() -> None:
    """AC-5: Depends() dependency is resolved when resource is read through the HTTP endpoint."""
    registry = ResourceRegistry()

    def get_tenant() -> str:
        """Return a fixed tenant identifier."""
        return "tenant-abc"

    @registry.resource("tenant://{key}", name="Tenant", description="Tenant value")
    async def get_tenant_value(key: str, tenant: str = Depends(get_tenant)) -> str:
        """Return tenant-scoped value."""
        return f"{tenant}:{key}"

    app = _make_app_with_resource_registry(registry)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=_rpc("resources/read", {"uri": "tenant://config"}),
            headers=_MCP_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body
    contents = body["result"]["contents"]
    assert contents[0]["text"] == "tenant-abc:config"


# ---------------------------------------------------------------------------
# AC-6: Resource handler without Depends() -> existing behavior unchanged
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_resource_without_depends_behavior_unchanged() -> None:
    """AC-6: Handler with no Depends() parameters still works correctly."""
    registry = ResourceRegistry()

    @registry.resource("plain://{slug}", name="Plain", description="Plain resource")
    async def get_plain(slug: str) -> str:
        """Return plain content."""
        return f"plain:{slug}"

    contents = await registry.read_resource("plain://hello")

    assert contents.text == "plain:hello"


@pytest.mark.unit
def test_resource_without_depends_list_resources_unchanged() -> None:
    """AC-6: list_resources() returns correct metadata for handlers without Depends()."""
    registry = ResourceRegistry()

    @registry.resource("doc://{id}", name="Doc", description="A document", mime_type="text/markdown")
    async def get_doc(id: str) -> str:
        """Return document content."""
        return f"doc:{id}"

    resources = registry.list_resources()

    assert len(resources) == 1
    assert resources[0].uri == "doc://{id}"
    assert resources[0].name == "Doc"
    assert resources[0].mime_type == "text/markdown"


# ---------------------------------------------------------------------------
# EC-3: Dependency raises exception -> error propagates to JSON-RPC response
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_resource_depends_exception_propagates_from_registry() -> None:
    """EC-3: When a Depends() dependency raises, the exception propagates from read_resource."""
    registry = ResourceRegistry()

    def failing_dep() -> str:
        """Dependency that always fails."""
        raise RuntimeError("dependency unavailable")

    @registry.resource("dep://{id}", name="Dep", description="Dep resource")
    async def get_dep(id: str, conn: str = Depends(failing_dep)) -> str:
        """Return dep content."""
        return f"dep:{id}"

    with pytest.raises(RuntimeError, match="dependency unavailable"):
        await registry.read_resource("dep://test")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resource_depends_exception_returns_error_via_http() -> None:
    """EC-3: Dependency exception during resources/read returns JSON-RPC error -32603."""
    registry = ResourceRegistry()

    def broken_dep() -> str:
        """Dependency that raises unconditionally."""
        raise ValueError("service unreachable")

    @registry.resource("broken://{id}", name="Broken", description="Broken resource")
    async def get_broken(id: str, svc: str = Depends(broken_dep)) -> str:
        """Return broken content."""
        return f"broken:{id}"

    app = _make_app_with_resource_registry(registry)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json=_rpc("resources/read", {"uri": "broken://anything"}),
            headers=_MCP_HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "error" in body
    assert body["error"]["code"] == -32603


# ---------------------------------------------------------------------------
# resources/templates/list dispatch branch
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resources_templates_list_returns_correct_structure() -> None:
    """With registered templates, resources/templates/list returns resourceTemplates."""
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
        resp = await client.post("/mcp", json=_rpc("resources/templates/list"), headers=_MCP_HEADERS)

    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body
    templates = body["result"]["resourceTemplates"]
    assert len(templates) == 1
    t = templates[0]
    assert t["uriTemplate"] == "notes://{id}"
    assert t["name"] == "Note"
    assert t["description"] == "A user note"
    assert t["mimeType"] == "text/plain"
    assert "nextCursor" not in body["result"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resources_templates_list_returns_32601_when_no_registry() -> None:
    """No resource registry configured returns -32601 naming resources/templates/list."""
    app = _make_app_without_resource_registry()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/mcp", json=_rpc("resources/templates/list"), headers=_MCP_HEADERS)

    assert resp.status_code == 200
    body = resp.json()
    assert "error" in body
    assert body["error"]["code"] == -32601
    assert "resources/templates/list" in body["error"]["message"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resources_templates_list_paginates_beyond_default_page_size() -> None:
    """More than 100 templates paginate per the shared list-method default."""
    registry = ResourceRegistry()

    def _register(index: int) -> None:
        @registry.resource(f"item://{index:03d}", name=f"item_{index:03d}")
        async def handler() -> str:
            """Test resource handler."""
            return "content"

    for i in range(105):
        _register(i)

    app = _make_app_with_resource_registry(registry)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.post("/mcp", json=_rpc("resources/templates/list"), headers=_MCP_HEADERS)
        first_result = first.json()["result"]
        assert len(first_result["resourceTemplates"]) == 100
        next_cursor = first_result["nextCursor"]

        second = await client.post(
            "/mcp",
            json=_rpc("resources/templates/list", {"cursor": next_cursor}),
            headers=_MCP_HEADERS,
        )
        second_result = second.json()["result"]
        assert len(second_result["resourceTemplates"]) == 5
        assert "nextCursor" not in second_result


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resources_templates_list_omits_mime_type_when_unset() -> None:
    """A template registered without mime_type omits mimeType instead of emitting null."""
    registry = ResourceRegistry()

    @registry.resource("plain://{id}", name="Plain", description="No mime type")
    async def get_plain(id: str) -> str:
        """Return plain content."""
        return f"plain:{id}"

    app = _make_app_with_resource_registry(registry)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        templates_resp = await client.post("/mcp", json=_rpc("resources/templates/list"), headers=_MCP_HEADERS)
        resources_resp = await client.post("/mcp", json=_rpc("resources/list"), headers=_MCP_HEADERS)

    templates_template = templates_resp.json()["result"]["resourceTemplates"][0]
    resources_template = resources_resp.json()["result"]["resourceTemplates"][0]

    assert "mimeType" not in templates_template
    assert "mimeType" not in resources_template


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resources_list_and_templates_list_emit_identical_template_shape() -> None:
    """resources/list's inline bundle and resources/templates/list share one shaper."""
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
        templates_resp = await client.post("/mcp", json=_rpc("resources/templates/list"), headers=_MCP_HEADERS)
        resources_resp = await client.post("/mcp", json=_rpc("resources/list"), headers=_MCP_HEADERS)

    templates_shape = templates_resp.json()["result"]["resourceTemplates"][0]
    resources_shape = resources_resp.json()["result"]["resourceTemplates"][0]
    assert templates_shape == resources_shape


# ---------------------------------------------------------------------------
# Icons: version-gated emission on resources/list and resources/templates/list
# ---------------------------------------------------------------------------


def _register_resource_with_icon(registry: ResourceRegistry) -> None:
    @registry.resource(
        "icon://{id}",
        name="IconResource",
        description="A resource with an icon",
        icons=[{"src": "https://example.com/icon.png"}],
    )
    async def get_icon_resource(id: str) -> str:
        """Return content for an icon-bearing resource."""
        return f"content:{id}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resources_list_emits_icons_when_negotiated_version_supports_them() -> None:
    """Icons appear in resources/list (both resources and resourceTemplates) at 2025-11-25."""
    registry = ResourceRegistry()
    _register_resource_with_icon(registry)

    app = _make_app_with_resource_registry(registry)
    transport = httpx.ASGITransport(app=app)
    headers = {**_MCP_HEADERS, "MCP-Protocol-Version": "2025-11-25"}
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/mcp", json=_rpc("resources/list"), headers=headers)

    result = resp.json()["result"]
    assert result["resources"][0]["icons"] == [{"src": "https://example.com/icon.png"}]
    assert result["resourceTemplates"][0]["icons"] == [{"src": "https://example.com/icon.png"}]


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("protocol_version", ["2025-06-18", "2025-03-26"])
async def test_resources_list_omits_icons_below_negotiated_version(protocol_version: str) -> None:
    """Icons are omitted from resources/list when negotiated version is below 2025-11-25."""
    registry = ResourceRegistry()
    _register_resource_with_icon(registry)

    app = _make_app_with_resource_registry(registry)
    transport = httpx.ASGITransport(app=app)
    headers = {**_MCP_HEADERS, "MCP-Protocol-Version": protocol_version}
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/mcp", json=_rpc("resources/list"), headers=headers)

    result = resp.json()["result"]
    assert "icons" not in result["resources"][0]
    assert "icons" not in result["resourceTemplates"][0]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resources_templates_list_emits_icons_when_negotiated_version_supports_them() -> None:
    """Icons appear in resources/templates/list at 2025-11-25."""
    registry = ResourceRegistry()
    _register_resource_with_icon(registry)

    app = _make_app_with_resource_registry(registry)
    transport = httpx.ASGITransport(app=app)
    headers = {**_MCP_HEADERS, "MCP-Protocol-Version": "2025-11-25"}
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/mcp", json=_rpc("resources/templates/list"), headers=headers)

    result = resp.json()["result"]
    assert result["resourceTemplates"][0]["icons"] == [{"src": "https://example.com/icon.png"}]


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("protocol_version", ["2025-06-18", "2025-03-26"])
async def test_resources_templates_list_omits_icons_below_negotiated_version(protocol_version: str) -> None:
    """Icons are omitted from resources/templates/list when negotiated version is below 2025-11-25."""
    registry = ResourceRegistry()
    _register_resource_with_icon(registry)

    app = _make_app_with_resource_registry(registry)
    transport = httpx.ASGITransport(app=app)
    headers = {**_MCP_HEADERS, "MCP-Protocol-Version": protocol_version}
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/mcp", json=_rpc("resources/templates/list"), headers=headers)

    result = resp.json()["result"]
    assert "icons" not in result["resourceTemplates"][0]


@pytest.mark.unit
def test_shape_resource_omits_icons_when_protocol_version_is_none() -> None:
    """_shape_resource() omits icons when called with the default protocol_version=None.

    Mirrors MCPToolRegistry.list_tools(protocol_version=None) unit-level
    coverage for tools/list: exercises the gating helper directly rather
    than only through the HTTP layer.
    """
    resource = Resource(
        uri="icon://direct",
        name="IconResource",
        description="A resource with an icon",
        icons=[Icon(src="https://example.com/icon.png")],
    )

    shaped = _shape_resource(resource)

    assert "icons" not in shaped


@pytest.mark.unit
def test_shape_resource_template_omits_icons_when_protocol_version_is_none() -> None:
    """_shape_resource_template() omits icons when called with the default protocol_version=None."""
    template = ResourceTemplate(
        uri_template="icon://{id}",
        name="IconResource",
        description="A resource template with an icon",
        icons=[Icon(src="https://example.com/icon.png")],
    )

    shaped = _shape_resource_template(template)

    assert "icons" not in shaped


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resources_list_omits_icons_when_no_protocol_version_header_sent() -> None:
    """resources/list omits icons when the MCP-Protocol-Version header is absent entirely.

    Distinct from test_resources_list_omits_icons_below_negotiated_version,
    which always sends an explicit (pre-2025-11-25) header value; this
    exercises the default-negotiation path used when a client sends no
    header at all.
    """
    registry = ResourceRegistry()
    _register_resource_with_icon(registry)

    app = _make_app_with_resource_registry(registry)
    transport = httpx.ASGITransport(app=app)
    headers = {k: v for k, v in _MCP_HEADERS.items() if k != "MCP-Protocol-Version"}
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/mcp", json=_rpc("resources/list"), headers=headers)

    result = resp.json()["result"]
    assert "icons" not in result["resources"][0]
    assert "icons" not in result["resourceTemplates"][0]


@pytest.mark.unit
def test_resource_icon_with_disallowed_scheme_rejected_at_registration() -> None:
    """@registry.resource(icons=...) rejects a non-HTTPS/data: icon src fail-fast.

    Proves Icon.model_validate() actually runs on the resource-icon surface,
    not just on tool icons.
    """
    registry = ResourceRegistry()

    with pytest.raises(Exception, match="not allowed"):

        @registry.resource(
            "bad-icon://{id}",
            name="BadIconResource",
            icons=[{"src": "http://example.com/icon.png"}],
        )
        async def get_bad_icon_resource(id: str) -> str:
            """Resource with a disallowed icon scheme."""
            return f"content:{id}"
