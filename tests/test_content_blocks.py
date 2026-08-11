"""
Unit and integration tests for the content-block builder in fastapi_mcp_router.router.

Unit tests cover build_content_block: wire shapes for text/image/audio/resource_link
(emitted regardless of negotiated protocol version, since none of these types
are 2025-11-25-only additions), and invalid data-scheme rejection. Also
cover handle_tools_call routing content-block tool results through the
builder with the negotiated version, including mixed-content ordering.

Integration tests cover the same behavior through a real HTTP round trip
(httpx.AsyncClient against a mounted FastAPI app), per the unit/integration
split pattern established by the removed tests/test_roots.py.
"""

from typing import cast

import httpx
import pytest
from fastapi import BackgroundTasks, FastAPI, Request

from fastapi_mcp_router import create_mcp_router
from fastapi_mcp_router.exceptions import MCPError
from fastapi_mcp_router.registry import MCPToolRegistry
from fastapi_mcp_router.router import build_content_block, handle_tools_call
from fastapi_mcp_router.types import (
    AudioContent,
    Icon,
    ImageContent,
    ResourceLinkContent,
    TextContent,
)


@pytest.mark.unit
def test_build_content_block_text_flat_shape():
    # Arrange
    block = TextContent(text="hello")

    # Act
    result = build_content_block(block, "2025-06-18")

    # Assert
    assert result == {"type": "text", "text": "hello"}


@pytest.mark.unit
def test_build_content_block_image_flat_shape():
    # Arrange
    block = ImageContent(data="aGVsbG8=", mimeType="image/png")

    # Act
    result = build_content_block(block, "2025-11-25")

    # Assert
    assert result == {"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"}


@pytest.mark.unit
def test_build_content_block_audio_flat_shape():
    # Arrange
    block = AudioContent(data="ZmFrZQ==", mimeType="audio/wav")

    # Act
    result = build_content_block(block, "2025-11-25")

    # Assert
    assert result == {"type": "audio", "data": "ZmFrZQ==", "mimeType": "audio/wav"}


@pytest.mark.unit
def test_build_content_block_resource_link_minimum_shape_omits_unset_fields():
    # Arrange
    block = ResourceLinkContent(uri="file:///tmp/report.csv", name="report.csv")

    # Act
    result = build_content_block(block, "2025-11-25")

    # Assert
    assert result == {"type": "resource_link", "uri": "file:///tmp/report.csv", "name": "report.csv"}


@pytest.mark.unit
def test_build_content_block_resource_link_includes_set_optional_fields():
    # Arrange
    block = ResourceLinkContent(uri="file:///tmp/report.csv", name="report.csv", mimeType="text/csv")

    # Act
    result = build_content_block(block, "2025-11-25")

    # Assert
    assert result is not None
    assert result["mimeType"] == "text/csv"


@pytest.mark.unit
def test_build_content_block_resource_link_full_shape_with_all_optional_fields_set():
    # Arrange
    block = ResourceLinkContent(
        uri="file:///tmp/report.csv",
        name="report.csv",
        title="Quarterly Report",
        description="A CSV export of quarterly figures",
        mimeType="text/csv",
        icons=[Icon(src="https://example.com/icon.png")],
        size=2048,
    )

    # Act
    result = build_content_block(block, "2025-11-25")

    # Assert
    assert result == {
        "type": "resource_link",
        "uri": "file:///tmp/report.csv",
        "name": "report.csv",
        "title": "Quarterly Report",
        "description": "A CSV export of quarterly figures",
        "mimeType": "text/csv",
        "icons": [{"src": "https://example.com/icon.png"}],
        "size": 2048,
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    "block",
    [
        ImageContent(data="aGVsbG8=", mimeType="image/png"),
        AudioContent(data="ZmFrZQ==", mimeType="audio/wav"),
        ResourceLinkContent(uri="file:///tmp/report.csv", name="report.csv"),
    ],
)
@pytest.mark.parametrize("negotiated_version", ["2025-06-18", "2025-03-26"])
def test_build_content_block_not_gated_below_2025_11_25(block, negotiated_version):
    # Act
    result = build_content_block(block, negotiated_version)

    # Assert
    assert result is not None
    assert result["type"] == block.type


@pytest.mark.unit
def test_build_content_block_text_never_gated_below_2025_11_25():
    # Arrange
    block = TextContent(text="hello")

    # Act
    result = build_content_block(block, "2025-03-26")

    # Assert
    assert result == {"type": "text", "text": "hello"}


@pytest.mark.unit
def test_build_content_block_image_invalid_data_scheme_raises_mcp_error_naming_scheme():
    # Arrange
    block = ImageContent(data="https://example.com/evil.png", mimeType="image/png")

    # Act / Assert
    with pytest.raises(MCPError) as exc_info:
        build_content_block(block, "2025-11-25")

    assert exc_info.value.code == -32602
    assert "https" in exc_info.value.message


@pytest.mark.unit
def test_build_content_block_audio_invalid_data_scheme_raises_mcp_error_naming_scheme():
    # Arrange
    block = AudioContent(data="ftp://example.com/evil.wav", mimeType="audio/wav")

    # Act / Assert
    with pytest.raises(MCPError) as exc_info:
        build_content_block(block, "2025-11-25")

    assert exc_info.value.code == -32602
    assert "ftp" in exc_info.value.message


@pytest.mark.unit
def test_build_content_block_audio_data_uri_scheme_raises_mcp_error():
    # Arrange
    block = AudioContent(data="data:audio/wav;base64,ZmFrZQ==", mimeType="audio/wav")

    # Act / Assert
    with pytest.raises(MCPError) as exc_info:
        build_content_block(block, "2025-11-25")

    assert exc_info.value.code == -32602


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_tools_call_mixed_content_all_blocks_present():
    # Arrange
    registry = MCPToolRegistry()

    @registry.tool()
    async def mixed_content_tool() -> list[TextContent | ImageContent]:
        return [TextContent(text="caption"), ImageContent(data="aGVsbG8=", mimeType="image/png")]

    # Act
    result = await handle_tools_call(
        registry,
        {"name": "mixed_content_tool", "arguments": {}},
        request=cast(Request, None),
        background_tasks=cast(BackgroundTasks, None),
        negotiated_version="2025-11-25",
    )

    # Assert
    assert result["content"] == [
        {"type": "text", "text": "caption"},
        {"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"},
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_tools_call_all_content_block_types_present_and_in_order():
    # Arrange
    registry = MCPToolRegistry()

    @registry.tool()
    async def all_types_tool() -> list[TextContent | ImageContent | AudioContent | ResourceLinkContent]:
        return [
            TextContent(text="caption"),
            ImageContent(data="aGVsbG8=", mimeType="image/png"),
            AudioContent(data="ZmFrZQ==", mimeType="audio/wav"),
            ResourceLinkContent(uri="file:///tmp/report.csv", name="report.csv"),
        ]

    # Act
    result = await handle_tools_call(
        registry,
        {"name": "all_types_tool", "arguments": {}},
        request=cast(Request, None),
        background_tasks=cast(BackgroundTasks, None),
        negotiated_version="2025-11-25",
    )

    # Assert
    assert result["content"] == [
        {"type": "text", "text": "caption"},
        {"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"},
        {"type": "audio", "data": "ZmFrZQ==", "mimeType": "audio/wav"},
        {"type": "resource_link", "uri": "file:///tmp/report.csv", "name": "report.csv"},
    ]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("negotiated_version", ["2025-06-18", "2025-03-26"])
async def test_handle_tools_call_emits_image_block_below_2025_11_25(
    negotiated_version,
):
    # Arrange
    registry = MCPToolRegistry()

    @registry.tool()
    async def mixed_content_tool() -> list[TextContent | ImageContent]:
        return [TextContent(text="caption"), ImageContent(data="aGVsbG8=", mimeType="image/png")]

    # Act
    result = await handle_tools_call(
        registry,
        {"name": "mixed_content_tool", "arguments": {}},
        request=cast(Request, None),
        background_tasks=cast(BackgroundTasks, None),
        negotiated_version=negotiated_version,
    )

    # Assert
    assert result["content"] == [
        {"type": "text", "text": "caption"},
        {"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"},
    ]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("negotiated_version", ["2025-06-18", "2025-03-26"])
async def test_handle_tools_call_emits_all_content_types_below_2025_11_25(negotiated_version):
    """Image, audio, and resource_link blocks are all emitted for pre-2025-11-25 clients, no omission."""
    # Arrange
    registry = MCPToolRegistry()

    @registry.tool()
    async def all_types_tool() -> list[TextContent | ImageContent | AudioContent | ResourceLinkContent]:
        return [
            TextContent(text="caption"),
            ImageContent(data="aGVsbG8=", mimeType="image/png"),
            AudioContent(data="ZmFrZQ==", mimeType="audio/wav"),
            ResourceLinkContent(uri="file:///tmp/report.csv", name="report.csv"),
        ]

    # Act
    result = await handle_tools_call(
        registry,
        {"name": "all_types_tool", "arguments": {}},
        request=cast(Request, None),
        background_tasks=cast(BackgroundTasks, None),
        negotiated_version=negotiated_version,
    )

    # Assert
    assert result["content"] == [
        {"type": "text", "text": "caption"},
        {"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"},
        {"type": "audio", "data": "ZmFrZQ==", "mimeType": "audio/wav"},
        {"type": "resource_link", "uri": "file:///tmp/report.csv", "name": "report.csv"},
    ]
    assert "isError" not in result or result["isError"] is False


@pytest.mark.unit
def test_build_content_block_accepts_dict_input():
    # Arrange
    block: dict[str, object] = {"type": "text", "text": "hi"}

    # Act
    result = build_content_block(block, "2025-06-18")

    # Assert
    assert result == {"type": "text", "text": "hi"}


@pytest.mark.unit
def test_build_content_block_unknown_type_raises_mcp_error():
    # Arrange
    block: dict[str, object] = {"type": "video", "data": "x"}

    # Act / Assert
    with pytest.raises(MCPError) as exc_info:
        build_content_block(block, "2025-11-25")

    assert exc_info.value.code == -32602


# ---------------------------------------------------------------------------
# Integration tests: HTTP round trip through a mounted FastAPI app
# ---------------------------------------------------------------------------


def _rpc_call_body(name: str) -> dict[str, object]:
    """Build a JSON-RPC tools/call request body for the given tool name.

    Args:
        name: Registered tool name to invoke.

    Returns:
        JSON-RPC 2.0 request body dict.
    """
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": {}},
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tools_call_image_content_over_http_flat_shape():
    """Integration: tools/call over HTTP returns a flat image block for a 2025-11-25 client."""
    # Arrange
    registry = MCPToolRegistry()

    @registry.tool()
    async def photo_tool() -> list[ImageContent]:
        return [ImageContent(data="aGVsbG8=", mimeType="image/png")]

    router = create_mcp_router(registry)
    app = FastAPI()
    app.include_router(router, prefix="/mcp")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"X-API-Key": "test-key", "MCP-Protocol-Version": "2025-11-25"},
    ) as client:
        # Act
        response = await client.post("/mcp", json=_rpc_call_body("photo_tool"))

    # Assert
    body = response.json()
    assert body["result"]["content"] == [{"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"}]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tools_call_audio_content_over_http_flat_shape():
    """Integration: tools/call over HTTP returns a flat audio block for a 2025-11-25 client."""
    # Arrange
    registry = MCPToolRegistry()

    @registry.tool()
    async def voice_memo_tool() -> list[AudioContent]:
        return [AudioContent(data="ZmFrZQ==", mimeType="audio/wav")]

    router = create_mcp_router(registry)
    app = FastAPI()
    app.include_router(router, prefix="/mcp")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"X-API-Key": "test-key", "MCP-Protocol-Version": "2025-11-25"},
    ) as client:
        # Act
        response = await client.post("/mcp", json=_rpc_call_body("voice_memo_tool"))

    # Assert
    body = response.json()
    assert body["result"]["content"] == [{"type": "audio", "data": "ZmFrZQ==", "mimeType": "audio/wav"}]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tools_call_resource_link_content_over_http_flat_shape():
    """Integration: tools/call over HTTP returns a flat resource_link block for a 2025-11-25 client."""
    # Arrange
    registry = MCPToolRegistry()

    @registry.tool()
    async def export_tool() -> list[ResourceLinkContent]:
        return [ResourceLinkContent(uri="file:///tmp/report.csv", name="report.csv")]

    router = create_mcp_router(registry)
    app = FastAPI()
    app.include_router(router, prefix="/mcp")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"X-API-Key": "test-key", "MCP-Protocol-Version": "2025-11-25"},
    ) as client:
        # Act
        response = await client.post("/mcp", json=_rpc_call_body("export_tool"))

    # Assert
    body = response.json()
    assert body["result"]["content"] == [
        {"type": "resource_link", "uri": "file:///tmp/report.csv", "name": "report.csv"}
    ]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tools_call_resource_link_content_with_icon_over_http_flat_shape():
    """Integration: a typed Icon instance attached to ResourceLinkContent.icons round-trips over HTTP."""
    # Arrange
    registry = MCPToolRegistry()

    @registry.tool()
    async def export_tool_with_icon() -> list[ResourceLinkContent]:
        return [
            ResourceLinkContent(
                uri="file:///tmp/report.csv",
                name="report.csv",
                icons=[Icon(src="https://example.com/icon.png")],
            )
        ]

    router = create_mcp_router(registry)
    app = FastAPI()
    app.include_router(router, prefix="/mcp")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"X-API-Key": "test-key", "MCP-Protocol-Version": "2025-11-25"},
    ) as client:
        # Act
        response = await client.post("/mcp", json=_rpc_call_body("export_tool_with_icon"))

    # Assert
    body = response.json()
    assert body["result"]["content"] == [
        {
            "type": "resource_link",
            "uri": "file:///tmp/report.csv",
            "name": "report.csv",
            "icons": [{"src": "https://example.com/icon.png"}],
        }
    ]


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("protocol_version", ["2025-06-18", "2025-03-26"])
async def test_tools_call_all_content_blocks_emitted_over_http_for_older_clients(protocol_version: str):
    """Integration: image/audio/resource_link blocks are emitted (not omitted) for older clients."""
    # Arrange
    registry = MCPToolRegistry()

    @registry.tool()
    async def all_types_tool() -> list[TextContent | ImageContent | AudioContent | ResourceLinkContent]:
        return [
            TextContent(text="caption"),
            ImageContent(data="aGVsbG8=", mimeType="image/png"),
            AudioContent(data="ZmFrZQ==", mimeType="audio/wav"),
            ResourceLinkContent(uri="file:///tmp/report.csv", name="report.csv"),
        ]

    router = create_mcp_router(registry)
    app = FastAPI()
    app.include_router(router, prefix="/mcp")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"X-API-Key": "test-key", "MCP-Protocol-Version": protocol_version},
    ) as client:
        # Act
        response = await client.post("/mcp", json=_rpc_call_body("all_types_tool"))

    # Assert
    body = response.json()
    assert "error" not in body
    assert body["result"]["content"] == [
        {"type": "text", "text": "caption"},
        {"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"},
        {"type": "audio", "data": "ZmFrZQ==", "mimeType": "audio/wav"},
        {"type": "resource_link", "uri": "file:///tmp/report.csv", "name": "report.csv"},
    ]
    assert body["result"].get("isError", False) is False
