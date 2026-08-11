"""
Unit tests for public API models in fastapi_mcp_router.types.

Covers TextContent, ToolResponse, ServerIcon, ServerInfo, McpSessionData,
CompletionRef, CompletionArgument, CompletionResult, ElicitationRequest,
and ElicitationResponse.
"""

import base64
from datetime import datetime
from typing import cast
from uuid import UUID

import pytest
from pydantic import ValidationError

from fastapi_mcp_router import ServerIcon, ServerInfo, TextContent, ToolResponse
from fastapi_mcp_router.exceptions import MCPError
from fastapi_mcp_router.router import handle_initialize
from fastapi_mcp_router.types import (
    AudioContent,
    CompletionArgument,
    CompletionRef,
    CompletionResult,
    ElicitationRequest,
    ElicitationResponse,
    Icon,
    ImageContent,
    McpSessionData,
    ResourceLinkContent,
    ToolAnnotations,
)


@pytest.mark.unit
def test_text_content_default_type_is_text() -> None:
    content = TextContent(text="hello")
    assert content.type == "text"


@pytest.mark.unit
def test_text_content_requires_text_field() -> None:
    with pytest.raises(ValidationError):
        TextContent.model_validate({})


@pytest.mark.unit
def test_tool_response_serializes_to_dict() -> None:
    response = ToolResponse(content=[TextContent(text="ok")])
    result = response.model_dump()
    assert "content" in result
    assert "isError" in result


@pytest.mark.unit
def test_tool_response_is_error_defaults_false() -> None:
    response = ToolResponse(content=[TextContent(text="ok")])
    assert response.isError is False


@pytest.mark.unit
def test_server_icon_accepts_required_fields() -> None:
    icon: ServerIcon = {"src": "https://example.com/icon.svg", "mimeType": "image/svg+xml"}
    assert icon["src"] == "https://example.com/icon.svg"
    assert icon["mimeType"] == "image/svg+xml"


@pytest.mark.unit
def test_server_info_accepts_all_optional_fields() -> None:
    info: ServerInfo = {
        "name": "my-server",
        "version": "1.0.0",
        "title": "My Server",
        "description": "A test MCP server",
        "icons": [{"src": "https://example.com/icon.svg", "mimeType": "image/svg+xml"}],
        "websiteUrl": "https://example.com",
    }
    assert info["name"] == "my-server"
    assert info["version"] == "1.0.0"
    assert info["title"] == "My Server"
    assert info["description"] == "A test MCP server"
    assert info["websiteUrl"] == "https://example.com"


@pytest.mark.unit
def test_mcp_session_data_stores_fields() -> None:
    session_id = "sess-abc123"
    oauth_client_id = UUID("12345678-1234-5678-1234-567812345678")
    connection_id = None
    last_event_id = 42
    created_at = datetime(2026, 2, 28, 12, 0, 0)

    session = McpSessionData(
        session_id=session_id,
        oauth_client_id=oauth_client_id,
        connection_id=connection_id,
        last_event_id=last_event_id,
        created_at=created_at,
    )

    assert session.session_id == session_id
    assert session.oauth_client_id == oauth_client_id
    assert session.connection_id is None
    assert session.last_event_id == last_event_id
    assert session.created_at == created_at


# --- CompletionRef ---


@pytest.mark.unit
def test_completion_ref_stores_type_and_name() -> None:
    ref = CompletionRef(type="ref/prompt", name="my_prompt")
    assert ref.type == "ref/prompt"
    assert ref.name == "my_prompt"


@pytest.mark.unit
def test_completion_ref_resource_type() -> None:
    ref = CompletionRef(type="ref/resource", name="my_resource")
    assert ref.type == "ref/resource"


@pytest.mark.unit
def test_completion_ref_requires_type_and_name() -> None:
    with pytest.raises(ValidationError):
        CompletionRef.model_validate({})


# --- CompletionArgument ---


@pytest.mark.unit
def test_completion_argument_stores_name_and_value() -> None:
    arg = CompletionArgument(name="query", value="par")
    assert arg.name == "query"
    assert arg.value == "par"


@pytest.mark.unit
def test_completion_argument_requires_name_and_value() -> None:
    with pytest.raises(ValidationError):
        CompletionArgument.model_validate({})


# --- CompletionResult ---


@pytest.mark.unit
def test_completion_result_stores_values() -> None:
    result = CompletionResult(values=["Paris", "Parma"])
    assert result.values == ["Paris", "Parma"]


@pytest.mark.unit
def test_completion_result_defaults_total_none_and_has_more_false() -> None:
    result = CompletionResult(values=[])
    assert result.total is None
    assert result.hasMore is False


@pytest.mark.unit
def test_completion_result_accepts_total_and_has_more() -> None:
    result = CompletionResult(values=["a"], total=50, hasMore=True)
    assert result.total == 50
    assert result.hasMore is True


@pytest.mark.unit
def test_completion_result_requires_values() -> None:
    with pytest.raises(ValidationError):
        CompletionResult.model_validate({})


# --- ElicitationRequest ---


@pytest.mark.unit
def test_elicitation_request_stores_message_and_schema() -> None:
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    req = ElicitationRequest(message="Enter your name", requestedSchema=schema)
    assert req.message == "Enter your name"
    assert req.requestedSchema == schema


@pytest.mark.unit
def test_elicitation_request_requires_message_and_schema() -> None:
    with pytest.raises(ValidationError):
        ElicitationRequest.model_validate({})


# --- ElicitationResponse ---


@pytest.mark.unit
def test_elicitation_response_accept_with_content() -> None:
    resp = ElicitationResponse(action="accept", content={"name": "Alice"})
    assert resp.action == "accept"
    assert resp.content == {"name": "Alice"}


@pytest.mark.unit
def test_elicitation_response_decline_has_no_content() -> None:
    resp = ElicitationResponse(action="decline")
    assert resp.action == "decline"
    assert resp.content is None


@pytest.mark.unit
def test_elicitation_response_cancel_has_no_content() -> None:
    resp = ElicitationResponse(action="cancel")
    assert resp.action == "cancel"
    assert resp.content is None


@pytest.mark.unit
def test_elicitation_response_requires_action() -> None:
    with pytest.raises(ValidationError):
        ElicitationResponse.model_validate({})


# --- ImageContent ---


@pytest.mark.unit
def test_image_content_constructs_with_flat_data_and_mime_type() -> None:
    content = ImageContent(data="aGVsbG8=", mimeType="image/png")
    assert content.type == "image"
    assert content.data == "aGVsbG8="
    assert content.mimeType == "image/png"


@pytest.mark.unit
def test_image_content_serializes_to_conformant_flat_shape() -> None:
    content = ImageContent(data="aGVsbG8=", mimeType="image/png")
    dumped = content.model_dump()
    assert dumped == {"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"}


@pytest.mark.unit
def test_image_content_requires_data_and_mime_type() -> None:
    with pytest.raises(ValidationError):
        ImageContent.model_validate({})


# --- AudioContent ---


@pytest.mark.unit
def test_audio_content_constructs_with_flat_data_and_mime_type() -> None:
    content = AudioContent(data="ZmFrZQ==", mimeType="audio/mpeg")
    assert content.type == "audio"
    assert content.data == "ZmFrZQ=="
    assert content.mimeType == "audio/mpeg"


@pytest.mark.unit
def test_audio_content_serializes_to_conformant_flat_shape() -> None:
    content = AudioContent(data="ZmFrZQ==", mimeType="audio/mpeg")
    dumped = content.model_dump()
    assert dumped == {"type": "audio", "data": "ZmFrZQ==", "mimeType": "audio/mpeg"}


@pytest.mark.unit
def test_audio_content_requires_data_and_mime_type() -> None:
    with pytest.raises(ValidationError):
        AudioContent.model_validate({})


# --- ResourceLinkContent ---


@pytest.mark.unit
def test_resource_link_content_constructs_with_minimum_fields() -> None:
    link = ResourceLinkContent(uri="file:///tmp/report.csv", name="report.csv")
    assert link.type == "resource_link"
    assert link.uri == "file:///tmp/report.csv"
    assert link.name == "report.csv"


@pytest.mark.unit
def test_resource_link_content_omits_unset_optionals_from_dump() -> None:
    link = ResourceLinkContent(uri="file:///tmp/report.csv", name="report.csv")
    dumped = link.model_dump(exclude_none=True)
    assert dumped == {
        "type": "resource_link",
        "uri": "file:///tmp/report.csv",
        "name": "report.csv",
    }
    assert "title" not in dumped
    assert "description" not in dumped
    assert "mimeType" not in dumped
    assert "icons" not in dumped
    assert "size" not in dumped


@pytest.mark.unit
def test_resource_link_content_accepts_icons_list() -> None:
    icon = Icon(src="https://example.com/icon.svg")
    link = ResourceLinkContent(uri="file:///tmp/report.csv", name="report.csv", icons=[icon])
    assert link.icons == [icon]


@pytest.mark.unit
def test_resource_link_content_requires_uri_and_name() -> None:
    with pytest.raises(ValidationError):
        ResourceLinkContent.model_validate({})


# --- Icon ---


@pytest.mark.unit
def test_icon_constructs_with_https_src() -> None:
    icon = Icon(src="https://example.com/icon.png")
    assert icon.src == "https://example.com/icon.png"


@pytest.mark.unit
def test_icon_constructs_with_data_uri_src() -> None:
    icon = Icon(src="data:image/png;base64,aGVsbG8=")
    assert icon.src == "data:image/png;base64,aGVsbG8="


@pytest.mark.unit
def test_icon_omits_unset_optionals_from_dump() -> None:
    icon = Icon(src="https://example.com/icon.png")
    dumped = icon.model_dump(exclude_none=True)
    assert dumped == {"src": "https://example.com/icon.png"}
    assert "mimeType" not in dumped
    assert "sizes" not in dumped
    assert "theme" not in dumped


@pytest.mark.unit
def test_icon_accepts_optional_fields() -> None:
    icon = Icon(
        src="https://example.com/icon.png",
        mimeType="image/png",
        sizes=["32x32", "16x16"],
        theme="dark",
    )
    assert icon.mimeType == "image/png"
    assert icon.sizes == ["32x32", "16x16"]
    assert icon.theme == "dark"


@pytest.mark.unit
def test_icon_rejects_http_scheme_naming_scheme() -> None:
    with pytest.raises(ValueError, match="http"):
        Icon(src="http://example.com/icon.png")


@pytest.mark.unit
def test_icon_rejects_file_scheme_naming_scheme() -> None:
    with pytest.raises(ValueError, match="file"):
        Icon(src="file:///etc/passwd")


@pytest.mark.unit
def test_icon_rejects_svg_with_script_tag() -> None:
    malicious_svg = "data:image/svg+xml,<svg><script>alert(1)</script></svg>"
    with pytest.raises(ValueError, match="script"):
        Icon(src=malicious_svg)


@pytest.mark.unit
def test_icon_rejects_base64_encoded_svg_with_script_tag() -> None:
    payload = base64.b64encode(b"<svg><script>alert(1)</script></svg>").decode()
    malicious_svg = "data:image/svg+xml;base64," + payload
    with pytest.raises(ValueError, match="script"):
        Icon(src=malicious_svg)


@pytest.mark.unit
def test_icon_rejects_svg_with_onload_event_handler() -> None:
    malicious_svg = "data:image/svg+xml,<svg onload=alert(1)></svg>"
    with pytest.raises(ValueError, match="executable content"):
        Icon(src=malicious_svg)


@pytest.mark.unit
def test_icon_rejects_svg_with_javascript_uri() -> None:
    malicious_svg = 'data:image/svg+xml,<svg><a href="javascript:alert(1)"></a></svg>'
    with pytest.raises(ValueError, match="executable content"):
        Icon(src=malicious_svg)


@pytest.mark.unit
@pytest.mark.parametrize("handler", ["onpointerdown", "onbeforeinput"])
def test_icon_rejects_svg_with_pointer_and_input_event_handlers(handler: str) -> None:
    malicious_svg = f"data:image/svg+xml,<svg {handler}=alert(1)></svg>"
    with pytest.raises(ValueError, match="executable content"):
        Icon(src=malicious_svg)


@pytest.mark.unit
def test_icon_rejects_svg_with_foreign_object() -> None:
    malicious_svg = (
        'data:image/svg+xml,<svg><foreignObject><body xmlns="http://www.w3.org/1999/xhtml">'
        "hi</body></foreignObject></svg>"
    )
    with pytest.raises(ValueError, match="executable content"):
        Icon(src=malicious_svg)


@pytest.mark.unit
def test_icon_rejects_percent_encoded_foreign_object() -> None:
    malicious_svg = "data:image/svg+xml,%3Csvg%3E%3CforeignObject%3E%3C/foreignObject%3E%3C/svg%3E"
    with pytest.raises(ValueError, match="executable content"):
        Icon(src=malicious_svg)


@pytest.mark.unit
def test_icon_rejects_base64_encoded_foreign_object() -> None:
    payload = base64.b64encode(b"<svg><foreignObject></foreignObject></svg>").decode()
    malicious_svg = "data:image/svg+xml;base64," + payload
    with pytest.raises(ValueError, match="executable content"):
        Icon(src=malicious_svg)


@pytest.mark.unit
def test_icon_rejects_svg_with_css_expression() -> None:
    malicious_svg = "data:image/svg+xml,<svg><style>rect{width:expression(alert(1))}</style></svg>"
    with pytest.raises(ValueError, match="executable content"):
        Icon(src=malicious_svg)


@pytest.mark.unit
def test_icon_rejects_percent_encoded_css_expression() -> None:
    malicious_svg = "data:image/svg+xml,%3Cstyle%3Ewidth%3Aexpression(alert(1))%3C/style%3E"
    with pytest.raises(ValueError, match="executable content"):
        Icon(src=malicious_svg)


@pytest.mark.unit
def test_icon_rejects_base64_encoded_css_expression() -> None:
    payload = base64.b64encode(b"<style>width:expression(alert(1))</style>").decode()
    malicious_svg = "data:image/svg+xml;base64," + payload
    with pytest.raises(ValueError, match="executable content"):
        Icon(src=malicious_svg)


@pytest.mark.unit
def test_icon_rejects_svg_with_moz_binding() -> None:
    malicious_svg = "data:image/svg+xml,<svg><style>rect{-moz-binding:url(evil.xml#x)}</style></svg>"
    with pytest.raises(ValueError, match="executable content"):
        Icon(src=malicious_svg)


@pytest.mark.unit
def test_icon_rejects_percent_encoded_moz_binding() -> None:
    malicious_svg = "data:image/svg+xml,%3Cstyle%3E-moz-binding%3Aurl(evil.xml%23x)%3C/style%3E"
    with pytest.raises(ValueError, match="executable content"):
        Icon(src=malicious_svg)


@pytest.mark.unit
def test_icon_rejects_base64_encoded_moz_binding() -> None:
    payload = base64.b64encode(b"<style>-moz-binding:url(evil.xml#x)</style>").decode()
    malicious_svg = "data:image/svg+xml;base64," + payload
    with pytest.raises(ValueError, match="executable content"):
        Icon(src=malicious_svg)


@pytest.mark.unit
def test_icon_rejects_svg_with_xml_entity_declaration() -> None:
    malicious_svg = (
        'data:image/svg+xml,<?xml version="1.0"?><!DOCTYPE svg [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
        "<svg>&xxe;</svg>"
    )
    with pytest.raises(ValueError, match="executable content"):
        Icon(src=malicious_svg)


@pytest.mark.unit
def test_icon_rejects_percent_encoded_xml_entity_declaration() -> None:
    malicious_svg = "data:image/svg+xml,%3C!DOCTYPE%20svg%20%5B%3C!ENTITY%20xxe%20%22boom%22%3E%5D%3E"
    with pytest.raises(ValueError, match="executable content"):
        Icon(src=malicious_svg)


@pytest.mark.unit
def test_icon_rejects_base64_encoded_xml_entity_declaration() -> None:
    payload = base64.b64encode(b'<!DOCTYPE svg [<!ENTITY xxe "boom">]><svg>&xxe;</svg>').decode()
    malicious_svg = "data:image/svg+xml;base64," + payload
    with pytest.raises(ValueError, match="executable content"):
        Icon(src=malicious_svg)


@pytest.mark.unit
def test_icon_accepts_legitimate_svg_without_new_blocklist_vectors() -> None:
    # A plain, legitimate SVG icon must still pass despite the expanded
    # blocklist: no <foreignObject>, no CSS expression()/-moz-binding, and
    # no XML entity/DOCTYPE declarations.
    legitimate_svg = (
        'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
        '<circle cx="12" cy="12" r="10" fill="blue"/></svg>'
    )
    icon = Icon(src=legitimate_svg)
    assert "circle" in icon.src


@pytest.mark.unit
def test_icon_rejects_svg_with_undecodable_base64_payload() -> None:
    # A payload that fails to decode under every known base64 variant
    # (standard and URL-safe) is itself treated as suspicious: a legitimate
    # encoder never emits an undecodable payload.
    malicious_svg = "data:image/svg+xml;base64,a"
    with pytest.raises(ValueError, match="could not be decoded"):
        Icon(src=malicious_svg)


@pytest.mark.unit
def test_icon_rejects_svg_with_urlsafe_base64_encoded_script_tag() -> None:
    # base64.b64decode (strict alphabet) cannot decode "-"/"_" characters and
    # raises, which must not be treated as "nothing to scan": lenient
    # consumers (e.g. Node's Buffer.from(str, 'base64')) decode the
    # URL-safe alphabet and would render the script verbatim.
    raw = b"<svg><script>alert(1)</script></svg>" * 3
    payload = base64.urlsafe_b64encode(raw).decode()
    malicious_svg = "data:image/svg+xml;base64," + payload
    with pytest.raises(ValueError, match="executable content"):
        Icon(src=malicious_svg)


@pytest.mark.unit
def test_icon_rejects_svg_with_percent_encoded_script_tag() -> None:
    # Per RFC 2397 / WHATWG data: URL parsing, when ";base64," is absent the
    # payload after the comma is percent-decoded before use by conformant
    # consumers, resolving this to a literal "<script>alert(1)</script>".
    malicious_svg = "data:image/svg+xml,%3Cscript%3Ealert(1)%3C/script%3E"
    with pytest.raises(ValueError, match="executable content"):
        Icon(src=malicious_svg)


@pytest.mark.unit
def test_icon_rejects_non_svg_media_type_base64_script_tag() -> None:
    # The executable-content scan must not be gated on a "svg" media-type
    # hint: a data: URI declaring an unrelated media type still gets its
    # decoded payload scanned.
    payload = base64.b64encode(b"<script>alert(1)</script>").decode()
    malicious_uri = "data:text/html;base64," + payload
    with pytest.raises(ValueError, match="executable content"):
        Icon(src=malicious_uri)


@pytest.mark.unit
def test_icon_rejects_data_uri_with_no_media_type_and_script_tag() -> None:
    malicious_uri = "data:,<script>alert(1)</script>"
    with pytest.raises(ValueError, match="executable content"):
        Icon(src=malicious_uri)


@pytest.mark.unit
def test_icon_rejects_mismatched_media_type_with_script_tag() -> None:
    malicious_uri = "data:image/png,<script>alert(1)</script>"
    with pytest.raises(ValueError, match="executable content"):
        Icon(src=malicious_uri)


@pytest.mark.unit
def test_icon_accepts_legitimate_binary_png_data_uri() -> None:
    # Unconditional scanning must not false-positive on real binary image
    # bytes decoded (with "replace" on invalid UTF-8) from a legitimate
    # data:image/png;base64, payload.
    png_bytes = bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010806000000") + bytes(range(256))
    payload = base64.b64encode(png_bytes).decode()
    icon = Icon(src="data:image/png;base64," + payload)
    assert icon.src.startswith("data:image/png;base64,")


@pytest.mark.unit
@pytest.mark.parametrize(
    "attribute",
    ["data-once='1'", "data-online='yes'", "data-only='true'"],
)
def test_icon_accepts_svg_with_legitimate_attributes_prefixed_by_on(attribute: str) -> None:
    # "\bon\w+\s*=" false-positives on legitimate words beginning with "on"
    # (once, online, only) because the preceding "-" already satisfies "\b".
    # Matching against an explicit event-handler allowlist avoids this.
    icon = Icon(src=f"data:image/svg+xml,<svg {attribute}></svg>")
    assert "data-on" in icon.src


@pytest.mark.unit
def test_icon_rejects_disallowed_mime_type() -> None:
    with pytest.raises(ValueError, match="text/html"):
        Icon(src="https://example.com/icon.png", mimeType="text/html")


@pytest.mark.unit
def test_icon_accepts_allowed_mime_types() -> None:
    icon = Icon(src="https://example.com/icon.webp", mimeType="image/webp")
    assert icon.mimeType == "image/webp"


@pytest.mark.unit
def test_icon_mime_type_remains_optional_and_omitted_from_dump() -> None:
    icon = Icon(src="https://example.com/icon.png")
    assert icon.mimeType is None
    dumped = icon.model_dump(exclude_none=True)
    assert "mimeType" not in dumped


# --- ToolAnnotations ---


@pytest.mark.unit
def test_tool_annotations_constructs_with_each_hint_field() -> None:
    annotations = ToolAnnotations(
        title="Delete Record",
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=True,
        openWorldHint=True,
    )
    assert annotations.title == "Delete Record"
    assert annotations.readOnlyHint is False
    assert annotations.destructiveHint is True
    assert annotations.idempotentHint is True
    assert annotations.openWorldHint is True


@pytest.mark.unit
def test_tool_annotations_omits_unset_hints_individually() -> None:
    annotations = ToolAnnotations(readOnlyHint=True)
    dumped = annotations.model_dump(exclude_none=True)
    assert dumped == {"readOnlyHint": True}
    assert "title" not in dumped
    assert "destructiveHint" not in dumped
    assert "idempotentHint" not in dumped
    assert "openWorldHint" not in dumped


@pytest.mark.unit
def test_tool_annotations_passes_through_unknown_vendor_keys() -> None:
    annotations = ToolAnnotations.model_validate({"readOnlyHint": True, "vendor:acme:priority": "high"})
    assert annotations.readOnlyHint is True
    dumped = annotations.model_dump()
    assert dumped["vendor:acme:priority"] == "high"


@pytest.mark.unit
def test_handle_initialize_merges_valid_server_icons() -> None:
    server_info: ServerInfo = {
        "name": "my-server",
        "icons": [{"src": "https://example.com/icon.png", "mimeType": "image/png"}],
    }
    result = handle_initialize({}, "2025-06-18", server_info)
    merged_server_info = cast("dict[str, object]", result["serverInfo"])
    assert merged_server_info["icons"] == server_info["icons"]


@pytest.mark.unit
def test_handle_initialize_rejects_server_icon_with_disallowed_scheme() -> None:
    server_info: ServerInfo = {
        "name": "my-server",
        "icons": [{"src": "http://example.com/icon.png"}],
    }
    with pytest.raises(MCPError) as exc_info:
        handle_initialize({}, "2025-06-18", server_info)
    assert exc_info.value.code == -32602


@pytest.mark.unit
def test_handle_initialize_rejects_server_icon_with_javascript_uri() -> None:
    server_info: ServerInfo = {
        "name": "my-server",
        "icons": [{"src": "javascript:alert(1)"}],
    }
    with pytest.raises(MCPError, match="Invalid server_info icon"):
        handle_initialize({}, "2025-06-18", server_info)


@pytest.mark.unit
def test_handle_initialize_rejects_server_icon_with_unsafe_svg_data_uri() -> None:
    server_info: ServerInfo = {
        "name": "my-server",
        "icons": [{"src": "data:image/svg+xml,<svg onload=alert(1)></svg>"}],
    }
    with pytest.raises(MCPError, match="Invalid server_info icon"):
        handle_initialize({}, "2025-06-18", server_info)


@pytest.mark.unit
def test_new_completion_elicitation_types_not_in_public_all() -> None:
    import fastapi_mcp_router

    new_types = [
        "CompletionRef",
        "CompletionArgument",
        "CompletionResult",
        "ElicitationRequest",
        "ElicitationResponse",
    ]
    for name in new_types:
        assert name not in fastapi_mcp_router.__all__, f"{name} must not be in __all__"
