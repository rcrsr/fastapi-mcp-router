"""
Unit tests for public API models in fastapi_mcp_router.types.

Covers TextContent, ToolResponse, ServerIcon, ServerInfo, McpSessionData,
CompletionRef, CompletionArgument, CompletionResult, ElicitationRequest,
and ElicitationResponse.
"""

from datetime import datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from fastapi_mcp_router import ServerIcon, ServerInfo, TextContent, ToolResponse
from fastapi_mcp_router.types import (
    CompletionArgument,
    CompletionRef,
    CompletionResult,
    ElicitationRequest,
    ElicitationResponse,
    McpSessionData,
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
