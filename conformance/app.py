"""Fixture server for the official MCP conformance suite.

Exposes the tools, resources, and prompts that the server-side scenarios in
``@modelcontextprotocol/conformance`` call by name. Every handler goes through
the public ``fastapi_mcp_router`` API so that a failing scenario points at a
library gap, not at fixture trickery.

Start with:  uvicorn conformance.app:app --port 3001
Run suite:   bash conformance/run.sh
"""

import asyncio
import base64
import json

from fastapi import FastAPI
from starlette.middleware.trustedhost import TrustedHostMiddleware

from fastapi_mcp_router import (
    AudioContent,
    ImageContent,
    InMemorySessionStore,
    MCPToolRegistry,
    ProgressCallback,
    PromptRegistry,
    ResourceRegistry,
    ServerInfo,
    TextContent,
    ToolError,
    create_mcp_router,
)
from fastapi_mcp_router.session import SamplingManager

# 1x1 red PNG and a 44-byte silent WAV header: the smallest valid payloads.
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
WAV_SILENT = b"RIFF" + (36).to_bytes(4, "little") + b"WAVEfmt " + (16).to_bytes(4, "little")
WAV_SILENT += (1).to_bytes(2, "little") + (1).to_bytes(2, "little") + (8000).to_bytes(4, "little")
WAV_SILENT += (8000).to_bytes(4, "little") + (1).to_bytes(2, "little") + (8).to_bytes(2, "little")
WAV_SILENT += b"data" + (0).to_bytes(4, "little")

JSON_SCHEMA_2020_12_FIXTURE: dict[str, object] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "$defs": {
        "address": {
            "$anchor": "addressDef",
            "type": "object",
            "properties": {"street": {"type": "string"}, "city": {"type": "string"}},
        }
    },
    "properties": {
        "name": {"type": "string"},
        "address": {"$ref": "#/$defs/address"},
        "contactMethod": {"type": "string", "enum": ["phone", "email"]},
        "phone": {"type": "string"},
        "email": {"type": "string"},
    },
    "allOf": [{"anyOf": [{"required": ["phone"]}, {"required": ["email"]}]}],
    "if": {"properties": {"contactMethod": {"const": "phone"}}, "required": ["contactMethod"]},
    "then": {"required": ["phone"]},
    "else": {"required": ["email"]},
    "additionalProperties": False,
}


async def _allow_all(_api_key: str | None, _bearer: str | None) -> bool:
    return True


async def _complete(ref: dict, argument: dict) -> dict:
    """Minimal completion handler: prefix-match a fixed vocabulary."""
    candidates = ["testValue1", "testValue2", "test", "paris", "park", "party"]
    prefix = str(argument.get("value", ""))
    values = [c for c in candidates if c.startswith(prefix)]
    return {"values": values, "total": len(values), "hasMore": False}


# create_mcp_router() is used instead of MCPRouter because only the factory
# accepts completion_handler (the completion-complete scenario needs it).
tools = MCPToolRegistry()
resources = ResourceRegistry()
prompts = PromptRegistry()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tools.tool()
async def test_simple_text() -> str:
    """Return a plain text response."""
    return "This is a simple text response for testing."


@tools.tool()
async def test_image_content() -> ImageContent:
    """Return a 1x1 PNG."""
    return ImageContent(data=base64.b64encode(PNG_1X1).decode(), mimeType="image/png")


@tools.tool()
async def test_audio_content() -> AudioContent:
    """Return a minimal WAV clip."""
    return AudioContent(data=base64.b64encode(WAV_SILENT).decode(), mimeType="audio/wav")


@tools.tool()
async def test_embedded_resource() -> dict:
    """Return an embedded resource content block.

    The public API has no EmbeddedResource content model, so this dict is
    JSON-stringified into a text block. Baselined.
    """
    return {
        "type": "resource",
        "resource": {
            "uri": "test://embedded-resource",
            "mimeType": "text/plain",
            "text": "This is an embedded resource for testing.",
        },
    }


@tools.tool()
async def test_multiple_content_types() -> list[TextContent | ImageContent]:
    """Return text and image blocks in one response.

    The scenario also expects an embedded resource block, which the public
    API cannot express. Baselined.
    """
    return [
        TextContent(text="Multiple content types test:"),
        ImageContent(data=base64.b64encode(PNG_1X1).decode(), mimeType="image/png"),
    ]


@tools.tool()
async def test_error_handling() -> str:
    """Always fail so the client sees isError: true."""
    raise ToolError("This tool intentionally returns an error for testing")


@tools.tool()
async def test_tool_with_progress(progress: ProgressCallback) -> str:
    """Emit three progress notifications 50ms apart."""
    await progress(0, 100, "started")
    await asyncio.sleep(0.05)
    await progress(50, 100, "halfway")
    await asyncio.sleep(0.05)
    await progress(100, 100, "done")
    return "Progress tool completed"


@tools.tool()
async def test_tool_with_logging() -> str:
    """Log three messages during execution.

    The library has no public handle for emitting notifications/message from
    inside a tool, so this returns text only. The scenario is baselined.
    """
    await asyncio.sleep(0.05)
    await asyncio.sleep(0.05)
    return "Logging tool completed"


@tools.tool()
async def test_sampling(prompt: str, sampling_manager: SamplingManager) -> str:
    """Ask the client's LLM to answer the prompt.

    SamplingManager.create_message() requires a session_id the tool cannot
    obtain from the public API, so this cannot complete. Baselined.
    """
    raise ToolError("test_sampling cannot resolve its session id via the public API")


@tools.tool()
async def test_elicitation(message: str) -> str:
    """Request structured input from the client.

    No public API lets a tool issue elicitation/create. Baselined.
    """
    raise ToolError(f"elicitation is not reachable from a tool handler: {message}")


@tools.tool()
async def test_elicitation_sep1034_defaults() -> str:
    """Elicitation with default values. Baselined, see test_elicitation."""
    raise ToolError("elicitation is not reachable from a tool handler")


@tools.tool()
async def test_elicitation_sep1330_enums() -> str:
    """Elicitation with enum variants. Baselined, see test_elicitation."""
    raise ToolError("elicitation is not reachable from a tool handler")


@tools.tool(name="json_schema_2020_12_tool", input_schema=JSON_SCHEMA_2020_12_FIXTURE)
async def json_schema_2020_12_tool(**kwargs: object) -> str:
    """Tool with JSON Schema 2020-12 features."""
    return json.dumps(kwargs)


@tools.tool()
async def test_reconnection() -> str:
    """SEP-1699 polling probe. The library does not close the stream mid-call."""
    await asyncio.sleep(0.1)
    return "reconnection tool completed"


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@resources.resource(uri_template="test://static-text", name="Static Text", mime_type="text/plain")
async def static_text() -> str:
    """A static text resource."""
    return "This is the content of the static text resource."


@resources.resource(uri_template="test://static-binary", name="Static Binary", mime_type="image/png")
async def static_binary() -> bytes:
    """A static binary resource."""
    return PNG_1X1


@resources.resource(uri_template="test://template/{id}/data", name="Template Data", mime_type="application/json")
async def template_data(id: str) -> str:
    """A templated resource."""
    return json.dumps({"id": id, "templateTest": True, "data": f"Data for ID: {id}"})


@resources.resource(uri_template="test://watched-resource", name="Watched Resource", mime_type="text/plain")
async def watched_resource() -> str:
    """A resource clients can subscribe to."""
    return "watched"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def _text(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


@prompts.prompt()
async def test_simple_prompt() -> list[dict]:
    """A prompt with no arguments, using the ``content: str`` shorthand.

    The registry wraps the string into a text content block on the wire.
    """
    return [{"role": "user", "content": "This is a simple prompt for testing."}]


@prompts.prompt()
async def test_prompt_with_arguments(arg1: str, arg2: str) -> list[dict]:
    """A prompt with two required arguments."""
    return [{"role": "user", "content": _text(f"Prompt with arguments: arg1='{arg1}', arg2='{arg2}'")}]


@prompts.prompt()
async def test_prompt_with_embedded_resource(resourceUri: str) -> list[dict]:
    """A prompt that embeds a resource."""
    return [
        {
            "role": "user",
            "content": {
                "type": "resource",
                "resource": {
                    "uri": resourceUri,
                    "mimeType": "text/plain",
                    "text": "Embedded resource content for testing.",
                },
            },
        },
        {"role": "user", "content": _text("Please process the embedded resource above.")},
    ]


@prompts.prompt()
async def test_prompt_with_image() -> list[dict]:
    """A prompt that includes an image."""
    return [
        {
            "role": "user",
            "content": {
                "type": "image",
                "data": base64.b64encode(PNG_1X1).decode(),
                "mimeType": "image/png",
            },
        },
        {"role": "user", "content": _text("Please analyze the image above.")},
    ]


app = FastAPI(title="fastapi-mcp-router conformance fixture")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]"])
app.include_router(
    create_mcp_router(
        tools,
        auth_validator=_allow_all,
        server_info=ServerInfo(name="fastapi-mcp-router-conformance", version="0.0.0"),
        session_store=InMemorySessionStore(),
        stateful=True,
        resource_registry=resources,
        prompt_registry=prompts,
        sampling_enabled=True,
        completion_handler=_complete,
        # Progress and log notifications are only drained on the GET stream,
        # never on the POST response stream. Enable it so clients that open
        # a GET stream (the TS SDK does) can receive them.
        legacy_sse=True,
    ),
    prefix="/mcp",
)
