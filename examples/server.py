"""Example MCP server exercising tools, resources, and prompts.

Start with: uvicorn examples.server:app --reload
Connect MCP Inspector: npx @modelcontextprotocol/inspector
  Transport: Streamable HTTP
  URL: http://localhost:8000/mcp
"""

import asyncio
import json
from datetime import UTC, datetime

from fastapi import FastAPI

from fastapi_mcp_router import InMemorySessionStore, MCPRouter, ProgressCallback, ServerInfo

app = FastAPI(title="MCP Example Server")


async def _allow_all(_api_key: str | None, _bearer: str | None) -> bool:
    """Skip authentication for this demo server."""
    return True


mcp = MCPRouter(
    server_info=ServerInfo(name="example-server", version="0.1.0"),
    session_store=InMemorySessionStore(),
    stateful=True,
    auth_validator=_allow_all,
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def echo(message: str) -> str:
    """Echo a message back to the caller."""
    return f"Echo: {message}"


@mcp.tool()
async def countdown(seconds: int, progress: ProgressCallback) -> str:
    """Count down for the given number of seconds, reporting progress each tick."""
    for i in range(seconds):
        await progress(i, seconds, f"Tick {i + 1}/{seconds}")
        await asyncio.sleep(1)
    return f"Countdown complete after {seconds}s"


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@mcp.resource(uri="time://now", name="Current Time", description="Server UTC timestamp")
async def get_current_time() -> str:
    """Return the current UTC time as ISO-8601."""
    return datetime.now(UTC).isoformat()


@mcp.resource(
    uri="config://{key}",
    name="Config Value",
    description="Read a configuration key",
    mime_type="application/json",
)
async def get_config(key: str) -> str:
    """Return a configuration value by key (demo data)."""
    store = {"debug": "true", "version": "0.1.0", "region": "us-east-1"}
    value = store.get(key)
    return json.dumps({"key": key, "value": value})


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


@mcp.prompt()
async def code_review(language: str, style: str = "concise") -> list[dict]:
    """Generate a code review prompt for the given language."""
    return [
        {
            "role": "user",
            "content": (
                f"Review the following {language} code. "
                f"Provide {style} feedback covering correctness, "
                "readability, and performance."
            ),
        }
    ]


app.include_router(mcp, prefix="/mcp")
