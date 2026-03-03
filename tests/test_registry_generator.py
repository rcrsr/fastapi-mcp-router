"""Unit tests for AsyncGenerator[dict] support in MCPToolRegistry.

Covers AC-11, AC-14, AC-91, EC-9, EC-10, IC-7 from the Phase 2 specification.
"""

import typing
from collections.abc import AsyncGenerator

import pytest

from fastapi_mcp_router import MCPToolRegistry, ToolError
from fastapi_mcp_router.registry import ToolDefinition

# IC-7: ToolDefinition has is_generator field


@pytest.mark.unit
def test_tooldefinition_has_is_generator_field():
    """ToolDefinition exposes is_generator=False by default (IC-7)."""

    async def handler() -> dict:
        return {}

    tool = ToolDefinition(
        name="t",
        description="",
        input_schema={"type": "object", "properties": {}, "required": []},
        handler=handler,
    )

    assert hasattr(tool, "is_generator")
    assert tool.is_generator is False


@pytest.mark.unit
def test_tooldefinition_is_generator_true_when_set():
    """ToolDefinition stores is_generator=True when explicitly passed (IC-7)."""

    async def handler() -> AsyncGenerator[dict]:
        yield {}

    tool = ToolDefinition(
        name="t",
        description="",
        input_schema={"type": "object", "properties": {}, "required": []},
        handler=handler,
        is_generator=True,
    )

    assert tool.is_generator is True


# AC-11: AsyncGenerator[dict] accepted as tool return type


@pytest.mark.unit
def test_register_async_generator_tool_sets_is_generator_true():
    """Registering an AsyncGenerator[dict] tool sets is_generator=True (AC-11)."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def gen_tool() -> AsyncGenerator[dict]:
        """A streaming tool."""
        yield {"step": 1}
        yield {"step": 2}

    tool_def = registry._tools["gen_tool"]
    assert tool_def.is_generator is True


@pytest.mark.unit
def test_register_typing_async_generator_tool_sets_is_generator_true():
    """typing.AsyncGenerator[dict] annotation also sets is_generator=True."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def gen_tool() -> typing.AsyncGenerator[dict, None]:
        """A streaming tool using typing alias."""
        yield {"n": 0}

    tool_def = registry._tools["gen_tool"]
    assert tool_def.is_generator is True


# AC-14: dict-returning tools are completely unaffected


@pytest.mark.unit
def test_register_dict_tool_is_generator_false():
    """dict-returning tool has is_generator=False (AC-14)."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def plain_tool() -> dict:
        return {"ok": True}

    tool_def = registry._tools["plain_tool"]
    assert tool_def.is_generator is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_call_dict_tool_returns_dict_unchanged():
    """dict-returning tool still returns its dict result (AC-14)."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def plain_tool(x: int) -> dict:
        return {"value": x}

    result = await registry.call_tool("plain_tool", {"x": 7})
    assert result == {"value": 7}


# AC-91: Empty generator returns empty list


@pytest.mark.unit
@pytest.mark.asyncio
async def test_call_empty_generator_returns_empty_list():
    """Empty AsyncGenerator yields empty content list (AC-91)."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def empty_gen() -> AsyncGenerator[dict]:
        """Empty generator."""
        return
        yield  # pragma: no cover

    result = await registry.call_tool("empty_gen", {})
    assert result == []


# Happy-path: generator yields multiple dicts -> collected list


@pytest.mark.unit
@pytest.mark.asyncio
async def test_call_generator_tool_collects_all_yields():
    """Generator tool collects all yielded dicts into a list."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def multi_gen(count: int) -> AsyncGenerator[dict]:
        """Yields count dicts."""
        for i in range(count):
            yield {"index": i}

    result = await registry.call_tool("multi_gen", {"count": 3})
    assert result == [{"index": 0}, {"index": 1}, {"index": 2}]


# EC-9: Generator yields non-dict -> ToolError


@pytest.mark.unit
@pytest.mark.asyncio
async def test_call_generator_non_dict_yield_raises_tool_error():
    """Generator that yields a non-dict raises ToolError (EC-9)."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def bad_yield() -> AsyncGenerator[dict]:
        """Yields a string instead of dict."""
        yield "not a dict"  # type: ignore[misc]

    with pytest.raises(ToolError) as exc_info:
        await registry.call_tool("bad_yield", {})

    assert "non-dict" in exc_info.value.message
    assert "str" in exc_info.value.message


# EC-10: Generator raises exception -> ToolError


@pytest.mark.unit
@pytest.mark.asyncio
async def test_call_generator_exception_raises_tool_error():
    """Generator that raises mid-iteration produces ToolError (EC-10)."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def exploding_gen() -> AsyncGenerator[dict]:
        """Yields one item then raises."""
        yield {"first": True}
        raise ValueError("mid-stream failure")

    with pytest.raises(ToolError) as exc_info:
        await registry.call_tool("exploding_gen", {})

    assert "mid-stream failure" in exc_info.value.message


@pytest.mark.unit
@pytest.mark.asyncio
async def test_call_generator_immediate_exception_raises_tool_error():
    """Generator that raises before any yield produces ToolError (EC-10)."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def fail_immediately() -> AsyncGenerator[dict]:
        """Raises before yielding anything."""
        raise RuntimeError("startup error")
        yield  # pragma: no cover

    with pytest.raises(ToolError) as exc_info:
        await registry.call_tool("fail_immediately", {})

    assert "startup error" in exc_info.value.message
