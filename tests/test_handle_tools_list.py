"""Unit tests for handle_tools_list function with excluded_tools parameter.

Tests the tool filtering logic in handle_tools_list to verify:
- Default behavior returns all tools when no exclusions specified
- Exclusions correctly filter out specified tools
- Empty exclusion list does not filter any tools
- Non-existent exclusion names are handled gracefully
"""

import pytest

from fastapi_mcp_router import MCPToolRegistry
from fastapi_mcp_router.router import handle_tools_list


@pytest.fixture(name="registry")
def registry_fixture() -> MCPToolRegistry:
    """Create tool registry with test tools for exclusion testing."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def tool_a() -> str:
        """First test tool."""
        return "a"

    @registry.tool()
    async def tool_b() -> str:
        """Second test tool."""
        return "b"

    @registry.tool()
    async def tool_c() -> str:
        """Third test tool."""
        return "c"

    return registry


def _get_tool_names(result: dict[str, object]) -> list[str]:
    """Extract tool names from handle_tools_list result."""
    tools = result["tools"]
    assert isinstance(tools, list)
    return [t["name"] for t in tools]  # type: ignore[index]


# Test: default behavior without exclusions


@pytest.mark.unit
def test_handle_tools_list_without_exclusions_returns_all_tools(
    registry: MCPToolRegistry,
) -> None:
    """Verify handle_tools_list returns all registered tools when excluded_tools is None."""
    # Act
    result = handle_tools_list(registry)

    # Assert
    tool_names = _get_tool_names(result)
    assert len(tool_names) == 3
    assert "tool_a" in tool_names
    assert "tool_b" in tool_names
    assert "tool_c" in tool_names


# Test: exclusions filter tools correctly


@pytest.mark.unit
def test_handle_tools_list_with_exclusions_filters_tools(
    registry: MCPToolRegistry,
) -> None:
    """Verify handle_tools_list filters out tools specified in excluded_tools list."""
    # Act
    result = handle_tools_list(registry, excluded_tools=["tool_b"])

    # Assert
    tool_names = _get_tool_names(result)
    assert len(tool_names) == 2
    assert "tool_a" in tool_names
    assert "tool_b" not in tool_names
    assert "tool_c" in tool_names


@pytest.mark.unit
def test_handle_tools_list_with_multiple_exclusions_filters_all(
    registry: MCPToolRegistry,
) -> None:
    """Verify handle_tools_list filters out multiple tools when specified."""
    # Act
    result = handle_tools_list(registry, excluded_tools=["tool_a", "tool_c"])

    # Assert
    tool_names = _get_tool_names(result)
    assert len(tool_names) == 1
    assert "tool_a" not in tool_names
    assert "tool_b" in tool_names
    assert "tool_c" not in tool_names


# Test: empty exclusion list returns all tools


@pytest.mark.unit
def test_handle_tools_list_with_empty_exclusions_returns_all_tools(
    registry: MCPToolRegistry,
) -> None:
    """Verify handle_tools_list returns all tools when excluded_tools is an empty list."""
    # Act
    result = handle_tools_list(registry, excluded_tools=[])

    # Assert
    tool_names = _get_tool_names(result)
    assert len(tool_names) == 3
    assert "tool_a" in tool_names
    assert "tool_b" in tool_names
    assert "tool_c" in tool_names


# Test: non-existent exclusion names handled gracefully


@pytest.mark.unit
def test_handle_tools_list_with_nonexistent_exclusion_returns_all_tools(
    registry: MCPToolRegistry,
) -> None:
    """Verify handle_tools_list ignores non-existent tool names in excluded_tools."""
    # Act
    result = handle_tools_list(registry, excluded_tools=["nonexistent_tool"])

    # Assert
    tool_names = _get_tool_names(result)
    assert len(tool_names) == 3
    assert "tool_a" in tool_names
    assert "tool_b" in tool_names
    assert "tool_c" in tool_names


@pytest.mark.unit
def test_handle_tools_list_with_mixed_existent_and_nonexistent_exclusions(
    registry: MCPToolRegistry,
) -> None:
    """Verify handle_tools_list filters existing tools and ignores non-existent ones."""
    # Act
    result = handle_tools_list(registry, excluded_tools=["tool_a", "nonexistent_tool", "another_missing"])

    # Assert
    tool_names = _get_tool_names(result)
    assert len(tool_names) == 2
    assert "tool_a" not in tool_names
    assert "tool_b" in tool_names
    assert "tool_c" in tool_names


# Test: response structure


@pytest.mark.unit
def test_handle_tools_list_returns_correct_structure(
    registry: MCPToolRegistry,
) -> None:
    """Verify handle_tools_list returns dict with 'tools' key containing list."""
    # Act
    result = handle_tools_list(registry, excluded_tools=["tool_a"])

    # Assert
    assert isinstance(result, dict)
    assert "tools" in result
    tools = result["tools"]
    assert isinstance(tools, list)
    for tool in tools:
        assert isinstance(tool, dict)
        assert "name" in tool
        assert "description" in tool
        assert "inputSchema" in tool
