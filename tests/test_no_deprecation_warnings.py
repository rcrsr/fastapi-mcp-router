"""Unit tests confirming inspect.iscoroutinefunction emits no DeprecationWarnings.

Covers AC-10 (async handler) and AC-11 (sync handler) from IR-4.
Registry and ResourceRegistry registration paths are both exercised.
"""

import warnings

import pytest

from fastapi_mcp_router import MCPToolRegistry, ResourceRegistry

# ---------------------------------------------------------------------------
# AC-10: Async handler registration -> 0 DeprecationWarnings
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_register_async_tool_handler_emits_no_deprecation_warnings():
    """Registering an async tool handler emits 0 DeprecationWarnings (AC-10).

    Verifies that inspect.iscoroutinefunction does not emit DeprecationWarnings
    when detecting an async coroutine function during tool registration.
    """
    registry = MCPToolRegistry()

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")

        @registry.tool()
        async def my_async_tool(value: str) -> str:
            """An async tool."""
            return value.upper()

    deprecation_warnings = [w for w in captured if issubclass(w.category, DeprecationWarning)]
    assert len(deprecation_warnings) == 0, (
        f"Expected 0 DeprecationWarnings, got {len(deprecation_warnings)}: "
        f"{[str(w.message) for w in deprecation_warnings]}"
    )


# ---------------------------------------------------------------------------
# AC-11: Sync handler registration -> 0 DeprecationWarnings
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_register_sync_tool_handler_emits_no_deprecation_warnings():
    """Registering a sync tool handler emits 0 DeprecationWarnings (AC-11).

    A sync function raises TypeError at registration time; that is expected.
    The test confirms inspect.iscoroutinefunction itself emits no deprecation
    warnings when evaluating a non-async function.
    """
    registry = MCPToolRegistry()

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")

        with pytest.raises(TypeError):

            @registry.tool()
            def my_sync_tool(value: str) -> str:
                """A sync tool (rejected by registry)."""
                return value.upper()

    deprecation_warnings = [w for w in captured if issubclass(w.category, DeprecationWarning)]
    assert len(deprecation_warnings) == 0, (
        f"Expected 0 DeprecationWarnings, got {len(deprecation_warnings)}: "
        f"{[str(w.message) for w in deprecation_warnings]}"
    )


# ---------------------------------------------------------------------------
# Resource registry: async handler registration -> 0 DeprecationWarnings
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_register_async_resource_handler_emits_no_deprecation_warnings():
    """Registering an async resource handler emits 0 DeprecationWarnings.

    Covers the resources.py iscoroutinefunction call path (IR-4, AC-10).
    """
    registry = ResourceRegistry()

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")

        @registry.resource(uri_template="data://{key}", name="DataResource")
        async def get_data(key: str) -> str:
            """Fetch data by key."""
            return f"value_of_{key}"

    deprecation_warnings = [w for w in captured if issubclass(w.category, DeprecationWarning)]
    assert len(deprecation_warnings) == 0, (
        f"Expected 0 DeprecationWarnings, got {len(deprecation_warnings)}: "
        f"{[str(w.message) for w in deprecation_warnings]}"
    )


# ---------------------------------------------------------------------------
# Resource registry: sync handler registration -> 0 DeprecationWarnings
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_register_sync_resource_handler_emits_no_deprecation_warnings():
    """Registering a sync resource handler emits 0 DeprecationWarnings (AC-11).

    A sync resource handler raises TypeError; the test confirms no deprecation
    warnings are emitted by inspect.iscoroutinefunction during that check.
    """
    registry = ResourceRegistry()

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")

        with pytest.raises(TypeError):

            @registry.resource(uri_template="data://{key}", name="DataResource")
            def get_data_sync(key: str) -> str:
                """Fetch data by key (sync, rejected)."""
                return f"value_of_{key}"

    deprecation_warnings = [w for w in captured if issubclass(w.category, DeprecationWarning)]
    assert len(deprecation_warnings) == 0, (
        f"Expected 0 DeprecationWarnings, got {len(deprecation_warnings)}: "
        f"{[str(w.message) for w in deprecation_warnings]}"
    )
