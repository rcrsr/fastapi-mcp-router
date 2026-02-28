"""Unit tests for fastapi-mcp-router registry module.

Tests MCPToolRegistry and ToolDefinition classes for tool registration,
schema generation, tool listing, and tool execution across various scenarios.
"""

from typing import Literal, cast

import pytest
from fastapi import Depends

from fastapi_mcp_router import MCPError, MCPToolRegistry, ToolError
from fastapi_mcp_router.registry import ToolDefinition

# ToolDefinition tests


@pytest.mark.unit
def test_tooldefinition_basic_instantiation():
    """Test ToolDefinition can be instantiated with all parameters."""
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}

    async def handler(name: str) -> str:
        return f"Hello, {name}"

    tool = ToolDefinition(
        name="greet",
        description="Greet a user",
        input_schema=schema,
        handler=handler,
    )

    assert tool.name == "greet"
    assert tool.description == "Greet a user"
    assert tool.input_schema == schema
    assert tool.handler == handler


@pytest.mark.unit
def test_tooldefinition_attributes_accessible():
    """Test ToolDefinition attributes are accessible after instantiation."""
    schema = {"type": "object", "properties": {}}

    async def my_handler() -> dict:
        return {"status": "ok"}

    tool = ToolDefinition(
        name="test_tool",
        description="Test description",
        input_schema=schema,
        handler=my_handler,
    )

    assert hasattr(tool, "name")
    assert hasattr(tool, "description")
    assert hasattr(tool, "input_schema")
    assert hasattr(tool, "handler")
    assert tool.name == "test_tool"
    assert tool.description == "Test description"
    assert tool.input_schema == schema
    assert tool.handler == my_handler


# MCPToolRegistry.tool() decorator tests


@pytest.mark.unit
def test_register_tool_default_name():
    """Test tool registration uses function name by default."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def my_tool(param: str) -> dict:
        return {"result": param}

    tools = registry.list_tools()
    assert len(tools) == 1
    assert tools[0]["name"] == "my_tool"


@pytest.mark.unit
def test_register_tool_custom_name():
    """Test tool registration with custom name."""
    registry = MCPToolRegistry()

    @registry.tool(name="custom_name")
    async def my_tool(param: str) -> dict:
        return {"result": param}

    tools = registry.list_tools()
    assert len(tools) == 1
    assert tools[0]["name"] == "custom_name"


@pytest.mark.unit
def test_register_tool_custom_description():
    """Test tool registration with custom description."""
    registry = MCPToolRegistry()

    @registry.tool(description="Custom description")
    async def my_tool(param: str) -> dict:
        """Original docstring."""
        return {"result": param}

    tools = registry.list_tools()
    assert len(tools) == 1
    assert tools[0]["description"] == "Custom description"


@pytest.mark.unit
def test_register_tool_explicit_input_schema():
    """Test tool registration with explicit input schema."""
    registry = MCPToolRegistry()
    custom_schema = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
    }

    @registry.tool(input_schema=custom_schema)
    async def my_tool(count: int) -> dict:
        return {"count": count}

    tools = registry.list_tools()
    assert len(tools) == 1
    assert tools[0]["inputSchema"] == custom_schema


@pytest.mark.unit
def test_register_tool_returns_original_function():
    """Test decorator returns original function unchanged."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def my_tool(value: str) -> str:
        return value.upper()

    # Function should still work normally
    import asyncio

    result = asyncio.run(my_tool("hello"))
    assert result == "HELLO"


@pytest.mark.unit
def test_register_tool_raises_on_non_async_function():
    """Test decorator raises TypeError when decorating non-async function."""
    registry = MCPToolRegistry()

    with pytest.raises(TypeError, match=r"Tool function .* must be async.*Add 'async def'"):

        @registry.tool()
        def sync_tool(param: str) -> str:
            return param


@pytest.mark.unit
def test_register_tool_uses_docstring_as_description():
    """Test that function docstring is used as description when not provided."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def my_tool(param: str) -> dict:
        """Process a parameter value."""
        return {"result": param}

    tools = registry.list_tools()
    assert len(tools) == 1
    assert tools[0]["description"] == "Process a parameter value."


@pytest.mark.unit
def test_register_tool_empty_docstring():
    """Test that empty docstring results in empty description."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def my_tool(param: str) -> dict:
        return {"result": param}

    tools = registry.list_tools()
    assert len(tools) == 1
    assert tools[0]["description"] == ""


# MCPToolRegistry._generate_schema() tests


@pytest.mark.unit
def test_generate_schema_basic_types():
    """Test schema generation for basic types (str, int, bool, float)."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def basic_types(name: str, age: int, active: bool, score: float) -> dict:
        return {"name": name, "age": age, "active": active, "score": score}

    tools = registry.list_tools()
    schema = cast(dict[str, object], tools[0]["inputSchema"])
    props = cast(dict[str, object], schema["properties"])

    assert schema["type"] == "object"
    assert "name" in props
    assert "age" in props
    assert "active" in props
    assert "score" in props
    assert cast(dict[str, object], props["name"])["type"] == "string"
    assert cast(dict[str, object], props["age"])["type"] == "integer"
    assert cast(dict[str, object], props["active"])["type"] == "boolean"
    assert cast(dict[str, object], props["score"])["type"] == "number"


@pytest.mark.unit
def test_generate_schema_optional_parameters_with_defaults():
    """Test optional parameters with defaults become optional in schema."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def with_defaults(name: str, greeting: str = "Hello") -> str:
        return f"{greeting}, {name}"

    tools = registry.list_tools()
    schema = cast(dict[str, object], tools[0]["inputSchema"])
    required = cast(list[object], schema["required"])
    props = cast(dict[str, object], schema["properties"])

    assert "name" in required
    assert "greeting" not in required
    assert cast(dict[str, object], props["greeting"])["default"] == "Hello"


@pytest.mark.unit
def test_generate_schema_required_parameters_without_defaults():
    """Test required parameters without defaults are marked as required."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def all_required(param1: str, param2: int, param3: bool) -> dict:
        return {"param1": param1, "param2": param2, "param3": param3}

    tools = registry.list_tools()
    schema = cast(dict[str, object], tools[0]["inputSchema"])

    assert schema["required"] == ["param1", "param2", "param3"]


@pytest.mark.unit
def test_generate_schema_complex_types():
    """Test schema generation for complex types (list[str], dict[str, int])."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def complex_types(tags: list[str], scores: dict[str, int]) -> dict:
        return {"tags": tags, "scores": scores}

    tools = registry.list_tools()
    schema = cast(dict[str, object], tools[0]["inputSchema"])
    props = cast(dict[str, object], schema["properties"])
    tags_schema = cast(dict[str, object], props["tags"])
    items_schema = cast(dict[str, object], tags_schema["items"])

    assert "tags" in props
    assert "scores" in props
    assert tags_schema["type"] == "array"
    assert items_schema["type"] == "string"
    assert cast(dict[str, object], props["scores"])["type"] == "object"


@pytest.mark.unit
def test_generate_schema_literal_types():
    """Test Literal types become enum in schema."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def with_literal(action: Literal["create", "update", "delete"]) -> dict:
        return {"action": action}

    tools = registry.list_tools()
    schema = cast(dict[str, object], tools[0]["inputSchema"])
    props = cast(dict[str, object], schema["properties"])
    action_schema = cast(dict[str, object], props["action"])

    assert "action" in props
    assert "enum" in action_schema
    assert set(cast(list[str], action_schema["enum"])) == {
        "create",
        "update",
        "delete",
    }


@pytest.mark.unit
def test_generate_schema_union_types_with_pipe():
    """Test union types with pipe operator (str | int)."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def with_union(value: str | int) -> dict:
        return {"value": value}

    tools = registry.list_tools()
    schema = cast(dict[str, object], tools[0]["inputSchema"])
    props = cast(dict[str, object], schema["properties"])
    value_schema = cast(dict[str, object], props["value"])

    assert "value" in props
    # Union types generate anyOf in JSON schema
    assert "anyOf" in value_schema


@pytest.mark.unit
def test_generate_schema_no_parameters():
    """Test functions with no parameters return empty properties."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def no_params() -> dict:
        return {"status": "ok"}

    tools = registry.list_tools()
    schema = cast(dict[str, object], tools[0]["inputSchema"])

    assert schema["type"] == "object"
    assert schema["properties"] == {}
    assert schema["required"] == []


@pytest.mark.unit
def test_generate_schema_filters_depends_parameters():
    """Test that FastAPI Depends() parameters are filtered out."""
    registry = MCPToolRegistry()

    def get_user() -> str:
        return "test_user"

    # Must use explicit schema when using Depends
    @registry.tool(
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        }
    )
    async def with_depends(message: str, user: str = Depends(get_user)) -> dict:
        return {"message": message, "user": user}

    tools = registry.list_tools()
    schema = cast(dict[str, object], tools[0]["inputSchema"])
    props = cast(dict[str, object], schema["properties"])

    # Schema should only include message, not user (dependency)
    assert "message" in props
    assert "user" not in props


@pytest.mark.unit
def test_generate_schema_mixed_required_and_optional():
    """Test mixed required and optional parameters."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def mixed_params(
        required1: str,
        optional1: int = 10,
        required2: object = ...,
        optional2: str = "default",
    ) -> dict:
        return {
            "required1": required1,
            "optional1": optional1,
            "required2": required2,
            "optional2": optional2,
        }

    tools = registry.list_tools()
    schema = cast(dict[str, object], tools[0]["inputSchema"])
    required = cast(list[object], schema["required"])

    # required1 has no default - required
    # optional1 has default - optional
    # required2 has ... (Ellipsis) as default but may vary
    # optional2 has default - optional
    assert "required1" in required
    assert "optional1" not in required
    assert "optional2" not in required


# MCPToolRegistry.list_tools() tests


@pytest.mark.unit
def test_list_tools_empty_registry():
    """Test listing tools from empty registry returns empty list."""
    registry = MCPToolRegistry()

    tools = registry.list_tools()

    assert tools == []
    assert isinstance(tools, list)
    assert len(tools) == 0


@pytest.mark.unit
def test_list_tools_single_tool():
    """Test listing single registered tool."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def single_tool(param: str) -> str:
        """Single tool description."""
        return param

    tools = registry.list_tools()

    assert len(tools) == 1
    assert tools[0]["name"] == "single_tool"
    assert tools[0]["description"] == "Single tool description."
    assert "inputSchema" in tools[0]


@pytest.mark.unit
def test_list_tools_multiple_tools():
    """Test listing multiple registered tools."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def tool_one(param1: str) -> str:
        """First tool."""
        return param1

    @registry.tool()
    async def tool_two(param2: int) -> int:
        """Second tool."""
        return param2

    @registry.tool()
    async def tool_three(param3: bool) -> bool:
        """Third tool."""
        return param3

    tools = registry.list_tools()

    assert len(tools) == 3
    tool_names = {tool["name"] for tool in tools}
    assert tool_names == {"tool_one", "tool_two", "tool_three"}


@pytest.mark.unit
def test_list_tools_correct_format():
    """Test that tools have correct format (name, description, inputSchema)."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def formatted_tool(value: str) -> dict:
        """Tool description."""
        return {"value": value}

    tools = registry.list_tools()

    assert len(tools) == 1
    tool = tools[0]
    assert "name" in tool
    assert "description" in tool
    assert "inputSchema" in tool
    assert isinstance(tool["name"], str)
    assert isinstance(tool["description"], str)
    assert isinstance(tool["inputSchema"], dict)


# MCPToolRegistry.call_tool() tests


@pytest.mark.unit
@pytest.mark.asyncio
async def test_call_tool_not_found():
    """Test calling tool that doesn't exist raises MCPError -32601."""
    registry = MCPToolRegistry()

    with pytest.raises(MCPError) as exc_info:
        await registry.call_tool("nonexistent", {})

    assert exc_info.value.code == -32601
    assert "Tool not found" in exc_info.value.message
    assert "nonexistent" in exc_info.value.message


@pytest.mark.unit
@pytest.mark.asyncio
async def test_call_tool_success():
    """Test calling tool with correct arguments."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def add(a: int, b: int) -> int:
        return a + b

    result = await registry.call_tool("add", {"a": 5, "b": 3})

    assert result == 8


@pytest.mark.unit
@pytest.mark.asyncio
async def test_call_tool_missing_required_argument():
    """Test calling tool with missing required argument raises MCPError -32602."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def requires_param(required: str) -> str:
        return required

    with pytest.raises(MCPError) as exc_info:
        await registry.call_tool("requires_param", {})

    assert exc_info.value.code == -32602
    assert "Invalid arguments" in exc_info.value.message


@pytest.mark.unit
@pytest.mark.asyncio
async def test_call_tool_unexpected_argument():
    """Test calling tool with unexpected argument raises MCPError -32602."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def simple_tool(expected: str) -> str:
        return expected

    with pytest.raises(MCPError) as exc_info:
        await registry.call_tool("simple_tool", {"expected": "value", "unexpected": "extra"})

    assert exc_info.value.code == -32602
    assert "Invalid arguments" in exc_info.value.message


@pytest.mark.unit
@pytest.mark.asyncio
async def test_call_tool_reraises_mcperror():
    """Test calling tool that raises MCPError re-raises without wrapping."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def raises_mcp() -> dict:
        raise MCPError(code=-32001, message="Custom MCP error")

    with pytest.raises(MCPError) as exc_info:
        await registry.call_tool("raises_mcp", {})

    # Should be the original MCPError, not wrapped
    assert exc_info.value.code == -32001
    assert exc_info.value.message == "Custom MCP error"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_call_tool_reraises_toolerror():
    """Test calling tool that raises ToolError re-raises without wrapping.

    Note: ToolError is handled at the router layer (not registry layer).
    The registry re-raises ToolError without wrapping so the router can
    convert it to isError: true response.
    """
    registry = MCPToolRegistry()

    @registry.tool()
    async def raises_tool() -> dict:
        raise ToolError(message="File not found", details={"path": "/test/file.txt"})

    with pytest.raises(ToolError) as exc_info:
        await registry.call_tool("raises_tool", {})

    # ToolError is re-raised without wrapping
    assert exc_info.value.message == "File not found"
    assert exc_info.value.details == {"path": "/test/file.txt"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_call_tool_wraps_other_exceptions():
    """Test calling tool that raises other Exception wraps in MCPError -32603."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def raises_generic() -> dict:
        raise ValueError("Something went wrong")

    with pytest.raises(MCPError) as exc_info:
        await registry.call_tool("raises_generic", {})

    assert exc_info.value.code == -32603
    assert "Tool execution failed" in exc_info.value.message
    assert "Something went wrong" in exc_info.value.message


# Additional integration tests


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_with_optional_parameters():
    """Test calling tool with optional parameters."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def greet(name: str, excited: bool = False) -> dict:
        greeting = f"Hello, {name}!" if not excited else f"Hello, {name}!!!"
        return {"message": greeting}

    # Call with only required parameter
    result1 = cast(dict[str, object], await registry.call_tool("greet", {"name": "Alice"}))
    assert result1["message"] == "Hello, Alice!"

    # Call with all parameters
    result2 = cast(dict[str, object], await registry.call_tool("greet", {"name": "Bob", "excited": True}))
    assert result2["message"] == "Hello, Bob!!!"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_with_complex_return_type():
    """Test tool with complex return type."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def get_data(count: int) -> dict[str, object]:
        return {
            "items": [f"item_{i}" for i in range(count)],
            "total": count,
            "metadata": {"generated": True},
        }

    result = cast(dict[str, object], await registry.call_tool("get_data", {"count": 3}))

    assert result["items"] == ["item_0", "item_1", "item_2"]
    assert result["total"] == 3
    assert cast(dict[str, object], result["metadata"])["generated"] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_with_literal_parameter():
    """Test tool with Literal parameter type."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def create_task(action: Literal["create", "update", "delete"], priority: int = 1) -> dict[str, object]:
        return {"action": action, "priority": priority}

    result = cast(
        dict[str, object],
        await registry.call_tool("create_task", {"action": "create", "priority": 5}),
    )

    assert result["action"] == "create"
    assert result["priority"] == 5


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_with_list_parameter():
    """Test tool with list parameter."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def sum_values(values: list[int]) -> int:
        return sum(values)

    result = await registry.call_tool("sum_values", {"values": [1, 2, 3, 4, 5]})

    assert result == 15


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_with_dict_parameter():
    """Test tool with dict parameter."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def process_config(config: dict[str, object]) -> dict[str, object]:
        return {"processed": True, "config": config}

    result = cast(
        dict[str, object],
        await registry.call_tool("process_config", {"config": {"key": "value", "number": 42}}),
    )
    config = cast(dict[str, object], result["config"])

    assert result["processed"] is True
    assert config["key"] == "value"
    assert config["number"] == 42


@pytest.mark.unit
def test_multiple_registries_independent():
    """Test multiple registries are independent."""
    registry1 = MCPToolRegistry()
    registry2 = MCPToolRegistry()

    @registry1.tool()
    async def tool1(param: str) -> str:
        return param

    @registry2.tool()
    async def tool2(param: str) -> str:
        return param

    tools1 = registry1.list_tools()
    tools2 = registry2.list_tools()

    assert len(tools1) == 1
    assert len(tools2) == 1
    assert tools1[0]["name"] == "tool1"
    assert tools2[0]["name"] == "tool2"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_with_none_default():
    """Test tool with None as default value."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def optional_param(required: str, optional: str | None = None) -> dict:
        return {"required": required, "optional": optional}

    # Call without optional parameter
    result1 = cast(dict[str, object], await registry.call_tool("optional_param", {"required": "test"}))
    assert result1["required"] == "test"
    assert result1["optional"] is None

    # Call with optional parameter
    result2 = cast(
        dict[str, object],
        await registry.call_tool("optional_param", {"required": "test", "optional": "value"}),
    )
    assert result2["required"] == "test"
    assert result2["optional"] == "value"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_call_tool_preserves_return_types():
    """Test that call_tool preserves various return types."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def return_string() -> str:
        return "hello"

    @registry.tool()
    async def return_int() -> int:
        return 42

    @registry.tool()
    async def return_bool() -> bool:
        return True

    @registry.tool()
    async def return_dict() -> dict:
        return {"key": "value"}

    @registry.tool()
    async def return_list() -> list:
        return [1, 2, 3]

    assert await registry.call_tool("return_string", {}) == "hello"
    assert await registry.call_tool("return_int", {}) == 42
    assert await registry.call_tool("return_bool", {}) is True
    assert await registry.call_tool("return_dict", {}) == {"key": "value"}
    assert await registry.call_tool("return_list", {}) == [1, 2, 3]


# IR-21: varargs schema exclusion


@pytest.mark.unit
def test_register_tool_with_varargs_schema_excludes_varargs():
    """Test *args and **kwargs are excluded from generated schema properties."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def my_tool(*args, **kwargs):
        pass

    tool = registry._tools["my_tool"]
    properties = cast(dict[str, object], tool.input_schema.get("properties", {}))
    assert "args" not in properties
    assert "kwargs" not in properties


# IR-22: unannotated param defaults to string


@pytest.mark.unit
def test_register_tool_with_unannotated_param_defaults_to_str():
    """Test that a parameter with no type annotation defaults to string in schema."""
    registry = MCPToolRegistry()

    @registry.tool()
    async def my_tool(x) -> dict:
        return {"x": x}

    tools = registry.list_tools()
    schema = cast(dict[str, object], tools[0]["inputSchema"])
    props = cast(dict[str, object], schema["properties"])
    assert cast(dict[str, object], props["x"])["type"] == "string"


# IR-23: duplicate tool name overwrites first registration


@pytest.mark.unit
def test_register_duplicate_tool_overwrites_first():
    """Test registering two tools with the same name keeps only the second handler."""
    registry = MCPToolRegistry()

    @registry.tool(name="same_name")
    async def first_handler() -> str:
        return "first"

    @registry.tool(name="same_name")
    async def second_handler() -> str:
        return "second"

    assert len(registry._tools) == 1
    assert registry._tools["same_name"].handler is second_handler


# IR-24: Depends(None) does not raise AttributeError


@pytest.mark.unit
@pytest.mark.asyncio
async def test_call_tool_with_none_dependency_continues():
    """Test that Depends(None) is skipped at call time without AttributeError."""
    registry = MCPToolRegistry()

    @registry.tool(
        input_schema={
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
    )
    async def my_tool(x: int, dep=Depends(None)) -> int:
        return x

    result = await registry.call_tool("my_tool", {"x": 1}, request=None, background_tasks=None)
    assert result == 1


# IR-25: bound method filters self from schema


@pytest.mark.unit
def test_register_class_method_filters_self_from_schema():
    """Test that registering a bound method excludes 'self' from schema properties."""
    registry = MCPToolRegistry()

    class MyService:
        async def my_method(self, value: str) -> str:
            return value

    service = MyService()

    @registry.tool()
    async def my_method(value: str) -> str:
        return await service.my_method(value)

    tools = registry.list_tools()
    schema = cast(dict[str, object], tools[0]["inputSchema"])
    props = cast(dict[str, object], schema.get("properties", {}))
    assert "self" not in props


# Phase 3 gap-closure tests


# Line 42: _is_type_or_contains_type __name__ string-match branch


@pytest.mark.unit
def test_is_type_or_contains_type_name_match():
    """Test _is_type_or_contains_type returns True when __name__ matches type_name.

    Line 42 fires when the annotation is not the same object as target_type
    but has a __name__ attribute equal to type_name.
    """
    from fastapi_mcp_router.registry import _is_type_or_contains_type

    # Create a fake class with matching __name__ but different identity
    class Request:
        pass

    from fastapi import Request as RealRequest

    assert Request is not RealRequest
    result = _is_type_or_contains_type(Request, RealRequest, "Request")
    assert result is True


# Line 349: _generate_schema continue for "self" parameter on unbound method


@pytest.mark.unit
def test_generate_schema_filters_self_from_unbound_method():
    """Test _generate_schema skips 'self' when called on an unbound method.

    Line 349 fires when sig.parameters contains 'self'; the unbound method
    path exposes self in inspect.signature unlike a wrapper async def.
    """
    registry = MCPToolRegistry()

    class MyService:
        async def handle(self, value: str) -> str:
            return value

    # Calling _generate_schema on the unbound method includes "self" in sig
    schema = registry._generate_schema(MyService.handle)
    props = cast(dict[str, object], schema.get("properties", {}))
    assert "self" not in props
    assert "value" in props


# Line 355: _generate_schema continue for Depends() default parameter


@pytest.mark.unit
def test_generate_schema_filters_depends_parameter():
    """Test _generate_schema skips parameters with Depends() defaults.

    Line 355 fires when param.default.__class__.__name__ == 'Depends';
    this confirms the continue on line 355 is reached.
    """
    registry = MCPToolRegistry()

    async def my_func(value: str, user: str = Depends(lambda: "anon")) -> str:
        return value

    schema = registry._generate_schema(my_func)
    props = cast(dict[str, object], schema.get("properties", {}))
    assert "value" in props
    assert "user" not in props


# Lines 578-580: call_tool raises MCPError when required Request param has no request


@pytest.mark.unit
@pytest.mark.asyncio
async def test_call_tool_raises_mcp_error_when_request_required_but_none():
    """Test call_tool raises MCPError(-32603) when a required Request param has no request.

    Lines 578-580 fire when param annotation matches Request, request is None,
    and param.default is Parameter.empty (no default provided).
    """
    from fastapi import Request

    registry = MCPToolRegistry()

    @registry.tool(input_schema={"type": "object", "properties": {}, "required": []})
    async def needs_request(req: Request) -> str:
        return "ok"

    with pytest.raises(MCPError) as exc_info:
        await registry.call_tool("needs_request", {}, request=None, background_tasks=None)

    assert exc_info.value.code == -32603
    assert "requires Request" in exc_info.value.message


# AC-41: ProgressCallback injection behavior


@pytest.mark.unit
def test_progress_callback_filtered_from_schema():
    """Test that progress: ProgressCallback is not included in generated inputSchema.

    The ProgressCallback parameter is injected at call time; clients never
    supply it. It must be absent from the tool's inputSchema properties.
    """
    from fastapi_mcp_router import ProgressCallback

    registry = MCPToolRegistry()

    @registry.tool()
    async def tool_with_progress(value: str, progress: ProgressCallback) -> str:
        """Tool that accepts a progress callback."""
        return value

    tools = registry.list_tools()
    schema = cast(dict[str, object], tools[0]["inputSchema"])
    props = cast(dict[str, object], schema["properties"])

    assert "value" in props
    assert "progress" not in props


@pytest.mark.unit
@pytest.mark.asyncio
async def test_call_tool_injects_progress_callback_when_provided():
    """Test call_tool injects a supplied progress_callback into the tool handler.

    When progress_callback is provided and the tool has a progress parameter,
    the callable is forwarded so the tool can call it.
    """
    from fastapi_mcp_router import ProgressCallback

    registry = MCPToolRegistry()
    received: list[object] = []

    @registry.tool()
    async def tool_with_progress(progress: ProgressCallback) -> str:
        """Tool that captures its injected progress callable."""
        received.append(progress)
        return "done"

    async def my_callback(current: int, total: int, message: str | None) -> None:
        pass

    await registry.call_tool("tool_with_progress", {}, progress_callback=my_callback)

    assert len(received) == 1
    assert received[0] is my_callback


@pytest.mark.unit
@pytest.mark.asyncio
async def test_call_tool_injects_noop_progress_callback_when_none():
    """Test call_tool injects a no-op async callable when progress_callback is None.

    The tool must receive an awaitable callable it can call unconditionally,
    even when the caller passes no progress_callback.
    """
    from fastapi_mcp_router import ProgressCallback

    registry = MCPToolRegistry()
    received: list[object] = []

    @registry.tool()
    async def tool_with_progress(progress: ProgressCallback) -> str:
        """Tool that captures its injected progress callable."""
        received.append(progress)
        await progress(1, 10, None)
        return "done"

    result = await registry.call_tool("tool_with_progress", {}, progress_callback=None)

    assert result == "done"
    assert len(received) == 1
    # No-op must be async callable (awaitable)
    import asyncio

    assert asyncio.iscoroutinefunction(received[0])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_call_tool_progress_callback_ignored_without_progress_param():
    """Test passing progress_callback does not affect tools without a progress parameter.

    Tools that have no ProgressCallback parameter must execute normally
    regardless of whether progress_callback is supplied.
    """
    registry = MCPToolRegistry()

    @registry.tool()
    async def simple_tool(value: str) -> str:
        """Tool without a progress parameter."""
        return value.upper()

    async def my_callback(current: int, total: int, message: str | None) -> None:
        pass

    result = await registry.call_tool(
        "simple_tool",
        {"value": "hello"},
        progress_callback=my_callback,
    )

    assert result == "HELLO"


# Line 625: TypeError with non-argument message raises MCPError(-32603)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_call_tool_raises_mcp_error_on_non_argument_type_error():
    """Test call_tool raises MCPError(-32603) for TypeError unrelated to arguments.

    Line 625 fires when a TypeError message does not contain 'missing',
    'unexpected', or 'argument', indicating an internal type error.
    """
    registry = MCPToolRegistry()

    @registry.tool(input_schema={"type": "object", "properties": {}, "required": []})
    async def bad_tool() -> str:
        raise TypeError("type mismatch in computation")

    with pytest.raises(MCPError) as exc_info:
        await registry.call_tool("bad_tool", {}, request=None, background_tasks=None)

    assert exc_info.value.code == -32603
    assert "Tool execution failed" in exc_info.value.message
