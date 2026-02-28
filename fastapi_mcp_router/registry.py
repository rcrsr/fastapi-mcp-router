"""Tool registry for MCP tool registration and management.

This module provides the core registry functionality for registering and managing
MCP tools. It includes automatic JSON schema generation from function signatures
using Pydantic TypeAdapter, supporting all Python type hints.

Classes:
    ToolDefinition: Internal storage for tool metadata
    MCPToolRegistry: Main registry for tool registration and execution
"""

import json
import types
from collections.abc import AsyncGenerator, Callable
from inspect import Parameter, signature
from typing import TypeVar, get_args, get_origin, get_type_hints

from pydantic import BaseModel, TypeAdapter

from fastapi_mcp_router.exceptions import MCPError, ToolError

F = TypeVar("F", bound=types.FunctionType)


def _is_type_or_contains_type(annotation: object, target_type: type, type_name: str) -> bool:
    """Check if annotation is target_type or contains it in a Union.

    Handles direct types, Union types (X | Y), and Optional types (X | None).
    Uses both identity check and string name comparison for robustness.

    Args:
        annotation: Type annotation to check
        target_type: Type class to search for (e.g., Request, BackgroundTasks)
        type_name: Simple type name for string comparison (e.g., "Request")

    Returns:
        True if annotation matches target_type or contains it in a Union
    """
    # Direct type match
    if annotation is target_type:
        return True

    # String name match (handles different import sources)
    if hasattr(annotation, "__name__") and annotation.__name__ == type_name:
        return True

    # Check for Union types (X | Y or Optional[X])
    origin = get_origin(annotation)
    if origin is not None:
        # For Union types, check all args
        args = get_args(annotation)
        for arg in args:
            # Skip NoneType in Union
            if arg is type(None):
                continue
            # Recursively check each union member
            if _is_type_or_contains_type(arg, target_type, type_name):
                return True

    return False


def _is_async_generator_annotation(hint: object) -> bool:
    """Check if a type hint is AsyncGenerator[dict, None].

    Accepts both collections.abc.AsyncGenerator and typing.AsyncGenerator
    as the origin. Only returns True when the first type argument is dict.

    Args:
        hint: Return type annotation to inspect

    Returns:
        True if hint is AsyncGenerator[dict, None]
    """
    if hint is None:
        return False

    origin = get_origin(hint)
    if origin is None:
        return False

    # Accept both collections.abc.AsyncGenerator and typing.AsyncGenerator
    # origins (they differ across Python versions and import paths).
    import typing as _typing

    if origin not in (AsyncGenerator, _typing.AsyncGenerator):
        return False

    args = get_args(hint)
    # AsyncGenerator[YieldType, SendType] — we require YieldType == dict
    if not args:
        return False

    return args[0] is dict


def _is_progress_callback_annotation(annotation: object) -> bool:
    """Check if an annotation is the ProgressCallback type alias.

    Performs identity comparison against the ProgressCallback type alias
    from fastapi_mcp_router.types. Import is deferred inside the function
    body to respect the registry → exceptions-only import constraint at
    module level (§LIB.3.1); types.py is a leaf module with no internal
    imports so the deferred import introduces no circular risk.

    Args:
        annotation: Type annotation to check

    Returns:
        True if annotation is exactly ProgressCallback
    """
    # Deferred import: types.py is a leaf module (no internal imports).
    # Module-level import of types.py is forbidden by §LIB.3.1.
    from fastapi_mcp_router.types import ProgressCallback

    return annotation is ProgressCallback


def _is_sampling_manager_annotation(annotation: object) -> bool:
    """Check if an annotation is the SamplingManager class.

    Performs identity comparison against SamplingManager from
    fastapi_mcp_router.session. Import is deferred inside the function
    body to avoid a circular import at module level (§LIB.3.1); session.py
    imports exceptions.py (a leaf module) only, so the deferred import
    introduces no circular risk.

    Args:
        annotation: Type annotation to check

    Returns:
        True if annotation is SamplingManager
    """
    # Deferred import: session.py imports only exceptions (leaf module).
    # Module-level import of session.py is forbidden by §LIB.3.1.
    from fastapi_mcp_router.session import SamplingManager

    return annotation is SamplingManager


class ToolDefinition:
    """Internal storage for tool metadata.

    Stores complete tool information including name, description, JSON schema,
    annotations, the async handler function, generator flag, and output schema.

    Attributes:
        name: Tool name identifier
        description: Human-readable tool description
        input_schema: JSON schema for tool parameters
        annotations: Optional MCP annotations for tool capabilities
        handler: Async function implementing tool logic
        is_generator: True when handler is an AsyncGenerator[dict, None]
        output_schema: Optional JSON schema describing structured tool output
    """

    name: str
    description: str
    input_schema: dict[str, object]
    annotations: dict[str, object] | None
    handler: Callable
    is_generator: bool
    output_schema: dict[str, object] | None

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict[str, object],
        handler: Callable,
        annotations: dict[str, object] | None = None,
        is_generator: bool = False,
        output_schema: dict[str, object] | None = None,
    ) -> None:
        """Initialize tool definition.

        Args:
            name: Tool name identifier
            description: Human-readable tool description
            input_schema: JSON schema for tool parameters
            handler: Async function implementing tool logic
            annotations: Optional MCP annotations for tool capabilities
            is_generator: True when handler yields via AsyncGenerator[dict, None]
            output_schema: Optional JSON schema describing structured tool output
        """
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.annotations = annotations
        self.handler = handler
        self.is_generator = is_generator
        self.output_schema = output_schema


class MCPToolRegistry:
    """Registry for MCP tool registration and management.

    Provides decorator-based tool registration with automatic JSON schema
    generation from function signatures using Pydantic TypeAdapter. Supports
    all Python type hints including Optional, Literal, Union, nested models,
    and complex generic types.

    The registry handles:
    - Tool registration via @tools.tool() decorator
    - Automatic schema generation from type hints
    - FastAPI Depends() parameter filtering
    - Tool discovery via list_tools()
    - Tool execution via call_tool()

    Example:
        >>> tools = MCPToolRegistry()
        >>>
        >>> @tools.tool()
        >>> async def greet(name: str, excited: bool = False) -> dict[str, str]:
        ...     '''Greet a user by name.'''
        ...     greeting = f"Hello, {name}!" if not excited else f"Hello, {name}!!!"
        ...     return {"message": greeting}
        >>>
        >>> # List registered tools
        >>> tool_list = tools.list_tools()
        >>> print(tool_list[0]["name"])
        greet
        >>>
        >>> # Execute tool
        >>> import asyncio
        >>> result = asyncio.run(tools.call_tool("greet", {"name": "Alice", "excited": True}))
        >>> print(result["message"])
        Hello, Alice!!!

    Example with complex types:
        >>> from typing import Literal
        >>>
        >>> @tools.tool()
        >>> async def create_task(
        ...     action: Literal["create", "update", "delete"],
        ...     priority: int = 1,
        ...     tags: list[str] | None = None
        ... ) -> dict[str, object]:
        ...     '''Create or modify a task.'''
        ...     return {"action": action, "priority": priority, "tags": tags or []}
        >>>
        >>> # Auto-generated schema includes enum for Literal type
        >>> schema = tools.list_tools()[0]["inputSchema"]
        >>> print(schema["properties"]["action"]["enum"])
        ['create', 'update', 'delete']

    Example with FastAPI dependencies (requires explicit schema):
        >>> from fastapi import Depends
        >>>
        >>> def get_user() -> str:
        ...     return "authenticated_user"
        >>>
        >>> @tools.tool(
        ...     input_schema={
        ...         "type": "object",
        ...         "properties": {"message": {"type": "string"}},
        ...         "required": ["message"]
        ...     }
        ... )
        >>> async def send_message(
        ...     message: str,
        ...     user: str = Depends(get_user)  # Filtered from schema
        ... ) -> dict[str, str]:
        ...     '''Send a message as authenticated user.'''
        ...     return {"from": user, "message": message}
    """

    _tools: dict[str, ToolDefinition]

    def __init__(self) -> None:
        """Initialize empty tool registry."""
        self._tools = {}

    def tool(
        self,
        name: str | None = None,
        description: str | None = None,
        input_schema: dict[str, object] | None = None,
        annotations: dict[str, object] | None = None,
        output_schema: dict[str, object] | None = None,
    ) -> Callable:
        """Register function as MCP tool.

        Decorator for registering async functions as MCP tools. Automatically
        generates JSON schema from function signature unless explicit schema
        provided. Validates function is async and stores tool metadata.

        Args:
            name: Tool name (defaults to function name)
            description: Tool description (defaults to function docstring)
            input_schema: JSON schema for parameters (auto-generated if not provided)
            annotations: MCP annotations for tool capabilities (e.g., {"readOnlyHint": True})
            output_schema: JSON schema describing structured tool output. When set,
                outputSchema is included in tools/list and call_tool returns
                structuredContent alongside the backward-compatible text content.

        Returns:
            Decorator function that returns original function unchanged

        Raises:
            TypeError: If decorated function is not async

        Example:
            >>> @tools.tool()
            >>> async def simple_tool(value: str) -> str:
            ...     '''Process a value.'''
            ...     return value.upper()

        Example with custom name and description:
            >>> @tools.tool(
            ...     name="custom_name",
            ...     description="Custom description"
            ... )
            >>> async def my_tool() -> dict:
            ...     return {"status": "ok"}

        Example with annotations (MCP 2025-06-18):
            >>> @tools.tool(annotations={"readOnlyHint": True})
            >>> async def get_data() -> dict:
            ...     '''Retrieve data without side effects.'''
            ...     return {"data": "value"}

        Example with explicit schema (required for FastAPI dependencies):
            >>> @tools.tool(
            ...     input_schema={
            ...         "type": "object",
            ...         "properties": {"count": {"type": "integer"}},
            ...         "required": ["count"]
            ...     }
            ... )
            >>> async def count_items(
            ...     count: int,
            ...     db: Database = Depends(get_db)
            ... ) -> dict:
            ...     return {"total": count}

        Example with output schema (MCP 2025-06-18 structured content):
            >>> @tools.tool(
            ...     output_schema={
            ...         "type": "object",
            ...         "properties": {"score": {"type": "number"}},
            ...         "required": ["score"]
            ...     }
            ... )
            >>> async def analyze(text: str) -> dict:
            ...     '''Analyze text and return a score.'''
            ...     return {"score": 0.95}
        """

        def decorator(func: F) -> F:
            # Import here to avoid circular dependency
            from asyncio import iscoroutinefunction
            from inspect import isasyncgenfunction

            # Validate function is async (coroutine or async generator)
            if not iscoroutinefunction(func) and not isasyncgenfunction(func):
                raise TypeError(f"Tool function {func.__name__} must be async. Add 'async def' to function definition.")

            tool_name = name or func.__name__
            tool_description = description or (func.__doc__ or "").strip()

            # Auto-generate input schema from function signature
            if input_schema is None:
                schema = self._generate_schema(func)
            else:
                schema = input_schema

            # Detect AsyncGenerator[dict, None] return annotation
            try:
                hints = get_type_hints(func)
                return_hint = hints.get("return")
            except Exception:
                return_hint = None
            is_gen = _is_async_generator_annotation(return_hint)

            # Store tool definition
            self._tools[tool_name] = ToolDefinition(
                name=tool_name,
                description=tool_description,
                input_schema=schema,
                handler=func,
                annotations=annotations,
                is_generator=is_gen,
                output_schema=output_schema,
            )

            return func

        return decorator

    def _generate_schema(self, func: types.FunctionType) -> dict[str, object]:
        """Generate JSON schema from function signature using Pydantic TypeAdapter.

        Uses Pydantic TypeAdapter to generate JSON schema from function type hints.
        Supports all Python type hints including:
        - Basic types: str, int, bool, float
        - Generics: list[T], dict[K,V], tuple[T, ...]
        - Optional types: T | None
        - Literal types: Literal["a", "b", "c"]
        - Union types: str | int
        - Nested models: Pydantic BaseModel subclasses

        Special parameter handling:
        - Skips self, cls parameters
        - Skips *args, **kwargs parameters
        - Skips FastAPI Depends() parameters (not part of MCP interface)
        - Parameters without defaults are required
        - Parameters with defaults are optional

        Args:
            func: Function to generate schema for

        Returns:
            JSON schema dict with type, properties, and required fields

        Example:
            >>> async def simple(name: str, count: int = 1) -> dict:
            ...     return {"name": name, "count": count}
            >>>
            >>> schema = registry._generate_schema(simple)
            >>> print(schema)
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "count": {"type": "integer", "default": 1}
                },
                "required": ["name"]
            }

        Example with complex types:
            >>> from typing import Literal
            >>>
            >>> async def complex(
            ...     action: Literal["create", "update"],
            ...     tags: list[str] | None = None
            ... ) -> dict:
            ...     return {}
            >>>
            >>> schema = registry._generate_schema(complex)
            >>> print(schema["properties"]["action"]["enum"])
            ['create', 'update']

        Example with FastAPI dependency (filtered from schema):
            >>> from fastapi import Depends
            >>>
            >>> async def with_dep(
            ...     message: str,
            ...     user: str = Depends(lambda: "user")
            ... ) -> dict:
            ...     return {}
            >>>
            >>> schema = registry._generate_schema(with_dep)
            >>> print("user" in schema["properties"])
            False
        """
        sig = signature(func)

        # Build dictionary of parameter types
        param_types: dict[str, object] = {}
        param_defaults: dict[str, object] = {}

        for param_name, param in sig.parameters.items():
            # Skip special parameters (self, cls, *args, **kwargs)
            if param_name in ("self", "cls"):
                continue
            if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
                continue

            # Skip FastAPI dependencies
            if hasattr(param.default, "__class__") and param.default.__class__.__name__ == "Depends":
                continue

            # Skip ProgressCallback parameters (injected at call time, not from client)
            if _is_progress_callback_annotation(param.annotation):
                continue

            # Skip SamplingManager parameters (injected at call time, not from client)
            if _is_sampling_manager_annotation(param.annotation):
                continue

            # Extract type annotation
            param_type = param.annotation
            if param_type is Parameter.empty:
                param_type = str  # Default to string

            param_types[param_name] = param_type

            # Track if parameter has default
            if param.default is not Parameter.empty:
                param_defaults[param_name] = param.default

        # No parameters case
        if not param_types:
            return {"type": "object", "properties": {}, "required": []}

        # Create a temporary model class for all parameters
        fields: dict[str, tuple[object, object]] = {}
        for param_name, param_type in param_types.items():
            if param_name in param_defaults:
                # Optional parameter with default
                fields[param_name] = (param_type, param_defaults[param_name])
            else:
                # Required parameter
                fields[param_name] = (param_type, ...)

        # Create dynamic model with all fields
        TempModel = type(
            f"{func.__name__}_params",
            (BaseModel,),
            {
                "__annotations__": {k: v[0] for k, v in fields.items()},
                **{k: v[1] for k, v in fields.items()},
            },
        )

        # Generate schema using TypeAdapter
        adapter: TypeAdapter = TypeAdapter(TempModel)
        schema = adapter.json_schema()

        # Extract parameter schema (strip model wrapper)
        return {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
        }

    def list_tools(self) -> list[dict[str, object]]:
        """List all registered tools with schemas.

        Returns list of tool definitions in MCP format with name, description,
        inputSchema, and optional annotations fields. Used by MCP protocol to
        expose available tools to clients.

        Returns:
            List of tool definition dicts with name, description, inputSchema,
            and annotations (if provided)

        Example:
            >>> tools = MCPToolRegistry()
            >>>
            >>> @tools.tool()
            >>> async def greet(name: str) -> str:
            ...     '''Greet a user.'''
            ...     return f"Hello, {name}"
            >>>
            >>> tool_list = tools.list_tools()
            >>> print(tool_list)
            [
                {
                    "name": "greet",
                    "description": "Greet a user.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"}
                        },
                        "required": ["name"]
                    }
                }
            ]

        Example with annotations:
            >>> @tools.tool(annotations={"readOnlyHint": True})
            >>> async def get_data() -> dict:
            ...     '''Retrieve data.'''
            ...     return {"data": "value"}
            >>>
            >>> tool_list = tools.list_tools()
            >>> print(tool_list[0]["annotations"])
            {'readOnlyHint': True}
        """
        result = []
        for tool in self._tools.values():
            tool_dict: dict[str, object] = {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
            }
            if tool.annotations is not None:
                tool_dict["annotations"] = tool.annotations
            if tool.output_schema is not None:
                tool_dict["outputSchema"] = tool.output_schema
            result.append(tool_dict)
        return result

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, object],
        request: object | None = None,
        background_tasks: object | None = None,
        stateful: bool = False,
        progress_callback: object | None = None,
        sampling_manager: object | None = None,
    ) -> object:
        """Call registered tool with arguments and optional injection.

        Executes registered tool handler with provided arguments. Automatically
        injects FastAPI Request, BackgroundTasks, ProgressCallback, and
        SamplingManager parameters if the tool handler signature includes them.
        This enables tools to access HTTP headers, authentication context,
        request metadata, schedule background tasks, report progress to the
        client, and issue server-to-client sampling requests.

        Generator tools (is_generator=True) are iterated to collect all yielded
        dicts into a list. Non-dict yields and generator exceptions produce a
        ToolError with isError: true so the LLM can recover. When stateful=True,
        generator tools return the raw AsyncGenerator instead of collecting.

        Request Injection:
            The registry inspects the tool handler signature for parameters typed
            as fastapi.Request. If found, the request parameter is automatically
            injected from the provided request object. This allows tools to:
            - Access HTTP headers (e.g., X-API-Key for authentication)
            - Read request metadata (method, URL, client info)
            - Use FastAPI dependency injection patterns

        BackgroundTasks Injection:
            The registry also inspects for parameters typed as fastapi.BackgroundTasks.
            If found, the background_tasks parameter is automatically injected. This
            allows tools to schedule asynchronous tasks for execution after the
            response is sent.

        ProgressCallback Injection:
            The registry inspects for parameters typed as ProgressCallback. If found
            and progress_callback is provided, it is injected. If progress_callback
            is None, a no-op async callback is injected so the tool can call it
            unconditionally without guarding against None.

        SamplingManager Injection:
            The registry inspects for parameters typed as SamplingManager. If found
            and sampling_manager is provided, it is injected directly. Tools call
            await sampling_manager.create_message(...) to issue server-to-client
            sampling requests. When sampling_manager is None the parameter is
            omitted; the tool must declare it optional or the call raises MCPError.

        Error Handling:
            - Tool not found → MCPError(-32601)
            - Invalid arguments → MCPError(-32602)
            - Execution failure → MCPError(-32603)
            - ToolError from tool → re-raised without wrapping (router converts to isError: true)
            - MCPError from tool → re-raised without wrapping
            - Generator non-dict yield → ToolError (isError: true)
            - Generator exception → ToolError (isError: true)

        Args:
            name: Tool name to execute
            arguments: Tool arguments dict matching input schema
            request: Optional FastAPI Request object for dependency injection.
                If tool handler has Request parameter and request is None,
                raises MCPError if parameter is required.
            background_tasks: Optional FastAPI BackgroundTasks object for async task
                scheduling. Injected into tools that have a BackgroundTasks parameter.
            stateful: When True and tool is_generator, returns the raw AsyncGenerator
                instead of collecting. Caller is responsible for iteration and error
                handling. Default False (collect into list).
            progress_callback: Optional ProgressCallback for reporting tool progress.
                Injected into tools that have a progress: ProgressCallback parameter.
                When None and tool has_progress, a no-op callback is injected.
            sampling_manager: Optional SamplingManager for server-to-client LLM
                sampling. Injected into tools that have a SamplingManager parameter.
                When None, parameter is not injected.

        Returns:
            Tool execution result (any JSON-serializable value).
            For generator tools: list[dict] collected from all yields.

        Raises:
            ToolError: If generator yields non-dict or raises an exception
            ToolError: If tool raises business logic error (propagated to router)
            MCPError: If tool not found, invalid arguments, or execution fails

        Example:
            >>> tools = MCPToolRegistry()
            >>>
            >>> @tools.tool()
            >>> async def add(a: int, b: int) -> int:
            ...     return a + b
            >>>
            >>> result = await tools.call_tool("add", {"a": 5, "b": 3})
            >>> print(result)
            8

        Example with Request injection:
            >>> from fastapi import Request
            >>>
            >>> @tools.tool(
            ...     input_schema={
            ...         "type": "object",
            ...         "properties": {"message": {"type": "string"}},
            ...         "required": ["message"]
            ...     }
            ... )
            >>> async def authenticated_tool(request: Request, message: str) -> dict:
            ...     api_key = request.headers.get("x-api-key")
            ...     return {"message": message, "api_key": api_key}
            >>>
            >>> # Request object is automatically injected
            >>> result = await tools.call_tool(
            ...     "authenticated_tool",
            ...     {"message": "Hello"},
            ...     request=request_object
            ... )

        Example with error handling:
            >>> try:
            ...     result = await tools.call_tool("nonexistent", {})
            ... except MCPError as e:
            ...     print(f"Error {e.code}: {e.message}")
            Error -32601: Tool not found: nonexistent

        Example with invalid arguments:
            >>> try:
            ...     result = await tools.call_tool("add", {"a": 5})  # Missing 'b'
            ... except MCPError as e:
            ...     print(f"Error {e.code}: {e.message}")
            Error -32602: Invalid arguments: ...
        """
        if name not in self._tools:
            raise MCPError(code=-32601, message=f"Tool not found: {name}")

        tool = self._tools[name]

        # Execute tool handler with Request, BackgroundTasks, and Depends injection
        try:
            # Import here to avoid circular dependency
            from asyncio import iscoroutinefunction

            from fastapi import BackgroundTasks, Request

            sig = signature(tool.handler)
            call_kwargs = dict(arguments)

            # Process parameters for injection
            for param_name, param in sig.parameters.items():
                # Check if parameter is typed as Request (handles Union types)
                if _is_type_or_contains_type(param.annotation, Request, "Request"):
                    if request is not None:
                        call_kwargs[param_name] = request
                    elif param.default is Parameter.empty:
                        # Required Request parameter but no request provided
                        raise MCPError(
                            code=-32603,
                            message=f"Tool {name} requires Request but none provided",
                        )

                # Check if parameter is typed as BackgroundTasks (handles Union types)
                elif _is_type_or_contains_type(param.annotation, BackgroundTasks, "BackgroundTasks"):
                    if background_tasks is not None:
                        call_kwargs[param_name] = background_tasks
                    # BackgroundTasks is typically optional, so no error if None

                # Check if parameter is typed as ProgressCallback
                elif _is_progress_callback_annotation(param.annotation):
                    if progress_callback is not None:
                        call_kwargs[param_name] = progress_callback
                    else:
                        # Inject a no-op so the tool can call progress unconditionally
                        async def _noop_progress(
                            current: int,
                            total: int,
                            message: str | None,
                        ) -> None:
                            pass

                        call_kwargs[param_name] = _noop_progress

                # Check if parameter is typed as SamplingManager
                elif _is_sampling_manager_annotation(param.annotation):
                    if sampling_manager is not None:
                        call_kwargs[param_name] = sampling_manager
                    elif param.default is Parameter.empty:
                        raise MCPError(
                            code=-32601,
                            message="Sampling requires stateful mode with sampling_enabled=True",
                        )
                    # else: optional param, pass None implicitly

                # Check if parameter has Depends() default - resolve at runtime
                # Use class name check since Depends is a special marker
                elif hasattr(param.default, "__class__") and param.default.__class__.__name__ == "Depends":
                    dependency_fn = param.default.dependency
                    if dependency_fn is None:
                        continue

                    # Call dependency function (may need Request as argument)
                    dep_sig = signature(dependency_fn)
                    dep_kwargs: dict[str, object] = {}

                    # Check if dependency needs Request (handles Union types)
                    for dep_param_name, dep_param in dep_sig.parameters.items():
                        if _is_type_or_contains_type(dep_param.annotation, Request, "Request"):
                            if request is not None:
                                dep_kwargs[dep_param_name] = request

                    # Resolve dependency (sync or async)
                    if iscoroutinefunction(dependency_fn):
                        resolved = await dependency_fn(**dep_kwargs)
                    else:
                        resolved = dependency_fn(**dep_kwargs)

                    call_kwargs[param_name] = resolved

            if tool.is_generator:
                if stateful:
                    return tool.handler(**call_kwargs)
                return await self._collect_generator(tool.handler(**call_kwargs))

            result = await tool.handler(**call_kwargs)

            # AC-69: when output_schema is set, return structured content alongside
            # backward-compatible text content (MCP 2025-06-18 structuredContent field).
            if tool.output_schema is not None:
                result_text = json.dumps(result) if not isinstance(result, str) else result
                return {
                    "structuredContent": result,
                    "content": [{"type": "text", "text": result_text}],
                }

            return result
        except TypeError as e:
            # Distinguish between argument errors and internal errors
            error_msg = str(e)
            if "missing" in error_msg or "unexpected" in error_msg or "argument" in error_msg:
                raise MCPError(code=-32602, message=f"Invalid arguments: {e}") from e
            else:
                # Internal type error during execution
                raise MCPError(code=-32603, message=f"Tool execution failed: {e}") from e
        except ToolError:
            # Re-raise ToolError without wrapping
            # Let router handler convert to isError: true response
            raise
        except MCPError:
            # Re-raise MCPError without wrapping
            raise
        except Exception as e:
            raise MCPError(code=-32603, message=f"Tool execution failed: {e}") from e

    async def _collect_generator(
        self,
        gen: AsyncGenerator[dict],
    ) -> list[dict]:
        """Iterate an AsyncGenerator tool, collecting yielded dicts into a list.

        Handles EC-9 (non-dict yield) and EC-10 (generator exception) by
        raising ToolError so the LLM can recover.

        Args:
            gen: AsyncGenerator produced by the tool handler call

        Returns:
            List of dicts yielded by the generator (empty list if no yields)

        Raises:
            ToolError: If the generator yields a non-dict value
            ToolError: If the generator raises an exception during iteration
        """
        results: list[dict] = []
        try:
            async for item in gen:
                if not isinstance(item, dict):
                    raise ToolError(
                        message=f"Generator yielded non-dict value: {type(item).__name__}",
                    )
                results.append(item)
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(message=f"Generator raised an exception: {e}") from e
        return results
