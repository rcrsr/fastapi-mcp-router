"""Prompt registry for MCP prompt registration and management.

This module provides the core registry functionality for registering and managing
MCP prompts. It includes automatic argument generation from function signatures
using inspect, supporting required and optional prompt arguments.

Classes:
    PromptArgument: Data model for a single prompt argument descriptor
    PromptMessage: Data model for a single prompt message
    PromptDefinition: Internal storage for prompt metadata
    PromptRegistry: Main registry for prompt registration and execution
"""

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass

from fastapi_mcp_router.exceptions import MCPError
from fastapi_mcp_router.types import Icon

logger = logging.getLogger(__name__)


@dataclass
class PromptArgument:
    """Descriptor for a single prompt argument.

    Attributes:
        name: Argument name as it appears in the function signature
        description: Human-readable argument description
        required: True if the argument has no default value
    """

    name: str
    description: str
    required: bool


@dataclass
class PromptMessage:
    """A single message in a prompt response.

    Attributes:
        role: Speaker role; either "user" or "assistant"
        content: Text content of the message
    """

    role: str
    content: str


class PromptDefinition:
    """Internal storage for prompt metadata.

    Stores complete prompt information including name, description,
    auto-generated arguments list, and the handler function.

    Attributes:
        name: Prompt name identifier
        description: Human-readable prompt description
        arguments: List of PromptArgument descriptors auto-generated from signature
        handler: Sync or async function implementing prompt logic
        icons: Optional list of validated Icon descriptors. Only emitted in
            prompts/list for clients negotiating a protocol version that
            supports icons.
    """

    name: str
    description: str
    arguments: list[PromptArgument]
    handler: Callable
    icons: list[Icon] | None

    def __init__(
        self,
        name: str,
        description: str,
        arguments: list[PromptArgument],
        handler: Callable,
        icons: list[Icon] | None = None,
    ) -> None:
        """Initialize prompt definition.

        Args:
            name: Prompt name identifier
            description: Human-readable prompt description
            arguments: List of PromptArgument descriptors
            handler: Sync or async function implementing prompt logic
            icons: Optional list of validated Icon descriptors
        """
        self.name = name
        self.description = description
        self.arguments = arguments
        self.handler = handler
        self.icons = icons


class PromptRegistry:
    """Registry for MCP prompt registration and management.

    Provides decorator-based prompt registration with automatic argument
    generation from function signatures using inspect. Supports required and
    optional arguments derived from parameter defaults.

    The registry handles:
    - Prompt registration via @prompts.prompt() decorator
    - Automatic argument generation from function signatures
    - Prompt discovery via list_prompts()
    - Prompt execution via get_prompt()

    Example:
        >>> prompts = PromptRegistry()
        >>>
        >>> @prompts.prompt()
        >>> async def validate_model(project_id: str) -> list[dict]:
        ...     '''Validate a project model.'''
        ...     return [{"role": "user", "content": f"Validate project {project_id}"}]
        >>>
        >>> # List registered prompts
        >>> prompt_list = prompts.list_prompts()
        >>> print(prompt_list[0]["name"])
        validate_model
        >>>
        >>> # Execute prompt
        >>> import asyncio
        >>> result = asyncio.run(prompts.get_prompt("validate_model", {"project_id": "abc"}))
        >>> print(result[0]["role"])
        user
    """

    _prompts: dict[str, PromptDefinition]

    def __init__(self) -> None:
        """Initialize empty prompt registry."""
        self._prompts = {}

    def prompt(
        self,
        name: str | None = None,
        description: str | None = None,
        icons: list[dict[str, object]] | None = None,
    ) -> Callable:
        """Register function as MCP prompt.

        Decorator for registering sync or async functions as MCP prompts.
        Automatically generates arguments from function signature. Uses the
        function name as prompt name if no name is provided, and the function
        docstring as description if no description is provided.

        Parameters without defaults become required arguments. Parameters with
        defaults become optional arguments. The `self` parameter is filtered out.

        Args:
            name: Prompt name (defaults to function name)
            description: Prompt description (defaults to function docstring)
            icons: Optional list of icon descriptor dicts. Each entry is
                validated via Icon.model_validate() at registration time.
                Only emitted in prompts/list for clients negotiating a
                protocol version that supports icons.

        Returns:
            Decorator function that returns the original function unchanged

        Raises:
            TypeError: If decorated function is not callable
            pydantic.ValidationError: If an icon dict fails Icon validation
                (e.g. src uses a disallowed scheme)

        Example:
            >>> @prompts.prompt()
            >>> async def greet_user(username: str, lang: str = "en") -> list[dict]:
            ...     '''Greet a user in the given language.'''
            ...     return [{"role": "user", "content": f"Hello {username}"}]

        Example with custom name and description:
            >>> @prompts.prompt(
            ...     name="custom_prompt",
            ...     description="Custom description"
            ... )
            >>> async def my_prompt(topic: str) -> list[dict]:
            ...     return [{"role": "user", "content": topic}]
        """

        def decorator(func: Callable) -> Callable:
            prompt_name = name or getattr(func, "__name__", "")
            prompt_description = description or (getattr(func, "__doc__", "") or "").strip()

            arguments = _extract_arguments(func)

            validated_icons = [Icon.model_validate(icon) for icon in icons] if icons is not None else None

            self._prompts[prompt_name] = PromptDefinition(
                name=prompt_name,
                description=prompt_description,
                arguments=arguments,
                handler=func,
                icons=validated_icons,
            )

            return func

        return decorator

    def list_prompts(self, protocol_version: str | None = None) -> list[dict[str, object]]:
        """List all registered prompts with arguments.

        Returns a list of prompt definitions in MCP format with name,
        description, and arguments fields. Used by the MCP protocol to expose
        available prompts to clients.

        Args:
            protocol_version: Negotiated MCP protocol version string (e.g.
                "2025-11-25"). Controls whether icons are emitted: icons
                require a version of "2025-11-25" or later. Defaults to None,
                which omits icons regardless of whether a prompt defines them.

        Returns:
            List of prompt definition dicts with name, description, arguments,
            and icons (if provided and protocol_version supports it)

        Example:
            >>> prompts = PromptRegistry()
            >>>
            >>> @prompts.prompt()
            >>> async def greet(username: str) -> list[dict]:
            ...     '''Greet a user.'''
            ...     return [{"role": "user", "content": f"Hello {username}"}]
            >>>
            >>> prompt_list = prompts.list_prompts()
            >>> print(prompt_list[0]["name"])
            greet
            >>> print(prompt_list[0]["arguments"][0]["name"])
            username
        """
        icons_supported = protocol_version is not None and protocol_version >= "2025-11-25"

        result = []
        for defn in self._prompts.values():
            prompt_dict: dict[str, object] = {
                "name": defn.name,
                "description": defn.description,
                "arguments": [
                    {
                        "name": arg.name,
                        "description": arg.description,
                        "required": arg.required,
                    }
                    for arg in defn.arguments
                ],
            }
            if defn.icons is not None and icons_supported:
                prompt_dict["icons"] = [icon.model_dump(exclude_none=True) for icon in defn.icons]
            result.append(prompt_dict)
        return result

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        """Call registered prompt handler with validated arguments.

        Validates that all required arguments are present, then calls the
        handler (sync or async) with the provided arguments as keyword
        arguments. Returns the list of message dicts from the handler.

        Args:
            name: Prompt name to execute
            arguments: Dict of argument values keyed by argument name

        Returns:
            List of message dicts with role and content keys

        Raises:
            MCPError: code -32602 if prompt not found
            MCPError: code -32602 if a required argument is missing
            MCPError: code -32603 if handler raises an exception

        Example:
            >>> result = await prompts.get_prompt(
            ...     "greet",
            ...     {"username": "Alice"}
            ... )
            >>> print(result[0]["role"])
            user
        """
        if name not in self._prompts:
            raise MCPError(code=-32602, message=f"Prompt not found: {name}")

        defn = self._prompts[name]
        call_args = arguments or {}

        # Validate required arguments are present
        for arg in defn.arguments:
            if arg.required and arg.name not in call_args:
                raise MCPError(
                    code=-32602,
                    message=f"Missing required argument: {arg.name}",
                )

        try:
            if inspect.iscoroutinefunction(defn.handler):
                result = await defn.handler(**call_args)
            else:
                result = defn.handler(**call_args)
            return result
        except MCPError:
            raise
        except Exception as e:
            raise MCPError(code=-32603, message=f"Prompt handler failed: {e}") from e

    def has_prompts(self) -> bool:
        """Check if any prompts are registered.

        Returns:
            True if at least one prompt is registered, False otherwise

        Example:
            >>> prompts = PromptRegistry()
            >>> print(prompts.has_prompts())
            False
            >>> @prompts.prompt()
            >>> async def example() -> list[dict]:
            ...     return []
            >>> print(prompts.has_prompts())
            True
        """
        return len(self._prompts) > 0


def _extract_arguments(func: Callable) -> list[PromptArgument]:
    """Extract prompt arguments from a function signature.

    Inspects the function signature and creates a PromptArgument descriptor
    for each parameter. Skips the `self` parameter. Parameters without
    defaults are marked as required; parameters with defaults are optional.

    Args:
        func: Function to extract arguments from

    Returns:
        List of PromptArgument descriptors ordered by signature position
    """
    sig = inspect.signature(func)
    arguments: list[PromptArgument] = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        required = param.default is inspect.Parameter.empty
        arguments.append(
            PromptArgument(
                name=param_name,
                description=param_name,
                required=required,
            )
        )

    return arguments
