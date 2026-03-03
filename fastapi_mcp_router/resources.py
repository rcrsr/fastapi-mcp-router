"""Resource registry for MCP resource registration and management.

This module provides resource registration and provider management for the
Model Context Protocol (MCP). It supports both decorator-based function
registration with URI templates and provider-based registration via the
ResourceProvider ABC.

Classes:
    Resource: Data model for a concrete resource
    ResourceTemplate: Data model for a URI-templated resource
    ResourceContents: Data model for resource content (text or binary)
    ResourceProvider: Abstract base class for resource providers
    FileResourceProvider: Local filesystem resource provider
    ResourceRegistry: Registry for MCP resource registration and execution
"""

import base64
import json
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from inspect import iscoroutinefunction, signature
from pathlib import Path
from typing import get_args, get_origin

from fastapi_mcp_router.exceptions import MCPError

logger = logging.getLogger(__name__)

_FILE_SIZE_LIMIT = 10 * 1024 * 1024  # 10 MB in bytes


def _is_type_or_contains_type(annotation: object, target_type: type, type_name: str) -> bool:
    """Check if annotation is target_type or contains it in a Union.

    Handles direct types, Union types (X | Y), and Optional types (X | None).
    Uses both identity check and string name comparison for robustness.

    Args:
        annotation: Type annotation to check
        target_type: Type class to search for (e.g., Request)
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
        args = get_args(annotation)
        for arg in args:
            if arg is type(None):
                continue
            if _is_type_or_contains_type(arg, target_type, type_name):
                return True

    return False


_DEFAULT_ALLOWED_EXTENSIONS = {".txt", ".md", ".json", ".yaml"}


@dataclass
class Resource:
    """A concrete MCP resource.

    Attributes:
        uri: Resource URI (unique identifier)
        name: Human-readable resource name
        description: Human-readable resource description
        mime_type: Optional MIME type for the resource content
    """

    uri: str
    name: str
    description: str
    mime_type: str | None = None


@dataclass
class ResourceTemplate:
    """An MCP resource template using RFC 6570 URI templates.

    Attributes:
        uri_template: URI template with {param} placeholders (RFC 6570)
        name: Human-readable resource name
        description: Human-readable resource description
        mime_type: Optional MIME type for the resource content
    """

    uri_template: str
    name: str
    description: str
    mime_type: str | None = None


@dataclass
class ResourceContents:
    """MCP resource content payload.

    Either text or blob must be set; they are mutually exclusive.

    Attributes:
        uri: Resource URI this content belongs to
        mime_type: Optional MIME type for the content
        text: Text content (mutually exclusive with blob)
        blob: Base64-encoded binary content (mutually exclusive with text)
    """

    uri: str
    mime_type: str | None = None
    text: str | None = None
    blob: str | None = None


class ResourceProvider(ABC):
    """Abstract base class for MCP resource providers.

    Implement this interface to expose a collection of resources from any
    backing store (filesystem, database, remote API, etc.).

    Example:
        >>> class MyProvider(ResourceProvider):
        ...     def list_resources(self) -> list[Resource]:
        ...         return [Resource(uri="my://item", name="item", description="An item")]
        ...
        ...     async def read_resource(self, uri: str) -> ResourceContents:
        ...         return ResourceContents(uri=uri, text="content")
        ...
        ...     def subscribe(self, uri: str) -> bool:
        ...         return False
        ...
        ...     def unsubscribe(self, uri: str) -> bool:
        ...         return False
        ...
        ...     async def watch(self) -> AsyncIterator[Resource]:
        ...         raise NotImplementedError
    """

    @abstractmethod
    def list_resources(self) -> list[Resource]:
        """Enumerate all resources available from this provider.

        Returns:
            List of Resource objects this provider exposes
        """

    @abstractmethod
    async def read_resource(self, uri: str) -> ResourceContents:
        """Read the content of a resource by URI.

        Args:
            uri: Resource URI to read

        Returns:
            ResourceContents with text or blob content

        Raises:
            MCPError: If the resource cannot be read
        """

    @abstractmethod
    def subscribe(self, uri: str) -> bool:
        """Subscribe to change notifications for a resource.

        Args:
            uri: Resource URI to subscribe to

        Returns:
            True if subscriptions are supported, False otherwise
        """

    @abstractmethod
    def unsubscribe(self, uri: str) -> bool:
        """Unsubscribe from change notifications for a resource.

        Args:
            uri: Resource URI to unsubscribe from

        Returns:
            True if unsubscription succeeded, False otherwise
        """

    @abstractmethod
    async def watch(self) -> AsyncIterator[Resource]:
        """Async generator yielding resources as they change.

        Returns:
            AsyncIterator of changed Resource objects

        Raises:
            NotImplementedError: If this provider does not support watching
        """


class _ResourceDefinition:
    """Internal storage for function-based resource handler metadata.

    Attributes:
        uri_template: URI template string with {param} placeholders
        name: Resource name
        description: Resource description
        mime_type: Optional MIME type
        handler: Async callable that returns resource content
        pattern: Compiled regex for matching incoming URIs
        param_names: Ordered list of parameter names from the template
    """

    def __init__(
        self,
        uri_template: str,
        name: str,
        description: str,
        handler: Callable,
        mime_type: str | None = None,
    ) -> None:
        """Initialize resource definition.

        Args:
            uri_template: URI template with {param} placeholders
            name: Resource name
            description: Resource description
            handler: Async callable implementing resource logic
            mime_type: Optional MIME type
        """
        self.uri_template = uri_template
        self.name = name
        self.description = description
        self.mime_type = mime_type
        self.handler = handler
        self.param_names, self.pattern = _compile_uri_template(uri_template)


def _compile_uri_template(uri_template: str) -> tuple[list[str], re.Pattern]:
    """Compile a URI template into a regex pattern and parameter name list.

    Converts {param} placeholders into named regex capture groups. All other
    characters in the template are treated as literals and regex-escaped.

    Args:
        uri_template: URI template string, e.g. "file://{path}"

    Returns:
        Tuple of (param_names, compiled_pattern) where param_names is an
        ordered list of parameter names extracted from the template.

    Example:
        >>> names, pattern = _compile_uri_template("file://{path}")
        >>> m = pattern.fullmatch("file:///home/user/readme.md")
        >>> m.group("path")
        '/home/user/readme.md'
    """
    param_names: list[str] = []
    regex_parts: list[str] = []
    last_end = 0

    for match in re.finditer(r"\{(\w+)\}", uri_template):
        # Escape the literal part before this placeholder
        literal = uri_template[last_end : match.start()]
        regex_parts.append(re.escape(literal))
        param_name = match.group(1)
        param_names.append(param_name)
        regex_parts.append(f"(?P<{param_name}>.+)")
        last_end = match.end()

    # Append any remaining literal suffix
    remaining = uri_template[last_end:]
    if remaining:
        regex_parts.append(re.escape(remaining))

    pattern = re.compile("".join(regex_parts))
    return param_names, pattern


class FileResourceProvider(ResourceProvider):
    """Local filesystem resource provider with sandboxed access.

    Provides read access to files under a root directory. Access is restricted
    to files with allowed extensions and enforces a 10 MB file size limit. Path
    traversal attempts using ".." segments are rejected.

    URIs accepted by read_resource follow the format:
        "file://<absolute-path>" or "<absolute-path>"

    Attributes:
        root_path: Resolved absolute root directory for all file access
        allowed_extensions: Set of permitted file extensions (with leading dot)

    Example:
        >>> provider = FileResourceProvider(root_path="/data/docs")
        >>> resources = provider.list_resources()
        >>> contents = await provider.read_resource("file:///data/docs/readme.md")
        >>> print(contents.text)
    """

    def __init__(
        self,
        root_path: str | Path,
        allowed_extensions: set[str] | None = None,
    ) -> None:
        """Initialize FileResourceProvider.

        Args:
            root_path: Root directory for sandboxed file access
            allowed_extensions: Permitted file extensions (default: {".txt", ".md", ".json", ".yaml"})
        """
        self.root_path = Path(root_path).resolve()
        self.allowed_extensions: set[str] = (
            allowed_extensions if allowed_extensions is not None else set(_DEFAULT_ALLOWED_EXTENSIONS)
        )

    def list_resources(self) -> list[Resource]:
        """List all files under root_path with allowed extensions.

        Returns:
            List of Resource objects for each accessible file

        Example:
            >>> provider = FileResourceProvider("/docs")
            >>> resources = provider.list_resources()
        """
        resources: list[Resource] = []
        if not self.root_path.exists():
            return resources

        for file_path in self.root_path.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix not in self.allowed_extensions:
                continue
            uri = f"file://{file_path}"
            mime_type = _guess_mime_type(file_path.suffix)
            resources.append(
                Resource(
                    uri=uri,
                    name=file_path.name,
                    description=f"File: {file_path.relative_to(self.root_path)}",
                    mime_type=mime_type,
                )
            )
        return resources

    async def read_resource(self, uri: str) -> ResourceContents:
        """Read a file by URI with sandbox and size validation.

        Args:
            uri: File URI in format "file://<path>" or "<path>"

        Returns:
            ResourceContents with text content for text files, blob for binary

        Raises:
            MCPError: -32602 if URI outside root, path traversal, unsupported
                extension, or file exceeds 10 MB

        Example:
            >>> contents = await provider.read_resource("file:///docs/readme.md")
            >>> print(contents.text)
        """
        file_path = _resolve_file_uri(uri, self.root_path)
        _validate_file_access(file_path, uri, self.allowed_extensions)
        mime_type = _guess_mime_type(file_path.suffix)

        raw_bytes = file_path.read_bytes()
        # Determine whether to return text or blob based on MIME type
        if mime_type and mime_type.startswith("text/"):
            return ResourceContents(
                uri=uri,
                mime_type=mime_type,
                text=raw_bytes.decode("utf-8", errors="replace"),
            )
        # JSON is text-based despite not starting with "text/"
        if mime_type == "application/json":
            return ResourceContents(
                uri=uri,
                mime_type=mime_type,
                text=raw_bytes.decode("utf-8", errors="replace"),
            )
        # Binary fallback
        return ResourceContents(
            uri=uri,
            mime_type=mime_type or "application/octet-stream",
            blob=base64.b64encode(raw_bytes).decode("ascii"),
        )

    def subscribe(self, uri: str) -> bool:
        """Return False; FileResourceProvider does not support subscriptions.

        Args:
            uri: Resource URI (unused)

        Returns:
            Always False
        """
        return False

    def unsubscribe(self, uri: str) -> bool:
        """Return False; FileResourceProvider does not support subscriptions.

        Args:
            uri: Resource URI (unused)

        Returns:
            Always False
        """
        return False

    async def watch(self) -> AsyncIterator[Resource]:
        """Raise NotImplementedError; file watching is not supported.

        Raises:
            NotImplementedError: Always
        """
        raise NotImplementedError("FileResourceProvider does not support watch()")


def _resolve_file_uri(uri: str, root_path: Path) -> Path:
    """Resolve a file URI to a validated absolute Path within root_path.

    Strips the "file://" scheme prefix if present, then resolves the path
    relative to root_path. Validates path traversal and sandbox containment.
    Extension and size checks are done separately by _validate_file_access.

    Args:
        uri: File URI (with or without "file://" prefix)
        root_path: Sandboxed root directory

    Returns:
        Validated absolute Path

    Raises:
        MCPError: -32602 if path contains "..", is outside root, or does not exist
    """
    raw = uri
    if raw.startswith("file://"):
        raw = raw[len("file://") :]

    # Reject path traversal before any resolution
    if ".." in raw.split("/"):
        raise MCPError(code=-32602, message=f"Path traversal not allowed: {uri}")

    # Also check for literal ".." in path components
    path_obj = Path(raw)
    for part in path_obj.parts:
        if part == "..":
            raise MCPError(code=-32602, message=f"Path traversal not allowed: {uri}")

    # Resolve absolute path (handle both absolute and relative inputs)
    if path_obj.is_absolute():
        candidate = path_obj.resolve()
    else:
        candidate = (root_path / path_obj).resolve()

    # Sandbox check: resolved path must be inside root_path
    try:
        candidate.relative_to(root_path)
    except ValueError as exc:
        raise MCPError(code=-32602, message=f"URI outside allowed root: {uri}") from exc

    if not candidate.exists():
        raise MCPError(code=-32602, message=f"Resource not found: {uri}")

    return candidate


def _validate_file_access(
    candidate: Path,
    uri: str,
    allowed_extensions: set[str],
) -> None:
    """Validate extension and file size for a resolved file path.

    Args:
        candidate: Resolved absolute file path
        uri: Original URI (for error messages)
        allowed_extensions: Set of permitted file extensions

    Raises:
        MCPError: -32602 if extension is not allowed or file exceeds 10 MB
    """
    if candidate.suffix not in allowed_extensions:
        raise MCPError(
            code=-32602,
            message=f"File extension not allowed: {candidate.suffix}",
        )

    file_size = candidate.stat().st_size
    if file_size > _FILE_SIZE_LIMIT:
        raise MCPError(
            code=-32602,
            message=f"File exceeds 10 MB limit ({file_size} bytes): {uri}",
        )


def _guess_mime_type(suffix: str) -> str | None:
    """Return a MIME type string for a file extension, or None if unknown.

    Args:
        suffix: File extension including leading dot (e.g. ".json")

    Returns:
        MIME type string or None
    """
    mime_map: dict[str, str] = {
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".json": "application/json",
        ".yaml": "text/yaml",
        ".yml": "text/yaml",
        ".html": "text/html",
        ".xml": "application/xml",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }
    return mime_map.get(suffix)


class ResourceRegistry:
    """Registry for MCP resource registration and management.

    Supports two registration modes:

    1. **Decorator-based** via ``@registry.resource(uri_template=...)``: the
       decorated async function is called with matched URI parameters as kwargs.
       The URI template may contain ``{param}`` placeholders.

    2. **Provider-based** via ``register_provider(prefix, provider)``: an
       instance of ``ResourceProvider`` handles all URIs that start with the
       given prefix.

    Example:
        >>> registry = ResourceRegistry()
        >>>
        >>> @registry.resource(
        ...     uri_template="docs://{slug}",
        ...     name="Document",
        ...     description="Fetch a document by slug",
        ... )
        >>> async def get_doc(slug: str) -> str:
        ...     return f"Content of {slug}"
        >>>
        >>> contents = await registry.read_resource("docs://readme")
        >>> print(contents.text)
        Content of readme
    """

    _handlers: list[_ResourceDefinition]
    _providers: list[tuple[str, ResourceProvider]]

    def __init__(self) -> None:
        """Initialize empty resource registry."""
        self._handlers = []
        self._providers = []

    def resource(
        self,
        uri_template: str,
        name: str | None = None,
        description: str | None = None,
        mime_type: str | None = None,
    ) -> Callable:
        """Decorator to register an async function as an MCP resource handler.

        The decorated function is called when a URI matching the template is
        requested. Template parameters (``{param}``) are extracted from the URI
        and passed as keyword arguments to the function.

        Supported return types:
        - ``str``: returned as text content
        - ``dict``: serialized as JSON text
        - ``bytes``: base64-encoded as blob content

        Args:
            uri_template: URI template string with optional {param} placeholders
            name: Resource name (defaults to function name)
            description: Resource description (defaults to function docstring)
            mime_type: Optional MIME type override

        Returns:
            Decorator that returns the original function unchanged

        Raises:
            TypeError: If the decorated function is not async

        Example:
            >>> @registry.resource(uri_template="config://{key}", name="Config")
            >>> async def get_config(key: str) -> str:
            ...     '''Fetch a configuration value.'''
            ...     return f"value_of_{key}"
        """

        def decorator(func: Callable) -> Callable:
            func_name = getattr(func, "__name__", repr(func))
            if not iscoroutinefunction(func):
                raise TypeError(f"Resource handler {func_name} must be async. Add 'async def' to function definition.")

            resource_name = name or func_name
            resource_description = description or (getattr(func, "__doc__", None) or "").strip()

            self._handlers.append(
                _ResourceDefinition(
                    uri_template=uri_template,
                    name=resource_name,
                    description=resource_description,
                    handler=func,
                    mime_type=mime_type,
                )
            )
            return func

        return decorator

    def register_provider(self, uri_prefix: str, provider: ResourceProvider) -> None:
        """Register a ResourceProvider for all URIs starting with uri_prefix.

        Args:
            uri_prefix: URI prefix string; any URI starting with this value is
                dispatched to the provider
            provider: ResourceProvider instance to handle matching URIs

        Example:
            >>> provider = FileResourceProvider("/data/docs")
            >>> registry.register_provider("file:///data/docs", provider)
        """
        self._providers.append((uri_prefix, provider))

    def has_resources(self) -> bool:
        """Return True if any function-based or provider-based resources are registered.

        Returns:
            True if at least one handler or provider has been registered

        Example:
            >>> registry = ResourceRegistry()
            >>> registry.has_resources()
            False
            >>> registry.register_provider("file://", FileResourceProvider("/tmp"))
            >>> registry.has_resources()
            True
        """
        return bool(self._handlers) or bool(self._providers)

    def list_resources(self) -> list[Resource]:
        """Return all resources from function-based handlers and providers.

        Function-based handlers contribute a single Resource where ``uri`` is
        set to the URI template (since the concrete URIs are not enumerable).
        Provider-based resources are enumerated by calling each provider's
        ``list_resources()`` method.

        Returns:
            Combined list of Resource objects from all sources

        Raises:
            MCPError: -32601 if no resources are registered

        Example:
            >>> resources = registry.list_resources()
            >>> for r in resources:
            ...     print(r.uri, r.name)
        """
        if not self.has_resources():
            raise MCPError(code=-32601, message="No resources registered")

        result: list[Resource] = []

        for defn in self._handlers:
            result.append(
                Resource(
                    uri=defn.uri_template,
                    name=defn.name,
                    description=defn.description,
                    mime_type=defn.mime_type,
                )
            )

        for _prefix, provider in self._providers:
            try:
                result.extend(provider.list_resources())
            except MCPError:
                raise
            except Exception as e:
                raise MCPError(code=-32603, message=f"Provider list_resources failed: {e}") from e

        return result

    def list_templates(self) -> list[ResourceTemplate]:
        """Return URI templates for function-based resource handlers only.

        Templates are returned for any URI template that contains at least one
        ``{param}`` placeholder. Static (parameter-free) URI templates are
        also included.

        Returns:
            List of ResourceTemplate objects for all function-based handlers

        Example:
            >>> templates = registry.list_templates()
            >>> for t in templates:
            ...     print(t.uri_template)
        """
        return [
            ResourceTemplate(
                uri_template=defn.uri_template,
                name=defn.name,
                description=defn.description,
                mime_type=defn.mime_type,
            )
            for defn in self._handlers
        ]

    async def read_resource(self, uri: str, request: object = None) -> ResourceContents:
        """Invoke the handler or provider matching the URI.

        Matching order:
        1. Function-based handlers are checked first using regex matching
           against their URI templates.
        2. Provider-based entries are checked next using URI prefix matching.

        Args:
            uri: Resource URI to read
            request: Optional FastAPI Request object for Depends() injection

        Returns:
            ResourceContents with text or blob content

        Raises:
            MCPError: -32602 if no handler or provider matches the URI
            MCPError: -32603 if the handler raises an unexpected exception
            MCPError: -32603 if the handler returns an unsupported type

        Example:
            >>> contents = await registry.read_resource("docs://readme")
            >>> print(contents.text)
        """
        # Try function-based handlers first
        for defn in self._handlers:
            match = defn.pattern.fullmatch(uri)
            if match is None:
                continue
            kwargs = match.groupdict()
            return await _invoke_handler(defn, uri, kwargs, request=request)

        # Try provider-based entries
        for prefix, provider in self._providers:
            if uri.startswith(prefix):
                try:
                    return await provider.read_resource(uri)
                except MCPError:
                    raise
                except Exception as e:
                    raise MCPError(code=-32603, message=f"Resource read failed: {e}") from e

        raise MCPError(code=-32602, message=f"Unknown resource URI: {uri}")


async def _invoke_handler(
    defn: _ResourceDefinition,
    uri: str,
    kwargs: dict[str, str],
    request: object = None,
) -> ResourceContents:
    """Call a function-based resource handler and normalize its return value.

    Resolves FastAPI Depends() parameters from the handler signature before
    invocation, injecting the optional Request into dependencies that need it.

    Args:
        defn: Resource definition containing handler and metadata
        uri: Original request URI (used in returned ResourceContents)
        kwargs: Matched URI parameters to pass as keyword arguments
        request: Optional FastAPI Request object for Depends() injection

    Returns:
        ResourceContents built from the handler's return value

    Raises:
        MCPError: -32603 if the handler raises an unexpected exception
        MCPError: -32603 if the handler returns an unsupported type
    """
    # Resolve Depends() OUTSIDE try — exceptions propagate per EC-3
    from fastapi import Request

    call_kwargs: dict[str, object] = dict(kwargs)
    handler_sig = signature(defn.handler)

    for param_name, param in handler_sig.parameters.items():
        if not (hasattr(param.default, "__class__") and param.default.__class__.__name__ == "Depends"):
            continue

        dependency_fn = param.default.dependency
        if dependency_fn is None:
            continue

        dep_sig = signature(dependency_fn)
        dep_kwargs: dict[str, object] = {}

        for dep_param_name, dep_param in dep_sig.parameters.items():
            if _is_type_or_contains_type(dep_param.annotation, Request, "Request"):
                if request is not None:
                    dep_kwargs[dep_param_name] = request

        if iscoroutinefunction(dependency_fn):
            resolved = await dependency_fn(**dep_kwargs)
        else:
            resolved = dependency_fn(**dep_kwargs)

        call_kwargs[param_name] = resolved

    # Only handler invocation inside try
    try:
        result = await defn.handler(**call_kwargs)
    except MCPError:
        raise
    except Exception as e:
        raise MCPError(code=-32603, message=f"Resource handler failed: {e}") from e

    mime_type = defn.mime_type

    if isinstance(result, str):
        return ResourceContents(
            uri=uri,
            mime_type=mime_type or "text/plain",
            text=result,
        )

    if isinstance(result, dict):
        return ResourceContents(
            uri=uri,
            mime_type=mime_type or "application/json",
            text=json.dumps(result),
        )

    if isinstance(result, bytes):
        return ResourceContents(
            uri=uri,
            mime_type=mime_type or "application/octet-stream",
            blob=base64.b64encode(result).decode("ascii"),
        )

    raise MCPError(
        code=-32603,
        message=f"Resource handler returned unsupported type: {type(result).__name__}",
    )
