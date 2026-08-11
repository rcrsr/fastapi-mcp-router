"""
FastAPI MCP Router - Lightweight FastAPI integration for Model Context Protocol.

This package provides decorator-based tool registration and stateless HTTP
transport for MCP protocol version 2025-06-18.
"""

__version__ = "0.4.0"

# Public API exports
from fastapi_mcp_router.exceptions import MCPError, ToolError
from fastapi_mcp_router.prompts import PromptRegistry
from fastapi_mcp_router.protocol import encode_cursor, paginate
from fastapi_mcp_router.registry import MCPToolRegistry
from fastapi_mcp_router.resources import ResourceRegistry
from fastapi_mcp_router.router import AuthValidator, MCPRouter, ToolFilter, create_mcp_router, create_prm_router
from fastapi_mcp_router.session import InMemorySessionStore, SessionStore
from fastapi_mcp_router.types import (
    AudioContent,
    EventSubscriber,
    Icon,
    ImageContent,
    ProgressCallback,
    ResourceLinkContent,
    ServerIcon,
    ServerInfo,
    TextContent,
    ToolAnnotations,
    ToolResponse,
)

__all__ = [
    "AudioContent",
    "AuthValidator",
    "EventSubscriber",
    "Icon",
    "ImageContent",
    "InMemorySessionStore",
    "MCPError",
    "MCPRouter",
    "MCPToolRegistry",
    "ProgressCallback",
    "PromptRegistry",
    "ResourceLinkContent",
    "ResourceRegistry",
    "ServerIcon",
    "ServerInfo",
    "SessionStore",
    "TextContent",
    "ToolAnnotations",
    "ToolError",
    "ToolFilter",
    "ToolResponse",
    "create_mcp_router",
    "create_prm_router",
    "encode_cursor",
    "paginate",
]
