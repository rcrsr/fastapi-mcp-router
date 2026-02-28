"""
FastAPI MCP Router - Lightweight FastAPI integration for Model Context Protocol.

This package provides decorator-based tool registration and stateless HTTP
transport for MCP protocol version 2025-06-18.
"""

__version__ = "0.1.0"

# Public API exports
from fastapi_mcp_router.exceptions import MCPError, ToolError
from fastapi_mcp_router.prompts import PromptRegistry
from fastapi_mcp_router.registry import MCPToolRegistry
from fastapi_mcp_router.resources import ResourceRegistry
from fastapi_mcp_router.router import MCPRouter, ToolFilter, create_mcp_router, create_prm_router
from fastapi_mcp_router.session import InMemorySessionStore, SessionStore
from fastapi_mcp_router.types import (
    EventSubscriber,
    ProgressCallback,
    ServerIcon,
    ServerInfo,
    TextContent,
    ToolResponse,
)

__all__ = [
    "EventSubscriber",
    "InMemorySessionStore",
    "MCPError",
    "MCPRouter",
    "MCPToolRegistry",
    "ProgressCallback",
    "PromptRegistry",
    "ResourceRegistry",
    "ServerIcon",
    "ServerInfo",
    "SessionStore",
    "TextContent",
    "ToolError",
    "ToolFilter",
    "ToolResponse",
    "create_mcp_router",
    "create_prm_router",
]
