"""
Docstring presence checks for public symbols added by the 0.4.0 spec-compliance work.

These are cheap containment checks (non-empty `__doc__`), not content assertions;
asserting on docstring text would make this test brittle against copy edits.
"""

import pytest


@pytest.mark.unit
@pytest.mark.parametrize(
    "target",
    [
        pytest.param("fastapi_mcp_router.protocol:encode_cursor", id="encode_cursor"),
        pytest.param("fastapi_mcp_router.protocol:decode_cursor", id="decode_cursor"),
        pytest.param("fastapi_mcp_router.protocol:paginate", id="paginate"),
        pytest.param("fastapi_mcp_router.router:build_content_block", id="build_content_block"),
        pytest.param(
            "fastapi_mcp_router.router:MCPRouter.notify_tools_list_changed",
            id="notify_tools_list_changed",
        ),
        pytest.param(
            "fastapi_mcp_router.router:MCPRouter.notify_resources_list_changed",
            id="notify_resources_list_changed",
        ),
        pytest.param(
            "fastapi_mcp_router.router:MCPRouter.notify_prompts_list_changed",
            id="notify_prompts_list_changed",
        ),
        pytest.param(
            "fastapi_mcp_router.router:MCPRouter.notify_resource_updated",
            id="notify_resource_updated",
        ),
        pytest.param("fastapi_mcp_router.session:SessionStore.list_sessions", id="list_sessions"),
        pytest.param("fastapi_mcp_router.session:SessionStore.find_subscribers", id="find_subscribers"),
        pytest.param("fastapi_mcp_router.types:ImageContent", id="ImageContent"),
        pytest.param("fastapi_mcp_router.types:AudioContent", id="AudioContent"),
        pytest.param("fastapi_mcp_router.types:ResourceLinkContent", id="ResourceLinkContent"),
        pytest.param("fastapi_mcp_router.types:Icon", id="Icon"),
        pytest.param("fastapi_mcp_router.types:ToolAnnotations", id="ToolAnnotations"),
        pytest.param("fastapi_mcp_router.registry:MCPToolRegistry.tool", id="MCPToolRegistry.tool"),
    ],
)
def test_public_symbol_has_docstring(target: str) -> None:
    """Every new public symbol from the spec-compliance work has a non-empty docstring."""
    module_path, _, attr_path = target.partition(":")
    module = __import__(module_path, fromlist=["_"])
    obj = module
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)

    assert obj.__doc__ is not None
    assert obj.__doc__.strip() != ""
