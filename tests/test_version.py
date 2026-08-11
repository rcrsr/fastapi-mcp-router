"""Release-metadata tests for fastapi-mcp-router.

Verifies that the importable `fastapi_mcp_router.__version__` stays in sync
with the `[project].version` declared in pyproject.toml.
"""

import tomllib
from pathlib import Path

import pytest

import fastapi_mcp_router


@pytest.mark.unit
def test_package_version_matches_pyproject():
    """Test fastapi_mcp_router.__version__ matches pyproject.toml's version."""
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        pyproject = tomllib.load(f)

    assert fastapi_mcp_router.__version__ == pyproject["project"]["version"]
