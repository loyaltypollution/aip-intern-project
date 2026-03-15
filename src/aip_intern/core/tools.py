"""MCP filesystem tool factory.

Returns langchain BaseTool objects backed by the @modelcontextprotocol/server-filesystem
Node.js server. Follows the same pattern as phase3's tools.py.

Requires Node.js + npx on PATH.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient


def _check_node() -> None:
    if shutil.which("npx") is None:
        raise RuntimeError(
            "Node.js/npx is required for MCP tools. Install from https://nodejs.org/"
        )


async def create_mcp_tools(workspace_root: Path) -> list:
    """Create MCP filesystem tools scoped to workspace_root.

    Args:
        workspace_root: Absolute or relative path to the workspace directory.
                        All file paths inside agents MUST be relative to this root.
    Returns:
        List of langchain BaseTool objects.
    """
    _check_node()
    client = MultiServerMCPClient(
        {
            "filesystem": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    str(Path(workspace_root).resolve()),
                ],
                "transport": "stdio",
            }
        }
    )
    return await client.get_tools()
