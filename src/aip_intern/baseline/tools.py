"""LangChain-native file tools for the baseline pipeline.

These are direct Python implementations — no MCP, no CrewAI. They are
compatible with ChatOpenAI.bind_tools() and _invoke_with_tools().

The workspace root is set once per run via set_workspace_root() before
calling get_tools(). This mirrors the pattern in mesh/tools.py but uses
langchain_core.tools.BaseTool so they work with LangChain's bind_tools().
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import BaseTool
from pydantic import BaseModel

_WORKSPACE_ROOT: Path = Path("workspace/")


def set_workspace_root(path: str | Path) -> None:
    """Set the workspace root for all tools. Call before get_tools() each run."""
    global _WORKSPACE_ROOT
    _WORKSPACE_ROOT = Path(path).resolve()


class _ReadFileInput(BaseModel):
    path: str  # relative to workspace root


class ReadFileTool(BaseTool):
    name: str = "read_file"
    description: str = (
        "Read a file from the workspace. path must be relative to workspace root."
    )
    args_schema: type[BaseModel] = _ReadFileInput

    def _run(self, path: str) -> str:
        full_path = _WORKSPACE_ROOT / path
        if not full_path.exists():
            return f"Error: {path} not found"
        return full_path.read_text()

    async def _arun(self, path: str) -> str:
        return self._run(path)


class _WriteFileInput(BaseModel):
    path: str
    content: str


class WriteFileTool(BaseTool):
    name: str = "write_file"
    description: str = (
        "Write content to a file in the workspace."
        " path must be relative to workspace root."
    )
    args_schema: type[BaseModel] = _WriteFileInput

    def _run(self, path: str, content: str) -> str:
        full_path = _WORKSPACE_ROOT / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        return f"Written: {path}"

    async def _arun(self, path: str, content: str) -> str:
        return self._run(path, content)


class _ListDirInput(BaseModel):
    path: str


class ListDirectoryTool(BaseTool):
    name: str = "list_directory"
    description: str = (
        "List files in a workspace directory."
        " path must be relative to workspace root."
    )
    args_schema: type[BaseModel] = _ListDirInput

    def _run(self, path: str) -> str:
        full_path = _WORKSPACE_ROOT / path
        if not full_path.is_dir():
            return f"Error: {path} is not a directory"
        return "\n".join(p.name for p in sorted(full_path.iterdir()))

    async def _arun(self, path: str) -> str:
        return self._run(path)


def get_tools() -> list[BaseTool]:
    """Return LangChain-compatible workspace tools for the baseline pipeline."""
    return [ReadFileTool(), WriteFileTool(), ListDirectoryTool()]
