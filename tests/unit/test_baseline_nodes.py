from __future__ import annotations

from pathlib import Path

import pytest

from aip_intern.baseline.graph import build_graph
from aip_intern.baseline.nodes import brief_node, response_node, triage_node
from aip_intern.baseline.runner import RunConfig, run_once
from aip_intern.baseline.state import BaselineState


def test_baseline_state_construction():
    state: BaselineState = {
        "run_id": "baseline_test",
        "task_description": "Triage feedback",
        "feedback_files": [],
        "policy_content": "",
        "triage_result": None,
        "brief_result": None,
        "response_result": None,
        "error": None,
        "step_trace": [],
    }
    assert state["run_id"] == "baseline_test"
    assert state["step_trace"] == []


def _make_state(**overrides) -> BaselineState:
    base: BaselineState = {
        "run_id": "test_run",
        "task_description": "test task",
        "feedback_files": ["data/feedback/msg_001.txt"],
        "policy_content": "SLA: 48h for HIGH",
        "triage_result": None,
        "brief_result": None,
        "response_result": None,
        "error": None,
        "step_trace": [],
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_triage_node_updates_step_trace(mock_llm):
    state = _make_state()
    result = await triage_node(state, llm=mock_llm, tools=[])
    assert "triage_node" in result["step_trace"]


@pytest.mark.asyncio
async def test_brief_node_requires_triage_result(mock_llm):
    """brief_node should set error if triage_result is None."""
    state = _make_state(triage_result=None)
    result = await brief_node(state, llm=mock_llm, tools=[])
    assert result.get("error") is not None


@pytest.mark.asyncio
async def test_response_node_requires_brief_result(mock_llm):
    """response_node should set error if brief_result is None."""
    state = _make_state(brief_result=None, triage_result="outputs/triage.csv")
    result = await response_node(state, llm=mock_llm, tools=[])
    assert result.get("error") is not None


def test_build_graph_compiles(mock_llm):
    graph = build_graph(mock_llm, tools=[])
    assert graph is not None


@pytest.mark.asyncio
async def test_run_once_returns_run_result(mock_llm, tmp_path):
    cfg = RunConfig(
        run_id_prefix="test",
        n_runs=1,
        config_path=Path("config/baseline.yaml"),
        llm_model="test-model",
        llm_base_url="mock",
        llm_api_key="x",
        workspace_root=Path("workspace/"),
        artifacts_dir=tmp_path,
    )
    # Patch _make_llm and create_mcp_tools so no real LLM or MCP server is needed
    from unittest.mock import AsyncMock, patch

    from aip_intern.baseline import runner as runner_mod

    with patch.object(runner_mod, "_make_llm", return_value=mock_llm), patch(
        "aip_intern.baseline.runner.create_mcp_tools", new=AsyncMock(return_value=[])
    ):
        result = await run_once(cfg)
    assert result.run_id.startswith("test_")
    assert isinstance(result.success, bool)
