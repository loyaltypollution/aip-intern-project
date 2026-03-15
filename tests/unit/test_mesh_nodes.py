from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aip_intern.mesh.crew_node import crew_node
from aip_intern.mesh.nodes import supervisor_node
from aip_intern.mesh.state import MeshState


def test_mesh_state_has_mesh_metrics():
    state: MeshState = {
        "run_id": "mesh_test",
        "task_description": "test",
        "error": None,
        "step_trace": [],
        "triage_result": None,
        "brief_result": None,
        "response_result": None,
        "message_count": 0,
        "state_size_bytes": 0,
    }
    assert state["message_count"] == 0
    assert state["state_size_bytes"] == 0


def _make_mesh_state(**overrides) -> MeshState:
    base: MeshState = {
        "run_id": "mesh_test",
        "task_description": "test",
        "triage_result": None,
        "brief_result": None,
        "response_result": None,
        "error": None,
        "step_trace": [],
        "message_count": 0,
        "state_size_bytes": 0,
    }
    base.update(overrides)
    return base


# --- supervisor_node tests ---

def test_supervisor_node_appends_step_trace():
    state = _make_mesh_state()
    result = supervisor_node(state)
    assert "supervisor_node" in result["step_trace"]


def test_supervisor_node_returns_no_error_on_valid_input():
    state = _make_mesh_state()
    result = supervisor_node(state)
    assert result.get("error") is None


def test_supervisor_node_returns_error_when_no_task_description():
    state = _make_mesh_state(task_description="")
    result = supervisor_node(state)
    assert "error" in result
    assert result["error"] is not None


# --- crew_node tests ---

def _make_mock_crew(tasks_output=None):
    """Helper: mock crew with tasks_output configured so message_count works."""
    mock_result = MagicMock(raw="done")
    mock_result.tasks_output = (
        tasks_output if tasks_output is not None else [MagicMock(), MagicMock()]
    )
    mock_crew = MagicMock()
    mock_crew.kickoff_async = AsyncMock(return_value=mock_result)
    mock_crew.tasks = [MagicMock(), MagicMock()]
    return mock_crew


@pytest.mark.asyncio
async def test_crew_node_updates_step_trace():
    state = _make_mesh_state()
    mock_crew = _make_mock_crew()
    result = await crew_node(state, crew=mock_crew)
    assert "crew_node" in result["step_trace"]


@pytest.mark.asyncio
async def test_crew_node_records_state_size():
    state = _make_mesh_state()
    mock_crew = _make_mock_crew()
    result = await crew_node(state, crew=mock_crew)
    assert result["state_size_bytes"] > 0


@pytest.mark.asyncio
async def test_crew_node_message_count_equals_tasks_output_length():
    state = _make_mesh_state()
    mock_crew = _make_mock_crew(tasks_output=[MagicMock(), MagicMock()])
    result = await crew_node(state, crew=mock_crew)
    assert result["message_count"] == 2
