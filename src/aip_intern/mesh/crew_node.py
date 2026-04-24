"""crew_node — wraps a CrewAI Crew as a single async LangGraph node.

The supervisor_node routes unconditionally to this node (Phase 2).
crew_node calls crew.kickoff_async() and returns state updates.

CrewAI's kickoff_async() is used (not kickoff()) to avoid blocking
the async LangGraph execution loop.

Inter-agent message count is read from the CrewAI crew result.
State transfer size is measured by serialising the state dict before returning.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from aip_intern.core.exceptions import AIPInternError
from aip_intern.mesh.state import MeshState

if TYPE_CHECKING:
    from crewai import Crew


async def crew_node(state: MeshState, crew: "Crew") -> dict:
    """Invoke a CrewAI Crew and write results back to LangGraph state.

    Args:
        state: Current MeshState.
        crew: Compiled CrewAI Crew (Triage Specialist + Brief+Response Specialist).

    Returns:
        Dict of state updates including outputs, message_count, state_size_bytes.
    """
    try:
        result = await crew.kickoff_async(
            inputs={
                "run_id": state["run_id"],
                "task_description": state["task_description"],
            }
        )
        # Measure state transfer size (serialised state at this checkpoint)
        state_snapshot = {**state, "step_trace": state["step_trace"] + ["crew_node"]}
        state_size = len(json.dumps(state_snapshot, default=str).encode())

        # Inter-agent handoff count = number of completed crew tasks
        # (one per specialist hand-off: Triage → Brief+Response = 2 tasks)
        msg_count = (
            len(result.tasks_output)
            if hasattr(result, "tasks_output") and result.tasks_output
            else len(crew.tasks)
        )

        # Token usage from CrewAI's LiteLLM aggregation
        token_usage = getattr(result, "token_usage", None)
        prompt_tokens = getattr(token_usage, "prompt_tokens", 0) if token_usage else 0
        completion_tokens = getattr(token_usage, "completion_tokens", 0) if token_usage else 0

        return {
            "triage_result": "outputs/triage.csv",
            "brief_result": "outputs/brief.md",
            "response_result": "outputs/response_templates.md",
            "message_count": msg_count,
            "state_size_bytes": state_size,
            "step_trace": state["step_trace"] + ["crew_node"],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
    except AIPInternError:
        raise
    except Exception as e:
        return {
            "error": f"crew_node: {e}",
            "step_trace": state["step_trace"] + ["crew_node"],
            "state_size_bytes": 0,
        }
