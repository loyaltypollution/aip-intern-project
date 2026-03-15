"""Mesh orchestrator node functions.

supervisor_node validates input and routes unconditionally to crew_node.
For Phase 2, routing logic is trivial — future phases may add branching.
"""

from __future__ import annotations

from aip_intern.mesh.state import MeshState


def supervisor_node(state: MeshState) -> dict:
    """Validate DocumentTask input and route to crew_node.

    Synchronous — no async work needed. LangGraph handles routing via graph edges.
    Phase 2: routes unconditionally to crew_node.
    Future phases: add branching logic here based on state fields.
    """
    if not state.get("task_description"):
        return {
            "error": "supervisor_node: task_description is required",
            "step_trace": state["step_trace"] + ["supervisor_node"],
        }
    return {"step_trace": state["step_trace"] + ["supervisor_node"]}
