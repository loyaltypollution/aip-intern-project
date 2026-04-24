"""LangGraph StateGraph for the Phase 2 mesh.

Graph topology:
    START → supervisor_node → crew_node → END

build_graph() returns a compiled graph ready for ainvoke().

Usage:
    graph = build_graph(llm_cfg, workspace_root=Path("workspace/"))
    result = await graph.ainvoke(initial_state, config=invoke_config)
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

from crewai import LLM, Crew, Process
from langgraph.graph import END, START, StateGraph

from aip_intern.mesh.crew.agents import (
    make_brief_response_specialist,
    make_triage_specialist,
)
from aip_intern.mesh.crew.tasks import make_brief_task, make_triage_task
from aip_intern.mesh.crew_node import crew_node
from aip_intern.mesh.nodes import supervisor_node
from aip_intern.mesh.state import MeshState
from aip_intern.mesh.tools import get_tools, set_workspace_root


def _build_crew(llm_cfg, workspace_root: Path) -> Crew:
    """Assemble the CrewAI Crew from agents and tasks."""
    set_workspace_root(workspace_root)
    tools = get_tools()

    # CrewAI's LLM() parses provider from the first `/` segment. vLLM's
    # OpenAI-compatible endpoint is registered under `hosted_vllm` — `openai/`
    # breaks when the model itself contains a `/` (e.g. `Qwen/Qwen2.5-...`),
    # because the second segment gets treated as the model ID and fails
    # native-provider lookup.
    model_name = llm_cfg.model
    if not model_name.startswith("hosted_vllm/"):
        # Strip any other provider prefix first (e.g. a stale `openai/`).
        if model_name.startswith("openai/"):
            model_name = model_name[len("openai/"):]
        model_name = f"hosted_vllm/{model_name}"
    crewai_llm = LLM(
        model=model_name,
        base_url=llm_cfg.base_url,
        api_key=llm_cfg.api_key,
        temperature=llm_cfg.temperature,
    )

    triage_agent = make_triage_specialist(crewai_llm, tools)
    brief_agent = make_brief_response_specialist(crewai_llm, tools)
    triage_task = make_triage_task(triage_agent)
    brief_task = make_brief_task(brief_agent, triage_task)

    return Crew(
        agents=[triage_agent, brief_agent],
        tasks=[triage_task, brief_task],
        process=Process.sequential,
        verbose=True,
    )


def build_graph(llm_cfg, workspace_root: Path = Path("workspace/")):
    """Build and compile the mesh StateGraph.

    Args:
        llm_cfg: LLMCfg dataclass with model, base_url, api_key etc.
        workspace_root: Path to workspace directory for CrewAI tools.

    Returns:
        Compiled LangGraph ready for ainvoke().
    """
    crew = _build_crew(llm_cfg, workspace_root)

    builder = StateGraph(MeshState)
    builder.add_node("supervisor_node", supervisor_node)
    builder.add_node("crew_node", partial(crew_node, crew=crew))

    builder.add_edge(START, "supervisor_node")
    builder.add_edge("supervisor_node", "crew_node")
    builder.add_edge("crew_node", END)

    return builder.compile()
