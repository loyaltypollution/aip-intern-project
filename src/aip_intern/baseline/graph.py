"""LangGraph StateGraph for the Phase 1 baseline.

Graph topology:
    START → triage_node → brief_node → response_node → END

build_graph() returns a compiled graph ready for ainvoke().
The LLM and tools are injected at construction time (not stored in state)
so they can be swapped for mocks in tests.

Usage:
    graph = build_graph(llm, tools)
    result = await graph.ainvoke(initial_state, config=invoke_config)
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from aip_intern.baseline.nodes import brief_node, response_node, triage_node
from aip_intern.baseline.state import BaselineState

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool


def build_graph(llm: "BaseChatModel", tools: list["BaseTool"]):
    """Build and compile the baseline StateGraph.

    Args:
        llm: Chat model instance (real or mock).
        tools: workspace filesystem tools. Pass [] for mock/unit test runs.

    Returns:
        Compiled LangGraph ready for ainvoke().
    """
    builder = StateGraph(BaselineState)

    # Bind llm + tools into each node function
    builder.add_node("triage_node", partial(triage_node, llm=llm, tools=tools))
    builder.add_node("brief_node", partial(brief_node, llm=llm, tools=tools))
    builder.add_node("response_node", partial(response_node, llm=llm, tools=tools))

    builder.add_edge(START, "triage_node")
    builder.add_edge("triage_node", "brief_node")
    builder.add_edge("brief_node", "response_node")
    builder.add_edge("response_node", END)

    return builder.compile()
