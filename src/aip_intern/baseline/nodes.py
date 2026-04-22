"""Baseline LangGraph node functions.

Each node is a pure async function: (state, llm, tools) → dict of state updates.
Nodes can be tested independently using the mock_llm fixture in conftest.py.

Node execution pattern (same as phase3):
1. Build messages from state + prompt file
2. Call llm.ainvoke(messages) — tools bound if available
3. Execute any tool_calls from the response
4. Return state update dict

Error handling: unexpected exceptions are caught and re-raised as AIPInternError
subclasses so Phase 3 can intercept and score them.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from aip_intern.baseline.state import BaselineState
from aip_intern.core.exceptions import AIPInternError, MalformedOutputError

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text().strip()


def _build_context(state: BaselineState) -> str:
    """Build a context string summarising available state for downstream nodes."""
    lines = [f"Task: {state['task_description']}"]
    if state.get("triage_result"):
        lines.append(f"Triage complete: {state['triage_result']}")
    if state.get("brief_result"):
        lines.append(f"Brief complete: {state['brief_result']}")
    return "\n".join(lines)


def _extract_token_counts(response) -> tuple[int, int]:
    """Extract prompt/completion token counts from model response metadata."""
    prompt_tokens = 0
    completion_tokens = 0

    usage_metadata = getattr(response, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        prompt_tokens += int(
            usage_metadata.get("input_tokens")
            or usage_metadata.get("prompt_tokens")
            or 0
        )
        completion_tokens += int(
            usage_metadata.get("output_tokens")
            or usage_metadata.get("completion_tokens")
            or 0
        )

    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage")
        if isinstance(token_usage, dict):
            prompt_tokens += int(token_usage.get("prompt_tokens") or 0)
            completion_tokens += int(token_usage.get("completion_tokens") or 0)

    return prompt_tokens, completion_tokens


def _content_to_text(content) -> str:
    """Normalize LangChain message content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(p for p in parts if p).strip()
    return str(content) if content is not None else ""


async def _invoke_with_tools(llm, messages, tools):
    """Invoke LLM with multi-turn tool execution loop.

    Loops until the model returns a response with no tool_calls, or 10 iterations.
    Each tool result is appended as a ToolMessage so the model can use it.
    Raises MalformedOutputError if a tool call fails or if 10 iterations are
    exhausted without a final text response.
    """
    from langchain_core.messages import ToolMessage

    bound = llm.bind_tools(tools) if tools else llm
    messages = list(messages)
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for _ in range(10):
        response = await bound.ainvoke(messages)
        pt, ct = _extract_token_counts(response)
        total_prompt_tokens += pt
        total_completion_tokens += ct
        messages.append(response)
        if not (hasattr(response, "tool_calls") and response.tool_calls):
            return response, total_prompt_tokens, total_completion_tokens
        tool_map = {t.name: t for t in tools}
        for tc in response.tool_calls:
            try:
                if tc["name"] in tool_map:
                    result = await tool_map[tc["name"]].ainvoke(tc["args"])
                else:
                    result = f"Unknown tool: {tc['name']}"
            except Exception as e:
                raise MalformedOutputError(f"Tool {tc['name']} failed: {e}") from e
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    raise MalformedOutputError(
        "_invoke_with_tools: exceeded 10 iterations without a non-tool response"
    )


async def triage_node(
    state: BaselineState,
    llm: "BaseChatModel",
    tools: list["BaseTool"],
) -> dict:
    """Read feedback files and write outputs/triage.csv.

    Returns a state update dict with triage_result and updated step_trace.
    """
    try:
        t0 = time.perf_counter()
        messages = [
            SystemMessage(content=_load_prompt("triage.txt")),
            HumanMessage(content=_build_context(state)),
        ]
        response, prompt_tokens, completion_tokens = await _invoke_with_tools(
            llm, messages, tools
        )
        triage_text = _content_to_text(getattr(response, "content", "")).strip()
        node_metrics = list(state.get("node_metrics", []))
        node_metrics.append(
            {
                "name": "triage_node",
                "latency_s": time.perf_counter() - t0,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "error": None,
            }
        )
        return {
            "triage_result": "outputs/triage.csv",
            "triage_result_text": triage_text,
            "node_metrics": node_metrics,
            "step_trace": state["step_trace"] + ["triage_node"],
        }
    except AIPInternError:
        raise
    except Exception as e:
        return {
            "error": f"triage_node: {e}",
            "step_trace": state["step_trace"] + ["triage_node"],
        }


async def brief_node(
    state: BaselineState,
    llm: "BaseChatModel",
    tools: list["BaseTool"],
) -> dict:
    """Read triage data from context and write outputs/brief.md."""
    if not state.get("triage_result"):
        return {"error": "brief_node: triage_result not set — run triage_node first"}
    try:
        t0 = time.perf_counter()
        messages = [
            SystemMessage(content=_load_prompt("brief.txt")),
            HumanMessage(content=_build_context(state)),
        ]
        response, prompt_tokens, completion_tokens = await _invoke_with_tools(
            llm, messages, tools
        )
        brief_text = _content_to_text(getattr(response, "content", "")).strip()
        node_metrics = list(state.get("node_metrics", []))
        node_metrics.append(
            {
                "name": "brief_node",
                "latency_s": time.perf_counter() - t0,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "error": None,
            }
        )
        return {
            "brief_result": "outputs/brief.md",
            "brief_result_text": brief_text,
            "node_metrics": node_metrics,
            "step_trace": state["step_trace"] + ["brief_node"],
        }
    except AIPInternError:
        raise
    except Exception as e:
        return {
            "error": f"brief_node: {e}",
            "step_trace": state["step_trace"] + ["brief_node"],
        }


async def response_node(
    state: BaselineState,
    llm: "BaseChatModel",
    tools: list["BaseTool"],
) -> dict:
    """Read triage + brief from context and write outputs/response_templates.md."""
    if not state.get("brief_result"):
        return {"error": "response_node: brief_result not set — run brief_node first"}
    try:
        t0 = time.perf_counter()
        messages = [
            SystemMessage(content=_load_prompt("response.txt")),
            HumanMessage(content=_build_context(state)),
        ]
        response, prompt_tokens, completion_tokens = await _invoke_with_tools(
            llm, messages, tools
        )
        response_text = _content_to_text(getattr(response, "content", "")).strip()
        node_metrics = list(state.get("node_metrics", []))
        node_metrics.append(
            {
                "name": "response_node",
                "latency_s": time.perf_counter() - t0,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "error": None,
            }
        )
        return {
            "response_result": "outputs/response_templates.md",
            "response_result_text": response_text,
            "node_metrics": node_metrics,
            "step_trace": state["step_trace"] + ["response_node"],
        }
    except AIPInternError:
        raise
    except Exception as e:
        return {
            "error": f"response_node: {e}",
            "step_trace": state["step_trace"] + ["response_node"],
        }
