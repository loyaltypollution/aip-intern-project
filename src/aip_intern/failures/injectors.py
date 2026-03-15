"""Fault injection library — Phase 3 implementation target.

These stubs define the injection interface. Phase 3 interns fill in the bodies.

Each injector wraps a node function and intercepts execution to simulate
a specific failure condition. The LangGraph graph does not need to change —
injectors are applied at graph construction time by replacing node functions.

Example (Phase 3 usage):
    from aip_intern.failures.injectors import inject_timeout
    patched_triage = inject_timeout(triage_node, after_seconds=5.0)
    builder.add_node("triage_node", partial(patched_triage, llm=llm, tools=tools))
"""

from __future__ import annotations

from typing import Callable

from aip_intern.core.exceptions import (
    AgentTimeoutError,  # noqa: F401
    CheckpointLostError,  # noqa: F401
    ContextOverflowError,  # noqa: F401
    MalformedOutputError,  # noqa: F401
)


def inject_timeout(node_fn: Callable, after_seconds: float = 5.0) -> Callable:
    """Wrap a node function to raise AgentTimeoutError after after_seconds.

    Args:
        node_fn: The original async node function to wrap.
        after_seconds: Time budget before timeout is raised.

    Returns:
        Wrapped async function with the same signature as node_fn.

    Phase 3 intern: implement using asyncio.wait_for or a threading.Timer.
    Measure recovery_time in scoring.py from exception raise to graph END.
    """
    ...


def inject_malformed_json(node_fn: Callable, fail_on_call: int = 1) -> Callable:
    """Wrap a node to raise MalformedOutputError on the Nth tool call.

    Args:
        node_fn: The original async node function.
        fail_on_call: Which tool call invocation to corrupt (1-indexed).

    Returns:
        Wrapped async function.

    Phase 3 intern: intercept tool call responses, replace JSON with
    invalid content on the specified call, verify MalformedOutputError is raised.
    """
    ...


def inject_checkpoint_loss(node_fn: Callable) -> Callable:
    """Wrap a node to raise CheckpointLostError after state is written.

    Simulates a LangGraph checkpoint that is written but cannot be read back
    (e.g., disk full, permissions error, corrupt state).

    Phase 3 intern: use LangGraph's checkpointer interface or monkeypatch
    the state serialiser to raise on read.
    """
    ...


def inject_context_overflow(node_fn: Callable, token_limit: int = 1000) -> Callable:
    """Wrap a node to raise ContextOverflowError when prompt exceeds token_limit.

    Phase 3 intern: count tokens before ainvoke(), raise ContextOverflowError
    if over limit. Use tiktoken for token counting.
    """
    ...
