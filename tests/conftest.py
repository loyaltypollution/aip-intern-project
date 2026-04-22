"""Shared test fixtures.

mock_llm: deterministic ChatModel for unit tests — no network required.
live_llm: real LLM from env — skipped if no OpenAI/Gemini endpoint is set.
sample_baseline_state: minimal valid BaselineState for node tests.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


DEFAULT_GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


def _live_llm_env() -> tuple[str | None, str | None, str]:
    base_url = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("GEMINI_BASE_URL")
        or os.environ.get("GOOGLE_BASE_URL")
    )
    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not base_url and api_key:
        base_url = DEFAULT_GEMINI_OPENAI_BASE_URL
    model = os.environ.get(
        "OPENAI_MODEL",
        os.environ.get("GEMINI_MODEL", "Qwen/Qwen3-32B-Instruct"),
    )
    return base_url, api_key, model


@pytest.fixture
def mock_llm():
    """A mock ChatModel that returns a deterministic AIMessage.

    Usage in tests:
        result = await triage_node(state, llm=mock_llm, tools=[])

    Override the return value for specific tests:
        mock_llm.ainvoke.return_value = AIMessage(content="custom response")
    """
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=AIMessage(
            content="mock response",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
        )
    )
    llm.bind_tools = MagicMock(return_value=llm)
    return llm


@pytest.fixture
def live_llm():
    """Real LLM from env. Skipped if no OpenAI/Gemini base URL is available."""
    base_url, api_key, model = _live_llm_env()
    if not base_url:
        pytest.skip("No OpenAI/Gemini base URL set — skipping live LLM test")
    from aip_intern.core.config import LLMCfg
    from aip_intern.core.llm import create_llm
    cfg = LLMCfg(
        model=model,
        base_url=base_url,
        api_key=api_key or "not-needed",
    )
    return create_llm(cfg)


@pytest.fixture
def sample_baseline_state():
    """Minimal valid BaselineState for node unit tests."""
    return {
        "run_id": "test_run_001",
        "task_description": "Triage citizen feedback → action brief → response drafts",
        "error": None,
        "step_trace": [],
        "node_metrics": [],
        "triage_result": None,
        "brief_result": None,
        "response_result": None,
    }
