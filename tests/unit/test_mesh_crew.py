from __future__ import annotations

import pytest
from crewai import LLM, Agent, Task

from aip_intern.mesh.crew.agents import (
    make_brief_response_specialist,
    make_triage_specialist,
)
from aip_intern.mesh.crew.tasks import make_brief_task, make_triage_task


@pytest.fixture
def stub_llm() -> LLM:
    """A crewai LLM stub that passes validation without real API credentials."""
    return LLM(model="openai/gpt-4o-mini", api_key="test-key")


def test_make_triage_specialist_returns_agent(stub_llm):
    agent = make_triage_specialist(llm=stub_llm, tools=[])
    assert isinstance(agent, Agent)
    assert "triage" in agent.role.lower()


def test_make_brief_response_specialist_returns_agent(stub_llm):
    agent = make_brief_response_specialist(llm=stub_llm, tools=[])
    assert isinstance(agent, Agent)


def test_make_triage_task_returns_task(stub_llm):
    agent = make_triage_specialist(llm=stub_llm, tools=[])
    task = make_triage_task(agent=agent)
    assert isinstance(task, Task)
    assert task.agent is agent


def test_make_brief_task_returns_task(stub_llm):
    agent = make_brief_response_specialist(llm=stub_llm, tools=[])
    task = make_brief_task(agent=agent)
    assert isinstance(task, Task)
    assert task.agent is agent
