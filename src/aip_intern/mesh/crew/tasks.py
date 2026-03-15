"""CrewAI Task definitions for the Phase 2 mesh.

Tasks encode the expected output and delegate to the correct agent.
"""

from __future__ import annotations

from crewai import Agent, Task


def make_triage_task(agent: Agent) -> Task:
    return Task(
        description=(
            "Read all feedback files from data/feedback/ and policy from "
            "data/policy_snippets.md. Classify each item and write "
            "outputs/triage.csv with columns: "
            "id,category,urgency,owner,summary,pii_flagged"
        ),
        expected_output="outputs/triage.csv written with one row per feedback item",
        agent=agent,
    )


def make_brief_task(agent: Agent, triage_task: Task | None = None) -> Task:
    return Task(
        description=(
            "Read outputs/triage.csv and data/policy_snippets.md. "
            "Write outputs/brief.md (executive summary, urgent items, "
            "theme analysis, recommended actions, statistics). "
            "Then write outputs/response_templates.md (one template per category)."
        ),
        expected_output=(
            "outputs/brief.md and outputs/response_templates.md written"
        ),
        agent=agent,
        context=[triage_task] if triage_task else [],
    )
