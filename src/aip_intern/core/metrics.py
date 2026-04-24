"""Metric collection for aip-intern-project.

Stores per-node and per-run metrics. Writes to artifacts/{run_id}/metrics.json.

Metric keys (see spec Metrics Table):
  - end_to_end_latency_s, per_node_latency (in nodes[].latency_s)
  - total_prompt_tokens, total_completion_tokens
  - error (bool-equivalent via error field being non-None)
  - step_trace: ordered list of node names visited
  - message_count, state_size_bytes (Phase 2 mesh only — leave None for baseline)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class NodeMetrics:
    """Metrics for a single node invocation."""

    name: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    error: Optional[str] = None


@dataclass
class RunMetrics:
    """Metrics for a single complete run (one ainvoke call).

    Instantiate at the start of run_once(), populate as nodes execute,
    call write() at the end.
    """

    run_id: str
    nodes: list[NodeMetrics] = field(default_factory=list)
    total_latency_s: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    step_trace: list[str] = field(default_factory=list)
    error: Optional[str] = None
    # Phase 2 mesh metrics — None for baseline
    message_count: Optional[int] = None
    state_size_bytes: Optional[int] = None
    # Sweep context — duplicated from parent path for self-contained metrics.
    scenario: Optional[str] = None
    sweep_stamp: Optional[str] = None

    def write(self, path: Path) -> None:
        """Write metrics to a JSON file. Creates parent directories if needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
