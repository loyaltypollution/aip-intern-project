"""Baseline run sweep.

run(config) → list[RunResult]: execute n_runs iterations, each writing
its own artifact directory.

run_once(config) → RunResult: execute a single run.

The runner is a pure library — no argparse, no print statements.
CLI concerns live in scripts/run_baseline.py.

Usage from notebooks:
    cfg = RunConfig(...)
    results = asyncio.run(run(cfg))
"""

from __future__ import annotations

import dataclasses
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from aip_intern.baseline.graph import build_graph
from aip_intern.baseline.state import BaselineState
from aip_intern.core.exceptions import AIPInternError
from aip_intern.core.metrics import NodeMetrics, RunMetrics
from aip_intern.core.tools import create_mcp_tools
from aip_intern.core.tracing import get_callback, get_langfuse


@dataclass
class RunConfig:
    """Configuration for a baseline sweep.

    Passed directly to run() or run_once(). One RunConfig per sweep;
    run_once() derives a fresh run_id per iteration.
    """

    run_id_prefix: str
    n_runs: int
    config_path: Path               # path to the YAML that produced this run
    llm_model: str
    llm_base_url: str
    llm_api_key: str
    workspace_root: Path
    artifacts_dir: Path
    llm_temperature: float = 0.0
    llm_max_tokens: int = 4096
    llm_request_timeout: int = 120
    task_description: str = "Triage citizen feedback → action brief → response drafts"


@dataclass
class RunResult:
    """Result of a single run_once() call.

    Consumed by analysis/aggregate.py to build comparison DataFrames.
    """

    run_id: str
    success: bool
    error: Optional[str]
    metrics: dict                  # serialised RunMetrics — keys per spec Metrics Table
    outputs_path: Path             # artifacts/{run_id}/outputs/
    langfuse_trace_url: Optional[str] = None


def _make_llm(cfg: RunConfig):
    """Build the LLM client from a RunConfig. Separated for test patching."""
    from aip_intern.core.config import LLMCfg
    from aip_intern.core.llm import create_llm

    llm_cfg = LLMCfg(
        model=cfg.llm_model,
        base_url=cfg.llm_base_url,
        api_key=cfg.llm_api_key,
        temperature=cfg.llm_temperature,
        max_tokens=cfg.llm_max_tokens,
        request_timeout=cfg.llm_request_timeout,
    )
    return create_llm(llm_cfg)


async def run_once(cfg: RunConfig) -> RunResult:
    """Execute a single complete graph run.

    Creates a fresh run_id, builds the graph, ainvokes it, writes metrics.json.
    """
    run_id = f"{cfg.run_id_prefix}_{uuid.uuid4().hex[:8]}"
    artifacts_run_dir = cfg.artifacts_dir / run_id
    outputs_dir = artifacts_run_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    shared_outputs_dir = (cfg.workspace_root / "outputs").resolve()
    shared_outputs_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in ("triage.csv", "brief.md", "response_templates.md"):
        stale_path = shared_outputs_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    metrics = RunMetrics(run_id=run_id)
    lf = get_langfuse()
    cb = get_callback(lf)
    invoke_config = {"callbacks": [cb]} if cb else {}

    # Build LLM and load MCP tools for this run
    llm = _make_llm(cfg)
    tools = await create_mcp_tools(cfg.workspace_root)
    graph = build_graph(llm, tools)

    initial_state: BaselineState = {
        "run_id": run_id,
        "task_description": cfg.task_description,
        "feedback_files": [],   # nodes use MCP list_directory instead
        "policy_content": "",   # nodes use MCP read_file instead
        "triage_result": None,
        "brief_result": None,
        "response_result": None,
        "error": None,
        "step_trace": [],
        "node_metrics": [],
    }

    t0 = time.perf_counter()
    try:
        result_state = await graph.ainvoke(initial_state, config=invoke_config)
        metrics.total_latency_s = time.perf_counter() - t0
        metrics.step_trace = result_state.get("step_trace", [])
        metrics.error = result_state.get("error")

        node_metrics_raw = result_state.get("node_metrics", []) or []
        metrics.nodes = [
            NodeMetrics(
                name=str(item.get("name", "unknown_node")),
                latency_s=float(item.get("latency_s", 0.0)),
                prompt_tokens=int(item.get("prompt_tokens", 0)),
                completion_tokens=int(item.get("completion_tokens", 0)),
                error=item.get("error"),
            )
            for item in node_metrics_raw
            if isinstance(item, dict)
        ]
        metrics.total_prompt_tokens = sum(n.prompt_tokens for n in metrics.nodes)
        metrics.total_completion_tokens = sum(n.completion_tokens for n in metrics.nodes)

        fallback_text_by_name = {
            "triage.csv": result_state.get("triage_result_text") or "",
            "brief.md": result_state.get("brief_result_text") or "",
            "response_templates.md": result_state.get("response_result_text") or "",
        }

        for out_name, fallback_text in fallback_text_by_name.items():
            source_path = shared_outputs_dir / out_name
            if not source_path.exists() and str(fallback_text).strip():
                source_path.write_text(str(fallback_text).strip() + "\n")

        for source_name in ("triage.csv", "brief.md", "response_templates.md"):
            source_path = shared_outputs_dir / source_name
            if source_path.exists():
                shutil.copy2(source_path, outputs_dir / source_name)

        success = result_state.get("error") is None
        error_msg = result_state.get("error")
    except AIPInternError as e:
        metrics.total_latency_s = time.perf_counter() - t0
        success = False
        error_msg = str(e)
        metrics.error = error_msg
    except Exception as e:
        metrics.total_latency_s = time.perf_counter() - t0
        success = False
        error_msg = f"Unexpected error: {e}"
        metrics.error = error_msg

    metrics_path = artifacts_run_dir / "metrics.json"
    metrics.write(metrics_path)

    return RunResult(
        run_id=run_id,
        success=success,
        error=error_msg if not success else None,
        metrics=dataclasses.asdict(metrics),
        outputs_path=outputs_dir,
    )


async def run(cfg: RunConfig) -> list[RunResult]:
    """Execute cfg.n_runs sequential runs. Returns all RunResult objects."""
    results = []
    for i in range(cfg.n_runs):
        print(f"  Run {i + 1}/{cfg.n_runs}...", end=" ", flush=True)
        result = await run_once(cfg)
        status = "OK" if result.success else f"ERR: {result.error}"
        print(status)
        results.append(result)
    return results
