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
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from aip_intern.baseline.graph import build_graph
from aip_intern.baseline.state import BaselineState
from aip_intern.core.exceptions import AIPInternError
from aip_intern.core.metrics import RunMetrics
from aip_intern.baseline.tools import get_tools, set_workspace_root
from aip_intern.core.tracing import get_callback, get_langfuse


@dataclass
class RunConfig:
    """Configuration for a sweep (baseline or mesh).

    Passed directly to run() or run_once(). One RunConfig per sweep;
    run_once() derives a fresh run_id per iteration.

    Artifact layout:
        {artifacts_dir}/sweeps/{sweep_stamp}/{scenario}/{run_id}/metrics.json
    where sweep_stamp identifies an overnight (baseline + mesh) pair.
    """

    scenario: str                   # "baseline" | "mesh" (used as folder + run_id prefix)
    sweep_stamp: str                # e.g. "20260424-1700" — shared by all runs in this sweep
    n_runs: int
    config_path: Path               # path to the YAML that produced this run
    llm_model: str
    llm_base_url: str
    llm_api_key: str
    workspace_root: Path
    artifacts_dir: Path              # repo-level artifacts root (without /sweeps)
    llm_temperature: float = 0.0
    llm_max_tokens: int = 4096
    llm_request_timeout: int = 120


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


def _build_public_trace_url(lf, trace_id: Optional[str]) -> Optional[str]:
    """Build a clickable trace URL from a trace_id.

    Prefers LANGFUSE_HOST_PUBLIC (external IP:port) over LANGFUSE_HOST (loopback
    from the CPU box) so the link is reachable from a laptop. Falls back to the
    SDK's own `get_trace_url` which uses LANGFUSE_HOST.
    """
    if lf is None or not trace_id:
        return None
    host = os.environ.get("LANGFUSE_HOST_PUBLIC")
    if host:
        return f"{host.rstrip('/')}/trace/{trace_id}"
    # SDK fallback — uses whatever host the client was constructed with.
    try:
        return lf.get_trace_url(trace_id=trace_id)
    except Exception:
        return None


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
    run_id = f"{cfg.scenario}_{cfg.sweep_stamp}_{uuid.uuid4().hex[:8]}"
    artifacts_run_dir = (
        cfg.artifacts_dir / "sweeps" / cfg.sweep_stamp / cfg.scenario / run_id
    )
    outputs_dir = artifacts_run_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    metrics = RunMetrics(
        run_id=run_id, scenario=cfg.scenario, sweep_stamp=cfg.sweep_stamp
    )
    lf = get_langfuse()
    cb = get_callback(lf)
    invoke_config = {"callbacks": [cb]} if cb else {}

    # Build LLM and tools for this run
    llm = _make_llm(cfg)
    set_workspace_root(cfg.workspace_root)
    tools = get_tools()
    graph = build_graph(llm, tools)

    # Clear workspace outputs dir before each run to prevent stale files from a
    # prior run being copied if this run writes fewer files than expected.
    workspace_outputs_dir = cfg.workspace_root / "outputs"
    if workspace_outputs_dir.exists():
        for _f in workspace_outputs_dir.iterdir():
            if _f.is_file():
                _f.unlink()

    initial_state: BaselineState = {
        "run_id": run_id,
        "task_description": "Triage citizen feedback → action brief → response drafts",
        "feedback_files": [],   # nodes use list_directory tool instead
        "policy_content": "",   # nodes use read_file tool instead
        "triage_result": None,
        "brief_result": None,
        "response_result": None,
        "error": None,
        "step_trace": [],
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }

    t0 = time.perf_counter()
    trace_id: Optional[str] = None
    try:
        # Open a Langfuse span so a trace_id exists for the duration of the
        # invoke. The langchain callback emits observations under that trace,
        # so the URL we stamp into metrics.json actually resolves in the UI.
        if lf is not None:
            with lf.start_as_current_span(name=f"run_once/{run_id}"):
                trace_id = lf.get_current_trace_id()
                result_state = await graph.ainvoke(initial_state, config=invoke_config)
        else:
            result_state = await graph.ainvoke(initial_state, config=invoke_config)
        metrics.total_latency_s = time.perf_counter() - t0
        metrics.step_trace = result_state.get("step_trace", [])
        metrics.total_prompt_tokens = result_state.get("prompt_tokens", 0)
        metrics.total_completion_tokens = result_state.get("completion_tokens", 0)
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

    # Flush the span to make sure the trace exists before we publish the URL.
    if lf is not None:
        try:
            lf.flush()
        except Exception:
            pass
    metrics.langfuse_trace_url = _build_public_trace_url(lf, trace_id)

    metrics_path = artifacts_run_dir / "metrics.json"
    metrics.write(metrics_path)

    # Copy workspace outputs to artifact outputs dir (tools write to workspace/outputs/)
    workspace_outputs = cfg.workspace_root / "outputs"
    if workspace_outputs.exists():
        for f in workspace_outputs.iterdir():
            if f.is_file():
                shutil.copy2(f, outputs_dir / f.name)

    return RunResult(
        run_id=run_id,
        success=success,
        error=error_msg if not success else None,
        metrics=dataclasses.asdict(metrics),
        outputs_path=outputs_dir,
        langfuse_trace_url=metrics.langfuse_trace_url,
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
