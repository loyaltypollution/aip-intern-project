"""Load Langfuse trace exports and join them to metrics.json rows.

Traces land at `artifacts/sweeps/{sweep_stamp}/langfuse_{scenario}.ndjson`
— one trace JSON per line — written by the `run_sweep.yml` playbook.

The runners do NOT set Langfuse session_id / trace_name to the `run_id`; instead
they pass the graph's initial_state (which contains `run_id`) as the invocation
input. LangChain's CallbackHandler records that state as the trace's root input,
so `trace["input"]["run_id"]` is the canonical join key. We also fall back to
scanning observations' input fields for anything containing a `run_id`, for
robustness against SDK changes.

Usage:
    from analysis.langfuse import load_traces, trace_for_run
    df = load_traces("artifacts/langfuse/")              # directory or file
    row = trace_for_run(df, "baseline_a1b2c3d4")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd


def _extract_run_id(trace: dict[str, Any]) -> Optional[str]:
    """Best-effort extract the aip-intern run_id from a Langfuse trace dict."""
    # 1) metadata (cheapest; populated if anyone ever adds it to invoke_config)
    md = trace.get("metadata") or {}
    if isinstance(md, dict) and isinstance(md.get("run_id"), str):
        return md["run_id"]
    # 2) sessionId (by convention)
    sid = trace.get("sessionId") or trace.get("session_id")
    if isinstance(sid, str) and sid:
        return sid
    # 3) trace.input — LangChain CallbackHandler stores the graph's initial_state
    inp = trace.get("input")
    if isinstance(inp, dict) and isinstance(inp.get("run_id"), str):
        return inp["run_id"]
    # 4) trace.name (some setups name the trace after run_id)
    name = trace.get("name")
    if isinstance(name, str) and (name.startswith("baseline_") or name.startswith("mesh_")):
        return name
    # 5) deep-scan observations
    for obs in trace.get("observations") or []:
        oi = obs.get("input")
        if isinstance(oi, dict) and isinstance(oi.get("run_id"), str):
            return oi["run_id"]
    return None


def _iter_ndjson_files(path: Path) -> Iterable[Path]:
    if path.is_dir():
        yield from sorted(path.glob("*.ndjson"))
    elif path.is_file():
        yield path


def load_traces(path: str | Path) -> pd.DataFrame:
    """Load one or more Langfuse NDJSON exports into a DataFrame.

    Args:
        path: a single .ndjson file OR a directory containing *.ndjson.

    Returns:
        DataFrame with at least: run_id, trace_id, name, timestamp,
        latency_ms, input, output, observations (raw list), source_file.
        Extra trace fields are preserved as-is.
    """
    p = Path(path)
    rows: list[dict[str, Any]] = []
    for f in _iter_ndjson_files(p):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    trace = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append({
                    "run_id": _extract_run_id(trace),
                    "trace_id": trace.get("id"),
                    "name": trace.get("name"),
                    "timestamp": trace.get("timestamp"),
                    "latency_ms": trace.get("latency"),
                    "input": trace.get("input"),
                    "output": trace.get("output"),
                    "observations": trace.get("observations"),
                    "source_file": str(f),
                    "_raw": trace,
                })
    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    return df


def trace_for_run(df_traces: pd.DataFrame, run_id: str) -> Optional[pd.Series]:
    """Return the trace row whose run_id matches, or None. If multiple match,
    returns the most recent by timestamp."""
    if df_traces.empty or "run_id" not in df_traces.columns:
        return None
    hits = df_traces[df_traces["run_id"] == run_id]
    if hits.empty:
        return None
    if "timestamp" in hits.columns:
        hits = hits.sort_values("timestamp", ascending=False)
    return hits.iloc[0]
