"""Load artifact run directories into a pandas DataFrame.

Artifact layout (ONLY supported layout):

    {artifacts_dir}/sweeps/{sweep_stamp}/{scenario}/{run_id}/metrics.json

A `sweep_stamp` identifies one overnight pair (baseline + mesh). A
`scenario` is "baseline" or "mesh".

Usage:
    from analysis.aggregate import load_runs, list_sweeps

    list_sweeps("artifacts/")
    # -> [('20260423-2003', {'baseline', 'mesh'}), ...]

    # One pair, both scenarios
    df = load_runs("artifacts/", sweep_stamp="20260423-2003")

    # One pair, one scenario
    df = load_runs("artifacts/", sweep_stamp="20260423-2003", scenario="mesh")

    # Everything
    df = load_runs("artifacts/")
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _trace_id_from_url(url: str | None) -> str | None:
    """Extract the trace_id from a Langfuse URL (`…/trace/<id>`)."""
    if not url:
        return None
    tail = url.rsplit("/trace/", 1)
    return tail[1] if len(tail) == 2 else None


def list_sweeps(
    artifacts_dir: str | Path, by: str = "mtime"
) -> list[tuple[str, set[str]]]:
    """Return [(sweep_stamp, {scenarios_present}), ...].

    `by="mtime"` (default) sorts by directory modification time ascending — the
    most-recently-written sweep is last. This is what you want for "latest
    sweep" pickers, because stale future-dated stamps can outrank real ones
    lexicographically (see phase1-2-run-log, incident 3).

    `by="stamp"` sorts lexicographically by stamp name, which only matches
    chronological order if every stamp on disk was written in stamp order.
    """
    root = Path(artifacts_dir) / "sweeps"
    if not root.exists():
        return []
    stamp_dirs = [p for p in root.iterdir() if p.is_dir()]
    if by == "mtime":
        stamp_dirs.sort(key=lambda p: p.stat().st_mtime)
    elif by == "stamp":
        stamp_dirs.sort(key=lambda p: p.name)
    else:
        raise ValueError(f"list_sweeps(by=...) must be 'mtime' or 'stamp', got {by!r}")
    out: list[tuple[str, set[str]]] = []
    for stamp_dir in stamp_dirs:
        scenarios = {p.name for p in stamp_dir.iterdir() if p.is_dir()}
        out.append((stamp_dir.name, scenarios))
    return out


def latest_sweep(artifacts_dir: str | Path) -> str | None:
    """Return the most-recently-written sweep_stamp on disk, or None."""
    pairs = list_sweeps(artifacts_dir, by="mtime")
    return pairs[-1][0] if pairs else None


def load_runs(
    artifacts_dir: str | Path,
    sweep_stamp: str | None = None,
    scenario: str | None = None,
) -> pd.DataFrame:
    """Load metrics.json files under `artifacts/sweeps/{sweep_stamp}/{scenario}/`.

    Returns one row per run with columns: run_id, scenario, sweep_stamp,
    total_latency_s, total_prompt_tokens, total_completion_tokens, error,
    step_trace, step_trace_len, message_count, state_size_bytes,
    langfuse_trace_id, langfuse_trace_url.

    `langfuse_trace_url` points at an ephemeral EC2 host and is dead once
    `terraform destroy` has run. `langfuse_trace_id` is the stable handle that
    cross-references the preserved `langfuse_<scenario>.ndjson` dumps.
    """
    root = Path(artifacts_dir) / "sweeps"
    if not root.exists():
        return pd.DataFrame()

    stamp_glob = sweep_stamp if sweep_stamp else "*"
    scenario_glob = scenario if scenario else "*"
    pattern = f"{stamp_glob}/{scenario_glob}/*/metrics.json"

    rows = []
    mismatches: list[tuple[str, str, str]] = []
    for metrics_path in sorted(root.glob(pattern)):
        try:
            m = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            continue
        parts = metrics_path.relative_to(root).parts  # (stamp, scenario, run_id, metrics.json)
        dir_stamp, dir_scenario = parts[0], parts[1]
        # Directory name is authoritative — if a sweep folder was renamed, the
        # embedded metrics.json["sweep_stamp"] can lag. Warn so it's visible.
        json_stamp = m.get("sweep_stamp")
        if json_stamp and json_stamp != dir_stamp:
            mismatches.append((m.get("run_id", "?"), json_stamp, dir_stamp))
        rows.append({
            "run_id": m.get("run_id"),
            "sweep_stamp": dir_stamp,
            "scenario": m.get("scenario") or dir_scenario,
            "total_latency_s": m.get("total_latency_s", 0),
            "total_prompt_tokens": m.get("total_prompt_tokens", 0),
            "total_completion_tokens": m.get("total_completion_tokens", 0),
            "error": m.get("error"),
            "step_trace": m.get("step_trace", []),
            "step_trace_len": len(m.get("step_trace", [])),
            "message_count": m.get("message_count"),
            "state_size_bytes": m.get("state_size_bytes"),
            "langfuse_trace_id": _trace_id_from_url(m.get("langfuse_trace_url")),
            "langfuse_trace_url": m.get("langfuse_trace_url"),
        })
    if mismatches:
        import warnings
        sample = ", ".join(f"{rid}: json={js} dir={ds}" for rid, js, ds in mismatches[:3])
        warnings.warn(
            f"{len(mismatches)} metrics.json rows have sweep_stamp disagreeing with their "
            f"directory; using directory. First: {sample}",
            stacklevel=2,
        )
    return pd.DataFrame(rows)
