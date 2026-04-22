"""Baseline vs mesh metric comparison.

Used in Phase 2's 02_analysis.ipynb and Phase 3's failure analysis.

Usage:
    from analysis.aggregate import load_runs
    from analysis.compare import compare_phases

    baseline_df = load_runs("artifacts/", prefix="baseline")
    mesh_df = load_runs("artifacts/", prefix="mesh")
    summary = compare_phases(baseline_df, mesh_df)
    print(summary)
"""

from __future__ import annotations

import pandas as pd


def compare_phases(baseline: pd.DataFrame, mesh: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics comparing baseline vs mesh across key metrics.

    Returns a DataFrame with metrics as rows and (baseline, mesh, delta) as columns.
    Delta = mesh - baseline (positive = mesh is higher/slower/more expensive).
    """
    metrics = [
        "total_latency_s",
        "total_prompt_tokens",
        "total_completion_tokens",
        "tool_call_count",
        "message_count",
        "state_size_bytes",
        "estimated_tpm",
    ]
    rows = []
    for m in metrics:
        if m not in baseline.columns or m not in mesh.columns:
            continue
        b_mean = baseline[m].mean() if len(baseline) else float("nan")
        m_mean = mesh[m].mean() if len(mesh) else float("nan")
        rows.append({
            "metric": m,
            "baseline_mean": round(b_mean, 3),
            "mesh_mean": round(m_mean, 3),
            "delta": round(m_mean - b_mean, 3),
        })
    # Error rate
    b_err = baseline["error"].notna().mean() if len(baseline) else float("nan")
    m_err = mesh["error"].notna().mean() if len(mesh) else float("nan")
    rows.append({
        "metric": "error_rate",
        "baseline_mean": round(b_err, 3),
        "mesh_mean": round(m_err, 3),
        "delta": round(m_err - b_err, 3),
    })
    rows.append({
        "metric": "success_rate",
        "baseline_mean": round(1 - b_err, 3),
        "mesh_mean": round(1 - m_err, 3),
        "delta": round((1 - m_err) - (1 - b_err), 3),
    })
    return pd.DataFrame(rows).set_index("metric")


def summarize_benchmark_records(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise benchmark records by system and task.

    Expected columns:
      - system
      - task
      - latency_ms
      - tokens_used
      - success
      - tool_calls
      - inter_agent_message_count
      - state_transfer_size
      - estimated_tpm
    """
    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby(["system", "task"], dropna=False)
    summary = grouped.agg(
        runs=("run_id", "count"),
        success_rate=("success", "mean"),
        latency_ms_mean=("latency_ms", "mean"),
        latency_ms_p95=("latency_ms", lambda s: s.quantile(0.95)),
        tokens_mean=("tokens_used", "mean"),
        prompt_tokens_mean=("prompt_tokens", "mean"),
        completion_tokens_mean=("completion_tokens", "mean"),
        tool_calls_mean=("tool_calls", "mean"),
        message_count_mean=("inter_agent_message_count", "mean"),
        state_transfer_bytes_mean=("state_transfer_size", "mean"),
        estimated_tpm_mean=("estimated_tpm", "mean"),
        error_rate=("success", lambda s: 1 - s.mean()),
    ).reset_index()
    return summary


def build_decision_matrix(summary: pd.DataFrame) -> pd.DataFrame:
    """Convert per-task summary rows into a decision matrix.

    Returns one row per task with fields that help decide whether to use the mesh.
    """
    if summary.empty:
        return pd.DataFrame()

    rows = []
    for task in sorted(summary["task"].unique()):
        task_rows = summary[summary["task"] == task]
        single = task_rows[task_rows["system"] == "single"]
        mesh = task_rows[task_rows["system"] == "mesh"]
        if single.empty or mesh.empty:
            continue
        single_row = single.iloc[0]
        mesh_row = mesh.iloc[0]
        rows.append(
            {
                "task": task,
                "recommendation": (
                    "mesh"
                    if mesh_row["success_rate"] >= single_row["success_rate"]
                    and mesh_row["latency_ms_mean"] <= single_row["latency_ms_mean"] * 1.25
                    else "single"
                ),
                "single_success_rate": round(single_row["success_rate"], 3),
                "mesh_success_rate": round(mesh_row["success_rate"], 3),
                "single_latency_ms": round(single_row["latency_ms_mean"], 3),
                "mesh_latency_ms": round(mesh_row["latency_ms_mean"], 3),
                "single_tokens": round(single_row["tokens_mean"], 3),
                "mesh_tokens": round(mesh_row["tokens_mean"], 3),
                "single_estimated_tpm": round(single_row["estimated_tpm_mean"], 3),
                "mesh_estimated_tpm": round(mesh_row["estimated_tpm_mean"], 3),
                "mesh_overhead_latency_ms": round(
                    mesh_row["latency_ms_mean"] - single_row["latency_ms_mean"], 3
                ),
                "mesh_overhead_tokens": round(
                    mesh_row["tokens_mean"] - single_row["tokens_mean"], 3
                ),
                "mesh_overhead_tool_calls": round(
                    mesh_row["tool_calls_mean"] - single_row["tool_calls_mean"], 3
                ),
            }
        )
    return pd.DataFrame(rows)
