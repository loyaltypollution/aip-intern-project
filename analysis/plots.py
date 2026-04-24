"""Reusable plotting functions for sweep analysis notebooks.

All helpers take DataFrames produced by analysis.aggregate.load_runs()
and return matplotlib Figures so notebooks can display or save freely.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------- within-one-sweep plots ----------

def plot_latency_distribution(df: pd.DataFrame, label: str = "") -> plt.Figure:
    """Histogram + rug of end-to-end latency across runs in one scenario."""
    fig, ax = plt.subplots(figsize=(8, 4))
    lat = df["total_latency_s"].dropna()
    ax.hist(lat, bins=min(15, max(5, len(lat) // 2)), edgecolor="black", alpha=0.7)
    for v in lat:
        ax.axvline(v, color="black", alpha=0.15, ymin=0, ymax=0.05)
    ax.axvline(lat.mean(), color="crimson", linestyle="--", label=f"mean={lat.mean():.1f}s")
    ax.axvline(lat.median(), color="navy", linestyle=":", label=f"median={lat.median():.1f}s")
    ax.set_xlabel("Latency (s)")
    ax.set_ylabel("Runs")
    ax.set_title(f"Latency distribution — {label}" if label else "Latency distribution")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_latency_box_per_scenario(df: pd.DataFrame, title: str = "") -> plt.Figure:
    """Box+strip of latency grouped by scenario (one sweep_stamp)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    scenarios = sorted(df["scenario"].unique())
    data = [df[df["scenario"] == s]["total_latency_s"].dropna().values for s in scenarios]
    bp = ax.boxplot(data, labels=scenarios, showmeans=True, meanline=True)
    for i, arr in enumerate(data, start=1):
        jitter = np.random.default_rng(0).normal(0, 0.03, size=len(arr))
        ax.scatter(np.full_like(arr, i) + jitter, arr, alpha=0.5, s=16)
    ax.set_ylabel("Latency (s)")
    ax.set_title(title or "Latency by scenario")
    fig.tight_layout()
    return fig


def plot_token_cost(df: pd.DataFrame, label: str = "") -> plt.Figure:
    """Stacked bar of prompt vs completion tokens per run."""
    fig, ax = plt.subplots(figsize=(10, 4))
    idx = np.arange(len(df))
    ax.bar(idx, df["total_prompt_tokens"], label="Prompt tokens")
    ax.bar(
        idx,
        df["total_completion_tokens"],
        bottom=df["total_prompt_tokens"],
        label="Completion tokens",
    )
    ax.set_xlabel("Run index")
    ax.set_ylabel("Tokens")
    ax.set_title(f"Token cost per run — {label}" if label else "Token cost per run")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_token_vs_latency(df: pd.DataFrame, title: str = "") -> plt.Figure:
    """Scatter: total tokens (prompt+completion) vs latency, coloured by scenario."""
    fig, ax = plt.subplots(figsize=(8, 5))
    total_tokens = df["total_prompt_tokens"] + df["total_completion_tokens"]
    for scenario, g in df.groupby("scenario"):
        tt = g["total_prompt_tokens"] + g["total_completion_tokens"]
        ax.scatter(tt, g["total_latency_s"], label=scenario, alpha=0.7, s=40)
    ax.set_xlabel("Total tokens (prompt + completion)")
    ax.set_ylabel("Latency (s)")
    ax.set_title(title or "Token cost vs. wall-clock latency")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_step_trace_heatmap(df: pd.DataFrame, title: str = "") -> plt.Figure:
    """Heatmap: rows = runs, cols = step ordinal position, cell = node name hash.

    Purpose: detect runs that took a different path through the graph.
    """
    traces = [t for t in df["step_trace"].dropna().tolist() if t]
    if not traces:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No step traces", ha="center")
        return fig
    all_steps = sorted({s for t in traces for s in t})
    step_to_idx = {s: i + 1 for i, s in enumerate(all_steps)}
    max_len = max(len(t) for t in traces)
    mat = np.zeros((len(traces), max_len))
    for r, t in enumerate(traces):
        for c, s in enumerate(t):
            mat[r, c] = step_to_idx[s]
    fig, ax = plt.subplots(figsize=(max(8, max_len * 0.5), max(3, len(traces) * 0.2)))
    im = ax.imshow(mat, aspect="auto", cmap="tab20", interpolation="nearest")
    ax.set_xlabel("Step ordinal")
    ax.set_ylabel("Run index")
    ax.set_title(title or "Step trace heatmap (colour = node)")
    cbar = fig.colorbar(im, ax=ax, ticks=list(step_to_idx.values()))
    cbar.ax.set_yticklabels(list(step_to_idx.keys()), fontsize=7)
    fig.tight_layout()
    return fig


# ---------- within-one-sweep baseline-vs-mesh ----------

def plot_pair_comparison_bars(summary: pd.DataFrame) -> plt.Figure:
    """Bar chart comparing baseline vs mesh means for each metric.

    Input: DataFrame from compare_phases() with index=metric and
    columns baseline_mean / mesh_mean / delta.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(summary))
    width = 0.38
    b = ax.bar(x - width / 2, summary["baseline_mean"], width, label="baseline")
    m = ax.bar(x + width / 2, summary["mesh_mean"], width, label="mesh")
    ax.set_xticks(x)
    ax.set_xticklabels(summary.index, rotation=20, ha="right")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylabel("Value (symlog)")
    ax.set_title("Baseline vs Mesh — per-metric means")
    ax.legend()
    for bar in list(b) + list(m):
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8)
    fig.tight_layout()
    return fig


# ---------- cross-sweep trend plots ----------

def plot_latency_trend_across_sweeps(df: pd.DataFrame) -> plt.Figure:
    """Line + scatter: mean latency per (sweep_stamp, scenario), error bars = stdev."""
    g = df.groupby(["sweep_stamp", "scenario"])["total_latency_s"].agg(["mean", "std", "count"]).reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    for scenario, sub in g.groupby("scenario"):
        sub = sub.sort_values("sweep_stamp")
        ax.errorbar(sub["sweep_stamp"], sub["mean"], yerr=sub["std"].fillna(0),
                    marker="o", label=f"{scenario} (n per point shown)", capsize=4)
        for _, r in sub.iterrows():
            ax.annotate(f"n={int(r['count'])}", (r["sweep_stamp"], r["mean"]),
                        textcoords="offset points", xytext=(0, 8), fontsize=8,
                        ha="center", color="dimgray")
    ax.set_ylabel("Latency mean (s)")
    ax.set_xlabel("sweep_stamp")
    ax.set_title("Latency trend across sweeps")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_token_trend_across_sweeps(df: pd.DataFrame) -> plt.Figure:
    """Two-panel: prompt-token mean and completion-token mean per sweep/scenario."""
    g = df.groupby(["sweep_stamp", "scenario"])[
        ["total_prompt_tokens", "total_completion_tokens"]
    ].mean().reset_index()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for scenario, sub in g.groupby("scenario"):
        sub = sub.sort_values("sweep_stamp")
        ax1.plot(sub["sweep_stamp"], sub["total_prompt_tokens"], marker="o", label=scenario)
        ax2.plot(sub["sweep_stamp"], sub["total_completion_tokens"], marker="o", label=scenario)
    ax1.set_title("Prompt tokens (mean)")
    ax2.set_title("Completion tokens (mean)")
    for ax in (ax1, ax2):
        ax.set_xlabel("sweep_stamp")
        ax.tick_params(axis="x", rotation=30)
        ax.legend()
    fig.tight_layout()
    return fig


def plot_delta_trend(df: pd.DataFrame) -> plt.Figure:
    """For each sweep_stamp that has both scenarios: mesh - baseline per metric.

    Shows whether the "mesh overhead" is stable or drifting across sweeps.
    """
    piv = df.groupby(["sweep_stamp", "scenario"]).agg(
        lat=("total_latency_s", "mean"),
        prompt=("total_prompt_tokens", "mean"),
        completion=("total_completion_tokens", "mean"),
    ).unstack("scenario")
    piv = piv.dropna()  # only sweeps with both scenarios
    if piv.empty:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No sweeps with both scenarios", ha="center")
        return fig
    fig, ax = plt.subplots(figsize=(10, 5))
    for metric in ["lat", "prompt", "completion"]:
        delta = piv[(metric, "mesh")] - piv[(metric, "baseline")]
        ax.plot(delta.index, delta.values, marker="o", label=f"Δ {metric} (mesh−baseline)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("sweep_stamp")
    ax.set_ylabel("mesh − baseline")
    ax.set_title("Mesh-over-baseline delta across paired sweeps")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    return fig
