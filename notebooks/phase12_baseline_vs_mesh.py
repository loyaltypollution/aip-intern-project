# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Baseline vs Mesh — Phase 1 & 2
#
# ## Research question
#
# On the citizen-feedback triage task, is a **LangGraph + CrewAI mesh** worth
# running in place of a **single-agent LangGraph baseline**?
#
# This notebook covers the cost half of that question using **token counts
# and latency as cost proxies** — `metrics.json` does not record a dollar
# figure, so neither does the comparison below. Brief *quality* is phase 5,
# not here. So we adopt the conservative framing:
#
# > **Assume mesh output is at best equal to baseline output on this task.**
# > Mesh is then adoptable only if its token/latency delta is bounded *and
# > stable*.
#
# ## What "stable" means, and why it comes first
#
# A cost ratio between scenarios (e.g. "mesh is 1.5× baseline on prompt tokens")
# is only meaningful if:
# 1. **Baseline is internally stable** — same inputs, same prompts, same
#    deterministic tool results should give tight, unimodal cost distributions.
#    A drifting baseline is a broken reference.
# 2. **The ratio is stable across sweeps** — if mesh/baseline flips from 3.2×
#    to 1.5× between two overnight runs with no code change, the ratio is
#    measuring environment (vLLM warm state, workspace contents, uncommitted
#    config) rather than a property of the design. That blocks any adoption
#    claim.
#
# ## Structure
#
# - §1 tests baseline internal stability.
# - §2 tests cross-sweep ratio stability (the main result).
# - §3–4 are supporting evidence for whoever wants to audit §2's numbers
#   (distributions, step traces).
# - §5 is the long-run trend across **every** sweep on disk (ignores the
#   widget selection).
# - §6 is the per-run lookup, back to respecting the widget selection.

# %% [markdown]
# ## Dashboard control
#
# The multi-select below lists every sweep on disk that has both scenarios
# present (oldest → newest, last three preselected). Pick the sweeps you want
# the rest of the notebook to analyse, then run the **"Apply selection"**
# cell immediately below to rebuild `df`. Downstream cells read `df`, so
# after applying a new selection use "Run All Below" (or Shift-Enter through
# §1–§7) to refresh the tables and plots.

# %%
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from analysis.aggregate import list_sweeps, load_runs
from analysis.plots import (
    plot_delta_trend,
    plot_latency_trend_across_sweeps,
    plot_step_trace_heatmap,
    plot_token_trend_across_sweeps,
)

ARTIFACTS = Path("../artifacts")

all_sweeps = [stamp for stamp, scen in list_sweeps(ARTIFACTS) if {"baseline", "mesh"} <= scen]
if not all_sweeps:
    raise SystemExit("No sweeps with both scenarios — can't compare.")

sweep_picker = widgets.SelectMultiple(
    options=all_sweeps,
    value=tuple(all_sweeps[-3:]),
    rows=min(12, len(all_sweeps)),
    description="sweeps",
    layout=widgets.Layout(width="420px"),
)
display(sweep_picker)

# %% [markdown]
# Running this cell rebinds `df` to the currently-picked sweeps. Re-run it
# whenever you change the selection above, then re-run §1 onward.

# %%
selected = list(sweep_picker.value)
if not selected:
    raise SystemExit("Pick at least one sweep in the widget above.")
print(f"Loading {len(selected)} sweep(s): {selected}")
df = pd.concat([load_runs(ARTIFACTS, sweep_stamp=s) for s in selected], ignore_index=True)

# %% [markdown]
# # §1 — Baseline stability
#
# Before comparing baseline to mesh we check baseline against itself. For
# each selected sweep: n runs, error rate, latency dispersion, token
# dispersion, and number of distinct step traces. The baseline graph is
# linear (`triage → brief → response`) — a single step-trace path is
# expected.
#
# **A row here is a problem if any of:**
# - `error_rate > 0` — baseline crashed on at least one run.
# - `p95/p50 > 1.3` — long latency tail suggests upstream non-determinism
#   (retries, warm-up).
# - `distinct_traces > 1` — a run skipped or added a node.

# %%
stability_rows = []
for stamp in selected:
    b = df[(df.sweep_stamp == stamp) & (df.scenario == "baseline")]
    if b.empty:
        continue
    p50 = b["total_latency_s"].median()
    p95 = b["total_latency_s"].quantile(0.95)
    distinct_traces = len({tuple(t) for t in b["step_trace"] if t})
    stability_rows.append({
        "sweep": stamp,
        "n": len(b),
        "error_rate": round(b["error"].notna().mean(), 3),
        "latency_mean_s": round(b["total_latency_s"].mean(), 2),
        "latency_p50_s": round(p50, 2),
        "latency_p95_s": round(p95, 2),
        "p95/p50": round(p95 / p50, 2) if p50 else np.nan,
        "prompt_tokens_mean": round(b["total_prompt_tokens"].mean(), 0),
        "prompt_tokens_std": round(b["total_prompt_tokens"].std(), 1),
        "distinct_traces": distinct_traces,
    })
stability = pd.DataFrame(stability_rows).set_index("sweep")
stability

# %% [markdown]
# ### Reading §1
#
# If any `error_rate > 0` or `distinct_traces > 1` in the table above,
# **stop**: the baseline for that sweep is compromised and its cost numbers
# shouldn't anchor a comparison. A mildly elevated `p95/p50` (1.3–1.5)
# usually means cold-start / KV-cache warming on the first few runs — look
# at §3 to see whether the slow runs cluster at the start of the sweep.

# %% [markdown]
# # §2 — Cross-sweep baseline vs mesh comparison
#
# The main table. One row per sweep, four cost metrics, each broken into
# `baseline mean`, `mesh mean`, and their ratio `m/b = mesh / baseline`.
#
# **How to read the ratio column:**
# - `m/b = 1.0` → mesh costs the same as baseline.
# - `m/b = 1.5` → mesh costs 50% more than baseline on that metric.
# - `m/b = 0.75` → mesh costs 25% less than baseline (unusual — flag it, don't
#   celebrate; it usually means baseline ran cold and mesh ran warm, or the
#   two scenarios didn't share the same workspace state).
#
# **How to read across rows:** if `m/b` moves by more than ~10% between
# sweeps, the mesh-vs-baseline ratio is tracking environmental drift rather
# than a property of the code. That's the blocker §1 can't catch.

# %%
METRICS = [
    ("total_latency_s", "latency (s)"),
    ("total_prompt_tokens", "prompt tokens"),
    ("total_completion_tokens", "completion tokens"),
]

rows = []
for stamp in selected:
    sub = df[df.sweep_stamp == stamp]
    b = sub[sub.scenario == "baseline"]
    m = sub[sub.scenario == "mesh"]
    row = {"sweep": stamp, "n_base": len(b), "n_mesh": len(m)}
    for col, label in METRICS:
        bm, mm = b[col].mean(), m[col].mean()
        row[f"{label} — base"] = round(bm, 1)
        row[f"{label} — mesh"] = round(mm, 1)
        row[f"{label} — m/b"] = round(mm / bm, 2) if bm else np.nan
    row["err — base"] = round(b["error"].notna().mean(), 3)
    row["err — mesh"] = round(m["error"].notna().mean(), 3)
    rows.append(row)

compare = pd.DataFrame(rows).set_index("sweep")
compare

# %% [markdown]
# ### Ratio drift (isolated view)
#
# Same `m/b` numbers as above, without the base/mesh columns crowding them.
# This is the view that answers the top-level question: **does mesh's cost
# premium hold still?**

# %%
ratio_cols = [c for c in compare.columns if c.endswith("— m/b")]
drift = compare[ratio_cols].copy()
drift.columns = [c.replace(" — m/b", "") for c in drift.columns]
drift

# %% [markdown]
# ### Reading §2 — worked example (historical, current disk state may differ)
#
# Early in this repo's history the ratios across `20260416-2336` vs the
# overnight sweep two days later looked like:
#
# | metric            | 16-2336 m/b | next-sweep m/b |
# | ----------------- | ----------: | -------------: |
# | latency           |        1.47 |           0.75 |
# | prompt tokens     |        3.21 |           1.55 |
# | completion tokens |        2.92 |           1.67 |
#
# Every ratio roughly halved between sweeps, and baseline got *slower* in
# absolute terms (37 s → 60 s). `git log` between the two sweeps showed
# **zero commits touching `src/`**, so the flip can't be a code change.
# Leading hypotheses (to be eliminated before trusting any adoption
# decision):
# - **Workspace pollution via MCP tools** — Apr 16 sweep's workspace may have
#   contained leftover files from prior runs, inflating tool-listing context
#   in the prompt. Apr 18 ran cleaner. (See `src/aip_intern/core/tools.py`
#   MCP factory.)
# - **vLLM warm state** — sweep 16 may have begun cold.
# - **Uncommitted config edits** at sweep time.
#
# None of these are settled. Flag them in the deliverable; don't claim an
# adoption verdict until one sweep-over-sweep pair holds still.

# %% [markdown]
# # §3 — Per-metric distributions
#
# Boxplots for every (sweep, scenario) pair. Useful for spotting:
# - A baseline whose **median moves between sweeps** — reference drift, kills
#   cross-sweep interpretation.
# - A **mesh tail** that shrinks or grows across sweeps — the leading
#   candidate for what actually changed in the environment.
# - Within-sweep **bimodality** in either scenario — a subset of runs took a
#   structurally different path.

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (col, label) in zip(axes, METRICS):
    labels, data = [], []
    for stamp in selected:
        for scen in ("baseline", "mesh"):
            vals = df[(df.sweep_stamp == stamp) & (df.scenario == scen)][col].values
            if len(vals):
                labels.append(f"{stamp[-4:]}\n{scen[:4]}")
                data.append(vals)
    ax.boxplot(data, tick_labels=labels, showmeans=True, meanline=True)
    ax.set_title(label)
    ax.tick_params(axis="x", labelsize=8)
fig.suptitle("Cost distributions — all selected sweeps", y=1.02)
fig.tight_layout()

# %% [markdown]
# # §4 — Mesh step traces (most recent selected sweep)
#
# Baseline traces are linear by construction — not plotted. Mesh delegates
# to CrewAI and may expand the trace depending on task decomposition. Look
# for: identical rows = stable orchestration; colour shifts = a run took a
# different path (could be legitimate task-dependent branching, could be a
# retry).

# %%
latest = selected[-1]
m_latest = df[(df.sweep_stamp == latest) & (df.scenario == "mesh")]
plot_step_trace_heatmap(m_latest, title=f"mesh step trace — {latest}")

# %% [markdown]
# # §5 — Trend across **all** sweeps (ignores the widget selection)
#
# The tables and boxplots above respect the widget selection so the reader
# can focus on a handful of sweeps. This section does the opposite: it reads
# every sweep
# on disk and draws trend lines so long-run drift is visible at a glance.
# Sweeps with only one scenario (e.g. a re-run of mesh alone) still show up
# on the line for whichever scenario is present.
#
# What the three charts answer:
# - **Latency trend.** A flat baseline line means model + infra are stable
#   sweep-over-sweep. A sloping baseline line is the top thing to fix — none
#   of the cross-sweep ratios are interpretable while your reference is
#   moving.
# - **Token trend.** Same shape for prompt + completion tokens. Step changes
#   usually mean a prompt / tool-schema change between sweeps.
# - **Mesh − baseline delta.** The headline: if these lines are roughly
#   horizontal, the adoption verdict generalises across sweeps. Divergence
#   means the tradeoff changed and earlier verdicts may no longer hold.
#
# Error bars are per-sweep stdev; annotations are the sample count per
# (sweep, scenario). **§6 returns to respecting the widget selection.**

# %%
df_all = load_runs(ARTIFACTS)
plot_latency_trend_across_sweeps(df_all)

# %%
plot_token_trend_across_sweeps(df_all)

# %%
plot_delta_trend(df_all)

# %% [markdown]
# # §6 — Per-run table
#
# Every run in every selected sweep. Sorted so the slowest
# run per (sweep, scenario) is on top — handy for outlier triage.

# %%
cols = [
    "sweep_stamp", "scenario", "run_id", "total_latency_s",
    "total_prompt_tokens", "total_completion_tokens",
    "step_trace_len", "error",
]
df[cols].sort_values(
    ["sweep_stamp", "scenario", "total_latency_s"],
    ascending=[True, True, False],
).reset_index(drop=True)
