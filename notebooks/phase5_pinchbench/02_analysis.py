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
# # Phase 5 — PinchBench / OpenClaw / KiloClaw safety analysis
#
# **Status:** placeholder. Fill in when Phase 5 scoring produces artifacts.
#
# ## Research question
# This phase supplies the *quality / safety* side of the tradeoff that
# Phase 2 deliberately left open. Phase 2 assumed mesh output was at-best
# equal to baseline output and tested only cost. Here we test whether that
# assumption is generous, accurate, or pessimistic.
#
# ## What this notebook will do
# 1. Per-benchmark score distribution, baseline vs mesh.
# 2. Per-scoring-dimension breakdown (the PinchBench family scores on
#    several axes; a single composite hides disagreements).
# 3. Tie the quality result back to the Phase 2 cost budget: if mesh
#    meaningfully outperforms baseline on safety, a *higher* cost budget
#    in Phase 2 becomes defensible, and the Phase 2 verdict should be
#    re-evaluated with the new budget.
#
# ## Decision rule
# Quality gain is "meaningful" if mesh beats baseline by ≥ 0.5 stdev on
# ≥ 2 of 3 benchmarks, CI not crossing 0.

# %%
# TODO: implement once artifacts/phase5/* exists.
print("Phase 5 artifacts not yet produced.")
