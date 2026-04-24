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
# # Phase 3 — Failure injection analysis
#
# **Status:** placeholder. Fill in when Phase 3 runs produce artifacts.
#
# ## Research question
# Under fault injection (4 failure types from the Phase 3 spec), does the
# mesh recover more reliably than the baseline, and at what cost premium?
#
# ## What this notebook will do
# 1. Per-fault-type recovery rate: baseline vs mesh (with bootstrap 95% CI
#    on the difference).
# 2. Latency-under-fault premium: mean latency conditional on fault type,
#    compared to Phase 1/2 no-fault baseline.
# 3. Step-trace divergence: mesh is expected to branch (retry / escalate)
#    under faults; baseline will either complete or fail outright. The
#    heatmap here is genuinely informative (unlike in Phase 2, where
#    baseline is linear by construction).
#
# ## Decision rule
# Mesh adoption on reliability grounds requires recovery-rate uplift of
# ≥ 20 pp on ≥ 3 of 4 fault types with CI not crossing 0.

# %%
# TODO: implement once artifacts/sweeps/{stamp}/phase3/* exists.
print("Phase 3 artifacts not yet produced.")
