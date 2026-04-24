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
# # Phase 4 — GCC network-constraint analysis
#
# **Status:** placeholder. Fill in when Phase 4 runs produce artifacts.
#
# ## Research question
# Under toxiproxy-simulated GCC network conditions (latency, bandwidth cap,
# packet loss), how does each scenario's wall-clock latency and error rate
# degrade relative to an unconstrained run?
#
# ## What this notebook will do
# 1. Latency × condition grid (one row per toxiproxy profile) for both
#    scenarios; chart shows degradation curves.
# 2. Error-rate × condition grid; mesh has more round-trips so is expected
#    to be more exposed to packet loss — quantify.
# 3. Token counts should be *invariant* to network conditions; flag any
#    sweep where they aren't (indicates silent retries at the model client).
#
# ## Decision rule
# Either scenario is "GCC-viable" if under the worst modelled profile it
# completes ≥ 95% of runs within 3× the unconstrained mean latency.

# %%
# TODO: implement once artifacts/sweeps/{stamp}/phase4/* exists.
print("Phase 4 artifacts not yet produced.")
