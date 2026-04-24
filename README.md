# aip-intern-project

Research repo comparing a single-agent LangGraph baseline against a
LangGraph+CrewAI mesh on a citizen feedback triage task.

**For interns:** Start with `notebooks/phaseN_*/00_overview.ipynb` for your phase.

## Quick Start

```bash
git clone <repo> && cd aip-intern-project
cp .env.example .env          # fill in OPENAI_BASE_URL + OPENAI_API_KEY
pip install -e ".[dev]"
nbstripout --install --attributes .gitattributes

# Run Phase 1 baseline (requires live LLM endpoint)
python scripts/run_baseline.py --config config/baseline.yaml --dry-run

# Open the experiment record
jupyter lab notebooks/phase1_baseline/01_run.ipynb
```

## Phase overview

| Phase | Who | What |
|-------|-----|------|
| 1 | Owner | Single-agent LangGraph baseline, 20-run sweep |
| 2 | Owner | LangGraph + CrewAI mesh, same metrics |
| 3 | Intern | Fault injection — 4 failure types, recovery scoring |
| 4 | Intern | GCC simulation via toxiproxy |
| 5 | Intern | PinchBench / OpenClaw / KiloClaw safety scoring |
| 6 | Intern | Reference repo cleanup |

## Infrastructure

```bash
# Deploy GPU + CPU EC2
terraform -chdir=infra/terraform apply

# Install dependencies
cd infra/ansible && ansible-playbook -i inventory.sh playbooks/site.yml

# Run sweep + collect artifacts
ansible-playbook -i inventory.sh playbooks/run_sweep.yml -e sweep_type=baseline
```

## Testing

```bash
pytest                           # unit tests only (no LLM required)
pytest -m integration            # requires OPENAI_BASE_URL in env
```

## Notebook layout and policy

Each phase folder contains three notebooks:

| File | Purpose | Outputs |
|---|---|---|
| `00_overview.ipynb` | Phase framing, architecture diagram | stripped (docs) |
| `01_run.ipynb` | Live execution record for one sweep | committed |
| `02_analysis.ipynb` | Multi-run analysis for the phase's question | committed |

Top-level notebooks (not tied to one phase):

| File | Purpose |
|---|---|
| `notebooks/drift.ipynb` | Cross-sweep trend monitor (baseline/mesh means, mesh-minus-baseline delta) across every sweep_stamp on disk |

### Authoring workflow (Jupytext pairing)

Every `02_analysis.ipynb` and `drift.ipynb` is paired with a sibling `.py`
(percent format). The `.py` is the edit target — diffable, reviewable, no
ipynb-output noise — and the `.ipynb` is the executed artifact.

```bash
# Edit the .py, then regenerate the .ipynb with outputs:
jupytext --to ipynb notebooks/phase2_mesh/02_analysis.py
AIP_SWEEP_STAMP=20260423-2003 \
  jupyter nbconvert --to notebook --execute --inplace notebooks/phase2_mesh/02_analysis.ipynb

# Or keep .py and .ipynb in sync after editing either side:
jupytext --sync notebooks/**/*.ipynb
```

Pick a sweep with `AIP_SWEEP_STAMP=<stamp>`; omit it to use the most recently
written sweep on disk (`analysis.aggregate.latest_sweep` — mtime-sorted, not
lexicographic).

### Deciding on mesh

`phase2_mesh/02_analysis.ipynb` is the decision notebook. It tests mesh
against a pre-registered cost budget (bounded latency/token premium over
baseline) with bootstrap 95% CIs. Quality comparison lives in Phase 5; until
then Phase 2 assumes mesh output is at-best equal to baseline output.
