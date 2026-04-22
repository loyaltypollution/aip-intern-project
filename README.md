# aip-intern-project

Research repo comparing a single-agent LangGraph baseline against a
LangGraph+CrewAI mesh on a citizen feedback triage task.

**For interns:** Start with `notebooks/phaseN_*/00_overview.ipynb` for your phase.

## Quick Start

```bash
git clone <repo> && cd aip-intern-project
cp .env.example .env          # fill in GEMINI_API_KEY (or OPENAI_* if using vLLM/OpenAI)
pip install -e ".[dev]"
nbstripout --install --attributes .gitattributes

# Run Phase 1 baseline (requires live LLM endpoint)
python scripts/run_baseline.py --config config/baseline.yaml --dry-run

# Run single-agent vs mesh benchmark harness
python scripts/run_benchmark.py --config config/benchmark.json --dry-run

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
pytest -m integration            # requires GEMINI_API_KEY or OPENAI_* in env
```

## Notebook output policy
- `00_overview.ipynb` — outputs stripped (documentation)
- `01_run.ipynb` — outputs committed (experiment record)
- `02_analysis.ipynb` — outputs committed (analysis record)

## Benchmark-to-decision pipeline

This repo now includes a thin evaluation layer that compares the existing single-agent baseline and mesh.

- Config: [config/benchmark.json](config/benchmark.json)
- Runner: [scripts/run_benchmark.py](scripts/run_benchmark.py)
- Comparison helpers: [analysis/compare.py](analysis/compare.py) and [analysis/plots.py](analysis/plots.py)
- Notebook: [notebooks/phase2_mesh/03_decision_pipeline.ipynb](notebooks/phase2_mesh/03_decision_pipeline.ipynb)

Artifacts are written to [artifacts/benchmarks](artifacts/benchmarks):

- `benchmark_records.jsonl` — raw per-run records
- `benchmark_records.csv` — flat table for plotting
- `benchmark_summary.csv` — aggregated task/system metrics
- `decision_framework.md` — presentation-ready recommendation matrix

The benchmark contract is: same model, same decoding settings, same task set, same run count, then compare latency, token usage, success rate, and mesh overhead.

### Gemini setup

If you only have a Gemini key, set `GEMINI_API_KEY` in `.env`. The repo will use Gemini's OpenAI-compatible endpoint automatically when no `OPENAI_BASE_URL` is present.

Minimal example:

```dotenv
GEMINI_API_KEY=...
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
GEMINI_MODEL=gemini-2.5-flash
```
