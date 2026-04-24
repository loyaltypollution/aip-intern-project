# /run-phase1-2 — Phase 1 & 2 Sweep Pipeline Orchestration

You orchestrate the Phase 1 (LangGraph baseline) and Phase 2 (LangGraph + CrewAI
mesh) benchmark sweeps. **All sweep execution is remote.** Sweeps run on a CPU
box (provisioned by Terraform) against a vLLM endpoint on a co-located GPU box,
both driven by Ansible. The local machine is the **control plane only** — it
runs Terraform/Ansible and receives rsync'd artifacts.

All commands assume cwd = repo root. If you're unsure, `cd "$(git rev-parse --show-toplevel)"`.

---

## Phase 0 — Infra Deploy

Check whether infra is already up:

```bash
(cd infra/terraform && terraform output -raw gpu_public_ip 2>/dev/null) \
  && echo "INFRA UP" || echo "INFRA NOT DEPLOYED"
```

If `INFRA UP`, jump to **vLLM health check**. Otherwise provision:

```bash
MY_IP=$(curl -s ifconfig.me)
echo "SSH will be locked to ${MY_IP}/32."
echo "If you'll roam networks, decide now: widen before apply, or re-apply"
echo "later from each new IP. Do NOT silently use 0.0.0.0/0."

cat > infra/terraform/terraform.tfvars <<EOF
allowed_ssh_cidr = "${MY_IP}/32"
EOF

(cd infra/terraform && terraform init && terraform apply -auto-approve)

# Strip CRLF from infra files (breaks ansible/terraform on macOS)
find infra -type f \( -name "*.tf" -o -name "*.yml" -o -name "*.yaml" \
      -o -name "*.sh" -o -name "*.j2" -o -name "*.cfg" \) \
  -exec perl -pi -e 's/\r\n/\n/g' {} +
chmod +x infra/ansible/inventory.sh

# Bootstrap GPU + CPU + Langfuse (~35–50 min)
(cd infra/ansible && ansible-playbook playbooks/site.yml)
```

### vLLM model and tool-call flags

`infra/ansible/group_vars/all.yml` pins the model and vLLM args. Current values
use `Qwen/Qwen2.5-32B-Instruct` with `--enable-auto-tool-choice --tool-call-parser hermes`.
Do not change these without verifying the model exists on HuggingFace and the
parser name is valid in the installed vLLM version — wrong parser name → every
LLM call returns HTTP 400 and `triage_node` silently fails, leaving
`step_trace: ["triage_node"]` and 0 tokens in every artifact.

### vLLM health check (up to 15 min for model load)

```bash
KEY=infra/ssh/aip-intern-generated-key.pem
GPU_IP=$(cd infra/terraform && terraform output -raw gpu_public_ip)

until ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ConnectTimeout=8 -i $KEY ubuntu@$GPU_IP \
    'curl -sf --max-time 5 http://127.0.0.1:8000/v1/models >/dev/null 2>&1'; do
  sleep 20; echo "waiting for vLLM..."
done && echo "vLLM UP"
```

If vLLM crashes in a loop (~60s between crashes): stale CUDA state. Reboot:
```bash
ssh -i $KEY ubuntu@$GPU_IP 'sudo reboot'
sleep 90 && ssh -i $KEY ubuntu@$GPU_IP 'sudo systemctl start vllm'
# then re-run the health loop
```

---

## Pre-Flight Checks (local control plane only)

The sweep does not need `.env` locally — `run_sweep.yml` exports
`OPENAI_BASE_URL` / `OPENAI_API_KEY` on the CPU box at runtime.

```bash
command -v ansible-playbook >/dev/null || { echo "STOP: install ansible"; exit 1; }
test -f infra/terraform/terraform.tfstate || { echo "STOP: run Phase 0 first"; exit 1; }
(cd infra/ansible && ./inventory.sh --list | python3 -m json.tool | head -30)
mkdir -p artifacts/ docs/
python3 -c "import aip_intern, pandas" 2>/dev/null || pip3 install -e ".[dev]"
```

---

## Artifact Layout & Pair Stamp

One `sweep_stamp` covers the (baseline + mesh) pair from a single
`/run-phase1-2` invocation. Pick a stamp at the start and pass it to both
ansible runs so their data co-locates:

    artifacts/sweeps/{sweep_stamp}/baseline/{run_id}/metrics.json
    artifacts/sweeps/{sweep_stamp}/mesh/{run_id}/metrics.json
    artifacts/sweeps/{sweep_stamp}/langfuse_baseline.ndjson
    artifacts/sweeps/{sweep_stamp}/langfuse_mesh.ndjson

```bash
export SWEEP_STAMP=$(date -u +%Y%m%d-%H%M)
echo "This run's pair stamp: $SWEEP_STAMP"
```

Gate on **successful** artifacts (failed runs can leave `error: null` with a
truncated `step_trace`):

```bash
count_ok() {   # args: sweep_stamp scenario
  local stamp=$1 scenario=$2 n=0
  for d in artifacts/sweeps/${stamp}/${scenario}/*/; do
    [ -f "${d}metrics.json" ] || continue
    python3 -c "
import json, sys
m = json.load(open('${d}metrics.json'))
sys.exit(0 if len(m.get('step_trace', [])) > 1 and not m.get('error') else 1)
" 2>/dev/null && n=$((n+1))
  done
  echo $n
}
# Example:  BASELINE_OK=$(count_ok "$SWEEP_STAMP" baseline)
```

---

## Pipeline

Track outcomes in `docs/phase1-2-run-log.md`.

### Stage 1: Baseline Sweep

```bash
LOG=$(pwd)/artifacts/ansible_baseline.log
set -o pipefail   # without this, tee masks ansible failures
(cd infra/ansible && ansible-playbook playbooks/run_sweep.yml \
  -e scenario=baseline -e sweep_type=baseline \
  -e sweep_stamp="$SWEEP_STAMP" \
  -e skip_aiperf=true -e skip_nemo_eval=true) \
  2>&1 | tee "$LOG"
```

The playbook: starts vLLM → probes `/v1/models` + inference (up to ~15min) →
uses the passed `sweep_stamp` (or fallback to CPU `date`) and exports
`AIP_SWEEP_STAMP` → runs `scripts/run_baseline.py --config config/baseline.yaml`
on the CPU box → stops vLLM → rsyncs
`artifacts/sweeps/{sweep_stamp}/{scenario}/` + `langfuse_{scenario}.ndjson`
back to local.

**NB on tee path:** use the absolute `$LOG` from above — `tee` runs in the
parent shell's cwd, not inside the `(cd ...)` subshell.

Success: tail of `ansible_baseline.log` shows `Complete: 20/20 runs succeeded.`
Partial success (≥ 1 OK) is fine — continue.

### Stage 2: Mesh Sweep

```bash
LOG=$(pwd)/artifacts/ansible_mesh.log
set -o pipefail
(cd infra/ansible && ansible-playbook playbooks/run_sweep.yml \
  -e scenario=mesh -e sweep_type=mesh \
  -e sweep_stamp="$SWEEP_STAMP" \
  -e skip_aiperf=true -e skip_nemo_eval=true) \
  2>&1 | tee "$LOG"
```

Same contract as Stage 1; mesh metrics (`message_count`, `state_size_bytes`)
appear in each `metrics.json`. Reusing `$SWEEP_STAMP` from Stage 1 is what
makes this a **paired** sweep.

### Stage 3: Analysis & Comparison (local)

Always run — read-only and cheap. Executes the two notebooks that ship with
the repo; both are parameterized and side-effect-free.

```bash
# Per-pair deep dive, parameterized via env var read by the first cell.
AIP_SWEEP_STAMP="$SWEEP_STAMP" jupyter nbconvert \
  --to notebook --execute notebooks/sweep_analysis.ipynb \
  --output "sweep_analysis_${SWEEP_STAMP}.ipynb" \
  --ExecutePreprocessor.timeout=600

# Cross-sweep trend across every pair on disk. No env var needed.
jupyter nbconvert \
  --to notebook --execute notebooks/cross_sweep.ipynb \
  --output "cross_sweep_${SWEEP_STAMP}.ipynb" \
  --ExecutePreprocessor.timeout=600

# Quick CLI summary also goes to docs/ for the run log.
TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT=docs/phase1-2-comparison-${TS}.md
python3 - <<EOF | tee "${OUT}"
from analysis.aggregate import load_runs
from analysis.compare import compare_phases
stamp = "${SWEEP_STAMP}"
baseline_df = load_runs("artifacts/", sweep_stamp=stamp, scenario="baseline")
mesh_df     = load_runs("artifacts/", sweep_stamp=stamp, scenario="mesh")
print(f"pair {stamp!r}: baseline={len(baseline_df)} mesh={len(mesh_df)}")
if not (baseline_df.empty and mesh_df.empty):
    print(compare_phases(baseline_df, mesh_df).to_string())
EOF
echo "Comparison written to ${OUT}"
```

### Stage 3b: Trace Inspection

Three trace levels, in order of "answers the question fastest":

**Per-run Langfuse URL** (requires infra alive). Every `metrics.json` has a
`langfuse_trace_url` scoped to `LANGFUSE_HOST_PUBLIC`:
```bash
python3 -c "
import json, glob, sys
stamp = sys.argv[1]
for p in sorted(glob.glob(f'artifacts/sweeps/{stamp}/mesh/*/metrics.json'))[:5]:
    m = json.load(open(p))
    print(m['run_id'], m.get('langfuse_trace_url'))
" "$SWEEP_STAMP"
```
Click a URL → LangGraph node spans, per-LLM-call prompt/completion, CrewAI task
tree, per-node latency. The right tool for "why is *this* run slow" and "what
did the agents actually say to each other".

**Offline NDJSON export** (infra can be torn down) — now co-located with
metrics under the pair dir:
```bash
ls artifacts/sweeps/${SWEEP_STAMP}/
jq '.observations | length' \
   artifacts/sweeps/${SWEEP_STAMP}/langfuse_mesh.ndjson | head
```

**Cross-sweep diff** — stamps are `YYYYMMDD-HHMM`; `git log --since` needs ISO:
```bash
stamp_to_iso() {   # "20260423-1917" -> "2026-04-23T19:17Z"
  local s=$1
  echo "${s:0:4}-${s:4:2}-${s:6:2}T${s:9:2}:${s:11:2}Z"
}
git log --since="$(stamp_to_iso 20260423-1917)" \
        --until="$(stamp_to_iso 20260424-1700)" --oneline \
  -- src/aip_intern/mesh/ src/aip_intern/baseline/ config/
```
Caveat: if a run was executed with uncommitted changes, the commit is a lower
bound; the working-tree snapshot at sweep time is not recorded. Treat
unexplained cross-sweep deltas as "measurement contract changed" until pinned
to a specific diff.

### Stage 4: Teardown Reminder

**Do not auto-teardown.** Print uptime:

```bash
APPLIED=$(stat -f %m infra/terraform/terraform.tfstate 2>/dev/null \
       || stat -c %Y infra/terraform/terraform.tfstate 2>/dev/null)
[ -n "$APPLIED" ] || { echo "couldn't stat terraform.tfstate"; exit 1; }
HOURS=$(( ($(date +%s) - APPLIED) / 3600 ))
echo "REMINDER: GPU+CPU boxes running ~${HOURS}h."
echo "Teardown:  (cd infra/terraform && terraform destroy -auto-approve)"
[ "$HOURS" -ge 8 ] && echo "!! GPU-hour cost is material — teardown is overdue."
```

---

## Error Recovery

When a stage fails, spawn diagnostic + fix as `subagent_type: general-purpose`
(fix needs Edit; `Explore` can't write).

**Diagnostic prompt:**
```
Read this error from {stage} and identify the root cause. Return:
(1) Root cause in one sentence
(2) Specific file/config/command to change
(3) Confidence 1–10

Error:
[paste tail of ansible_{sweep_type}.log — the failed task's stdout/stderr,
 plus `journalctl -u vllm` output if vLLM is implicated]
```

**Fix prompt:**
```
Apply the minimal fix. Do not refactor.
Do not modify workspace/data/, artifacts/, or terraform.tfstate.
Root cause: [from diagnostic]
Fix target: [from diagnostic]
```

Apply fix if confidence ≥ 7 and safe. Retry the stage up to twice with
backoff (60s, then 180s). If both retries fail: skip, log, continue.

### Known Issues

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| `vLLM /v1/models` probe times out ~1h | Model still loading OR OOM on 32B weights | `ssh $GPU 'journalctl -u vllm -n 200'`; if OOM, reduce `--max-model-len` in `group_vars/all.yml` |
| All runs end with `step_trace: ["triage_node"]` and 0 tokens | Tool-call parser mismatch, 400s swallowed | Confirm `--tool-call-parser` value is valid for installed vLLM version |
| `CrewAI kickoff timeout` | Remote vLLM slow under concurrent load | Raise `llm.request_timeout` in `config/mesh.yaml` |
| `load_runs()` empty DataFrame | rsync didn't bring artifacts back | Check the "Sync sweep artifacts" task in `ansible_{sweep}.log`; re-run the playbook (UUIDs differ, no clobber) |

---

## Run Log Format

Append to `docs/phase1-2-run-log.md`:

```markdown
## Run: YYYY-MM-DD HH:MM UTC

| Stage | Status | Notes |
|-------|--------|-------|
| 1. Baseline Sweep | ✅ / ❌ / ⏭ skipped | N/20 succeeded |
| 2. Mesh Sweep     | ✅ / ❌ / ⏭ skipped | N/20 succeeded |
| 3. Analysis       | ✅ / ❌ | → docs/phase1-2-comparison-<TS>.md |
| 4. Teardown       | reminded / executed | uptime ~Xh |

### Errors
<!-- For each ❌: root cause + attempted fix -->
```

---

## Key Paths

| What | Path |
|------|------|
| Terraform state / vars (gitignored) | `infra/terraform/terraform.tfstate`, `terraform.tfvars` |
| Ansible | `infra/ansible/playbooks/site.yml` (bootstrap), `run_sweep.yml` (execute) |
| Ansible vars (model, vllm_args, scenarios) | `infra/ansible/group_vars/all.yml` |
| SSH key | `infra/ssh/aip-intern-generated-key.pem` |
| Sweep scripts (remote) | `~/aip-intern-project/scripts/run_{baseline,mesh}.py` |
| Workspace tools (local Python) | `src/aip_intern/{baseline,mesh}/tools.py` |
| Artifacts (after rsync) | `artifacts/sweeps/{sweep_stamp}/{scenario}/{run_id}/` |
| Langfuse NDJSON exports | `artifacts/sweeps/{sweep_stamp}/langfuse_{scenario}.ndjson` |
| Notebooks (parameterized) | `notebooks/sweep_analysis.ipynb`, `notebooks/cross_sweep.ipynb` |
| Analysis | `analysis/aggregate.py`, `analysis/compare.py` |
| Run log | `docs/phase1-2-run-log.md` |

## Metrics Captured

| Metric | Phase 1 | Phase 2 | Notes |
|--------|---------|---------|-------|
| `total_latency_s` | ✓ | ✓ | Wall-clock of full graph invocation |
| `total_prompt_tokens` | ✓ | ✓ | |
| `total_completion_tokens` | ✓ | ✓ | |
| `step_trace` | ✓ | ✓ | Ordered node names |
| `error` | ✓ | ✓ | Non-null string on failure |
| `message_count` | — | ✓ | Inter-agent messages in crew |
| `state_size_bytes` | — | ✓ | Serialised LangGraph state at crew_node return |

## Langfuse (self-hosted on CPU box)

Runs in Docker on the CPU box via `playbooks/setup_langfuse.yml` (included by
`site.yml`). `run_sweep.yml` reads `infra/langfuse/credentials.env` and injects
`LANGFUSE_HOST` + `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` into the sweep
subprocess. If the credentials file is absent the sweep runs without tracing
— no error.

**UI access:**
```bash
CPU_IP=$(cd infra/terraform && terraform output -raw cpu_public_ip)
echo "http://${CPU_IP}:3000"
cat infra/langfuse/credentials.env   # UI user + password + API keys
```

**First-boot seed caveat:** `LANGFUSE_INIT_*` only applies against an empty
Postgres. If you keep the DB volume across restarts, later secret changes are
silently ignored. To rotate keys: `docker compose down -v` on the CPU box *and*
`rm -rf infra/langfuse/secrets/ infra/langfuse/credentials.env` locally, then
re-run the playbook. Clickhouse first-boot migrations take 2–5 min — the
playbook waits up to 10 min for `/api/public/health`.
