import json
from pathlib import Path
from aip_intern.core.metrics import NodeMetrics, RunMetrics

def test_run_metrics_write_read(tmp_path):
    m = RunMetrics(run_id="baseline_abc")
    m.nodes.append(NodeMetrics(name="triage_node", latency_s=1.2, prompt_tokens=100, completion_tokens=50))
    m.nodes.append(NodeMetrics(name="brief_node", latency_s=0.8, prompt_tokens=200, completion_tokens=80))
    m.total_latency_s = 2.0
    m.total_prompt_tokens = 300
    m.total_completion_tokens = 130
    m.step_trace = ["triage_node", "brief_node"]

    out = tmp_path / "metrics.json"
    m.write(out)

    data = json.loads(out.read_text())
    assert data["run_id"] == "baseline_abc"
    assert data["total_latency_s"] == 2.0
    assert len(data["nodes"]) == 2
    assert data["nodes"][0]["name"] == "triage_node"

def test_run_metrics_error_flag(tmp_path):
    m = RunMetrics(run_id="baseline_err")
    m.error = "AgentTimeoutError: timed out"
    out = tmp_path / "metrics.json"
    m.write(out)
    data = json.loads(out.read_text())
    assert data["error"] == "AgentTimeoutError: timed out"
