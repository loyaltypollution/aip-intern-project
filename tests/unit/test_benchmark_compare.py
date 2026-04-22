from __future__ import annotations

from analysis.compare import build_decision_matrix, pd as pd_module, summarize_benchmark_records


def test_summarize_benchmark_records_groups_by_system_and_task():
    df = pd_module.DataFrame(
        [
            {
                "run_id": "single_simple_1",
                "system": "single",
                "task": "simple",
                "latency_ms": 100.0,
                "tokens_used": 1000,
                "prompt_tokens": 700,
                "completion_tokens": 300,
                "success": True,
                "tool_calls": 2,
                "inter_agent_message_count": None,
                "state_transfer_size": None,
                "estimated_tpm": 600.0,
            },
            {
                "run_id": "mesh_simple_1",
                "system": "mesh",
                "task": "simple",
                "latency_ms": 180.0,
                "tokens_used": 1400,
                "prompt_tokens": 1000,
                "completion_tokens": 400,
                "success": True,
                "tool_calls": 6,
                "inter_agent_message_count": 4,
                "state_transfer_size": 512,
                "estimated_tpm": 900.0,
            },
        ]
    )

    summary = summarize_benchmark_records(df)

    assert len(summary) == 2
    assert set(summary["system"]) == {"single", "mesh"}
    assert set(summary["task"]) == {"simple"}


def test_build_decision_matrix_prefers_single_for_simple_heavier_mesh():
    summary = pd_module.DataFrame(
        [
            {
                "system": "single",
                "task": "simple",
                "runs": 20,
                "success_rate": 1.0,
                "latency_ms_mean": 100.0,
                "latency_ms_p95": 110.0,
                "tokens_mean": 1000.0,
                "prompt_tokens_mean": 700.0,
                "completion_tokens_mean": 300.0,
                "tool_calls_mean": 2.0,
                "message_count_mean": 0.0,
                "state_transfer_bytes_mean": 0.0,
                "estimated_tpm_mean": 600.0,
                "error_rate": 0.0,
            },
            {
                "system": "mesh",
                "task": "simple",
                "runs": 20,
                "success_rate": 1.0,
                "latency_ms_mean": 180.0,
                "latency_ms_p95": 200.0,
                "tokens_mean": 1400.0,
                "prompt_tokens_mean": 1000.0,
                "completion_tokens_mean": 400.0,
                "tool_calls_mean": 6.0,
                "message_count_mean": 4.0,
                "state_transfer_bytes_mean": 512.0,
                "estimated_tpm_mean": 900.0,
                "error_rate": 0.0,
            },
        ]
    )

    decision = build_decision_matrix(summary)

    assert len(decision) == 1
    assert decision.iloc[0]["recommendation"] == "single"
    assert decision.iloc[0]["mesh_overhead_tool_calls"] == 4.0
