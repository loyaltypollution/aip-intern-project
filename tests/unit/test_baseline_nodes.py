from aip_intern.baseline.state import BaselineState

def test_baseline_state_construction():
    state: BaselineState = {
        "run_id": "baseline_test",
        "task_description": "Triage feedback",
        "feedback_files": [],
        "policy_content": "",
        "triage_result": None,
        "brief_result": None,
        "response_result": None,
        "error": None,
        "step_trace": [],
    }
    assert state["run_id"] == "baseline_test"
    assert state["step_trace"] == []
