from __future__ import annotations

import os
from pathlib import Path

import pytest

from aip_intern.baseline.runner import RunConfig, run_once


@pytest.mark.integration
@pytest.mark.asyncio
async def test_baseline_single_run_produces_outputs(live_llm, tmp_path):
    """Full baseline run: all 3 output files produced, no error."""
    base_url = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("GEMINI_BASE_URL")
        or os.environ.get("GOOGLE_BASE_URL")
    )
    if not base_url and (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
    cfg = RunConfig(
        run_id_prefix="integration_test",
        n_runs=1,
        config_path=Path("config/baseline.yaml"),
        llm_model=os.environ.get(
            "OPENAI_MODEL", os.environ.get("GEMINI_MODEL", "Qwen/Qwen3-32B-Instruct")
        ),
        llm_base_url=base_url or os.environ["OPENAI_BASE_URL"],
        llm_api_key=os.environ.get(
            "OPENAI_API_KEY",
            os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", "not-needed")),
        ),
        workspace_root=Path("workspace/"),
        artifacts_dir=tmp_path,
    )
    result = await run_once(cfg)
    assert result.success, f"Run failed: {result.error}"
    assert (result.outputs_path / "triage.csv").exists()
    assert (result.outputs_path / "brief.md").exists()
    assert (result.outputs_path / "response_templates.md").exists()
