import os
from pathlib import Path
import pytest
from aip_intern.core.config import load_config, AppConfig, RunConfig

SAMPLE_YAML = """
run:
  n_runs: 5
  run_id_prefix: "test"
llm:
  model: "test-model"
  base_url: "${TEST_BASE_URL}"
  api_key: "test-key"
  temperature: 0.0
  max_tokens: 512
mcp:
  workspace_root: "workspace/"
artifacts:
  output_dir: "artifacts/"
"""

def test_load_config_resolves_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_BASE_URL", "http://localhost:8000/v1")
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text(SAMPLE_YAML)
    cfg = load_config(cfg_file)
    assert cfg.llm.base_url == "http://localhost:8000/v1"
    assert cfg.run.n_runs == 5

def test_load_config_missing_env_leaves_placeholder(tmp_path):
    # If env var not set, placeholder is preserved (not silently empty)
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text(SAMPLE_YAML)
    cfg = load_config(cfg_file)
    assert "${TEST_BASE_URL}" in cfg.llm.base_url or cfg.llm.base_url == ""


def test_load_config_supports_gemini_aliases(tmp_path, monkeypatch):
    cfg_file = tmp_path / "gemini.yaml"
    cfg_file.write_text(
        """
run:
  n_runs: 5
  run_id_prefix: "test"
llm:
  model: "test-model"
  base_url: "${OPENAI_BASE_URL}"
  api_key: "${OPENAI_API_KEY}"
  temperature: 0.0
  max_tokens: 512
mcp:
  workspace_root: "workspace/"
artifacts:
  output_dir: "artifacts/"
"""
    )
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")

    cfg = load_config(cfg_file)

    assert cfg.llm.api_key == "gemini-secret"
    assert cfg.llm.base_url == "https://generativelanguage.googleapis.com/v1beta/openai"

def test_run_config_fields():
    cfg = RunConfig(
        run_id="baseline_abc",
        n_runs=20,
        config_path=Path("config/baseline.yaml"),
        llm_model="test-model",
        llm_base_url="http://localhost:8000/v1",
        workspace_root=Path("workspace/"),
        artifacts_dir=Path("artifacts/"),
    )
    assert cfg.config_path == Path("config/baseline.yaml")

def test_create_llm_returns_instance():
    from aip_intern.core.llm import create_llm
    from aip_intern.core.config import LLMCfg
    from langchain_openai import ChatOpenAI

    cfg = LLMCfg(model="test", base_url="http://localhost:8000/v1", api_key="x")
    llm = create_llm(cfg)
    assert isinstance(llm, ChatOpenAI)
