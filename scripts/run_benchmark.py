"""Benchmark-to-decision pipeline for single-agent vs mesh evaluation.

This wrapper reuses the existing runners and only adds benchmark orchestration,
telemetry collection, aggregation, and a decision-framework artifact.

Usage:
    python scripts/run_benchmark.py --config config/benchmark.json
    python scripts/run_benchmark.py --config config/benchmark.json --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import statistics
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

load_dotenv()

from aip_intern.baseline import runner as baseline_runner
from aip_intern.baseline.runner import RunConfig
from aip_intern.mesh import graph as mesh_graph
from aip_intern.mesh import runner as mesh_runner


@dataclass
class BenchmarkTelemetry:
    """Lightweight per-run telemetry collected via callback/tool wrapping."""

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    tool_calls: int = 0
    llm_calls: int = 0

    def record_llm_result(self, response: Any) -> None:
        prompt_tokens, completion_tokens = _extract_token_usage(response)
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.llm_calls += 1


def _extract_token_usage(response: Any) -> tuple[int, int]:
    """Best-effort extraction of prompt/completion tokens from an LLMResult/AIMessage."""
    usage = {}
    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        usage = llm_output.get("token_usage") or llm_output.get("usage") or {}

    if not usage:
        generations = getattr(response, "generations", None) or []
        for batch in generations:
            for generation in batch:
                message = getattr(generation, "message", None)
                metadata = getattr(message, "usage_metadata", None)
                if metadata:
                    usage = metadata
                    break
            if usage:
                break

    prompt_tokens = int(
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or usage.get("prompt_token_count")
        or 0
    )
    completion_tokens = int(
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or usage.get("candidates_token_count")
        or 0
    )
    return prompt_tokens, completion_tokens


def _wrap_tool(tool: Any, telemetry: BenchmarkTelemetry) -> Any:
    """Wrap tool methods so every invocation increments the tool-call counter."""

    def _set_tool_attr(obj: Any, attr_name: str, value: Any) -> None:
        """Set attributes on dynamic/pydantic tool objects safely."""
        try:
            setattr(obj, attr_name, value)
        except (AttributeError, ValueError, TypeError):
            object.__setattr__(obj, attr_name, value)

    def _wrap_sync(method):
        def wrapped(*args: Any, **kwargs: Any):
            telemetry.tool_calls += 1
            return method(*args, **kwargs)

        return wrapped

    async def _wrap_async(method, *args: Any, **kwargs: Any):
        telemetry.tool_calls += 1
        return await method(*args, **kwargs)

    for attr_name in ("ainvoke", "invoke", "_run"):
        original = getattr(tool, attr_name, None)
        if not callable(original):
            continue
        if asyncio.iscoroutinefunction(original):
            _set_tool_attr(
                tool,
                attr_name,
                lambda *args, _original=original, **kwargs: _wrap_async(
                    _original, *args, **kwargs
                ),
            )
        else:
            _set_tool_attr(tool, attr_name, _wrap_sync(original))
    return tool


@contextmanager
def _patched(obj: Any, attr_name: str, replacement: Any):
    original = getattr(obj, attr_name)
    setattr(obj, attr_name, replacement)
    try:
        yield
    finally:
        setattr(obj, attr_name, original)


async def _run_single_task(
    system: str,
    run_cfg: RunConfig,
    telemetry: BenchmarkTelemetry,
) -> Any:
    if system == "single":
        return await baseline_runner.run_once(run_cfg)
    if system == "mesh":
        return await mesh_runner.run_once(run_cfg)
    raise ValueError(f"Unknown system: {system}")


def _resolve_env(value: str) -> str:
    def _replacement(match) -> str:
        key = match.group(1)

        alias_candidates = [key]
        if key.startswith("OPENAI_"):
            suffix = key[len("OPENAI_") :]
            alias_candidates.extend([f"GEMINI_{suffix}", f"GOOGLE_{suffix}"])

        for candidate in alias_candidates:
            resolved = os.environ.get(candidate)
            if resolved:
                return resolved

        if key == "OPENAI_BASE_URL" and (
            os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        ):
            return "https://generativelanguage.googleapis.com/v1beta/openai"

        return match.group(0)

    return re.sub(r"\$\{(\w+)\}", _replacement, value)


def _resolve_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_dict(item) for item in value]
    if isinstance(value, str):
        return _resolve_env(value)
    return value


def _load_benchmark_config(path: Path) -> dict[str, Any]:
    return _resolve_dict(json.loads(path.read_text()))


def _decision_note(single_row: dict[str, Any], mesh_row: dict[str, Any]) -> str:
    if mesh_row["success_rate"] > single_row["success_rate"]:
        return "mesh"
    if mesh_row["success_rate"] < single_row["success_rate"]:
        return "single"
    if mesh_row["latency_ms_mean"] < single_row["latency_ms_mean"]:
        return "mesh"
    return "single"


def _build_decision_framework(summary: list[dict[str, Any]]) -> str:
    if not summary:
        return "# Mesh vs Single-Agent Decision Framework\n\nNo benchmark data found."

    task_sections: list[str] = []
    by_task = {}
    for row in summary:
        by_task.setdefault(row["task"], {})[row["system"]] = row

    for task in sorted(by_task):
        systems = by_task[task]
        single_row = systems.get("single")
        mesh_row = systems.get("mesh")
        if not single_row or not mesh_row:
            continue
        recommendation = _decision_note(single_row, mesh_row)
        task_sections.append(
            "\n".join(
                [
                    f"### {task}",
                    f"- Recommendation: **{recommendation}**",
                    f"- Success rate: single={single_row['success_rate']:.2f}, mesh={mesh_row['success_rate']:.2f}",
                    f"- Latency (ms): single={single_row['latency_ms_mean']:.1f}, mesh={mesh_row['latency_ms_mean']:.1f}",
                    f"- Tokens: single={single_row['tokens_mean']:.1f}, mesh={mesh_row['tokens_mean']:.1f}",
                    f"- TPM estimate: single={single_row['estimated_tpm_mean']:.1f}, mesh={mesh_row['estimated_tpm_mean']:.1f}",
                    f"- Mesh overhead: latency={mesh_row['latency_ms_mean'] - single_row['latency_ms_mean']:.1f} ms, tokens={mesh_row['tokens_mean'] - single_row['tokens_mean']:.1f}, tool calls={mesh_row['tool_calls_mean'] - single_row['tool_calls_mean']:.1f}",
                ]
            )
        )

    return "\n\n".join(
        [
            "# Mesh vs Single-Agent Decision Framework",
            "",
            "## Core takeaway",
            "Mesh is only worth it when task complexity and specialization gains outweigh coordination overhead, failure propagation, and higher token pressure.",
            "## Decision matrix",
            "",
            "| Scenario | Use Single Agent | Use Mesh |",
            "|---|---:|---:|",
            "| Simple tasks | ✅ | ❌ |",
            "| Low latency required | ✅ | ❌ |",
            "| High TPM constraints | ❌ | ⚠️ |",
            "| Complex multi-step tasks | ❌ | ✅ |",
            "| Tool-heavy workflows | ❌ | ✅ |",
            "| Strict reliability required | ✅ | ❌ |",
            "",
            "## GCC constraints",
            "- Added latency hurts mesh more than a single agent.",
            "- TPM caps become throughput bottlenecks because mesh can increase total token volume.",
            "- Egress/tool restrictions amplify mesh overhead because more tool calls become coordination points.",
            "- Reliability matters more than peak capability in constrained environments.",
            "",
            "## Task-level recommendations",
            *task_sections,
        ]
    )


async def run_benchmark(config_path: Path, dry_run: bool = False) -> Path:
    config = _load_benchmark_config(config_path)
    benchmark_cfg = config["benchmark"]
    llm_cfg = config["llm"]
    mcp_cfg = config.get("mcp", {})
    artifacts_cfg = config.get("artifacts", {})
    tasks = config["tasks"]

    n_runs = 1 if dry_run else int(benchmark_cfg["n_runs_per_task"])
    systems = benchmark_cfg.get("systems", ["single", "mesh"])
    output_dir = Path(benchmark_cfg.get("output_dir", "artifacts/benchmarks"))
    output_dir.mkdir(parents=True, exist_ok=True)

    workspace_root = Path(mcp_cfg.get("workspace_root", "workspace/"))
    artifacts_dir = Path(artifacts_cfg.get("output_dir", "artifacts/"))

    records: list[dict[str, Any]] = []

    for task_name, task_description in tasks.items():
        for system in systems:
            for run_index in range(1, n_runs + 1):
                telemetry = BenchmarkTelemetry()
                run_cfg = RunConfig(
                    run_id_prefix=f"{system}_{task_name}",
                    n_runs=1,
                    config_path=config_path,
                    llm_model=llm_cfg["model"],
                    llm_base_url=llm_cfg["base_url"],
                    llm_api_key=llm_cfg["api_key"],
                    llm_temperature=float(llm_cfg.get("temperature", 0.0)),
                    llm_max_tokens=int(llm_cfg.get("max_tokens", 4096)),
                    llm_request_timeout=int(llm_cfg.get("request_timeout", 120)),
                    workspace_root=workspace_root,
                    artifacts_dir=artifacts_dir,
                    task_description=task_description,
                )

                print(f"[{system}][{task_name}] run {run_index}/{n_runs} ...", end=" ", flush=True)
                result = await _run_single_task(system, run_cfg, telemetry)
                latency_ms = float(result.metrics.get("total_latency_s", 0.0)) * 1000.0
                prompt_tokens = int(result.metrics.get("total_prompt_tokens", 0) or 0)
                completion_tokens = int(result.metrics.get("total_completion_tokens", 0) or 0)
                total_tokens = prompt_tokens + completion_tokens
                estimated_tpm = (
                    (total_tokens / result.metrics.get("total_latency_s", 0.0)) * 60.0
                    if result.metrics.get("total_latency_s", 0.0)
                    else 0.0
                )
                record = {
                    "run_id": result.run_id,
                    "system": system,
                    "task": task_name,
                    "task_description": task_description,
                    "run_index": run_index,
                    "latency_ms": round(latency_ms, 3),
                    "tokens_used": total_tokens,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "estimated_tpm": round(estimated_tpm, 3),
                    "success": bool(result.success),
                    "llm_judge_score": None,
                    "steps": result.metrics.get("step_trace", []),
                    "errors": result.error,
                    "tool_calls": telemetry.tool_calls,
                    "inter_agent_message_count": result.metrics.get("message_count"),
                    "state_transfer_size": result.metrics.get("state_size_bytes"),
                }
                records.append(record)
                print("OK" if result.success else f"ERR: {result.error}")

    records_path = output_dir / "benchmark_records.jsonl"
    csv_path = output_dir / "benchmark_records.csv"
    summary_path = output_dir / "benchmark_summary.csv"
    decision_path = output_dir / "decision_framework.md"

    records_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    csv_fields = [
        "run_id",
        "system",
        "task",
        "task_description",
        "run_index",
        "latency_ms",
        "tokens_used",
        "prompt_tokens",
        "completion_tokens",
        "estimated_tpm",
        "success",
        "llm_judge_score",
        "steps",
        "errors",
        "tool_calls",
        "inter_agent_message_count",
        "state_transfer_size",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in records:
            csv_row = dict(row)
            csv_row["steps"] = json.dumps(csv_row["steps"])
            writer.writerow(csv_row)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault((row["system"], row["task"]), []).append(row)

    summary: list[dict[str, Any]] = []
    for (system, task), rows in sorted(grouped.items()):
        latencies = [float(row["latency_ms"]) for row in rows]
        tokens = [float(row["tokens_used"]) for row in rows]
        prompt_tokens = [float(row["prompt_tokens"]) for row in rows]
        completion_tokens = [float(row["completion_tokens"]) for row in rows]
        tpm = [float(row["estimated_tpm"]) for row in rows]
        tool_calls = [float(row["tool_calls"]) for row in rows]
        message_counts = [float(row.get("inter_agent_message_count") or 0) for row in rows]
        state_sizes = [float(row.get("state_transfer_size") or 0) for row in rows]
        success_rate = sum(1 for row in rows if row["success"]) / len(rows)

        summary.append(
            {
                "system": system,
                "task": task,
                "runs": len(rows),
                "success_rate": round(success_rate, 4),
                "latency_ms_mean": round(statistics.mean(latencies), 3),
                "latency_ms_p95": round(sorted(latencies)[max(0, int(round(len(latencies) * 0.95)) - 1)], 3),
                "tokens_mean": round(statistics.mean(tokens), 3),
                "prompt_tokens_mean": round(statistics.mean(prompt_tokens), 3),
                "completion_tokens_mean": round(statistics.mean(completion_tokens), 3),
                "estimated_tpm_mean": round(statistics.mean(tpm), 3),
                "tool_calls_mean": round(statistics.mean(tool_calls), 3),
                "message_count_mean": round(statistics.mean(message_counts), 3),
                "state_transfer_size_mean": round(statistics.mean(state_sizes), 3),
                "error_rate": round(1.0 - success_rate, 4),
            }
        )

    with summary_path.open("w", newline="") as handle:
        fieldnames = list(summary[0].keys()) if summary else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if summary:
            writer.writeheader()
            writer.writerows(summary)

    decision_path.write_text(_build_decision_framework(summary))

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-agent vs mesh benchmark sweep")
    parser.add_argument("--config", default="config/benchmark.json")
    parser.add_argument("--dry-run", action="store_true", help="Run once per task/system")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    out_dir = asyncio.run(run_benchmark(config_path, dry_run=args.dry_run))
    print(f"\nBenchmark complete. Artifacts written to: {out_dir}/")


if __name__ == "__main__":
    main()
