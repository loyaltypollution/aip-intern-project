"""Recovery scoring — Phase 3 implementation target.

Stubs define the scoring interface. Phase 3 interns fill in the bodies.

Scoring rubric (from spec):
- recovery_time_s: time from exception raise to graph reaching END (or giving up)
- output_quality: 0.0–1.0, based on whether expected output files were produced
- recovery_mode: "automatic" (graph handled it) | "manual" (human intervention needed)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class RecoveryScore:
    failure_type: str                        # e.g. "AgentTimeoutError"
    recovery_time_s: float                   # seconds from fault injection to END
    output_quality: float                    # 0.0 = no outputs, 1.0 = all 3 files
    recovery_mode: Literal["automatic", "manual", "unrecoverable"]
    notes: str = ""


def score_recovery(
    failure_type: str,
    t_fault: float,
    t_end: float,
    outputs_path: Path,
) -> RecoveryScore:
    """Score a single failure injection run.

    Args:
        failure_type: Name of the injected fault (e.g. "AgentTimeoutError").
        t_fault: Unix timestamp when the fault was injected.
        t_end: Unix timestamp when the graph reached END (or gave up).
        outputs_path: Path to artifacts/{run_id}/outputs/.

    Returns:
        RecoveryScore with recovery_time_s, output_quality, recovery_mode.

    Phase 3 intern: implement output_quality by checking which of
    triage.csv, brief.md, response_templates.md were produced.
    Determine recovery_mode by inspecting the graph's final state error field.
    """
    ...


def score_output_quality(outputs_path: Path) -> float:
    """Score output completeness: fraction of expected files produced.

    Expected files: triage.csv, brief.md, response_templates.md
    Returns: 0.0, 0.33, 0.67, or 1.0

    Phase 3 intern: also consider file size > 0 and basic format validation.
    """
    ...
