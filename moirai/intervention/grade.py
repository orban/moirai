"""Deterministic grader adapter with a blinded, outcome-independent retry policy."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from moirai.intervention.checkpoint import Workspace
from moirai.intervention.schema import GradeResult


class Grader(Protocol):
    version: str

    def grade(self, workspace: Workspace, task_id: str) -> GradeResult: ...


def grade_with_retry(grader: Grader, workspace: Workspace, task_id: str, max_attempts: int = 3) -> GradeResult:
    """Rerun only the evaluator, only on evaluator error, at most max_attempts times.

    The agent is never rerun here. The policy does not look at the arm or at
    any provisional outcome.
    """
    last: GradeResult | None = None
    for attempt in range(1, max_attempts + 1):
        r = grader.grade(workspace, task_id)
        last = GradeResult(r.passed, r.status, r.detail, r.evaluator_version or grader.version, attempt)
        if r.status == "ok":
            return last
    assert last is not None
    return last


class FixtureGrader:
    """Reads ``outcome.json`` that a synthetic runtime leaves in the workspace."""
    version = "fixture/1"

    def grade(self, workspace: Workspace, task_id: str) -> GradeResult:
        p = Path(workspace.path) / "outcome.json"
        if not p.exists():
            return GradeResult(None, "grader_error", "outcome.json missing", self.version)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            return GradeResult(None, "grader_error", f"unreadable outcome: {e}", self.version)
        if data.get("task_id") != task_id:
            return GradeResult(None, "grader_error", "task mismatch", self.version)
        if data.get("grader_flaky_once") and not p.with_suffix(".retry").exists():
            p.with_suffix(".retry").write_text("1", encoding="utf-8")
            return GradeResult(None, "grader_error", "transient evaluator failure (fixture)", self.version)
        return GradeResult(bool(data["passed"]), "ok", "", self.version)
