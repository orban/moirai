"""Generate narrative findings from per-task divergence analysis.

Takes alignment, divergence points, and runs for a single task
and produces structured findings that explain what happened.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from moirai.compress import step_enriched_name, NOISE_STEPS
from moirai.schema import Alignment, DivergencePoint, GAP, Run


@dataclass
class StepDetail:
    """One step in a trajectory with full context."""
    position: int
    enriched_name: str
    detail: str  # file name, command, or pattern
    is_fork: bool = False
    reasoning: str | None = None


@dataclass
class BranchExample:
    """One branch at a divergence point with a representative trajectory."""
    value: str                    # the step choice at the fork
    run_count: int
    success_rate: float | None
    representative_run_id: str
    steps: list[StepDetail]       # full trajectory with fork marked
    fork_position: int            # which step index is the fork
    reasoning: str | None = None  # agent reasoning at the fork step


@dataclass
class Finding:
    """A narrative finding for one divergence point within a task."""
    task_id: str
    total_runs: int
    pass_count: int
    fail_count: int
    fork_column: int
    p_value: float | None
    summary: str                  # one-line plain-language summary
    branches: list[BranchExample]
    recommendation: str           # what to do about it


def narrate_task(
    task_id: str,
    runs: list[Run],
    alignment: Alignment,
    points: list[DivergencePoint],
) -> list[Finding]:
    """Generate narrative findings for a single task's divergence points."""
    if not points:
        return []

    n_pass = sum(1 for r in runs if r.result.success)
    n_fail = len(runs) - n_pass
    run_map = {r.run_id: r for r in runs}

    findings: list[Finding] = []

    for point in points[:3]:  # top 3 divergence points per task
        branches = _build_branches(point, alignment, runs)
        if len(branches) < 2:
            continue

        summary = _generate_summary(task_id, branches, point, n_pass, n_fail)
        recommendation = _generate_recommendation(branches, point)

        findings.append(Finding(
            task_id=task_id,
            total_runs=len(runs),
            pass_count=n_pass,
            fail_count=n_fail,
            fork_column=point.column,
            p_value=point.p_value,
            summary=summary,
            branches=branches,
            recommendation=recommendation,
        ))

    return findings


def _build_branches(
    point: DivergencePoint,
    alignment: Alignment,
    runs: list[Run],
) -> list[BranchExample]:
    """Build branch examples with full trajectory context."""
    run_map = {r.run_id: r for r in runs}
    branches: list[BranchExample] = []

    sorted_values = sorted(
        point.value_counts.items(),
        key=lambda x: -(point.success_by_value.get(x[0]) or 0)
    )

    for value, count in sorted_values:
        rate = point.success_by_value.get(value)

        # Find representative run (prefer pass for high-success branches, fail for low)
        rep_run = None
        rep_idx = None
        for run_idx, rid in enumerate(alignment.run_ids):
            if run_idx < len(alignment.matrix) and point.column < len(alignment.matrix[run_idx]):
                if alignment.matrix[run_idx][point.column] == value:
                    run = run_map.get(rid)
                    if run:
                        rep_run = run
                        rep_idx = run_idx
                        break

        if not rep_run or rep_idx is None:
            continue

        # Build step details with fork marked
        steps: list[StepDetail] = []
        # Count non-gap positions before the fork to find the actual step index
        fork_step_idx = sum(
            1 for c in range(point.column)
            if c < len(alignment.matrix[rep_idx]) and alignment.matrix[rep_idx][c] != GAP
        )

        filtered_idx = 0
        for step in rep_run.steps:
            enriched = step_enriched_name(step)
            if enriched is None:
                continue

            detail = _step_detail(step)
            is_fork = (filtered_idx == fork_step_idx)

            reasoning = None
            if is_fork:
                reasoning = step.output.get("reasoning") if step.output else None

            steps.append(StepDetail(
                position=filtered_idx,
                enriched_name=enriched,
                detail=detail,
                is_fork=is_fork,
                reasoning=reasoning,
            ))
            filtered_idx += 1

        # Extract fork reasoning for the branch-level field
        fork_reasoning = None
        for s in steps:
            if s.is_fork and s.reasoning:
                fork_reasoning = s.reasoning
                break

        branches.append(BranchExample(
            value=value,
            run_count=count,
            success_rate=rate,
            representative_run_id=rep_run.run_id,
            steps=steps,
            fork_position=fork_step_idx,
            reasoning=fork_reasoning,
        ))

    return branches


def _step_detail(step) -> str:
    """Extract human-readable detail from step attrs."""
    if not step.attrs:
        return ""
    fp = step.attrs.get("file_path", "")
    if fp:
        return os.path.basename(fp)
    cmd = step.attrs.get("command", "")
    if cmd:
        return cmd[:60]
    pat = step.attrs.get("pattern", "")
    if pat:
        return pat[:60]
    return ""


def _generate_summary(
    task_id: str,
    branches: list[BranchExample],
    point: DivergencePoint,
    n_pass: int,
    n_fail: int,
) -> str:
    """Generate a one-line summary of the finding."""
    best = max(branches, key=lambda b: b.success_rate or 0)
    worst = min(branches, key=lambda b: b.success_rate if b.success_rate is not None else 1)

    best_detail = ""
    worst_detail = ""
    for s in best.steps:
        if s.is_fork:
            best_detail = f" ({s.detail})" if s.detail else ""
            break
    for s in worst.steps:
        if s.is_fork:
            worst_detail = f" ({s.detail})" if s.detail else ""
            break

    best_rate = f"{best.success_rate:.0%}" if best.success_rate is not None else "?"
    worst_rate = f"{worst.success_rate:.0%}" if worst.success_rate is not None else "?"

    return (
        f"At step {best.fork_position}, runs that chose "
        f"`{best.value}`{best_detail} succeeded {best_rate} of the time, "
        f"while runs that chose `{worst.value}`{worst_detail} succeeded {worst_rate}."
    )


def _generate_recommendation(branches: list[BranchExample], point: DivergencePoint) -> str:
    """Generate a recommendation based on the branches."""
    best = max(branches, key=lambda b: b.success_rate or 0)
    worst = min(branches, key=lambda b: b.success_rate if b.success_rate is not None else 1)

    # What did the best branch do BEFORE the fork vs the worst?
    best_before = [s for s in best.steps if s.position < best.fork_position]
    worst_before = [s for s in worst.steps if s.position < worst.fork_position]

    # Find the first step where they differ
    diff_step = None
    for i in range(min(len(best_before), len(worst_before))):
        if best_before[i].enriched_name != worst_before[i].enriched_name:
            diff_step = i
            break

    if diff_step is not None:
        return (
            f"The successful branch diverged earlier at step {diff_step}: "
            f"`{best_before[diff_step].enriched_name}` "
            f"({best_before[diff_step].detail or 'no detail'}) "
            f"vs `{worst_before[diff_step].enriched_name}` "
            f"({worst_before[diff_step].detail or 'no detail'})."
        )

    best_detail = ""
    worst_detail = ""
    for s in best.steps:
        if s.is_fork:
            best_detail = s.detail
            break
    for s in worst.steps:
        if s.is_fork:
            worst_detail = s.detail
            break

    return (
        f"Prefer `{best.value}`"
        + (f" ({best_detail})" if best_detail else "")
        + f" over `{worst.value}`"
        + (f" ({worst_detail})" if worst_detail else "")
        + f" at this decision point."
    )
