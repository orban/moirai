"""Signature compression and phase classification.

Transforms raw step sequences into human-readable compressed representations.
"""
from __future__ import annotations

from moirai.schema import Run, Step


# --- Phase classification ---

PHASE_MAP: dict[str, str] = {
    # Step names → phases
    "read": "explore",
    "search": "explore",
    "glob": "explore",
    "grep": "explore",
    "edit": "modify",
    "write": "modify",
    "test": "verify",
    "test_result": "verify",
    "bash": "execute",
    "reason": "think",
    "subagent": "think",
    "plan": "system",
    "result": "system",
    "error_observation": "system",
    "unknown": "other",
    "action": "act",
}

# Step types → fallback phases (when name isn't in PHASE_MAP)
TYPE_PHASE_MAP: dict[str, str] = {
    "llm": "think",
    "tool": "act",
    "system": "system",
    "judge": "verify",
    "error": "error",
    "memory": "system",
    "compaction": "system",
    "handoff": "system",
    "other": "other",
}


def step_phase(step: Step) -> str:
    """Classify a step into a high-level phase."""
    if step.name in PHASE_MAP:
        return PHASE_MAP[step.name]
    return TYPE_PHASE_MAP.get(step.type, "other")


NOISE_STEPS = {"error_observation", "action"}


def step_display_name(step: Step) -> str | None:
    """Get a human-readable display name for a step.

    Returns None for noisy steps that should be filtered.
    Annotates test results with pass/fail.
    """
    if step.name in NOISE_STEPS:
        return None
    if step.name == "test_result":
        if step.status != "ok":
            return "test(fail)"
        return "test(pass)"
    return step.name


# --- Run-length encoding ---

def _rle(items: list[str]) -> list[tuple[str, int]]:
    """Run-length encode a list of strings."""
    if not items:
        return []
    result: list[tuple[str, int]] = []
    current = items[0]
    count = 1
    for item in items[1:]:
        if item == current:
            count += 1
        else:
            result.append((current, count))
            current = item
            count = 1
    result.append((current, count))
    return result


def _format_rle(runs: list[tuple[str, int]]) -> str:
    """Format RLE tuples into a readable string."""
    parts = []
    for name, count in runs:
        if count == 1:
            parts.append(name)
        else:
            parts.append(f"{name}×{count}")
    return " → ".join(parts)


# --- Compressed signatures ---

def compress_steps(steps: list[Step]) -> str:
    """Compress a step sequence into a human-readable signature.

    Filters noise (error_observation, generic action), annotates test results,
    and run-length encodes consecutive identical steps.
    """
    names: list[str] = []
    for step in steps:
        display = step_display_name(step)
        if display is not None:
            names.append(display)

    # If all steps were filtered, fall back to step types
    if not names:
        names = [step.type for step in steps]

    if not names:
        return "(empty)"

    return _format_rle(_rle(names))


def compress_run(run: Run) -> str:
    """Compress a run's trajectory into a readable signature."""
    return compress_steps(run.steps)


# --- Phase sequences ---

def phase_sequence(run: Run) -> list[str]:
    """Get the phase sequence for a run, filtering noise."""
    phases: list[str] = []
    for step in run.steps:
        if step.name == "error_observation":
            continue  # noise
        phase = step_phase(step)
        phases.append(phase)
    return phases


def compress_phases(run: Run) -> str:
    """Compress a run into a phase-level signature.

    e.g. "explore → modify → verify(fail) → explore → modify → verify(pass)"
    """
    phases: list[str] = []
    for step in run.steps:
        if step.name == "error_observation":
            continue
        if step.name == "action":
            continue

        phase = step_phase(step)

        # Annotate verify phases with pass/fail
        if phase == "verify":
            if step.status != "ok" or step.name == "test_result" and step.status == "error":
                phase = "verify(fail)"
            else:
                phase = "verify(pass)"

        phases.append(phase)

    if not phases:
        return "(empty)"

    return _format_rle(_rle(phases))


def phase_summary(run: Run) -> dict[str, int]:
    """Count steps per phase for a run (excluding noise)."""
    counts: dict[str, int] = {}
    for step in run.steps:
        if step.name in NOISE_STEPS:
            continue
        phase = step_phase(step)
        counts[phase] = counts.get(phase, 0) + 1
    return counts


def phase_summary_str(run: Run) -> str:
    """Format phase summary as a readable string like '38% explore, 10% modify, 19% verify'."""
    counts = phase_summary(run)
    total = sum(counts.values())
    if total == 0:
        return "no steps"

    parts = []
    for phase, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        parts.append(f"{pct:.0f}% {phase}")
    return ", ".join(parts)


# --- Cluster sub-pattern detection ---

def classify_run_subpattern(run: Run, avg_steps: float) -> str:
    """Classify a run into a behavioral sub-pattern based on length and outcome.

    Returns one of: "fast_solve", "retry_loop", "stuck", "normal"
    """
    n_steps = len(run.steps)
    success = run.result.success

    if success and n_steps < avg_steps * 0.5:
        return "fast_solve"
    if not success and n_steps > avg_steps * 1.5:
        return "stuck"
    if n_steps > avg_steps * 1.2:
        return "retry_loop"
    return "normal"


def cluster_subpatterns(runs: list[Run]) -> dict[str, list[Run]]:
    """Group runs in a cluster by behavioral sub-pattern."""
    if not runs:
        return {}

    avg_steps = sum(len(r.steps) for r in runs) / len(runs)
    groups: dict[str, list[Run]] = {}

    for run in runs:
        pattern = classify_run_subpattern(run, avg_steps)
        if pattern not in groups:
            groups[pattern] = []
        groups[pattern].append(run)

    return groups
