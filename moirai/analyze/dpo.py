"""Extract DPO preference pairs from aligned trajectory divergence points.

Given runs of the same task with mixed outcomes, aligns trajectories,
finds divergence points, and extracts (prompt, preferred, dispreferred)
triples where the agent's choice at the divergence predicts success.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from moirai.analyze.align import align_runs, _get_sequence
from moirai.analyze.divergence import find_divergence_points
from moirai.schema import Alignment, DivergencePoint, GAP, Run


@dataclass
class PreferencePair:
    """A single DPO preference pair extracted from a divergence point."""
    task_id: str
    divergence_column: int              # position in alignment
    divergence_position_norm: float     # 0-1 normalized position

    # The shared context leading up to the divergence
    context_steps: list[dict[str, Any]]  # steps before the divergence (from preferred run)
    context_sequence: list[str]          # enriched step names before divergence

    # The divergence itself
    preferred_step: dict[str, Any]       # full step dict from a passing run
    dispreferred_step: dict[str, Any]    # full step dict from a failing run
    preferred_label: str                 # enriched step name
    dispreferred_label: str              # enriched step name

    # Metadata
    preferred_run_id: str
    dispreferred_run_id: str
    p_value: float | None
    q_value: float | None
    preferred_success_rate: float        # pass rate of runs taking the preferred branch
    dispreferred_success_rate: float     # pass rate of runs taking the dispreferred branch
    n_preferred_runs: int
    n_dispreferred_runs: int


def _find_step_at_alignment_col(
    run: Run,
    alignment: Alignment,
    run_idx: int,
    col: int,
) -> dict[str, Any] | None:
    """Map an alignment column back to the original step in a run.

    Walks the alignment row, counting non-gap positions to find the
    original step index, then returns the full step dict.
    """
    if run_idx >= len(alignment.matrix):
        return None
    row = alignment.matrix[run_idx]
    if col >= len(row) or row[col] == GAP:
        return None

    # Count non-gap positions up to and including this column
    original_idx = sum(1 for c in range(col + 1) if c < len(row) and row[c] != GAP) - 1

    if 0 <= original_idx < len(run.steps):
        step = run.steps[original_idx]
        return {
            "idx": step.idx,
            "type": step.type,
            "name": step.name,
            "status": step.status,
            "input": step.input,
            "output": step.output,
            "attrs": step.attrs,
        }
    return None


def _get_context_steps(
    run: Run,
    alignment: Alignment,
    run_idx: int,
    col: int,
    max_context: int = 5,
) -> list[dict[str, Any]]:
    """Get the steps leading up to a divergence point from a specific run."""
    steps = []
    start_col = max(0, col - max_context)
    for c in range(start_col, col):
        step = _find_step_at_alignment_col(run, alignment, run_idx, c)
        if step is not None:
            steps.append(step)
    return steps


def extract_preference_pairs(
    runs: list[Run],
    task_id: str,
    level: str = "name",
    q_threshold: float = 0.10,
    min_branch_size: int = 2,
    max_context_steps: int = 5,
) -> list[PreferencePair]:
    """Extract DPO preference pairs from runs of the same task.

    For each significant divergence point:
    1. Identifies the best (highest pass rate) and worst (lowest pass rate) branches
    2. Picks a representative passing run from the best branch and a failing run from worst
    3. Extracts context, preferred step, and dispreferred step with full content

    Returns preference pairs sorted by statistical significance.
    """
    if len(runs) < 3:
        return []

    pass_runs = [r for r in runs if r.result.success is True]
    fail_runs = [r for r in runs if r.result.success is False]
    if not pass_runs or not fail_runs:
        return []

    # Align all runs
    alignment = align_runs(runs, level=level)
    n_cols = len(alignment.matrix[0]) if alignment.matrix else 0

    # Find divergence points
    divergence_points, _ = find_divergence_points(
        alignment, runs, min_branch_size=min_branch_size, q_threshold=q_threshold,
    )

    if not divergence_points:
        return []

    # Build run_id -> index maps
    run_id_to_idx = {rid: i for i, rid in enumerate(alignment.run_ids)}
    run_id_to_run = {r.run_id: r for r in runs}

    pairs = []

    for dp in divergence_points:
        # Find the best and worst branches by success rate
        best_branch = None
        worst_branch = None
        best_rate = -1.0
        worst_rate = 2.0

        for value, rate in dp.success_by_value.items():
            if rate is None:
                continue
            if rate > best_rate:
                best_rate = rate
                best_branch = value
            if rate < worst_rate:
                worst_rate = rate
                worst_branch = value

        if best_branch is None or worst_branch is None:
            continue
        if best_branch == worst_branch:
            continue
        # Need a meaningful gap
        if best_rate - worst_rate < 0.15:
            continue

        # Find representative runs from each branch
        preferred_run_id = None
        dispreferred_run_id = None

        for run_idx, run_id in enumerate(alignment.run_ids):
            if run_idx >= len(alignment.matrix):
                continue
            row = alignment.matrix[run_idx]
            if dp.column >= len(row):
                continue
            val = row[dp.column]

            run = run_id_to_run[run_id]
            if val == best_branch and run.result.success is True and preferred_run_id is None:
                preferred_run_id = run_id
            if val == worst_branch and run.result.success is False and dispreferred_run_id is None:
                dispreferred_run_id = run_id

        if preferred_run_id is None or dispreferred_run_id is None:
            continue

        pref_idx = run_id_to_idx[preferred_run_id]
        dis_idx = run_id_to_idx[dispreferred_run_id]
        pref_run = run_id_to_run[preferred_run_id]
        dis_run = run_id_to_run[dispreferred_run_id]

        # Extract the actual steps at the divergence
        pref_step = _find_step_at_alignment_col(pref_run, alignment, pref_idx, dp.column)
        dis_step = _find_step_at_alignment_col(dis_run, alignment, dis_idx, dp.column)

        if pref_step is None or dis_step is None:
            continue

        # Context from the preferred run (shared prefix)
        context = _get_context_steps(pref_run, alignment, pref_idx, dp.column, max_context_steps)
        context_seq = [alignment.matrix[pref_idx][c]
                       for c in range(max(0, dp.column - max_context_steps), dp.column)
                       if c < len(alignment.matrix[pref_idx]) and alignment.matrix[pref_idx][c] != GAP]

        pairs.append(PreferencePair(
            task_id=task_id,
            divergence_column=dp.column,
            divergence_position_norm=dp.column / n_cols if n_cols > 0 else 0.0,
            context_steps=context,
            context_sequence=context_seq,
            preferred_step=pref_step,
            dispreferred_step=dis_step,
            preferred_label=best_branch,
            dispreferred_label=worst_branch,
            preferred_run_id=preferred_run_id,
            dispreferred_run_id=dispreferred_run_id,
            p_value=dp.p_value,
            q_value=dp.q_value,
            preferred_success_rate=best_rate,
            dispreferred_success_rate=worst_rate,
            n_preferred_runs=dp.value_counts.get(best_branch, 0),
            n_dispreferred_runs=dp.value_counts.get(worst_branch, 0),
        ))

    return pairs


def extract_pairwise_preferences(
    runs: list[Run],
    task_id: str,
    level: str = "name",
    max_context_steps: int = 5,
) -> list[PreferencePair]:
    """Extract DPO pairs by pairwise alignment of pass/fail run pairs.

    Unlike extract_preference_pairs which requires population-level
    statistical significance, this function aligns each (pass, fail) pair
    individually and finds where they first diverge. This works even with
    just 2 runs — one pass, one fail.

    Returns one pair per (pass_run, fail_run) combination with a clear
    divergence point.
    """
    pass_runs = [r for r in runs if r.result.success is True]
    fail_runs = [r for r in runs if r.result.success is False]
    if not pass_runs or not fail_runs:
        return []

    pairs = []

    for pref_run in pass_runs:
        for dis_run in fail_runs:
            pair = _extract_pairwise_divergence(
                pref_run, dis_run, task_id, level, max_context_steps,
            )
            if pair is not None:
                pairs.append(pair)

    return pairs


def _extract_pairwise_divergence(
    pass_run: Run,
    fail_run: Run,
    task_id: str,
    level: str,
    max_context_steps: int,
) -> PreferencePair | None:
    """Find the first divergence between a pass/fail pair and extract a preference."""
    from moirai.analyze.align import _nw_align, _get_sequence

    seq_pass = _get_sequence(pass_run, level)
    seq_fail = _get_sequence(fail_run, level)

    if not seq_pass or not seq_fail:
        return None

    aligned_pass, aligned_fail = _nw_align(seq_pass, seq_fail)

    # Find first divergence (non-gap mismatch)
    first_div = None
    for col in range(len(aligned_pass)):
        if aligned_pass[col] != aligned_fail[col]:
            # Skip if either is a gap — gaps are insertions, not choices
            if aligned_pass[col] == GAP or aligned_fail[col] == GAP:
                continue
            first_div = col
            break

    if first_div is None:
        return None

    # Map alignment column back to original step indices
    pass_orig_idx = sum(1 for c in range(first_div + 1) if aligned_pass[c] != GAP) - 1
    fail_orig_idx = sum(1 for c in range(first_div + 1) if aligned_fail[c] != GAP) - 1

    if pass_orig_idx < 0 or pass_orig_idx >= len(pass_run.steps):
        return None
    if fail_orig_idx < 0 or fail_orig_idx >= len(fail_run.steps):
        return None

    pref_step_obj = pass_run.steps[pass_orig_idx]
    dis_step_obj = fail_run.steps[fail_orig_idx]

    def _step_to_dict(s):
        return {
            "idx": s.idx, "type": s.type, "name": s.name,
            "status": s.status, "input": s.input,
            "output": s.output, "attrs": s.attrs,
        }

    # Context: steps before divergence from the pass run
    context_start = max(0, pass_orig_idx - max_context_steps)
    context_steps = [_step_to_dict(pass_run.steps[i])
                     for i in range(context_start, pass_orig_idx)]
    context_seq = aligned_pass[max(0, first_div - max_context_steps):first_div]
    context_seq = [s for s in context_seq if s != GAP]

    n_cols = len(aligned_pass)
    return PreferencePair(
        task_id=task_id,
        divergence_column=first_div,
        divergence_position_norm=first_div / n_cols if n_cols > 0 else 0.0,
        context_steps=context_steps,
        context_sequence=context_seq,
        preferred_step=_step_to_dict(pref_step_obj),
        dispreferred_step=_step_to_dict(dis_step_obj),
        preferred_label=aligned_pass[first_div],
        dispreferred_label=aligned_fail[first_div],
        preferred_run_id=pass_run.run_id,
        dispreferred_run_id=fail_run.run_id,
        p_value=None,
        q_value=None,
        preferred_success_rate=1.0,
        dispreferred_success_rate=0.0,
        n_preferred_runs=1,
        n_dispreferred_runs=1,
    )


def format_pair_as_dpo(pair: PreferencePair) -> dict[str, Any]:
    """Format a preference pair as a DPO training example.

    Returns a dict with:
      - prompt: the task context + trajectory prefix
      - chosen: the preferred step (reasoning + action)
      - rejected: the dispreferred step (reasoning + action)
      - metadata: statistical evidence and provenance
    """
    # Build the prompt from context
    context_parts = []
    for step in pair.context_steps:
        out = step.get("output", {})
        reasoning = out.get("reasoning", "")
        action = out.get("action", "")
        result = out.get("result", "")
        observation = out.get("observation", "")

        part = f"[Step {step['idx']}: {step['type']}:{step['name']}]"
        if reasoning:
            part += f"\nThinking: {reasoning}"
        if action:
            part += f"\nAction: {action}"
        if result:
            part += f"\nResult: {str(result)[:500]}"
        elif observation:
            part += f"\nObservation: {str(observation)[:500]}"
        context_parts.append(part)

    prompt = f"Task: {pair.task_id}\n\nTrajectory so far:\n" + "\n\n".join(context_parts)
    prompt += "\n\nWhat should the agent do next?"

    def _format_step(step: dict) -> str:
        out = step.get("output", {})
        inp = step.get("input", {})
        reasoning = out.get("reasoning", "") or inp.get("reasoning", "")
        action = out.get("action", "")
        result = out.get("result", "")
        observation = out.get("observation", "")
        text = inp.get("text", "")
        parts = []
        if reasoning:
            parts.append(f"Thinking: {reasoning}")
        if action:
            parts.append(f"Action: {action}")
        if result and isinstance(result, str):
            parts.append(f"Result: {result[:1000]}")
        elif observation:
            parts.append(f"Observation: {str(observation)[:1000]}")
        if not parts and text:
            parts.append(f"Content: {str(text)[:1000]}")
        if not parts:
            parts.append(f"{step['type']}:{step['name']}")
        return "\n".join(parts)

    return {
        "prompt": prompt,
        "chosen": _format_step(pair.preferred_step),
        "rejected": _format_step(pair.dispreferred_step),
        "metadata": {
            "task_id": pair.task_id,
            "divergence_position": pair.divergence_position_norm,
            "preferred_label": pair.preferred_label,
            "dispreferred_label": pair.dispreferred_label,
            "preferred_success_rate": pair.preferred_success_rate,
            "dispreferred_success_rate": pair.dispreferred_success_rate,
            "p_value": pair.p_value,
            "q_value": pair.q_value,
            "n_preferred": pair.n_preferred_runs,
            "n_dispreferred": pair.n_dispreferred_runs,
            "preferred_run_id": pair.preferred_run_id,
            "dispreferred_run_id": pair.dispreferred_run_id,
        },
    }
