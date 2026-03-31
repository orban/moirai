from __future__ import annotations

import math

from moirai.analyze.stats import (
    benjamini_hochberg,
    chi_squared_test,
    fishers_exact_branches,
)
from moirai.schema import Alignment, DivergencePoint, GAP, Run


def find_divergence_points(
    alignment: Alignment,
    runs: list[Run],
    min_branch_size: int = 2,
    q_threshold: float = 0.05,
) -> tuple[list[DivergencePoint], int]:
    """Find columns where runs diverge and correlate with outcome.

    Returns (points, n_candidates_tested). Points are sorted by q-value
    ascending with entropy as tiebreaker. BH correction is applied internally.
    """
    if not alignment.matrix or not alignment.matrix[0]:
        return [], 0

    success_map = {r.run_id: r.result.success for r in runs}
    n_cols = len(alignment.matrix[0])

    candidates: list[DivergencePoint] = []

    for col in range(n_cols):
        values: dict[str, list[str]] = {}
        for run_idx, run_id in enumerate(alignment.run_ids):
            if run_idx < len(alignment.matrix):
                val = alignment.matrix[run_idx][col] if col < len(alignment.matrix[run_idx]) else GAP
                if val != GAP:
                    if val not in values:
                        values[val] = []
                    values[val].append(run_id)

        if len(values) <= 1:
            continue

        value_counts = {v: len(ids) for v, ids in values.items()}

        smallest = min(value_counts.values())
        if smallest < min_branch_size:
            continue

        total = sum(value_counts.values())
        entropy = 0.0
        for count in value_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        success_by_value: dict[str, float | None] = {}
        for val, run_ids in values.items():
            successes = [success_map.get(rid) for rid in run_ids]
            known = [s for s in successes if s is not None]
            if known:
                success_by_value[val] = sum(1 for s in known if s) / len(known)
            else:
                success_by_value[val] = None

        p_val = _compute_significance(values, success_map)

        phase_ctx = _compute_phase_context(col, alignment, runs)

        candidates.append(DivergencePoint(
            column=col,
            value_counts=value_counts,
            entropy=entropy,
            success_by_value=success_by_value,
            p_value=p_val,
            min_branch_size=smallest,
            phase_context=phase_ctx,
        ))

    n_tested = len(candidates)

    # Apply BH correction
    raw_p = [dp.p_value for dp in candidates]
    adjusted = benjamini_hochberg(raw_p, q=q_threshold)
    for dp, qv in zip(candidates, adjusted):
        dp.q_value = qv

    # Filter by q-value. Keep candidates where significance couldn't be computed
    # (p_value=None means insufficient outcome data, not "not significant").
    points = [dp for dp in candidates if dp.q_value is None or dp.q_value <= q_threshold]

    # Sort by q-value ascending, entropy as tiebreaker
    points.sort(key=lambda dp: (dp.q_value if dp.q_value is not None else 1.0, -dp.entropy))

    return points, n_tested


def _compute_significance(
    values: dict[str, list[str]],
    success_map: dict[str, bool | None],
) -> float | None:
    """Compute statistical significance of branch-outcome association."""
    branches: list[tuple[int, int]] = []
    for _, run_ids in values.items():
        s = 0
        f = 0
        for rid in run_ids:
            outcome = success_map.get(rid)
            if outcome is True:
                s += 1
            elif outcome is False:
                f += 1
        branches.append((s, f))

    total_known = sum(succ + fail for succ, fail in branches)
    if total_known < 4:
        return None

    total_s = sum(b[0] for b in branches)
    total_f = sum(b[1] for b in branches)
    if total_s == 0 or total_f == 0:
        return None

    if len(branches) == 2:
        return fishers_exact_branches(branches[0], branches[1])
    else:
        return chi_squared_test(branches)


def _compute_phase_context(col: int, alignment: Alignment, runs: list[Run] | None = None) -> str | None:
    """Compute the phase transition context at a divergence column.

    Returns a string like "explore→[modify vs explore]" describing what
    phase the step before the divergence is in and what phases the branches go to.
    """
    # Find the phase of the step before this column (look at the previous non-gap column)
    prev_phase = None
    if col > 0:
        for run_idx in range(len(alignment.run_ids)):
            prev_col = col - 1
            while prev_col >= 0:
                prev_val = alignment.matrix[run_idx][prev_col] if prev_col < len(alignment.matrix[run_idx]) else GAP
                if prev_val != GAP:
                    prev_phase = _value_to_phase(prev_val, alignment.level)
                    break
                prev_col -= 1
            if prev_phase:
                break

    # Collect the distinct phases at this column
    branch_phases: dict[str, str] = {}  # alignment_value -> phase
    for run_idx in range(len(alignment.run_ids)):
        if run_idx < len(alignment.matrix) and col < len(alignment.matrix[run_idx]):
            val = alignment.matrix[run_idx][col]
            if val != GAP and val not in branch_phases:
                branch_phases[val] = _value_to_phase(val, alignment.level)

    if len(branch_phases) < 2:
        return None

    distinct_phases = set(branch_phases.values())
    if len(distinct_phases) == 1:
        # All branches go to the same phase — the divergence is within a phase
        phase = list(distinct_phases)[0]
        if prev_phase:
            return f"{prev_phase}→[{phase}: {' vs '.join(sorted(branch_phases.keys()))}]"
        return f"[{phase}: {' vs '.join(sorted(branch_phases.keys()))}]"
    else:
        # Branches go to different phases
        parts = [f"{v}={p}" for v, p in sorted(branch_phases.items())]
        if prev_phase:
            return f"{prev_phase}→[{' vs '.join(parts)}]"
        return f"[{' vs '.join(parts)}]"


def _value_to_phase(value: str, level: str) -> str:
    """Map an alignment value to a phase name."""
    from moirai.compress import PHASE_MAP, TYPE_PHASE_MAP

    if level == "name":
        return PHASE_MAP.get(value, "other")
    else:
        return TYPE_PHASE_MAP.get(value, "other")
