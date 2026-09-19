from __future__ import annotations

import math

from moirai.analyze.stats import (
    benjamini_hochberg,
    chi_squared_test,
    fishers_exact_2x2,
    fishers_exact_branches,
)
from moirai.schema import ActivityDivergence, Alignment, DivergencePoint, GAP, Run


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
    adjusted = benjamini_hochberg(raw_p)
    for dp, qv in zip(candidates, adjusted):
        dp.q_value = qv

    # Filter by q-value. Keep candidates where significance couldn't be computed
    # (p_value=None means insufficient outcome data, not "not significant").
    points = [dp for dp in candidates if dp.q_value is None or dp.q_value <= q_threshold]

    # Sort by q-value ascending, entropy as tiebreaker
    points.sort(key=lambda dp: (dp.q_value if dp.q_value is not None else 1.0, -dp.entropy))

    return points, n_tested


def find_activity_divergences(
    alignment: Alignment,
    runs: list[Run],
    q_threshold: float = 1.0,
) -> list[ActivityDivergence]:
    """Find columns where having a step (vs gap) predicts outcome.

    For each alignment column, builds a 2x2 table:
        (active/gap) x (pass/fail)
    and runs Fisher's exact test. BH correction is applied across all columns.

    Complements ``find_divergence_points`` which tests whether different
    step *types* predict outcome. This function tests whether *being active
    at all* predicts outcome — useful when the key signal is that one group
    skips a step entirely.

    Returns all columns sorted by q-value ascending. Use ``q_threshold``
    to filter (default 1.0 = return everything, let caller decide).
    """
    if not alignment.matrix or not alignment.matrix[0]:
        return []

    success_map = {r.run_id: r.result.success for r in runs}
    n_cols = len(alignment.matrix[0])
    n_runs = len(alignment.run_ids)

    # Partition run indices by outcome
    pass_idxs = [i for i, rid in enumerate(alignment.run_ids)
                 if success_map.get(rid) is True]
    fail_idxs = [i for i, rid in enumerate(alignment.run_ids)
                 if success_map.get(rid) is False]

    if not pass_idxs or not fail_idxs:
        return []

    candidates: list[ActivityDivergence] = []

    for col in range(n_cols):
        pa = sum(1 for i in pass_idxs if alignment.matrix[i][col] != GAP)
        pg = len(pass_idxs) - pa
        fa = sum(1 for i in fail_idxs if alignment.matrix[i][col] != GAP)
        fg = len(fail_idxs) - fa

        # Skip columns where everyone is active or everyone is a gap
        if (pa + fa == 0) or (pg + fg == 0):
            continue

        p = fishers_exact_2x2(pa, pg, fa, fg)
        if p is None:
            continue

        # Direction: positive = pass runs more active here
        pass_rate = pa / len(pass_idxs) if pass_idxs else 0
        fail_rate = fa / len(fail_idxs) if fail_idxs else 0

        # Collect step labels at this column
        labels: dict[str, int] = {}
        for i in range(n_runs):
            val = alignment.matrix[i][col]
            if val != GAP:
                labels[val] = labels.get(val, 0) + 1

        phase_ctx = _compute_phase_context(col, alignment, runs)

        candidates.append(ActivityDivergence(
            column=col,
            pass_active=pa,
            pass_gap=pg,
            fail_active=fa,
            fail_gap=fg,
            p_value=p,
            direction=pass_rate - fail_rate,
            active_labels=labels,
            phase_context=phase_ctx,
        ))

    # BH correction
    raw_p = [c.p_value for c in candidates]
    adjusted = benjamini_hochberg(raw_p)
    for c, qv in zip(candidates, adjusted):
        c.q_value = qv

    result = [c for c in candidates if c.q_value is not None and c.q_value <= q_threshold]
    result.sort(key=lambda c: (c.q_value if c.q_value is not None else 1.0, -abs(c.direction)))

    return result


def _action_label(name: str) -> str:
    """Convert a step name like 'read(test_file)' to readable prose."""
    base = name.split("(")[0] if "(" in name else name
    target = name.split("(")[1].rstrip(")") if "(" in name else ""

    labels = {
        "read": "read" + (f" {target}" if target else ""),
        "search": "searched" + (f" ({target})" if target else ""),
        "edit": "edited" + (f" {target}" if target else ""),
        "write": "wrote" + (f" {target}" if target else ""),
        "test": "ran tests" + (f" ({target})" if target else ""),
        "bash": "ran a command" + (f" ({target})" if target else ""),
        "reason": "stopped to reason",
        "subagent": "delegated to subagent",
        "plan": "planned",
    }
    return labels.get(base, name)


def _best_worst(
    point: DivergencePoint,
    min_support: int = 1,
) -> tuple[str, float, int, str, float, int] | None:
    """Find the best and worst variants by pass rate.

    Only considers variants with at least min_support runs.
    Returns (best_name, best_rate, best_count, worst_name, worst_rate, worst_count)
    or None if rates are unavailable or insufficient support.
    """
    rated = [
        (v, point.success_by_value[v], c)
        for v, c in point.value_counts.items()
        if point.success_by_value.get(v) is not None and c >= min_support
    ]
    if len(rated) < 2:
        return None
    rated.sort(key=lambda x: -x[1])
    best_name, best_rate, best_count = rated[0]
    worst_name, worst_rate, worst_count = rated[-1]
    if best_rate == worst_rate:
        return None
    return best_name, best_rate, best_count, worst_name, worst_rate, worst_count


def summarize_point(point: DivergencePoint) -> str:
    """Sharp, opinionated summary of a divergence point.

    Takes a stance: names the better path and the worse path explicitly.
    """
    variants = sorted(point.value_counts.items(), key=lambda x: -x[1])
    if not variants:
        return f"Divergence at position {point.column}"

    bw = _best_worst(point)
    if bw is None:
        # No rate difference — just describe the split
        top = [v for v, _ in variants if v != GAP][:2]
        if not top:
            return f"Divergence at position {point.column}"
        return f"Runs diverge at step {point.column}: {_action_label(top[0])} vs {_action_label(top[1]) if len(top) > 1 else 'gap'}"

    best_name, best_rate, best_count, worst_name, worst_rate, worst_count = bw
    delta = best_rate - worst_rate

    # GAP pattern
    if best_name == GAP:
        return (
            f"Skipping step {point.column} correlates with success: "
            f"{best_rate:.0%} pass ({best_count} runs) vs {worst_rate:.0%} when runs {_action_label(worst_name)} "
            f"({worst_count} runs)"
        )
    if worst_name == GAP:
        return (
            f"Runs that {_action_label(best_name)} at step {point.column} pass {best_rate:.0%} "
            f"({best_count} runs) — those that skip it pass {worst_rate:.0%} ({worst_count} runs)"
        )

    # Edit-vs-test ordering
    best_base = best_name.split("(")[0] if "(" in best_name else best_name
    worst_base = worst_name.split("(")[0] if "(" in worst_name else worst_name
    edit_types = {"edit", "write"}
    verify_types = {"test"}
    if (best_base in edit_types and worst_base in verify_types) or (best_base in verify_types and worst_base in edit_types):
        winner = _action_label(best_name)
        loser = _action_label(worst_name)
        return (
            f"Runs that {winner} at step {point.column} pass {best_rate:.0%} — "
            f"those that {loser} pass {worst_rate:.0%} ({delta:+.0%} difference)"
        )

    # Same action type, different targets
    if best_base == worst_base and best_name != worst_name:
        best_target = best_name.split("(")[1].rstrip(")") if "(" in best_name else best_name
        worst_target = worst_name.split("(")[1].rstrip(")") if "(" in worst_name else worst_name
        return (
            f"Choice of {best_base} target matters: {best_target} → {best_rate:.0%} pass "
            f"vs {worst_target} → {worst_rate:.0%} pass ({delta:+.0%})"
        )

    # Generic — but still take a stance
    return (
        f"At step {point.column}, {_action_label(best_name)} correlates with success ({best_rate:.0%} pass, "
        f"{best_count} runs) while {_action_label(worst_name)} correlates with failure "
        f"({worst_rate:.0%} pass, {worst_count} runs)"
    )


def generate_claim(
    points: list[DivergencePoint],
    n_runs: int,
    min_support: int = 3,
) -> str | None:
    """Generate the one-sentence claim for a task — the strongest divergence.

    This is the hero: the single most provocative finding.
    Requires at least min_support runs on BOTH sides of the split.
    A "100% gap" from 1 run vs 1 run is noise, not a finding.
    Returns None if no divergence has a meaningful, well-supported outcome gap.
    """
    if not points:
        return None

    # Find the point with the largest pass rate gap, requiring support
    best_point = None
    best_delta = 0.0
    best_bw = None
    for p in points:
        bw = _best_worst(p, min_support=min_support)
        if bw is None:
            continue
        delta = bw[1] - bw[4]  # best_rate - worst_rate
        if delta > best_delta:
            best_delta = delta
            best_point = p
            best_bw = bw

    if best_point is None or best_delta < 0.10:
        return None

    best_name, best_rate, best_count, worst_name, worst_rate, worst_count = best_bw

    return (
        f"Critical divergence at step {best_point.column}: "
        f"{_action_label(best_name)} → {best_rate:.0%} success ({best_count} runs) vs "
        f"{_action_label(worst_name)} → {worst_rate:.0%} success ({worst_count} runs). "
        f"{best_delta:.0%} outcome gap."
    )


def build_variant_list(point: DivergencePoint) -> list[dict]:
    """Build a canonical list of variant dicts from a DivergencePoint.

    Shared by terminal output, JSON export, and HTML report to avoid
    duplicating the value_counts → variant conversion logic.
    """
    variants = []
    for value, count in sorted(point.value_counts.items(), key=lambda x: -x[1]):
        rate = point.success_by_value.get(value)
        n_pass = round(rate * count) if rate is not None else None
        variants.append({
            "value": value,
            "n_runs": count,
            "n_pass": n_pass,
            "n_fail": count - n_pass if n_pass is not None else None,
            "pass_rate": rate,
        })
    return variants


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
