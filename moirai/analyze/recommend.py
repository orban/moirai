"""Synthesize analysis results into actionable recommendations."""
from __future__ import annotations

from dataclasses import dataclass

from moirai.analyze.motifs import Motif
from moirai.schema import DivergencePoint


@dataclass
class Recommendation:
    """An actionable recommendation with supporting evidence."""
    action: str           # what to do, in plain language
    impact: str           # expected effect
    evidence: list[str]   # supporting findings
    priority: float       # higher = more important (for sorting)


def synthesize(
    motifs: list[Motif],
    divergence_points: list[DivergencePoint],
    n_runs: int,
    baseline_rate: float,
) -> list[Recommendation]:
    """Synthesize motifs and divergence points into ranked recommendations.

    Groups related findings, frames them as actions, and ranks by impact.
    """
    recs: list[Recommendation] = []

    # 1. Find the biggest failure pattern and recommend against it
    fail_motifs = [m for m in motifs if m.lift < 0.5 and m.p_value is not None and m.p_value < 0.1]
    if fail_motifs:
        biggest_fail = max(fail_motifs, key=lambda m: m.total_runs)
        # Find related failure motifs (overlapping steps)
        fail_steps = set(biggest_fail.pattern)
        related = [m for m in fail_motifs if m != biggest_fail and set(m.pattern) & fail_steps]

        action = _describe_avoid_pattern(biggest_fail)
        impact = f"{biggest_fail.total_runs} of {n_runs} runs ({biggest_fail.total_runs/n_runs:.0%}) follow this pattern and succeed only {biggest_fail.success_rate:.0%} of the time."
        evidence = [f"`{biggest_fail.display}` → {biggest_fail.success_rate:.0%} success ({biggest_fail.total_runs} runs, p={biggest_fail.p_value:.3f})"]
        for m in related[:2]:
            evidence.append(f"`{m.display}` → {m.success_rate:.0%} success ({m.total_runs} runs)")

        # Check if any divergence point corroborates
        for dp in divergence_points:
            for val, rate in dp.success_by_value.items():
                if rate is not None and rate <= 0.15 and val in fail_steps:
                    evidence.append(f"Position {dp.column}: `{val}` branch → {rate:.0%} success ({dp.value_counts[val]} runs)")
                    break

        recs.append(Recommendation(
            action=action,
            impact=impact,
            evidence=evidence,
            priority=biggest_fail.total_runs * (baseline_rate - biggest_fail.success_rate),
        ))

    # 2. Find the biggest success pattern and recommend it
    success_motifs = [m for m in motifs if m.lift > 2.0 and m.p_value is not None and m.p_value < 0.1]
    if success_motifs:
        biggest_success = max(success_motifs, key=lambda m: m.total_runs)
        related = [m for m in success_motifs if m != biggest_success
                   and set(m.pattern) & set(biggest_success.pattern)]

        action = _describe_encourage_pattern(biggest_success)
        n_without = n_runs - biggest_success.total_runs
        rate_without = _rate_without(motifs, biggest_success, n_runs, baseline_rate)
        impact = f"Runs with this pattern succeed {biggest_success.success_rate:.0%} vs {rate_without:.0%} without it."
        evidence = [f"`{biggest_success.display}` → {biggest_success.success_rate:.0%} success ({biggest_success.total_runs} runs, p={biggest_success.p_value:.3f})"]
        for m in related[:2]:
            evidence.append(f"`{m.display}` → {m.success_rate:.0%} success ({m.total_runs} runs)")

        recs.append(Recommendation(
            action=action,
            impact=impact,
            evidence=evidence,
            priority=biggest_success.total_runs * (biggest_success.success_rate - baseline_rate),
        ))

    # 3. Find divergence points where one branch clearly dominates
    for dp in divergence_points[:5]:
        branches = sorted(dp.success_by_value.items(), key=lambda x: -(x[1] or 0))
        if len(branches) < 2:
            continue
        best_val, best_rate = branches[0]
        worst_val, worst_rate = branches[-1]
        if best_rate is None or worst_rate is None:
            continue
        if best_rate - worst_rate < 0.4:
            continue  # not a strong enough difference
        # Skip if this is already covered by the motif recommendations
        if recs and any(best_val in r.evidence[0] or worst_val in r.evidence[0] for r in recs):
            continue

        best_count = dp.value_counts[best_val]
        worst_count = dp.value_counts[worst_val]
        phase = dp.phase_context or ""

        action = f"At the {_position_label(dp, len(divergence_points))} decision point"
        if phase:
            action += f" ({phase.split('→')[0].strip()} phase)"
        action += f", prefer `{best_val}` over `{worst_val}`."

        impact = f"`{best_val}` succeeds {best_rate:.0%} ({best_count} runs) vs `{worst_val}` at {worst_rate:.0%} ({worst_count} runs)."
        evidence = [f"Position {dp.column} (p={dp.p_value:.3f})" if dp.p_value else f"Position {dp.column}"]

        recs.append(Recommendation(
            action=action,
            impact=impact,
            evidence=evidence,
            priority=worst_count * (best_rate - worst_rate),
        ))

    recs.sort(key=lambda r: -r.priority)
    return recs[:5]  # top 5 recommendations


def _describe_avoid_pattern(motif: Motif) -> str:
    """Describe a failure pattern as an action to avoid."""
    steps = motif.pattern
    pos = "early" if motif.avg_position < 0.3 else ("late" if motif.avg_position > 0.7 else ""  )

    # Look for common anti-patterns
    if "bash" in steps and ("test" in steps or "test_result" in steps):
        return f"Avoid falling back to bash commands after test failures{_pos_phrase(pos)}."
    if steps.count(steps[0]) >= 3:
        return f"Break out of `{steps[0]}` loops{_pos_phrase(pos)} — repeated {steps[0]} without progress correlates with failure."
    if "read" in steps and steps.count("read") >= 2 and "edit" not in steps and "write" not in steps:
        return f"Avoid excessive reading without modification{_pos_phrase(pos)} — this pattern correlates with getting stuck."

    return f"Avoid the `{motif.display}` pattern{_pos_phrase(pos)}."


def _describe_encourage_pattern(motif: Motif) -> str:
    """Describe a success pattern as an action to encourage."""
    steps = motif.pattern
    pos = "early" if motif.avg_position < 0.3 else ("late" if motif.avg_position > 0.7 else "")

    if "subagent" in steps:
        return f"Use subagent delegation{_pos_phrase(pos)} for exploration."
    if "reason" in steps:
        return f"Add reasoning steps{_pos_phrase(pos)} — thinking before acting correlates with success."
    if "plan" in steps:
        return f"Plan{_pos_phrase(pos)} — runs that pause to plan recover from failures."
    if "search" in steps and ("edit" in steps or "write" in steps):
        return f"Search before modifying{_pos_phrase(pos)} — informed edits succeed more."

    return f"Encourage the `{motif.display}` pattern{_pos_phrase(pos)}."


def _pos_phrase(pos: str) -> str:
    if pos == "early":
        return " in early steps"
    if pos == "late":
        return " in later steps"
    return ""


def _position_label(dp: DivergencePoint, total: int) -> str:
    """Describe a divergence point's position in human terms."""
    if dp.column < 20:
        return "an early"
    if dp.column > 60:
        return "a late"
    return "a mid-trajectory"


def _rate_without(motifs: list[Motif], target: Motif, n_runs: int, baseline: float) -> float:
    """Estimate success rate for runs WITHOUT the target pattern."""
    n_with = target.total_runs
    n_success_with = target.success_runs
    total_success = int(baseline * n_runs)
    n_without = n_runs - n_with
    if n_without <= 0:
        return baseline
    success_without = total_success - n_success_with
    return max(0, success_without / n_without)
