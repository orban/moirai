"""Evidence extraction — compare two variants via canonical behavioral features.

Extracts deterministic behavioral features from trajectory structure,
computes shifts between baseline and current variants, and packages
everything into a VariantComparison for downstream cause ranking.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable

from moirai.analyze.stats import cohens_h, effect_magnitude, proportion_delta_ci
from moirai.compress import step_enriched_name
from moirai.schema import FeatureShift, Run, TaskBreakdown, VariantComparison


# --- Canonical behavioral features ---
# Each feature is a function: list[Run] -> float
# Rate features return values in [0, 1]; non-rate features are unbounded.

RATE_FEATURES = {
    "TEST_AFTER_EDIT_RATE",
    "ITERATIVE_FIX_RATE",
    "BLIND_SUBMIT_RATE",
    "TOOL_TIMEOUT_ERROR_RATE",
    "STEP_FAILURE_RATE",
    "SEARCH_BEFORE_EDIT_RATE",
    "REASONING_DENSITY",
}


_names_cache: dict[int, list[str]] = {}


def _enriched_names(run: Run) -> list[str]:
    """Get enriched names, cached by object identity to avoid recomputation in bootstrap."""
    key = id(run)
    cached = _names_cache.get(key)
    if cached is not None:
        return cached
    names = [n for s in run.steps if (n := step_enriched_name(s)) is not None]
    _names_cache[key] = names
    return names


def _has_bigram(names: list[str], a: str, b: str) -> bool:
    """Check if names contains the bigram (a, b) anywhere."""
    for i in range(len(names) - 1):
        if names[i].startswith(a) and names[i + 1].startswith(b):
            return True
    return False


def _has_trigram(names: list[str], a: str, b: str, c: str) -> bool:
    for i in range(len(names) - 2):
        if names[i].startswith(a) and names[i + 1].startswith(b) and names[i + 2].startswith(c):
            return True
    return False



def _test_after_edit_rate(runs: list[Run]) -> float:
    """Fraction of runs where a test step follows an edit step."""
    if not runs:
        return 0.0
    count = sum(1 for r in runs if _has_bigram(_enriched_names(r), "edit", "test"))
    return count / len(runs)


def _iterative_fix_rate(runs: list[Run]) -> float:
    """Fraction of runs containing edit→test→edit motif."""
    if not runs:
        return 0.0
    count = sum(1 for r in runs if _has_trigram(_enriched_names(r), "edit", "test", "edit"))
    return count / len(runs)


def _blind_submit_rate(runs: list[Run]) -> float:
    """Fraction of runs with edit→finish/submit and no intervening test."""
    if not runs:
        return 0.0
    count = 0
    for r in runs:
        names = _enriched_names(r)
        # Look for edit followed by finish without test in between
        has_edit = False
        has_test_after_edit = False
        for name in names:
            if name.startswith("edit"):
                has_edit = True
                has_test_after_edit = False
            elif name.startswith("test"):
                has_test_after_edit = True
            elif name.startswith("finish") or name == "submit":
                if has_edit and not has_test_after_edit:
                    count += 1
                    break
    return count / len(runs)


def _tool_timeout_error_rate(runs: list[Run]) -> float:
    """Fraction of runs that hit at least one tool/bash error (not test failures)."""
    if not runs:
        return 0.0
    count = sum(
        1 for r in runs
        if any(
            s.status in ("error", "timeout") and s.name in ("bash", "execute", "unknown")
            for s in r.steps
        )
    )
    return count / len(runs)


def _step_failure_rate(runs: list[Run]) -> float:
    """Fraction of steps with non-ok status."""
    total = 0
    failures = 0
    for r in runs:
        for s in r.steps:
            total += 1
            if s.status != "ok":
                failures += 1
    return failures / total if total > 0 else 0.0


def _avg_step_count(runs: list[Run]) -> float:
    """Mean trajectory length."""
    if not runs:
        return 0.0
    return sum(len(r.steps) for r in runs) / len(runs)


def _search_before_edit_rate(runs: list[Run]) -> float:
    """Fraction of runs with search→edit pattern."""
    if not runs:
        return 0.0
    count = sum(1 for r in runs if _has_bigram(_enriched_names(r), "search", "edit"))
    return count / len(runs)


def _reasoning_density(runs: list[Run]) -> float:
    """Fraction of steps with non-empty reasoning."""
    total = 0
    with_reasoning = 0
    for r in runs:
        for s in r.steps:
            total += 1
            if s.output and s.output.get("reasoning"):
                with_reasoning += 1
    return with_reasoning / total if total > 0 else 0.0


FEATURE_EXTRACTORS: dict[str, Callable[[list[Run]], float]] = {
    "TEST_AFTER_EDIT_RATE": _test_after_edit_rate,
    "ITERATIVE_FIX_RATE": _iterative_fix_rate,
    "BLIND_SUBMIT_RATE": _blind_submit_rate,
    "TOOL_TIMEOUT_ERROR_RATE": _tool_timeout_error_rate,
    "STEP_FAILURE_RATE": _step_failure_rate,
    "AVG_STEP_COUNT": _avg_step_count,
    "SEARCH_BEFORE_EDIT_RATE": _search_before_edit_rate,
    "REASONING_DENSITY": _reasoning_density,
}


def extract_behavioral_features(runs: list[Run]) -> dict[str, float]:
    """Extract canonical behavioral features from a set of runs."""
    _names_cache.clear()
    return {name: fn(runs) for name, fn in FEATURE_EXTRACTORS.items()}


def compare_variants(
    baseline_runs: list[Run],
    current_runs: list[Run],
    baseline_label: str = "baseline",
    current_label: str = "current",
) -> VariantComparison:
    """Compare two variants and extract structured evidence.

    1. Extract behavioral features per variant
    2. Compute feature shifts with effect sizes and CIs
    3. Run per-task breakdown
    4. Package into VariantComparison
    """
    baseline_features = extract_behavioral_features(baseline_runs)
    current_features = extract_behavioral_features(current_runs)

    # Compute feature shifts
    feature_shifts: list[FeatureShift] = []
    for name in FEATURE_EXTRACTORS:
        bv = baseline_features[name]
        cv = current_features[name]
        shift = cv - bv

        is_rate = name in RATE_FEATURES
        if is_rate:
            es = cohens_h(cv, bv)
            ci_lo, ci_hi = proportion_delta_ci(bv, len(baseline_runs), cv, len(current_runs))
        else:
            # For non-rate features, use a simple normalized difference
            denom = max(abs(bv), abs(cv), 1.0)
            es = shift / denom
            # Rough CI via standard error of the mean
            if len(baseline_runs) > 1 and len(current_runs) > 1:
                b_vals = [len(r.steps) for r in baseline_runs]
                c_vals = [len(r.steps) for r in current_runs]
                b_var = sum((x - bv) ** 2 for x in b_vals) / (len(b_vals) - 1)
                c_var = sum((x - cv) ** 2 for x in c_vals) / (len(c_vals) - 1)
                se = math.sqrt(b_var / len(b_vals) + c_var / len(c_vals))
                ci_lo, ci_hi = shift - 1.96 * se, shift + 1.96 * se
            else:
                ci_lo, ci_hi = shift - abs(shift), shift + abs(shift)

        feature_shifts.append(FeatureShift(
            feature=name,
            baseline_value=bv,
            current_value=cv,
            shift=shift,
            effect_size=es,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            magnitude=effect_magnitude(es),
        ))

    # Sort by absolute shift descending
    feature_shifts.sort(key=lambda fs: -abs(fs.shift))

    # Per-task breakdown
    baseline_by_task: dict[str, list[Run]] = defaultdict(list)
    current_by_task: dict[str, list[Run]] = defaultdict(list)
    for r in baseline_runs:
        baseline_by_task[r.task_id].append(r)
    for r in current_runs:
        current_by_task[r.task_id].append(r)

    all_tasks = set(baseline_by_task.keys()) | set(current_by_task.keys())
    task_breakdown: list[TaskBreakdown] = []
    for tid in sorted(all_tasks):
        b_runs = baseline_by_task.get(tid, [])
        c_runs = current_by_task.get(tid, [])
        b_known = [r for r in b_runs if r.result.success is not None]
        c_known = [r for r in c_runs if r.result.success is not None]
        b_rate = sum(1 for r in b_known if r.result.success) / len(b_known) if b_known else 0.0
        c_rate = sum(1 for r in c_known if r.result.success) / len(c_known) if c_known else 0.0
        task_breakdown.append(TaskBreakdown(
            task_id=tid,
            baseline_pass_rate=b_rate,
            current_pass_rate=c_rate,
            delta=c_rate - b_rate,
            baseline_runs=len(b_runs),
            current_runs=len(c_runs),
        ))

    task_breakdown.sort(key=lambda tb: tb.delta)

    # Overall pass rates
    b_known = [r for r in baseline_runs if r.result.success is not None]
    c_known = [r for r in current_runs if r.result.success is not None]
    b_rate = sum(1 for r in b_known if r.result.success) / len(b_known) if b_known else 0.0
    c_rate = sum(1 for r in c_known if r.result.success) / len(c_known) if c_known else 0.0

    return VariantComparison(
        baseline_label=baseline_label,
        current_label=current_label,
        baseline_pass_rate=b_rate,
        current_pass_rate=c_rate,
        pass_rate_delta=c_rate - b_rate,
        feature_shifts=feature_shifts,
        task_breakdown=task_breakdown,
    )
