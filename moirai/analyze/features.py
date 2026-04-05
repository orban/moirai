"""Behavioral feature computation and within-task natural experiments."""
from __future__ import annotations

import random
import re
from collections.abc import Callable

from moirai.analyze.stats import benjamini_hochberg, sign_test
from moirai.schema import ExperimentResult, FeatureResult, FeatureSpec, Run


# ── Constants ──────────────────────────────────────────────────────

_IDEAL_PHASE_POSITIONS: dict[str, float] = {
    "explore": 0.2,
    "modify": 0.5,
    "verify": 0.85,
}

_HYPOTHESIS_RE = re.compile(
    r'\bi think (?:the|this|it)\b'
    r'|\bthe (?:root )?cause (?:is|might be|could be|seems)\b'
    r'|\bthis (?:fails?|errors?|breaks?) because\b'
    r'|\bthe (?:issue|problem|bug|error) (?:is|seems|appears|might be)\b'
    r'|\bmy (?:hypothesis|theory|guess)\b'
    r'|\bi (?:suspect|believe|hypothesize)\b'
    r'|\bthis (?:is likely|is probably|seems to be|appears to be)\b'
    r'|\bthe reason (?:is|for)\b'
    r"|\bwhat(?:'s| is) happening (?:is|here)\b"
    r'|\bthis (?:suggests?|indicates?|means?)\b',
    re.IGNORECASE,
)


# ── Feature functions ──────────────────────────────────────────────
# Each: (Run) -> float | None. Returns None when data is insufficient.


def _test_position_centroid(run: Run) -> float | None:
    """Normalized center of mass of test step positions (0-1).

    Uses swarm convention: position = i / (n-1) where n = total enriched steps.
    Returns None if < 5 enriched steps or < 2 test steps.
    """
    from moirai.compress import step_enriched_name

    enriched = [step_enriched_name(s) for s in run.steps]
    enriched = [e for e in enriched if e is not None]
    n = len(enriched)
    if n < 5:
        return None

    test_positions = [
        i / (n - 1)
        for i, name in enumerate(enriched)
        if name.startswith("test(")
    ]
    if len(test_positions) < 2:
        return None

    return sum(test_positions) / len(test_positions)


def _symmetry_index(run: Run) -> float | None:
    """Mean ideal-position score across classifiable steps.

    Each explore/modify/verify step scored as 1 - |actual_pos - ideal_pos|.
    Ideal: explore@0.2, modify@0.5, verify@0.85.
    Returns None if < 6 steps or < 4 classifiable steps.
    """
    from moirai.compress import phase_sequence

    phases = phase_sequence(run)
    n = len(phases)
    if n < 6:
        return None

    scores: list[float] = []
    for i, phase in enumerate(phases):
        ideal = _IDEAL_PHASE_POSITIONS.get(phase)
        if ideal is None:
            continue
        actual_pos = i / (n - 1)
        scores.append(1.0 - abs(actual_pos - ideal))

    if len(scores) < 4:
        return None

    return sum(scores) / len(scores)


def _uncertainty_density(run: Run) -> float | None:
    """Fraction of reasoning steps matching UNCERTAINTY_RE.

    Returns None if run has no reasoning steps.
    """
    # Import the regex -- handle both the renamed public version and the
    # current private name for backward compatibility during transition.
    try:
        from moirai.analyze.content import UNCERTAINTY_RE
    except ImportError:
        from moirai.analyze.content import _UNCERTAINTY_RE as UNCERTAINTY_RE  # type: ignore[attr-defined]

    n_reasoning = 0
    n_matches = 0
    for step in run.steps:
        reasoning = str(step.output.get("reasoning", ""))
        if not reasoning:
            continue
        n_reasoning += 1
        if UNCERTAINTY_RE.search(reasoning):
            n_matches += 1

    if n_reasoning == 0:
        return None

    return n_matches / n_reasoning


def _hypothesis_formation(run: Run) -> float | None:
    """Count of hypothesis pattern matches per reasoning step.

    Returns None if run has no reasoning steps.
    """
    n_reasoning = 0
    total_matches = 0
    for step in run.steps:
        reasoning = str(step.output.get("reasoning", ""))
        if not reasoning:
            continue
        n_reasoning += 1
        total_matches += len(_HYPOTHESIS_RE.findall(reasoning.lower()))

    if n_reasoning == 0:
        return None

    return total_matches / n_reasoning


def _edit_dispersion(run: Run) -> float | None:
    """unique_files / total_edits. 1.0 = every edit hits a different file.

    Returns None if run has < 2 edit steps.
    """
    from moirai.compress import step_enriched_name

    file_paths: list[str] = []
    for step in run.steps:
        enriched = step_enriched_name(step)
        if enriched is None:
            continue
        if not enriched.startswith("edit("):
            continue
        fp = step.attrs.get("file_path", "")
        file_paths.append(fp)

    if len(file_paths) < 2:
        return None

    unique = len(set(file_paths))
    return unique / len(file_paths)


# ── Feature registry ───────────────────────────────────────────────

FEATURES: list[FeatureSpec] = [
    FeatureSpec("test_position_centroid", _test_position_centroid, "positive",
                "Where in the trajectory testing is concentrated"),
    FeatureSpec("symmetry_index", _symmetry_index, "positive",
                "How well the trajectory follows explore\u2192modify\u2192verify"),
    FeatureSpec("uncertainty_density", _uncertainty_density, "negative",
                "Hedging language density in reasoning text"),
    FeatureSpec("reasoning_hypothesis_formation", _hypothesis_formation, "negative",
                "Explicit hypothesis formation in reasoning"),
    FeatureSpec("edit_dispersion", _edit_dispersion, "positive",
                "Whether edits spread across files (higher = more spread)"),
]


# ── Natural experiment ─────────────────────────────────────────────


def within_task_experiment(
    task_runs: dict[str, list[Run]],
    feature_fn: Callable[[Run], float | None],
    min_group_size: int = 3,
) -> ExperimentResult:
    """Within-task median-split natural experiment.

    For each task:
      1. Compute feature value per run
      2. Split at median: HIGH = value > median, LOW = value <= median
      3. Compare pass rates: HIGH group vs LOW group
      4. Record the delta for this task

    Across tasks: sign test (exact binomial) on the per-task deltas.
    Skip tasks where either group has < min_group_size runs.
    """
    deltas: list[float] = []
    all_pass_vals: list[float] = []
    all_fail_vals: list[float] = []

    for task_id, runs in task_runs.items():
        # Compute feature for each run
        valued: list[tuple[float, bool]] = []
        for r in runs:
            v = feature_fn(r)
            if v is not None and r.result.success is not None:
                valued.append((v, r.result.success))

        if len(valued) < 2 * min_group_size:
            continue

        # Collect pass/fail values for global means
        for v, success in valued:
            if success:
                all_pass_vals.append(v)
            else:
                all_fail_vals.append(v)

        # Median split
        values = sorted(v for v, _ in valued)
        median = values[len(values) // 2]

        high = [(v, s) for v, s in valued if v > median]
        low = [(v, s) for v, s in valued if v <= median]

        if len(high) < min_group_size or len(low) < min_group_size:
            continue

        high_pass_rate = sum(1 for _, s in high if s) / len(high)
        low_pass_rate = sum(1 for _, s in low if s) / len(low)
        deltas.append(high_pass_rate - low_pass_rate)

    if not deltas:
        return ExperimentResult(delta_pp=0.0, p_value=None, n_tasks=0)

    mean_delta = sum(deltas) / len(deltas)
    p = sign_test(deltas)

    return ExperimentResult(
        delta_pp=mean_delta * 100,  # convert to percentage points
        p_value=p,
        n_tasks=len(deltas),
    )


def per_task_deltas(
    task_runs: dict[str, list[Run]],
    feature_fn: Callable[[Run], float | None],
    min_group_size: int = 3,
) -> list[float]:
    """Return the raw per-task pass-rate deltas from a median-split experiment.

    Same logic as within_task_experiment but returns the individual deltas
    instead of aggregating. Useful for histograms and diagnostics.
    """
    deltas: list[float] = []

    for task_id, runs in task_runs.items():
        valued: list[tuple[float, bool]] = []
        for r in runs:
            v = feature_fn(r)
            if v is not None and r.result.success is not None:
                valued.append((v, r.result.success))

        if len(valued) < 2 * min_group_size:
            continue

        values = sorted(v for v, _ in valued)
        median = values[len(values) // 2]

        high = [(v, s) for v, s in valued if v > median]
        low = [(v, s) for v, s in valued if v <= median]

        if len(high) < min_group_size or len(low) < min_group_size:
            continue

        high_pass_rate = sum(1 for _, s in high if s) / len(high)
        low_pass_rate = sum(1 for _, s in low if s) / len(low)
        deltas.append(high_pass_rate - low_pass_rate)

    return deltas


# ── Main pipeline ──────────────────────────────────────────────────


def rank_features(
    runs: list[Run],
    min_runs: int = 10,
    seed: int = 42,
) -> list[FeatureResult]:
    """Full pipeline: compute features, run natural experiments, validate, rank.

    1. Group runs by task_id, filter to mixed-outcome tasks with min_runs
    2. For each feature: within_task_experiment → delta, p-value
    3. Inline split-half: random 50/50 task split, check both halves p < 0.05
    4. BH-correct p-values across features
    5. Sort by |delta_pp| descending
    """
    from moirai.analyze.content import select_task_groups

    task_runs, _ = select_task_groups(runs, min_runs=min_runs)
    if not task_runs:
        return []

    # Split-half task keys (deterministic via seed)
    task_keys = sorted(task_runs.keys())
    rng = random.Random(seed)
    shuffled = list(task_keys)
    rng.shuffle(shuffled)
    mid = len(shuffled) // 2
    half_a_keys = set(shuffled[:mid])
    half_b_keys = set(shuffled[mid:])
    half_a = {k: v for k, v in task_runs.items() if k in half_a_keys}
    half_b = {k: v for k, v in task_runs.items() if k in half_b_keys}

    results: list[FeatureResult] = []

    for spec in FEATURES:
        # Full experiment
        exp = within_task_experiment(task_runs, spec.compute)

        # Compute pass/fail means
        pass_vals = [spec.compute(r) for r in runs
                     if r.result.success is True]
        fail_vals = [spec.compute(r) for r in runs
                     if r.result.success is False]
        pass_vals = [v for v in pass_vals if v is not None]
        fail_vals = [v for v in fail_vals if v is not None]
        pass_mean = sum(pass_vals) / len(pass_vals) if pass_vals else 0.0
        fail_mean = sum(fail_vals) / len(fail_vals) if fail_vals else 0.0

        # Split-half validation (inline)
        exp_a = within_task_experiment(half_a, spec.compute)
        exp_b = within_task_experiment(half_b, spec.compute)
        split_half = (
            exp_a.p_value is not None and exp_a.p_value < 0.05
            and exp_b.p_value is not None and exp_b.p_value < 0.05
        )

        results.append(FeatureResult(
            name=spec.name,
            description=spec.description,
            direction=spec.direction,
            delta_pp=exp.delta_pp,
            p_value=exp.p_value,
            q_value=None,  # filled after BH
            split_half=split_half,
            pass_mean=pass_mean,
            fail_mean=fail_mean,
            n_tasks=exp.n_tasks,
            n_runs=len(pass_vals) + len(fail_vals),
        ))

    # BH correction across features
    raw_p = [r.p_value for r in results]
    adjusted = benjamini_hochberg(raw_p)
    for r, q in zip(results, adjusted):
        r.q_value = q

    # Sort by |delta_pp| descending
    results.sort(key=lambda r: -abs(r.delta_pp))

    return results
