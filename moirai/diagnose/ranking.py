"""Evidence-weighted cause ranking.

Scores candidate causes by matching observed feature shifts against
each cause's declared expected_shifts. This is an honest scoring
heuristic, not Bayesian inference.
"""
from __future__ import annotations

import math
import random

from moirai.analyze.evidence import compare_variants
from moirai.diagnose.causes import CandidateCause
from moirai.schema import (
    CauseScore,
    DiagnosisResult,
    Run,
    VariantComparison,
)


def _direction_matches(shift: float, expected: str) -> bool:
    """Check if an observed shift matches the expected direction."""
    if expected == "increase":
        return shift > 0
    if expected == "decrease":
        return shift < 0
    return False


def _auto_priors(causes: list[CandidateCause], unknown_prior: float) -> tuple[dict[str, float], float]:
    """Assign uniform priors to causes with prior=0.0.

    Returns (cause_id → prior, unknown_prior).
    """
    n = len(causes) + 1  # +1 for unknown bucket
    auto_prior = 1.0 / n

    priors = {}
    for c in causes:
        priors[c.id] = c.prior if c.prior > 0 else auto_prior

    if unknown_prior <= 0:
        unknown_prior = auto_prior

    # Normalize so all priors sum to 1.0
    total = sum(priors.values()) + unknown_prior
    priors = {k: v / total for k, v in priors.items()}
    unknown_prior = unknown_prior / total

    return priors, unknown_prior


def score_causes(
    comparison: VariantComparison,
    causes: list[CandidateCause],
    unknown_prior: float = 0.0,
) -> DiagnosisResult:
    """Rank candidate causes by evidence-weighted scoring.

    For each cause:
    1. Iterate expected_shifts: direction match → +|shift|, mismatch → -|shift|×0.5
    2. Unclaimed features → unknown bucket
    3. Apply priors, exp(raw), normalize to simplex
    """
    priors, unk_prior = _auto_priors(causes, unknown_prior)

    # Build feature shift lookup
    shift_by_feature = {fs.feature: fs for fs in comparison.feature_shifts}

    # All features claimed by any cause
    claimed_features: set[str] = set()
    for c in causes:
        claimed_features.update(c.expected_shifts.keys())

    # Score each cause
    raw_scores: dict[str, float] = {}
    matched_features: dict[str, list[str]] = {}

    for c in causes:
        raw = 0.0
        matched = []
        for feature, expected_dir in c.expected_shifts.items():
            fs = shift_by_feature.get(feature)
            if fs is None:
                continue
            # Use effect_size (normalized) not raw shift (scale-dependent)
            strength = abs(fs.effect_size)
            if _direction_matches(fs.shift, expected_dir):
                raw += strength
                matched.append(feature)
            else:
                raw -= strength * 0.5  # evidence against, damped

        raw_scores[c.id] = raw
        matched_features[c.id] = matched

    # Unknown bucket: unclaimed features with non-trivial effect sizes
    unclaimed_shift = sum(
        abs(fs.effect_size) for fs in comparison.feature_shifts
        if fs.feature not in claimed_features and abs(fs.effect_size) > 0.1
    )
    raw_scores["_unknown"] = unclaimed_shift

    # Apply priors and exponentiate (clamp to avoid overflow)
    weighted: dict[str, float] = {}
    for c in causes:
        weighted[c.id] = priors[c.id] * math.exp(min(raw_scores[c.id], 50.0))
    weighted["_unknown"] = unk_prior * math.exp(min(raw_scores["_unknown"], 50.0))

    # Normalize to simplex
    total = sum(weighted.values())
    if total == 0:
        total = 1.0

    # Build CauseScore list
    cause_scores: list[CauseScore] = []
    for c in causes:
        score = weighted[c.id] / total
        cause_scores.append(CauseScore(
            cause_id=c.id,
            cause_description=c.description,
            score=score,
            ci_lower=score,  # will be updated by bootstrap
            ci_upper=score,
            matched_features=matched_features[c.id],
            evidence_strength=raw_scores[c.id],
        ))

    cause_scores.sort(key=lambda cs: -cs.score)
    unknown_score = weighted["_unknown"] / total

    return DiagnosisResult(
        cause_scores=cause_scores,
        unknown_score=unknown_score,
        feature_shifts=comparison.feature_shifts,
        task_breakdown=comparison.task_breakdown,
        baseline_pass_rate=comparison.baseline_pass_rate,
        current_pass_rate=comparison.current_pass_rate,
    )


def bootstrap_confidence(
    baseline_runs: list[Run],
    current_runs: list[Run],
    causes: list[CandidateCause],
    n_bootstrap: int = 200,
    seed: int = 42,
) -> DiagnosisResult:
    """Score causes with bootstrap confidence intervals.

    Resamples runs with replacement, recomputes evidence + ranking each time.
    Returns the point-estimate result with CIs filled in.
    """
    rng = random.Random(seed)

    # Point estimate
    comparison = compare_variants(baseline_runs, current_runs)
    result = score_causes(comparison, causes)

    # Bootstrap
    score_samples: dict[str, list[float]] = {cs.cause_id: [] for cs in result.cause_scores}
    score_samples["_unknown"] = []

    for _ in range(n_bootstrap):
        b_sample = rng.choices(baseline_runs, k=len(baseline_runs))
        c_sample = rng.choices(current_runs, k=len(current_runs))

        boot_comparison = compare_variants(b_sample, c_sample)
        boot_result = score_causes(boot_comparison, causes)

        for cs in boot_result.cause_scores:
            if cs.cause_id in score_samples:
                score_samples[cs.cause_id].append(cs.score)
        score_samples["_unknown"].append(boot_result.unknown_score)

    # Compute 95% CIs from bootstrap samples
    for cs in result.cause_scores:
        samples = sorted(score_samples.get(cs.cause_id, [cs.score]))
        if len(samples) >= 20:
            lo_idx = int(len(samples) * 0.025)
            hi_idx = int(len(samples) * 0.975)
            cs.ci_lower = samples[lo_idx]
            cs.ci_upper = samples[hi_idx]

    return result
