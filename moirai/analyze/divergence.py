from __future__ import annotations

import math

from moirai.schema import Alignment, DivergencePoint, GAP, Run


def find_divergence_points(
    alignment: Alignment,
    runs: list[Run],
    min_branch_size: int = 2,
    p_threshold: float = 0.2,
) -> list[DivergencePoint]:
    """Find columns where runs diverge and correlate with outcome.

    Improvements over v0:
    1. Stability: filters out points where any branch has fewer than min_branch_size runs
    2. Significance: computes Fisher's exact test p-value, filters by p_threshold
    3. Phase context: annotates each point with the phase transition happening

    Points are sorted by p_value ascending (most significant first),
    then by entropy descending as tiebreaker.
    """
    if not alignment.matrix or not alignment.matrix[0]:
        return []

    success_map = {r.run_id: r.result.success for r in runs}
    n_cols = len(alignment.matrix[0])

    points: list[DivergencePoint] = []

    for col in range(n_cols):
        values: dict[str, list[str]] = {}  # value -> list of run_ids
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

        # 1. Stability filter: skip if any branch is too small
        smallest = min(value_counts.values())
        if smallest < min_branch_size:
            continue

        # Entropy
        total = sum(value_counts.values())
        entropy = 0.0
        for count in value_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        # Success rates per value
        success_by_value: dict[str, float | None] = {}
        for val, run_ids in values.items():
            successes = [success_map.get(rid) for rid in run_ids]
            known = [s for s in successes if s is not None]
            if known:
                success_by_value[val] = sum(1 for s in known if s) / len(known)
            else:
                success_by_value[val] = None

        # 2. Significance: Fisher's exact test (or chi-squared for >2 branches)
        # None = not enough outcome data to test (all-null, all-same, <4 known).
        # Only filter when we CAN compute significance and it exceeds threshold.
        p_val = _compute_significance(values, success_map)
        if p_val is not None and p_val > p_threshold:
            continue

        # 3. Phase context
        phase_ctx = _compute_phase_context(col, alignment, runs)

        points.append(DivergencePoint(
            column=col,
            value_counts=value_counts,
            entropy=entropy,
            success_by_value=success_by_value,
            p_value=p_val,
            min_branch_size=smallest,
            phase_context=phase_ctx,
        ))

    # Sort by significance first (lowest p-value), then entropy as tiebreaker
    def sort_key(p: DivergencePoint) -> tuple[float, float]:
        pv = p.p_value if p.p_value is not None else 1.0
        return (pv, -p.entropy)

    points.sort(key=sort_key)
    return points


def _compute_significance(
    values: dict[str, list[str]],
    success_map: dict[str, bool | None],
) -> float | None:
    """Compute statistical significance of branch-outcome association.

    Uses Fisher's exact test for 2 branches, chi-squared for >2.
    Returns p-value, or None if not enough data.
    """
    # Build contingency: for each branch, count successes and failures
    branches: list[tuple[int, int]] = []  # (successes, failures) per branch
    for _val, run_ids in values.items():
        s = 0
        f = 0
        for rid in run_ids:
            outcome = success_map.get(rid)
            if outcome is True:
                s += 1
            elif outcome is False:
                f += 1
        branches.append((s, f))

    # Need at least some known outcomes
    total_known = sum(succ + fail for succ, fail in branches)
    if total_known < 4:
        return None

    # Need variation in outcomes (not all-pass or all-fail)
    total_s = sum(succ for succ, _fail in branches)
    total_f = sum(fail for _succ, fail in branches)
    if total_s == 0 or total_f == 0:
        return None

    if len(branches) == 2:
        return _fishers_exact(branches[0], branches[1])
    else:
        return _chi_squared(branches)


def _fishers_exact(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Fisher's exact test for a 2x2 contingency table.

    Hand-rolled to avoid scipy dependency in the hot path.
    Uses the hypergeometric distribution.
    """
    # Table: [[a_success, a_fail], [b_success, b_fail]]
    a_s, a_f = a
    b_s, b_f = b

    n = a_s + a_f + b_s + b_f
    row1 = a_s + a_f
    col1 = a_s + b_s

    # P-value: sum of probabilities of tables as extreme or more extreme
    # than observed, under null hypothesis of independence.
    # Use the hypergeometric distribution.
    observed_p = _hypergeom_pmf(a_s, n, col1, row1)
    if observed_p is None:
        return 1.0

    p_value = 0.0
    for x in range(max(0, row1 + col1 - n), min(row1, col1) + 1):
        px = _hypergeom_pmf(x, n, col1, row1)
        if px is not None and px <= observed_p + 1e-10:
            p_value += px

    return min(p_value, 1.0)


def _hypergeom_pmf(k: int, N: int, K: int, n: int) -> float | None:
    """Probability mass function of hypergeometric distribution.

    P(X=k) where X ~ Hypergeometric(N, K, n)
    """
    if k < max(0, n + K - N) or k > min(n, K):
        return 0.0

    # Use log-space to avoid overflow
    try:
        log_p = (
            _log_comb(K, k)
            + _log_comb(N - K, n - k)
            - _log_comb(N, n)
        )
        return math.exp(log_p)
    except (ValueError, OverflowError):
        return None


def _log_comb(n: int, k: int) -> float:
    """Log of binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return float("-inf")
    if k == 0 or k == n:
        return 0.0
    # Use lgamma: log(C(n,k)) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _chi_squared(branches: list[tuple[int, int]]) -> float:
    """Chi-squared test for independence with >2 branches.

    Returns approximate p-value using chi-squared distribution.
    """
    total_s = sum(s for s, f in branches)
    total_f = sum(f for s, f in branches)
    total = total_s + total_f

    if total == 0:
        return 1.0

    chi2 = 0.0
    for s, f in branches:
        n_branch = s + f
        if n_branch == 0:
            continue
        expected_s = n_branch * total_s / total
        expected_f = n_branch * total_f / total
        if expected_s > 0:
            chi2 += (s - expected_s) ** 2 / expected_s
        if expected_f > 0:
            chi2 += (f - expected_f) ** 2 / expected_f

    # Degrees of freedom = (rows - 1) * (cols - 1) = (n_branches - 1) * 1
    df = len(branches) - 1
    if df <= 0:
        return 1.0

    # Approximate p-value using regularized incomplete gamma function
    # For small df, use the survival function approximation
    return _chi2_sf(chi2, df)


def _chi2_sf(x: float, df: int) -> float:
    """Survival function (1 - CDF) of chi-squared distribution.

    Simple approximation using the regularized incomplete gamma function.
    """
    if x <= 0:
        return 1.0

    # Use Wilson-Hilferty approximation for chi-squared p-value
    # Good enough for our filtering purposes
    z = ((x / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))

    # Standard normal survival function approximation
    return 0.5 * math.erfc(z / math.sqrt(2))


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
