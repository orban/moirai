"""Statistical primitives — Fisher's exact, chi-squared, BH correction, permutation FDR.

Consolidated from duplicated implementations in motifs.py and divergence.py.
"""
from __future__ import annotations

import math

import numpy as np


def fishers_exact_2x2(a: int, b: int, c: int, d: int) -> float | None:
    """Fisher's exact test for a 2x2 contingency table [[a,b],[c,d]].

    Returns two-sided p-value, or None if not computable.
    """
    n = a + b + c + d
    if n == 0:
        return None

    row1 = a + b
    col1 = a + c

    observed_p = _hypergeom_pmf(a, n, col1, row1)
    if observed_p is None:
        return None

    p_value = 0.0
    for x in range(max(0, row1 + col1 - n), min(row1, col1) + 1):
        px = _hypergeom_pmf(x, n, col1, row1)
        if px is not None and px <= observed_p + 1e-10:
            p_value += px

    return min(p_value, 1.0)


def fishers_exact_branches(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Fisher's exact test for two branches given as (success, failure) tuples."""
    a_s, a_f = a
    b_s, b_f = b
    result = fishers_exact_2x2(a_s, a_f, b_s, b_f)
    return result if result is not None else 1.0


def chi_squared_test(branches: list[tuple[int, int]]) -> float:
    """Chi-squared test for independence with >2 branches.

    Each branch is (successes, failures). Returns approximate p-value.
    """
    total_s = sum(s for s, _ in branches)
    total_f = sum(f for _, f in branches)
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

    df = len(branches) - 1
    if df <= 0:
        return 1.0

    return chi2_sf(chi2, df)


def chi2_sf(x: float, df: int) -> float:
    """Survival function (1 - CDF) of chi-squared distribution.

    Wilson-Hilferty approximation — accurate enough for filtering.
    """
    if x <= 0:
        return 1.0

    z = ((x / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return 0.5 * math.erfc(z / math.sqrt(2))


def benjamini_hochberg(
    p_values: list[float | None],
    q: float = 0.05,
) -> list[float | None]:
    """Benjamini-Hochberg FDR correction.

    Takes a list of raw p-values (None entries are skipped and preserved).
    Returns adjusted q-values in the same order.
    """
    if not p_values:
        return []

    # Collect non-None entries with their original indices
    indexed: list[tuple[int, float]] = []
    for i, p in enumerate(p_values):
        if p is not None:
            indexed.append((i, p))

    if not indexed:
        return list(p_values)

    m = len(indexed)

    # Sort by p-value ascending
    indexed.sort(key=lambda x: x[1])

    # Compute adjusted p-values: p_adj[i] = p[i] * m / rank
    adjusted = [0.0] * m
    for rank_idx, (orig_idx, p) in enumerate(indexed):
        rank = rank_idx + 1
        adjusted[rank_idx] = p * m / rank

    # Enforce monotonicity: working backwards, each value must be <= the next
    for i in range(m - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])

    # Cap at 1.0
    adjusted = [min(a, 1.0) for a in adjusted]

    # Map back to original positions
    result: list[float | None] = list(p_values)
    for rank_idx, (orig_idx, _) in enumerate(indexed):
        result[orig_idx] = adjusted[rank_idx]

    return result


def permutation_fdr(
    membership: np.ndarray,
    outcomes: np.ndarray,
    q_threshold: float = 0.05,
    n_permutations: int = 1000,
    seed: int = 42,
) -> tuple[float, list[int]]:
    """Estimate empirical FDR via label permutation.

    Args:
        membership: (n_patterns, n_runs) boolean matrix
        outcomes: (n_runs,) boolean array (True=success)
        q_threshold: BH correction threshold
        n_permutations: number of permutations
        seed: random seed for reproducibility

    Returns:
        (empirical_fdr, discoveries_per_permutation)

    Uses vectorized chi-squared approximation (not Fisher's exact)
    inside the permutation loop for performance.
    """
    n_patterns, n_runs = membership.shape
    if n_patterns == 0 or n_runs == 0:
        return 0.0, []

    rng = np.random.default_rng(seed)

    # Count actual discoveries
    actual_p_values = _vectorized_p_values(membership, outcomes)
    actual_q = benjamini_hochberg(actual_p_values.tolist(), q=q_threshold)
    actual_discoveries = sum(1 for qv in actual_q if qv is not None and qv <= q_threshold)

    if actual_discoveries == 0:
        return 0.0, [0] * n_permutations

    discoveries_per_perm: list[int] = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(outcomes)
        perm_p_values = _vectorized_p_values(membership, shuffled)
        perm_q = benjamini_hochberg(perm_p_values.tolist(), q=q_threshold)
        n_disc = sum(1 for qv in perm_q if qv is not None and qv <= q_threshold)
        discoveries_per_perm.append(n_disc)

    mean_null = sum(discoveries_per_perm) / len(discoveries_per_perm)
    empirical_fdr = mean_null / actual_discoveries

    return min(empirical_fdr, 1.0), discoveries_per_perm


def _vectorized_p_values(
    membership: np.ndarray,
    outcomes: np.ndarray,
) -> np.ndarray:
    """Compute chi-squared p-values for all patterns against outcomes.

    Uses vectorized operations — no per-pattern Python loop for the math.
    """
    n_patterns, n_runs = membership.shape
    total_s = outcomes.sum()
    total_f = n_runs - total_s

    if total_s == 0 or total_f == 0:
        return np.ones(n_patterns)

    # For each pattern: count successes and failures with/without pattern
    # membership is (n_patterns, n_runs), outcomes is (n_runs,)
    with_s = membership @ outcomes.astype(float)        # (n_patterns,)
    with_total = membership.sum(axis=1).astype(float)   # (n_patterns,)
    with_f = with_total - with_s

    without_s = total_s - with_s
    without_f = total_f - with_f

    # Chi-squared statistic for each 2x2 table
    total = float(n_runs)
    chi2 = np.zeros(n_patterns)

    for obs, exp_num_s, exp_num_f in [
        (with_s, with_total * total_s / total, with_total * total_f / total),
        (without_s, (total - with_total) * total_s / total, (total - with_total) * total_f / total),
    ]:
        # Success cell
        mask_s = exp_num_s > 0
        chi2[mask_s] += (obs[mask_s] - exp_num_s[mask_s]) ** 2 / exp_num_s[mask_s]
        # Failure cell
        obs_f = (with_f if obs is with_s else without_f)
        mask_f = exp_num_f > 0
        chi2[mask_f] += (obs_f[mask_f] - exp_num_f[mask_f]) ** 2 / exp_num_f[mask_f]

    # Chi-squared survival function with df=1
    # Wilson-Hilferty approximation
    df = 1
    z = ((chi2 / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    p_values = 0.5 * np.vectorize(math.erfc)(z / math.sqrt(2))

    # Clamp to [0, 1]
    p_values = np.clip(p_values, 0.0, 1.0)

    # Patterns with zero total (shouldn't happen with membership matrix) get p=1
    p_values[with_total == 0] = 1.0

    return p_values


# --- Internal helpers ---

def _hypergeom_pmf(k: int, N: int, K: int, n: int) -> float | None:
    """PMF of hypergeometric distribution: P(X=k) where X ~ Hypergeometric(N, K, n)."""
    if k < max(0, n + K - N) or k > min(n, K):
        return 0.0
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
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
