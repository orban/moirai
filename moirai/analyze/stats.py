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

    # Filter out zero-count branches to avoid inflating df
    active = [(s, f) for s, f in branches if s + f > 0]

    chi2 = 0.0
    for s, f in active:
        n_branch = s + f
        expected_s = n_branch * total_s / total
        expected_f = n_branch * total_f / total
        if expected_s > 0:
            chi2 += (s - expected_s) ** 2 / expected_s
        if expected_f > 0:
            chi2 += (f - expected_f) ** 2 / expected_f

    df = len(active) - 1
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
) -> list[float | None]:
    """Benjamini-Hochberg FDR correction.

    Takes a list of raw p-values (None entries are skipped and preserved).
    Returns adjusted q-values in the same order. Callers filter by their
    own q-threshold after calling this function.
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
    actual_q = benjamini_hochberg(actual_p_values.tolist())
    actual_discoveries = sum(1 for qv in actual_q if qv is not None and qv <= q_threshold)

    if actual_discoveries == 0:
        return 0.0, [0] * n_permutations

    discoveries_per_perm: list[int] = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(outcomes)
        perm_p_values = _vectorized_p_values(membership, shuffled)
        perm_q = benjamini_hochberg(perm_p_values.tolist())
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

    # Chi-squared survival function with df=1 (Wilson-Hilferty approximation)
    # Guard: chi2 <= 0 → p = 1.0 (consistent with scalar chi2_sf)
    df = 1
    z = np.where(
        chi2 > 0,
        ((chi2 / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df)),
        -np.inf,
    )
    p_values = np.where(chi2 > 0, 0.5 * np.vectorize(math.erfc)(z / math.sqrt(2)), 1.0)

    p_values = np.clip(p_values, 0.0, 1.0)
    p_values[with_total == 0] = 1.0

    return p_values


def kendall_tau_b(x: list[float], y: list[float]) -> tuple[float, float | None]:
    """Kendall's Tau-b rank correlation.

    Returns (tau, p_value). Returns (0.0, None) for degenerate inputs
    (n < 5, no variation in x or y, or scipy returns NaN).
    """
    if len(x) != len(y) or len(x) < 5:
        return 0.0, None

    # Check for zero variance
    if len(set(x)) < 2 or len(set(y)) < 2:
        return 0.0, None

    from scipy.stats import kendalltau

    result = kendalltau(x, y, variant="b")
    tau_val = float(result.statistic)
    p_raw = float(result.pvalue)

    # scipy returns NaN for some degenerate cases
    if math.isnan(tau_val):
        return 0.0, None

    p_val = p_raw if not math.isnan(p_raw) else None
    return tau_val, p_val


# --- Effect sizes and confidence intervals ---


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions.

    h = 2 × arcsin(sqrt(p1)) - 2 × arcsin(sqrt(p2))
    Thresholds: |h| < 0.2 = small, 0.2-0.8 = medium, > 0.8 = large.
    """
    p1 = max(0.0, min(1.0, p1))
    p2 = max(0.0, min(1.0, p2))
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def proportion_delta_ci(
    baseline_p: float,
    baseline_n: int,
    current_p: float,
    current_n: int,
    z: float = 1.96,
) -> tuple[float, float]:
    """CI on the difference (current - baseline) of two independent proportions.

    Wald interval. Acceptable for n >= 30; bootstrap for small samples.
    """
    if baseline_n == 0 or current_n == 0:
        return (-1.0, 1.0)
    se = math.sqrt(
        baseline_p * (1 - baseline_p) / baseline_n
        + current_p * (1 - current_p) / current_n
    )
    delta = current_p - baseline_p
    return (delta - z * se, delta + z * se)


def effect_magnitude(h: float) -> str:
    """Classify effect size magnitude from Cohen's h."""
    ah = abs(h)
    if ah < 0.2:
        return "negligible"
    if ah < 0.5:
        return "small"
    if ah < 0.8:
        return "medium"
    return "large"


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


def sign_test(values: list[float]) -> float | None:
    """Two-sided sign test on a list of values.

    Counts positives and negatives (zeros EXCLUDED).
    Returns p-value from exact binomial test under H0: P(positive) = 0.5.
    Returns None if fewer than 5 non-zero values.
    """
    n_pos = sum(1 for v in values if v > 0)
    n_neg = sum(1 for v in values if v < 0)
    n = n_pos + n_neg
    if n < 5:
        return None
    # Exact two-sided binomial: P(X >= max(n_pos, n_neg)) * 2, capped at 1.0
    k = max(n_pos, n_neg)
    p_tail = 0.0
    for x in range(k, n + 1):
        p_tail += _binom_pmf(x, n, 0.5)
    return min(2.0 * p_tail, 1.0)


def _binom_pmf(k: int, n: int, p: float) -> float:
    """Binomial probability mass function."""
    if k < 0 or k > n:
        return 0.0
    log_coeff = _log_comb(n, k)
    if log_coeff == float("-inf"):
        return 0.0
    return math.exp(log_coeff + k * math.log(p) + (n - k) * math.log(1 - p))


def _log_comb(n: int, k: int) -> float:
    """Log of binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return float("-inf")
    if k == 0 or k == n:
        return 0.0
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
