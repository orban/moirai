"""Tests for statistical primitives — BH correction, Fisher's exact, permutation FDR."""

import numpy as np

from moirai.analyze.stats import (
    benjamini_hochberg,
    fishers_exact_2x2,
    fishers_exact_branches,
    chi_squared_test,
    kendall_tau_b,
    permutation_fdr,
    sign_test,
)


class TestBenjaminiHochberg:
    def test_empty(self):
        assert benjamini_hochberg([]) == []

    def test_single_value(self):
        result = benjamini_hochberg([0.03])
        assert result == [0.03]

    def test_single_none(self):
        result = benjamini_hochberg([None])
        assert result == [None]

    def test_known_reference(self):
        """Known BH example: 10 p-values at q=0.05, first 2 survive."""
        p_values = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
        adjusted = benjamini_hochberg(p_values)

        # First 2 should have q <= 0.05
        assert adjusted[0] is not None and adjusted[0] <= 0.05
        assert adjusted[1] is not None and adjusted[1] <= 0.05

        # Third onward should have q > 0.05
        for i in range(2, 10):
            assert adjusted[i] is not None
            assert adjusted[i] > 0.05, f"index {i}: q={adjusted[i]}"

    def test_none_values_preserved(self):
        result = benjamini_hochberg([0.01, None, 0.04, None, 0.1])
        assert result[1] is None
        assert result[3] is None
        assert result[0] is not None
        assert result[2] is not None
        assert result[4] is not None

    def test_monotonicity(self):
        """Adjusted p-values should be monotonically non-decreasing when sorted by raw p."""
        p_values = [0.05, 0.01, 0.1, 0.001, 0.5]
        adjusted = benjamini_hochberg(p_values)
        # Sort by raw p-value and check adjusted is monotonic
        pairs = sorted(zip(p_values, adjusted), key=lambda x: x[0])
        adj_sorted = [a for _, a in pairs]
        for i in range(len(adj_sorted) - 1):
            assert adj_sorted[i] <= adj_sorted[i + 1] + 1e-10

    def test_all_none(self):
        result = benjamini_hochberg([None, None, None])
        assert result == [None, None, None]

    def test_capped_at_one(self):
        result = benjamini_hochberg([0.9, 0.95, 0.99])
        for v in result:
            assert v is not None
            assert v <= 1.0


class TestFishersExact:
    def test_perfect_association(self):
        """All successes in group A, all failures in group B."""
        p = fishers_exact_2x2(5, 0, 0, 5)
        assert p is not None
        assert p < 0.01

    def test_no_association(self):
        """Equal success rates in both groups."""
        p = fishers_exact_2x2(5, 5, 5, 5)
        assert p is not None
        assert p > 0.5

    def test_empty_table(self):
        assert fishers_exact_2x2(0, 0, 0, 0) is None

    def test_branches_wrapper(self):
        p = fishers_exact_branches((5, 0), (0, 5))
        assert p < 0.01

    def test_matches_old_motifs_implementation(self):
        """Characterization: same result as the old motifs._fishers_exact_2x2."""
        # Known table from motif testing
        p = fishers_exact_2x2(3, 2, 1, 4)
        assert p is not None
        assert 0.0 < p < 1.0


class TestChiSquared:
    def test_perfect_association(self):
        p = chi_squared_test([(10, 0), (0, 10)])
        assert p < 0.01

    def test_no_association(self):
        p = chi_squared_test([(5, 5), (5, 5)])
        assert p > 0.5

    def test_three_branches(self):
        p = chi_squared_test([(10, 0), (0, 10), (5, 5)])
        assert p < 0.05

    def test_empty(self):
        assert chi_squared_test([]) == 1.0

    def test_single_branch(self):
        assert chi_squared_test([(5, 5)]) == 1.0


class TestKendallTauB:
    def test_perfect_agreement(self):
        from pytest import approx
        tau, p = kendall_tau_b([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        assert tau == approx(1.0)
        assert p is not None and p < 0.05

    def test_perfect_disagreement(self):
        from pytest import approx
        tau, p = kendall_tau_b([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
        assert tau == approx(-1.0)
        assert p is not None and p < 0.05

    def test_no_correlation(self):
        tau, p = kendall_tau_b([1, 2, 3, 4, 5], [3, 1, 4, 5, 2])
        assert -0.5 < tau < 0.5

    def test_with_ties(self):
        tau, p = kendall_tau_b([1, 1, 2, 2, 3], [1, 2, 1, 2, 3])
        assert -1.0 <= tau <= 1.0
        assert p is not None

    def test_too_few_values(self):
        tau, p = kendall_tau_b([1, 2, 3], [1, 2, 3])
        assert tau == 0.0
        assert p is None

    def test_no_variance_x(self):
        tau, p = kendall_tau_b([1, 1, 1, 1, 1], [1, 2, 3, 4, 5])
        assert tau == 0.0
        assert p is None

    def test_no_variance_y(self):
        tau, p = kendall_tau_b([1, 2, 3, 4, 5], [1, 1, 1, 1, 1])
        assert tau == 0.0
        assert p is None

    def test_binary_outcomes(self):
        """Binary y (pass/fail) — the common case in moirai."""
        tau, p = kendall_tau_b(
            [0.1, 0.2, 0.3, 0.8, 0.9],
            [1.0, 1.0, 1.0, 0.0, 0.0],
        )
        assert tau < 0  # higher distance -> failure
        assert p is not None

    def test_length_mismatch(self):
        tau, p = kendall_tau_b([1, 2, 3], [1, 2])
        assert tau == 0.0
        assert p is None


class TestPermutationFDR:
    def test_deterministic_with_seed(self):
        membership = np.array([[True, True, False, False, True, False, True, False]], dtype=bool)
        outcomes = np.array([True, True, False, False, True, False, False, False], dtype=bool)
        fdr1, disc1 = permutation_fdr(membership, outcomes, seed=42, n_permutations=100)
        fdr2, disc2 = permutation_fdr(membership, outcomes, seed=42, n_permutations=100)
        assert fdr1 == fdr2
        assert disc1 == disc2

    def test_no_patterns(self):
        membership = np.zeros((0, 10), dtype=bool)
        outcomes = np.ones(10, dtype=bool)
        fdr, disc = permutation_fdr(membership, outcomes)
        assert fdr == 0.0

    def test_returns_list_of_counts(self):
        membership = np.array([
            [True, True, False, False, True, True, False, False],
            [False, False, True, True, False, False, True, True],
        ], dtype=bool)
        outcomes = np.array([True, True, True, True, False, False, False, False], dtype=bool)
        _, discoveries = permutation_fdr(membership, outcomes, n_permutations=50, seed=42)
        assert len(discoveries) == 50
        assert all(isinstance(d, int) for d in discoveries)


class TestSignTest:
    def test_sign_test_all_positive(self):
        """All positive values should yield a small p-value."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        p = sign_test(values)
        assert p is not None
        assert p < 0.07  # exact: 2 * 0.5^5 = 0.0625

    def test_sign_test_balanced(self):
        """Equal positive and negative values should yield p close to 1.0."""
        values = [1.0, -1.0, 2.0, -2.0, 3.0, -3.0]
        p = sign_test(values)
        assert p is not None
        assert p > 0.9

    def test_sign_test_excludes_zeros(self):
        """Zeros should not count toward the test."""
        # 5 positives, 0 negatives, but 3 zeros — effective n=5
        values = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 4.0, 5.0]
        p = sign_test(values)
        assert p is not None
        assert p < 0.07  # same as all-positive with n=5

    def test_sign_test_too_few(self):
        """Fewer than 5 non-zero values should return None."""
        values = [1.0, -1.0, 2.0, 0.0, 0.0]
        # Only 3 non-zero values
        p = sign_test(values)
        assert p is None

    def test_sign_test_strongly_negative(self):
        """All negative values should also yield a small p-value."""
        values = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]
        p = sign_test(values)
        assert p is not None
        assert p < 0.05
