"""Tests for evidence extraction — behavioral features and variant comparison."""
from __future__ import annotations

from moirai.analyze.evidence import (
    compare_variants,
    extract_behavioral_features,
)
from moirai.analyze.stats import cohens_h, effect_magnitude, proportion_delta_ci
from moirai.schema import Result, Run, Step


def _step(name: str, status: str = "ok", attrs: dict | None = None, output: dict | None = None) -> Step:
    return Step(idx=0, type="tool", name=name, status=status, attrs=attrs or {}, output=output or {})


def _make_run(
    run_id: str,
    task_id: str,
    steps: list[Step],
    success: bool,
    tags: dict | None = None,
) -> Run:
    return Run(
        run_id=run_id,
        task_id=task_id,
        steps=steps,
        result=Result(success=success),
        tags=tags or {},
    )


# --- Test behavioral feature extraction ---


class TestExtractBehavioralFeatures:
    def test_test_after_edit_rate(self):
        # Run with edit→test pattern
        r1 = _make_run("r1", "t1", [_step("edit"), _step("test")], True)
        # Run without
        r2 = _make_run("r2", "t1", [_step("edit"), _step("bash")], False)

        features = extract_behavioral_features([r1, r2])
        assert features["TEST_AFTER_EDIT_RATE"] == 0.5

    def test_test_after_edit_rate_empty(self):
        features = extract_behavioral_features([])
        assert features["TEST_AFTER_EDIT_RATE"] == 0.0

    def test_iterative_fix_rate(self):
        r1 = _make_run("r1", "t1", [_step("edit"), _step("test"), _step("edit")], True)
        r2 = _make_run("r2", "t1", [_step("edit"), _step("test")], False)
        r3 = _make_run("r3", "t1", [_step("read"), _step("edit")], False)

        features = extract_behavioral_features([r1, r2, r3])
        assert abs(features["ITERATIVE_FIX_RATE"] - 1 / 3) < 0.01

    def test_blind_submit_rate(self):
        # edit→finish with no test in between = blind submit
        r1 = _make_run("r1", "t1", [_step("edit"), _step("finish")], False)
        # edit→test→finish = not blind
        r2 = _make_run("r2", "t1", [_step("edit"), _step("test"), _step("finish")], True)

        features = extract_behavioral_features([r1, r2])
        assert features["BLIND_SUBMIT_RATE"] == 0.5

    def test_tool_timeout_error_rate(self):
        r1 = _make_run("r1", "t1", [
            _step("edit", "ok"),
            _step("test", "error"),
            _step("bash", "ok"),
        ], False)

        features = extract_behavioral_features([r1])
        assert abs(features["TOOL_TIMEOUT_ERROR_RATE"] - 1 / 3) < 0.01

    def test_step_failure_rate(self):
        r1 = _make_run("r1", "t1", [
            _step("edit", "ok"),
            _step("test", "error"),
        ], False)

        features = extract_behavioral_features([r1])
        assert features["STEP_FAILURE_RATE"] == 0.5

    def test_avg_step_count(self):
        r1 = _make_run("r1", "t1", [_step("a"), _step("b")], True)
        r2 = _make_run("r2", "t1", [_step("a"), _step("b"), _step("c"), _step("d")], True)

        features = extract_behavioral_features([r1, r2])
        assert features["AVG_STEP_COUNT"] == 3.0

    def test_search_before_edit_rate(self):
        r1 = _make_run("r1", "t1", [_step("search"), _step("edit")], True)
        r2 = _make_run("r2", "t1", [_step("edit"), _step("test")], True)

        features = extract_behavioral_features([r1, r2])
        assert features["SEARCH_BEFORE_EDIT_RATE"] == 0.5

    def test_all_features_present(self):
        r = _make_run("r1", "t1", [_step("edit")], True)
        features = extract_behavioral_features([r])
        expected_keys = {
            "TEST_AFTER_EDIT_RATE", "ITERATIVE_FIX_RATE", "BLIND_SUBMIT_RATE",
            "TOOL_TIMEOUT_ERROR_RATE", "STEP_FAILURE_RATE", "AVG_STEP_COUNT",
            "SEARCH_BEFORE_EDIT_RATE", "REASONING_DENSITY",
        }
        assert set(features.keys()) == expected_keys


# --- Test stats functions ---


class TestCohensH:
    def test_identical_proportions(self):
        assert cohens_h(0.5, 0.5) == 0.0

    def test_large_difference(self):
        h = cohens_h(0.9, 0.1)
        assert abs(h) > 1.0  # large effect

    def test_small_difference(self):
        h = cohens_h(0.51, 0.49)
        assert abs(h) < 0.1  # negligible

    def test_boundary_values(self):
        h = cohens_h(1.0, 0.0)
        assert h > 0
        h = cohens_h(0.0, 1.0)
        assert h < 0


class TestProportionDeltaCI:
    def test_identical_proportions(self):
        lo, hi = proportion_delta_ci(0.5, 100, 0.5, 100)
        assert lo < 0 < hi  # CI contains 0

    def test_different_proportions(self):
        lo, hi = proportion_delta_ci(0.3, 100, 0.8, 100)
        # delta = 0.8 - 0.3 = 0.5, CI should be positive
        assert lo > 0

    def test_zero_n(self):
        lo, hi = proportion_delta_ci(0.5, 0, 0.5, 100)
        assert lo == -1.0 and hi == 1.0


class TestEffectMagnitude:
    def test_negligible(self):
        assert effect_magnitude(0.1) == "negligible"

    def test_small(self):
        assert effect_magnitude(0.3) == "small"

    def test_medium(self):
        assert effect_magnitude(0.6) == "medium"

    def test_large(self):
        assert effect_magnitude(1.0) == "large"

    def test_negative(self):
        assert effect_magnitude(-0.9) == "large"


# --- Test variant comparison ---


class TestCompareVariants:
    def _build_cohort(self, prefix: str, task_id: str, n: int, pattern: str, success_rate: float) -> list[Run]:
        """Build a cohort of runs with a specific behavioral pattern."""
        runs = []
        for i in range(n):
            success = i < int(n * success_rate)
            if pattern == "test_verify":
                steps = [_step("read"), _step("search"), _step("edit"), _step("test"), _step("edit"), _step("test"), _step("finish")]
            elif pattern == "blind_submit":
                steps = [_step("read"), _step("edit"), _step("finish")]
            else:
                steps = [_step("edit")]
            runs.append(_make_run(f"{prefix}_{i}", task_id, steps, success, tags={"variant": prefix}))
        return runs

    def test_basic_comparison(self):
        baseline = self._build_cohort("base", "t1", 10, "test_verify", 0.7)
        current = self._build_cohort("curr", "t1", 10, "blind_submit", 0.7)

        result = compare_variants(baseline, current)

        assert result.baseline_label == "baseline"
        assert result.current_label == "current"
        assert len(result.feature_shifts) > 0
        assert len(result.task_breakdown) == 1

    def test_detects_test_after_edit_shift(self):
        baseline = self._build_cohort("base", "t1", 20, "test_verify", 0.7)
        current = self._build_cohort("curr", "t1", 20, "blind_submit", 0.7)

        result = compare_variants(baseline, current)

        # Find the TEST_AFTER_EDIT_RATE shift
        tae = next(fs for fs in result.feature_shifts if fs.feature == "TEST_AFTER_EDIT_RATE")
        assert tae.baseline_value == 1.0  # all test_verify runs have edit→test
        assert tae.current_value == 0.0  # blind_submit runs have no test after edit
        assert tae.shift == -1.0
        assert tae.magnitude == "large"

    def test_same_pass_rate_different_structure(self):
        baseline = self._build_cohort("base", "t1", 20, "test_verify", 0.7)
        current = self._build_cohort("curr", "t1", 20, "blind_submit", 0.7)

        result = compare_variants(baseline, current)

        # Pass rates should be identical
        assert result.baseline_pass_rate == result.current_pass_rate
        assert result.pass_rate_delta == 0.0

        # But structural features should show large shifts
        large_shifts = [fs for fs in result.feature_shifts if fs.magnitude == "large"]
        assert len(large_shifts) >= 1

    def test_task_breakdown(self):
        b1 = self._build_cohort("base", "easy", 5, "test_verify", 0.8)
        b2 = self._build_cohort("base", "hard", 5, "test_verify", 0.6)
        c1 = self._build_cohort("curr", "easy", 5, "blind_submit", 1.0)
        c2 = self._build_cohort("curr", "hard", 5, "blind_submit", 0.4)

        result = compare_variants(b1 + b2, c1 + c2)

        assert len(result.task_breakdown) == 2
        easy = next(tb for tb in result.task_breakdown if tb.task_id == "easy")
        hard = next(tb for tb in result.task_breakdown if tb.task_id == "hard")
        assert easy.delta > 0  # easy improved
        assert hard.delta < 0  # hard regressed

    def test_empty_cohorts_handled(self):
        r = _make_run("r1", "t1", [_step("edit")], True)
        result = compare_variants([r], [r])
        assert result.baseline_pass_rate == 1.0
        assert result.current_pass_rate == 1.0

    def test_feature_shifts_sorted_by_magnitude(self):
        baseline = self._build_cohort("base", "t1", 20, "test_verify", 0.7)
        current = self._build_cohort("curr", "t1", 20, "blind_submit", 0.7)

        result = compare_variants(baseline, current)

        shifts = [abs(fs.shift) for fs in result.feature_shifts]
        assert shifts == sorted(shifts, reverse=True)
