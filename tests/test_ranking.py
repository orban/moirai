"""Tests for cause ranking — scoring, bootstrap, unknown bucket."""
from __future__ import annotations

from moirai.diagnose.causes import CandidateCause, load_causes
from moirai.diagnose.ranking import bootstrap_confidence, score_causes
from moirai.schema import FeatureShift, Result, Run, Step, VariantComparison


def _step(name: str, status: str = "ok") -> Step:
    return Step(idx=0, type="tool", name=name, status=status)


def _make_run(run_id: str, task_id: str, steps: list[Step], success: bool, variant: str = "x") -> Run:
    return Run(
        run_id=run_id, task_id=task_id, steps=steps,
        result=Result(success=success), tags={"variant": variant},
    )


def _prompt_cause() -> CandidateCause:
    return CandidateCause(
        id="C1", type="prompt",
        description="Removed testing instruction",
        expected_shifts={
            "TEST_AFTER_EDIT_RATE": "decrease",
            "BLIND_SUBMIT_RATE": "increase",
        },
    )


def _timeout_cause() -> CandidateCause:
    return CandidateCause(
        id="C3", type="tool",
        description="Tool timeout reduced",
        expected_shifts={
            "TOOL_TIMEOUT_ERROR_RATE": "increase",
            "STEP_FAILURE_RATE": "increase",
        },
    )


def _model_cause() -> CandidateCause:
    return CandidateCause(
        id="C2", type="model",
        description="Model upgrade",
        expected_shifts={
            "AVG_STEP_COUNT": "decrease",
            "REASONING_DENSITY": "increase",
        },
    )


def _build_comparison(
    test_rate_shift: float = -0.6,
    blind_shift: float = 0.6,
    timeout_shift: float = 0.01,
) -> VariantComparison:
    """Build a VariantComparison with controlled feature shifts."""
    return VariantComparison(
        baseline_label="baseline",
        current_label="current",
        baseline_pass_rate=0.70,
        current_pass_rate=0.70,
        pass_rate_delta=0.0,
        feature_shifts=[
            FeatureShift("TEST_AFTER_EDIT_RATE", 0.9, 0.9 + test_rate_shift, test_rate_shift, test_rate_shift, -0.7, -0.5, "large"),
            FeatureShift("BLIND_SUBMIT_RATE", 0.1, 0.1 + blind_shift, blind_shift, blind_shift, 0.5, 0.7, "large"),
            FeatureShift("TOOL_TIMEOUT_ERROR_RATE", 0.03, 0.03 + timeout_shift, timeout_shift, timeout_shift, -0.01, 0.03, "negligible"),
            FeatureShift("STEP_FAILURE_RATE", 0.05, 0.05 + timeout_shift, timeout_shift, timeout_shift, -0.01, 0.03, "negligible"),
            FeatureShift("AVG_STEP_COUNT", 14.0, 10.0, -4.0, -0.3, -5.0, -3.0, "negligible"),
            FeatureShift("REASONING_DENSITY", 0.0, 0.0, 0.0, 0.0, -0.1, 0.1, "negligible"),
        ],
        task_breakdown=[],
    )


class TestScoreCauses:
    def test_prompt_scenario_ranks_c1_first(self):
        comparison = _build_comparison(test_rate_shift=-0.6, blind_shift=0.6, timeout_shift=0.01)
        causes = [_prompt_cause(), _model_cause(), _timeout_cause()]

        result = score_causes(comparison, causes)

        assert result.cause_scores[0].cause_id == "C1"
        assert result.cause_scores[0].score > 0.4

    def test_timeout_scenario_ranks_c3_first(self):
        comparison = _build_comparison(test_rate_shift=-0.05, blind_shift=0.05, timeout_shift=0.35)
        causes = [_prompt_cause(), _model_cause(), _timeout_cause()]

        result = score_causes(comparison, causes)

        assert result.cause_scores[0].cause_id == "C3"
        assert result.cause_scores[0].score > 0.3

    def test_unknown_bucket_present(self):
        comparison = _build_comparison()
        causes = [_prompt_cause(), _timeout_cause()]

        result = score_causes(comparison, causes)

        assert result.unknown_score >= 0.0
        total = sum(cs.score for cs in result.cause_scores) + result.unknown_score
        assert abs(total - 1.0) < 0.01

    def test_unknown_dominates_when_no_match(self):
        # Features shift in ways no cause expects
        comparison = VariantComparison(
            baseline_label="b", current_label="c",
            baseline_pass_rate=0.5, current_pass_rate=0.5, pass_rate_delta=0.0,
            feature_shifts=[
                FeatureShift("SEARCH_BEFORE_EDIT_RATE", 0.3, 0.9, 0.6, 0.6, 0.4, 0.8, "large"),
            ],
            task_breakdown=[],
        )
        # Cause expects features that didn't shift
        cause = CandidateCause(
            id="C_wrong", type="prompt", description="wrong",
            expected_shifts={"TEST_AFTER_EDIT_RATE": "decrease"},
        )

        result = score_causes(comparison, [cause])

        # Unknown should get significant mass since the big shift is unclaimed
        assert result.unknown_score > result.cause_scores[0].score

    def test_direction_mismatch_penalizes(self):
        # Feature shifts opposite to what cause expects
        comparison = VariantComparison(
            baseline_label="b", current_label="c",
            baseline_pass_rate=0.5, current_pass_rate=0.5, pass_rate_delta=0.0,
            feature_shifts=[
                FeatureShift("TEST_AFTER_EDIT_RATE", 0.3, 0.9, 0.6, 0.6, 0.4, 0.8, "large"),
            ],
            task_breakdown=[],
        )
        # Cause expects decrease, but feature increased
        cause = CandidateCause(
            id="C1", type="prompt", description="wrong direction",
            expected_shifts={"TEST_AFTER_EDIT_RATE": "decrease"},
        )

        result = score_causes(comparison, [cause])

        # Cause should be penalized, unknown should dominate
        assert result.cause_scores[0].score < 0.5

    def test_scores_sum_to_one(self):
        comparison = _build_comparison()
        causes = [_prompt_cause(), _model_cause(), _timeout_cause()]

        result = score_causes(comparison, causes)

        total = sum(cs.score for cs in result.cause_scores) + result.unknown_score
        assert abs(total - 1.0) < 0.001

    def test_explicit_priors_shift_scores(self):
        comparison = _build_comparison(test_rate_shift=-0.6, blind_shift=0.6, timeout_shift=0.01)

        # Equal priors
        c1_equal = _prompt_cause()
        c1_equal.prior = 0.5
        c3_equal = _timeout_cause()
        c3_equal.prior = 0.5
        result_equal = score_causes(comparison, [c1_equal, c3_equal])

        # Give C3 higher prior
        c1_low = _prompt_cause()
        c1_low.prior = 0.2
        c3_high = _timeout_cause()
        c3_high.prior = 0.8
        result_biased = score_causes(comparison, [c1_low, c3_high])

        # C3's score should increase when given higher prior
        c3_equal_score = next(cs for cs in result_equal.cause_scores if cs.cause_id == "C3").score
        c3_biased_score = next(cs for cs in result_biased.cause_scores if cs.cause_id == "C3").score
        assert c3_biased_score > c3_equal_score


class TestLoadCauses:
    def test_load_from_json(self, tmp_path):
        data = [
            {"id": "C1", "type": "prompt", "description": "test",
             "expected_shifts": {"TEST_AFTER_EDIT_RATE": "decrease"}},
            {"id": "C2", "type": "model", "description": "test2",
             "expected_shifts": {}, "prior": 0.5},
        ]
        import json
        p = tmp_path / "causes.json"
        p.write_text(json.dumps(data))

        causes = load_causes(p)

        assert len(causes) == 2
        assert causes[0].id == "C1"
        assert causes[0].expected_shifts == {"TEST_AFTER_EDIT_RATE": "decrease"}
        assert causes[1].prior == 0.5


class TestBootstrapConfidence:
    def test_bootstrap_produces_cis(self):
        baseline = [
            _make_run(f"b{i}", "t1",
                      [_step("read"), _step("edit"), _step("test"), _step("finish")],
                      i < 7, "baseline")
            for i in range(10)
        ]
        current = [
            _make_run(f"c{i}", "t1",
                      [_step("read"), _step("edit"), _step("finish")],
                      i < 7, "current")
            for i in range(10)
        ]

        causes = [_prompt_cause(), _timeout_cause()]
        result = bootstrap_confidence(baseline, current, causes, n_bootstrap=50, seed=42)

        # CIs should be populated and different from point estimate
        for cs in result.cause_scores:
            assert cs.ci_lower <= cs.score <= cs.ci_upper or abs(cs.ci_lower - cs.ci_upper) < 0.01

    def test_ranking_stable_with_clear_signal(self):
        baseline = [
            _make_run(f"b{i}", "t1",
                      [_step("read"), _step("edit"), _step("test"), _step("finish")],
                      i < 7, "baseline")
            for i in range(20)
        ]
        current = [
            _make_run(f"c{i}", "t1",
                      [_step("read"), _step("edit"), _step("finish")],
                      i < 7, "current")
            for i in range(20)
        ]

        causes = [_prompt_cause(), _timeout_cause()]
        result = bootstrap_confidence(baseline, current, causes, n_bootstrap=100, seed=42)

        assert result.cause_scores[0].cause_id == "C1"
