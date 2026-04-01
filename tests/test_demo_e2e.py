"""End-to-end tests for the diagnosis demo scenarios."""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from moirai.analyze.evidence import compare_variants
from moirai.diagnose.causes import load_causes
from moirai.diagnose.ranking import score_causes
from moirai.load import load_runs
from moirai.filters import apply_kv_filters


# Import the generator directly
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "diagnosis_demo"))
from generate import _generate_prompt_regression, _generate_timeout_regression


@pytest.fixture
def causes_path():
    return Path(__file__).parent.parent / "examples" / "diagnosis_demo" / "causes.json"


class TestPromptRegressionScenario:
    """The Invisible Regression: same pass rate, different strategy."""

    def test_c1_ranks_first(self, tmp_path, causes_path):
        rng = random.Random(42)
        _generate_prompt_regression(rng, tmp_path)

        runs, _ = load_runs(tmp_path)
        baseline = apply_kv_filters(runs, ["variant=baseline"])
        current = apply_kv_filters(runs, ["variant=current"])

        comparison = compare_variants(baseline, current)
        causes = load_causes(causes_path)
        result = score_causes(comparison, causes)

        assert result.cause_scores[0].cause_id == "C1"

    def test_test_after_edit_rate_drops(self, tmp_path):
        rng = random.Random(42)
        _generate_prompt_regression(rng, tmp_path)

        runs, _ = load_runs(tmp_path)
        baseline = apply_kv_filters(runs, ["variant=baseline"])
        current = apply_kv_filters(runs, ["variant=current"])

        comparison = compare_variants(baseline, current)

        tae = next(fs for fs in comparison.feature_shifts if fs.feature == "TEST_AFTER_EDIT_RATE")
        assert tae.shift < -0.3  # large drop

    def test_stable_across_seeds(self, tmp_path, causes_path):
        causes = load_causes(causes_path)
        for seed in [42, 123, 456]:
            seed_dir = tmp_path / f"seed_{seed}"
            seed_dir.mkdir()
            rng = random.Random(seed)
            _generate_prompt_regression(rng, seed_dir)

            runs, _ = load_runs(seed_dir)
            baseline = apply_kv_filters(runs, ["variant=baseline"])
            current = apply_kv_filters(runs, ["variant=current"])

            comparison = compare_variants(baseline, current)
            result = score_causes(comparison, causes)

            assert result.cause_scores[0].cause_id == "C1", f"seed {seed}: got {result.cause_scores[0].cause_id}"


class TestTimeoutRegressionScenario:
    """Negative control: tool timeout is the actual cause."""

    def test_c3_ranks_first(self, tmp_path, causes_path):
        rng = random.Random(42)
        _generate_timeout_regression(rng, tmp_path)

        runs, _ = load_runs(tmp_path)
        baseline = apply_kv_filters(runs, ["variant=baseline"])
        current = apply_kv_filters(runs, ["variant=current"])

        comparison = compare_variants(baseline, current)
        causes = load_causes(causes_path)
        result = score_causes(comparison, causes)

        assert result.cause_scores[0].cause_id == "C3"

    def test_timeout_error_rate_spikes(self, tmp_path):
        rng = random.Random(42)
        _generate_timeout_regression(rng, tmp_path)

        runs, _ = load_runs(tmp_path)
        baseline = apply_kv_filters(runs, ["variant=baseline"])
        current = apply_kv_filters(runs, ["variant=current"])

        comparison = compare_variants(baseline, current)

        ter = next(fs for fs in comparison.feature_shifts if fs.feature == "TOOL_TIMEOUT_ERROR_RATE")
        assert ter.shift > 0.2  # significant spike


class TestUnknownCauseDominates:
    """When no candidate cause matches, unknown bucket should be large."""

    def test_unknown_wins_with_wrong_causes(self, tmp_path):
        rng = random.Random(42)
        _generate_prompt_regression(rng, tmp_path)

        runs, _ = load_runs(tmp_path)
        baseline = apply_kv_filters(runs, ["variant=baseline"])
        current = apply_kv_filters(runs, ["variant=current"])

        comparison = compare_variants(baseline, current)

        # Causes that don't match the actual evidence
        from moirai.diagnose.causes import CandidateCause
        wrong_causes = [
            CandidateCause(
                id="Cwrong", type="runtime",
                description="Unrelated runtime change",
                expected_shifts={"REASONING_DENSITY": "increase"},
            ),
        ]

        result = score_causes(comparison, wrong_causes)

        # Unknown should have significant mass since the big shifts are unclaimed
        assert result.unknown_score > 0.3
