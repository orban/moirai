"""Tests for cohort comparison."""

import pytest

from moirai.schema import Step, Result, Run
from moirai.analyze.compare import compare_cohorts
from moirai.filters import parse_kv_filter, apply_kv_filters


def _make_run(run_id: str, types: list[str], success: bool = True,
              harness: str = "h1", model: str = "m1",
              error_type: str | None = None) -> Run:
    steps = [Step(idx=i, type=t, name=f"s{i}",
                  metrics={"tokens_in": 100.0, "tokens_out": 50.0, "latency_ms": 1000.0})
             for i, t in enumerate(types)]
    return Run(
        run_id=run_id, task_id="t1", harness=harness, model=model,
        steps=steps, result=Result(success=success, error_type=error_type),
    )


class TestCompare:
    def test_identical_cohorts_zero_delta(self):
        runs = [_make_run(f"r{i}", ["llm", "tool", "judge"], True) for i in range(4)]
        diff = compare_cohorts(runs[:2], runs[2:])
        assert diff.a_summary.run_count == 2
        assert diff.b_summary.run_count == 2
        assert diff.a_summary.success_rate == diff.b_summary.success_rate

    def test_different_cohorts(self):
        a_runs = [
            _make_run("a1", ["llm", "tool", "judge"], True),
            _make_run("a2", ["llm", "tool", "judge"], False),
        ]
        b_runs = [
            _make_run("b1", ["llm", "tool", "judge"], True),
            _make_run("b2", ["llm", "tool", "judge"], True),
        ]
        diff = compare_cohorts(a_runs, b_runs)
        assert diff.a_summary.success_rate == 0.5
        assert diff.b_summary.success_rate == 1.0

    def test_empty_cohort_raises(self):
        runs = [_make_run("r1", ["llm"])]
        with pytest.raises(ValueError, match="cohort A is empty"):
            compare_cohorts([], runs)
        with pytest.raises(ValueError, match="cohort B is empty"):
            compare_cohorts(runs, [])

    def test_cluster_matching_by_prototype(self):
        # Cohort A: 2 direct_solve, 1 error_loop
        a_runs = [
            _make_run("a1", ["llm", "tool", "judge"], True),
            _make_run("a2", ["llm", "tool", "judge"], True),
            _make_run("a3", ["llm", "tool", "tool", "error"], False),
        ]
        # Cohort B: 1 direct_solve, 2 error_loop
        b_runs = [
            _make_run("b1", ["llm", "tool", "judge"], True),
            _make_run("b2", ["llm", "tool", "tool", "error"], False),
            _make_run("b3", ["llm", "tool", "tool", "error"], False),
        ]
        diff = compare_cohorts(a_runs, b_runs, threshold=0.5)
        # Should have shifts
        assert len(diff.cluster_shifts) > 0

    def test_unique_signatures(self):
        a_runs = [_make_run("a1", ["llm", "tool", "judge"])]
        b_runs = [_make_run("b1", ["llm", "error"])]
        diff = compare_cohorts(a_runs, b_runs)
        assert len(diff.a_only_signatures) > 0
        assert len(diff.b_only_signatures) > 0


class TestParseKVFilter:
    def test_valid(self):
        assert parse_kv_filter("harness=baseline") == ("harness", "baseline")

    def test_value_with_equals(self):
        assert parse_kv_filter("tag=a=b") == ("tag", "a=b")

    def test_missing_equals(self):
        with pytest.raises(ValueError, match="expected key=value"):
            parse_kv_filter("harness")

    def test_empty_key(self):
        with pytest.raises(ValueError, match="empty key"):
            parse_kv_filter("=value")


class TestApplyKVFilters:
    def test_field_filter(self):
        runs = [_make_run("r1", ["llm"], harness="a"), _make_run("r2", ["llm"], harness="b")]
        result = apply_kv_filters(runs, ["harness=a"])
        assert len(result) == 1
        assert result[0].run_id == "r1"

    def test_tag_filter(self):
        r1 = _make_run("r1", ["llm"])
        r1.tags = {"cohort": "baseline"}
        r2 = _make_run("r2", ["llm"])
        r2.tags = {"cohort": "experiment"}
        result = apply_kv_filters([r1, r2], ["cohort=baseline"])
        assert len(result) == 1

    def test_multiple_filters(self):
        runs = [
            _make_run("r1", ["llm"], harness="a", model="x"),
            _make_run("r2", ["llm"], harness="a", model="y"),
        ]
        result = apply_kv_filters(runs, ["harness=a", "model=x"])
        assert len(result) == 1
        assert result[0].run_id == "r1"
