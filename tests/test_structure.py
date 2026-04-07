"""Tests for per-task structure score."""

from moirai.schema import Step, Result, Run
from moirai.analyze.structure import (
    compute_structure_score,
    compute_all_structure_scores,
)


def _make_run(run_id, task_id, names, success):
    steps = [Step(idx=i, type="tool", name=name, attrs={}) for i, name in enumerate(names)]
    return Run(run_id=run_id, task_id=task_id, steps=steps, result=Result(success=success))


def _make_task_runs(task_id, n_pass, n_fail, pass_names, fail_names):
    runs = []
    for i in range(n_pass):
        runs.append(_make_run(f"{task_id}_p{i}", task_id, pass_names, True))
    for i in range(n_fail):
        runs.append(_make_run(f"{task_id}_f{i}", task_id, fail_names, False))
    return runs


class TestStructureScore:
    def test_high_structure_task(self):
        """Perfectly separable pass/fail patterns should score high."""
        runs = _make_task_runs("t1", 5, 5,
            ["edit", "edit", "test(pass)", "edit", "test(pass)", "edit", "edit"],
            ["search", "search", "test(fail)", "search", "test(fail)", "search", "search"])
        score = compute_structure_score(runs, "t1", n_resamples=10)
        assert score.branch_gap > 0.5
        assert score.composite > 0.3

    def test_identical_runs_low_structure(self):
        """Runs with identical step sequences should score low on branch_gap."""
        names = ["read", "edit", "test(pass)", "read", "edit", "read", "edit"]
        runs = _make_task_runs("t1", 4, 4, names, names)
        score = compute_structure_score(runs, "t1", n_resamples=5)
        # Same steps for pass and fail → no divergence points → low score
        assert score.branch_gap == 0.0

    def test_too_few_runs(self):
        """Tasks with very few runs should still return a score (with low stability)."""
        runs = _make_task_runs("t1", 2, 2,
            ["read", "edit", "test(pass)"],
            ["search", "edit", "test(fail)"])
        score = compute_structure_score(runs, "t1", n_resamples=5)
        assert 0.0 <= score.composite <= 1.0
        assert score.stability == 0.0  # < 5 runs

    def test_score_components_in_range(self):
        """All components should be in [0, 1]."""
        runs = _make_task_runs("t1", 5, 5,
            ["read", "edit", "test(pass)", "edit", "test(pass)", "read", "edit"],
            ["read", "search", "test(fail)", "search", "test(fail)", "read", "search"])
        score = compute_structure_score(runs, "t1", n_resamples=10)
        assert 0.0 <= score.branch_gap <= 1.0
        assert 0.0 <= score.earlyness <= 1.0
        assert 0.0 <= score.stability <= 1.0
        assert 0.0 <= score.composite <= 1.0

    def test_deterministic_with_same_seed(self):
        """Same seed should give same result."""
        runs = _make_task_runs("t1", 5, 5,
            ["edit", "edit", "test(pass)", "edit", "test(pass)", "edit", "edit"],
            ["search", "search", "test(fail)", "search", "test(fail)", "search", "search"])
        s1 = compute_structure_score(runs, "t1", seed=42, n_resamples=10)
        s2 = compute_structure_score(runs, "t1", seed=42, n_resamples=10)
        assert s1.composite == s2.composite
        assert s1.stability == s2.stability


class TestComputeAllStructureScores:
    def test_filters_non_mixed(self):
        """All-pass or all-fail tasks should be excluded."""
        task_runs = {
            "all_pass": [_make_run(f"p{i}", "all_pass", ["read", "edit"], True) for i in range(6)],
            "mixed": _make_task_runs("mixed", 4, 4,
                ["read", "edit", "test(pass)", "edit"],
                ["read", "search", "test(fail)", "search"]),
        }
        scores = compute_all_structure_scores(task_runs, min_runs=4, n_resamples=5, n_workers=1)
        assert len(scores) == 1
        assert scores[0].task_id == "mixed"

    def test_sorted_by_composite_desc(self):
        """Results should be sorted by composite score descending."""
        task_runs = {}
        for t in range(5):
            task_runs[f"task_{t}"] = _make_task_runs(f"task_{t}", 4, 4,
                ["read", "edit", "test(pass)", "edit"],
                ["read", "search", "test(fail)", "search"])
        scores = compute_all_structure_scores(task_runs, min_runs=4, n_resamples=5, n_workers=1)
        composites = [s.composite for s in scores]
        assert composites == sorted(composites, reverse=True)
