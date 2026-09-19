"""Tests for reranking experiment."""

from moirai.schema import Step, Result, Run
from moirai.analyze.rerank import (
    rerank_experiment,
    _wilson_ci,
)


def _make_run(
    run_id: str,
    task_id: str,
    names: list[str],
    success: bool,
) -> Run:
    steps = []
    for i, name in enumerate(names):
        steps.append(Step(idx=i, type="tool", name=name, attrs={}))
    return Run(run_id=run_id, task_id=task_id, steps=steps, result=Result(success=success))


def _make_task_runs(
    task_id: str,
    n_pass: int,
    n_fail: int,
    pass_names: list[str],
    fail_names: list[str],
) -> list[Run]:
    runs = []
    for i in range(n_pass):
        runs.append(_make_run(f"{task_id}_p{i}", task_id, pass_names, True))
    for i in range(n_fail):
        runs.append(_make_run(f"{task_id}_f{i}", task_id, fail_names, False))
    return runs


class TestWilsonCI:
    def test_all_ones(self):
        lo, hi = _wilson_ci([1.0, 1.0, 1.0])
        assert lo > 0.3  # Wilson doesn't give exactly 1.0 for small n
        assert hi <= 1.0

    def test_ci_contains_proportion(self):
        values = [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
        lo, hi = _wilson_ci(values)
        p = sum(values) / len(values)
        assert lo <= p <= hi

    def test_empty(self):
        lo, hi = _wilson_ci([])
        assert lo == 0.0
        assert hi == 0.0


class TestRerankExperiment:
    def test_runs_on_synthetic_data(self):
        task_runs = {}
        for t in range(5):
            task_id = f"task_{t}"
            pass_names = ["read", "edit", "test(pass)", "edit", "test(pass)", "read", "edit"]
            fail_names = ["read", "search", "test(fail)", "search", "test(fail)", "read", "search"]
            task_runs[task_id] = _make_task_runs(task_id, 4, 4, pass_names, fail_names)

        results = rerank_experiment(task_runs, k=3, n_samples=100, seed=42)
        assert results.n_tasks > 0
        assert results.k == 3
        assert len(results.method_results) > 0

        # All accuracies in valid range
        for mr in results.method_results:
            assert 0.0 <= mr.selection_accuracy <= 1.0

    def test_oracle_beats_random(self):
        task_runs = {}
        for t in range(10):
            task_id = f"task_{t}"
            pass_names = ["edit", "edit", "test(pass)", "edit", "test(pass)", "edit", "edit"]
            fail_names = ["search", "search", "test(fail)", "search", "test(fail)", "search", "search"]
            task_runs[task_id] = _make_task_runs(task_id, 4, 4, pass_names, fail_names)

        results = rerank_experiment(task_runs, k=3, n_samples=200, seed=42)
        assert results.oracle_best_of_k >= results.random_best_of_k

    def test_pass_at_1_is_reasonable(self):
        """50/50 pass/fail data should give ~50% pass@1."""
        task_runs = {}
        for t in range(20):
            task_id = f"task_{t}"
            task_runs[task_id] = _make_task_runs(
                task_id, 5, 5,
                ["read", "edit", "test(pass)", "edit"],
                ["read", "search", "test(fail)", "search"],
            )

        results = rerank_experiment(task_runs, k=3, n_samples=100, seed=42)
        # With 50/50 pass/fail, pass@1 should be around 0.5
        assert 0.3 <= results.random_pass_at_1 <= 0.7

    def test_k_sweep(self):
        """Higher K should give higher oracle rate."""
        task_runs = {}
        for t in range(10):
            task_id = f"task_{t}"
            task_runs[task_id] = _make_task_runs(
                task_id, 4, 4,
                ["read", "edit", "test(pass)", "edit", "test(pass)", "read", "edit"],
                ["read", "search", "test(fail)", "search", "test(fail)", "read", "search"],
            )

        r2 = rerank_experiment(task_runs, k=2, n_samples=100, seed=42)
        r3 = rerank_experiment(task_runs, k=3, n_samples=100, seed=42)
        # Oracle with K=3 should be >= oracle with K=2
        assert r3.oracle_best_of_k >= r2.oracle_best_of_k - 0.05  # small tolerance

    def test_excludes_non_mixed_tasks(self):
        """Tasks that are all-pass or all-fail should be excluded."""
        task_runs = {
            "all_pass": [_make_run(f"p{i}", "all_pass", ["read", "edit"], True) for i in range(6)],
            "mixed": _make_task_runs("mixed", 4, 4,
                ["read", "edit", "test(pass)", "edit"],
                ["read", "search", "test(fail)", "search"]),
        }
        results = rerank_experiment(task_runs, k=3, n_samples=50, seed=42)
        assert results.n_tasks == 1  # only "mixed" qualifies
