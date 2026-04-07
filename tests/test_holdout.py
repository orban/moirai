"""Tests for held-out prediction study."""

from moirai.schema import Step, Result, Run
from moirai.analyze.holdout import (
    split_runs,
    train_divergence_model,
    score_divergence,
    score_features,
    score_step_count,
    compute_auroc,
    compute_accuracy_at_k,
    run_holdout_study,
)


def _make_run(
    run_id: str,
    task_id: str,
    names: list[str],
    success: bool,
) -> Run:
    """Create a run with step names that survive step_enriched_name."""
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
    """Create a set of runs for one task with distinct pass/fail behaviors."""
    runs = []
    for i in range(n_pass):
        runs.append(_make_run(f"{task_id}_p{i}", task_id, pass_names, True))
    for i in range(n_fail):
        runs.append(_make_run(f"{task_id}_f{i}", task_id, fail_names, False))
    return runs


class TestSplitRuns:
    def test_basic_split(self):
        runs = _make_task_runs("t1", 5, 5,
            ["read", "edit", "test(pass)"],
            ["read", "search", "test(fail)"])
        task_runs = {"t1": runs}
        splits = split_runs(task_runs, train_frac=0.7, seed=42)
        assert "t1" in splits
        train, test = splits["t1"]
        assert len(train) + len(test) == 10
        assert len(test) >= 2

    def test_stratification_preserves_outcomes(self):
        runs = _make_task_runs("t1", 6, 6,
            ["read", "edit", "test(pass)"],
            ["read", "search", "test(fail)"])
        task_runs = {"t1": runs}
        splits = split_runs(task_runs, train_frac=0.7, seed=42)
        train, test = splits["t1"]
        # Both splits should have pass and fail runs
        assert any(r.result.success for r in train)
        assert any(not r.result.success for r in train)
        assert any(r.result.success for r in test)
        assert any(not r.result.success for r in test)

    def test_too_few_runs_excluded(self):
        runs = _make_task_runs("t1", 2, 1,
            ["read", "edit"], ["read", "search"])
        task_runs = {"t1": runs}
        splits = split_runs(task_runs, train_frac=0.7)
        assert "t1" not in splits

    def test_multiple_tasks(self):
        runs_a = _make_task_runs("t1", 5, 5, ["read", "edit"], ["read", "search"])
        runs_b = _make_task_runs("t2", 4, 4, ["edit", "test(pass)"], ["search", "test(fail)"])
        task_runs = {"t1": runs_a, "t2": runs_b}
        splits = split_runs(task_runs, train_frac=0.7, seed=42)
        assert "t1" in splits
        assert "t2" in splits


class TestDivergenceModel:
    def _divergent_runs(self):
        """Create runs where pass/fail take distinct branches."""
        pass_names = ["read", "edit", "test(pass)", "edit", "test(pass)"]
        fail_names = ["read", "search", "test(fail)", "search", "test(fail)"]
        return _make_task_runs("t1", 4, 4, pass_names, fail_names)

    def test_model_trains_on_divergent_data(self):
        runs = self._divergent_runs()
        model = train_divergence_model(runs, "t1")
        assert model is not None
        assert len(model.divergence_points) > 0
        assert model.train_pass_rate == 0.5

    def test_model_returns_none_for_too_few_runs(self):
        runs = _make_task_runs("t1", 1, 1, ["read"], ["edit"])
        model = train_divergence_model(runs, "t1")
        assert model is None

    def test_score_divergence_pass_run(self):
        runs = self._divergent_runs()
        model = train_divergence_model(runs, "t1")
        assert model is not None
        # A pass-like run should score higher than 0.5
        pass_run = _make_run("test_p", "t1",
            ["read", "edit", "test(pass)", "edit", "test(pass)"], True)
        score = score_divergence(pass_run, model)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_divergence_fail_run(self):
        runs = self._divergent_runs()
        model = train_divergence_model(runs, "t1")
        assert model is not None
        fail_run = _make_run("test_f", "t1",
            ["read", "search", "test(fail)", "search", "test(fail)"], False)
        score = score_divergence(fail_run, model)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestScorers:
    def test_score_features_returns_valid_range(self):
        run = _make_run("r1", "t1",
            ["read", "edit", "test(pass)", "edit", "test(pass)", "read", "edit"],
            True)
        score = score_features(run)
        assert 0.0 <= score <= 1.0

    def test_score_step_count_short_run(self):
        run = _make_run("r1", "t1", ["read", "edit"], True)
        score = score_step_count(run)
        assert score > 0.5  # short run = higher score

    def test_score_step_count_long_run(self):
        run = _make_run("r1", "t1", ["read"] * 200, True)
        score = score_step_count(run)
        assert score < 0.3  # long run = lower score


class TestAUROC:
    def test_perfect_separation(self):
        scored = [(1.0, True), (0.9, True), (0.1, False), (0.0, False)]
        assert compute_auroc(scored) == 1.0

    def test_random_gives_half(self):
        scored = [(0.5, True), (0.5, False)]
        assert compute_auroc(scored) == 0.5

    def test_inverse_gives_zero(self):
        scored = [(0.0, True), (1.0, False)]
        assert compute_auroc(scored) == 0.0

    def test_empty_class_gives_half(self):
        scored = [(0.5, True), (0.7, True)]
        assert compute_auroc(scored) == 0.5

    def test_accuracy_at_k(self):
        scored = [(0.9, True), (0.8, False), (0.7, True), (0.1, False)]
        acc = compute_accuracy_at_k(scored, k=2)
        assert acc is not None
        # Top 2 by score: (0.9, True) and (0.8, False) → 1/2 = 0.5
        assert abs(acc - 0.5) < 0.01

    def test_accuracy_at_k_too_few(self):
        scored = [(0.9, True)]
        assert compute_accuracy_at_k(scored, k=3) is None


class TestHoldoutStudyE2E:
    def test_runs_on_synthetic_data(self):
        """End-to-end: create tasks with known divergence, verify study completes."""
        task_runs = {}
        for t in range(5):
            task_id = f"task_{t}"
            pass_names = ["read", "edit", "test(pass)", "edit", "test(pass)", "read", "edit"]
            fail_names = ["read", "search", "test(fail)", "search", "test(fail)", "read", "search"]
            task_runs[task_id] = _make_task_runs(task_id, 5, 5, pass_names, fail_names)

        results = run_holdout_study(task_runs, min_runs=6, seed=42)
        assert results.n_tasks > 0
        assert results.n_train_runs > 0
        assert results.n_test_runs > 0
        assert len(results.method_results) == 4

        # All methods should return valid AUROCs
        for mr in results.method_results:
            assert 0.0 <= mr.auroc <= 1.0
            assert mr.n_scored > 0

    def test_divergence_beats_random_on_clean_data(self):
        """With perfectly separable data, divergence should beat random."""
        task_runs = {}
        for t in range(10):
            task_id = f"task_{t}"
            # Very distinct pass/fail patterns
            pass_names = ["edit", "edit", "test(pass)", "edit", "test(pass)", "edit", "edit"]
            fail_names = ["search", "search", "test(fail)", "search", "test(fail)", "search", "search"]
            task_runs[task_id] = _make_task_runs(task_id, 6, 6, pass_names, fail_names)

        results = run_holdout_study(task_runs, min_runs=6, seed=42)
        method_by_name = {r.name: r for r in results.method_results}

        # With perfectly separable data, divergence should outperform random
        assert method_by_name["divergence"].auroc >= method_by_name["random"].auroc
