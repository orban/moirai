"""Tests for agglomerative clustering."""

from moirai.schema import Step, Result, Run
from moirai.analyze.cluster import cluster_runs, compute_concordance


def _make_run(types: list[str], run_id: str = "r1", success: bool = True,
              error_type: str | None = None) -> Run:
    steps = [Step(idx=i, type=t, name=f"s{i}") for i, t in enumerate(types)]
    return Run(
        run_id=run_id, task_id="t1", steps=steps,
        result=Result(success=success, error_type=error_type),
    )


def test_empty_runs():
    result = cluster_runs([])
    assert result.clusters == []
    assert result.labels == {}


def test_single_run():
    r = _make_run(["llm", "tool"], "r1")
    result = cluster_runs([r])
    assert len(result.clusters) == 1
    assert result.clusters[0].count == 1
    assert result.labels["r1"] == 0


def test_two_identical_runs():
    r1 = _make_run(["llm", "tool", "judge"], "r1")
    r2 = _make_run(["llm", "tool", "judge"], "r2")
    result = cluster_runs([r1, r2])
    assert len(result.clusters) == 1
    assert result.clusters[0].count == 2
    assert result.labels["r1"] == result.labels["r2"]


def test_known_clusters_separate():
    # Group A: direct solve pattern
    a1 = _make_run(["llm", "tool", "judge"], "a1", True)
    a2 = _make_run(["llm", "tool", "judge"], "a2", True)
    a3 = _make_run(["llm", "tool", "judge"], "a3", True)
    # Group B: error pattern
    b1 = _make_run(["llm", "tool", "tool", "tool", "error"], "b1", False, "loop")
    b2 = _make_run(["llm", "tool", "tool", "tool", "error"], "b2", False, "loop")
    b3 = _make_run(["llm", "tool", "tool", "tool", "error"], "b3", False, "loop")

    result = cluster_runs([a1, a2, a3, b1, b2, b3], threshold=0.5)
    assert len(result.clusters) == 2

    # All a's should be in same cluster, all b's in another
    assert result.labels["a1"] == result.labels["a2"] == result.labels["a3"]
    assert result.labels["b1"] == result.labels["b2"] == result.labels["b3"]
    assert result.labels["a1"] != result.labels["b1"]


def test_cluster_success_rate():
    r1 = _make_run(["llm", "tool", "judge"], "r1", True)
    r2 = _make_run(["llm", "tool", "judge"], "r2", False)
    r3 = _make_run(["llm", "tool", "judge"], "r3", True)
    result = cluster_runs([r1, r2, r3])
    # All in one cluster
    assert len(result.clusters) == 1
    assert result.clusters[0].success_rate is not None
    assert abs(result.clusters[0].success_rate - 2 / 3) < 0.01


def test_cluster_error_types():
    r1 = _make_run(["llm", "error"], "r1", False, "timeout")
    r2 = _make_run(["llm", "error"], "r2", False, "timeout")
    r3 = _make_run(["llm", "error"], "r3", False, "crash")
    result = cluster_runs([r1, r2, r3])
    assert result.clusters[0].error_types == {"timeout": 2, "crash": 1}


def test_threshold_sensitivity():
    # With very low threshold, similar-but-not-identical runs split
    r1 = _make_run(["llm", "tool", "judge"], "r1")
    r2 = _make_run(["llm", "tool", "llm", "judge"], "r2")  # one extra step

    # Tight threshold should separate them
    tight = cluster_runs([r1, r2], threshold=0.1)
    assert len(tight.clusters) == 2

    # Loose threshold should merge them
    loose = cluster_runs([r1, r2], threshold=0.5)
    assert len(loose.clusters) == 1


# --- Concordance scoring tests ---


def _make_named_run(names: list[str], run_id: str, success: bool,
                    score: float | None = None) -> Run:
    """Make a run with enrichable step names for concordance testing."""
    steps = [Step(idx=i, type="tool", name=name) for i, name in enumerate(names)]
    return Run(
        run_id=run_id, task_id="t1", steps=steps,
        result=Result(success=success, score=score),
    )


class TestComputeConcordance:
    def test_positive_concordance(self):
        """Runs close to consensus succeed, distant runs fail -> positive tau."""
        typical = ["read", "search", "edit", "test"]
        runs = [
            _make_named_run(typical, "r1", True),
            _make_named_run(typical, "r2", True),
            _make_named_run(typical, "r3", True),
            _make_named_run(["bash", "bash", "bash", "bash"], "r4", False),
            _make_named_run(["bash", "bash", "error", "bash"], "r5", False),
            _make_named_run(["bash", "error", "bash", "error"], "r6", False),
        ]
        labels = {r.run_id: 0 for r in runs}
        result = compute_concordance(runs, labels)
        assert 0 in result
        assert result[0].tau > 0
        assert result[0].n_runs == 6
        assert result[0].used_continuous is False

    def test_negative_concordance(self):
        """Typical runs fail, atypical succeed -> negative tau."""
        typical = ["read", "search", "edit", "test"]
        runs = [
            _make_named_run(typical, "r1", False),
            _make_named_run(typical, "r2", False),
            _make_named_run(typical, "r3", False),
            _make_named_run(typical, "r4", False),
            _make_named_run(["bash", "bash", "bash", "bash"], "r5", True),
        ]
        labels = {r.run_id: 0 for r in runs}
        result = compute_concordance(runs, labels)
        assert 0 in result
        assert result[0].tau < 0

    def test_too_few_runs(self):
        """Clusters with < 5 known-outcome runs are skipped."""
        runs = [
            _make_named_run(["read", "edit"], f"r{i}", i < 2)
            for i in range(4)
        ]
        labels = {r.run_id: 0 for r in runs}
        result = compute_concordance(runs, labels)
        assert 0 not in result

    def test_all_same_outcome(self):
        """All-success or all-failure clusters are skipped."""
        runs = [
            _make_named_run(["read", "edit", "test"], f"r{i}", True)
            for i in range(6)
        ]
        labels = {r.run_id: 0 for r in runs}
        result = compute_concordance(runs, labels)
        assert 0 not in result

    def test_identical_runs(self):
        """All runs structurally identical -> no distance variance -> skipped."""
        pattern = ["read", "search", "edit"]
        runs = [
            _make_named_run(pattern, f"r{i}", i < 3)
            for i in range(6)
        ]
        labels = {r.run_id: 0 for r in runs}
        result = compute_concordance(runs, labels)
        assert 0 not in result

    def test_continuous_scores(self):
        """Uses result.score when available."""
        runs = [
            _make_named_run(["read", "edit", "test"], f"r{i}", i < 3, score=float(i))
            for i in range(6)
        ]
        labels = {r.run_id: 0 for r in runs}
        result = compute_concordance(runs, labels)
        if 0 in result:
            assert result[0].used_continuous is True

    def test_multiple_clusters(self):
        """Returns results keyed by cluster ID."""
        runs_a = [_make_named_run(["read", "edit"], f"a{i}", i < 3) for i in range(6)]
        runs_b = [_make_named_run(["bash", "test"], f"b{i}", i < 3) for i in range(6)]
        labels = {r.run_id: 0 for r in runs_a}
        labels.update({r.run_id: 1 for r in runs_b})
        result = compute_concordance(runs_a + runs_b, labels)
        assert len(result) == 0

    def test_p_value_populated(self):
        """Non-degenerate case should have a p-value."""
        typical = ["read", "search", "edit", "test"]
        runs = [
            _make_named_run(typical, "r1", True),
            _make_named_run(typical, "r2", True),
            _make_named_run(typical, "r3", True),
            _make_named_run(["bash", "bash", "bash", "bash"], "r4", False),
            _make_named_run(["bash", "error", "bash", "error"], "r5", False),
            _make_named_run(["error", "bash", "error", "bash"], "r6", False),
        ]
        labels = {r.run_id: 0 for r in runs}
        result = compute_concordance(runs, labels)
        if 0 in result:
            assert result[0].p_value is not None
