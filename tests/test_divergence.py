"""Tests for divergence detection."""

from moirai.schema import Alignment, GAP, Step, Result, Run
from moirai.analyze.divergence import find_divergence_points
from moirai.analyze.align import align_runs


def _make_run(run_id: str, types: list[str], success: bool | None = True) -> Run:
    steps = [Step(idx=i, type=t, name=f"s{i}") for i, t in enumerate(types)]
    return Run(run_id=run_id, task_id="t1", steps=steps, result=Result(success=success))


class TestDivergencePoints:
    def test_uniform_column_not_divergence(self):
        alignment = Alignment(
            run_ids=["r1", "r2", "r3"],
            matrix=[
                ["llm", "tool"],
                ["llm", "tool"],
                ["llm", "tool"],
            ],
            level="type",
        )
        runs = [_make_run("r1", ["llm", "tool"]),
                _make_run("r2", ["llm", "tool"]),
                _make_run("r3", ["llm", "tool"])]
        points, _ = find_divergence_points(alignment, runs)
        assert points == []

    def test_mixed_column_detected(self):
        alignment = Alignment(
            run_ids=["r1", "r2", "r3"],
            matrix=[
                ["llm", "judge"],
                ["llm", "tool"],
                ["llm", "judge"],
            ],
            level="type",
        )
        runs = [_make_run("r1", ["llm", "judge"], True),
                _make_run("r2", ["llm", "tool"], False),
                _make_run("r3", ["llm", "judge"], True)]
        points, _ = find_divergence_points(alignment, runs, min_branch_size=1, q_threshold=1.0)
        assert len(points) == 1
        assert points[0].column == 1
        assert points[0].value_counts == {"judge": 2, "tool": 1}

    def test_entropy_calculation(self):
        alignment = Alignment(
            run_ids=["r1", "r2"],
            matrix=[
                ["llm"],
                ["tool"],
            ],
            level="type",
        )
        runs = [_make_run("r1", ["llm"]), _make_run("r2", ["tool"])]
        points, _ = find_divergence_points(alignment, runs, min_branch_size=1, q_threshold=1.0)
        assert len(points) == 1
        assert abs(points[0].entropy - 1.0) < 0.01

    def test_success_correlation(self):
        alignment = Alignment(
            run_ids=["r1", "r2", "r3", "r4"],
            matrix=[
                ["llm", "judge"],
                ["llm", "judge"],
                ["llm", "tool"],
                ["llm", "tool"],
            ],
            level="type",
        )
        runs = [_make_run("r1", ["llm", "judge"], True),
                _make_run("r2", ["llm", "judge"], True),
                _make_run("r3", ["llm", "tool"], False),
                _make_run("r4", ["llm", "tool"], False)]
        points, _ = find_divergence_points(alignment, runs, min_branch_size=1, q_threshold=1.0)
        assert len(points) == 1
        assert points[0].success_by_value["judge"] == 1.0
        assert points[0].success_by_value["tool"] == 0.0

    def test_significance_filtering(self):
        """Points where branch doesn't predict outcome get filtered by q_threshold."""
        alignment = Alignment(
            run_ids=["r1", "r2", "r3", "r4"],
            matrix=[
                ["llm", "judge"],
                ["llm", "judge"],
                ["llm", "tool"],
                ["llm", "tool"],
            ],
            level="type",
        )
        # Branch doesn't predict outcome: one pass one fail in each branch
        runs = [_make_run("r1", ["llm", "judge"], True),
                _make_run("r2", ["llm", "judge"], False),
                _make_run("r3", ["llm", "tool"], True),
                _make_run("r4", ["llm", "tool"], False)]
        # p-value should be 1.0 — filtered at q=0.05
        points, _ = find_divergence_points(alignment, runs, min_branch_size=1, q_threshold=0.05)
        assert len(points) == 0

    def test_min_branch_size_filtering(self):
        """Points with tiny branches get filtered."""
        alignment = Alignment(
            run_ids=["r1", "r2", "r3"],
            matrix=[
                ["llm", "judge"],
                ["llm", "tool"],
                ["llm", "judge"],
            ],
            level="type",
        )
        runs = [_make_run("r1", ["llm", "judge"], True),
                _make_run("r2", ["llm", "tool"], False),
                _make_run("r3", ["llm", "judge"], True)]
        points, _ = find_divergence_points(alignment, runs, min_branch_size=2, q_threshold=1.0)
        assert len(points) == 0

    def test_phase_context_populated(self):
        alignment = Alignment(
            run_ids=["r1", "r2", "r3", "r4"],
            matrix=[
                ["read", "edit"],
                ["read", "edit"],
                ["read", "search"],
                ["read", "search"],
            ],
            level="name",
        )
        runs = [_make_run("r1", ["read", "edit"], True),
                _make_run("r2", ["read", "edit"], True),
                _make_run("r3", ["read", "search"], False),
                _make_run("r4", ["read", "search"], False)]
        points, _ = find_divergence_points(alignment, runs, min_branch_size=1, q_threshold=1.0)
        assert len(points) == 1
        assert points[0].phase_context is not None

    def test_all_null_success_partition(self):
        alignment = Alignment(
            run_ids=["r1", "r2"],
            matrix=[
                ["llm"],
                ["tool"],
            ],
            level="type",
        )
        runs = [_make_run("r1", ["llm"], None), _make_run("r2", ["tool"], None)]
        points, _ = find_divergence_points(alignment, runs, min_branch_size=1, q_threshold=1.0)
        assert len(points) == 1
        assert points[0].success_by_value["llm"] is None
        assert points[0].success_by_value["tool"] is None

    def test_sorted_by_significance(self):
        alignment = Alignment(
            run_ids=["r1", "r2", "r3", "r4"],
            matrix=[
                ["llm", "tool", "judge"],
                ["llm", "tool", "error"],
                ["llm", "judge", "judge"],
                ["llm", "judge", "error"],
            ],
            level="type",
        )
        runs = [_make_run("r1", ["x"] * 3, True),
                _make_run("r2", ["x"] * 3, False),
                _make_run("r3", ["x"] * 3, True),
                _make_run("r4", ["x"] * 3, False)]
        points, _ = find_divergence_points(alignment, runs, min_branch_size=1, q_threshold=1.0)
        # Should be sorted by q_value ascending
        q_values = [p.q_value if p.q_value is not None else (p.p_value or 1.0) for p in points]
        for i in range(len(q_values) - 1):
            assert q_values[i] <= q_values[i + 1] + 0.01

    def test_q_value_populated(self):
        """BH correction sets q_value on divergence points."""
        alignment = Alignment(
            run_ids=["r1", "r2", "r3", "r4"],
            matrix=[
                ["llm", "judge"],
                ["llm", "judge"],
                ["llm", "tool"],
                ["llm", "tool"],
            ],
            level="type",
        )
        runs = [_make_run("r1", ["llm", "judge"], True),
                _make_run("r2", ["llm", "judge"], True),
                _make_run("r3", ["llm", "tool"], False),
                _make_run("r4", ["llm", "tool"], False)]
        points, n_tested = find_divergence_points(alignment, runs, min_branch_size=1, q_threshold=1.0)
        assert n_tested > 0
        for p in points:
            assert p.q_value is not None

    def test_empty_alignment(self):
        alignment = Alignment(run_ids=[], matrix=[], level="type")
        points, _ = find_divergence_points(alignment, [])
        assert points == []


class TestAlignRunsProgressive:
    def test_identical_runs_no_gaps(self):
        runs = [
            _make_run("r1", ["llm", "tool", "judge"]),
            _make_run("r2", ["llm", "tool", "judge"]),
            _make_run("r3", ["llm", "tool", "judge"]),
        ]
        alignment = align_runs(runs)
        for row in alignment.matrix:
            assert GAP not in row
        assert len(alignment.matrix) == 3
        assert all(len(row) == 3 for row in alignment.matrix)

    def test_single_run(self):
        runs = [_make_run("r1", ["llm", "tool"])]
        alignment = align_runs(runs)
        assert alignment.matrix == [["llm", "tool"]]

    def test_gap_insertion_for_extra_step(self):
        runs = [
            _make_run("r1", ["llm", "tool", "judge"]),
            _make_run("r2", ["llm", "tool", "llm", "judge"]),
        ]
        alignment = align_runs(runs)
        assert len(alignment.matrix) == 2
        assert len(alignment.matrix[0]) == len(alignment.matrix[1])
        assert GAP in alignment.matrix[0]

    def test_determinism(self):
        """Same runs in same order should produce same alignment."""
        runs = [
            _make_run("r1", ["llm", "tool", "judge"]),
            _make_run("r2", ["llm", "llm", "judge"]),
            _make_run("r3", ["llm", "tool", "tool", "judge"]),
        ]
        a1 = align_runs(runs)
        a2 = align_runs(runs)
        assert a1.matrix == a2.matrix
