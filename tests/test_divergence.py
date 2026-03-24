"""Tests for divergence detection."""

import math

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
        points = find_divergence_points(alignment, runs)
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
        points = find_divergence_points(alignment, runs)
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
        points = find_divergence_points(alignment, runs)
        assert len(points) == 1
        # Two equally frequent values: entropy = 1.0
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
        points = find_divergence_points(alignment, runs)
        assert len(points) == 1
        assert points[0].success_by_value["judge"] == 1.0
        assert points[0].success_by_value["tool"] == 0.0

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
        points = find_divergence_points(alignment, runs)
        assert len(points) == 1
        assert points[0].success_by_value["llm"] is None
        assert points[0].success_by_value["tool"] is None

    def test_sorted_by_entropy_descending(self):
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
        runs = [_make_run(f"r{i+1}", ["x"] * 3) for i in range(4)]
        points = find_divergence_points(alignment, runs)
        # Should be sorted by entropy descending
        for i in range(len(points) - 1):
            assert points[i].entropy >= points[i + 1].entropy

    def test_empty_alignment(self):
        alignment = Alignment(run_ids=[], matrix=[], level="type")
        points = find_divergence_points(alignment, [])
        assert points == []


class TestAlignRunsProgressive:
    def test_identical_runs_no_gaps(self):
        runs = [
            _make_run("r1", ["llm", "tool", "judge"]),
            _make_run("r2", ["llm", "tool", "judge"]),
            _make_run("r3", ["llm", "tool", "judge"]),
        ]
        alignment = align_runs(runs)
        # No gaps expected
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
        # r1 should have a gap somewhere
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
