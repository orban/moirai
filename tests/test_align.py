"""Tests for NW alignment and distance computation."""

from moirai.schema import Step, Result, Run
from moirai.analyze.align import _nw_align, trajectory_distance, distance_matrix, GAP


def _make_run(types: list[str], run_id: str = "r1") -> Run:
    steps = [Step(idx=i, type=t, name=f"s{i}") for i, t in enumerate(types)]
    return Run(run_id=run_id, task_id="t1", steps=steps, result=Result(success=True))


class TestNWAlign:
    def test_identical(self):
        a, b = _nw_align(["llm", "tool", "judge"], ["llm", "tool", "judge"])
        assert a == ["llm", "tool", "judge"]
        assert b == ["llm", "tool", "judge"]

    def test_one_empty(self):
        a, b = _nw_align(["llm", "tool"], [])
        assert b == [GAP, GAP]
        assert a == ["llm", "tool"]

    def test_both_empty(self):
        a, b = _nw_align([], [])
        assert a == []
        assert b == []

    def test_different_lengths(self):
        a, b = _nw_align(["llm", "tool", "judge"], ["llm", "judge"])
        # Should align with a gap for the missing "tool"
        assert len(a) == len(b)
        assert GAP in b

    def test_completely_different(self):
        a, b = _nw_align(["llm", "llm"], ["tool", "tool"])
        assert len(a) == len(b)

    def test_gap_insertion(self):
        a, b = _nw_align(["llm", "tool", "llm", "judge"], ["llm", "llm", "judge"])
        assert len(a) == len(b)
        # The "tool" in seq_a should cause a gap in seq_b
        assert GAP in b


class TestTrajectoryDistance:
    def test_identical_runs(self):
        r1 = _make_run(["llm", "tool", "judge"])
        r2 = _make_run(["llm", "tool", "judge"], "r2")
        assert trajectory_distance(r1, r2) == 0.0

    def test_completely_different(self):
        r1 = _make_run(["llm", "llm", "llm"])
        r2 = _make_run(["tool", "tool", "tool"], "r2")
        assert trajectory_distance(r1, r2) == 1.0

    def test_empty_vs_nonempty(self):
        r1 = _make_run([])
        r2 = _make_run(["llm", "tool"], "r2")
        assert trajectory_distance(r1, r2) == 1.0

    def test_empty_vs_empty(self):
        r1 = _make_run([])
        r2 = _make_run([], "r2")
        assert trajectory_distance(r1, r2) == 0.0

    def test_symmetry(self):
        r1 = _make_run(["llm", "tool", "judge"])
        r2 = _make_run(["llm", "tool", "tool", "judge"], "r2")
        assert trajectory_distance(r1, r2) == trajectory_distance(r2, r1)

    def test_range_0_to_1(self):
        r1 = _make_run(["llm", "tool"])
        r2 = _make_run(["llm", "judge"], "r2")
        d = trajectory_distance(r1, r2)
        assert 0.0 <= d <= 1.0

    def test_partial_overlap(self):
        r1 = _make_run(["llm", "tool", "judge"])
        r2 = _make_run(["llm", "tool", "error"], "r2")
        d = trajectory_distance(r1, r2)
        assert 0.0 < d < 1.0


class TestDistanceMatrix:
    def test_empty(self):
        m = distance_matrix([])
        assert len(m) == 0

    def test_single_run(self):
        m = distance_matrix([_make_run(["llm"])])
        assert len(m) == 0

    def test_two_identical(self):
        r1 = _make_run(["llm", "tool"], "r1")
        r2 = _make_run(["llm", "tool"], "r2")
        m = distance_matrix([r1, r2])
        assert len(m) == 1
        assert m[0] == 0.0

    def test_three_runs(self):
        r1 = _make_run(["llm", "tool"], "r1")
        r2 = _make_run(["llm", "tool"], "r2")
        r3 = _make_run(["tool", "judge"], "r3")
        m = distance_matrix([r1, r2, r3])
        # Condensed matrix for 3 items has 3 entries
        assert len(m) == 3
        assert m[0] == 0.0  # r1 vs r2
        assert m[1] > 0.0   # r1 vs r3
        assert m[2] > 0.0   # r2 vs r3
