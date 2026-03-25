"""Tests for signature compression and phase classification."""

from moirai.schema import Step, Result, Run
from moirai.compress import (
    compress_run,
    compress_phases,
    compress_steps,
    phase_summary,
    step_display_name,
    step_phase,
    cluster_subpatterns,
    _rle,
    _format_rle,
)


def _step(name: str, type: str = "tool", status: str = "ok") -> Step:
    return Step(idx=0, type=type, name=name, status=status)


def _make_run(steps: list[Step], success: bool = True, run_id: str = "r1") -> Run:
    for i, s in enumerate(steps):
        s.idx = i
    return Run(run_id=run_id, task_id="t1", steps=steps, result=Result(success=success))


class TestStepDisplayName:
    def test_noise_filtered(self):
        assert step_display_name(_step("error_observation", "system")) is None
        assert step_display_name(_step("action")) is None

    def test_test_result_annotated(self):
        assert step_display_name(_step("test_result", "judge", "ok")) == "test(pass)"
        assert step_display_name(_step("test_result", "judge", "error")) == "test(fail)"

    def test_normal_step(self):
        assert step_display_name(_step("read")) == "read"
        assert step_display_name(_step("edit")) == "edit"
        assert step_display_name(_step("search")) == "search"


class TestStepPhase:
    def test_explore(self):
        assert step_phase(_step("read")) == "explore"
        assert step_phase(_step("search")) == "explore"

    def test_modify(self):
        assert step_phase(_step("edit")) == "modify"
        assert step_phase(_step("write")) == "modify"

    def test_verify(self):
        assert step_phase(_step("test")) == "verify"
        assert step_phase(_step("test_result", "judge")) == "verify"

    def test_think(self):
        assert step_phase(_step("reason", "llm")) == "think"

    def test_fallback_to_type(self):
        assert step_phase(_step("unknown_name", "llm")) == "think"
        assert step_phase(_step("unknown_name", "error")) == "error"


class TestRLE:
    def test_empty(self):
        assert _rle([]) == []

    def test_no_repeats(self):
        assert _rle(["a", "b", "c"]) == [("a", 1), ("b", 1), ("c", 1)]

    def test_repeats(self):
        assert _rle(["a", "a", "b", "b", "b"]) == [("a", 2), ("b", 3)]

    def test_single(self):
        assert _rle(["a"]) == [("a", 1)]

    def test_format(self):
        assert _format_rle([("read", 1), ("edit", 3), ("test", 1)]) == "read → edit×3 → test"


class TestCompressRun:
    def test_basic(self):
        steps = [_step("read"), _step("edit"), _step("test_result", "judge", "ok")]
        run = _make_run(steps)
        assert compress_run(run) == "read → edit → test(pass)"

    def test_noise_filtered(self):
        steps = [
            _step("error_observation", "system"),
            _step("read"),
            _step("error_observation", "system"),
            _step("action"),
            _step("edit"),
        ]
        run = _make_run(steps)
        assert compress_run(run) == "read → edit"

    def test_rle(self):
        steps = [_step("read"), _step("read"), _step("read"), _step("edit")]
        run = _make_run(steps)
        assert compress_run(run) == "read×3 → edit"

    def test_empty_steps(self):
        run = _make_run([])
        assert compress_run(run) == "(empty)"

    def test_all_noise_falls_back_to_types(self):
        steps = [_step("error_observation", "system"), _step("action")]
        run = _make_run(steps)
        result = compress_run(run)
        assert result != "(empty)"  # should fall back to types


class TestCompressPhases:
    def test_basic(self):
        steps = [_step("read"), _step("edit"), _step("test_result", "judge", "ok")]
        run = _make_run(steps)
        result = compress_phases(run)
        assert "explore" in result
        assert "modify" in result
        assert "verify(pass)" in result

    def test_verify_fail(self):
        steps = [_step("read"), _step("test_result", "judge", "error")]
        run = _make_run(steps)
        result = compress_phases(run)
        assert "verify(fail)" in result


class TestPhaseSummary:
    def test_counts(self):
        steps = [_step("read"), _step("read"), _step("edit"), _step("test_result", "judge")]
        run = _make_run(steps)
        summary = phase_summary(run)
        assert summary["explore"] == 2
        assert summary["modify"] == 1
        assert summary["verify"] == 1

    def test_noise_excluded(self):
        steps = [_step("read"), _step("error_observation", "system"), _step("action")]
        run = _make_run(steps)
        summary = phase_summary(run)
        assert "explore" in summary
        assert sum(summary.values()) == 1  # only the read


class TestClusterSubpatterns:
    def test_fast_solve(self):
        runs = [
            _make_run([_step("read"), _step("edit")], True, "r1"),           # 2 steps
            _make_run([_step("read")] * 20 + [_step("edit")], True, "r2"),   # 21 steps
            _make_run([_step("read")] * 20 + [_step("edit")], False, "r3"),  # 21 steps
        ]
        groups = cluster_subpatterns(runs)
        assert "fast_solve" in groups
        assert groups["fast_solve"][0].run_id == "r1"

    def test_stuck(self):
        short = _make_run([_step("read")] * 5, True, "r1")
        long_fail = _make_run([_step("read")] * 50, False, "r2")
        groups = cluster_subpatterns([short, long_fail])
        assert "stuck" in groups
        assert groups["stuck"][0].run_id == "r2"
