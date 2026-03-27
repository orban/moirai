"""Tests for motif discovery."""

from moirai.schema import Step, Result, Run
from moirai.analyze.motifs import find_motifs, _filtered_names, _extract_ngrams


def _make_run(run_id: str, names: list[str], success: bool) -> Run:
    steps = [Step(idx=i, type="tool", name=n) for i, n in enumerate(names)]
    return Run(run_id=run_id, task_id="t1", steps=steps, result=Result(success=success))


class TestFilteredNames:
    def test_filters_noise(self):
        run = _make_run("r1", ["read", "error_observation", "action", "edit"], True)
        assert _filtered_names(run) == ["read", "edit"]

    def test_keeps_meaningful(self):
        run = _make_run("r1", ["read", "search", "write", "test_result"], True)
        result = _filtered_names(run)
        assert result[0] == "read"
        assert result[1] == "search"
        assert result[2] == "write"
        assert result[3] == "test(pass)"  # test_result with ok status → test(pass)


class TestExtractNgrams:
    def test_basic(self):
        grams = _extract_ngrams(["a", "b", "c", "d"], min_n=3, max_n=3)
        patterns = [g for g, _ in grams]
        assert ("a", "b", "c") in patterns
        assert ("b", "c", "d") in patterns

    def test_positions_normalized(self):
        grams = _extract_ngrams(["a", "b", "c", "d"], min_n=3, max_n=3)
        # First gram starts at position 0
        assert grams[0][1] == 0.0

    def test_empty(self):
        assert _extract_ngrams([], min_n=3, max_n=3) == []

    def test_too_short(self):
        assert _extract_ngrams(["a", "b"], min_n=3, max_n=3) == []


class TestFindMotifs:
    def test_finds_discriminative_pattern(self):
        # Pattern "read → edit → test" appears only in successful runs
        pass_runs = [
            _make_run(f"p{i}", ["read", "edit", "test_result", "write"], True)
            for i in range(5)
        ]
        fail_runs = [
            _make_run(f"f{i}", ["read", "write", "write", "write"], False)
            for i in range(5)
        ]
        motifs = find_motifs(pass_runs + fail_runs, min_n=3, max_n=3, min_count=3, p_threshold=0.5)
        # Should find patterns unique to pass or fail runs
        assert len(motifs) > 0

    def test_no_motifs_when_all_same_outcome(self):
        runs = [_make_run(f"r{i}", ["read", "edit", "test"], True) for i in range(5)]
        motifs = find_motifs(runs, min_n=3, max_n=3)
        assert motifs == []

    def test_no_motifs_when_too_few_runs(self):
        runs = [
            _make_run("r1", ["read", "edit", "test"], True),
            _make_run("r2", ["read", "write", "test"], False),
        ]
        motifs = find_motifs(runs, min_n=3, max_n=3, min_count=3)
        assert motifs == []

    def test_lift_direction(self):
        pass_runs = [
            _make_run(f"p{i}", ["read", "search", "edit", "test_result"], True)
            for i in range(6)
        ]
        fail_runs = [
            _make_run(f"f{i}", ["write", "write", "write", "test_result"], False)
            for i in range(6)
        ]
        motifs = find_motifs(pass_runs + fail_runs, min_n=3, max_n=3, min_count=3, p_threshold=1.0)
        positive = [m for m in motifs if m.lift > 1]
        negative = [m for m in motifs if m.lift < 1]
        # read→search→edit should be positive, write→write→write should be negative
        assert any("read" in m.pattern and "search" in m.pattern for m in positive)
        assert any(m.pattern == ("write", "write", "write") for m in negative)

    def test_empty_runs(self):
        assert find_motifs([]) == []
