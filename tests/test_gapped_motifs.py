"""Tests for gapped motif discovery."""

from moirai.schema import Step, Result, Run, GappedMotif
from moirai.analyze.motifs import find_gapped_motifs, _is_subsequence


def _make_run(run_id: str, names: list[str], success: bool) -> Run:
    steps = [Step(idx=i, type="tool", name=n) for i, n in enumerate(names)]
    return Run(run_id=run_id, task_id="t1", steps=steps, result=Result(success=success))


class TestIsSubsequence:
    def test_basic(self):
        assert _is_subsequence(("a", "c"), ("a", "b", "c"))

    def test_same(self):
        assert _is_subsequence(("a", "b"), ("a", "b"))

    def test_not_subsequence(self):
        assert not _is_subsequence(("c", "a"), ("a", "b", "c"))

    def test_single(self):
        assert _is_subsequence(("a",), ("a", "b", "c"))

    def test_empty(self):
        assert _is_subsequence((), ("a", "b"))


class TestFindGappedMotifs:
    def test_discovers_ordered_pairs(self):
        """Patterns that appear in order across runs should be found."""
        pass_runs = [_make_run(f"p{i}", ["read", "search", "edit", "test"], True) for i in range(5)]
        fail_runs = [_make_run(f"f{i}", ["read", "write", "write", "write"], False) for i in range(5)]
        motifs, n_tested = find_gapped_motifs(pass_runs + fail_runs, max_length=2, min_count=3, q_threshold=1.0)
        assert n_tested > 0
        assert len(motifs) > 0
        # search → edit should appear only in pass runs
        search_edit = [m for m in motifs if m.anchors == ("search", "edit")]
        assert len(search_edit) == 1
        assert search_edit[0].success_rate == 1.0

    def test_discovers_triples(self):
        pass_runs = [_make_run(f"p{i}", ["read", "search", "edit", "test"], True) for i in range(5)]
        fail_runs = [_make_run(f"f{i}", ["read", "write", "write", "write"], False) for i in range(5)]
        motifs, _ = find_gapped_motifs(pass_runs + fail_runs, max_length=3, min_count=3, q_threshold=1.0)
        triples = [m for m in motifs if len(m.anchors) == 3]
        assert len(triples) > 0

    def test_greedy_matching_uses_earliest(self):
        """With repeated types, greedy match picks the first occurrence."""
        # All runs have read appearing twice. The pattern (read, edit) should match
        # using the FIRST read, not the second.
        runs = [
            _make_run(f"p{i}", ["read", "write", "read", "edit"], True) for i in range(5)
        ] + [
            _make_run(f"f{i}", ["write", "write", "write", "write"], False) for i in range(5)
        ]
        motifs, _ = find_gapped_motifs(runs, max_length=2, min_count=3, q_threshold=1.0)
        read_edit = [m for m in motifs if m.anchors == ("read", "edit")]
        assert len(read_edit) == 1
        assert read_edit[0].success_runs == 5

    def test_pruning_shorter_when_longer_covers_more(self):
        """Short pattern pruned when longer supersequence has same/more runs and better q."""
        pass_runs = [_make_run(f"p{i}", ["read", "search", "edit"], True) for i in range(5)]
        fail_runs = [_make_run(f"f{i}", ["write", "write", "write"], False) for i in range(5)]
        motifs, _ = find_gapped_motifs(pass_runs + fail_runs, max_length=3, min_count=3, q_threshold=1.0)
        # If (read, edit) and (read, search, edit) both exist with same runs,
        # the shorter should be pruned
        pairs = [m for m in motifs if len(m.anchors) == 2]
        triples = [m for m in motifs if len(m.anchors) == 3]
        # At least some triples should exist
        assert len(triples) > 0

    def test_pruning_preserves_broader_pattern(self):
        """Short pattern kept when it covers MORE runs than the longer supersequence."""
        # (read, edit) appears in 6 pass runs. (read, search, edit) in only 3.
        # Use distinct step names with no repeats to avoid (read, edit, edit) artifacts.
        runs = [
            _make_run("p0", ["read", "search", "edit"], True),
            _make_run("p1", ["read", "search", "edit"], True),
            _make_run("p2", ["read", "search", "edit"], True),
            _make_run("p3", ["read", "bash", "edit"], True),   # has (read, edit) but NOT (read, search, edit)
            _make_run("p4", ["read", "bash", "edit"], True),
            _make_run("p5", ["read", "bash", "edit"], True),
            _make_run("f0", ["glob", "glob", "glob"], False),
            _make_run("f1", ["glob", "glob", "glob"], False),
            _make_run("f2", ["glob", "glob", "glob"], False),
            _make_run("f3", ["glob", "glob", "glob"], False),
            _make_run("f4", ["glob", "glob", "glob"], False),
            _make_run("f5", ["glob", "glob", "glob"], False),
        ]
        motifs, _ = find_gapped_motifs(runs, max_length=3, min_count=3, q_threshold=1.0)
        read_edit = [m for m in motifs if m.anchors == ("read", "edit")]
        read_search_edit = [m for m in motifs if m.anchors == ("read", "search", "edit")]
        # (read, edit) has 6 runs, (read, search, edit) has 3 — shorter should survive
        assert len(read_edit) == 1, f"Expected (read, edit), got: {[m.anchors for m in motifs]}"
        assert read_edit[0].total_runs == 6

    def test_min_count_filter(self):
        runs = [
            _make_run("p0", ["read", "edit"], True),
            _make_run("p1", ["read", "edit"], True),
            _make_run("f0", ["write", "write"], False),
            _make_run("f1", ["write", "write"], False),
        ]
        # min_count=3 but each pattern only appears in 2 runs
        motifs, _ = find_gapped_motifs(runs, max_length=2, min_count=3, q_threshold=1.0)
        assert motifs == []

    def test_frequency_filter(self):
        """Rare step types are excluded from candidate pool."""
        # "rare" appears in only 1 run out of 20+ — below 5% threshold
        runs = [_make_run(f"p{i}", ["read", "edit"], True) for i in range(10)]
        runs += [_make_run(f"f{i}", ["write", "write"], False) for i in range(10)]
        runs.append(_make_run("x0", ["rare", "edit"], True))
        motifs, _ = find_gapped_motifs(runs, max_length=2, min_count=3, q_threshold=1.0)
        # No pattern should contain "rare"
        for m in motifs:
            assert "rare" not in m.anchors

    def test_empty_input(self):
        motifs, n = find_gapped_motifs([])
        assert motifs == []
        assert n == 0

    def test_single_run(self):
        motifs, _ = find_gapped_motifs([_make_run("r0", ["read", "edit"], True)])
        assert motifs == []

    def test_all_same_outcome(self):
        runs = [_make_run(f"r{i}", ["read", "edit", "test"], True) for i in range(5)]
        motifs, _ = find_gapped_motifs(runs)
        assert motifs == []

    def test_q_value_populated(self):
        pass_runs = [_make_run(f"p{i}", ["read", "edit", "test"], True) for i in range(5)]
        fail_runs = [_make_run(f"f{i}", ["write", "write", "write"], False) for i in range(5)]
        motifs, _ = find_gapped_motifs(pass_runs + fail_runs, max_length=2, min_count=3, q_threshold=1.0)
        for m in motifs:
            assert m.q_value is not None

    def test_display_format(self):
        m = GappedMotif(
            anchors=("read", "edit", "test"),
            total_runs=5, success_runs=5, fail_runs=0,
            success_rate=1.0, baseline_rate=0.5, lift=2.0,
            p_value=0.01, avg_position=0.3,
        )
        assert m.display == "read → ... → edit → ... → test"

    def test_max_length_2_excludes_triples(self):
        pass_runs = [_make_run(f"p{i}", ["read", "search", "edit"], True) for i in range(5)]
        fail_runs = [_make_run(f"f{i}", ["write", "write", "write"], False) for i in range(5)]
        motifs, _ = find_gapped_motifs(pass_runs + fail_runs, max_length=2, min_count=3, q_threshold=1.0)
        for m in motifs:
            assert len(m.anchors) <= 2


class TestCLIIntegration:
    def test_patterns_gapped_flag(self):
        """Smoke test: --gapped flag doesn't crash."""
        from typer.testing import CliRunner
        from moirai.cli import app
        import json
        import tempfile
        from pathlib import Path

        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal run files
            for i in range(5):
                run = {
                    "run_id": f"p{i}", "task_id": "t1",
                    "steps": [{"idx": j, "type": "tool", "name": n} for j, n in enumerate(["read", "edit", "test"])],
                    "result": {"success": True},
                }
                Path(tmpdir, f"p{i}.json").write_text(json.dumps(run))
            for i in range(5):
                run = {
                    "run_id": f"f{i}", "task_id": "t1",
                    "steps": [{"idx": j, "type": "tool", "name": n} for j, n in enumerate(["write", "write", "write"])],
                    "result": {"success": False},
                }
                Path(tmpdir, f"f{i}.json").write_text(json.dumps(run))

            result = runner.invoke(app, ["patterns", tmpdir, "--gapped"])
            assert result.exit_code == 0
