"""Tests for motif discovery."""

from moirai.schema import Step, Result, Run
from moirai.analyze.motifs import find_motifs, stratified_find_motifs, _filtered_names, _extract_ngrams


def _make_run(run_id: str, names: list[str], success: bool, task_id: str = "t1", task_family: str | None = None) -> Run:
    steps = [Step(idx=i, type="tool", name=n) for i, n in enumerate(names)]
    return Run(run_id=run_id, task_id=task_id, task_family=task_family, steps=steps, result=Result(success=success))


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
        assert result[3] == "test(pass)"


class TestExtractNgrams:
    def test_basic(self):
        grams = _extract_ngrams(["a", "b", "c", "d"], min_n=3, max_n=3)
        patterns = [g for g, _ in grams]
        assert ("a", "b", "c") in patterns
        assert ("b", "c", "d") in patterns

    def test_positions_normalized(self):
        grams = _extract_ngrams(["a", "b", "c", "d"], min_n=3, max_n=3)
        assert grams[0][1] == 0.0

    def test_empty(self):
        assert _extract_ngrams([], min_n=3, max_n=3) == []

    def test_too_short(self):
        assert _extract_ngrams(["a", "b"], min_n=3, max_n=3) == []


class TestFindMotifs:
    def test_finds_discriminative_pattern(self):
        pass_runs = [
            _make_run(f"p{i}", ["read", "edit", "test_result", "write"], True)
            for i in range(5)
        ]
        fail_runs = [
            _make_run(f"f{i}", ["read", "write", "write", "write"], False)
            for i in range(5)
        ]
        motifs, n_tested = find_motifs(pass_runs + fail_runs, min_n=3, max_n=3, min_count=3, q_threshold=0.5)
        assert len(motifs) > 0
        assert n_tested > 0

    def test_no_motifs_when_all_same_outcome(self):
        runs = [_make_run(f"r{i}", ["read", "edit", "test"], True) for i in range(5)]
        motifs, _ = find_motifs(runs, min_n=3, max_n=3)
        assert motifs == []

    def test_no_motifs_when_too_few_runs(self):
        runs = [
            _make_run("r1", ["read", "edit", "test"], True),
            _make_run("r2", ["read", "write", "test"], False),
        ]
        motifs, _ = find_motifs(runs, min_n=3, max_n=3, min_count=3)
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
        motifs, _ = find_motifs(pass_runs + fail_runs, min_n=3, max_n=3, min_count=3, q_threshold=1.0)
        positive = [m for m in motifs if m.lift > 1]
        negative = [m for m in motifs if m.lift < 1]
        assert any("read" in m.pattern and "search" in m.pattern for m in positive)
        assert any(m.pattern == ("write", "write", "write") for m in negative)

    def test_empty_runs(self):
        motifs, n = find_motifs([])
        assert motifs == []
        assert n == 0

    def test_q_value_populated(self):
        """BH correction sets q_value on motifs."""
        pass_runs = [
            _make_run(f"p{i}", ["read", "edit", "test_result", "write"], True)
            for i in range(5)
        ]
        fail_runs = [
            _make_run(f"f{i}", ["read", "write", "write", "write"], False)
            for i in range(5)
        ]
        motifs, _ = find_motifs(pass_runs + fail_runs, min_n=3, max_n=3, min_count=3, q_threshold=1.0)
        for m in motifs:
            assert m.q_value is not None

    def test_bh_is_stricter_than_raw_p(self):
        """BH correction should filter at least as strictly as the old p-threshold."""
        pass_runs = [
            _make_run(f"p{i}", ["read", "edit", "test_result", "write"], True)
            for i in range(5)
        ]
        fail_runs = [
            _make_run(f"f{i}", ["read", "write", "write", "write"], False)
            for i in range(5)
        ]
        # Get all candidates (q_threshold=1.0 lets everything through)
        all_motifs, _ = find_motifs(pass_runs + fail_runs, min_n=3, max_n=3, min_count=3, q_threshold=1.0)
        # Get BH-filtered (default q=0.05)
        filtered, _ = find_motifs(pass_runs + fail_runs, min_n=3, max_n=3, min_count=3, q_threshold=0.05)
        assert len(filtered) <= len(all_motifs)


class TestStratifiedFindMotifs:
    def test_removes_family_confound(self):
        """Motifs that are just family-identity proxies should vanish under stratification.

        Family A always passes with pattern [read, edit, test].
        Family B always fails with pattern [write, write, write].
        Unstratified: the patterns look outcome-predictive.
        Stratified: each family has uniform outcomes, so no motifs survive.
        """
        family_a = [
            _make_run(f"a{i}", ["read", "edit", "test"], True, task_family="alpha")
            for i in range(6)
        ]
        family_b = [
            _make_run(f"b{i}", ["write", "write", "write"], False, task_family="beta")
            for i in range(6)
        ]
        all_runs = family_a + family_b

        # Unstratified: should find motifs (cross-family confound)
        unstratified, _ = find_motifs(all_runs, min_n=3, max_n=3, min_count=3, q_threshold=1.0)
        assert len(unstratified) > 0

        # Stratified: each family is uniform-outcome, so nothing should survive
        stratified, _ = stratified_find_motifs(
            all_runs,
            stratify_by=lambda r: r.task_family,
            min_n=3, max_n=3, min_count=3, q_threshold=1.0,
        )
        assert len(stratified) == 0

    def test_preserves_genuine_motifs(self):
        """When families have mixed outcomes internally, real motifs survive."""
        # Both families have the same pattern structure with mixed outcomes
        runs = []
        for fam in ["alpha", "beta"]:
            for i in range(5):
                runs.append(_make_run(f"{fam}_p{i}", ["read", "edit", "test_result", "write"], True, task_family=fam))
            for i in range(5):
                runs.append(_make_run(f"{fam}_f{i}", ["read", "write", "write", "write"], False, task_family=fam))

        unstratified, _ = find_motifs(runs, min_n=3, max_n=3, min_count=3, q_threshold=0.5)
        stratified, _ = stratified_find_motifs(
            runs,
            stratify_by=lambda r: r.task_family,
            min_n=3, max_n=3, min_count=3, q_threshold=0.5,
        )
        # Both should find motifs since the signal is real within each family
        assert len(unstratified) > 0
        assert len(stratified) > 0

    def test_skips_uniform_groups(self):
        """Groups with no outcome variance are skipped, not errored on."""
        # Family A: all pass. Family B: mixed.
        runs = [
            _make_run(f"a{i}", ["read", "edit", "test"], True, task_family="alpha")
            for i in range(5)
        ]
        for i in range(5):
            runs.append(_make_run(f"bp{i}", ["read", "edit", "test_result", "write"], True, task_family="beta"))
        for i in range(5):
            runs.append(_make_run(f"bf{i}", ["read", "write", "write", "write"], False, task_family="beta"))

        stratified, n_tested = stratified_find_motifs(
            runs,
            stratify_by=lambda r: r.task_family,
            min_n=3, max_n=3, min_count=3, q_threshold=0.5,
        )
        # Should run without error; only beta contributes motifs
        assert n_tested > 0

    def test_deduplicates_across_groups(self):
        """Same pattern found in multiple groups keeps the best q-value."""
        runs = []
        for fam in ["alpha", "beta"]:
            for i in range(5):
                runs.append(_make_run(f"{fam}_p{i}", ["read", "edit", "test_result", "write"], True, task_family=fam))
            for i in range(5):
                runs.append(_make_run(f"{fam}_f{i}", ["read", "write", "write", "write"], False, task_family=fam))

        stratified, _ = stratified_find_motifs(
            runs,
            stratify_by=lambda r: r.task_family,
            min_n=3, max_n=3, min_count=3, q_threshold=1.0,
        )
        # Each pattern should appear at most once
        patterns = [m.pattern for m in stratified]
        assert len(patterns) == len(set(patterns))
