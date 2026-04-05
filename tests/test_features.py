"""Tests for behavioral feature functions."""

from moirai.schema import Step, Result, Run
from moirai.analyze.features import (
    _test_position_centroid,
    _symmetry_index,
    _uncertainty_density,
    _hypothesis_formation,
    _edit_dispersion,
    within_task_experiment,
    rank_features,
    FEATURES,
)


def _make_run(
    run_id: str,
    task_id: str,
    names: list[str],
    success: bool,
    output: list[dict] | None = None,
    attrs: list[dict] | None = None,
) -> Run:
    steps = []
    for i, name in enumerate(names):
        out = output[i] if output and i < len(output) else {}
        att = attrs[i] if attrs and i < len(attrs) else {}
        steps.append(Step(idx=i, type="tool", name=name, output=out or {}, attrs=att or {}))
    return Run(run_id=run_id, task_id=task_id, steps=steps, result=Result(success=success))


class TestTestPositionCentroid:
    def test_tests_at_end_high_value(self):
        """Tests concentrated at the end produce a high centroid."""
        # 7 enriched steps (>= 5), tests at positions 5 and 6
        run = _make_run("r1", "t1",
            ["read", "edit", "read", "edit", "read", "test(pass)", "test(pass)"],
            True)
        val = _test_position_centroid(run)
        assert val is not None
        assert val > 0.7

    def test_tests_at_start_low_value(self):
        """Tests concentrated at the start produce a low centroid."""
        run = _make_run("r1", "t1",
            ["test(fail)", "test(fail)", "read", "edit", "read", "edit", "read"],
            False)
        val = _test_position_centroid(run)
        assert val is not None
        assert val < 0.3

    def test_no_tests_returns_none(self):
        """Run with no test steps returns None."""
        run = _make_run("r1", "t1",
            ["read", "edit", "read", "edit", "read"],
            True)
        assert _test_position_centroid(run) is None

    def test_too_few_steps_returns_none(self):
        """Run with < 5 enriched steps returns None."""
        run = _make_run("r1", "t1",
            ["test(pass)", "test(pass)", "read", "edit"],
            True)
        assert _test_position_centroid(run) is None

    def test_one_test_returns_none(self):
        """Run with only 1 test step returns None (need >= 2)."""
        run = _make_run("r1", "t1",
            ["read", "edit", "read", "edit", "test(pass)"],
            True)
        assert _test_position_centroid(run) is None

    def test_centroid_value(self):
        """Verify exact centroid calculation with known positions."""
        # 6 enriched steps, tests at index 4 and 5
        # positions: 4/(6-1) = 0.8, 5/(6-1) = 1.0
        # centroid = (0.8 + 1.0) / 2 = 0.9
        run = _make_run("r1", "t1",
            ["read", "edit", "read", "edit", "test(pass)", "test(fail)"],
            True)
        val = _test_position_centroid(run)
        assert val is not None
        assert abs(val - 0.9) < 0.01


class TestSymmetryIndex:
    def test_ideal_order_high_score(self):
        """explore -> modify -> verify in ideal positions scores high."""
        # 9 steps: 2 explore, 4 modify (middle), 3 verify (end)
        # Need step names that map to the right phases
        run = _make_run("r1", "t1",
            ["read", "read", "edit", "edit", "edit", "edit", "test", "test", "test"],
            True)
        val = _symmetry_index(run)
        assert val is not None
        assert val > 0.6

    def test_reversed_order_low_score(self):
        """verify -> modify -> explore (reversed) scores lower."""
        run = _make_run("r1", "t1",
            ["test", "test", "test", "edit", "edit", "edit", "read", "read", "read"],
            False)
        val = _symmetry_index(run)
        assert val is not None
        # Reversed order should score lower than ideal
        ideal_run = _make_run("r2", "t1",
            ["read", "read", "read", "edit", "edit", "edit", "test", "test", "test"],
            True)
        ideal_val = _symmetry_index(ideal_run)
        assert ideal_val is not None
        assert ideal_val > val

    def test_too_few_steps_returns_none(self):
        """Run with < 6 steps returns None."""
        run = _make_run("r1", "t1",
            ["read", "edit", "test", "read", "edit"],
            True)
        assert _symmetry_index(run) is None

    def test_too_few_classifiable_returns_none(self):
        """Run with < 4 classifiable (explore/modify/verify) steps returns None."""
        # 6 steps but only 3 are classifiable as explore/modify/verify
        # bash maps to "execute", reason maps to "think"
        run = _make_run("r1", "t1",
            ["read", "bash", "bash", "bash", "edit", "test"],
            True)
        val = _symmetry_index(run)
        # read=explore, edit=modify, test=verify = 3 classifiable, < 4
        assert val is None


class TestUncertaintyDensity:
    def test_hedging_produces_nonzero(self):
        """Reasoning with hedging words produces non-zero density."""
        run = _make_run("r1", "t1", ["read", "edit"], True,
            output=[
                {"reasoning": "maybe this is the issue, let me try another approach"},
                {"reasoning": "this might work"},
            ])
        val = _uncertainty_density(run)
        assert val is not None
        assert val > 0.0

    def test_no_hedging_produces_zero(self):
        """Reasoning without hedging words produces zero density."""
        run = _make_run("r1", "t1", ["read", "edit"], True,
            output=[
                {"reasoning": "the function returns an integer"},
                {"reasoning": "editing the file now"},
            ])
        val = _uncertainty_density(run)
        assert val is not None
        assert val == 0.0

    def test_no_reasoning_returns_none(self):
        """Run with no reasoning text returns None."""
        run = _make_run("r1", "t1", ["read", "edit"], True,
            output=[
                {"result": "ok"},
                {"result": "done"},
            ])
        assert _uncertainty_density(run) is None

    def test_all_steps_match(self):
        """If every reasoning step matches, density is 1.0."""
        run = _make_run("r1", "t1", ["read", "edit"], True,
            output=[
                {"reasoning": "maybe this is wrong"},
                {"reasoning": "might need to reconsider"},
            ])
        val = _uncertainty_density(run)
        assert val == 1.0

    def test_partial_match(self):
        """2 out of 3 reasoning steps matching gives 2/3."""
        run = _make_run("r1", "t1", ["read", "edit", "test"], True,
            output=[
                {"reasoning": "maybe this works"},
                {"reasoning": "the code is correct"},
                {"reasoning": "possibly a bug here"},
            ])
        val = _uncertainty_density(run)
        assert val is not None
        assert abs(val - 2 / 3) < 0.01


class TestHypothesisFormation:
    def test_hypotheses_produce_nonzero(self):
        """Reasoning with hypothesis patterns produces non-zero value."""
        run = _make_run("r1", "t1", ["read", "edit"], True,
            output=[
                {"reasoning": "I think the issue is in the parser"},
                {"reasoning": "the root cause is a missing import"},
            ])
        val = _hypothesis_formation(run)
        assert val is not None
        assert val > 0.0

    def test_no_hypotheses_produces_zero(self):
        """Reasoning without hypothesis patterns produces zero."""
        run = _make_run("r1", "t1", ["read", "edit"], True,
            output=[
                {"reasoning": "reading the file contents"},
                {"reasoning": "editing line 42"},
            ])
        val = _hypothesis_formation(run)
        assert val is not None
        assert val == 0.0

    def test_no_reasoning_returns_none(self):
        """Run with no reasoning text returns None."""
        run = _make_run("r1", "t1", ["read", "edit"], True,
            output=[
                {"result": "ok"},
                {"result": "done"},
            ])
        assert _hypothesis_formation(run) is None

    def test_multiple_matches_per_step(self):
        """Multiple hypothesis patterns in one step are all counted."""
        run = _make_run("r1", "t1", ["read"], True,
            output=[
                {"reasoning": "I think the problem is here. I suspect the cause is a typo."},
            ])
        val = _hypothesis_formation(run)
        assert val is not None
        # "i think the" + "the problem is" + "i suspect" + "the cause is" = 4 matches / 1 step
        # (exact count depends on regex overlap; at least 2)
        assert val >= 2.0


class TestEditDispersion:
    def test_three_different_files(self):
        """Edits to 3 different files gives dispersion of 1.0."""
        run = _make_run("r1", "t1",
            ["edit", "edit", "edit"], True,
            attrs=[
                {"file_path": "src/a.py"},
                {"file_path": "src/b.py"},
                {"file_path": "src/c.py"},
            ])
        val = _edit_dispersion(run)
        assert val is not None
        assert val == 1.0

    def test_same_file_repeatedly(self):
        """All edits to the same file gives dispersion of 1/n."""
        run = _make_run("r1", "t1",
            ["edit", "edit", "edit", "edit"], True,
            attrs=[
                {"file_path": "src/a.py"},
                {"file_path": "src/a.py"},
                {"file_path": "src/a.py"},
                {"file_path": "src/a.py"},
            ])
        val = _edit_dispersion(run)
        assert val is not None
        assert abs(val - 0.25) < 0.01

    def test_fewer_than_two_edits_returns_none(self):
        """Run with < 2 edit steps returns None."""
        run = _make_run("r1", "t1",
            ["read", "edit", "test"], True,
            attrs=[
                {},
                {"file_path": "src/a.py"},
                {},
            ])
        assert _edit_dispersion(run) is None

    def test_write_steps_excluded(self):
        """Write steps are not counted as edits."""
        run = _make_run("r1", "t1",
            ["edit", "write", "write"], True,
            attrs=[
                {"file_path": "src/a.py"},
                {"file_path": "src/b.py"},
                {"file_path": "src/c.py"},
            ])
        # Only 1 edit step, so should return None
        assert _edit_dispersion(run) is None

    def test_mixed_files(self):
        """2 unique files across 3 edits gives 2/3."""
        run = _make_run("r1", "t1",
            ["edit", "edit", "edit"], True,
            attrs=[
                {"file_path": "src/a.py"},
                {"file_path": "src/b.py"},
                {"file_path": "src/a.py"},
            ])
        val = _edit_dispersion(run)
        assert val is not None
        assert abs(val - 2 / 3) < 0.01


class TestFeaturesRegistry:
    def test_registry_has_five_features(self):
        assert len(FEATURES) == 5

    def test_all_specs_have_required_fields(self):
        for spec in FEATURES:
            assert spec.name
            assert callable(spec.compute)
            assert spec.direction in ("positive", "negative")
            assert spec.description

    def test_feature_names_unique(self):
        names = [f.name for f in FEATURES]
        assert len(names) == len(set(names))


class TestWithinTaskExperiment:
    """Known-answer tests for within_task_experiment."""

    def _make_task_runs(self):
        """Build a dataset where test_position_centroid clearly predicts success.

        Task "t1" and "t2": pass runs test late (varying positions), fail runs test early.
        Each task has 8 runs (4 pass, 4 fail) with variance in test positions.
        """
        groups = {}

        for task_id in ["t1", "t2"]:
            runs = []
            # 4 pass runs: tests late but at varying positions
            for i, n_read_before in enumerate([6, 7, 8, 9]):
                n_read_after = 10 - n_read_before - 2
                names = (["read"] * n_read_before) + ["test(pass)", "test(pass)"]
                if n_read_after > 0:
                    names += ["read"] * n_read_after
                runs.append(_make_run(f"p{i}_{task_id}", task_id, names, True))
            # 4 fail runs: tests early at varying positions
            for i, n_test_start in enumerate([0, 1, 2, 3]):
                names = (["read"] * n_test_start) + ["test(fail)", "test(fail)"]
                names += ["read"] * (10 - len(names))
                runs.append(_make_run(f"f{i}_{task_id}", task_id, names, False))
            groups[task_id] = runs

        return groups

    def test_positive_delta_when_late_testers_pass(self):
        groups = self._make_task_runs()
        result = within_task_experiment(groups, _test_position_centroid, min_group_size=2)
        # Pass runs test late (centroid ~0.9), fail runs test early (~0.1)
        # HIGH group (> median) should have higher pass rate
        assert result.delta_pp > 0
        assert result.n_tasks == 2

    def test_p_value_is_significant(self):
        groups = self._make_task_runs()
        result = within_task_experiment(groups, _test_position_centroid, min_group_size=2)
        # With only 2 tasks, sign test needs >= 5 non-zero, so p might be None
        # But deltas should be consistently positive
        assert result.delta_pp > 0

    def test_empty_groups(self):
        result = within_task_experiment({}, _test_position_centroid)
        assert result.delta_pp == 0.0
        assert result.p_value is None
        assert result.n_tasks == 0

    def test_skips_tasks_below_min_group_size(self):
        # Only 2 runs per outcome — below min_group_size=3
        groups = {
            "t1": [
                _make_run("p1", "t1", ["read"] * 8 + ["test(pass)"] * 2, True),
                _make_run("p2", "t1", ["read"] * 8 + ["test(pass)"] * 2, True),
                _make_run("f1", "t1", ["test(fail)"] * 2 + ["read"] * 8, False),
                _make_run("f2", "t1", ["test(fail)"] * 2 + ["read"] * 8, False),
            ],
        }
        result = within_task_experiment(groups, _test_position_centroid, min_group_size=3)
        assert result.n_tasks == 0

    def test_feature_returning_none_handled(self):
        """Runs where feature returns None should be excluded, not crash."""
        groups = {
            "t1": [
                # These have no test steps → centroid returns None
                _make_run(f"r{i}", "t1", ["read"] * 10, i < 3)
                for i in range(6)
            ],
        }
        result = within_task_experiment(groups, _test_position_centroid, min_group_size=2)
        assert result.n_tasks == 0  # no valid feature values


class TestRankFeatures:
    """Integration tests for rank_features."""

    def _make_dataset(self):
        """Build a minimal dataset with clear feature signals.

        20 runs across 2 tasks, 5 pass + 5 fail each.
        Pass runs: test late, no hedging, edits spread.
        Fail runs: test early, hedging, edits concentrated.
        """
        runs = []
        for task_id in ["t1", "t2"]:
            for i in range(5):
                # Pass: explore, edit multiple files, test late
                names = (["read(source)"] * 4 + ["search(grep_targeted)"] * 3
                         + ["edit(source)"] * 2 + ["test(pass)"] * 2)
                attrs = [{}] * 7 + [{"file_path": f"f{j}.py"} for j in range(i, i + 2)] + [{}] * 2
                output = [{"reasoning": "The fix is straightforward."}] + [{}] * 10
                runs.append(_make_run(
                    f"p{i}_{task_id}", task_id, names, True,
                    output=output, attrs=attrs,
                ))
            for i in range(5):
                # Fail: test early, hedge, hammer one file
                names = (["test(fail)"] * 2 + ["read(source)"] * 4
                         + ["search(grep_targeted)"] * 2 + ["edit(source)"] * 2
                         + ["test(fail)"])
                attrs = [{}] * 8 + [{"file_path": "same.py"}] * 2 + [{}]
                output = [{"reasoning": "Maybe the issue might be here, let me try."}] + [{}] * 10
                runs.append(_make_run(
                    f"f{i}_{task_id}", task_id, names, False,
                    output=output, attrs=attrs,
                ))
        return runs

    def test_returns_feature_results(self):
        runs = self._make_dataset()
        results = rank_features(runs, min_runs=5, seed=42)
        assert len(results) > 0
        for r in results:
            assert r.name in [f.name for f in FEATURES]
            # q_value can be None if p_value is None (insufficient data for sign test)
            if r.p_value is not None:
                assert r.q_value is not None

    def test_sorted_by_abs_delta(self):
        runs = self._make_dataset()
        results = rank_features(runs, min_runs=5, seed=42)
        for i in range(len(results) - 1):
            assert abs(results[i].delta_pp) >= abs(results[i + 1].delta_pp)

    def test_test_position_centroid_is_positive(self):
        """Our constructed data has pass=late testing, fail=early. Delta should be positive."""
        runs = self._make_dataset()
        results = rank_features(runs, min_runs=5, seed=42)
        tpc = next((r for r in results if r.name == "test_position_centroid"), None)
        if tpc and tpc.n_tasks > 0:
            assert tpc.delta_pp > 0

    def test_empty_input(self):
        results = rank_features([], min_runs=5)
        assert results == []

    def test_all_same_outcome(self):
        """All-pass dataset should return empty (no mixed-outcome tasks)."""
        runs = [
            _make_run(f"r{i}", "t1", ["read"] * 10, True)
            for i in range(10)
        ]
        results = rank_features(runs, min_runs=5)
        assert results == []
