"""Tests for divergence detection."""

from moirai.schema import Alignment, GAP, Step, Result, Run
from moirai.analyze.divergence import find_activity_divergences, find_divergence_points, generate_claim, summarize_point
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


class TestActivityDivergences:
    def test_pass_active_fail_gap(self):
        """Column where pass runs have steps and fail runs have gaps."""
        alignment = Alignment(
            run_ids=["p1", "p2", "p3", "f1", "f2", "f3"],
            matrix=[
                ["llm", "tool"],
                ["llm", "tool"],
                ["llm", "tool"],
                ["llm", GAP],
                ["llm", GAP],
                ["llm", GAP],
            ],
            level="type",
        )
        runs = [_make_run("p1", ["llm", "tool"], True),
                _make_run("p2", ["llm", "tool"], True),
                _make_run("p3", ["llm", "tool"], True),
                _make_run("f1", ["llm"], False),
                _make_run("f2", ["llm"], False),
                _make_run("f3", ["llm"], False)]
        result = find_activity_divergences(alignment, runs)
        # Column 1 should be detected: 3/3 pass active, 0/3 fail active
        col1 = [r for r in result if r.column == 1]
        assert len(col1) == 1
        assert col1[0].pass_active == 3
        assert col1[0].fail_active == 0
        assert col1[0].direction > 0  # pass-biased

    def test_fail_active_pass_gap(self):
        """Column where fail runs have steps and pass runs have gaps."""
        alignment = Alignment(
            run_ids=["p1", "p2", "f1", "f2"],
            matrix=[
                ["llm", GAP],
                ["llm", GAP],
                ["llm", "tool"],
                ["llm", "tool"],
            ],
            level="type",
        )
        runs = [_make_run("p1", ["llm"], True),
                _make_run("p2", ["llm"], True),
                _make_run("f1", ["llm", "tool"], False),
                _make_run("f2", ["llm", "tool"], False)]
        result = find_activity_divergences(alignment, runs)
        col1 = [r for r in result if r.column == 1]
        assert len(col1) == 1
        assert col1[0].direction < 0  # fail-biased

    def test_no_divergence_when_all_active(self):
        """Column where everyone is active should not appear."""
        alignment = Alignment(
            run_ids=["p1", "f1"],
            matrix=[["llm"], ["llm"]],
            level="type",
        )
        runs = [_make_run("p1", ["llm"], True),
                _make_run("f1", ["llm"], False)]
        result = find_activity_divergences(alignment, runs)
        assert len(result) == 0

    def test_bh_correction_applied(self):
        """q_value should be populated on all results."""
        alignment = Alignment(
            run_ids=["p1", "p2", "f1", "f2"],
            matrix=[
                ["llm", "tool", GAP],
                ["llm", GAP, "tool"],
                ["llm", GAP, "tool"],
                ["llm", "tool", GAP],
            ],
            level="type",
        )
        runs = [_make_run("p1", ["x"] * 3, True),
                _make_run("p2", ["x"] * 3, True),
                _make_run("f1", ["x"] * 3, False),
                _make_run("f2", ["x"] * 3, False)]
        result = find_activity_divergences(alignment, runs)
        for r in result:
            assert r.q_value is not None

    def test_sorted_by_q_value(self):
        """Results should be sorted by q-value ascending."""
        alignment = Alignment(
            run_ids=["p1", "p2", "p3", "f1", "f2", "f3"],
            matrix=[
                ["a", "b", "c"],
                ["a", "b", GAP],
                ["a", GAP, "c"],
                ["a", GAP, GAP],
                ["a", GAP, GAP],
                ["a", GAP, GAP],
            ],
            level="type",
        )
        runs = [_make_run("p1", ["x"] * 3, True),
                _make_run("p2", ["x"] * 3, True),
                _make_run("p3", ["x"] * 3, True),
                _make_run("f1", ["x"] * 3, False),
                _make_run("f2", ["x"] * 3, False),
                _make_run("f3", ["x"] * 3, False)]
        result = find_activity_divergences(alignment, runs)
        q_vals = [r.q_value for r in result if r.q_value is not None]
        for i in range(len(q_vals) - 1):
            assert q_vals[i] <= q_vals[i + 1] + 0.01

    def test_active_labels_populated(self):
        """active_labels should contain the step names at the column."""
        alignment = Alignment(
            run_ids=["p1", "p2", "f1", "f2"],
            matrix=[
                ["llm", "tool"],
                ["llm", "judge"],
                ["llm", GAP],
                ["llm", GAP],
            ],
            level="type",
        )
        runs = [_make_run("p1", ["llm", "tool"], True),
                _make_run("p2", ["llm", "judge"], True),
                _make_run("f1", ["llm"], False),
                _make_run("f2", ["llm"], False)]
        result = find_activity_divergences(alignment, runs)
        col1 = [r for r in result if r.column == 1]
        assert len(col1) == 1
        assert col1[0].active_labels == {"tool": 1, "judge": 1}

    def test_empty_alignment(self):
        alignment = Alignment(run_ids=[], matrix=[], level="type")
        assert find_activity_divergences(alignment, []) == []

    def test_q_threshold_filtering(self):
        """Strict q_threshold should filter non-significant results."""
        alignment = Alignment(
            run_ids=["p1", "f1"],
            matrix=[["llm", "tool"], ["llm", GAP]],
            level="type",
        )
        runs = [_make_run("p1", ["llm", "tool"], True),
                _make_run("f1", ["llm"], False)]
        # With only 2 runs, nothing should be significant at q<0.05
        strict = find_activity_divergences(alignment, runs, q_threshold=0.05)
        loose = find_activity_divergences(alignment, runs, q_threshold=1.0)
        assert len(strict) <= len(loose)


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


class TestSummarizePoint:
    def test_generic_with_rates(self):
        """Two non-GAP variants with rates produces opinionated summary."""
        from moirai.schema import DivergencePoint
        point = DivergencePoint(
            column=5,
            value_counts={"edit": 4, "search": 3},
            entropy=1.0,
            success_by_value={"edit": 0.75, "search": 0.33},
        )
        s = summarize_point(point)
        assert "step 5" in s
        assert "75%" in s
        assert "33%" in s
        assert "success" in s or "correlates" in s

    def test_gap_pattern_active_better(self):
        """GAP variant where active runs win — takes a stance."""
        from moirai.schema import DivergencePoint
        point = DivergencePoint(
            column=7,
            value_counts={"read(test)": 5, GAP: 3},
            entropy=0.95,
            success_by_value={"read(test)": 0.80, GAP: 0.33},
        )
        s = summarize_point(point)
        assert "80%" in s
        assert "33%" in s
        assert "step 7" in s

    def test_gap_pattern_gap_better(self):
        """GAP variant where skipping is better — says so."""
        from moirai.schema import DivergencePoint
        point = DivergencePoint(
            column=3,
            value_counts={"bash": 4, GAP: 6},
            entropy=0.97,
            success_by_value={"bash": 0.25, GAP: 0.83},
        )
        s = summarize_point(point)
        assert "83%" in s
        assert "25%" in s
        assert "kip" in s.lower()  # "Skipping" or "skip"

    def test_no_rates_still_describes_split(self):
        """No success rates produces a description without percentages."""
        from moirai.schema import DivergencePoint
        point = DivergencePoint(
            column=2,
            value_counts={"edit": 3, GAP: 2},
            entropy=0.97,
            success_by_value={"edit": None, GAP: None},
        )
        s = summarize_point(point)
        assert "step 2" in s
        assert "edit" in s.lower()

    def test_empty_variants(self):
        """Empty value_counts produces fallback."""
        from moirai.schema import DivergencePoint
        point = DivergencePoint(
            column=0,
            value_counts={},
            entropy=0.0,
            success_by_value={},
        )
        s = summarize_point(point)
        assert "0" in s

    def test_no_rate_no_percentages(self):
        """No rates means no percentages in output."""
        from moirai.schema import DivergencePoint
        point = DivergencePoint(
            column=4,
            value_counts={"read": 3, "write": 2},
            entropy=0.97,
            success_by_value={"read": None, "write": None},
        )
        s = summarize_point(point)
        assert "step 4" in s
        assert "%" not in s

    def test_edit_vs_test_pattern_test_wins(self):
        """Edit-vs-test pattern where testing first is better."""
        from moirai.schema import DivergencePoint
        point = DivergencePoint(
            column=10,
            value_counts={"edit(source)": 5, "test(fail)": 4},
            entropy=1.0,
            success_by_value={"edit(source)": 0.20, "test(fail)": 0.75},
        )
        s = summarize_point(point)
        assert "75%" in s
        assert "20%" in s
        assert "step 10" in s

    def test_edit_vs_test_pattern_edit_wins(self):
        """Edit-vs-test pattern where editing first is better."""
        from moirai.schema import DivergencePoint
        point = DivergencePoint(
            column=8,
            value_counts={"write(source)": 6, "test(pass)": 3},
            entropy=0.92,
            success_by_value={"write(source)": 0.83, "test(pass)": 0.33},
        )
        s = summarize_point(point)
        assert "83%" in s
        assert "33%" in s
        assert "wrote" in s.lower() or "write" in s.lower()

    def test_different_file_targets(self):
        """Same base action but different targets."""
        from moirai.schema import DivergencePoint
        point = DivergencePoint(
            column=12,
            value_counts={"read(source)": 5, "read(test_file)": 4},
            entropy=1.0,
            success_by_value={"read(source)": 0.40, "read(test_file)": 0.75},
        )
        s = summarize_point(point)
        assert "source" in s
        assert "test_file" in s
        assert "75%" in s
        assert "40%" in s

    def test_different_targets_no_rates_falls_through(self):
        """Same base action, different targets, but no rates — falls to generic."""
        from moirai.schema import DivergencePoint
        point = DivergencePoint(
            column=6,
            value_counts={"edit(source)": 3, "edit(config)": 2},
            entropy=0.97,
            success_by_value={"edit(source)": None, "edit(config)": None},
        )
        s = summarize_point(point)
        assert "step 6" in s
        assert "edit" in s.lower()


class TestGenerateClaim:
    def test_claim_picks_strongest_divergence(self):
        """Claim picks the point with the largest outcome gap."""
        from moirai.schema import DivergencePoint
        weak = DivergencePoint(
            column=3, value_counts={"a": 5, "b": 5}, entropy=1.0,
            success_by_value={"a": 0.60, "b": 0.40},
        )
        strong = DivergencePoint(
            column=7, value_counts={"x": 4, "y": 6}, entropy=1.0,
            success_by_value={"x": 1.0, "y": 0.0},
        )
        claim = generate_claim([weak, strong], 10)
        assert claim is not None
        assert "step 7" in claim
        assert "100%" in claim

    def test_claim_none_when_no_gap(self):
        """No claim when all variants have equal rates."""
        from moirai.schema import DivergencePoint
        point = DivergencePoint(
            column=1, value_counts={"a": 5, "b": 5}, entropy=1.0,
            success_by_value={"a": 0.50, "b": 0.50},
        )
        claim = generate_claim([point], 10)
        assert claim is None

    def test_claim_none_for_empty_points(self):
        assert generate_claim([], 10) is None

    def test_claim_includes_outcome_gap(self):
        """Claim states the percentage gap."""
        from moirai.schema import DivergencePoint
        point = DivergencePoint(
            column=5, value_counts={"read": 8, "edit": 4}, entropy=0.9,
            success_by_value={"read": 0.875, "edit": 0.25},
        )
        claim = generate_claim([point], 12)
        assert claim is not None
        assert "62%" in claim or "63%" in claim  # 87.5% - 25% = 62.5%
        assert "gap" in claim.lower()
