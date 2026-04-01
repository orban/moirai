"""Tests for stream geometry computation and run metadata serialization."""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage

from moirai.schema import Alignment, GAP, Result, Run, Step
from moirai.viz.stream import build_run_detail, build_run_meta, build_stream_tree


def _make_run(run_id: str, task_id: str, success: bool, steps: list[str]) -> Run:
    return Run(
        run_id=run_id, task_id=task_id, task_family=None, agent=None,
        model=None, harness=None, timestamp=None, tags={},
        steps=[Step(idx=i, type="tool", name=s, status="ok") for i, s in enumerate(steps)],
        result=Result(success=success, score=None, label=None, error_type=None, summary=None),
    )


def _make_alignment_and_linkage(runs: list[Run]) -> tuple[Alignment, np.ndarray]:
    """Build alignment + linkage from runs for testing."""
    from moirai.analyze.align import align_runs, distance_matrix

    alignment = align_runs(runs, level="name")
    condensed = distance_matrix(runs, level="name")
    Z = linkage(condensed, method="average")
    return alignment, Z


# --- Test 1: build_stream_tree basic structure ---

class TestBuildStreamTreeBasic:
    def test_basic_structure(self):
        runs = [
            _make_run("r1", "task1", True, ["read", "edit", "test"]),
            _make_run("r2", "task1", True, ["read", "edit", "test"]),
            _make_run("r3", "task1", False, ["read", "bash", "bash"]),
            _make_run("r4", "task1", False, ["read", "bash", "bash"]),
        ]
        alignment, Z = _make_alignment_and_linkage(runs)

        tree = build_stream_tree(alignment, runs, Z)

        # Top-level keys
        assert tree["task_id"] == "task1"
        assert tree["n_runs"] == 4
        assert tree["n_pass"] == 2
        assert tree["n_fail"] == 2
        assert tree["n_cols"] > 0
        assert tree["root_branch_id"] is not None

        # Branches: should have at least 2 (the two leaf groups) + root
        assert len(tree["branches"]) >= 2

        # Root branch has all 4 runs
        root = tree["branches"][tree["root_branch_id"]]
        assert len(root["run_ids"]) == 4
        assert root["success_rate"] == 0.5

        # At least 1 bifurcation
        assert len(tree["bifurcations"]) >= 1

        # Tree structure: each bifurcation maps to [left, right]
        for bif_id, children in tree["tree"].items():
            assert len(children) == 2
            assert bif_id in tree["bifurcations"]

    def test_branch_fields(self):
        runs = [
            _make_run("r1", "task1", True, ["read", "edit", "test"]),
            _make_run("r2", "task1", False, ["read", "bash", "bash"]),
        ]
        alignment, Z = _make_alignment_and_linkage(runs)

        tree = build_stream_tree(alignment, runs, Z)

        for branch in tree["branches"].values():
            assert "id" in branch
            assert "run_ids" in branch
            assert "success_rate" in branch
            assert "step_proportions" in branch
            assert "trajectory" in branch
            assert "phase_mix" in branch
            assert "col_start" in branch
            assert "col_end" in branch


# --- Test 2: step proportions ---

class TestStepProportions:
    def test_uniform_column(self):
        """When all runs start with 'read', column 0 should be {read: 1.0}."""
        runs = [
            _make_run("r1", "task1", True, ["read", "edit", "test"]),
            _make_run("r2", "task1", True, ["read", "edit", "test"]),
            _make_run("r3", "task1", False, ["read", "bash", "bash"]),
            _make_run("r4", "task1", False, ["read", "bash", "bash"]),
        ]
        alignment, Z = _make_alignment_and_linkage(runs)

        tree = build_stream_tree(alignment, runs, Z)

        # Root branch: all 4 runs
        root = tree["branches"][tree["root_branch_id"]]
        props = root["step_proportions"]

        # Find column 0 proportions — "read" should be 1.0
        assert 0 in props
        assert "read" in props[0]
        assert props[0]["read"] == 1.0

    def test_mixed_column(self):
        """Column 1 should show the mix of edit vs bash."""
        runs = [
            _make_run("r1", "task1", True, ["read", "edit", "test"]),
            _make_run("r2", "task1", True, ["read", "edit", "test"]),
            _make_run("r3", "task1", False, ["read", "bash", "bash"]),
            _make_run("r4", "task1", False, ["read", "bash", "bash"]),
        ]
        alignment, Z = _make_alignment_and_linkage(runs)

        tree = build_stream_tree(alignment, runs, Z)

        root = tree["branches"][tree["root_branch_id"]]
        props = root["step_proportions"]

        # Column 1 should have both edit and bash
        col1 = props[1]
        total = sum(col1.values())
        assert abs(total - 1.0) < 0.01  # proportions sum to ~1.0
        # Should have at least 2 distinct values
        assert len(col1) >= 2


# --- Test 3: bifurcation significance ---

class TestBifurcationSignificance:
    def test_significant_split(self):
        """Clear pass/fail split should give significant p-value."""
        runs = [
            _make_run("p1", "task1", True, ["read", "edit", "test"]),
            _make_run("p2", "task1", True, ["read", "edit", "test"]),
            _make_run("p3", "task1", True, ["read", "edit", "test"]),
            _make_run("f1", "task1", False, ["read", "bash", "bash"]),
            _make_run("f2", "task1", False, ["read", "bash", "bash"]),
            _make_run("f3", "task1", False, ["read", "bash", "bash"]),
        ]
        alignment, Z = _make_alignment_and_linkage(runs)

        tree = build_stream_tree(alignment, runs, Z)

        # Find a bifurcation with a significant split
        significant_bifs = [
            b for b in tree["bifurcations"].values()
            if b["significant"]
        ]
        assert len(significant_bifs) >= 1

        best = min(significant_bifs, key=lambda b: b["p_value"] or 1.0)
        assert best["p_value"] is not None
        assert best["p_value"] < 0.2  # 3v3 Fisher gives p~0.1
        assert best["significant"] is True

        # Success rates should differ
        assert best["left_success_rate"] != best["right_success_rate"]


# --- Test 4: build_run_meta ---

class TestBuildRunMeta:
    def test_fields(self):
        run = _make_run("r1", "task1", True, ["read", "edit", "test"])
        run.model = "gpt-4"
        run.harness = "swe-bench"

        meta = build_run_meta(run)

        assert meta["run_id"] == "r1"
        assert meta["success"] is True
        assert meta["step_count"] == 3
        assert isinstance(meta["trajectory"], str)
        assert len(meta["trajectory"]) > 0
        assert isinstance(meta["phase_mix"], str)
        assert meta["harness"] == "swe-bench"
        assert meta["model"] == "gpt-4"

    def test_failed_run(self):
        run = _make_run("r2", "task1", False, ["read", "bash"])

        meta = build_run_meta(run)

        assert meta["success"] is False
        assert meta["step_count"] == 2


# --- Test 5: build_run_detail ---

class TestBuildRunDetail:
    def test_step_list(self):
        run = Run(
            run_id="r1", task_id="task1",
            steps=[
                Step(idx=0, type="tool", name="read", status="ok",
                     attrs={"file_path": "/src/main.py"}),
                Step(idx=1, type="tool", name="edit", status="ok",
                     attrs={"file_path": "/src/main.py"}),
                Step(idx=2, type="tool", name="test", status="ok"),
            ],
            result=Result(success=True),
        )

        detail = build_run_detail(run)

        assert detail["run_id"] == "r1"
        assert len(detail["steps"]) == 3

        step0 = detail["steps"][0]
        assert step0["idx"] == 0
        assert step0["enriched"] is not None
        assert "read" in step0["enriched"]
        assert step0["phase"] is not None
        assert step0["color"].startswith("#")

    def test_noise_filtered(self):
        """Noise steps (error_observation) should be skipped."""
        run = Run(
            run_id="r1", task_id="task1",
            steps=[
                Step(idx=0, type="tool", name="read", status="ok"),
                Step(idx=1, type="system", name="error_observation", status="ok"),
                Step(idx=2, type="tool", name="edit", status="ok"),
            ],
            result=Result(success=True),
        )

        detail = build_run_detail(run)

        # error_observation is noise — should be filtered
        assert len(detail["steps"]) == 2
        enriched_names = [s["enriched"] for s in detail["steps"]]
        assert all("error" not in e for e in enriched_names)

    def test_colors_from_step_colors(self):
        """Colors should come from STEP_COLORS dict."""
        from moirai.viz.html import STEP_COLORS

        run = Run(
            run_id="r1", task_id="task1",
            steps=[
                Step(idx=0, type="tool", name="read", status="ok",
                     attrs={"file_path": "/src/main.py"}),
            ],
            result=Result(success=True),
        )

        detail = build_run_detail(run)
        step0 = detail["steps"][0]
        # read(source) should map to its color in STEP_COLORS
        assert step0["color"] == STEP_COLORS.get(step0["enriched"], "#6e7681")

    def test_detail_string(self):
        """Detail should extract filename from file_path attr."""
        run = Run(
            run_id="r1", task_id="task1",
            steps=[
                Step(idx=0, type="tool", name="read", status="ok",
                     attrs={"file_path": "/path/to/foo.py"}),
                Step(idx=1, type="tool", name="bash", status="ok",
                     attrs={"command": "python test.py --verbose"}),
            ],
            result=Result(success=True),
        )

        detail = build_run_detail(run)

        # file_path -> just filename
        assert "foo.py" in detail["steps"][0]["detail"]
        # command -> truncated
        assert "python test.py" in detail["steps"][1]["detail"]
