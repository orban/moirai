"""Tests for HTML visualization: dendrogram + heatmap."""

import numpy as np
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram, linkage

from moirai.schema import Step, Result, Run, Alignment, GAP
from moirai.viz.html import _build_dendrogram_heatmap, _scipy_coords_to_svg
from moirai.analyze.splits import find_split_divergences


def _make_run(run_id: str, names: list[str], success: bool) -> Run:
    steps = [Step(idx=i, type="tool", name=n) for i, n in enumerate(names)]
    return Run(run_id=run_id, task_id="t1", steps=steps, result=Result(success=success))


class TestBuildDendrogramHeatmap:
    def test_three_runs_has_dendrogram(self):
        """3+ runs should produce SVG with dendrogram paths and outcome strip."""
        runs = [
            _make_run("r1", ["read", "edit", "test"], True),
            _make_run("r2", ["read", "edit", "bash"], False),
            _make_run("r3", ["read", "edit", "test"], True),
        ]
        alignment = Alignment(
            run_ids=["r1", "r2", "r3"],
            matrix=[["read", "edit", "test"], ["read", "edit", "bash"], ["read", "edit", "test"]],
            level="name",
        )

        splits, Z, dendro = find_split_divergences(alignment, runs)
        svg = _build_dendrogram_heatmap(alignment, runs, splits, Z, dendro)

        assert "<path" in svg  # dendrogram bracket paths
        assert "#3fb950" in svg  # green outcome strip (pass)
        assert "#f85149" in svg  # red outcome strip (fail)
        assert "showTip" in svg  # interactive tooltips on cells

    def test_one_run_no_dendrogram(self):
        """1 run should render heatmap without dendrogram paths."""
        runs = [_make_run("r1", ["read", "edit"], True)]
        alignment = Alignment(
            run_ids=["r1"],
            matrix=[["read", "edit"]],
            level="name",
        )

        svg = _build_dendrogram_heatmap(alignment, runs, [], None, {})

        assert "<path" not in svg  # no dendrogram
        assert "<rect" in svg  # heatmap cells still present
        assert "r1" in svg  # run label

    def test_identical_trajectories_no_crash(self):
        """All identical trajectories (zero distances) should not crash."""
        runs = [
            _make_run("r1", ["read", "edit", "test"], True),
            _make_run("r2", ["read", "edit", "test"], True),
            _make_run("r3", ["read", "edit", "test"], False),
        ]
        alignment = Alignment(
            run_ids=["r1", "r2", "r3"],
            matrix=[["read", "edit", "test"]] * 3,
            level="name",
        )

        splits, Z, dendro = find_split_divergences(alignment, runs)
        svg = _build_dendrogram_heatmap(alignment, runs, splits, Z, dendro)
        assert "<svg" in svg


class TestScipyCoordsToSvg:
    def test_basic_brackets(self):
        """Known 3-item dendrogram should produce 2 bracket paths."""
        icoord = [[5.0, 5.0, 15.0, 15.0], [10.0, 10.0, 25.0, 25.0]]
        dcoord = [[0.0, 0.5, 0.5, 0.0], [0.0, 1.0, 1.0, 0.0]]

        paths = _scipy_coords_to_svg(icoord, dcoord, cell_h=10, dendro_w=100)

        assert len(paths) == 2
        assert all("<path" in p for p in paths)
        assert all('fill="none"' in p for p in paths)

    def test_zero_max_distance(self):
        """All-zero distances should not crash (division by zero guard)."""
        icoord = [[5.0, 5.0, 15.0, 15.0]]
        dcoord = [[0.0, 0.0, 0.0, 0.0]]

        paths = _scipy_coords_to_svg(icoord, dcoord, cell_h=10, dendro_w=100)
        assert len(paths) == 1

    def test_empty_input(self):
        """Empty icoord/dcoord returns empty list."""
        assert _scipy_coords_to_svg([], [], cell_h=10, dendro_w=100) == []
