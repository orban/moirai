"""Tests for HTML visualization."""

from moirai.schema import Step, Result, Run, Alignment
from moirai.viz.html import _build_dendrogram_heatmap_svg, _scipy_coords_to_svg, _build_data
from moirai.analyze.splits import find_split_divergences


def _make_run(run_id: str, names: list[str], success: bool) -> Run:
    steps = [Step(idx=i, type="tool", name=n) for i, n in enumerate(names)]
    return Run(run_id=run_id, task_id="t1", steps=steps, result=Result(success=success))


class TestBuildDendrogramHeatmapSvg:
    def test_three_runs_has_dendrogram(self):
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
        svg = _build_dendrogram_heatmap_svg(alignment, runs, splits, dendro)

        assert "<path" in svg
        assert "#3fb950" in svg  # pass color
        assert "#f85149" in svg  # fail color

    def test_one_run_no_dendrogram(self):
        runs = [_make_run("r1", ["read", "edit"], True)]
        alignment = Alignment(run_ids=["r1"], matrix=[["read", "edit"]], level="name")

        svg = _build_dendrogram_heatmap_svg(alignment, runs, [], {})

        assert "<path" not in svg
        assert "<rect" in svg

    def test_identical_trajectories_no_crash(self):
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
        svg = _build_dendrogram_heatmap_svg(alignment, runs, splits, dendro)
        assert "<svg" in svg


class TestBuildData:
    def test_builds_json_serializable_data(self):
        """The data payload should be JSON-serializable."""
        import json
        runs = [
            _make_run("r1", ["read", "edit", "test"], True),
            _make_run("r2", ["read", "edit", "bash"], False),
        ]
        data = _build_data(runs, None)
        # Should not raise
        json.dumps(data, default=str)
        assert data["stats"]["total_runs"] == 2
        assert data["stats"]["pass_rate"] == "50%"


class TestScipyCoordsToSvg:
    def test_basic_brackets(self):
        icoord = [[5.0, 5.0, 15.0, 15.0], [10.0, 10.0, 25.0, 25.0]]
        dcoord = [[0.0, 0.5, 0.5, 0.0], [0.0, 1.0, 1.0, 0.0]]
        paths = _scipy_coords_to_svg(icoord, dcoord, cell_h=10, dendro_w=100)
        assert len(paths) == 2
        assert all("<path" in p for p in paths)

    def test_zero_max_distance(self):
        icoord = [[5.0, 5.0, 15.0, 15.0]]
        dcoord = [[0.0, 0.0, 0.0, 0.0]]
        paths = _scipy_coords_to_svg(icoord, dcoord, cell_h=10, dendro_w=100)
        assert len(paths) == 1

    def test_empty_input(self):
        assert _scipy_coords_to_svg([], [], cell_h=10, dendro_w=100) == []
