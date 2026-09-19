"""End-to-end golden test for the moirai branch pipeline.

Synthetic dataset with a known divergence point:
- 5 runs, same task
- 3 pass: read → read(test) → edit → test(pass)
- 2 fail: read → edit → test(fail) → edit → test(fail)
- Known split at position 1: read(test) vs edit
"""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from moirai.cli import app
from moirai.schema import Step, Result, Run
from moirai.analyze.align import align_runs
from moirai.analyze.divergence import find_divergence_points, summarize_point


def _make_run(run_id: str, names: list[str], success: bool) -> Run:
    steps = [Step(idx=i, type="tool", name=name, attrs={}) for i, name in enumerate(names)]
    return Run(run_id=run_id, task_id="task_golden", steps=steps, result=Result(success=success))


PASS_STEPS = ["read", "read(test)", "edit", "test(pass)"]
FAIL_STEPS = ["read", "edit", "test(fail)", "edit", "test(fail)"]

GOLDEN_RUNS = [
    _make_run("p1", PASS_STEPS, True),
    _make_run("p2", PASS_STEPS, True),
    _make_run("p3", PASS_STEPS, True),
    _make_run("f1", FAIL_STEPS, False),
    _make_run("f2", FAIL_STEPS, False),
]


class TestGoldenAlignment:
    def test_alignment_detects_divergence(self):
        """Alignment + divergence detection finds the known split."""
        alignment = align_runs(GOLDEN_RUNS, level="name")
        points, _ = find_divergence_points(
            alignment, GOLDEN_RUNS, min_branch_size=2, q_threshold=1.0,
        )
        assert len(points) >= 1

        # At least one point should separate pass from fail
        found_separating = False
        for p in points:
            rates = [r for r in p.success_by_value.values() if r is not None]
            if len(rates) >= 2 and max(rates) - min(rates) >= 0.5:
                found_separating = True
                break
        assert found_separating, f"No separating divergence point found. Points: {points}"

    def test_summarize_point_produces_text(self):
        """summarize_point returns non-empty string for detected points."""
        alignment = align_runs(GOLDEN_RUNS, level="name")
        points, _ = find_divergence_points(
            alignment, GOLDEN_RUNS, min_branch_size=1, q_threshold=1.0,
        )
        for p in points:
            s = summarize_point(p)
            assert isinstance(s, str)
            assert len(s) > 10
            # New summaries take a stance — check they're substantive
            assert len(s) > 20, f"Summary too short: {s}"


class TestGoldenCLI:
    def _write_golden_traces(self, tmp_path: Path) -> Path:
        """Write golden run traces as JSON files."""
        for run in GOLDEN_RUNS:
            data = {
                "run_id": run.run_id,
                "task_id": run.task_id,
                "steps": [
                    {"idx": s.idx, "type": s.type, "name": s.name}
                    for s in run.steps
                ],
                "result": {"success": run.result.success},
            }
            (tmp_path / f"{run.run_id}.json").write_text(json.dumps(data))
        return tmp_path

    def test_branch_terminal_output(self, tmp_path):
        """Branch command runs end-to-end and shows branch cards."""
        self._write_golden_traces(tmp_path)
        runner = CliRunner()
        result = runner.invoke(app, ["branch", str(tmp_path)])
        assert result.exit_code == 0
        assert "task_golden" in result.output
        assert "Branch 1" in result.output

    def test_branch_json_output(self, tmp_path):
        """--json flag produces valid JSON with branch points."""
        self._write_golden_traces(tmp_path)
        json_path = tmp_path / "report.json"
        runner = CliRunner()
        result = runner.invoke(app, ["branch", str(tmp_path), "--json", str(json_path)])
        assert result.exit_code == 0
        assert json_path.exists()

        data = json.loads(json_path.read_text())
        assert "tasks" in data
        assert len(data["tasks"]) == 1

        task = data["tasks"][0]
        assert task["task_id"] == "task_golden"
        assert task["n_runs"] == 5
        assert task["n_pass"] == 3
        assert task["n_fail"] == 2
        assert len(task["branch_points"]) >= 1

        bp = task["branch_points"][0]
        assert "position" in bp
        assert "summary" in bp
        assert isinstance(bp["summary"], str)
        assert len(bp["summary"]) > 10
        assert "variants" in bp
        assert len(bp["variants"]) >= 2

        # Each variant has required fields
        for v in bp["variants"]:
            assert "value" in v
            assert "n_runs" in v
            assert "pass_rate" in v

    def test_branch_html_output(self, tmp_path):
        """--html flag produces HTML file with branch cards."""
        self._write_golden_traces(tmp_path)
        html_path = tmp_path / "report.html"
        runner = CliRunner()
        result = runner.invoke(app, ["branch", str(tmp_path), "--html", str(html_path)])
        assert result.exit_code == 0
        assert html_path.exists()

        html = html_path.read_text()
        assert "branch_cards" in html
        assert "moirai" in html.lower()

    def test_branch_json_and_html_together(self, tmp_path):
        """Both --json and --html can be used simultaneously."""
        self._write_golden_traces(tmp_path)
        json_path = tmp_path / "report.json"
        html_path = tmp_path / "report.html"
        runner = CliRunner()
        result = runner.invoke(app, [
            "branch", str(tmp_path),
            "--json", str(json_path),
            "--html", str(html_path),
        ])
        assert result.exit_code == 0
        assert json_path.exists()
        assert html_path.exists()
