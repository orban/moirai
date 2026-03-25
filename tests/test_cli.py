"""CLI integration tests using Typer's CliRunner."""

import json
from pathlib import Path

from typer.testing import CliRunner

from moirai.cli import app

runner = CliRunner()


def _write_run(tmp_path: Path, filename: str, run_id: str = "r1",
               harness: str = "h1", success: bool = True) -> Path:
    data = {
        "run_id": run_id,
        "task_id": "t1",
        "harness": harness,
        "model": "m1",
        "steps": [
            {"idx": 0, "type": "llm", "name": "plan", "metrics": {"tokens_in": 100, "tokens_out": 50, "latency_ms": 500}},
            {"idx": 1, "type": "tool", "name": "search", "metrics": {"tokens_in": 200, "tokens_out": 80, "latency_ms": 1000}},
        ],
        "result": {"success": success},
    }
    p = tmp_path / filename
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


class TestValidateCommand:
    def test_valid_file_exit_0(self, tmp_path):
        _write_run(tmp_path, "run.json")
        result = runner.invoke(app, ["validate", str(tmp_path / "run.json")])
        assert result.exit_code == 0

    def test_invalid_file_exit_1(self, tmp_path):
        (tmp_path / "bad.json").write_text("{bad", encoding="utf-8")
        result = runner.invoke(app, ["validate", str(tmp_path / "bad.json")])
        assert result.exit_code == 1

    def test_nonexistent_path_exit_1(self):
        result = runner.invoke(app, ["validate", "/nonexistent/path"])
        assert result.exit_code == 1


class TestSummaryCommand:
    def test_basic_summary(self, tmp_path):
        _write_run(tmp_path, "a.json", "r1")
        _write_run(tmp_path, "b.json", "r2")
        result = runner.invoke(app, ["summary", str(tmp_path)])
        assert result.exit_code == 0
        assert "Runs: 2" in result.output

    def test_filtered_summary(self, tmp_path):
        _write_run(tmp_path, "a.json", "r1", harness="baseline")
        _write_run(tmp_path, "b.json", "r2", harness="router")
        result = runner.invoke(app, ["summary", str(tmp_path), "--harness", "baseline"])
        assert result.exit_code == 0
        assert "Runs: 1" in result.output

    def test_no_matches_exit_1(self, tmp_path):
        _write_run(tmp_path, "a.json", "r1", harness="baseline")
        result = runner.invoke(app, ["summary", str(tmp_path), "--harness", "nonexistent"])
        assert result.exit_code == 1


class TestTraceCommand:
    def test_trace_file(self, tmp_path):
        _write_run(tmp_path, "run.json")
        result = runner.invoke(app, ["trace", str(tmp_path / "run.json")])
        assert result.exit_code == 0
        assert "Run: r1" in result.output

    def test_trace_directory_error(self, tmp_path):
        _write_run(tmp_path, "run.json")
        result = runner.invoke(app, ["trace", str(tmp_path)])
        assert result.exit_code == 1
        assert "single run file" in result.output


class TestClustersCommand:
    def test_basic_clusters(self, tmp_path):
        for i in range(3):
            _write_run(tmp_path, f"r{i}.json", f"r{i}")
        result = runner.invoke(app, ["clusters", str(tmp_path)])
        assert result.exit_code == 0
        assert "Cluster" in result.output


class TestBranchCommand:
    def test_basic_branch(self, tmp_path):
        _write_run(tmp_path, "a.json", "r1")
        _write_run(tmp_path, "b.json", "r2", success=False)
        result = runner.invoke(app, ["branch", str(tmp_path)])
        assert result.exit_code == 0


class TestDiffCommand:
    def test_basic_diff(self, tmp_path):
        _write_run(tmp_path, "a.json", "r1", harness="baseline")
        _write_run(tmp_path, "b.json", "r2", harness="router")
        result = runner.invoke(app, ["diff", str(tmp_path), "--a", "harness=baseline", "--b", "harness=router"])
        assert result.exit_code == 0
        assert "Cohort A" in result.output
        assert "Cohort B" in result.output

    def test_missing_flags_exit_2(self, tmp_path):
        _write_run(tmp_path, "a.json", "r1")
        result = runner.invoke(app, ["diff", str(tmp_path)])
        assert result.exit_code == 2

    def test_empty_cohort_exit_1(self, tmp_path):
        _write_run(tmp_path, "a.json", "r1", harness="baseline")
        result = runner.invoke(app, ["diff", str(tmp_path), "--a", "harness=baseline", "--b", "harness=nonexistent"])
        assert result.exit_code == 1
        assert "0 runs" in result.output


class TestExplainCommand:
    def test_explain_found(self, tmp_path):
        _write_run(tmp_path, "a.json", "r1")
        _write_run(tmp_path, "b.json", "r2", success=False)
        _write_run(tmp_path, "c.json", "r3")
        result = runner.invoke(app, ["explain", str(tmp_path), "--run", "r1"])
        assert result.exit_code == 0
        assert "Run:" in result.output
        assert "Trajectory:" in result.output

    def test_explain_prefix_match(self, tmp_path):
        _write_run(tmp_path, "a.json", "long-run-id-001")
        result = runner.invoke(app, ["explain", str(tmp_path), "--run", "long-run"])
        assert result.exit_code == 0
        assert "long-run-id-001" in result.output

    def test_explain_not_found(self, tmp_path):
        _write_run(tmp_path, "a.json", "r1")
        result = runner.invoke(app, ["explain", str(tmp_path), "--run", "nonexistent"])
        assert result.exit_code == 1
        assert "not found" in result.output
