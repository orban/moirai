"""Tests for file loading."""

import json
from pathlib import Path

import pytest

from moirai.load import load_runs, validate_file


def _write_run(tmp_path: Path, filename: str, data: dict) -> Path:
    p = tmp_path / filename
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _minimal_run(run_id: str = "r1", **overrides) -> dict:
    base = {
        "run_id": run_id,
        "task_id": "t1",
        "steps": [{"idx": 0, "type": "llm", "name": "plan"}],
        "result": {"success": True},
    }
    base.update(overrides)
    return base


class TestValidateFile:
    def test_valid_file(self, tmp_path):
        p = _write_run(tmp_path, "run.json", _minimal_run())
        result = validate_file(p)
        assert result.passed
        assert result.errors == []

    def test_invalid_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not json", encoding="utf-8")
        result = validate_file(p)
        assert not result.passed
        assert any("invalid JSON" in e for e in result.errors)

    def test_json_array_rejected(self, tmp_path):
        p = tmp_path / "array.json"
        p.write_text(json.dumps([_minimal_run()]), encoding="utf-8")
        result = validate_file(p)
        assert not result.passed
        assert any("JSON array" in e for e in result.errors)

    def test_missing_required_field(self, tmp_path):
        p = _write_run(tmp_path, "run.json", {"run_id": "r1"})
        result = validate_file(p)
        assert not result.passed
        assert any("task_id" in e for e in result.errors)

    def test_strict_mode_warnings_become_errors(self, tmp_path):
        data = _minimal_run()
        data["steps"] = [{"type": "banana", "name": "x"}]
        p = _write_run(tmp_path, "run.json", data)
        result = validate_file(p, strict=True)
        assert not result.passed


class TestLoadRuns:
    def test_single_file(self, tmp_path):
        _write_run(tmp_path, "run.json", _minimal_run())
        runs, warnings = load_runs(tmp_path / "run.json")
        assert len(runs) == 1
        assert runs[0].run_id == "r1"

    def test_directory(self, tmp_path):
        _write_run(tmp_path, "a.json", _minimal_run("r1"))
        _write_run(tmp_path, "b.json", _minimal_run("r2"))
        runs, warnings = load_runs(tmp_path)
        assert len(runs) == 2

    def test_nested_directory(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        _write_run(tmp_path, "a.json", _minimal_run("r1"))
        _write_run(sub, "b.json", _minimal_run("r2"))
        runs, _ = load_runs(tmp_path)
        assert len(runs) == 2

    def test_invalid_json_skipped(self, tmp_path):
        _write_run(tmp_path, "good.json", _minimal_run("r1"))
        (tmp_path / "bad.json").write_text("{nope", encoding="utf-8")
        runs, warnings = load_runs(tmp_path)
        assert len(runs) == 1
        assert any("skipped" in w for w in warnings)

    def test_json_array_skipped(self, tmp_path):
        _write_run(tmp_path, "good.json", _minimal_run("r1"))
        (tmp_path / "arr.json").write_text(json.dumps([_minimal_run("r2")]), encoding="utf-8")
        runs, warnings = load_runs(tmp_path)
        assert len(runs) == 1
        assert any("JSON array" in w for w in warnings)

    def test_duplicate_run_id_skipped(self, tmp_path):
        _write_run(tmp_path, "a.json", _minimal_run("r1"))
        _write_run(tmp_path, "b.json", _minimal_run("r1"))
        runs, warnings = load_runs(tmp_path)
        assert len(runs) == 1
        assert any("duplicate run_id" in w for w in warnings)

    def test_empty_directory(self, tmp_path):
        runs, warnings = load_runs(tmp_path)
        assert runs == []
        assert any("no JSON files" in w for w in warnings)

    def test_path_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_runs("/nonexistent/path")

    def test_strict_rejects_invalid(self, tmp_path):
        (tmp_path / "bad.json").write_text("{nope", encoding="utf-8")
        with pytest.raises(ValueError):
            load_runs(tmp_path, strict=True)
