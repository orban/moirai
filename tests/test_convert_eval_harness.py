"""Tests for eval-harness converter with transcript enrichment."""

import json
from pathlib import Path

import pytest

# Add scripts dir to path so we can import the converter
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from convert_eval_harness import parse_transcript, convert_trial, _merge_transcript, parse_log_steps


def _write_jsonl(path: Path, lines: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(l) for l in lines), encoding="utf-8")


def _make_assistant_text(text: str) -> dict:
    return {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}}


def _make_assistant_thinking(thinking: str) -> dict:
    return {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": thinking}]}}


def _make_assistant_tool_use(name: str, tool_use_id: str, input_data: dict | None = None) -> dict:
    return {"type": "assistant", "message": {"role": "assistant", "content": [
        {"type": "tool_use", "name": name, "id": tool_use_id, "input": input_data or {}}
    ]}}


def _make_user_tool_result(tool_use_id: str, content: str) -> dict:
    return {"type": "user", "message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    ]}}


class TestParseTranscript:
    def test_multi_tool_use_each_gets_own_reasoning(self, tmp_path):
        """Each tool_use gets the reasoning from its immediately preceding text/thinking block."""
        jsonl_path = tmp_path / "transcript.jsonl"
        _write_jsonl(jsonl_path, [
            _make_assistant_text("Let me read the source file."),
            _make_assistant_tool_use("Read", "tu_1", {"file_path": "/src/main.py"}),
            _make_user_tool_result("tu_1", "def main(): pass"),
            _make_assistant_text("Now I should search for tests."),
            _make_assistant_tool_use("Glob", "tu_2", {"pattern": "tests/**"}),
            _make_user_tool_result("tu_2", "tests/test_main.py"),
            _make_assistant_thinking("I need to edit the source."),
            _make_assistant_tool_use("Edit", "tu_3", {"file_path": "/src/main.py"}),
            _make_user_tool_result("tu_3", "File edited"),
        ])

        result = parse_transcript(jsonl_path)

        assert len(result) == 3
        assert result[0]["reasoning"] == "Let me read the source file."
        assert result[0]["tool_name"] == "Read"
        assert result[0]["result"] == "def main(): pass"
        assert result[1]["reasoning"] == "Now I should search for tests."
        assert result[1]["tool_name"] == "Glob"
        assert result[2]["reasoning"] == "I need to edit the source."
        assert result[2]["tool_name"] == "Edit"

    def test_thinking_preferred_over_text(self, tmp_path):
        """When a thinking block is present, text doesn't override it."""
        jsonl_path = tmp_path / "transcript.jsonl"
        _write_jsonl(jsonl_path, [
            _make_assistant_thinking("Deep thought about the problem."),
            _make_assistant_tool_use("Read", "tu_1"),
            _make_user_tool_result("tu_1", "file contents"),
        ])

        result = parse_transcript(jsonl_path)
        assert result[0]["reasoning"] == "Deep thought about the problem."

    def test_tool_result_matched_by_id(self, tmp_path):
        """tool_result blocks are matched to tool_use by tool_use_id."""
        jsonl_path = tmp_path / "transcript.jsonl"
        _write_jsonl(jsonl_path, [
            _make_assistant_tool_use("Read", "tu_alpha"),
            _make_assistant_tool_use("Glob", "tu_beta"),
            _make_user_tool_result("tu_beta", "beta result"),
            _make_user_tool_result("tu_alpha", "alpha result"),
        ])

        result = parse_transcript(jsonl_path)
        assert result[0]["tool_name"] == "Read"
        assert result[0]["result"] == "alpha result"
        assert result[1]["tool_name"] == "Glob"
        assert result[1]["result"] == "beta result"

    def test_malformed_lines_skipped(self, tmp_path):
        """Malformed JSONL lines don't crash the parser."""
        jsonl_path = tmp_path / "transcript.jsonl"
        jsonl_path.write_text(
            '{"not valid json\n'
            + json.dumps(_make_assistant_text("reasoning")) + "\n"
            + json.dumps(_make_assistant_tool_use("Read", "tu_1")) + "\n",
            encoding="utf-8",
        )

        result = parse_transcript(jsonl_path)
        assert len(result) == 1
        assert result[0]["reasoning"] == "reasoning"

    def test_reasoning_truncated(self, tmp_path):
        """Reasoning is truncated to 500 chars."""
        long_text = "x" * 1000
        jsonl_path = tmp_path / "transcript.jsonl"
        _write_jsonl(jsonl_path, [
            _make_assistant_text(long_text),
            _make_assistant_tool_use("Read", "tu_1"),
        ])

        result = parse_transcript(jsonl_path)
        assert len(result[0]["reasoning"]) == 500


class TestConvertTrialMerge:
    def test_merge_reasoning_onto_log_steps(self, tmp_path):
        """Transcript reasoning merges into log-parsed steps by sequential position."""
        # Create trial JSON
        trial_path = tmp_path / "trial.json"
        trial_path.write_text(json.dumps({
            "task_id": "test_task-123",
            "condition": "none",
            "rep": 0,
            "success": True,
        }))

        # Create log file
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        log_path = log_dir / "trial.log"
        log_path.write_text(
            "[tool] Read /src/main.py\n"
            "[tool] Edit /src/main.py\n"
            "[result] 2 turns, $0.10\n"
        )

        # Create transcript
        transcripts_dir = tmp_path / "projects"
        # We need a workspace path in the file_path attrs to trigger matching
        # For this test, just call _merge_transcript directly
        steps = parse_log_steps(log_path)
        transcript = [
            {"reasoning": "Let me read the file.", "result": "file contents", "tool_name": "Read", "tool_use_id": "tu_1"},
            {"reasoning": "Now I'll edit it.", "result": "edited", "tool_name": "Edit", "tool_use_id": "tu_2"},
        ]
        _merge_transcript(steps, transcript)

        # Tool steps should have reasoning, result step should not
        assert steps[0]["output"]["reasoning"] == "Let me read the file."
        assert steps[0]["output"]["result"] == "file contents"
        assert steps[1]["output"]["reasoning"] == "Now I'll edit it."
        assert "output" not in steps[2] or "reasoning" not in steps[2].get("output", {})

    def test_merge_stops_on_name_mismatch(self, tmp_path):
        """When tool names don't match, merge stops at that point."""
        log_path = tmp_path / "test.log"
        log_path.write_text(
            "[tool] Read /src/main.py\n"
            "[tool] Bash: pytest\n"
            "[tool] Edit /src/main.py\n"
        )
        steps = parse_log_steps(log_path)

        transcript = [
            {"reasoning": "Reading file.", "result": None, "tool_name": "Read", "tool_use_id": "tu_1"},
            {"reasoning": "Searching.", "result": None, "tool_name": "Glob", "tool_use_id": "tu_2"},  # mismatch: Glob vs test
            {"reasoning": "Editing.", "result": None, "tool_name": "Edit", "tool_use_id": "tu_3"},
        ]
        _merge_transcript(steps, transcript)

        # First step should be enriched, rest should not (merge stopped)
        assert steps[0]["output"]["reasoning"] == "Reading file."
        assert "output" not in steps[1] or "reasoning" not in steps[1].get("output", {})
        assert "output" not in steps[2] or "reasoning" not in steps[2].get("output", {})


class TestConvertTrialFallback:
    def test_no_transcript_produces_valid_run(self, tmp_path):
        """Without transcripts, conversion still works as before."""
        trial_path = tmp_path / "trial.json"
        trial_path.write_text(json.dumps({
            "task_id": "test_task-123",
            "condition": "none",
            "rep": 0,
            "success": True,
        }))

        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "trial.log").write_text("[tool] Read /src/main.py\n")

        result = convert_trial(trial_path, log_dir, transcripts_dir=None)

        assert result is not None
        assert result["run_id"] == "test_task-123-none-r0"
        assert len(result["steps"]) == 1
        assert result["steps"][0]["name"] == "read"
        assert "reasoning" not in result["steps"][0].get("output", {})
