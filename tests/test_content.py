"""Tests for content-aware analysis."""

from moirai.schema import Step, Result, Run, Alignment, DivergencePoint, GAP
from moirai.analyze.content import (
    select_task_groups, sample_runs, build_prompt, parse_response, invoke_llm,
)


def _make_run(run_id, task_id, names, success, output=None):
    """Make a Run with named steps and optional output dicts."""
    steps = []
    for i, name in enumerate(names):
        out = {}
        if output and i < len(output):
            out = output[i] or {}
        steps.append(Step(idx=i, type="tool", name=name, output=out))
    return Run(
        run_id=run_id,
        task_id=task_id,
        steps=steps,
        result=Result(success=success),
    )


class TestSelectTaskGroups:
    def test_mixed_outcomes_qualify(self):
        """3 pass + 3 fail -> qualifies."""
        runs = [
            _make_run(f"p{i}", "t1", ["read", "edit"], True)
            for i in range(3)
        ] + [
            _make_run(f"f{i}", "t1", ["read", "error"], False)
            for i in range(3)
        ]
        groups, skipped = select_task_groups(runs, min_runs=5)
        assert "t1" in groups
        assert len(groups["t1"]) == 6
        assert "t1" not in skipped

    def test_all_same_outcome_skipped(self):
        """All-pass -> skipped with reason."""
        runs = [
            _make_run(f"r{i}", "t1", ["read", "edit"], True)
            for i in range(6)
        ]
        groups, skipped = select_task_groups(runs, min_runs=5)
        assert "t1" not in groups
        assert skipped["t1"] == "all-pass"

    def test_too_few_runs_skipped(self):
        """3 runs -> skipped for too few."""
        runs = [
            _make_run(f"r{i}", "t1", ["read"], True if i < 2 else False)
            for i in range(3)
        ]
        groups, skipped = select_task_groups(runs, min_runs=5)
        assert "t1" not in groups
        assert skipped["t1"] == "too few runs"

    def test_task_filter_restricts(self):
        """--task filters to one group."""
        runs = [
            _make_run(f"a{i}", "t1", ["read"], True if i < 3 else False)
            for i in range(6)
        ] + [
            _make_run(f"b{i}", "t2", ["edit"], True if i < 3 else False)
            for i in range(6)
        ]
        groups, skipped = select_task_groups(runs, min_runs=5, task_filter="t1")
        assert "t1" in groups
        assert "t2" not in groups
        assert "t2" not in skipped

    def test_task_not_found(self):
        """--task with nonexistent ID."""
        runs = [
            _make_run(f"r{i}", "t1", ["read"], True if i < 3 else False)
            for i in range(6)
        ]
        groups, skipped = select_task_groups(runs, task_filter="nope")
        assert groups == {}
        assert skipped == {"nope": "not found"}


class TestSampleRuns:
    def _make_alignment(self, run_ids, matrix):
        return Alignment(run_ids=run_ids, matrix=matrix, level="name")

    def test_picks_near_and_far(self):
        """Returns runs from both ends of distance from consensus."""
        # Consensus will be ["read", "edit", "test"] (majority)
        runs = [
            _make_run("p1", "t1", ["read", "edit", "test"], True),
            _make_run("p2", "t1", ["read", "edit", "test"], True),
            _make_run("f1", "t1", ["bash", "bash", "bash"], False),
            _make_run("f2", "t1", ["read", "edit", "test"], False),
        ]
        alignment = self._make_alignment(
            ["p1", "p2", "f1", "f2"],
            [
                ["read", "edit", "test"],   # p1: 0 mismatches
                ["read", "edit", "test"],   # p2: 0 mismatches
                ["bash", "bash", "bash"],   # f1: 3 mismatches (far)
                ["read", "edit", "test"],   # f2: 0 mismatches (near)
            ],
        )
        sampled = sample_runs(runs, alignment, seed=42)
        sampled_ids = {r.run_id for r in sampled}
        # Should have at least one pass and one fail
        assert any(r.result.success for r in sampled)
        assert any(not r.result.success for r in sampled)
        # Should include the far-divergent fail run f1
        assert "f1" in sampled_ids

    def test_deterministic_with_seed(self):
        """Same seed = same result."""
        runs = [
            _make_run(f"p{i}", "t1", ["read", "edit"], True)
            for i in range(5)
        ] + [
            _make_run(f"f{i}", "t1", ["bash", "error"], False)
            for i in range(5)
        ]
        alignment = self._make_alignment(
            [r.run_id for r in runs],
            [["read", "edit"]] * 5 + [["bash", "error"]] * 5,
        )
        result1 = sample_runs(runs, alignment, seed=99)
        result2 = sample_runs(runs, alignment, seed=99)
        assert [r.run_id for r in result1] == [r.run_id for r in result2]

    def test_single_run_class(self):
        """Class with 1 run doesn't crash."""
        runs = [
            _make_run("p1", "t1", ["read"], True),
            _make_run("f1", "t1", ["bash"], False),
        ]
        alignment = self._make_alignment(
            ["p1", "f1"],
            [["read"], ["bash"]],
        )
        sampled = sample_runs(runs, alignment, seed=42)
        assert len(sampled) == 2  # one near per class, no far because only 1 run each


class TestBuildPrompt:
    def _make_alignment(self, run_ids, matrix):
        return Alignment(run_ids=run_ids, matrix=matrix, level="name")

    def test_includes_context_and_columns(self):
        """task_id, counts, column info all appear in output."""
        runs = [
            _make_run("p1", "task-42", ["read", "edit"], True,
                       output=[{"reasoning": "looks right"}, {"result": "ok"}]),
            _make_run("f1", "task-42", ["read", "bash"], False,
                       output=[{"reasoning": "trying bash"}, {"result": "error"}]),
        ]
        alignment = self._make_alignment(
            ["p1", "f1"],
            [["read", "edit"], ["read", "bash"]],
        )
        dp = DivergencePoint(
            column=1,
            value_counts={"edit": 1, "bash": 1},
            entropy=1.0,
            success_by_value={"edit": 1.0, "bash": 0.0},
        )
        prompt = build_prompt("task-42", runs, alignment, [dp], runs)
        assert "task-42" in prompt
        assert "1 pass" in prompt
        assert "1 fail" in prompt
        assert "column 1" in prompt
        assert "edit: 1 runs" in prompt
        assert "bash: 1 runs" in prompt

    def test_truncates_long_content(self):
        """2000-char content gets truncated."""
        long_text = "x" * 2000
        runs = [
            _make_run("p1", "t1", ["read"], True,
                       output=[{"reasoning": long_text}]),
        ]
        alignment = self._make_alignment(["p1"], [["read"]])
        dp = DivergencePoint(
            column=0,
            value_counts={"read": 1},
            entropy=0.0,
            success_by_value={"read": 1.0},
        )
        prompt = build_prompt("t1", runs, alignment, [dp], runs)
        # The reasoning field is truncated to 800 chars by _truncate
        assert long_text not in prompt
        assert "..." in prompt


class TestParseResponse:
    def test_valid_json(self):
        """Well-formed JSON parses correctly."""
        raw = '''{
          "findings": [
            {
              "category": "wrong_file",
              "column": 3,
              "description": "Agent edited the wrong file",
              "evidence": "p1 edited foo.py, f1 edited bar.py",
              "pass_runs": ["p1"],
              "fail_runs": ["f1"]
            }
          ],
          "summary": "Failing runs edit the wrong file.",
          "confidence": "high"
        }'''
        findings, summary, confidence = parse_response(raw)
        assert len(findings) == 1
        assert findings[0].category == "wrong_file"
        assert findings[0].column == 3
        assert findings[0].description == "Agent edited the wrong file"
        assert findings[0].pass_runs == ["p1"]
        assert findings[0].fail_runs == ["f1"]
        assert summary == "Failing runs edit the wrong file."
        assert confidence == "high"

    def test_strips_markdown_fences(self):
        """```json ... ``` stripped."""
        raw = '''Here is the analysis:
```json
{
  "findings": [],
  "summary": "No issues found.",
  "confidence": "low"
}
```
Done.'''
        findings, summary, confidence = parse_response(raw)
        assert findings == []
        assert summary == "No issues found."
        assert confidence == "low"

    def test_malformed_json_retry(self):
        '''"some text { valid json } trailing" works via brace extraction.'''
        raw = 'The analysis shows: {"findings": [], "summary": "ok", "confidence": "medium"} end of response'
        findings, summary, confidence = parse_response(raw)
        assert summary == "ok"
        assert confidence == "medium"

    def test_normalizes_known_categories(self):
        '''"wrong_file" stays as-is (known category preserved).'''
        raw = '''{
          "findings": [
            {
              "category": "wrong_file",
              "column": 0,
              "description": "test",
              "evidence": "test"
            }
          ],
          "summary": "",
          "confidence": ""
        }'''
        findings, _, _ = parse_response(raw)
        assert findings[0].category == "wrong_file"


class TestInvokeLlm:
    def test_structural_returns_none(self):
        """mode='structural' -> None without subprocess."""
        result = invoke_llm("some prompt", mode="structural")
        assert result is None
