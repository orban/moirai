"""Tests for narrative generation with reasoning support."""

from moirai.schema import Step, Result, Run, Alignment, DivergencePoint, GAP
from moirai.analyze.narrate import narrate_task


def _make_run(run_id: str, names: list[str], success: bool, outputs: dict[int, dict] | None = None) -> Run:
    steps = []
    for i, n in enumerate(names):
        output = (outputs or {}).get(i, {})
        steps.append(Step(idx=i, type="tool", name=n, output=output))
    return Run(run_id=run_id, task_id="t1", steps=steps, result=Result(success=success))


class TestNarrateWithReasoning:
    def test_reasoning_populated_when_present(self):
        """BranchExample.reasoning is populated from fork step output."""
        runs = [
            _make_run("pass1", ["read", "edit"], True, {1: {"reasoning": "I should edit this"}}),
            _make_run("fail1", ["read", "bash"], False, {1: {"reasoning": "Let me run it"}}),
        ]
        alignment = Alignment(
            run_ids=["pass1", "fail1"],
            matrix=[["read", "edit"], ["read", "bash"]],
            level="name",
        )
        point = DivergencePoint(
            column=1,
            value_counts={"edit": 1, "bash": 1},
            entropy=1.0,
            success_by_value={"edit": 1.0, "bash": 0.0},
            p_value=0.1,
        )

        findings = narrate_task("t1", runs, alignment, [point])

        assert len(findings) == 1
        assert len(findings[0].branches) == 2
        # Best branch (edit, 100%) should have reasoning
        edit_branch = next(b for b in findings[0].branches if b.value == "edit")
        assert edit_branch.reasoning == "I should edit this"
        # Worst branch (bash, 0%) should have reasoning
        bash_branch = next(b for b in findings[0].branches if b.value == "bash")
        assert bash_branch.reasoning == "Let me run it"

    def test_reasoning_none_when_absent(self):
        """BranchExample.reasoning is None when step has no output.reasoning."""
        runs = [
            _make_run("pass1", ["read", "edit"], True),
            _make_run("fail1", ["read", "bash"], False),
        ]
        alignment = Alignment(
            run_ids=["pass1", "fail1"],
            matrix=[["read", "edit"], ["read", "bash"]],
            level="name",
        )
        point = DivergencePoint(
            column=1,
            value_counts={"edit": 1, "bash": 1},
            entropy=1.0,
            success_by_value={"edit": 1.0, "bash": 0.0},
            p_value=0.1,
        )

        findings = narrate_task("t1", runs, alignment, [point])

        assert len(findings) == 1
        for branch in findings[0].branches:
            assert branch.reasoning is None
