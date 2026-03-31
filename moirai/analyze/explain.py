"""Generate structured pass/fail comparison for LLM analysis.

Takes a task with mixed outcomes, finds the most informative pass/fail pair,
aligns their trajectories, and outputs a rich comparison document with full
context at the divergence point — designed for consumption by an LLM.
"""
from __future__ import annotations

import os
from collections import defaultdict

from moirai.analyze.align import _nw_align
from moirai.compress import step_enriched_name
from moirai.schema import GAP, Run


def explain_task(task_id: str, runs: list[Run]) -> str:
    """Generate a structured comparison document for a task's pass/fail runs.

    Returns a formatted text document ready for LLM analysis.
    """
    pass_runs = [r for r in runs if r.result.success is True]
    fail_runs = [r for r in runs if r.result.success is False]

    if not pass_runs or not fail_runs:
        return f"Task {task_id}: no mixed outcomes ({len(pass_runs)} pass, {len(fail_runs)} fail)"

    # Pick the best pair
    pass_run, fail_run = _pick_pair(pass_runs, fail_runs)

    # Build enriched step lists
    p_steps = _enrich_steps(pass_run)
    f_steps = _enrich_steps(fail_run)

    # Align
    p_names = [s["enriched"] for s in p_steps]
    f_names = [s["enriched"] for s in f_steps]
    aligned_p, aligned_f = _nw_align(p_names, f_names)

    # Find divergence points
    divergences = []
    for i, (a, b) in enumerate(zip(aligned_p, aligned_f)):
        if a != b:
            divergences.append(i)

    first_div = divergences[0] if divergences else None

    # Build the document
    lines = []
    lines.append(f"# Divergence analysis: {task_id}")
    lines.append("")
    lines.append(f"Task: {task_id}")
    lines.append(f"Total runs: {len(runs)} ({len(pass_runs)} pass, {len(fail_runs)} fail)")
    lines.append(f"Pass run: {pass_run.run_id} (harness: {pass_run.harness})")
    lines.append(f"Fail run: {fail_run.run_id} (harness: {fail_run.harness})")
    lines.append("")

    # Narrative
    if first_div is not None:
        n_same = first_div
        p_val = aligned_p[first_div]
        f_val = aligned_f[first_div]
        p_detail = _detail_at(p_steps, aligned_p, first_div)
        f_detail = _detail_at(f_steps, aligned_f, first_div)
        lines.append(f"## Summary")
        lines.append("")
        lines.append(f"After {n_same} identical steps, the runs diverge:")
        lines.append(f"- Pass run: {p_val} ({p_detail})" if p_val != GAP else f"- Pass run: skipped this step")
        lines.append(f"- Fail run: {f_val} ({f_detail})" if f_val != GAP else f"- Fail run: skipped this step")
        lines.append(f"- {len(divergences)} total differences across {len(aligned_p)} aligned positions")
        lines.append("")

    # Aligned trajectory diff
    lines.append("## Aligned trajectory")
    lines.append("")
    lines.append(f"{'PASS':<50} {'FAIL':<50}")
    lines.append(f"{'-'*50} {'-'*50}")

    p_idx, f_idx = 0, 0
    for i, (a, b) in enumerate(zip(aligned_p, aligned_f)):
        is_div = (a != b)
        marker = ">>>" if i == first_div else ("  *" if is_div else "   ")

        if a == GAP:
            p_text = "---"
        else:
            p_step = p_steps[p_idx]
            p_text = f"{a} {p_step['detail']}" if p_step["detail"] else a
            p_idx += 1

        if b == GAP:
            f_text = "---"
        else:
            f_step = f_steps[f_idx]
            f_text = f"{b} {f_step['detail']}" if f_step["detail"] else b
            f_idx += 1

        lines.append(f"{marker} {p_text:<48} {f_text:<48}")

    lines.append("")

    # Context at the divergence point
    if first_div is not None:
        lines.append("## Context at the divergence point")
        lines.append("")

        p_orig = _orig_idx(aligned_p, first_div)
        f_orig = _orig_idx(aligned_f, first_div)

        if p_orig is not None and p_orig < len(p_steps):
            ps = p_steps[p_orig]
            lines.append("### Pass run's reasoning and action")
            lines.append("")
            if ps["reasoning"]:
                lines.append(f"Reasoning: {ps['reasoning']}")
                lines.append("")
            lines.append(f"Action: {ps['enriched']} ({ps['detail']})")
            if ps["tool_input"]:
                lines.append(f"Input: {_format_input(ps['tool_input'])}")
            if ps["result"]:
                lines.append(f"Result ({len(ps['result'])} chars): {ps['result'][:500]}")
            lines.append("")

        if f_orig is not None and f_orig < len(f_steps):
            fs = f_steps[f_orig]
            lines.append("### Fail run's reasoning and action")
            lines.append("")
            if fs["reasoning"]:
                lines.append(f"Reasoning: {fs['reasoning']}")
                lines.append("")
            lines.append(f"Action: {fs['enriched']} ({fs['detail']})")
            if fs["tool_input"]:
                lines.append(f"Input: {_format_input(fs['tool_input'])}")
            if fs["result"]:
                lines.append(f"Result ({len(fs['result'])} chars): {fs['result'][:500]}")
            lines.append("")

    # Edit diffs if present
    p_edits = [s for s in p_steps if s["tool_input"].get("old_string")]
    f_edits = [s for s in f_steps if s["tool_input"].get("old_string")]
    if p_edits or f_edits:
        lines.append("## Code changes")
        lines.append("")
        if p_edits:
            lines.append("### Pass run's edits")
            for s in p_edits:
                lines.append(f"File: {s['detail']}")
                lines.append(f"```diff")
                lines.append(f"- {s['tool_input']['old_string']}")
                lines.append(f"+ {s['tool_input']['new_string']}")
                lines.append(f"```")
                if s["reasoning"]:
                    lines.append(f"Reasoning: {s['reasoning']}")
                lines.append("")
        if f_edits:
            lines.append("### Fail run's edits")
            for s in f_edits:
                lines.append(f"File: {s['detail']}")
                lines.append(f"```diff")
                lines.append(f"- {s['tool_input']['old_string']}")
                lines.append(f"+ {s['tool_input']['new_string']}")
                lines.append(f"```")
                if s["reasoning"]:
                    lines.append(f"Reasoning: {s['reasoning']}")
                lines.append("")

    # Test results
    p_tests = [s for s in p_steps if s["enriched"].startswith("test")]
    f_tests = [s for s in f_steps if s["enriched"].startswith("test")]
    if p_tests or f_tests:
        lines.append("## Test results")
        lines.append("")
        for s in p_tests:
            lines.append(f"Pass run test: {s['detail']}")
            if s["result"]:
                lines.append(f"```\n{s['result'][:1000]}\n```")
            lines.append("")
        for s in f_tests:
            lines.append(f"Fail run test: {s['detail']}")
            if s["result"]:
                lines.append(f"```\n{s['result'][:1000]}\n```")
            lines.append("")

    # Prompt for analysis
    lines.append("## Analysis prompt")
    lines.append("")
    lines.append("Given the above divergence between a passing and failing run of the same task:")
    lines.append("1. Why did the pass run's approach work while the fail run's didn't?")
    lines.append("2. What specific knowledge or decision at the divergence point made the difference?")
    lines.append("3. What system prompt change or tool configuration would steer agents toward the successful approach?")

    return "\n".join(lines)


def _pick_pair(pass_runs: list[Run], fail_runs: list[Run]) -> tuple[Run, Run]:
    """Pick the best pass/fail pair. Same harness > has reasoning > any."""
    def _has_reasoning(r: Run) -> bool:
        return any(s.output.get("reasoning") for s in r.steps if s.output)

    for p in pass_runs:
        for f in fail_runs:
            if p.harness == f.harness and _has_reasoning(p) and _has_reasoning(f):
                return p, f
    for p in pass_runs:
        for f in fail_runs:
            if p.harness == f.harness:
                return p, f
    for p in pass_runs:
        for f in fail_runs:
            if _has_reasoning(p) or _has_reasoning(f):
                return p, f
    return pass_runs[0], fail_runs[0]


def _enrich_steps(run: Run) -> list[dict]:
    """Build enriched step list with all available context."""
    steps = []
    for s in run.steps:
        enriched = step_enriched_name(s)
        if enriched is None:
            continue

        detail = ""
        if s.attrs:
            fp = s.attrs.get("file_path", "")
            if fp:
                detail = os.path.basename(fp)
            cmd = s.attrs.get("command", "")
            if cmd:
                detail = cmd[:80]
            pat = s.attrs.get("pattern", "")
            if pat:
                detail = pat[:60]

        steps.append({
            "enriched": enriched,
            "detail": detail,
            "reasoning": s.output.get("reasoning", "") if s.output else "",
            "result": s.output.get("result", "") if s.output else "",
            "tool_input": s.output.get("tool_input", {}) if s.output else {},
        })
    return steps


def _detail_at(steps: list[dict], aligned: list[str], idx: int) -> str:
    orig = _orig_idx(aligned, idx)
    if orig is not None and orig < len(steps):
        return steps[orig]["detail"]
    return ""


def _orig_idx(aligned: list[str], aligned_idx: int) -> int | None:
    """Map aligned index to original step index (skipping gaps)."""
    count = 0
    for i in range(aligned_idx):
        if aligned[i] != GAP:
            count += 1
    if aligned[aligned_idx] == GAP:
        return None
    return count


def _format_input(tool_input: dict) -> str:
    """Format tool input for display, omitting large fields."""
    parts = []
    for k, v in tool_input.items():
        if k in ("old_string", "new_string"):
            parts.append(f"{k}: ({len(str(v))} chars)")
        else:
            s = str(v)
            if len(s) > 100:
                s = s[:97] + "..."
            parts.append(f"{k}: {s}")
    return ", ".join(parts)
