"""Content-aware analysis — LLM-assisted differential diagnosis of agent traces."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections import defaultdict

import numpy as np

from moirai.schema import (
    Alignment,
    ContentFinding,
    DivergencePoint,
    ExplanationReport,
    GAP,
    Run,
)


def select_task_groups(
    runs: list[Run],
    min_runs: int = 5,
    task_filter: str | None = None,
) -> tuple[dict[str, list[Run]], dict[str, str]]:
    """Partition runs by task_id, filter to qualifying groups.

    Returns (qualifying_groups, skip_reasons).
    """
    by_task: dict[str, list[Run]] = defaultdict(list)
    for run in runs:
        by_task[run.task_id].append(run)

    if task_filter is not None:
        if task_filter not in by_task:
            return {}, {task_filter: "not found"}
        by_task = {task_filter: by_task[task_filter]}

    qualifying: dict[str, list[Run]] = {}
    skip_reasons: dict[str, str] = {}

    for task_id, task_runs in by_task.items():
        known = [r for r in task_runs if r.result.success is not None]
        if len(known) < min_runs:
            skip_reasons[task_id] = "too few runs"
            continue

        has_pass = any(r.result.success for r in known)
        has_fail = any(not r.result.success for r in known)
        if not has_pass:
            skip_reasons[task_id] = "all-fail"
            continue
        if not has_fail:
            skip_reasons[task_id] = "all-pass"
            continue

        qualifying[task_id] = task_runs

    return qualifying, skip_reasons


def sample_runs(
    runs: list[Run],
    alignment: Alignment,
    cons: list[str],
    seed: int = 42,
) -> list[Run]:
    """Stratified sampling: near-consensus + high-divergence per outcome class."""
    n_cols = len(cons) if cons else 0

    # Compute per-run distance from consensus
    run_distances: dict[str, float] = {}
    for run_idx, run_id in enumerate(alignment.run_ids):
        if run_idx < len(alignment.matrix) and n_cols > 0:
            row = alignment.matrix[run_idx]
            mismatches = sum(1 for a, b in zip(row, cons) if a != b)
            run_distances[run_id] = mismatches / n_cols
        else:
            run_distances[run_id] = 1.0
    rng = np.random.default_rng(seed)

    sampled: list[Run] = []
    for success_val in [True, False]:
        class_runs = [
            r for r in runs
            if r.result.success is not None and r.result.success == success_val
        ]
        if not class_runs:
            continue

        # Sort by distance; use rng for tie-breaking
        shuffled = list(class_runs)
        rng.shuffle(shuffled)
        sorted_runs = sorted(shuffled, key=lambda r: run_distances.get(r.run_id, 1.0))

        # Near-consensus (closest)
        sampled.append(sorted_runs[0])
        # High-divergence (furthest)
        if len(sorted_runs) > 1:
            sampled.append(sorted_runs[-1])

    return sampled


def _step_at_column(run: Run, alignment: Alignment, col: int) -> int | None:
    """Map alignment column to original step index in a run.

    Counts non-GAP entries in the run's alignment row up to the column
    to determine the original step index.
    """
    run_idx = None
    for i, rid in enumerate(alignment.run_ids):
        if rid == run.run_id:
            run_idx = i
            break
    if run_idx is None or run_idx >= len(alignment.matrix):
        return None

    row = alignment.matrix[run_idx]
    if col >= len(row) or row[col] == GAP:
        return None

    # Count non-GAP entries before this column to get original step index
    orig_idx = sum(1 for c in range(col) if row[c] != GAP)
    if orig_idx >= len(run.steps):
        return None
    return orig_idx


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit - 3] + "..."


def _extract_error(result_text: str) -> str | None:
    """Extract error string from step result text."""
    error_patterns = [
        r"(?:Traceback.*?\n(?:.*\n)*?.*(?:Error|Exception):.*)",
        r"(?:Error|Exception|FAILED|FAIL):?\s+.+",
        r"(?:FileNotFoundError|PermissionError|ImportError|SyntaxError|TypeError|ValueError):?\s+.+",
    ]
    for pattern in error_patterns:
        m = re.search(pattern, result_text, re.IGNORECASE)
        if m:
            return m.group(0)[:200]
    return None


def _detect_test_outcome(result_text: str) -> str | None:
    """Detect test pass/fail from result text."""
    lower = result_text.lower()
    if any(kw in lower for kw in ["passed", "tests passed", "ok", "all tests"]):
        if "failed" not in lower and "error" not in lower:
            return "pass"
    if any(kw in lower for kw in ["failed", "failure", "tests failed"]):
        return "fail"
    if "error" in lower and "test" in lower:
        return "error"
    return None


def build_prompt(
    task_id: str,
    runs: list[Run],
    alignment: Alignment,
    divergent_columns: list[DivergencePoint],
    sampled_runs: list[Run],
    cons: list[str] | None = None,
) -> str:
    """Assemble the analysis prompt with anchor-then-contrast structure."""
    n_runs = len(runs)
    n_pass = sum(1 for r in runs if r.result.success)
    n_fail = sum(1 for r in runs if r.result.success is not None and not r.result.success)

    if cons is None:
        from moirai.analyze.align import consensus
        cons = consensus(alignment.matrix)
    cons_str = " → ".join(cons[:20]) if cons else "(empty)"

    parts: list[str] = []
    parts.append(f"## Task: {task_id}")
    parts.append(f"Runs: {n_runs} total, {n_pass} pass, {n_fail} fail")
    parts.append(f"Consensus sequence: {cons_str}")
    parts.append("")

    pass_runs = [r for r in sampled_runs if r.result.success]
    fail_runs = [r for r in sampled_runs if r.result.success is not None and not r.result.success]

    for dp in divergent_columns:
        vals_str = ", ".join(f"{v}: {c} runs" for v, c in dp.value_counts.items())
        succ_str = ", ".join(
            f"{v}: {r:.0%}" for v, r in (dp.success_by_value or {}).items()
            if r is not None
        )

        parts.append(f"### Divergent position: column {dp.column}")
        if dp.phase_context:
            parts.append(f"Phase context: {dp.phase_context}")
        parts.append(f"Values: {vals_str}")
        if succ_str:
            parts.append(f"Success rates: {succ_str}")
        parts.append("")

        # Passing runs at this column
        parts.append("#### Passing runs")
        for run in pass_runs:
            _append_run_at_column(parts, run, alignment, dp.column)

        # Failing runs at this column
        parts.append("#### Failing runs")
        for run in fail_runs:
            _append_run_at_column(parts, run, alignment, dp.column)

        parts.append("")

    # Response format instruction
    parts.append("## Instructions")
    parts.append("Analyze the content differences between passing and failing runs at these divergence points.")
    parts.append("Explain why the failing runs fail — what specific decisions, tool calls, or reasoning led to failure?")
    parts.append("")
    parts.append("Respond with JSON in this exact format:")
    parts.append('```json')
    parts.append("""{
  "findings": [
    {
      "category": "string (e.g. wrong_file, missing_test, error_ignored, wrong_command, reasoning_gap)",
      "column": 0,
      "description": "one sentence explaining the finding",
      "evidence": "specific content from the runs that supports this",
      "pass_runs": ["run_id1"],
      "fail_runs": ["run_id2"]
    }
  ],
  "summary": "2-3 sentence narrative explaining the overall pattern",
  "confidence": "high, medium, or low"
}""")
    parts.append('```')

    prompt = "\n".join(parts)

    # Budget check: if over ~8K tokens (~32K chars), reduce to top-3 columns
    if len(prompt) > 32000 and len(divergent_columns) > 3:
        return build_prompt(task_id, runs, alignment, divergent_columns[:3], sampled_runs, cons)

    return prompt


def _append_run_at_column(
    parts: list[str],
    run: Run,
    alignment: Alignment,
    col: int,
) -> None:
    """Append a run's content at a specific alignment column to prompt parts."""
    orig_idx = _step_at_column(run, alignment, col)
    if orig_idx is None:
        parts.append(f"  Run {run.run_id} ({run.result.success}): GAP at this column")
        return

    step = run.steps[orig_idx]
    parts.append(f"  Run {run.run_id}:")

    # Structured metadata
    file_path = step.attrs.get("file_path") or ""
    command = step.attrs.get("command") or ""
    result_text = str(step.output.get("result", ""))
    error = _extract_error(result_text)
    test_outcome = _detect_test_outcome(result_text)

    if file_path:
        parts.append(f"    file: {file_path}")
    if command:
        parts.append(f"    command: {command}")
    if error:
        parts.append(f"    error: {error}")
    if test_outcome:
        parts.append(f"    test: {test_outcome}")

    # Raw content
    reasoning = str(step.output.get("reasoning", ""))
    tool_input = str(step.output.get("tool_input", ""))
    if reasoning:
        parts.append(f"    reasoning: {_truncate(reasoning, 800)}")
    if result_text:
        parts.append(f"    result: {_truncate(result_text, 800)}")
    if tool_input:
        parts.append(f"    tool_input: {_truncate(tool_input, 400)}")

    # Before/after context
    if orig_idx > 0:
        prev = run.steps[orig_idx - 1]
        parts.append(f"    (before: {prev.type}:{prev.name})")
    if orig_idx < len(run.steps) - 1:
        nxt = run.steps[orig_idx + 1]
        parts.append(f"    (after: {nxt.type}:{nxt.name})")


def parse_response(raw: str) -> tuple[list[ContentFinding], str, str]:
    """Parse LLM output into findings, summary, confidence.

    Handles markdown fences, envelope wrapping, malformed JSON.
    """
    # Handle claude --output-format json envelope
    try:
        envelope = json.loads(raw)
        if isinstance(envelope, dict) and "result" in envelope:
            raw = envelope["result"]
    except (json.JSONDecodeError, TypeError):
        pass

    # Strip markdown fences
    m = re.search(r"```(?:json)?\s*\n(.*?)```", str(raw), re.DOTALL)
    text = m.group(1) if m else str(raw)

    # Try parsing
    data = _try_parse_json(text.strip())
    if data is None:
        return [], "", ""

    findings: list[ContentFinding] = []
    for f in data.get("findings", []):
        if not isinstance(f, dict):
            continue
        category = str(f.get("category", "other"))
        # Category is kept as-is from LLM output
        findings.append(ContentFinding(
            category=category,
            column=int(f.get("column", 0)),
            description=str(f.get("description", "")),
            evidence=str(f.get("evidence", "")),
            pass_runs=f.get("pass_runs", []),
            fail_runs=f.get("fail_runs", []),
        ))

    summary = str(data.get("summary", ""))
    confidence = str(data.get("confidence", ""))

    return findings, summary, confidence


def _try_parse_json(text: str) -> dict | None:
    """Try to parse JSON, with one retry stripping to outermost braces."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Retry: find first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


def invoke_llm(
    prompt: str, mode: str, timeout: int = 60,
) -> tuple[str | None, str | None]:
    """Call LLM CLI via subprocess.

    Returns (stdout, error_reason). stdout is None on failure,
    error_reason is None on success. Uses stdin for prompt.
    """
    if mode == "structural":
        return None, None  # not a failure, intentional skip

    if mode == "auto":
        cli = shutil.which("claude") or shutil.which("codex")
        if not cli:
            return None, None  # auto mode: graceful degradation, not an error
    else:
        cli = shutil.which(mode)
        if not cli:
            return None, f"{mode} CLI not found"

    # Build CLI args per tool
    cli_name = cli.rsplit("/", 1)[-1] if "/" in cli else cli
    if cli_name == "claude":
        cmd = [cli, "-p", "-", "--output-format", "json"]
    elif cli_name == "codex":
        cmd = [cli, "--prompt", "-"]
    else:
        cmd = [cli, "-p", "-"]

    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()[:200] if result.stderr else ""
            return None, f"{cli_name} exited {result.returncode}: {stderr}"
        return result.stdout, None
    except subprocess.TimeoutExpired:
        return None, f"{cli_name} timed out after {timeout}s"
    except OSError as e:
        return None, f"{cli_name} failed: {e}"


def run_explain(
    runs: list[Run],
    mode: str = "auto",
    task_filter: str | None = None,
    top_n: int = 5,
    max_runs: int = 50,
    timeout: int = 60,
    seed: int = 42,
    cluster: bool = False,
) -> list[ExplanationReport]:
    """Top-level orchestrator: group -> align -> diverge -> sample -> prompt -> LLM -> parse."""
    from moirai.analyze.align import align_runs, consensus
    from moirai.analyze.divergence import find_divergence_points
    from moirai.compress import compress_phases

    groups, skip_reasons = select_task_groups(runs, task_filter=task_filter)
    n_skipped = len(skip_reasons)
    n_qualifying = len(groups)

    reports: list[ExplanationReport] = []

    for task_id, group_runs in groups.items():
        # Pre-alignment sampling if > max_runs
        if len(group_runs) > max_runs:
            group_runs = _presample(group_runs, max_runs, seed)

        alignment = align_runs(group_runs, level="name")
        cons = consensus(alignment.matrix)
        divpoints, _ = find_divergence_points(alignment, group_runs)

        # Take top-N by entropy
        top_divpoints = sorted(divpoints, key=lambda d: d.entropy, reverse=True)[:top_n]

        # Consensus as compressed phase notation from first run (representative)
        if group_runs:
            consensus_str = compress_phases(group_runs[0])
        else:
            consensus_str = "(empty)"

        sampled = sample_runs(group_runs, alignment, cons, seed=seed)

        # Optional clustering
        concordance_tau = None
        concordance_p = None
        if cluster:
            from moirai.analyze.cluster import cluster_runs, compute_concordance
            cr = cluster_runs(group_runs)
            conc = compute_concordance(group_runs, cr.labels)
            # Use largest cluster's concordance
            if conc:
                largest_cid = max(conc.keys(), key=lambda cid: conc[cid].n_runs)
                concordance_tau = conc[largest_cid].tau
                concordance_p = conc[largest_cid].p_value

        prompt = build_prompt(task_id, group_runs, alignment, top_divpoints, sampled, cons)
        raw, llm_error = invoke_llm(prompt, mode, timeout)

        if raw is not None:
            findings, summary, confidence = parse_response(raw)
        else:
            findings, summary, confidence = [], "", ""
            if llm_error:
                # Surface error in summary so terminal display shows it
                summary = f"[LLM error: {llm_error}]"

        known = [r for r in group_runs if r.result.success is not None]
        pass_rate = sum(1 for r in known if r.result.success) / len(known) if known else 0.0

        reports.append(ExplanationReport(
            task_id=task_id,
            n_runs=len(group_runs),
            pass_rate=pass_rate,
            findings=findings,
            summary=summary,
            confidence=confidence,
            consensus=consensus_str,
            divergent_columns=top_divpoints,
            n_qualifying=n_qualifying,
            n_skipped=n_skipped,
            concordance_tau=concordance_tau,
            concordance_p=concordance_p,
        ))

    return reports


def _presample(
    runs: list[Run],
    max_runs: int,
    seed: int,
) -> list[Run]:
    """Downsample runs while preserving outcome balance."""
    rng = np.random.default_rng(seed)
    pass_runs = [r for r in runs if r.result.success]
    fail_runs = [r for r in runs if r.result.success is not None and not r.result.success]
    # Proportional allocation
    total_known = len(pass_runs) + len(fail_runs)
    if total_known == 0:
        indices = rng.choice(len(runs), size=min(max_runs, len(runs)), replace=False)
        return [runs[i] for i in sorted(indices)]

    n_pass = max(1, round(max_runs * len(pass_runs) / total_known))
    n_fail = max(1, max_runs - n_pass)

    sampled: list[Run] = []
    if len(pass_runs) <= n_pass:
        sampled.extend(pass_runs)
    else:
        indices = rng.choice(len(pass_runs), size=n_pass, replace=False)
        sampled.extend(pass_runs[i] for i in sorted(indices))

    if len(fail_runs) <= n_fail:
        sampled.extend(fail_runs)
    else:
        indices = rng.choice(len(fail_runs), size=n_fail, replace=False)
        sampled.extend(fail_runs[i] for i in sorted(indices))

    return sampled
