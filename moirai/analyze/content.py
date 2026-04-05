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
    ReasoningMetrics,
    Run,
    TransitionSignal,
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


UNCERTAINTY_RE = re.compile(
    r'\bmaybe\b|\bmight\b|\bpossibly\b'           # hedging words
    r'|\bactually,'                                 # self-correction marker
    r'|\blet me (?:try|step back|reconsider)\b'     # exploration/backtracking
    r'|\banother approach\b',                        # approach abandonment
    re.IGNORECASE,
)
_CAUSAL_RE = re.compile(
    r'\bbecause\b|\bsince\b|\btherefore\b|\bthus\b'
    r'|\bcaused by\b|\bdue to\b|\bresult of\b',
    re.IGNORECASE,
)
_DIAGNOSIS_RE = re.compile(
    r'\bthe issue is\b|\bthe problem is\b|\broot cause\b'
    r'|\bthis happens when\b|\bfails? because\b|\berror.{0,20}because\b',
    re.IGNORECASE,
)
_CODE_REF_RE = re.compile(
    r'\.py\b|line \d+|def \w+|class \w+|import \w+',
)


def compute_test_centroid(run: Run) -> float | None:
    """Normalized position (0-1) of the center of mass of test steps.

    Returns None if the run has no test steps. Uses enriched step names
    to identify test steps (any name starting with 'test(').
    """
    from moirai.compress import step_enriched_name

    enriched = [step_enriched_name(s) for s in run.steps]
    enriched = [e for e in enriched if e is not None]
    if not enriched:
        return None

    n = len(enriched)
    if n == 1:
        # Single step: position is 0.0 regardless
        test_positions = [
            0.0
            for _i, name in enumerate(enriched)
            if name.startswith("test(")
        ]
    else:
        test_positions = [
            i / (n - 1)
            for i, name in enumerate(enriched)
            if name.startswith("test(")
        ]
    if not test_positions:
        return None

    return sum(test_positions) / len(test_positions)


def find_exemplar_task(
    task_groups: dict[str, list[Run]],
    feature_fn,
    min_pass: int = 5,
    min_fail: int = 5,
    min_delta: float = 0.05,
    prefer_step_range: tuple[int, int] = (30, 70),
) -> str | None:
    """Find a task that best exemplifies a behavioral feature difference.

    Args:
        task_groups: {task_id: [runs]} from select_task_groups()
        feature_fn: callable(Run) -> float | None, the feature to compare
        min_pass/min_fail: minimum runs per outcome
        min_delta: minimum pass_mean - fail_mean to qualify
        prefer_step_range: ideal average step count range

    Returns the task_id with the largest feature delta that has
    enough runs and reasonable step counts, or None.
    """
    candidates = []
    for task_id, runs in task_groups.items():
        pass_vals = [feature_fn(r) for r in runs if r.result.success is True]
        fail_vals = [feature_fn(r) for r in runs if r.result.success is False]
        pass_vals = [v for v in pass_vals if v is not None]
        fail_vals = [v for v in fail_vals if v is not None]

        if len(pass_vals) < min_pass or len(fail_vals) < min_fail:
            continue

        delta = sum(pass_vals) / len(pass_vals) - sum(fail_vals) / len(fail_vals)
        if abs(delta) < min_delta:
            continue

        avg_steps = sum(len(r.steps) for r in runs) / len(runs)
        lo, hi = prefer_step_range
        in_range = lo <= avg_steps <= hi

        candidates.append((task_id, delta, avg_steps, in_range))

    if not candidates:
        return None

    # Prefer tasks in step range, then by delta
    candidates.sort(key=lambda x: (-x[3], -abs(x[1])))
    return candidates[0][0]


def compute_reasoning_metrics(runs: list[Run]) -> ReasoningMetrics | None:
    """Compute aggregate reasoning quality metrics for a set of runs."""
    total_uncertainty = 0
    total_causal = 0
    total_diagnosis = 0
    total_code_refs = 0
    total_chars = 0
    total_reasoning_steps = 0
    total_steps = 0

    for r in runs:
        total_steps += len(r.steps)
        for s in r.steps:
            reasoning = str(s.output.get("reasoning", ""))
            if not reasoning:
                continue
            total_reasoning_steps += 1
            total_chars += len(reasoning)
            total_uncertainty += len(UNCERTAINTY_RE.findall(reasoning))
            total_causal += len(_CAUSAL_RE.findall(reasoning))
            total_diagnosis += len(_DIAGNOSIS_RE.findall(reasoning))
            total_code_refs += len(_CODE_REF_RE.findall(reasoning))

    if total_reasoning_steps == 0:
        return None

    return ReasoningMetrics(
        uncertainty_density=total_uncertainty / total_reasoning_steps,
        causal_density=total_causal / total_reasoning_steps,
        diagnosis_density=total_diagnosis / total_reasoning_steps,
        code_ref_density=total_code_refs / total_reasoning_steps,
        reasoning_per_step=total_chars / total_steps,
        n_reasoning_steps=total_reasoning_steps,
    )


def compute_transition_bigrams(
    runs: list[Run],
) -> list[TransitionSignal]:
    """Compute transition bigram signals between passing and failing runs.

    Only analyzes runs with at least one edit/write step (controls for
    the trivial confound that runs without edits always fail).

    Normalization: per-run first (bigram_count / (n_steps - 1)), then
    averaged across runs in each group. This avoids length-weighting bias
    where one long run dominates the group signal.

    Returns top 5 signals sorted by absolute delta descending.
    Min count threshold: 10 (hardcoded, not a parameter).
    """
    from moirai.compress import step_enriched_name

    # Step 1: filter to runs with known outcome and at least one edit/write step
    eligible: list[Run] = []
    for r in runs:
        if r.result.success is None:
            continue
        has_edit = any(s.name in ("edit", "write") for s in r.steps)
        if not has_edit:
            continue
        eligible.append(r)

    # Step 2: extract enriched step sequences, skip runs with <2 enriched steps
    run_sequences: list[tuple[Run, list[str]]] = []
    for r in eligible:
        seq = [
            name
            for s in r.steps
            if (name := step_enriched_name(s)) is not None
        ]
        if len(seq) < 2:
            continue
        run_sequences.append((r, seq))

    # Step 3: split into pass/fail groups; require >=2 runs per group
    pass_entries = [(r, seq) for r, seq in run_sequences if r.result.success]
    fail_entries = [(r, seq) for r, seq in run_sequences if not r.result.success]

    if len(pass_entries) < 2 or len(fail_entries) < 2:
        return []

    # Steps 4-6: compute per-run normalized bigram rates, then average per group
    all_bigrams: set[tuple[str, str]] = set()
    raw_counts: dict[tuple[str, str], int] = defaultdict(int)

    def _group_rates(
        entries: list[tuple[Run, list[str]]],
    ) -> dict[tuple[str, str], float]:
        rates: dict[tuple[str, str], list[float]] = defaultdict(list)
        for _r, seq in entries:
            counts: dict[tuple[str, str], int] = defaultdict(int)
            for a, b in zip(seq, seq[1:]):
                bigram = (a, b)
                counts[bigram] += 1
                raw_counts[bigram] += 1
                all_bigrams.add(bigram)
            norm = max(1, len(seq) - 1)
            for bigram, count in counts.items():
                rates[bigram].append(count / norm)
        return {bg: sum(vals) / len(vals) for bg, vals in rates.items()}

    pass_rates = _group_rates(pass_entries)
    fail_rates = _group_rates(fail_entries)

    # Step 7-8: compute delta, filter by raw count >= 10, take top 5
    signals: list[TransitionSignal] = []
    for bigram in all_bigrams:
        total = raw_counts[bigram]
        if total < 10:
            continue
        pr = pass_rates.get(bigram, 0.0)
        fr = fail_rates.get(bigram, 0.0)
        delta = pr - fr
        signals.append(TransitionSignal(
            from_step=bigram[0],
            to_step=bigram[1],
            pass_rate=pr,
            fail_rate=fr,
            delta=delta,
            total_count=total,
        ))

    # Step 9: sort by absolute delta descending, take top 5
    signals.sort(key=lambda s: abs(s.delta), reverse=True)
    return signals[:5]


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
    prompt: str, mode: str, timeout: int = 120,
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
        cmd = [cli, "-p", "-", "--output-format", "json", "--model", "sonnet"]
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

    # Fallback: if no task groups qualify (e.g., 1 run per task), use
    # cluster-based grouping instead. Cluster runs by structural similarity,
    # then analyze each cluster with mixed outcomes.
    if not groups:
        all_clusters, cluster_skips = _cluster_based_groups(runs)
        if task_filter:
            # Apply filter to cluster-based groups (e.g., --task cluster_8)
            if task_filter in all_clusters:
                groups = {task_filter: all_clusters[task_filter]}
                skip_reasons = {}
            else:
                skip_reasons = {task_filter: "not found"}
        else:
            groups, skip_reasons = all_clusters, cluster_skips

    n_skipped = len(skip_reasons)
    n_qualifying = len(groups)

    reports: list[ExplanationReport] = []

    for task_id, group_runs in groups.items():
        # Pre-alignment sampling if > max_runs
        if len(group_runs) > max_runs:
            group_runs = _presample(group_runs, max_runs, seed)

        alignment = align_runs(group_runs, level="name")
        cons = consensus(alignment.matrix)
        # Use q_threshold=1.0 to get ALL divergent columns regardless of
        # statistical significance — the LLM does the interpretive work.
        # Structural significance filtering is the wrong gate for content analysis.
        divpoints, _ = find_divergence_points(alignment, group_runs, q_threshold=1.0)

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

        # Reasoning metrics: overall, pass, fail
        pass_runs_list = [r for r in known if r.result.success]
        fail_runs_list = [r for r in known if not r.result.success]
        reasoning_all = compute_reasoning_metrics(known)
        reasoning_pass = compute_reasoning_metrics(pass_runs_list) if pass_runs_list else None
        reasoning_fail = compute_reasoning_metrics(fail_runs_list) if fail_runs_list else None

        transitions = compute_transition_bigrams(group_runs)

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
            reasoning=reasoning_all,
            reasoning_pass=reasoning_pass,
            reasoning_fail=reasoning_fail,
            transitions=transitions,
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


def _cluster_based_groups(
    runs: list[Run],
    min_runs: int = 5,
) -> tuple[dict[str, list[Run]], dict[str, str]]:
    """Fall back to cluster-based grouping when task-based grouping fails.

    Clusters runs by structural similarity, then treats each cluster
    with mixed outcomes as a group for content analysis.
    """
    from moirai.analyze.cluster import cluster_runs

    cr = cluster_runs(runs)

    groups: dict[str, list[Run]] = {}
    skip_reasons: dict[str, str] = {}

    for info in cr.clusters:
        cluster_runs_list = [
            r for r in runs if cr.labels.get(r.run_id) == info.cluster_id
        ]
        known = [r for r in cluster_runs_list if r.result.success is not None]
        label = f"cluster_{info.cluster_id}"

        if len(known) < min_runs:
            skip_reasons[label] = "too few runs"
            continue

        has_pass = any(r.result.success for r in known)
        has_fail = any(not r.result.success for r in known)
        if not has_pass:
            skip_reasons[label] = "all-fail"
            continue
        if not has_fail:
            skip_reasons[label] = "all-pass"
            continue

        groups[label] = cluster_runs_list

    return groups, skip_reasons
