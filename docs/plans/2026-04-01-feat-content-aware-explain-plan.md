---
title: "feat: content-aware explain command"
type: feat
date: 2026-04-01
spec: docs/specs/2026-04-01-content-aware-diagnosis-design.md
deepened: true
---

# Content-Aware Explain Command

## Enhancement summary

**Deepened on:** 2026-04-01
**Research agents used:** subprocess patterns, prompt engineering, architecture review, simplicity review, spec flow analysis

### Key improvements from research
1. Use stdin (`input=`) for subprocess prompts — avoids ARG_MAX shell limits
2. Anchor-then-contrast prompt structure — divergence point first, then pass/fail groups
3. Dropped 3 unnecessary dataclasses, 2 unnecessary functions, ~13 unnecessary tests
4. Clarified relationship with existing `analyze/explain.py` module
5. Fixed 7 spec/plan gaps found by flow analysis

## Overview

Replace the existing `moirai explain` command with a content-aware version that answers "why do runs fail?" by comparing step content (reasoning, tool inputs, results) at structurally divergent positions across runs. The existing single-run explain behavior is preserved via `--run <id>`.

Spec: `docs/specs/2026-04-01-content-aware-diagnosis-design.md`

### Relationship to existing modules

- `analyze/explain.py` — existing module with `explain_task()`. Used by the `divergence` command (cli.py:529+), **not** by the existing `explain` command. Left untouched.
- `analyze/content.py` — new module for cross-run content analysis. Does not import or depend on `analyze/explain.py`.
- The existing `explain` CLI code (cli.py:382-527) stays inline for the `--run` path. No refactoring of existing code.

## Phase A: Core implementation

Schema, content analysis module, CLI command, and terminal display.

### A.1 Add dataclasses to schema.py

Add after the `# --- Evidence and diagnosis dataclasses ---` section:

```python
# --- Content-aware analysis dataclasses ---

@dataclass
class ContentFinding:
    category: str              # free-text from LLM; normalized to known values when possible
    column: int                # alignment column where observed
    description: str           # one-sentence explanation
    evidence: str              # specific content from runs
    pass_runs: list[str]       # run_ids with passing behavior
    fail_runs: list[str]       # run_ids with failing behavior

@dataclass
class ExplanationReport:
    task_id: str
    n_runs: int
    pass_rate: float
    findings: list[ContentFinding]
    summary: str                           # free-form narrative; empty if structural mode
    confidence: str                        # "high", "medium", "low"; empty if structural
    consensus: str                         # compressed phase notation
    divergent_columns: list[DivergencePoint]  # reuse existing dataclass
    n_qualifying: int                      # task groups that qualified
    n_skipped: int                         # task groups skipped
    concordance_tau: float | None = None   # only with --cluster
    concordance_p: float | None = None     # only with --cluster
```

Known finding categories (for normalization, not enforcement): `wrong_file`, `missing_test`, `error_ignored`, `wrong_command`, `reasoning_gap`. Validated with a set check, not a StrEnum.

### A.2 Create moirai/analyze/content.py

```python
"""Content-aware analysis — LLM-assisted differential diagnosis of agent traces."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections import defaultdict

from moirai.schema import (
    Alignment, ContentFinding, DivergencePoint, ExplanationReport, Run,
)

KNOWN_CATEGORIES = {
    "wrong_file", "missing_test", "error_ignored",
    "wrong_command", "reasoning_gap",
}


def select_task_groups(
    runs: list[Run],
    min_runs: int = 5,
    task_filter: str | None = None,
) -> tuple[dict[str, list[Run]], dict[str, str]]:
    """Partition runs by task_id, filter to qualifying groups.

    Returns (qualifying_groups, skip_reasons).
    """
    # Group by task_id
    # If task_filter, check task exists — if not, return error reason "not found"
    # Filter: mixed outcomes, >= min_runs known outcomes
    # Track skip reasons: "all-pass", "all-fail", "too few runs"


def sample_runs(
    runs: list[Run],
    alignment: Alignment,
    n_per_class: int = 2,
    seed: int = 42,
) -> list[Run]:
    """Stratified sampling: near-consensus + high-divergence per outcome class."""
    # Compute per-run distance from consensus (reuse _consensus from align.py)
    # For each outcome class (pass/fail):
    #   Pick the run closest to consensus
    #   Pick the run furthest from consensus
    # Use numpy RNG with seed for deterministic tie-breaking


def build_prompt(
    task_id: str,
    runs: list[Run],
    alignment: Alignment,
    divergent_columns: list[DivergencePoint],
    sampled_runs: list[Run],
) -> str:
    """Assemble the analysis prompt with hybrid context.

    Prompt structure (anchor-then-contrast):
      1. Context: task_id, run count, pass/fail split, consensus
      2. For each divergent column:
         - Shared context: position, phase, what came before
         - Pass group: steps from passing sampled runs at this column
         - Fail group: steps from failing sampled runs at this column
         Per step: extracted metadata (error, file_path, command, test outcome)
                   + raw reasoning (~800 chars) + raw result (~800 chars)
                   + raw tool_input (~400 chars) + one step before/after
      3. Response format: JSON schema with findings[] + summary + confidence

    Budget: ~8K tokens. If exceeded, reduce to top-3 columns.
    """
    # Inline metadata extraction here (no separate function):
    # - Parse step.output.get("result", "") for error strings/tracebacks
    # - Pull file_path from step.attrs.get("file_path") or step.output.get("tool_input")
    # - Pull command from step.attrs.get("command")
    # - Detect test pass/fail from keywords in result
    # - Classify output: empty if no output, error if error detected, normal otherwise
    # - Truncate raw fields to limits


def parse_response(raw: str) -> tuple[list[ContentFinding], str, str]:
    """Parse LLM output into findings, summary, confidence.

    Handles: markdown fences, envelope wrapping (claude --output-format json),
    malformed JSON (one retry stripping to first { ... last }).
    Returns (findings, summary, confidence).
    """
    # Strip markdown fences: re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.DOTALL)
    # json.loads; if fails, try extracting between first { and last }
    # Validate required fields: findings (list), summary (str), confidence (str)
    # Normalize finding categories against KNOWN_CATEGORIES


def invoke_llm(prompt: str, mode: str, timeout: int = 60) -> str | None:
    """Call LLM CLI via subprocess. Returns stdout or None.

    Uses stdin (input=) for prompt to avoid ARG_MAX limits.
    """
    if mode == "structural":
        return None

    if mode == "auto":
        cli = shutil.which("claude") or shutil.which("codex")
        if not cli:
            return None
    else:
        cli = shutil.which(mode)  # "claude" or "codex"
        if not cli:
            return None  # caller handles exit code based on mode

    try:
        result = subprocess.run(
            [cli, "-p", "-", "--output-format", "json"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return None
        return result.stdout
    except subprocess.TimeoutExpired:
        return None


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
    from moirai.analyze.align import align_runs, _consensus
    from moirai.analyze.divergence import find_divergence_points
    from moirai.analyze.compress import compress_sequence

    groups, skip_reasons = select_task_groups(runs, task_filter=task_filter)
    n_skipped = len(skip_reasons)
    n_qualifying = len(groups)

    reports = []
    any_llm_failure = False

    for task_id, group_runs in groups.items():
        # Pre-alignment sampling if > max_runs
        if len(group_runs) > max_runs:
            group_runs = _presample(group_runs, max_runs, seed)

        alignment = align_runs(group_runs, level="name")
        divpoints, _ = find_divergence_points(alignment, group_runs)

        # Take top-N by entropy weighted by outcome separation
        top_divpoints = sorted(divpoints, key=lambda d: d.entropy, reverse=True)[:top_n]

        consensus = compress_sequence(...)  # compressed phase notation
        sampled = sample_runs(group_runs, alignment, seed=seed)

        # Optional clustering
        concordance_tau = None
        concordance_p = None
        if cluster:
            from moirai.analyze.cluster import cluster_runs, compute_concordance
            cr = cluster_runs(group_runs)
            conc = compute_concordance(group_runs, cr.labels)
            # Pick the cluster with most runs for tau
            ...

        prompt = build_prompt(task_id, group_runs, alignment, top_divpoints, sampled)
        raw = invoke_llm(prompt, mode, timeout)

        if raw is not None:
            findings, summary, confidence = parse_response(raw)
        else:
            findings, summary, confidence = [], "", ""
            if mode not in ("structural", "auto"):
                any_llm_failure = True

        reports.append(ExplanationReport(
            task_id=task_id,
            n_runs=len(group_runs),
            pass_rate=...,
            findings=findings,
            summary=summary,
            confidence=confidence,
            consensus=consensus,
            divergent_columns=top_divpoints,
            n_qualifying=n_qualifying,
            n_skipped=n_skipped,
            concordance_tau=concordance_tau,
            concordance_p=concordance_p,
        ))

    return reports
```

### A.3 Replace explain command in cli.py

```python
@app.command()
def explain(
    path: Path = typer.Argument(..., help="Path to run file or directory"),
    run_id: str | None = typer.Option(None, "--run", help="Explain a single run by ID"),
    task: str | None = typer.Option(None, "--task", help="Restrict to a specific task_id"),
    top_n: int = typer.Option(5, "--top-n", help="Divergent columns to analyze"),
    mode: str = typer.Option("auto", "--mode", help="auto, claude, codex, or structural"),
    do_cluster: bool = typer.Option(False, "--cluster", help="Enable clustering + concordance"),
    format: str = typer.Option("terminal", "--format", help="Output format: terminal or json"),
    max_runs: int = typer.Option(50, "--max-runs", help="Cap runs per task group"),
    timeout: int = typer.Option(60, "--timeout", help="LLM subprocess timeout"),
    seed: int = typer.Option(42, "--seed", help="Random seed for sampling"),
    # Preserved from existing --run path:
    level: str = typer.Option("type", help="Alignment level for --run mode"),
    threshold: float = typer.Option(0.3, help="Cluster threshold for --run mode"),
    strict: bool = typer.Option(False, help="Treat warnings as errors"),
) -> None:
    """Explain why agent runs succeed or fail.

    Without --run: cross-run differential analysis using LLM content comparison.
    With --run: explain a single run within its cluster context.
    """
    runs = _load_and_filter(path, strict)

    if run_id:
        # === Existing single-run path (inline, unchanged) ===
        # Find target run by ID or prefix match
        # Cluster all runs, find target's cluster
        # Align siblings, find divergence points
        # Show where target diverged
        # (existing code from cli.py:382-527, using level and threshold)
        ...
        return

    # === New cross-run content analysis path ===
    from moirai.analyze.content import run_explain

    if mode not in ("auto", "claude", "codex", "structural"):
        err_console.print(f"[red]error:[/red] unknown mode: {mode}")
        raise typer.Exit(1)

    reports = run_explain(
        runs, mode=mode, task_filter=task, top_n=top_n,
        max_runs=max_runs, timeout=timeout, seed=seed, cluster=do_cluster,
    )

    # Print skip summary
    if reports:
        r0 = reports[0]
        if r0.n_skipped > 0:
            err_console.print(f"[dim]{r0.n_skipped} task groups skipped[/dim]")

    # Auto-mode degradation indicator
    if mode == "auto" and all(r.summary == "" for r in reports) and reports:
        err_console.print("[dim]structural only — no LLM CLI found[/dim]")

    if format == "json":
        import json as json_mod
        from dataclasses import asdict
        # All warnings go to stderr, JSON to stdout
        console.print(json_mod.dumps([asdict(r) for r in reports], indent=2, default=str))
    else:
        from moirai.viz.terminal import print_explanation
        for report in reports:
            print_explanation(report)

    # Exit codes
    if mode not in ("auto", "structural"):
        if any(r.summary == "" for r in reports):
            raise typer.Exit(2)  # LLM failure in explicit mode
```

### A.4 Add print_explanation to viz/terminal.py

```python
def print_explanation(report: ExplanationReport) -> None:
    """Print content-aware explanation report."""
    # Header
    rate_str = f"{report.pass_rate:.0%}"
    console.print(f"\n[bold]{report.task_id}[/bold]: {report.n_runs} runs, {rate_str} pass")

    # Summary (if present — empty in structural mode)
    if report.summary:
        console.print(f"\n{report.summary}")

    # Findings
    if report.findings:
        console.print(f"\n[bold]Findings[/bold] ({len(report.findings)})")
        for f in report.findings:
            color = "red" if f.category in ("error_ignored", "wrong_file") else "yellow"
            console.print(f"  [{color}]{f.category}[/{color}] col {f.column}: {f.description}")
            console.print(f"    [dim]{f.evidence}[/dim]")

    # Divergent columns (always present — the structural layer)
    if report.divergent_columns:
        console.print(f"\n[bold]Divergent positions[/bold]")
        for dp in report.divergent_columns:
            vals = ", ".join(f"{v}: {c}" for v, c in dp.value_counts.items())
            succ = ", ".join(
                f"{v}: {r:.0%}" for v, r in (dp.success_by_value or {}).items()
                if r is not None
            )
            console.print(f"  col {dp.column}: {vals}")
            if succ:
                console.print(f"    [dim]success rates: {succ}[/dim]")

    # Consensus
    console.print(f"\n[dim]Consensus: {report.consensus}[/dim]")

    # Concordance (if --cluster)
    if report.concordance_tau is not None:
        p_str = f", p={report.concordance_p:.3f}" if report.concordance_p is not None else ""
        console.print(f"[dim]Concordance: τ={report.concordance_tau:.2f}{p_str}[/dim]")
```

## Phase B: Tests

### B.1 Unit tests — tests/test_content.py

```python
"""Tests for content-aware analysis."""
from moirai.schema import Step, Result, Run
from moirai.analyze.content import (
    select_task_groups, sample_runs, build_prompt, parse_response, invoke_llm,
)


def _make_run(run_id: str, task_id: str, names: list[str],
              success: bool, output: dict | None = None) -> Run:
    steps = [
        Step(idx=i, type="tool", name=n,
             output=output or {}, attrs={})
        for i, n in enumerate(names)
    ]
    return Run(run_id=run_id, task_id=task_id, steps=steps,
               result=Result(success=success))


class TestSelectTaskGroups:
    def test_mixed_outcomes_qualify(self): ...
    def test_all_same_outcome_skipped(self): ...
    def test_too_few_runs_skipped(self): ...
    def test_task_filter_restricts(self): ...
    def test_task_not_found_error(self): ...


class TestSampleRuns:
    def test_picks_near_and_far(self): ...
    def test_deterministic_with_seed(self): ...
    def test_single_run_class(self): ...


class TestBuildPrompt:
    def test_includes_context_and_columns(self): ...
    def test_truncates_long_content(self): ...
    def test_reduces_to_3_columns_over_budget(self): ...


class TestParseResponse:
    def test_valid_json(self): ...
    def test_strips_markdown_fences(self): ...
    def test_malformed_json_retry(self): ...
    def test_normalizes_known_categories(self): ...


class TestInvokeLlm:
    def test_structural_returns_none(self): ...
```

### B.2 CLI tests — tests/test_cli.py

Update existing `TestExplainCommand`:

```python
class TestExplainCommand:
    # Existing tests preserved with --run flag
    def test_explain_single_run(self): ...        # existing, add --run
    def test_explain_single_run_not_found(self): ...  # existing, add --run

    # New cross-run tests
    def test_explain_structural_mode(self): ...    # --mode structural, exit 0
    def test_explain_no_qualifying_groups(self): ...  # all-pass, exit 0 + message
    def test_explain_json_output(self): ...        # --format json, valid JSON
```

## Acceptance criteria

- [ ] `moirai explain <path>` runs cross-run differential analysis (default behavior)
- [ ] `moirai explain <path> --run <id>` preserves existing single-run behavior with `--level` and `--threshold`
- [ ] `--mode structural` produces deterministic output without subprocess calls
- [ ] `--mode auto` degrades to structural with visible "[structural only]" indicator
- [ ] `--mode claude` / `--mode codex` exits 2 on LLM failure
- [ ] `--cluster` adds concordance tau/p to the report
- [ ] `--format json` produces valid JSON list to stdout, warnings to stderr
- [ ] `--task` restricts to one group; prints "not found" if task_id absent from data
- [ ] `--seed` makes sampling deterministic
- [ ] Existing `explain` CLI tests pass with `--run` flag
- [ ] All new tests pass (~15 total)
- [ ] No changes to align.py, cluster.py, stats.py, compress.py, or converters
- [ ] `analyze/explain.py` left untouched (used by `divergence` command)
