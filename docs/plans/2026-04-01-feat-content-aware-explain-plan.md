---
title: "feat: content-aware explain command"
type: feat
date: 2026-04-01
spec: docs/specs/2026-04-01-content-aware-diagnosis-design.md
---

# Content-Aware Explain Command

## Overview

Replace the existing `moirai explain` command with a content-aware version that answers "why do runs fail?" by comparing step content (reasoning, tool inputs, results) at structurally divergent positions across runs. The existing single-run explain behavior is preserved via `--run <id>`.

Spec: `docs/specs/2026-04-01-content-aware-diagnosis-design.md`

## Phase 1: Schema + content extraction

Add dataclasses to `schema.py` and build the content extraction layer. No CLI changes yet.

### 1.1 Add dataclasses to schema.py

Add after the `# --- Evidence and diagnosis dataclasses ---` section:

```python
# --- Content-aware analysis dataclasses ---

class FindingCategory(StrEnum):
    WRONG_FILE = "wrong_file"
    MISSING_TEST = "missing_test"
    ERROR_IGNORED = "error_ignored"
    WRONG_COMMAND = "wrong_command"
    REASONING_GAP = "reasoning_gap"
    OTHER = "other"

@dataclass
class StepMetadata:
    error: str | None
    file_path: str | None
    command: str | None
    test_outcome: str | None
    output_status: str

@dataclass
class DivergentColumn:
    column: int
    phase: str
    values: dict[str, int]
    outcome_split: dict[str, float]

@dataclass
class ContentFinding:
    category: str
    column: int
    description: str
    evidence: str
    pass_runs: list[str]
    fail_runs: list[str]

@dataclass
class StructuralBaseline:
    consensus: str
    divergent_columns: list[DivergentColumn]
    n_qualifying: int
    n_skipped: int

@dataclass
class ClusterContext:
    concordance_tau: float | None
    concordance_p: float | None
    n_clusters: int
    cluster_labels: dict[str, int]

@dataclass
class ExplanationReport:
    task_id: str
    n_runs: int
    pass_rate: float
    findings: list[ContentFinding]
    summary: str
    baseline: StructuralBaseline
    cluster: ClusterContext | None
```

### 1.2 Create moirai/analyze/content.py

Module with these functions:

```python
"""Content-aware analysis — LLM-assisted differential diagnosis of agent traces."""

def extract_metadata(step: Step) -> StepMetadata
    # Parse step.output["result"] for error strings/tracebacks
    # Pull file_path from step.attrs or step.output["tool_input"]
    # Pull command from step.attrs
    # Detect test pass/fail in result text
    # Classify output_status: empty, truncated, error, normal

def select_task_groups(runs: list[Run], min_runs: int = 5) -> list[TaskGroup]
    # Partition by task_id
    # Filter: mixed outcomes, >= min_runs known outcomes
    # Return with skip reasons for reporting

def sample_runs(
    runs: list[Run],
    alignment: Alignment,
    n_per_class: int = 2,
    seed: int = 42,
) -> list[Run]
    # Compute per-run distance from consensus
    # For each outcome class (pass/fail):
    #   Pick the run closest to consensus
    #   Pick the run furthest from consensus
    # Use seed for deterministic tie-breaking

def build_prompt(
    task_id: str,
    runs: list[Run],
    alignment: Alignment,
    divergent_columns: list[DivergentColumn],
    sampled_runs: list[Run],
) -> str
    # Context block: task_id, run count, pass/fail split, consensus
    # For each divergent column:
    #   Position, phase context, value distribution, outcome split
    #   For each sampled run at this column:
    #     Structured metadata (always preserved)
    #     Raw reasoning (~800 chars)
    #     Raw result (~800 chars)
    #     Raw tool_input (~400 chars)
    #     One step before/after
    # Response format instruction with JSON schema
    # If prompt > ~8K tokens, reduce to top-3 columns

def parse_response(raw: str) -> list[ContentFinding]
    # Strip markdown fences if present
    # json.loads
    # Retry once on malformed JSON (strip to first { ... last })
    # Validate required fields
    # Normalize category to FindingCategory when possible

def invoke_llm(prompt: str, mode: str, timeout: int) -> str | None
    # mode: "auto", "claude", "codex", "structural"
    # If "structural": return None
    # If "auto": try claude, then codex, then return None
    # subprocess.run(["claude", "-p", prompt], capture_output=True,
    #                text=True, timeout=timeout)
    # Return stdout or None on failure

def run_explain(
    runs: list[Run],
    mode: str = "auto",
    task_filter: str | None = None,
    top_n: int = 5,
    max_runs: int = 50,
    timeout: int = 60,
    seed: int = 42,
    cluster: bool = False,
) -> list[ExplanationReport]
    # Orchestrator: group -> align -> diverge -> sample -> prompt -> LLM -> parse
```

### 1.3 Tests for content extraction

File: `tests/test_content.py`

```
class TestExtractMetadata:
    test_error_in_result — step with traceback in output["result"]
    test_file_path_from_attrs — step with attrs["file_path"]
    test_file_path_from_tool_input — parsed from output["tool_input"]
    test_test_outcome_pass — result containing "PASSED" or similar
    test_test_outcome_fail — result containing "FAILED"
    test_empty_output — step with no output
    test_normal_output — step with normal content

class TestSelectTaskGroups:
    test_mixed_outcomes — returns groups with both pass and fail
    test_all_same_outcome — skips all-pass and all-fail
    test_too_few_runs — skips groups with < 5 runs
    test_task_filter — --task restricts to one group

class TestSampleRuns:
    test_picks_near_and_far — one near-consensus, one far per class
    test_deterministic_with_seed — same seed = same selection
    test_small_class — class with 1 run still works

class TestBuildPrompt:
    test_includes_context_block — task_id, counts, consensus present
    test_includes_divergent_columns — column content in prompt
    test_truncation — long content truncated to limits
    test_budget_reduction — > 8K tokens reduces to 3 columns

class TestParseResponse:
    test_valid_json — parses correctly
    test_markdown_fences — strips ```json ... ```
    test_malformed_json — retries extraction
    test_category_normalization — "wrong_file" maps to FindingCategory
```

## Phase 2: CLI integration

### 2.1 Refactor existing explain command

The current `explain` command (cli.py:382-527) has inline analysis logic. Extract the single-run analysis into a function in `analyze/content.py`:

```python
def explain_single_run(
    target_run: Run,
    all_runs: list[Run],
    mode: str = "auto",
    timeout: int = 60,
) -> ExplanationReport
    # Existing logic: cluster, find target's cluster, align siblings,
    # find divergence points, show where target diverged
    # Wrap result in ExplanationReport with single-run findings
```

### 2.2 Replace explain command in cli.py

```python
@app.command()
def explain(
    path: Path = typer.Argument(..., help="Path to run file or directory"),
    run: str | None = typer.Option(None, "--run", help="Explain a single run by ID"),
    task: str | None = typer.Option(None, "--task", help="Restrict to a specific task_id"),
    top_n: int = typer.Option(5, "--top-n", help="Divergent columns to analyze"),
    mode: str = typer.Option("auto", "--mode", help="auto, claude, codex, or structural"),
    do_cluster: bool = typer.Option(False, "--cluster", help="Enable clustering + concordance"),
    format: str = typer.Option("terminal", "--format", help="Output format: terminal or json"),
    max_runs: int = typer.Option(50, "--max-runs", help="Cap runs per task group"),
    timeout: int = typer.Option(60, "--timeout", help="LLM subprocess timeout in seconds"),
    seed: int = typer.Option(42, "--seed", help="Random seed for sampling"),
    strict: bool = typer.Option(False, help="Treat warnings as errors"),
) -> None:
    """Explain why agent runs succeed or fail.

    Without --run: cross-run differential analysis using LLM content comparison.
    With --run: explain a single run within its cluster context.
    """
    runs = _load_and_filter(path, strict)

    if run:
        # Single-run path (existing behavior, refactored)
        from moirai.analyze.content import explain_single_run
        report = explain_single_run(target_run, runs, mode=mode, timeout=timeout)
        reports = [report]
    else:
        # Cross-run content analysis (new behavior)
        from moirai.analyze.content import run_explain
        reports = run_explain(
            runs, mode=mode, task_filter=task, top_n=top_n,
            max_runs=max_runs, timeout=timeout, seed=seed, cluster=do_cluster,
        )

    if format == "json":
        # JSON output
    else:
        from moirai.viz.terminal import print_explanation
        for report in reports:
            print_explanation(report)
```

### 2.3 Update existing CLI tests

File: `tests/test_cli.py`

The existing `TestExplainCommand` class (lines 120-140) tests the old single-run behavior. Update these tests to use the new `--run` flag, and add tests for the new cross-run behavior:

```
class TestExplainCommand:
    # Existing tests updated to use --run flag
    test_explain_single_run — existing test with --run <id>
    test_explain_single_run_not_found — existing test

    # New tests
    test_explain_cross_run — basic cross-run analysis with --mode structural
    test_explain_no_qualifying_groups — all-pass dataset, exit 0 with message
    test_explain_json_output — --format json produces valid JSON
    test_explain_task_filter — --task restricts to one group
    test_explain_structural_mode — --mode structural skips LLM, exit 0
```

## Phase 3: Terminal display

### 3.1 Add print_explanation to viz/terminal.py

```python
def print_explanation(report: ExplanationReport) -> None:
    # Header: task_id, run count, pass rate
    # Skip reasons if n_skipped > 0

    # Summary (if present — empty in structural mode)
    if report.summary:
        console.print(f"\n[bold]Summary[/bold]")
        console.print(report.summary)

    # Findings
    if report.findings:
        console.print(f"\n[bold]Findings[/bold] ({len(report.findings)})")
        for f in report.findings:
            color = "red" if f.category in ("error_ignored", "wrong_file") else "yellow"
            console.print(f"  [{color}]{f.category}[/{color}] column {f.column}: {f.description}")
            console.print(f"    [dim]{f.evidence}[/dim]")
            console.print(f"    pass: {', '.join(f.pass_runs)} | fail: {', '.join(f.fail_runs)}")

    # Structural baseline (always present)
    console.print(f"\n[dim]Consensus: {report.baseline.consensus}[/dim]")
    for dc in report.baseline.divergent_columns:
        vals = ", ".join(f"{v}: {c}" for v, c in dc.values.items())
        console.print(f"  [dim]col {dc.column} ({dc.phase}): {vals}[/dim]")

    # Cluster context (if --cluster)
    if report.cluster:
        tau = report.cluster.concordance_tau
        p = report.cluster.concordance_p
        if tau is not None:
            p_str = f", p={p:.3f}" if p is not None else ""
            console.print(f"  [dim]concordance: τ={tau:.2f}{p_str}[/dim]")
```

## Phase 4: End-to-end integration

### 4.1 Wire up the orchestrator

`run_explain` in `content.py` ties everything together:

1. `select_task_groups(runs)` — partition and filter
2. For each qualifying group:
   a. Pre-alignment sampling if > max_runs (stratified by outcome, seeded)
   b. `align_runs(group_runs, level="name")`
   c. `find_divergence_points(alignment, group_runs)` — existing function
   d. Convert to `DivergentColumn` list, take top-N by entropy × outcome separation
   e. `sample_runs(group_runs, alignment, seed=seed)`
   f. `build_prompt(...)` 
   g. `invoke_llm(prompt, mode, timeout)`
   h. If LLM returned content: `parse_response(raw)` → findings + summary
   i. If LLM failed or structural mode: findings=[], summary=""
   j. Build `ExplanationReport` with baseline always populated
3. Return list of reports

### 4.2 Integration tests

File: `tests/test_content.py` (append to existing)

```
class TestRunExplain:
    test_structural_mode_no_subprocess — mode="structural" never calls subprocess
    test_auto_mode_fallback — mode="auto" with no CLI returns structural-only report
    test_report_has_baseline — StructuralBaseline always populated
    test_report_findings_empty_without_llm — findings=[] in structural mode
    test_max_runs_caps_alignment — > max_runs triggers pre-sampling
    test_seed_determinism — same seed = same report
```

## Acceptance criteria

- [ ] `moirai explain <path>` runs cross-run differential analysis (default behavior)
- [ ] `moirai explain <path> --run <id>` preserves existing single-run behavior
- [ ] `--mode structural` produces deterministic output without LLM calls
- [ ] `--mode auto` tries claude, then codex, then degrades to structural
- [ ] `--mode claude` / `--mode codex` exits 2 on LLM failure
- [ ] `--cluster` adds concordance metadata to the report
- [ ] `--format json` produces valid JSON matching ExplanationReport schema
- [ ] `--task` restricts analysis to a specific task group
- [ ] `--seed` makes sampling deterministic
- [ ] Existing `explain` CLI tests pass with `--run` flag
- [ ] All new tests pass
- [ ] No changes to align.py, cluster.py, stats.py, compress.py, or converters
