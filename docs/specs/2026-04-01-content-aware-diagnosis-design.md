---
title: "Content-aware diagnosis — LLM-assisted differential analysis of agent traces"
type: feat
date: 2026-04-01
origin: "Concordance scoring results across 3 datasets (~2000 runs) showing |τ| < 0.1 for most clusters — structural similarity doesn't predict outcomes"
---

# Content-aware diagnosis

## The problem

moirai's structural analysis (clustering by tool-call sequence similarity, divergence detection, motif extraction) doesn't explain outcomes for the majority of populations. Concordance scoring across eval_harness (483 runs), swe_smith (1000 runs), and swe_agent (500 runs) shows |τ| < 0.1 in nearly every cluster. Two runs can follow identical tool sequences — read, search, edit, test — and have opposite outcomes because the *content* of those calls (which file was read, what was edited, what the test returned) is where the signal lives.

The structural layer captures *what tools were called*. It misses *what those tools did*. Content-aware diagnosis fills that gap.

## What it does

`moirai diagnose <path>` runs a four-stage pipeline that starts with structural alignment (to pair steps across runs) and then uses an LLM to compare the *content* at steps where runs diverge.

The output is a `DiagnosticReport` per task: a list of findings ("failing runs read the wrong file at step 14", "passing runs ran tests after editing, failing runs didn't") plus a narrative summary.

## Pipeline

### Stage 1: Group

Partition runs by `task_id`. A task group qualifies if:
- Mixed outcomes (at least one pass, one fail)
- At least 5 runs with known outcomes

Groups that don't qualify are skipped silently.

### Stage 2: Align

Run existing `align_runs()` at the enriched-name level within each qualifying task group. No clustering split — just align the task group as-is.

### Stage 3: Diverge + sample

Find divergent columns using existing entropy-based detection. Select top-N columns (default N=5) by entropy weighted by outcome separation.

For each selected column, sample runs using stratified selection:
- One near-consensus run per outcome class (pass/fail)
- One high-divergence run per outcome class
- Resulting in 2-4 sampled runs per column

### Stage 4: Analyze

Build a prompt with hybrid context and pipe through `claude -p` or `codex` CLI.

#### Content preparation

For each step at a divergent column, extract structured metadata into a typed `StepMetadata`:
- `error`: error string or traceback if present in result
- `file_path`: from attrs or parsed from tool_input
- `command`: from attrs or parsed from tool_input
- `test_outcome`: pass/fail/error if result contains test output
- `output_status`: empty, truncated, error, or normal

These fields are always preserved regardless of prompt budget. Raw content fields (`reasoning`, `result`, `tool_input`) are included with generous limits (~800 chars for reasoning and result, ~400 chars for tool_input).

#### Prompt structure

```
1. Context block
   - Task ID, run count, pass/fail split
   - Consensus sequence (compressed phase notation)

2. Divergent columns (top-N)
   For each column:
   - Position and phase context ("step 14 of 22, during modify phase")
   - Value distribution + outcome split
   - Sampled runs:
     Per run:
       - Run ID, outcome
       - Structured metadata (error, file_path, command, test_outcome)
       - Raw reasoning (~800 chars)
       - Raw result (~800 chars)
       - Raw tool_input (~400 chars)
       - One step before/after for local context

3. Response format instruction (JSON schema)
```

Target: ~8K tokens per task group. If the assembled prompt exceeds this, reduce to top-3 columns.

#### Graceful degradation

If no LLM CLI is available, moirai prints the structural analysis (alignment, divergent columns, outcome splits, extracted metadata) without the LLM synthesis. Still useful — you see "at position 14, failing runs hit `FileNotFoundError` on `tests/conftest.py` while passing runs read it successfully." No crash, just no narrative explanation.

## Output schema

```python
from enum import StrEnum

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
    test_outcome: str | None       # "pass", "fail", "error", or None
    output_status: str             # "empty", "truncated", "error", "normal"

@dataclass
class ContentFinding:
    category: str              # free-text from LLM; mapped to FindingCategory when possible
    column: int                # alignment column where observed
    description: str           # one-sentence explanation
    evidence: str              # specific content from runs
    pass_runs: list[str]       # run_ids with passing behavior
    fail_runs: list[str]       # run_ids with failing behavior

@dataclass
class StructuralContext:
    consensus: str             # compressed phase notation
    concordance_tau: float | None
    concordance_p: float | None

@dataclass
class DiagnosticReport:
    task_id: str
    n_runs: int
    pass_rate: float
    findings: list[ContentFinding]
    summary: str                           # free-form narrative diagnosis
    structural: StructuralContext | None    # populated when --cluster is used
```

### Finding categories

The LLM returns free-text categories. `FindingCategory` provides a known set for filtering/aggregation, but the LLM is not constrained to it. When a returned category matches a known value, it's normalized; otherwise it's kept as-is. The taxonomy grows from observed data, not upfront assumptions.

## CLI interface

```
moirai diagnose <path> [options]

Options:
  --task TEXT        Restrict to a specific task_id
  --top-n INT       Divergent columns to analyze (default: 5)
  --llm TEXT        CLI to use: "claude" or "codex" (default: "claude")
  --cluster         Enable clustering + concordance scoring
  --format json     Machine-readable output (default: terminal)
  --max-runs INT    Cap runs per task group for alignment (default: 50)
  --timeout INT     Seconds before LLM subprocess is killed (default: 60)
```

### Exit codes

- 0: findings produced, or no qualifying task groups (with informational message)
- 1: error (malformed input, CLI failure)
- 2: LLM CLI not available (structural output still printed)

### Sampling

Two independent caps:

- `--max-runs`: If a task group exceeds this, sample down (stratified by outcome) before alignment. Controls alignment cost (O(n²) pairwise).
- Per-column sampling (stage 3): After divergent columns are found, select 2-4 runs per column for the LLM prompt. Uses alignment distances to pick near-consensus and high-divergence representatives. Controls prompt size.

### Integration with existing commands

`moirai diagnose` is a new top-level command. It internally calls `align_runs()` and divergence detection but doesn't depend on `cluster_runs()` unless `--cluster` is passed. No changes to existing commands.

## Module structure

```
moirai/
  analyze/
    content.py      (NEW)  — content extraction, prompt building, LLM invocation
  schema.py         (MOD)  — add StepMetadata, ContentFinding, StructuralContext,
                             DiagnosticReport, FindingCategory
  cli.py            (MOD)  — add diagnose command
  viz/
    terminal.py     (MOD)  — add print_diagnostic_report()
```

### content.py functions

- `extract_metadata(step) -> StepMetadata` — pull structured signals from step content
- `select_task_groups(runs) -> list[TaskGroup]` — partition by task_id, filter qualifying groups
- `sample_runs(runs, alignment, n_per_class) -> list[Run]` — stratified near-consensus + high-divergence
- `build_prompt(task_group, alignment, divergent_columns, sampled_runs) -> str` — assemble prompt
- `parse_response(raw_json) -> DiagnosticReport` — validate LLM output; strips markdown fences, retries on malformed JSON once, raises on persistent failure
- `run_diagnosis(runs, options) -> list[DiagnosticReport]` — top-level pipeline orchestrator

### What stays untouched

`align.py`, `cluster.py`, `stats.py`, `compress.py`, all converters. The content layer reads from existing data structures without modifying the structural pipeline.

### Dependencies

No new Python dependencies. LLM interaction uses `subprocess.run()` with `input=` for prompt text and `timeout=` for the kill deadline. JSON parsing uses stdlib.

## Relationship to structural analysis

The structural layer becomes scaffolding for content analysis, not the analysis itself:

- **Alignment** is load-bearing — it pairs steps across runs so content comparison is position-specific
- **Divergent column detection** is load-bearing — it focuses the LLM on the decision points that matter
- **Clustering** is optional — available via `--cluster` for cases where a task group has fundamentally different strategies
- **Concordance** is optional — reported in `StructuralContext` when `--cluster` is used
