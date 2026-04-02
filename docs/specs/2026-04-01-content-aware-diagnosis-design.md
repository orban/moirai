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

`moirai explain <path>` runs a four-stage pipeline that starts with structural alignment (to pair steps across runs) and then uses an LLM to compare the *content* at steps where runs diverge.

The output is a `ExplanationReport` per task: a list of findings ("failing runs read the wrong file at step 14", "passing runs ran tests after editing, failing runs didn't") plus a narrative summary.

## Pipeline

### Stage 1: Group

Partition runs by `task_id`. A task group qualifies if:
- Mixed outcomes (at least one pass, one fail)
- At least 5 runs with known outcomes

Groups that don't qualify are skipped with a summary line ("3 task groups skipped: all-pass (2), too few runs (1)").

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
class DivergentColumn:
    column: int                # position in alignment
    phase: str                 # phase context ("modify", "verify", etc.)
    values: dict[str, int]     # value distribution
    outcome_split: dict[str, float]  # value -> pass rate

@dataclass
class StructuralBaseline:
    """Always populated — the structural context available without --cluster."""
    consensus: str                         # compressed phase notation
    divergent_columns: list[DivergentColumn]
    n_qualifying: int                      # task groups that qualified
    n_skipped: int                         # task groups skipped

@dataclass
class ClusterContext:
    """Only populated when --cluster is used."""
    concordance_tau: float | None
    concordance_p: float | None
    n_clusters: int
    cluster_labels: dict[str, int]         # run_id -> cluster_id

@dataclass
class ExplanationReport:
    task_id: str
    n_runs: int
    pass_rate: float
    findings: list[ContentFinding]
    summary: str                           # free-form narrative; empty if --mode structural
    baseline: StructuralBaseline           # always present
    cluster: ClusterContext | None          # only with --cluster
```

### Finding categories

The LLM returns free-text categories. `FindingCategory` provides a known set for filtering/aggregation, but the LLM is not constrained to it. When a returned category matches a known value, it's normalized; otherwise it's kept as-is. The taxonomy grows from observed data, not upfront assumptions.

## CLI interface

```
moirai explain <path> [options]

Options:
  --task TEXT        Restrict to a specific task_id
  --top-n INT       Divergent columns to analyze (default: 5)
  --mode TEXT        Analysis mode: auto, claude, codex, or structural (default: auto)
  --cluster         Enable clustering + concordance scoring
  --format json     Machine-readable output (default: terminal)
  --max-runs INT    Cap runs per task group for alignment (default: 50)
  --timeout INT     Seconds before LLM subprocess is killed (default: 60)
  --seed INT        Random seed for sampling reproducibility (default: 42)
```

### Modes

| Mode | Behavior |
|---|---|
| `auto` (default) | Try `claude` CLI, then `codex`. If neither available, fall back to structural-only output. Exit 0. |
| `claude` | Use `claude -p`. If missing/timeout/malformed JSON, print structural output and exit 2. |
| `codex` | Use `codex`. Same failure behavior as `claude`. |
| `structural` | Skip LLM entirely. Deterministic, CI-safe. Exit 0. |

### Exit codes

| Condition | Exit code |
|---|---|
| Findings produced | 0 |
| No qualifying task groups (informational message printed) | 0 |
| `--mode structural` completed successfully | 0 |
| `--mode auto` with no LLM CLI found (structural output printed) | 0 |
| `--mode claude\|codex` with LLM CLI not found (structural output printed) | 2 |
| LLM subprocess timeout (structural output printed, partial findings if any) | 2 |
| LLM returned malformed JSON after retry (structural output printed) | 2 |
| Malformed input, missing path, internal error | 1 |

All exit-2 cases print structural output before exiting. In `auto` mode, LLM unavailability is not a failure — it degrades gracefully and exits 0.

### Sampling

Two independent caps, both using `--seed` for deterministic tie-breaking:

- `--max-runs`: If a task group exceeds this, sample down (stratified by outcome) before alignment. Controls alignment cost (O(n²) pairwise).
- Per-column sampling (stage 3): After divergent columns are found, select 2-4 runs per column for the LLM prompt. Uses alignment distances to pick near-consensus and high-divergence representatives. Controls prompt size.

### Multi-group behavior

Each task group is diagnosed independently. If the LLM fails for one group (timeout, malformed JSON), that group's report contains structural output only and a warning. Other groups continue. The command exits 2 if any group had an LLM failure, 0 if all succeeded.

### Integration with existing commands

`moirai explain` replaces the existing `explain` command. The old single-run behavior is preserved via `--run <id>`, which explains one run within its cluster context (existing logic, moved from inline CLI code into `analyze/content.py`). Without `--run`, the command runs cross-run differential analysis (the new behavior).

## Module structure

```
moirai/
  analyze/
    content.py      (NEW)  — content extraction, prompt building, LLM invocation
  schema.py         (MOD)  — add StepMetadata, ContentFinding, DivergentColumn,
                             StructuralBaseline, ClusterContext, ExplanationReport,
                             FindingCategory
  cli.py            (MOD)  — add diagnose command
  viz/
    terminal.py     (MOD)  — add print_diagnostic_report()
```

### content.py functions

- `extract_metadata(step) -> StepMetadata` — pull structured signals from step content
- `select_task_groups(runs) -> list[TaskGroup]` — partition by task_id, filter qualifying groups
- `sample_runs(runs, alignment, n_per_class) -> list[Run]` — stratified near-consensus + high-divergence
- `build_prompt(task_group, alignment, divergent_columns, sampled_runs) -> str` — assemble prompt
- `parse_response(raw_json) -> ExplanationReport` — validate LLM output; strips markdown fences, retries on malformed JSON once, raises on persistent failure
- `run_diagnosis(runs, options) -> list[ExplanationReport]` — top-level pipeline orchestrator

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
