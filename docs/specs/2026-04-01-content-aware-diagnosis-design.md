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

The output is a `DiagnosticReport` per task: a list of typed findings ("failing runs read the wrong file at step 14", "passing runs ran tests after editing, failing runs didn't") plus a narrative summary.

## Pipeline

### Stage 1: Group

Partition runs by `task_id`. A task group qualifies if:
- Mixed outcomes (at least one pass, one fail)
- At least 5 runs with known outcomes

Groups that don't qualify are skipped silently.

### Stage 2: Align

Run existing `align_runs()` at the enriched-name level within each qualifying task group.

If a task group has high structural heterogeneity, trigger a clustering split first (`cluster_runs()`), then align within each sub-group. This is automatic, not user-configured.

Heterogeneity is detected by computing the condensed NW distance matrix (already needed for alignment) and checking: if the mean pairwise distance > 0.7 (on the 0-1 NW distance scale), or if sequence lengths have a >2x ratio between the 25th and 75th percentile, the group is heterogeneous. These thresholds are constants in `content.py`, not user-configurable.

### Stage 3: Diverge + sample

Find divergent columns using existing entropy-based detection. Select top-N columns (default N=5) by entropy weighted by outcome separation.

For each selected column, sample runs using stratified selection:
- One near-consensus run per outcome class (pass/fail)
- One high-divergence run per outcome class
- Resulting in 2-4 sampled runs per column

### Stage 4: Analyze

Build a prompt with hybrid context and pipe through `claude -p` or `codex` CLI.

#### Content preparation

For each step at a divergent column, extract structured metadata that's always preserved regardless of truncation:
- `error`: error string or traceback if present in result
- `file_path`: from attrs or parsed from tool_input
- `command`: from attrs or parsed from tool_input
- `test_outcome`: pass/fail/error if result contains test output
- `output_status`: empty, truncated, error, or normal

Raw content fields (`reasoning`, `result`, `tool_input`) are included with generous limits (~800 chars for reasoning and result, ~400 chars for tool_input).

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

Total prompt target: ~8K tokens per task group. If the assembled prompt exceeds this, reduce by: (1) drop `tool_input` fields, (2) reduce raw content limits to ~400 chars, (3) reduce to top-3 columns. The reduction is automatic and logged.

#### Fallback on truncation

If the LLM returns low-confidence findings or flags insufficient context, moirai re-runs with a summarization pass: a first LLM call summarizes the flagged columns' content, then the analysis runs again on summaries instead of truncated raw text. If still low confidence after retry, findings are kept but flagged.

#### Graceful degradation

If no LLM CLI is available, moirai prints the structural analysis (alignment, divergent columns, outcome splits, extracted metadata) without the LLM synthesis. Still useful — you see "at position 14, failing runs hit `FileNotFoundError` on `tests/conftest.py` while passing runs read it successfully." No crash, just no narrative explanation.

## Output schema

```python
@dataclass
class ContentFinding:
    category: str          # from fixed taxonomy below
    severity: str          # "high", "medium", "low"
    column: int            # alignment column where observed
    description: str       # one-sentence explanation
    evidence: str          # specific content from runs
    pass_runs: list[str]   # run_ids with passing behavior
    fail_runs: list[str]   # run_ids with failing behavior

@dataclass
class DiagnosticReport:
    task_id: str
    n_runs: int
    pass_rate: float
    findings: list[ContentFinding]    # sorted by severity desc
    summary: str                       # free-form narrative diagnosis
    confidence: str                    # "high", "medium", "low"
    structural_metadata: dict          # concordance τ, consensus, etc.
```

### Finding categories

| Category | Meaning |
|---|---|
| `wrong_file` | Agent read/edited the wrong file |
| `missing_test` | Agent didn't verify its changes |
| `error_ignored` | Agent saw an error and didn't act on it |
| `wrong_command` | Agent used an ineffective tool/command |
| `reasoning_gap` | Agent's reasoning missed the actual problem |
| `other` | Doesn't fit above categories |

Kept intentionally small so findings are filterable and aggregatable across tasks. The `other` category catches novel patterns; if a category appears frequently in `other`, it gets promoted to a named category in a future version.

## CLI interface

```
moirai diagnose <path> [options]

Options:
  --task TEXT        Restrict to a specific task_id
  --top-n INT       Divergent columns to analyze (default: 5)
  --llm TEXT        CLI to use: "claude" or "codex" (default: "claude")
  --structural      Enable clustering + concordance scoring
  --output json     Machine-readable output (default: terminal)
  --max-runs INT    Cap runs per task group for alignment (default: 50)
```

### Exit codes

- 0: findings produced
- 1: no qualifying task groups
- 2: LLM CLI not available (structural output still printed)

### Two-stage sampling

There are two distinct sampling operations:

1. **Pre-alignment sampling** (`--max-runs`): If a task group has more than `max-runs` runs, sample down (stratified by outcome) before computing the distance matrix and alignment. This controls alignment cost (O(n^2) pairwise).

2. **Per-column sampling** (stage 3): After divergent columns are identified from the full alignment, select 2-4 specific runs per column for LLM context. This controls prompt size. Uses the alignment distances to pick near-consensus and high-divergence representatives.

These are independent. The first is about computational cost, the second is about prompt construction.

### Integration with existing commands

`moirai diagnose` is a new top-level command. It internally calls `align_runs()` and divergence detection but doesn't depend on `cluster_runs()` unless `--structural` is passed. No changes to existing commands.

## Module structure

```
moirai/
  analyze/
    content.py      (NEW)  — content extraction, prompt building, LLM invocation
  schema.py         (MOD)  — add ContentFinding, DiagnosticReport
  cli.py            (MOD)  — add diagnose command
  viz/
    terminal.py     (MOD)  — add print_diagnostic_report()
```

### content.py functions

- `extract_metadata(step) -> dict` — pull structured signals from step content
- `select_task_groups(runs) -> list[TaskGroup]` — partition by task_id, filter qualifying groups
- `detect_heterogeneity(alignment) -> bool` — check if task group needs clustering split
- `sample_runs(runs, alignment, n_per_class) -> list[Run]` — stratified near-consensus + high-divergence
- `build_prompt(task_group, alignment, divergent_columns, sampled_runs) -> str` — assemble prompt
- `parse_response(raw_json) -> DiagnosticReport` — validate and parse LLM output
- `run_diagnosis(runs, options) -> list[DiagnosticReport]` — top-level pipeline orchestrator

### What stays untouched

`align.py`, `cluster.py`, `stats.py`, `compress.py`, all converters. The content layer reads from existing data structures without modifying the structural pipeline.

### Dependencies

No new Python dependencies. LLM interaction is a subprocess call to `claude` or `codex` CLI. JSON parsing uses stdlib.

## Relationship to structural analysis

The structural layer (alignment, clustering, concordance) becomes scaffolding for content analysis, not the analysis itself:

- **Alignment** is load-bearing — it pairs steps across runs so content comparison is position-specific rather than whole-trajectory
- **Divergent column detection** is load-bearing — it focuses the LLM on the decision points that matter
- **Clustering** is optional — triggers automatically only when a task group is structurally heterogeneous, or when the user passes `--structural`
- **Concordance** is optional — reported as metadata when `--structural` is used, doesn't gate analysis

## Future: full LLM comparison layer

The interface is designed so that the prompt-building and response-parsing functions can be swapped for a more sophisticated LLM interaction:
- Multi-turn: the LLM asks for more context on specific columns
- Full-trajectory: send complete run content instead of sampled columns
- Comparative: "here are two runs, explain why one passed and one failed"

The `build_prompt` / `parse_response` interface isolates this — the rest of the pipeline doesn't change when the LLM interaction gets richer.
