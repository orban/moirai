# moirai v0 design spec

Trajectory-level debugging for stochastic agent systems.

## Problem

Agent evaluations report final metrics (pass/fail, score) but hide *how* the agent got there. When 30% of runs fail, you can't tell if they all fail the same way or five different ways. When a harness change improves pass rate, you can't tell if it fixed the failure mode or just shuffled it. Final metrics are lossy summaries of rich behavioral data.

## v0 goal

Given a directory of agent run traces for the same task family, a user can:

1. Load and normalize them
2. Inspect individual runs
3. Cluster runs by structural similarity
4. See where trajectories diverge and how divergence correlates with outcome
5. Compare two cohorts (harness A vs B, model X vs Y)

That is v0. Nothing else.

## v0 user

A technical person running repeated agent experiments locally or in CI who wants to understand why outcomes differ across runs.

## Use cases

- **Single bad run**: "Show me the steps and where this run went wrong."
- **Repeated runs**: "Why do 30% fail and 70% succeed?"
- **Harness comparison**: "Did harness v2 fix failures or shift them?"
- **Model comparison**: "Did the new model behave differently or just cost more?"

## Hard boundaries

**v0 supports:**
- Offline JSON trace files
- Repeated runs of similar tasks
- Step-type sequence analysis
- Clustering by structural similarity
- Rich terminal output, optional HTML export

**v0 does not support:**
- Distributed collection, OpenTelemetry, real-time streaming
- Auth, multi-user, hosted dashboard
- Semantic embedding analysis, full prompt diffing
- Plugin SDK, arbitrary execution engines
- Video replay, per-token tracing

---

## Data model

Three levels: Run, Step, Result.

### Run

```json
{
  "run_id": "run_001",
  "task_id": "task_bugfix_123",
  "task_family": "bugfix",
  "agent": "swe_agent",
  "model": "claude-3.7-sonnet",
  "harness": "baseline_v1",
  "timestamp": "2026-03-24T12:00:00Z",
  "tags": {
    "branch": "main",
    "dataset": "mini_bugs",
    "cohort": "baseline"
  },
  "steps": [],
  "result": {}
}
```

**Required:** `run_id`, `task_id`, `steps`, `result`

**Optional (strongly recommended):** `task_family`, `agent`, `model`, `harness`, `timestamp`, `tags`

### Step

```json
{
  "idx": 0,
  "type": "llm",
  "name": "reason",
  "status": "ok",
  "input": {"summary": "User asks how to fix failing unit test"},
  "output": {"summary": "Agent decides to inspect repository files"},
  "metrics": {"tokens_in": 412, "tokens_out": 137, "latency_ms": 1840},
  "attrs": {"tool_name": null, "file_path": null}
}
```

**Required:** `idx`, `type`, `name`

**Optional:** `status`, `input`, `output`, `metrics`, `attrs`

`input` and `output` are arbitrary dicts. Don't over-constrain them.

### Allowed step types

`llm`, `tool`, `system`, `memory`, `compaction`, `judge`, `error`, `handoff`

Unknown types normalize to `other`.

### Result

```json
{
  "success": true,
  "score": 0.91,
  "label": "pass",
  "error_type": null,
  "summary": "Patched bug and tests passed"
}
```

**Required:** `success` (bool or null)

**Optional:** `score`, `label`, `error_type`, `summary`

`success: null` is allowed in non-strict mode (excluded from success rate calculations). Strict mode rejects null.

---

## Internal representation

```python
@dataclass
class Step:
    idx: int
    type: str
    name: str
    status: str = "ok"
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)

@dataclass
class Result:
    success: bool | None
    score: float | None = None
    label: str | None = None
    error_type: str | None = None
    summary: str | None = None

@dataclass
class Run:
    run_id: str
    task_id: str
    task_family: str | None = None
    agent: str | None = None
    model: str | None = None
    harness: str | None = None
    timestamp: str | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)
    result: Result = field(default_factory=lambda: Result(success=None))
```

---

## Normalization rules

`normalize_run()` applies these transformations:

**Step ordering:** Sort by `idx`. If `idx` missing, assign by sequence position.

**Step type aliases:**
- `assistant`, `completion` → `llm`
- `function_call`, `retrieval` → `tool`
- `compress` → `compaction`
- `critic` → `judge`
- Unknown → `other`

**Missing status:** Default to `"ok"`.

**Missing result (strict mode):** Reject the file. **Non-strict:** Infer `success=false` if any error step exists, otherwise `success=null`.

**Metric normalization:** Flatten nested metric structures into `tokens_in`, `tokens_out`, `latency_ms`.

**Sequence signature:** Compute for each run:
- Type sequence: `["llm", "tool", "llm", "judge"]`
- Name sequence: `["plan", "retrieve_docs", "reason", "verify"]`
- Compact signature: `llm:plan > tool:retrieve_docs > llm:reason > judge:verify`

---

## Analysis primitives

### Summary stats

```python
summarize_runs(runs: list[Run]) -> RunSummary
```

Outputs: run count, success rate, avg/median steps, avg tokens in/out, avg latency, top signatures, error type counts.

### Sequence extraction

```python
step_type_sequence(run: Run) -> list[str]
step_name_sequence(run: Run) -> list[str]
signature(run: Run) -> str
```

### Cohort filtering

```python
filter_runs(runs, model=None, harness=None, task_family=None, tags=None) -> list[Run]
```

Exact match on model/harness/task_family. Key-value match on tags.

### Alignment

```python
align_runs(runs: list[Run], level: str = "type") -> Alignment
```

Uses Needleman-Wunsch global sequence alignment. Levels: `type`, `name`.

Produces aligned columns with gap markers. Handles insertions/deletions properly — runs of different lengths align correctly via gap penalties.

Example output:

| col | run1  | run2  | run3  |
|-----|-------|-------|-------|
| 0   | llm   | llm   | llm   |
| 1   | tool  | tool  | tool  |
| 2   | llm   | llm   | llm   |
| 3   | judge | llm   | judge |
| 4   | -     | judge | -     |

For pairwise alignment: standard Needleman-Wunsch with match=1, mismatch=-1, gap=-1. For multi-run alignment: progressive alignment — align most similar pair first, then align remaining runs against the consensus.

### Divergence detection

```python
find_divergence_points(alignment: Alignment) -> list[DivergencePoint]
```

Each point includes: column index, value frequencies, entropy, downstream success rates per branch value.

### Distance

```python
trajectory_distance(run_a: Run, run_b: Run, level: str = "type") -> float
```

Default: normalized Levenshtein distance over step type sequence. Secondary: type+name hybrid (weight type 0.7, name 0.3).

### Clustering

```python
cluster_runs(runs: list[Run], level: str = "type", method: str = "agglomerative", threshold: float = 0.3) -> ClusterResult
```

Pairwise distance matrix → agglomerative clustering with user-specified threshold. Each cluster reports: id, count, success rate, prototype signature, avg length, error types.

### Cohort diff

```python
compare_cohorts(a_runs: list[Run], b_runs: list[Run]) -> CohortDiff
```

Outputs: success rate delta, avg steps delta, avg tokens delta, avg latency delta, cluster distribution delta, divergence point shifts, signatures unique to each cohort.

---

## CLI

Built with Typer. Package name: `moirai`. Command: `moirai`.

### Commands

```
moirai validate <path>        # Check JSON validity and schema compliance
moirai summary <path>         # Aggregate stats across runs
moirai trace <run.json>       # Inspect single run
moirai branch <dir>           # Multi-run divergence analysis
moirai clusters <dir>         # Cluster runs by trajectory structure
moirai diff <dir> --a K=V --b K=V  # Compare two cohorts
```

### Common flags

- `--level type|name` (default: type)
- `--strict` (reject invalid files instead of warning)
- `--html <path>` (write HTML output)

### Command-specific flags

**branch:** `--task-family`, `--model`, `--harness`

**clusters:** `--threshold 0.3`

**diff:** `--a harness=baseline_v1 --b harness=router_v2`

### Input conventions

- Single file: `run.json`
- Directory: `runs/*.json` (recursive)
- Invalid files: warn and skip (unless `--strict`)

---

## Visualization

Terminal: Rich tables and formatted text.
HTML: Plotly.

### trace

Run metadata + result + ordered step table (idx, type, name, status, tokens_in, tokens_out, latency_ms). Optional `--expand` shows input/output summaries.

### branch

HTML: Sankey-style frequency graph. Nodes = aligned positions, edges = transitions, edge weight = frequency, node color = downstream success rate.
Terminal fallback: top divergence points table.

### clusters

Terminal: cluster list with counts, success rates, prototypes.
HTML: bar chart of cluster sizes and success rates.

### diff

Terminal: metric deltas, cluster shifts, signature differences.
HTML: side-by-side success rates, step length distributions, cluster composition.

---

## Example output

### summary

```
Runs: 42
Success rate: 61.9%
Avg steps: 7.8
Median steps: 7
Avg tokens in/out: 2310 / 710
Avg latency: 12.4s

Top signatures:
1. llm > tool > llm > judge (12)
2. llm > tool > tool > llm > judge (9)
3. llm > llm > tool > judge (7)

Errors:
tool_timeout: 6
loop_detected: 4
judge_false_positive: 2
```

### divergence

```
Top divergence points:
col=3 entropy=1.32
  judge: 14 runs, success 92.9%
  tool: 8 runs, success 37.5%
  llm: 5 runs, success 20.0%

col=5 entropy=0.98
  compaction: 6 runs, success 16.7%
  llm: 12 runs, success 83.3%
```

### clusters

```
Cluster 0
  count: 15
  success: 86.7%
  prototype: llm > tool > llm > judge

Cluster 1
  count: 10
  success: 20.0%
  prototype: llm > tool > tool > tool > error

Cluster 2
  count: 8
  success: 37.5%
  prototype: llm > compaction > llm > judge
```

### diff

```
Cohort A: harness=baseline_v1 (21 runs)
Cohort B: harness=router_v2 (21 runs)

Success rate: 47.6% -> 71.4% (+23.8)
Avg steps: 8.9 -> 6.1 (-2.8)
Avg tokens out: 891 -> 624 (-267)

Cluster shifts:
  direct_solve: +9 runs
  retrieval_loop: -6 runs
  compaction_drift: -2 runs

Largest divergence change:
  col=2 tool branch frequency dropped from 61.9% to 23.8%
```

---

## File layout

```
moirai/
  README.md
  pyproject.toml
  .gitignore

  moirai/
    __init__.py
    cli.py
    schema.py
    normalize.py
    load.py
    filters.py

    analyze/
      __init__.py
      summary.py
      align.py
      divergence.py
      distance.py
      cluster.py
      compare.py

    viz/
      __init__.py
      trace_table.py
      branch_graph.py
      cluster_plot.py
      diff_plot.py

  scripts/
    generate_synthetic.py

  examples/
    synthetic_helpdesk/
    synthetic_codefix/

  tests/
    test_normalize.py
    test_distance.py
    test_align.py
    test_cluster.py
    test_compare.py
```

---

## Synthetic data

Two datasets: `synthetic_helpdesk` and `synthetic_codefix`, 20-50 runs each.

Behavior modes to cover:
- Successful direct solve
- Retrieval loop (tool → tool → tool)
- Premature judging
- Tool failure and retry recovery
- Compaction-induced drift
- Over-exploration before solve

Vary across: model, harness, outcome. Two harness variants per dataset for diff testing.

---

## Implementation order

### Milestone 1: schema + loader + validator + summary
Load run files, normalize, validate, print summary.

### Milestone 2: single-run trace inspection
`moirai trace` with clean terminal output.

### Milestone 3: distance + clustering
Pairwise Levenshtein distance, agglomerative clustering, cluster output.

### Milestone 4: alignment + divergence
Needleman-Wunsch alignment, divergence point detection with entropy and success correlation.

### Milestone 5: cohort diff
Filter, compare, produce change summaries.

### Milestone 6: HTML export
Branch graph (Sankey), cluster charts, diff plots via Plotly.

---

## Dependencies

- `typer` — CLI
- `rich` — terminal output
- `plotly` — HTML visualizations
- `scipy` — agglomerative clustering
- `numpy` — distance matrices

No other dependencies for v0.

---

## Non-goals

- OpenTelemetry ingestion
- Live streaming traces
- Semantic clustering via embeddings
- Prompt token heatmaps
- LLM-generated run summaries
- Browser backend / auth
- Plugin SDK
- Distributed storage
- Replay engine
- Per-token attention visualization

---

## What makes v0 compelling

Not logging traces (everyone does that). Not visualizing one run (everyone can render a tree).

v0 proves:
1. Repeated runs cluster into distinct behavior modes
2. Divergence points correlate with outcome
3. Harness changes alter trajectory distribution in measurable ways
4. Model upgrades don't always matter as much as harness changes
