---
title: moirai v0 implementation
type: feat
date: 2026-03-24
---

# moirai v0 implementation plan

Spec: `docs/superpowers/specs/2026-03-24-moirai-v0-design.md`

## Implementation decisions (from spec review and flow analysis)

These clarifications fill gaps identified during spec review and are binding for implementation:

1. **One run per JSON file.** A file containing a JSON array is rejected with a clear error.
2. **Empty `steps: []` is valid.** Produces empty signature `""`, distance 1.0 from any non-empty run, gets its own cluster. Warn during validation.
3. **Duplicate `run_id` across files:** warn and skip the duplicate (keep first loaded). Add to validate checks.
4. **Duplicate `idx` within a run:** warn during normalization, resolve by original array order (stable sort).
5. **Exit codes:** 0 success, 1 error (validation failure, runtime error), 2 invalid CLI usage.
6. **Clustering:** average linkage, distance criterion. Threshold 0.3 means clusters merge until inter-cluster distance exceeds 0.3.
7. **Filter flags on all directory commands:** `summary`, `clusters`, `branch`, and `diff` all support `--model`, `--harness`, `--task-family`.
8. **`--expand` is a real flag on `trace`.** Added to CLI spec.
9. **Metrics:** only exact keys `tokens_in`, `tokens_out`, `latency_ms` are recognized. No aliases in v0. Other metric keys are preserved but not used in summaries.
10. **Zero matching runs:** all commands print "No runs matched the given filters" and exit 1.
11. **`--html` overwrites existing files silently.** Standard CLI behavior.
12. **Tag filter matching:** values compared as strings. `--a retries=3` matches `tags.retries == "3"` or `tags.retries == 3` (string coercion on the tag value).
13. **`--verbose` / `--quiet`:** not in v0. Default output shows warnings inline.

---

## Phase 0: project skeleton

**Goal:** Installable package with `moirai` CLI entry point, no commands yet.

### Files to create

- `pyproject.toml` — package metadata, dependencies, `[project.scripts] moirai = "moirai.cli:app"`, Python >=3.11
- `.gitignore` — Python standard (`__pycache__`, `.venv`, `*.egg-info`, `dist/`, `.pytest_cache/`)
- `moirai/__init__.py` — version string only
- `moirai/cli.py` — Typer app with no commands, just the app object
- `moirai/analyze/__init__.py` — empty
- `moirai/viz/__init__.py` — empty

### Dependencies in pyproject.toml

```toml
[project]
dependencies = [
  "typer>=0.9",
  "rich>=13.0",
  "plotly>=5.0",
  "scipy>=1.11",
  "numpy>=1.24",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov"]
```

### Done when

- `pip install -e .` succeeds
- `moirai --help` prints usage
- `pytest` runs (0 tests, 0 failures)

---

## Phase 1: schema + loader + validator + summary

**Goal:** Load, validate, normalize, and summarize run files. First two CLI commands work.

### Task 1.1: schema.py — data model

All dataclasses from the spec: `Step`, `Result`, `Run`, `GAP`, `Alignment`, `DivergencePoint`, `RunSummary`, `ClusterInfo`, `ClusterResult`, `CohortDiff`.

Plus constants:
- `KNOWN_STEP_TYPES = {"llm", "tool", "system", "memory", "compaction", "judge", "error", "handoff"}`
- `STEP_TYPE_ALIASES = {"assistant": "llm", "completion": "llm", "function_call": "tool", "retrieval": "tool", "compress": "compaction", "critic": "judge"}`

**File:** `moirai/schema.py`
**Test:** `tests/test_schema.py` — dataclass construction, field defaults, GAP sentinel value

### Task 1.2: normalize.py — normalization

```python
def normalize_run(raw: dict, strict: bool = False) -> Run
```

Implements all normalization rules from spec section 5:
- Step ordering (sort by idx, assign if missing)
- Step type alias mapping
- Unknown type → `"other"`
- Default status to `"ok"`
- Metric key recognition (exact match only)
- Result inference in non-strict mode
- Warn on: missing idx, unknown step types, empty steps, duplicate idx

Also:
```python
def step_type_sequence(run: Run) -> list[str]
def step_name_sequence(run: Run) -> list[str]
def signature(run: Run) -> str  # "llm:plan > tool:retrieve > llm:reason"
```

**File:** `moirai/normalize.py`
**Test:** `tests/test_normalize.py` — alias mapping, idx assignment, empty steps, strict mode rejection, signature computation, duplicate idx handling

### Task 1.3: load.py — file loading

```python
def load_runs(path: str | Path, strict: bool = False) -> list[Run]
```

- Single `.json` file: parse, normalize, return `[run]`
- Directory: recursively glob `**/*.json`, skip non-JSON, skip symlinks
- Reject JSON arrays with clear error ("Expected a single Run object, got array")
- Detect and warn on duplicate `run_id`, keep first
- Skip invalid files with warning (unless strict)
- Return list of normalized Run objects

**File:** `moirai/load.py`
**Test:** `tests/test_load.py` — single file, directory, nested directory, invalid JSON, JSON array rejection, duplicate run_id, symlink skipping

### Task 1.4: filters.py — cohort filtering

```python
def filter_runs(
    runs: list[Run],
    model: str | None = None,
    harness: str | None = None,
    task_family: str | None = None,
    tags: dict[str, str] | None = None,
) -> list[Run]

def parse_kv_filter(kv: str) -> tuple[str, str, str]
    # Returns (field_or_tag, key, value)
    # "harness=baseline" -> ("field", "harness", "baseline")
    # "cohort=A" -> ("tag", "cohort", "A")
```

Tag matching: coerce tag value to string before comparison.

**File:** `moirai/filters.py`
**Test:** `tests/test_filters.py` — exact match, tag match, multiple filters AND'd, no matches returns empty, numeric tag coercion

### Task 1.5: analyze/summary.py — summary stats

```python
def summarize_runs(runs: list[Run]) -> RunSummary
```

- Count, success rate (excluding null), avg/median steps
- Avg tokens_in, tokens_out, latency_ms (from step metrics)
- Top signatures (Counter, sorted by count)
- Error type counts (from result.error_type)

**File:** `moirai/analyze/summary.py`
**Test:** `tests/test_summary.py` — basic summary, all-null success, mixed success/null, empty metrics

### Task 1.6: viz/trace_table.py — terminal output for summary and trace

Rich console output functions:

```python
def print_summary(summary: RunSummary) -> None
def print_trace(run: Run, expand: bool = False) -> None
def print_validation_result(results: list[ValidationResult]) -> None
```

Match the exact output format from spec section 12.

**File:** `moirai/viz/trace_table.py`

### Task 1.7: cli.py — validate, summary, trace commands

Wire up three commands:

```python
@app.command()
def validate(path: str, strict: bool = False) -> None

@app.command()
def summary(
    path: str,
    strict: bool = False,
    model: str | None = None,
    harness: str | None = None,
    task_family: str | None = None,
) -> None

@app.command()
def trace(path: str, expand: bool = False) -> None
```

- `validate`: load each file, report pass/fail/warnings, exit 1 if any failures
- `summary`: load, filter, summarize, print
- `trace`: load single file, print trace table

### Task 1.8: synthetic data generator (first batch)

`scripts/generate_synthetic.py` — generate 30 runs for `examples/synthetic_helpdesk/`:
- 6 behavior modes (direct_solve, retrieval_loop, premature_judge, tool_failure_retry, compaction_drift, over_exploration)
- 2 harness variants (baseline_v1, router_v2)
- 2 models
- Controlled success/failure rates per mode

This is needed before phase 1 is "done" — the commands need real data to verify against.

### Phase 1 done when

- `moirai validate examples/synthetic_helpdesk/` shows pass for all files
- `moirai summary examples/synthetic_helpdesk/` matches expected output format
- `moirai trace examples/synthetic_helpdesk/run_001.json` shows step table
- `moirai summary examples/synthetic_helpdesk/ --harness baseline_v1` filters correctly
- All tests pass

---

## Phase 2: distance + clustering

**Goal:** Group runs into behavior modes by structural similarity.

### Task 2.1: analyze/align.py — Needleman-Wunsch core

The alignment module is needed by both distance (phase 2) and divergence (phase 3). Build the core algorithm here, but the multi-run progressive alignment can wait until phase 3.

```python
def nw_align(seq_a: list[str], seq_b: list[str], match: int = 1, mismatch: int = -1, gap: int = -1) -> tuple[list[str], list[str]]
    # Returns two aligned sequences (with GAP markers)
```

~30 lines. Standard DP matrix + traceback.

**File:** `moirai/analyze/align.py`
**Test:** `tests/test_align.py` — identical sequences, one empty, different lengths, gap insertion

### Task 2.2: analyze/distance.py — trajectory distance

```python
def trajectory_distance(run_a: Run, run_b: Run, level: str = "type") -> float
```

- Extract sequences based on level
- Call `nw_align`
- Count mismatches + gaps in aligned output
- Divide by alignment length
- Range [0.0, 1.0]

Also:
```python
def distance_matrix(runs: list[Run], level: str = "type") -> np.ndarray
    # Returns condensed distance matrix for scipy
```

**File:** `moirai/analyze/distance.py`
**Test:** `tests/test_distance.py` — identical runs (0.0), completely different (1.0), empty vs non-empty (1.0), symmetry, triangle inequality spot check

### Task 2.3: analyze/cluster.py — agglomerative clustering

```python
def cluster_runs(
    runs: list[Run],
    level: str = "type",
    threshold: float = 0.3,
) -> ClusterResult
```

- Compute distance matrix
- `scipy.cluster.hierarchy.linkage(condensed, method="average")`
- `scipy.cluster.hierarchy.fcluster(Z, t=threshold, criterion="distance")`
- Build ClusterInfo for each cluster: id, prototype (most common signature), success rate, count, avg length, error types
- Handle edge cases: 0 runs (return empty), 1 run (single cluster)

**File:** `moirai/analyze/cluster.py`
**Test:** `tests/test_cluster.py` — known clusters separate correctly, single run, two identical runs (one cluster), threshold sensitivity

### Task 2.4: viz/cluster_plot.py — cluster terminal output

```python
def print_clusters(result: ClusterResult) -> None
```

Match the cluster output format from spec section 12.3.

**File:** `moirai/viz/cluster_plot.py`

### Task 2.5: cli.py — clusters command

```python
@app.command()
def clusters(
    path: str,
    level: str = "type",
    threshold: float = 0.3,
    strict: bool = False,
    model: str | None = None,
    harness: str | None = None,
    task_family: str | None = None,
    html: str | None = None,
) -> None
```

HTML output deferred to phase 5.

### Phase 2 done when

- `moirai clusters examples/synthetic_helpdesk/` produces believable clusters
- Known behavior modes (direct_solve, retrieval_loop, etc.) land in separate clusters
- Cluster prototype signatures match expected patterns
- All tests pass

---

## Phase 3: alignment + divergence

**Goal:** Identify where trajectories split and how splits correlate with outcome.

### Task 3.1: analyze/align.py — progressive multi-run alignment

Extend `align.py` with:

```python
def align_runs(runs: list[Run], level: str = "type") -> Alignment
```

Progressive alignment algorithm from spec:
1. Compute pairwise distances (reuse `distance_matrix`)
2. Find closest pair, align with NW
3. Build consensus (majority non-gap, first-occurrence tie-break)
4. Align next closest against consensus, update
5. Repeat until all incorporated
6. Map all runs back through consensus to produce Alignment

**File:** `moirai/analyze/align.py` (extend)
**Test:** `tests/test_align.py` (extend) — 3 similar runs align correctly, gap insertion in right position, all-identical runs produce no gaps

### Task 3.2: analyze/divergence.py — divergence detection

```python
def find_divergence_points(alignment: Alignment, runs: list[Run]) -> list[DivergencePoint]
```

- For each column: count distinct non-gap values
- If >1 distinct value: compute entropy, partition runs by value, compute per-partition success rate
- Sort by entropy descending
- Handle null success: exclude from rate, return None if all null in partition

**File:** `moirai/analyze/divergence.py`
**Test:** `tests/test_divergence.py` — uniform column (not a divergence), mixed column, entropy calculation, success correlation, all-null partition

### Task 3.3: viz/branch_graph.py — terminal divergence output

```python
def print_divergence(points: list[DivergencePoint]) -> None
```

Match the divergence output format from spec section 12.2.

**File:** `moirai/viz/branch_graph.py`

### Task 3.4: cli.py — branch command

```python
@app.command()
def branch(
    path: str,
    level: str = "type",
    strict: bool = False,
    model: str | None = None,
    harness: str | None = None,
    task_family: str | None = None,
    html: str | None = None,
) -> None
```

Terminal: print divergence points. HTML deferred to phase 5.

### Phase 3 done when

- `moirai branch examples/synthetic_helpdesk/` shows divergence points
- Divergence points correlate with success/failure in expected ways (e.g., runs that hit `judge` early succeed more often)
- All tests pass

---

## Phase 4: cohort diff

**Goal:** Compare two slices of runs and produce a change summary.

### Task 4.1: analyze/compare.py — cohort comparison

```python
def compare_cohorts(
    a_runs: list[Run],
    b_runs: list[Run],
    level: str = "type",
    threshold: float = 0.3,
) -> CohortDiff
```

- Summarize each cohort independently
- Compute deltas (success rate, steps, tokens, latency)
- Cluster each cohort independently
- Match clusters by prototype label
- Compute cluster shifts (positive = more in B)
- Find unique signatures per cohort (pre-clustering)

**File:** `moirai/analyze/compare.py`
**Test:** `tests/test_compare.py` — identical cohorts (zero delta), different cohorts, empty cohort handling, cluster matching by prototype

### Task 4.2: viz/diff_plot.py — terminal diff output

```python
def print_diff(diff: CohortDiff, a_label: str, b_label: str) -> None
```

Match the diff output format from spec section 12.4.

**File:** `moirai/viz/diff_plot.py`

### Task 4.3: cli.py — diff command

```python
@app.command()
def diff(
    path: str,
    a: list[str] = typer.Option([], "--a"),
    b: list[str] = typer.Option([], "--b"),
    level: str = "type",
    threshold: float = 0.3,
    strict: bool = False,
    html: str | None = None,
) -> None
```

- Parse `--a` and `--b` K=V pairs using `parse_kv_filter`
- Filter runs into two cohorts
- Error if either cohort is empty (show available values)
- Error if `--a` or `--b` not provided
- Compare and print

### Task 4.4: synthetic data — second dataset

Generate `examples/synthetic_codefix/` (30 runs, same behavior modes, different task family). Verify diff works across both datasets.

### Phase 4 done when

- `moirai diff examples/synthetic_helpdesk/ --a harness=baseline_v1 --b harness=router_v2` produces correct output
- Success rate deltas match manual calculation
- Cluster shifts are directionally correct
- Empty cohort produces helpful error with available values
- All tests pass

---

## Phase 5: HTML export

**Goal:** Plotly visualizations for branch, clusters, and diff.

### Task 5.1: viz/branch_graph.py — Sankey HTML

Extend with:
```python
def write_branch_html(alignment: Alignment, points: list[DivergencePoint], runs: list[Run], path: str) -> None
```

Plotly Sankey diagram:
- Nodes: aligned position + step type label
- Links: transitions between adjacent positions, weighted by run count
- Node color: downstream success rate (green to red gradient)
- Self-contained HTML file (Plotly include)

### Task 5.2: viz/cluster_plot.py — cluster bar chart HTML

Extend with:
```python
def write_clusters_html(result: ClusterResult, path: str) -> None
```

Grouped bar chart: cluster ID on x-axis, bars for count and success rate.

### Task 5.3: viz/diff_plot.py — diff comparison HTML

Extend with:
```python
def write_diff_html(diff: CohortDiff, a_label: str, b_label: str, path: str) -> None
```

Side-by-side panels:
- Success rate comparison (bar)
- Step count distribution (histogram)
- Cluster composition (stacked bar)

### Task 5.4: wire HTML into CLI commands

Add `--html` flag handling to `branch`, `clusters`, `diff` commands. When provided, write HTML and print path to console.

### Phase 5 done when

- `moirai branch examples/synthetic_helpdesk/ --html branch.html` produces a viewable Sankey
- `moirai clusters examples/synthetic_helpdesk/ --html clusters.html` produces bar chart
- `moirai diff examples/synthetic_helpdesk/ --a harness=baseline_v1 --b harness=router_v2 --html diff.html` produces comparison view
- All HTML files are self-contained (single file, no external dependencies)

---

## Phase 6: polish and ship

**Goal:** README, final synthetic data, end-to-end verification.

### Task 6.1: README.md

Sections: what this is, why final metrics are insufficient, example output (paste real diff output), install, quickstart, commands reference, data schema, roadmap (v1 directions from spec).

Short and ruthless. Under 300 lines.

### Task 6.2: end-to-end verification

Run every command against both synthetic datasets. Verify:
- All outputs match spec examples in format
- HTML files render correctly in browser
- `--strict` mode works
- Filter flags work on all directory commands
- Error messages are helpful for: empty directory, no matches, invalid JSON, missing required fields

### Task 6.3: final commit and tag

Tag as `v0.1.0`.

---

## Build sequence summary

| Phase | Depends on | Produces | Files |
|-------|-----------|----------|-------|
| 0 | nothing | installable skeleton | pyproject.toml, cli.py, __init__.py |
| 1 | phase 0 | validate, summary, trace commands + synthetic data | schema.py, normalize.py, load.py, filters.py, analyze/summary.py, viz/trace_table.py, cli.py, generate_synthetic.py |
| 2 | phase 1 | clusters command | analyze/align.py (core NW), analyze/distance.py, analyze/cluster.py, viz/cluster_plot.py |
| 3 | phase 2 | branch command | analyze/align.py (progressive), analyze/divergence.py, viz/branch_graph.py |
| 4 | phases 1-3 | diff command + second dataset | analyze/compare.py, viz/diff_plot.py |
| 5 | phases 2-4 | HTML export for all viz commands | viz/*.py (extend) |
| 6 | all above | shippable repo | README.md |

## Test strategy

- Each analysis module has its own test file
- Tests use small hand-crafted fixtures (3-5 runs with known properties), not the synthetic datasets
- Synthetic datasets are for manual verification and README examples
- No mocking of internal modules — tests exercise real code paths
- Test runner: pytest

## Risk areas

1. **Typer `--a`/`--b` repeated flag parsing** — prototype this early in phase 4. If Typer can't handle `--a harness=X --a model=Y`, switch to `--filter-a` with comma separation.
2. **Sankey diagram complexity** — the Plotly Sankey for the branch graph is the hardest visualization. If it takes too long, fall back to a simpler transition-frequency heatmap.
3. **Progressive alignment quality** — if the progressive alignment produces noisy results at 50 runs, consider using only the top-k most common sequences as cluster representatives for alignment.
