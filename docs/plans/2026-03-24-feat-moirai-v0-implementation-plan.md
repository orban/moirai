---
title: moirai v0 implementation
type: feat
date: 2026-03-24
---

# moirai v0 implementation plan

Spec: `docs/superpowers/specs/2026-03-24-moirai-v0-design.md`

## Implementation decisions

These clarifications fill gaps from spec review, flow analysis, and plan review. Binding for implementation.

1. **One run per JSON file.** A file containing a JSON array is rejected with a clear error.
2. **Empty `steps: []` is valid.** Produces empty signature `""`, distance 1.0 from any non-empty run, gets its own cluster. Warn during validation.
3. **Duplicate `run_id` across files:** warn and skip the duplicate (keep first loaded). Add to validate checks.
4. **Duplicate `idx` within a run:** warn during normalization, resolve by original array order (stable sort).
5. **Exit codes:** 0 success, 1 error (validation failure, runtime error), 2 invalid CLI usage.
6. **Clustering:** average linkage, distance criterion. Threshold 0.3 means clusters merge until inter-cluster distance exceeds 0.3.
7. **Filter flags on all directory commands:** `summary`, `clusters`, `branch`, and `diff` all support `--model`, `--harness`, `--task-family`.
8. **`--expand` is a real flag on `trace`.**
9. **Metrics:** only exact keys `tokens_in`, `tokens_out`, `latency_ms` recognized. No aliases in v0. Other keys preserved but not used in summaries.
10. **Zero matching runs:** all commands print "No runs matched the given filters" and exit 1.
11. **`--html` overwrites existing files silently.**
12. **Tag filter matching:** values compared as strings (coerce tag value to string).
13. **`--verbose` / `--quiet`:** not in v0. Warnings inline.
14. **`find_divergence_points` takes `(alignment, runs)`.** Deviates from spec which only takes `(alignment)`. The function needs runs for per-partition success rates. Spec deviation is intentional.
15. **`load_runs` returns `tuple[list[Run], list[str]]`.** Runs plus warning messages. Caller decides how to display warnings. Makes the function testable without capturing stdout.
16. **`Alignment` uses matrix representation.** `matrix: list[list[str]]` indexed by `[run_index][column_index]`, not dict-per-column.
17. **`RunSummary` metric fields are `float | None`.** None when no runs have metrics. 0.0 is misleading.

---

## Phase 1a: skeleton + schema + loader + validate

**Goal:** Installable package, data pipeline works, `moirai validate` command.

### Setup (folded from old phase 0)

- `pyproject.toml` — package metadata, `[project.scripts] moirai = "moirai.cli:app"`, Python >=3.11
- `.gitignore` — `__pycache__`, `.venv`, `*.egg-info`, `dist/`, `.pytest_cache/`
- `moirai/__init__.py` — version string only
- `moirai/cli.py` — Typer app
- `moirai/analyze/__init__.py`, `moirai/viz/__init__.py` — empty

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

**Also: prototype Typer `--a`/`--b` repeated flags.** 15-minute spike. Confirm `a: list[str] = typer.Option([], "--a")` works with `--a harness=X --a model=Y`. If not, design alternative now (e.g., `--filter-a` with comma separation). This de-risks phase 4.

### Task 1a.1: schema.py — phase 1 types only

Dataclasses for this phase: `Step`, `Result`, `Run`, `RunSummary`, `ValidationResult`.

Constants:
- `KNOWN_STEP_TYPES = {"llm", "tool", "system", "memory", "compaction", "judge", "error", "handoff"}`
- `STEP_TYPE_ALIASES = {"assistant": "llm", "completion": "llm", "function_call": "tool", "retrieval": "tool", "compress": "compaction", "critic": "judge"}`

`Alignment`, `DivergencePoint`, `ClusterInfo`, `ClusterResult`, `CohortDiff` added in their respective phases.

Sequence extraction functions live here (analysis primitives, not normalization):
```python
def step_type_sequence(run: Run) -> list[str]
def step_name_sequence(run: Run) -> list[str]
def signature(run: Run) -> str  # "llm:plan > tool:retrieve > llm:reason"
```

```python
@dataclass
class ValidationResult:
    file_path: str
    passed: bool
    warnings: list[str]
    errors: list[str]
```

`RunSummary` metric fields are `float | None`.

**File:** `moirai/schema.py`

### Task 1a.2: normalize.py — normalization only

```python
def normalize_run(raw: dict, strict: bool = False) -> Run
```

- Step ordering (sort by idx, assign if missing)
- Step type alias mapping, unknown → `"other"`
- Default status to `"ok"`
- Metric key recognition (exact match only, coerce to float)
- Result inference in non-strict mode
- Warn on: missing idx, unknown step types, empty steps, duplicate idx

No sequence extraction here — that's in `schema.py`.

**File:** `moirai/normalize.py`
**Test:** `tests/test_normalize.py` — alias mapping, idx assignment, empty steps, strict mode rejection, duplicate idx, metric coercion. Also covers schema dataclass construction and sequence extraction (no separate test_schema.py).

### Task 1a.3: load.py — file loading

```python
def load_runs(path: str | Path, strict: bool = False) -> tuple[list[Run], list[str]]
```

Returns `(runs, warnings)`. Caller prints warnings.

- Single `.json` file: parse, normalize, return `[run]`
- Directory: recursively glob `**/*.json`, skip symlinks
- Reject JSON arrays ("Expected a single Run object, got array")
- Detect and warn on duplicate `run_id`, keep first
- Skip invalid files with warning (unless strict)
- Path not found → raise `FileNotFoundError`
- Directory with no .json files → return `([], ["No JSON files found in ..."])`

**File:** `moirai/load.py`
**Test:** `tests/test_load.py` — single file, directory, nested, invalid JSON, JSON array rejection, duplicate run_id, empty directory. Uses `tmp_path` fixture.

### Task 1a.4: filters.py — cohort filtering

```python
def filter_runs(
    runs: list[Run],
    model: str | None = None,
    harness: str | None = None,
    task_family: str | None = None,
    tags: dict[str, str] | None = None,
) -> list[Run]
```

`parse_kv_filter` deferred to phase 4 when `diff` needs it.

**File:** `moirai/filters.py`
**Test:** `tests/test_filters.py` — exact match, tag match, AND logic, no matches, numeric tag coercion

### Task 1a.5: cli.py — validate command + hand-crafted test data

Wire up `validate`:
```python
@app.command()
def validate(path: Path, strict: bool = False) -> None
```

CLI params use `Path`, not `str`.

Write 10-15 hand-crafted JSON files in `examples/synthetic_helpdesk/`:
- 3-4 behavior modes (direct_solve, retrieval_loop, tool_failure_retry, compaction_drift)
- 2 harness variants (baseline_v1, router_v2)
- 2 models
- Mix of success/failure

These are simple, readable, manually verifiable. No generator script.

### Phase 1a done when

- `pip install -e .` succeeds
- `moirai validate examples/synthetic_helpdesk/` shows pass for all files
- All tests pass

---

## Phase 1b: summary + trace

**Goal:** `moirai summary` and `moirai trace` work.

### Task 1b.1: analyze/summary.py

```python
def summarize_runs(runs: list[Run]) -> RunSummary
```

- Count, success rate (excluding null; None if all null)
- Avg/median steps
- Avg tokens_in/out, latency (None if no metrics)
- Top signatures (Counter, descending)
- Error type counts (from result.error_type)

**File:** `moirai/analyze/summary.py`
**Test:** `tests/test_summary.py` — basic, all-null success, no metrics → None fields

### Task 1b.2: viz/terminal.py — all terminal output

Single file for all Rich console print functions:

```python
def print_summary(summary: RunSummary) -> None
def print_trace(run: Run, expand: bool = False) -> None
def print_validation(results: list[ValidationResult]) -> None
```

Match output format from spec section 12. Add `print_clusters`, `print_divergence`, `print_diff` in later phases.

**File:** `moirai/viz/terminal.py`

### Task 1b.3: cli.py — summary and trace commands

```python
@app.command()
def summary(
    path: Path,
    strict: bool = False,
    model: str | None = None,
    harness: str | None = None,
    task_family: str | None = None,
) -> None

@app.command()
def trace(path: Path, expand: bool = False) -> None
```

`trace` errors clearly if given a directory: "trace requires a single run file, not a directory."

### Phase 1b done when

- `moirai summary examples/synthetic_helpdesk/` matches expected format
- `moirai summary examples/synthetic_helpdesk/ --harness baseline_v1` filters correctly
- `moirai trace examples/synthetic_helpdesk/run_001.json` shows step table
- All tests pass

---

## Phase 2: distance + clustering

**Goal:** Group runs into behavior modes by structural similarity.

### Task 2.1: analyze/align.py — NW core + distance

Pairwise Needleman-Wunsch alignment and distance in one module. `_nw_align` is private (prefixed).

```python
def _nw_align(seq_a: list[str], seq_b: list[str]) -> tuple[list[str], list[str]]
    # match=+1, mismatch=-1, gap=-1. Returns aligned sequences with GAP.

def trajectory_distance(run_a: Run, run_b: Run, level: str = "type") -> float
    # NW align, count non-matches, divide by alignment length. [0.0, 1.0].

def distance_matrix(runs: list[Run], level: str = "type") -> np.ndarray
    # Condensed distance matrix for scipy.
```

Progressive multi-run alignment (`align_runs`) added in phase 3.

**File:** `moirai/analyze/align.py`
**Test:** `tests/test_align.py` — NW: identical, one empty, different lengths, gap insertion. Distance: 0.0 for identical, 1.0 for disjoint, empty vs non-empty, symmetry, empty vs empty = 0.0.

### Task 2.2: schema.py — add ClusterInfo, ClusterResult

```python
@dataclass
class ClusterInfo:
    cluster_id: int
    count: int
    success_rate: float | None
    prototype: str       # most common signature; also serves as label
    avg_length: float
    error_types: dict[str, int]

@dataclass
class ClusterResult:
    clusters: list[ClusterInfo]
    labels: dict[str, int]  # run_id -> cluster_id
```

No separate `label` field. Display functions derive a short label from `prototype` if needed.

### Task 2.3: analyze/cluster.py — agglomerative clustering

```python
def cluster_runs(
    runs: list[Run],
    level: str = "type",
    threshold: float = 0.3,
) -> ClusterResult
```

- `scipy.cluster.hierarchy.linkage(condensed, method="average")`
- `scipy.cluster.hierarchy.fcluster(Z, t=threshold, criterion="distance")`
- 0 runs → empty ClusterResult. 1 run → single cluster.

**File:** `moirai/analyze/cluster.py`
**Test:** `tests/test_cluster.py` — known clusters separate, single run, two identical runs, threshold sensitivity

### Task 2.4: viz/terminal.py — add print_clusters

```python
def print_clusters(result: ClusterResult) -> None
```

Match spec section 12.3.

### Task 2.5: cli.py — clusters command

```python
@app.command()
def clusters(
    path: Path,
    level: str = "type",
    threshold: float = 0.3,
    strict: bool = False,
    model: str | None = None,
    harness: str | None = None,
    task_family: str | None = None,
    html: Path | None = None,
) -> None
```

HTML deferred to phase 5.

### Phase 2 done when

- `moirai clusters examples/synthetic_helpdesk/` produces believable clusters
- Behavior modes land in separate clusters
- All tests pass

---

## Phase 3: alignment + divergence

**Goal:** Identify where trajectories split and how splits correlate with outcome.

### Task 3.1: schema.py — add Alignment, DivergencePoint

```python
GAP = "-"

@dataclass
class Alignment:
    run_ids: list[str]
    matrix: list[list[str]]  # [run_index][column_index] = value or GAP
    level: str

@dataclass
class DivergencePoint:
    column: int
    value_counts: dict[str, int]
    entropy: float
    success_by_value: dict[str, float | None]
```

### Task 3.2: analyze/align.py — progressive multi-run alignment

```python
def align_runs(runs: list[Run], level: str = "type") -> Alignment
```

Algorithm:
1. Compute pairwise distances (reuse `distance_matrix`)
2. Closest pair → align with `_nw_align`
3. Consensus: majority non-gap, first-occurrence tie-break
4. Align next closest against consensus, update
5. Repeat until all runs incorporated
6. Build `Alignment` with matrix representation

**Test:** `tests/test_align.py` (extend) — 3 similar runs align, gap insertion, all-identical runs → no gaps, determinism under shuffled input order

### Task 3.3: analyze/divergence.py

```python
def find_divergence_points(alignment: Alignment, runs: list[Run]) -> list[DivergencePoint]
```

Spec deviation: takes `runs` for per-partition success rates (spec says `alignment` only).

- Column has >1 distinct non-gap value → divergence point
- Entropy, value counts, success rates per partition
- Sorted by entropy descending
- All-null success in partition → `None`

**File:** `moirai/analyze/divergence.py`
**Test:** `tests/test_divergence.py` — uniform column, mixed column, entropy math, success correlation, all-null partition

### Task 3.4: viz/terminal.py — add print_divergence

```python
def print_divergence(points: list[DivergencePoint]) -> None
```

Match spec section 12.2.

### Task 3.5: cli.py — branch command

```python
@app.command()
def branch(
    path: Path,
    level: str = "type",
    strict: bool = False,
    model: str | None = None,
    harness: str | None = None,
    task_family: str | None = None,
    html: Path | None = None,
) -> None
```

### Phase 3 done when

- `moirai branch examples/synthetic_helpdesk/` shows divergence points
- Points correlate with success/failure
- All tests pass

---

## Phase 4: cohort diff

**Goal:** Compare two cohorts and produce a change summary.

### Task 4.1: schema.py — add CohortDiff

```python
@dataclass
class CohortDiff:
    a_summary: RunSummary
    b_summary: RunSummary
    a_only_signatures: list[tuple[str, int]]  # pre-clustering
    b_only_signatures: list[tuple[str, int]]  # pre-clustering
    cluster_shifts: list[tuple[str, int]]     # post-clustering: (prototype, count_delta)
```

No stored delta fields. Display functions compute deltas from the two summaries.

### Task 4.2: filters.py — add parse_kv_filter

```python
def parse_kv_filter(kv: str) -> tuple[str, str]
    # "harness=baseline" -> ("harness", "baseline")
    # Simple split on "=". Caller dispatches to field vs tag.
```

**Test:** `tests/test_filters.py` (extend) — valid K=V, missing =, empty value

### Task 4.3: analyze/compare.py

```python
def compare_cohorts(
    a_runs: list[Run],
    b_runs: list[Run],
    level: str = "type",
    threshold: float = 0.3,
) -> CohortDiff
```

- Summarize each cohort
- Cluster each independently
- Match clusters by prototype
- Compute shifts (positive = more in B)
- Find unique signatures per cohort (pre-clustering)

**File:** `moirai/analyze/compare.py`
**Test:** `tests/test_compare.py` — identical cohorts, different cohorts, empty cohort raises ValueError, cluster matching by prototype

### Task 4.4: viz/terminal.py — add print_diff

```python
def print_diff(diff: CohortDiff, a_label: str, b_label: str) -> None
```

Computes deltas inline from `a_summary` and `b_summary`. Match spec section 12.4.

### Task 4.5: cli.py — diff command

```python
@app.command()
def diff(
    path: Path,
    a: list[str] = typer.Option([], "--a"),
    b: list[str] = typer.Option([], "--b"),
    level: str = "type",
    threshold: float = 0.3,
    strict: bool = False,
    html: Path | None = None,
) -> None
```

- Error if `--a` or `--b` not provided (exit 2)
- Error if either cohort empty (show available values, exit 1)

### Phase 4 done when

- `moirai diff examples/synthetic_helpdesk/ --a harness=baseline_v1 --b harness=router_v2` works
- Empty cohort shows available values
- All tests pass

---

## Phase 5: HTML export + polish

**Goal:** Plotly visualizations, README, ship.

### Task 5.1: viz/html.py — all HTML output

Single file for all Plotly HTML functions:

```python
def write_branch_html(alignment: Alignment, points: list[DivergencePoint], runs: list[Run], path: Path) -> Path
def write_clusters_html(result: ClusterResult, path: Path) -> Path
def write_diff_html(diff: CohortDiff, a_label: str, b_label: str, path: Path) -> Path
```

All return the path written. All produce self-contained HTML files.

**branch:** Start with a transition-frequency heatmap (colored alignment matrix). Upgrade to Sankey only if straightforward. Don't block the phase on Sankey complexity.

**clusters:** Grouped bar chart (cluster x-axis, bars for count and success rate).

**diff:** Single panel: success rate comparison bar chart. More panels in v0.2 if needed.

### Task 5.2: wire `--html` into CLI commands

Add conditional HTML writing to `branch`, `clusters`, `diff` commands.

### Task 5.3: README.md

Sections: what this is, why final metrics are insufficient, example output (paste real diff/cluster output), install, quickstart, commands, data schema, roadmap.

Under 300 lines.

### Task 5.4: test_cli.py — CLI integration tests

```python
# tests/test_cli.py
# Uses typer.testing.CliRunner
```

- validate: valid file → exit 0, invalid → exit 1
- summary: filtered output, zero matches → exit 1
- trace: directory input → clear error
- diff: missing --a/--b → exit 2, empty cohort → exit 1 with available values

### Phase 5 done when

- All HTML commands produce viewable files
- README exists
- CLI integration tests pass
- Tag as `v0.1.0`

---

## Build sequence

| Phase | Produces | Key files |
|-------|----------|-----------|
| 1a | validate command + test data | pyproject.toml, schema.py, normalize.py, load.py, filters.py, cli.py |
| 1b | summary + trace commands | analyze/summary.py, viz/terminal.py |
| 2 | clusters command | analyze/align.py, analyze/cluster.py |
| 3 | branch command | analyze/align.py (extend), analyze/divergence.py |
| 4 | diff command | analyze/compare.py, filters.py (extend) |
| 5 | HTML export + README + ship | viz/html.py, README.md, test_cli.py |

## File layout (revised)

```
moirai/
  pyproject.toml
  .gitignore
  README.md

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
      align.py          # NW core + distance + progressive alignment
      divergence.py
      cluster.py
      compare.py
    viz/
      __init__.py
      terminal.py       # all Rich print_* functions
      html.py           # all Plotly write_*_html functions

  examples/
    synthetic_helpdesk/   # 10-15 hand-crafted JSON files

  tests/
    test_normalize.py     # also covers schema types + sequence extraction
    test_load.py
    test_filters.py
    test_summary.py
    test_align.py         # NW, distance, progressive alignment
    test_divergence.py
    test_cluster.py
    test_compare.py
    test_cli.py           # Typer CliRunner integration tests
```

## Test strategy

- Hand-crafted fixtures (3-5 runs with known properties), not synthetic datasets
- No mocking — tests exercise real code paths
- `tmp_path` for file I/O tests
- No viz output tests beyond CLI integration tests (low value for terminal formatting)
- pytest runner

## Risk areas

1. **Typer `--a`/`--b` repeated flags** — prototyped in phase 1a setup. If broken, redesign before phase 4.
2. **Progressive alignment quality** — if noisy at 50 runs, fall back to aligning only the top-k most common sequences as cluster representatives.
3. **Branch HTML** — start with heatmap, not Sankey. Upgrade if time allows.
