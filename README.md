# moirai

Trajectory-level debugging for stochastic agent systems.

## The problem

Agent evaluations report final metrics — pass rate, score, cost. But when 30% of runs fail, those numbers don't tell you whether they all fail the same way or five different ways. When a harness change improves pass rate, you can't tell if it fixed the failure mode or just shifted it. Final metrics are lossy summaries of rich behavioral data.

moirai looks at the *trajectories* — the sequence of steps each run took — and shows you where behavior diverges and how divergence correlates with outcome.

## Example: what patterns predict failure?

```
$ moirai patterns runs/

Discriminative patterns (1000 runs, 75% baseline success)

Patterns correlated with success:
  edit → test_result → write → test_result → reason  100% (12 runs) vs 75% baseline  p=0.044
  write → edit → reason                              100% (13 runs) vs 75% baseline  p=0.047

Patterns correlated with failure:
  write → test_result → read → read → test_result    0% (4 runs) vs 75% baseline    p=0.004
  read → read → read → test_result                   17% (6 runs) vs 75% baseline   p=0.005
```

Thinking after testing works. Reading without modifying doesn't.

## Example: where do trajectories split?

```
$ moirai branch runs/

Cluster 13 (109 runs, 74% success)
  Aligned to 55 columns, 1 divergence points

  Position 35 (p=0.075)
  verify→
    test_result: 77 runs, 74% success
    search: 2 runs, 0% success
```

At position 35, runs that keep testing succeed 74% of the time. Runs that go back to searching at that point always fail.

## Example: explain a failing run

```
$ moirai explain runs/ --run django-money__835c

Run: django-money__835c... FAIL
Cluster: 8 (11 runs, 91% success)

Trajectory: read → test(fail)×2 → search → test(fail)×3 → write → test(fail) → read×2 → test(fail)
Mix: 58% verify, 33% explore, 8% modify

Key divergence (position 38):
  write: 1 runs, 0% success  <-- this run
  read: 1 runs, 100% success
```

This run is in a cluster where 91% succeed. It failed because it tried to write a fix at position 38 instead of reading more — 0% vs 100% success for that choice.

## Install

```bash
pip install -e .
```

Requires Python 3.11+.

## Quickstart

```bash
# validate trace files
moirai validate path/to/runs/

# aggregate stats
moirai summary path/to/runs/

# inspect a single run
moirai trace path/to/run.json

# cluster by trajectory structure
moirai clusters path/to/runs/

# find where trajectories diverge
moirai branch path/to/runs/

# find patterns that predict success/failure
moirai patterns path/to/runs/

# explain why a specific run failed
moirai explain path/to/runs/ --run <run_id>

# compare two cohorts
moirai diff path/to/runs/ --a harness=baseline --b harness=router
```

## Commands

| Command | Description |
|---------|-------------|
| `validate` | Check JSON validity and schema compliance |
| `summary` | Aggregate stats across runs |
| `trace` | Inspect a single run with compressed trajectory |
| `clusters` | Cluster runs by trajectory structure, show sub-patterns |
| `branch` | Per-cluster divergence analysis with significance testing |
| `patterns` | Find step patterns that predict success or failure |
| `explain` | Explain why a specific run succeeded or failed |
| `diff` | Compare two cohorts |

### Common flags

- `--strict` — treat warnings as errors
- `--html <path>` — write HTML visualization (clusters, branch, diff)
- `--model`, `--harness`, `--task-family` — filter runs

### Diff flags

`--a` and `--b` take `K=V` filters. Repeatable for multiple conditions (AND'd):

```bash
moirai diff runs/ --a harness=baseline --a model=claude-3.7 --b harness=router
```

## Output format

moirai compresses raw step sequences into readable summaries:

**Raw** (28 steps):
```
error_observation → read → error_observation → action → error_observation → action → ...
```

**Compressed** (8 meaningful tokens):
```
read×2 → edit → write → test(fail) → search → test(fail) → read
```

**Phase level**:
```
explore → modify → verify(fail) → explore → verify(fail) → explore
```

**Phase mix**: `50% verify, 25% modify, 25% explore`

## Data schema

Each JSON file is a single run object:

```json
{
  "run_id": "run_001",
  "task_id": "task_bugfix_123",
  "task_family": "bugfix",
  "agent": "swe_agent",
  "model": "claude-3.7-sonnet",
  "harness": "baseline_v1",
  "timestamp": "2026-03-24T12:00:00Z",
  "tags": {"branch": "main", "cohort": "baseline"},
  "steps": [
    {
      "idx": 0, "type": "llm", "name": "plan", "status": "ok",
      "input": {"summary": "User asks about failing test"},
      "output": {"summary": "Agent decides to inspect files"},
      "metrics": {"tokens_in": 412, "tokens_out": 137, "latency_ms": 1840}
    }
  ],
  "result": {"success": true, "score": 0.91, "label": "pass"}
}
```

**Required fields:** `run_id`, `task_id`, `steps`, `result` (with `success`).

**Step types:** `llm`, `tool`, `system`, `memory`, `compaction`, `judge`, `error`, `handoff`. Unknown types normalize to `other`.

## How it works

1. **Load and normalize** — reads JSON files, maps type aliases, filters noise steps
2. **Cluster** — pairwise Needleman-Wunsch distance over step-type sequences, agglomerative clustering (average linkage)
3. **Align** — clusters first at type level, then progressive NW alignment within each cluster at name level for fine-grained divergence
4. **Diverge** — Fisher's exact test at each aligned position, filters by statistical significance (p<0.2) and minimum branch size (≥2)
5. **Patterns** — extracts 3-5 step n-grams, tests which patterns discriminate between success and failure
6. **Diff** — cluster each cohort independently, match by prototype, compute shifts

## Converters

moirai includes converters for existing trace formats:

- `scripts/convert_eval_harness.py` — intent-layer eval-harness trial results + logs
- `scripts/convert_swe_smith.py` — SWE-bench/SWE-smith trajectories from HuggingFace

## Roadmap

- Motif-aware clustering (cluster by patterns, not just type sequences)
- Content-aware diffs (what the agent was looking at, not just what tool it used)
- Time-series drift detection over nightly runs
- OpenTelemetry adapters
- DTW alignment for better retry-loop handling
