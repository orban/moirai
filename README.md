# moirai

Trajectory-level debugging for stochastic agent systems.

## The problem

Agent evaluations report final metrics — pass rate, score, cost. But when 30% of runs fail, those numbers don't tell you whether they all fail the same way or five different ways. When a harness change improves pass rate, you can't tell if it fixed the failure mode or just shifted it. Final metrics are lossy summaries of rich behavioral data.

moirai looks at the *trajectories* — the sequence of steps each run took — and shows you where behavior diverges and how divergence correlates with outcome.

## Example: cohort diff

```
$ moirai diff runs/ --a harness=baseline_v1 --b harness=router_v2

Cohort A: harness=baseline_v1 (6 runs)
Cohort B: harness=router_v2 (6 runs)

Success rate: 33.3% -> 66.7% (+33.3)
Avg steps: 4.7 -> 5.3 (+0.7)
Avg tokens in: 765 -> 869 (+104)
Avg tokens out: 270 -> 258 (-12)

Cluster shifts:
  llm > tool > llm > tool > llm > judge: +4 runs
  llm > tool > tool > tool > error: -3 runs
  llm > tool > compaction > llm > error: +1 runs
```

router_v2 doubled the success rate by eliminating retrieval loops, but introduced a compaction failure mode. That's a real story, not a number.

## Install

```bash
pip install -e .
```

Requires Python 3.11+.

## Quickstart

```bash
# validate your trace files
moirai validate path/to/runs/

# see aggregate stats
moirai summary path/to/runs/

# inspect a single run
moirai trace path/to/run.json

# cluster runs by trajectory structure
moirai clusters path/to/runs/

# find where trajectories diverge
moirai branch path/to/runs/

# compare two cohorts
moirai diff path/to/runs/ --a harness=baseline --b harness=router
```

## Commands

| Command | Description |
|---------|-------------|
| `validate` | Check JSON validity and schema compliance |
| `summary` | Aggregate stats across runs |
| `trace` | Inspect a single run |
| `clusters` | Cluster runs by trajectory structure |
| `branch` | Multi-run divergence analysis |
| `diff` | Compare two cohorts |

### Common flags

- `--level type|name` — sequence level for analysis (default: type)
- `--strict` — treat warnings as errors
- `--html <path>` — write HTML visualization (clusters, branch, diff)
- `--model`, `--harness`, `--task-family` — filter runs

### Diff flags

`--a` and `--b` take `K=V` filters. Repeatable for multiple conditions (AND'd):

```bash
moirai diff runs/ --a harness=baseline --a model=claude-3.7 --b harness=router
```

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
      "idx": 0,
      "type": "llm",
      "name": "plan",
      "status": "ok",
      "input": {"summary": "User asks about failing test"},
      "output": {"summary": "Agent decides to inspect files"},
      "metrics": {"tokens_in": 412, "tokens_out": 137, "latency_ms": 1840}
    }
  ],
  "result": {
    "success": true,
    "score": 0.91,
    "label": "pass",
    "error_type": null,
    "summary": "Patched bug and tests passed"
  }
}
```

**Required fields:** `run_id`, `task_id`, `steps`, `result` (with `success`).

**Step types:** `llm`, `tool`, `system`, `memory`, `compaction`, `judge`, `error`, `handoff`. Unknown types normalize to `other`.

**Recognized metrics:** `tokens_in`, `tokens_out`, `latency_ms`.

## How it works

1. **Load and normalize** — reads JSON files, maps type aliases, assigns missing indices
2. **Cluster** — pairwise Needleman-Wunsch distance over step-type sequences, agglomerative clustering
3. **Align** — progressive multi-run alignment to find where trajectories diverge
4. **Diverge** — entropy at each aligned position, correlated with downstream success rates
5. **Diff** — cluster each cohort independently, match by prototype, compute shifts

## Roadmap

v0 is done. v1 directions:

- Semantic step grouping
- Content-aware diffs
- Time-series drift detection over nightly runs
- OpenTelemetry adapters
- Compaction event analysis
- Judge disagreement overlays
