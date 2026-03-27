# moirai

**Align agent trajectories. Find where they diverge. See which branches predict failure.**

Most agent failures aren't random. Across 1000 SWE-bench runs, failures collapse into a small number of recurring behavioral branches — and those branches are identifiable from the trajectory structure alone, before you look at any code or logs.

```
$ moirai branch runs/

Cluster 13: 109 runs, 74% success
  Position 35 (p=0.075)
  verify→
    test_result: 77 runs, 74% success     ← agents that keep testing here pass
    search:       2 runs,  0% success     ← agents that go back to searching here always fail
```

moirai is a trajectory analysis tool for stochastic agent systems. Point it at a directory of agent run traces, and it tells you *where* runs diverge and *which* behavioral patterns predict success or failure — with statistical significance, not vibes.

## Why pass/fail isn't enough

You run 100 agents on the same task. 70 pass. 30 fail. You know the pass rate, but you don't know:

- Do all 30 failures fail the same way, or five different ways?
- Did the failures explore too much, or not enough?
- Is there a specific step in the trajectory where pass and fail runs split?
- Would a different harness change the distribution, or just move the failures around?

moirai answers these by treating runs as sequences, aligning them (Needleman-Wunsch, borrowed from bioinformatics), clustering by structural similarity, and testing each aligned position for outcome-predictive divergence using Fisher's exact test.

## Walkthrough: 1000 SWE-bench agent traces

### 1. Runs cluster into behavioral modes

The 1000 runs aren't 1000 unique trajectories. They're variations on ~10 behavioral patterns:

```
$ moirai clusters runs/

Cluster 18: 149 runs, 85% success
  explore → verify(fail) → modify → verify(fail)×2
  normal (143 runs, 85%): read → test(fail) → write → edit → test(fail) → write...

Cluster 25: 116 runs, 77% success
  explore → verify(fail)×2 → modify → verify(fail)×2 → think
  normal (116 runs, 77%): read → test(fail)×5 → write → edit → test(fail) → read...

Cluster 8:  83 runs, 49% success
  explore → verify(fail)×2 → explore → verify(fail)×4
  normal (83 runs, 49%): read → test(fail)×2 → search → test(fail)×3 → write...
```

Cluster 8 has a 49% success rate — almost a coin flip. The others range from 74% to 85%. Why?

### 2. Within each cluster, divergence points split outcomes

moirai aligns the runs within each cluster and finds positions where the agent's choice statistically predicts success or failure:

```
$ moirai branch runs/

Cluster 13: 109 runs, 74% success
  Aligned to 55 columns, 1 significant divergence point

  Position 35 (p=0.075)
  verify→
    test_result: 77 runs, 74% success
    context: ...test_result → write → [test_result] → test_result...
    search: 2 runs, 0% success
    context: ...test_result → [search] → search → test_result...
```

At position 35, 77 runs continue testing and succeed 74% of the time. 2 runs go back to searching and succeed 0%. The divergence is in a `verify→` phase — the agent has already been testing. Runs that abandon the current approach to search again at this point never recover.

### 3. Pattern mining finds recurring failure signatures

Beyond positional divergence, moirai extracts step n-grams that correlate with outcome:

```
$ moirai patterns runs/

Discriminative patterns (1000 runs, 75% baseline success)

Patterns that predict success:
  write → edit → reason                              100% success (13 runs)  p=0.047
  edit → test_result → write → test_result → reason  100% success (12 runs)  p=0.044

Patterns that predict failure:
  write → test_result → read → read → test_result      0% success (4 runs)  p=0.004
  read → read → read → test_result                    17% success (6 runs)  p=0.005
  reason → test_result → reason → test_result → reason 20% success (5 runs)  p=0.016
```

The success patterns have `reason` in them — the agent pauses to think between actions. The failure patterns are loops: `read → read → read` (exploring without acting) or `reason → test → reason → test → reason` (oscillating without converging).

### 4. Explain any specific run

```
$ moirai explain runs/ --run Project-MONAI

Run: Project-MONAI__MONAI... FAIL
Cluster: 23 (5 runs, 40% success)

Trajectory: read → search → test(fail)×2 → reason → test(fail) → write →
            reason → edit → test(fail)×2 → write → test(fail) → test(pass) → reason
Mix: 47% verify, 20% think, 20% modify, 13% explore

Compared to passing runs in this cluster:
  avg 30 steps (vs 29 in this run)
  e.g. read → search → test(fail) → write → test(fail) → reason → write → ...
```

## HTML dashboard

`moirai branch --html report.html` generates a dashboard with the analysis synthesized:

![moirai dashboard](docs/images/dashboard.png)

## Install

```bash
pip install -e .
```

Requires Python 3.11+.

## Commands

```bash
moirai validate path/to/runs/           # check trace files
moirai summary path/to/runs/            # aggregate stats
moirai trace path/to/run.json           # inspect one run
moirai clusters path/to/runs/           # find behavioral modes
moirai branch path/to/runs/             # find divergence points
moirai patterns path/to/runs/           # find success/failure patterns
moirai explain path/to/runs/ --run ID   # explain a specific run
moirai diff path/to/runs/ --a K=V --b K=V  # compare cohorts
```

All directory commands support `--model`, `--harness`, `--task-family` filters. `branch`, `clusters`, and `diff` support `--html`.

## How it works

1. **Normalize** — loads JSON trace files, maps step type aliases, filters noise
2. **Cluster** — pairwise Needleman-Wunsch edit distance over step sequences, agglomerative clustering
3. **Align** — progressive multi-sequence alignment within each cluster at step-name level
4. **Test** — Fisher's exact test at each aligned position; filters by significance (p<0.2) and minimum branch size (≥2)
5. **Mine** — extracts 3-5 step n-grams, tests which patterns discriminate between success and failure
6. **Synthesize** — groups related findings into ranked recommendations

## Data format

Each JSON file is one run:

```json
{
  "run_id": "run_001",
  "task_id": "task_bugfix_123",
  "harness": "baseline_v1",
  "model": "claude-3.7-sonnet",
  "steps": [
    {"idx": 0, "type": "llm", "name": "plan", "status": "ok",
     "metrics": {"tokens_in": 412, "tokens_out": 137, "latency_ms": 1840}}
  ],
  "result": {"success": true}
}
```

Step types: `llm`, `tool`, `system`, `memory`, `compaction`, `judge`, `error`, `handoff`. Converters included for [SWE-smith](scripts/convert_swe_smith.py) and [eval-harness](scripts/convert_eval_harness.py) formats.

## Limitations

moirai works on tool-call sequences — what the agent *did*, not what it *thought*. The patterns it finds are structural correlations, not causal explanations. A richer trace format (with reasoning content, error messages, and file context) would enable deeper analysis. The alignment algorithm (progressive NW) works well within clusters of similar-length runs but degrades when trajectories have very different structures.

## Roadmap

- Content-aware analysis (what the agent read/wrote, not just which tool it called)
- Semantic step grouping
- Time-series drift detection across nightly runs
- OpenTelemetry adapters
