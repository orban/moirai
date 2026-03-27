# moirai

Trajectory-level debugging for stochastic agent systems.

Align rollouts, identify divergence points, and surface behavioral branches that predict success or failure.

```
                 ┌─ test → plan → edit ────── success (74%)
run 1─┐          │
run 2─┼─ read → test → test ─┤
run 3─┤          │            └─ search → search ──── failure (0%)
run 4─┘          │
                 divergence point (p=0.075)
```

## The core idea

Stochastic agents don't fail randomly. When you run the same agent 100 times on the same task, the failures cluster into a small number of recurring behavioral branches. Those branches are identifiable from trajectory structure — *before* you inspect reasoning traces, model internals, or individual logs.

moirai treats agent runs as sequences of steps (tool calls, reasoning turns, test executions), aligns them using Needleman-Wunsch (the same algorithm used for DNA sequence alignment), and statistically tests each aligned position for outcome-predictive divergence.

Across 1000 SWE-bench agent traces, this surfaces findings like:

```
$ moirai branch runs/

Cluster 13: 109 runs, 74% success
  Position 35 (p=0.075)
  verify→
    test_result: 77 runs, 74% success     ← agents that keep testing pass
    search:       2 runs,  0% success     ← agents that go back to searching always fail
```

## Why existing tools miss this

**Aggregate metrics** (pass rate, avg tokens, cost) tell you *that* 30% of runs fail but not *whether* they fail the same way or five different ways. A harness change that improves pass rate by 10% might have fixed one failure mode and introduced another. You can't see this from the numbers.

**Raw traces and trace viewers** show you what happened in one run. But if you're running 100 agents on the same task, reading individual traces doesn't scale. You need population-level analysis — what do the successful runs have in common that the failures don't?

**Pass/fail thinking** treats each run as an independent coin flip. But agent behavior is structured. Runs that start with the same exploration strategy often converge to the same outcome. The question isn't "did it pass" but "which behavioral branch did it take, and does that branch predict the outcome?"

moirai operates at the layer between individual traces and aggregate metrics: structural analysis of trajectory populations.

## Why this matters

### For interpretability
Trajectory alignment identifies *where* in a rollout interpretability is most useful. Instead of reasoning about every token across a 50-step trajectory, you can focus on the 2-3 aligned positions where the agent's choice statistically predicts success or failure. These divergence points are where the model's decision-making actually matters — and where mechanistic analysis has the highest leverage.

### For agent monitoring
Recurring failure branches are more useful than raw anomaly detection on individual traces. If 45% of your failing runs share the same structural pattern (`test → bash → bash`, 5% success rate), that's a detector you can build and deploy — not a one-off incident to investigate. moirai gives you the recurring behavioral signatures, not just the individual alerts.

### For simulation and environment design
Divergence points reveal which environment choices or early decisions dominate downstream outcomes. If agents that get a specific observation at step 16 succeed 67% of the time while those that don't succeed 0%, that's a signal about environment design — not agent capability. Trajectory alignment separates "the agent made a bad choice" from "the environment presented a hard fork."

## Walkthrough: 1000 SWE-bench agent traces

### Runs cluster into behavioral modes

```
$ moirai clusters runs/

Cluster 18: 149 runs, 85% success
  explore → verify(fail) → modify → verify(fail)×2
  normal (143 runs, 85%): read → test(fail) → write → edit → test(fail) → write...

Cluster 8:  83 runs, 49% success
  explore → verify(fail)×2 → explore → verify(fail)×4
  normal (83 runs, 49%): read → test(fail)×2 → search → test(fail)×3 → write...
```

1000 runs aren't 1000 unique strategies. They're variations on ~10 behavioral patterns with different success rates. Cluster 18 (explore, then modify) succeeds 85%. Cluster 8 (explore, then explore more) succeeds 49%.

### Within clusters, divergence points split outcomes

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

At position 35, agents are in a verification phase. 77 runs continue testing — 74% of them succeed. 2 runs go back to searching — both fail. Abandoning a verification strategy to restart exploration at this point is a death sentence.

### Pattern mining finds recurring failure signatures

```
$ moirai patterns runs/

Patterns that predict success:
  write → edit → reason                              100% success (13 runs)  p=0.047
  edit → test_result → write → test_result → reason  100% success (12 runs)  p=0.044

Patterns that predict failure:
  write → test_result → read → read → test_result      0% success (4 runs)  p=0.004
  read → read → read → test_result                    17% success (6 runs)  p=0.005
```

Success patterns contain `reason` — the agent pauses to think between actions. Failure patterns are loops: `read → read → read` (exploring without acting) or repeated oscillation without convergence.

### Explain any specific run

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

### HTML dashboard

`moirai branch --html report.html` generates an analysis dashboard:

![moirai dashboard](docs/images/dashboard.png)

## What this is not

- **Not a trace viewer.** moirai doesn't visualize individual runs. It analyzes populations of runs.
- **Not a benchmark harness.** It doesn't run agents or compute pass/fail. It takes existing traces as input.
- **Not a generic analytics dashboard.** There are no time-series charts, percentile breakdowns, or cost summaries. The only question moirai answers is: *where do trajectories diverge, and does it matter?*
- **Not a replacement for interpretability tools.** It doesn't look at model internals, attention patterns, or activations. It works at the behavioral level — what the agent *did*, not what it *computed*.

What it is: a way of treating stochastic rollouts as structured sequences, aligning them, and finding the recurring behavioral branches that statistically predict outcomes.

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

All directory commands support `--model`, `--harness`, `--task-family` filters.

## How it works

1. **Normalize** — loads JSON trace files, maps step type aliases, filters noise
2. **Cluster** — pairwise Needleman-Wunsch edit distance over step sequences, agglomerative clustering
3. **Align** — progressive multi-sequence alignment within each cluster at step-name level
4. **Test** — Fisher's exact test at each aligned position; filters by significance (p<0.2) and stability (min 2 runs per branch)
5. **Mine** — extracts 3-5 step n-grams, tests which patterns discriminate between success and failure

## Data format

Each JSON file is one run. Step types: `llm`, `tool`, `system`, `memory`, `compaction`, `judge`, `error`, `handoff`. Converters included for [SWE-smith](scripts/convert_swe_smith.py) and [eval-harness](scripts/convert_eval_harness.py) formats.

## Limitations

moirai works on tool-call sequences — what the agent *did*, not what it *thought*. The patterns it finds are structural correlations, not causal explanations. "Runs that loop on bash fail" might mean "stuck agents thrash with bash," not "bash causes failure." Richer traces (with reasoning content, error messages, file context) would enable deeper analysis.

## Roadmap

- Content-aware analysis (what the agent read/wrote, not just which tool it called)
- Semantic step grouping beyond tool names
- Time-series drift detection across nightly runs
- OpenTelemetry adapters
