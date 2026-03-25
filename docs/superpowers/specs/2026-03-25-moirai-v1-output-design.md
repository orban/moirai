# moirai v1: output redesign

The analysis engine works. The output doesn't. This spec fixes the output layer.

## Problem

Real agent trajectories are 20-60 steps long. v0 output has three failures:

1. **Signatures are unreadable.** `system > tool > judge > tool > system > tool > system > tool > ...` for 30+ steps. Nobody can parse this.
2. **Divergence points lack context.** "col=30, judge: 4 runs 100%, tool: 4 runs 50%" — which runs? What led there? What happened after?
3. **No path from insight to action.** You see a divergence, then what? There's no way to drill into specific runs or understand what makes a cluster succeed.

## Design: three changes

### 1. Compressed signatures

Replace raw step sequences with run-length encoded, semantically grouped signatures.

**Before:**
```
error_observation > read > error_observation > action > error_observation > action >
error_observation > action > error_observation > read > error_observation > action >
error_observation > edit > error_observation > action > write > test_result > action >
error_observation > search > test_result > action > error_observation > read >
error_observation > action > action
```

**After:**
```
read×2 → action×3 → edit → write → test(fail) → search → test(fail) → read → action×2
[28 steps, 5 phases]
```

**Rules:**

- Consecutive identical steps collapse: `read > read > read` → `read×3`
- Noisy pairs collapse: `error_observation > action` is just `action` (the observation is implicit)
- Test results annotate: `test_result` with `status=error` → `test(fail)`, `status=ok` → `test(pass)`
- Phase boundaries at step type changes: a "phase" is a contiguous block of the same high-level activity (exploration, modification, testing)

**Phase classification:**

| Steps | Phase |
|---|---|
| read, search | explore |
| edit, write | modify |
| test, test_result | verify |
| bash | execute |
| reason, subagent | think |
| error_observation, plan | system |

**Cluster prototype becomes:**
```
Cluster 0 (21 runs, 71% success)
  explore → modify → verify(fail) → explore → modify → verify(pass)
  avg 33 steps across 5 phases, 38% read, 10% edit, 19% test
```

### 2. Divergence narratives

When a divergence point is found, show the *story* of what happened, not just the statistics.

**Before:**
```
col=30 entropy=0.89
  judge: 4 runs, success 100.0%
  tool: 4 runs, success 50.0%
  system: 34 runs, success 67.6%
```

**After:**
```
Divergence at position 30 (entropy 0.89)

  Branch A: verify (4 runs, 100% success)
    Runs that test here always succeed.
    Context: explore×4 → modify → [VERIFY] → pass
    Runs: django-money__835c, marshmallow__8b42, ...

  Branch B: modify (4 runs, 50% success)
    Runs that keep editing here succeed half the time.
    Context: explore×4 → [MODIFY] → verify(fail) → modify → verify(pass|fail)
    Runs: mozillazg__e42d, pydantic__3a1c, ...

  Branch C: explore (34 runs, 68% success)
    Most runs are still exploring at this point.
    Context: explore×3 → [EXPLORE more] → modify → verify
```

**Implementation:**

For each branch at a divergence point:
1. Collect the runs that took that branch
2. Show the compressed signature of the 5 steps before and 5 steps after the divergence
3. Show success rate and run count
4. List 2-3 representative run IDs (closest to the branch centroid)

### 3. `moirai explain <run_id>`

New command that answers: "What happened in this run and why did it succeed/fail compared to similar runs?"

**Output:**
```
$ moirai explain examples/swe_smith/ --run django-money__835c...

Run: django-money__835c... [PASS]
Task: django-money__django-money.835c1ab8
Cluster: 0 (21 runs, 71% success)

Trajectory:
  explore: read×2, search×1
  modify:  edit×1
  verify:  test(pass)
  [5 phases, 12 steps — shorter than cluster avg of 33]

Compared to cluster:
  This run is 64% shorter than average.
  It edited once and passed on first test — typical of the fast-solve subpattern.

  In failing runs from this cluster:
  - 83% have a verify(fail) followed by more modify steps
  - 67% have 3+ edit attempts before passing
  - The most common failure pattern: explore → modify → verify(fail) → modify → verify(fail) → explore → ...

Key divergence:
  At position 6, this run went to VERIFY while most failing runs went to MODIFY.
  Runs that verify early: 4/4 success (100%)
  Runs that modify again: 2/4 success (50%)
```

**Implementation:**

1. Load all runs, find the target run
2. Cluster, find which cluster it belongs to
3. Compute the alignment of runs in that cluster
4. Find where the target run diverges from failing runs
5. Show compressed trajectory, cluster context, and the key divergence

### 4. Updated cluster output

**Before:**
```
Cluster 0
  count: 21
  success: 71.4%
  prototype: system > tool > system > tool > system > tool > system > tool > ...
```

**After:**
```
Cluster 0: 21 runs, 71% success
  Pattern: explore → modify → verify(fail) → explore → modify → verify(pass)
  Avg 33 steps, 5 phases
  Steps: 38% explore, 10% modify, 19% verify, 33% other
  Errors: test failures drive retry loops

  Fast solvers (8 runs, 100% success):
    explore → modify → verify(pass)  [<15 steps]

  Retry loop (10 runs, 60% success):
    explore → modify → verify(fail) → modify → verify  [20-40 steps]

  Stuck (3 runs, 0% success):
    explore → modify → verify(fail) → explore → modify → verify(fail) → ...  [>40 steps]
```

**Implementation:**

Within each cluster, sub-cluster by trajectory length or phase count to identify behavioral sub-patterns. Show the 2-3 most common sub-patterns with their success rates.

### 5. Updated diff output

**Before:**
```
Cluster shifts:
  tool > tool > tool > tool > tool > tool > ... : +20 runs
```

**After:**
```
Cluster shifts:
  explore→modify→verify(pass)  [fast solve]:  +9 runs in B
  explore→modify→verify(fail)→...  [retry loop]:  -6 runs in B
  explore→...→explore  [stuck]:  -2 runs in B

B eliminated most retry loops and stuck runs.
The fast-solve pattern (+9) accounts for most of the success rate improvement.
```

---

## What stays the same

- Data model (Run, Step, Result)
- Analysis engine (NW alignment, clustering, divergence detection)
- CLI structure (validate, summary, trace, clusters, branch, diff)
- JSON input format

The changes are purely in the presentation layer (`viz/terminal.py`) plus one new command (`explain`) and a new module for signature compression.

## Implementation approach

1. `moirai/compress.py` — signature compression, phase classification, run-length encoding
2. Update `viz/terminal.py` — use compressed signatures everywhere
3. Divergence narratives in `print_divergence` — context windows around split points
4. `moirai explain` command — new CLI command, new analysis combining cluster + divergence + comparison
5. Sub-pattern detection in clusters — length-based sub-clustering within each cluster

## Priority

1 and 2 are the highest leverage — they fix the unreadable output everywhere.
3 makes divergence actionable.
4 is the crown jewel but depends on 1-3.
5 is polish.
