---
title: "feat: Validate moirai's predictive signal — held-out study + reranking"
type: feat
date: 2026-04-06
---

# Validate moirai's predictive signal

The core question is narrow:

> Given multiple trajectories on the same task, can moirai identify divergence points that predict success/failure better than naive whole-trajectory scoring?

If yes, the signal is real and downstream training work (ORM, DPO) is justified.
If no, everything downstream is theater.

Two deliverables. No training infrastructure. No cloud GPUs. Just analysis on existing data.

## Brainstorm reference

`docs/brainstorms/2026-04-06-trajectory-learning-pipeline-brainstorm.md` — contains the full training pipeline design (SFT → ORM → DPO). That plan is deferred until these two deliverables validate the signal.

---

## Deliverable 1: Held-out prediction study

**Question:** Do divergence points found on training runs predict outcomes on held-out runs of the same task?

### Design

For each of the 1,096 mixed-outcome tasks (min 4 runs each, median 11):

1. **Split runs**: 70% train, 30% held-out (stratified by outcome so both splits have pass and fail runs)
2. **On train runs**: align trajectories, find divergence points via Fisher's exact test, identify high-signal branches (the ones with large pass-rate gaps)
3. **On held-out runs**: for each held-out run, check which branch it takes at each divergence point. Score the run by accumulating branch success deltas.
4. **Predict**: does the accumulated divergence-point score predict the held-out run's actual outcome?

### Scoring methods to compare

| Method | Description | What it tests |
|--------|-------------|---------------|
| **Divergence-point score** | Sum of branch success deltas at top-K divergence points from train split | Moirai's core claim: localized divergence predicts outcome |
| **Feature score** | Moirai's behavioral features (test centroid, uncertainty density, etc.) computed on the held-out run | Whether per-run features predict without alignment |
| **Step count** | Just trajectory length | Dumb baseline — longer runs might just fail more |
| **Random** | Coin flip weighted by task-level pass rate | Floor |

### Metrics

| Metric | What it measures |
|--------|-----------------|
| AUROC | Overall discrimination between pass/fail |
| Accuracy at top-K | If we pick the K highest-scored runs, what fraction actually pass? |
| Calibration | Does a score of 0.7 actually mean 70% pass rate? |

### Implementation

```
moirai/analyze/holdout.py
```

- [ ] `split_runs(task_runs, train_frac=0.7, seed=42) -> (train, test)` — stratified split preserving outcome ratio
- [ ] `train_divergence_model(train_runs) -> DivergenceModel` — align train runs, find significant divergence points, store branch success rates
- [ ] `score_run(model, run) -> float` — for a held-out run, compute score from divergence-point branch matching
- [ ] `run_holdout_study(all_task_runs, methods) -> StudyResults` — run all methods, compute metrics, return comparison table
- [ ] CLI: `moirai holdout path/to/data/ --min-runs 6 --output results.json`

Min runs bumped to 6 (need at least 2 train pass, 2 train fail, 1 test pass, 1 test fail).

### Expected output

```
$ moirai holdout /Volumes/mnemosyne/moirai/swe_rebench_v2/ --min-runs 6

Held-out prediction study — 847 tasks, 10,293 runs (70/30 split)

  Method                        AUROC    Accuracy@3   Calibration
  ────────────────────────────  ──────   ──────────   ───────────
  Divergence-point score         0.68        0.74        0.04
  Feature score (5 features)     0.63        0.69        0.06
  Step count                     0.55        0.58        0.12
  Random                         0.50        0.52        0.08
```

(Numbers are illustrative — the real question is whether divergence-point score meaningfully beats feature score and step count.)

### Success criteria

- Divergence-point AUROC > 0.60 (better than step count)
- Divergence-point AUROC > feature-score AUROC (localization adds value over per-run features)
- Signal exists at early divergence points (top-K restricted to first 30% of trajectory still discriminates)

### Failure modes

- AUROC ~0.55 for everything → signal is too weak, moirai's divergence analysis doesn't generalize to unseen runs. Stop here.
- Feature score beats divergence-point score → alignment-based divergence localization doesn't add value over simpler per-run features. Moirai simplifies to a feature extractor.
- Step count matches divergence-point score → we're just measuring trajectory length with extra steps.

---

## Deliverable 2: Reranking experiment

**Question:** If you run the agent K times on a task and use moirai to pick the best run, do you get better results than random selection?

This is the operational product claim: moirai improves best-of-K without training anything.

### Design

For each task with K runs (K ≥ 4):

1. **Sample K=3 runs** (simulating "run your agent 3 times")
2. **Score each run** using moirai-derived metrics
3. **Select the top-scored run** as the "prediction"
4. **Measure**: what fraction of the time does the selected run actually pass?

Compare against:
- **Random selection**: pick one of the 3 at random
- **Oracle**: pick the passing run if one exists (upper bound)
- **Longest run**: pick the run with the most steps (naive heuristic)

Repeat 1000 times per task with different random K=3 samples to get stable estimates.

### Scoring methods for reranking

| Method | Source |
|--------|--------|
| **Divergence-point score** | From Deliverable 1's trained model applied to each run |
| **Feature composite** | Weighted sum of moirai's 5 behavioral features |
| **Uncertainty density (inverted)** | Single strongest feature — does one feature suffice? |
| **Random** | Baseline |
| **Longest** | Naive heuristic |

### Metrics

| Metric | Definition |
|--------|-----------|
| **Selection accuracy** | P(selected run passes) across all tasks and samples |
| **Lift over random** | Selection accuracy - random baseline accuracy |
| **Lift over pass@1** | How much better than just running once? |
| **Pass@1 vs selected-of-3** | The headline number: does picking-with-moirai-of-3 beat pass@1? |

### Implementation

```
moirai/analyze/rerank.py
```

- [ ] `rerank_experiment(task_runs, scorer, k=3, n_samples=1000, seed=42) -> RerankResults`
- [ ] `RerankResults` — per-task and aggregate selection accuracy, lift, confidence intervals
- [ ] CLI: `moirai rerank path/to/data/ --k 3 --scorer divergence --output results.json`
- [ ] Sweep K=2,3,5 to see how value scales with number of runs

### Expected output

```
$ moirai rerank /Volumes/mnemosyne/moirai/swe_rebench_v2/ --k 3

Reranking experiment — 1,096 tasks, K=3, 1000 samples/task

  Random pass@1:              52.1%
  Random selection of 3:      67.8%      (binomial best-of-3)
  Longest run:                69.2%      (+1.4pp over random)
  Uncertainty (inverted):     71.5%      (+3.7pp)
  Feature composite:          72.8%      (+5.0pp)
  Divergence-point score:     73.4%      (+5.6pp)
  Oracle (best-of-3):         82.1%      (upper bound)
```

### Success criteria

- Divergence-point or feature-composite selection beats random selection of K by ≥ 3 percentage points
- The lift is consistent across task families (not driven by one family, per Simpson's Paradox concern)
- Moirai-based selection captures ≥30% of the gap between random-of-K and oracle-of-K

### Failure modes

- All methods within 1pp of random → scoring doesn't help selection. The runs are too similar or the features don't discriminate at the individual-run level.
- Longest-run matches moirai methods → we're just measuring trajectory length. No behavioral insight.
- Lift exists but disappears when stratified by task family → Simpson's Paradox again.

---

## Execution order

```
Phase 0: Implement holdout infrastructure
  - split_runs, train_divergence_model, score_run
  - ~2-3 days of moirai library work

Phase 1: Run Deliverable 1 (held-out study)
  - Run on existing SWE-rebench data (local, no GPU needed)
  - ~1 day compute + analysis

  GATE: AUROC > 0.60 for divergence-point score?
    NO  → Document findings. Moirai's divergence analysis doesn't generalize. Stop.
    YES → Proceed.

Phase 2: Implement reranking infrastructure
  - rerank_experiment, scoring methods
  - ~1-2 days

Phase 3: Run Deliverable 2 (reranking experiment)
  - Run on existing data (local, no GPU)
  - ~1 day compute + analysis

  GATE: Lift > 3pp over random selection?
    NO  → Moirai has analytical but not operational value. Reframe as pure debugging tool.
    YES → Signal is real AND operationally useful. ORM is justified.

Phase 4 (deferred): ORM, SFT, DPO
  - Only if both gates pass
  - Full plan in brainstorm doc
```

### Cost

$0. Everything runs on existing data, locally, on CPU. No cloud GPUs, no training, no eval infrastructure.

---

## New code required

| File | What | Deliverable |
|------|------|-------------|
| `moirai/analyze/holdout.py` | Hold-out study infrastructure | 1 |
| `moirai/analyze/rerank.py` | Reranking experiment infrastructure | 2 |
| `moirai/cli.py` | `moirai holdout` and `moirai rerank` commands | 1, 2 |
| `moirai/viz/terminal.py` | Print functions for study results | 1, 2 |
| `tests/test_holdout.py` | Tests for holdout study | 1 |
| `tests/test_rerank.py` | Tests for reranking experiment | 2 |

---

## What this plan explicitly defers

Everything in the brainstorm's training pipeline:
- SFT on mixed-outcome successes
- ORM / process reward model (tabular or otherwise)
- Localized DPO fine-tuning
- OpenHands eval infrastructure
- SWE-bench Verified benchmark runs
- Cloud GPU compute
- Four-way composition experiment

Those are earned by passing the gates above, not assumed.

---

## Acceptance criteria

### Deliverable 1 (held-out study)
- [ ] `moirai holdout` command works end-to-end on SWE-rebench data
- [ ] Four scoring methods compared (divergence-point, feature, step-count, random)
- [ ] Results include AUROC, accuracy@K, calibration for each
- [ ] Analysis of early vs late divergence points (restricted to first 30% of trajectory)
- [ ] Results stratified by task family
- [ ] Clear go/no-go answer on whether divergence localization predicts held-out outcomes

### Deliverable 2 (reranking experiment)
- [ ] `moirai rerank` command works end-to-end
- [ ] Sweep over K=2,3,5
- [ ] Five scoring methods compared (divergence, feature composite, uncertainty, longest, random)
- [ ] Lift computed with confidence intervals (bootstrap)
- [ ] Results stratified by task family
- [ ] Clear go/no-go answer on whether moirai-based selection has operational value
