# moirai

Turn your agent's stochastic variation into training signal.

When the same agent runs the same task multiple times, it sometimes passes and sometimes fails. That variation isn't noise — it's data. moirai extracts it: aligns trajectories, finds where behavior diverges, identifies which choices predict success, and produces preference pairs for fine-tuning.

## The pipeline

```bash
# 1. What predicts success?
moirai features examples/swe_rebench --min-runs 4 --output results.json

# 2. Where do runs diverge on a specific task?
moirai branch examples/swe_rebench --task "vyperlang__vyper-4385" \
    --feature test_position_centroid

# 3. What failure modes recur across tasks?
moirai divergences examples/swe_rebench --output divergences.json

# 4. Extract training signal
moirai export --format dpo examples/swe_rebench --output pairs.jsonl
```

### Step 1: Discover behavioral features

```
$ moirai features examples/swe_rebench --min-runs 4

Behavioral features — 1,038 mixed-outcome tasks, 12,854 runs

  Feature                              Delta   p-value   q-value  Split-half
  ──────────────────────────────────  ───────  ────────  ────────  ──────────
  uncertainty_density                  -7.0pp   <0.001    <0.001  ✓
  symmetry_index                       +6.6pp   <0.001    <0.001  ✓
  test_position_centroid               +6.5pp   <0.001    <0.001  ✓
  reasoning_hypothesis_formation       -5.6pp   <0.001    <0.001  ✓
  edit_dispersion                      +4.9pp   <0.001    <0.001  ✓
```

For each feature, moirai runs a within-task natural experiment: split runs at the median, compare pass rates. This controls for task identity — the comparison is always within the same task, between runs of the same agent.

### Step 2: Drill into one task

```
$ moirai branch examples/swe_rebench --task "vyperlang__vyper-4385" \
    --feature test_position_centroid

  Feature: test_position_centroid (higher = better)
  Median split: HIGH (>0.69) = 100% pass | LOW (≤0.69) = 50% pass

    fail  test_position_centroid: 0.14  LOW
    fail  test_position_centroid: 0.28  LOW
    pass  test_position_centroid: 0.68  LOW
    pass  test_position_centroid: 0.78  HIGH
    pass  test_position_centroid: 0.84  HIGH
```

The `--feature` flag connects the two tools: the feature name from step 1 feeds into `moirai branch`, which annotates each run with the feature value and shows the within-task median split.

### Step 3: Find recurring failure modes

```
$ moirai divergences examples/swe_rebench

  Cluster                              Size    %   Position
  ────────────────────────────────    ────  ────   ────────
  Search strategy divergence           944   31%      0.13
  Wasted orientation                   532   18%      0.03
  Execution approach divergence        443   15%      0.11
  Wrong file targeted                  363   12%      0.07
  Test timing divergence               298   10%      0.12
  ...
```

### Step 4: Extract DPO pairs

```
$ moirai export --format dpo examples/swe_rebench --output pairs.jsonl
# → 11,006 preference pairs from 1,096 tasks
```

At each divergence point, moirai extracts (context, chosen, rejected) triples — what the pass run did vs what the fail run did at the same decision point. These are input-ready for Direct Preference Optimization.

## Why this matters

**Your agent's failures contain the training signal to fix them.** Every eval run produces trajectories. Most get a pass/fail label and get discarded. When the same agent passes and fails on the same task, the difference between those trajectories is a preference signal — a specific moment where a different choice would have led to a different outcome.

moirai extracts that signal:
- **Behavioral features** → process reward signals (dense, step-level, not just sparse pass/fail)
- **Divergence points** → DPO preference pairs (context + chosen + rejected)
- **Failure mode clusters** → diagnostic patterns for monitoring and debugging

You're already generating this data. Stop throwing it away.

## Install

```bash
pip install -e .
```

Requires Python 3.11+.

## Commands

```bash
moirai features path/to/runs/          # rank behavioral features by predictive power
moirai branch path/to/runs/            # per-task alignment and divergence analysis
moirai divergences path/to/runs/       # cluster divergence points into failure modes
moirai export --format dpo path/       # extract DPO preference pairs
moirai validate path/to/runs/          # check trace files
moirai summary path/to/runs/           # aggregate stats
moirai trace path/to/run.json          # inspect one run
moirai clusters path/to/runs/          # find behavioral strategy clusters
moirai patterns path/to/runs/          # find success/failure step patterns
moirai explain path/to/runs/           # content-aware divergence analysis
moirai diff path/ --a K=V --b K=V     # compare cohorts
```

All commands support `--model`, `--harness`, `--task-family` filters.
`moirai branch` supports `--task` (filter to one task) and `--feature` (annotate with a behavioral feature).

## How it works

1. **Normalize** — loads JSON trace files, enriches step labels from attrs (file type, command type, search specificity)
2. **Align** — Needleman-Wunsch alignment over step sequences, progressive multi-sequence alignment
3. **Test** — Fisher's exact test at each aligned position: does having a step here (vs a gap) predict outcome?
4. **Features** — compute behavioral features per run, validate with within-task median-split natural experiments, split-half validation, BH correction
5. **Extract** — at each divergence point, extract (context, chosen, rejected) preference pairs for DPO

## Data

moirai works with any agent traces in its JSON format. Converters included for:
- [SWE-rebench](scripts/convert_swe_rebench.py) (Nebius/OpenHands trajectories)
- [SWE-bench experiments](scripts/convert_swebench_experiments.py) (SWE-bench/experiments S3 data)
- [SWE-smith](scripts/convert_swe_smith.py)
- [eval-harness](scripts/convert_eval_harness.py)

The results in the blog post use [nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) (CC-BY-4.0): 12,854 runs of OpenHands + Qwen3-Coder-480B across 1,096 mixed-outcome tasks.

## End-to-end example

```bash
# Download and convert data
python scripts/convert_swe_rebench.py /path/to/trajectories examples/swe_rebench

# Discover features
moirai features examples/swe_rebench --min-runs 4 --output results.json

# Drill into one task
moirai branch examples/swe_rebench --task "vyperlang__vyper-4385" \
    --feature test_position_centroid

# Extract DPO pairs
moirai export --format dpo examples/swe_rebench --output pairs.jsonl
```

Or run the documented example script:
```bash
python examples/feature_analysis.py examples/swe_rebench --output results.json --verbose
```

## Limitations

- **One agent, one benchmark.** The current findings are from OpenHands + Qwen3-Coder on SWE-rebench. They may not generalize.
- **Observational, not causal.** The features predict outcomes but we haven't run an intervention study. Early testing might cause failure, or confused agents might test early — both are detectable, neither is proven.
- **Small effect sizes.** 5-7 percentage points from stochastic variation. The significance comes from consistency across 1,000+ tasks, not from any single task.
- **Behavioral, not mechanistic.** moirai works on what the agent *did*, not what it *computed*. It doesn't look at model internals.

## Blog post

[What Stochastic Variation Reveals About AI Agents](https://ryanorban.com/posts/what-stochastic-variation-reveals/) walks through the methodology and findings with inline visualizations.
