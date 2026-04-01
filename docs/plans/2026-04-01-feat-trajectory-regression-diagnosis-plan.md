---
title: "feat: trajectory-aware regression diagnosis"
type: feat
date: 2026-04-01
---

# Trajectory-Aware Regression Diagnosis

## Overview

Extend moirai from an observation engine into a diagnosis system. Given baseline and current agent rollouts plus a list of candidate system changes, produce ranked causes of regression with uncertainty estimates, grounded in trajectory-level structural evidence.

The key insight: **pass-rate-based evals are blind to behavioral regressions**. An agent can maintain 70% pass rate while fundamentally changing its problem-solving strategy — solving different tasks, taking different paths. Trajectory structure reveals what aggregate metrics hide.

This plan is **demo-first**. Every design decision serves a runnable end-to-end demonstration that proves three things:

1. Pass-rate evals miss regressions that trajectory analysis catches
2. Structural evidence (branch shifts, motif changes) localizes the cause
3. Evidence-weighted cause ranking identifies the responsible change

## Problem statement

Today's agent eval workflow:

```
deploy change → run eval suite → check pass rate → ship or revert
```

This misses:
- **Simpson's paradox**: pass rate unchanged, but the agent passes easy tasks it used to fail and fails hard tasks it used to pass
- **Strategy regression**: agent stops testing fixes, starts blind-submitting — works on simple tasks, catastrophic on complex ones
- **Masked degradation**: model upgrade improves capability but prompt change simultaneously degrades planning — net effect is zero, but both problems exist

Moirai already detects structural differences between pass and fail trajectories. What's missing is the cause-ranking layer: given observed trajectory shifts between variants, which system change is responsible?

## The demo: "The Invisible Regression"

### Scenario

A team deploys a system prompt change to their SWE-bench agent and wants to verify nothing broke.

The change: **removed "always run tests before submitting your fix"** from the system prompt.

Results:
| Variant   | Runs | Pass rate |
|-----------|------|-----------|
| baseline  | 100  | 68%       |
| current   | 100  | 68%       |

Traditional eval: **"No regression detected. Ship it."**

Three candidate causes are under investigation:
- **C1**: System prompt edit (removed testing instruction)
- **C2**: Model upgrade (claude-3.5-sonnet → claude-4-sonnet)
- **C3**: Tool timeout reduction (120s → 60s)

### What trajectory analysis reveals

**Canonical behavioral features** (computed from trajectory structure):

| Feature | Baseline | Current | Shift |
|---------|----------|---------|-------|
| `TEST_AFTER_EDIT_RATE` | 0.88 | 0.31 | -0.57 |
| `ITERATIVE_FIX_RATE` (edit→test→edit motif) | 0.72 | 0.24 | -0.48 |
| `BLIND_SUBMIT_RATE` (edit→submit, no test) | 0.12 | 0.67 | +0.55 |
| `TOOL_TIMEOUT_ERROR_RATE` | 0.03 | 0.04 | +0.01 |
| `AVG_STEP_COUNT` | 14.2 | 9.8 | -4.4 |

**Per-task breakdown** (10 tasks, 10 runs each per variant):
- Hard tasks (5): baseline 76% → current 48% (-28pp)
- Easy tasks (5): baseline 60% → current 88% (+28pp)
- Net: 68% → 68% — invisible to pass-rate

### Cause ranking

Each candidate cause declares which behavioral features it would affect:

```json
{
  "id": "C1", "type": "prompt",
  "description": "Removed testing instruction",
  "expected_shifts": {
    "TEST_AFTER_EDIT_RATE": "decrease",
    "ITERATIVE_FIX_RATE": "decrease",
    "BLIND_SUBMIT_RATE": "increase"
  }
}
```

Evidence-weighted ranking (not Bayesian — this is honest scoring with uncertainty):

| Cause | Score | Confidence | Key evidence |
|-------|-------|------------|--------------|
| C1: prompt change | **0.82** | high | TEST_AFTER_EDIT_RATE shifted -0.57, BLIND_SUBMIT_RATE +0.55, hard tasks regressed |
| C2: model upgrade | 0.12 | low | easy task improvement, but no feature matches |
| C3: tool timeout | 0.04 | low | TOOL_TIMEOUT_ERROR_RATE unchanged (+0.01) |
| Unknown cause | 0.02 | — | residual |

### Negative control

To validate the ranking model, the demo includes a second scenario where the *tool timeout* is the actual cause:

| Feature | Baseline | Current (timeout) | Shift |
|---------|----------|--------------------|-------|
| `TEST_AFTER_EDIT_RATE` | 0.88 | 0.85 | -0.03 |
| `TOOL_TIMEOUT_ERROR_RATE` | 0.03 | 0.28 | +0.25 |
| `STEP_FAILURE_RATE` | 0.05 | 0.19 | +0.14 |

The ranking correctly flips: C3 scores highest. This proves the model isn't hardwired to pick prompt changes.

### Robustness

The generator runs with 3 random seeds. Ranking order must be stable across all 3 seeds to pass.

### Demo output

```
$ moirai diagnose runs/ --baseline variant=baseline --current variant=current \
    --causes causes.json

Regression diagnosis
====================

Pass rate: baseline 68.0% → current 68.0%
          (no change detected by aggregate metrics)

Trajectory structure reveals behavioral shift:

  Feature                    Baseline  Current  Shift   Effect
  TEST_AFTER_EDIT_RATE       0.88      0.31     -0.57   large
  ITERATIVE_FIX_RATE         0.72      0.24     -0.48   large
  BLIND_SUBMIT_RATE          0.12      0.67     +0.55   large
  TOOL_TIMEOUT_ERROR_RATE    0.03      0.04     +0.01   negligible

  Per-task:
    Hard tasks (5): 76% → 48%  (-28pp)
    Easy tasks (5): 60% → 88%  (+28pp)

Cause ranking:
  #1  C1: prompt change (removed testing instruction)
      score: 0.82  [0.71, 0.91]
      matched features: TEST_AFTER_EDIT_RATE, ITERATIVE_FIX_RATE, BLIND_SUBMIT_RATE

  #2  C2: model upgrade
      score: 0.12  [0.04, 0.22]
      matched features: (none — easy task improvement is indirect)

  #3  C3: tool timeout reduction
      score: 0.04  [0.01, 0.10]
      matched features: TOOL_TIMEOUT_ERROR_RATE (negligible shift)

  Unknown/unmodeled: 0.02

3 seeds tested: ranking stable (C1 first in all 3)
```

## Technical approach

### Architecture

```
Layer 1: Moirai (existing — no schema changes needed)
├── ingestion + normalization
├── alignment (NW)
├── clustering (scipy hierarchical)
├── divergence detection
├── motif mining
└── explain (pairwise comparison)

Layer 2: Evidence extraction (new — moirai/analyze/evidence.py)
├── compare_variants()           ← orchestrator
├── extract_behavioral_features() ← canonical feature extractor
├── extract_branch_evidence()    ← per-divergence-point comparison
├── extract_pattern_evidence()   ← motif frequency comparison
└── extract_task_breakdown()     ← per-task pass-rate delta

Layer 3: Cause ranking (new — moirai/diagnose/)
├── CandidateCause schema + loader
├── score_causes()              ← evidence-weighted ranking
├── DiagnosisResult             ← ranked causes + confidence
└── bootstrap_confidence()      ← rank stability via resampling
```

No experiment selection layer in MVP. That's deferred.

### Key design decisions

**1. No schema changes for variant awareness.**
The existing `apply_kv_filters()` in `filters.py` already supports `variant=baseline` via `run.tags`. The `diff` command already uses `--a` and `--b` K=V filters. No need to add `variant_id` to `Run` — just use tags. (Credit: Codex caught this.)

**2. Canonical behavioral features, not raw trajectory tokens.**
Instead of mapping causes to `affected_components` (free-text that won't bind to step names), we extract deterministic canonical features from trajectory structure:

```python
BEHAVIORAL_FEATURES = {
    "TEST_AFTER_EDIT_RATE": "fraction of runs where a test step follows an edit step",
    "ITERATIVE_FIX_RATE": "fraction of runs containing edit→test→edit motif",
    "BLIND_SUBMIT_RATE": "fraction of runs with edit→submit and no intervening test",
    "TOOL_TIMEOUT_ERROR_RATE": "fraction of steps with timeout/error status",
    "STEP_FAILURE_RATE": "fraction of steps with error status",
    "AVG_STEP_COUNT": "mean trajectory length",
    "SEARCH_BEFORE_EDIT_RATE": "fraction of runs with search→edit pattern",
    "REASONING_DENSITY": "fraction of steps with non-empty reasoning",
}
```

Each `CandidateCause` declares which features it expects to shift *and in which direction*:
```json
{
  "expected_shifts": {
    "TEST_AFTER_EDIT_RATE": "decrease",
    "ITERATIVE_FIX_RATE": "decrease",
    "BLIND_SUBMIT_RATE": "increase"
  }
}
```

The scoring function matches observed shifts against expected direction. A feature that shifts the expected way is evidence *for* the cause; a feature that shifts the opposite way is evidence *against*. This is deterministic, inspectable, direction-aware, and won't break on vocabulary drift.

**3. Evidence-weighted ranking, not Bayesian inference.**
The plan originally called this "Bayesian." It isn't — it's additive scoring normalized to a simplex. Calling it "Bayesian" is dishonest. The actual model has one formulation (not two):

```
# For each cause:
raw_score = 0.0
for feature, expected_direction in cause.expected_shifts.items():
    actual_shift = feature_shifts[feature].shift
    if direction_matches(actual_shift, expected_direction):
        raw_score += abs(actual_shift)       # evidence FOR
    else:
        raw_score -= abs(actual_shift) * 0.5  # evidence AGAINST (damped)

# Unclaimed features go to unknown bucket:
unclaimed_shift = sum(abs(fs.shift) for fs in feature_shifts
                      if fs.feature not in any cause's expected_shifts)
unknown_raw = unclaimed_shift

# Normalize with priors:
weighted = {cause.id: cause.prior * exp(raw_scores[cause.id]) for cause in causes}
weighted["unknown"] = unknown_prior * exp(unknown_raw)
total = sum(weighted.values())
final_scores = {k: v / total for k, v in weighted.items()}
```

Direction matching: `"decrease"` matches negative shift, `"increase"` matches positive shift. A feature that shifts the opposite way from expectation is evidence *against* the cause (damped by 0.5 to avoid overcorrection from noisy single features).

**Priors**: `prior: 0.0` means "auto-assign uniform." Before scoring, any cause with prior=0.0 gets `1.0 / (n_causes + 1)` (the +1 is for the unknown bucket). The unknown bucket's default prior is also `1.0 / (n_causes + 1)`.

This is a scoring heuristic with explicit priors. It's interpretable, testable, and honest about what it is.

**4. Deduplication of correlated evidence.**
Branch shifts, motif shifts, and cluster drift can all reflect the same behavioral change (e.g., agents stopped testing). To avoid double-counting, we extract features at the **behavioral feature** level (one number per feature), not at the raw evidence level. Each feature contributes once to the score, regardless of how many evidence types detected it.

**Acknowledged limitation**: some canonical features are mechanically correlated (e.g., `TEST_AFTER_EDIT_RATE` dropping causes `BLIND_SUBMIT_RATE` to rise). A cause listing both will get inflated scores. The bootstrap CI captures this honestly (higher variance in resampled scores). For MVP we accept this; a v2 improvement would group correlated features and take the strongest signal from each group.

**5. Pooled alignment for divergence, per-task for breakdown.**
Branch evidence uses pooled alignment across all runs (both variants, all tasks). This is what the existing `align_runs()` + `find_divergence_points()` already does. Task breakdown is separate: group by task_id, compute per-task pass rate delta. These are independent analyses, not a single alignment.

**6. Unknown-cause bucket.**
The posterior always includes an "unknown/unmodeled" entry that captures residual mass. If no candidate cause explains the observed shifts well, the unknown bucket dominates. This prevents false certainty.

### Implementation phases

#### Phase 1: Evidence extraction API (`moirai/analyze/evidence.py`)

Build first, before the generator. This defines the outputs the generator must target.

**Behavioral feature extraction:**

```python
# moirai/analyze/evidence.py

def extract_behavioral_features(runs: list[Run]) -> dict[str, float]:
    """Extract canonical behavioral features from a set of runs.

    Returns feature_name → value (typically a rate in [0, 1]).
    """

def compare_variants(
    baseline_runs: list[Run],
    current_runs: list[Run],
    level: str = "name",
) -> VariantComparison:
    """Compare two variants and extract structured evidence.

    1. Extract behavioral features per variant
    2. Compute feature shifts with effect sizes
    3. Run per-task breakdown
    4. Package into VariantComparison
    """
```

**Evidence dataclasses** (in `schema.py`):

```python
@dataclass
class FeatureShift:
    """A single behavioral feature comparison between variants."""
    feature: str            # canonical name (e.g., "TEST_AFTER_EDIT_RATE")
    baseline_value: float
    current_value: float
    shift: float            # current - baseline
    effect_size: float      # Cohen's h for proportions, Cohen's d for means
    ci_lower: float         # 95% CI on shift
    ci_upper: float
    magnitude: str          # "negligible", "small", "medium", "large"

@dataclass
class TaskBreakdown:
    """Per-task success rate change between variants."""
    task_id: str
    baseline_pass_rate: float
    current_pass_rate: float
    delta: float
    baseline_runs: int
    current_runs: int

@dataclass
class VariantComparison:
    """Complete structured comparison of two variants."""
    baseline_label: str
    current_label: str
    baseline_pass_rate: float
    current_pass_rate: float
    pass_rate_delta: float
    feature_shifts: list[FeatureShift]
    task_breakdown: list[TaskBreakdown]
```

**Effect sizes and CIs** (in `stats.py`):

```python
def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h for two proportions."""
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))

def proportion_delta_ci(
    baseline_p: float, baseline_n: int,
    current_p: float, current_n: int,
    z: float = 1.96,
) -> tuple[float, float]:
    """CI on the difference (current - baseline) of two independent proportions.

    Wald interval. For FeatureShift CIs on rate-type features.
    Bootstrap CI used for non-rate features (e.g., AVG_STEP_COUNT).
    """
    se = math.sqrt(baseline_p * (1 - baseline_p) / baseline_n
                   + current_p * (1 - current_p) / current_n)
    delta = current_p - baseline_p  # matches FeatureShift.shift = current - baseline
    return (delta - z * se, delta + z * se)
```

Note: `FeatureShift.ci_lower`/`ci_upper` come from `proportion_delta_ci` for rate features and from bootstrap resampling for non-rate features (like `AVG_STEP_COUNT`). This is a Wald interval — acceptable for n>=30; bootstrap is the fallback for small samples.

- [ ] Define `FeatureShift`, `TaskBreakdown`, `VariantComparison` in `schema.py`
- [ ] Implement `extract_behavioral_features()` in `moirai/analyze/evidence.py`
- [ ] Implement `compare_variants()` in `moirai/analyze/evidence.py`
- [ ] Add `cohens_h()`, `proportion_delta_ci()` to `stats.py`
- [ ] Add `moirai evidence` CLI command
- [ ] Tests for feature extraction and variant comparison

#### Phase 2: Synthetic data generator (`examples/diagnosis_demo/`)

Built *after* Phase 1, targeting the evidence API's actual outputs.

**Primary scenario** ("The Invisible Regression"):
- 10 tasks, 10 runs each per variant (100 baseline + 100 current)
- Pass rates: ~68% for both variants
- Behavioral shift: testing frequency drops dramatically in current
- Cause: prompt change (C1)

**Negative control** ("Tool Timeout"):
- Same task structure, but the actual cause is tool timeout (C3)
- TEST_AFTER_EDIT_RATE barely changes, TOOL_TIMEOUT_ERROR_RATE spikes
- Validates that the ranking model correctly identifies C3

**Robustness**:
- Generator accepts `--seed` flag
- Demo script runs with seeds 42, 123, 456
- Ranking must be stable across all seeds

```python
# examples/diagnosis_demo/generate.py

def generate_scenario(
    scenario: str,        # "prompt_regression" or "timeout_regression"
    output_dir: Path,
    n_tasks: int = 10,
    runs_per_task: int = 10,
    seed: int = 42,
) -> None:
    """Generate synthetic moirai-format runs for a regression scenario."""
```

- [ ] `generate.py` with primary + negative control scenarios
- [ ] `causes.json` — 3 candidate causes with `expected_shifts`
- [ ] Verify: pass rates match, structural differences detectable by evidence API
- [ ] Verify: negative control scenario flips the ranking

#### Phase 3: Cause ranking (`moirai/diagnose/`)

**`moirai/diagnose/causes.py`** — schema and loader:

```python
@dataclass
class CandidateCause:
    id: str
    type: str                               # prompt, tool, model, runtime, config
    description: str
    expected_shifts: dict[str, str]         # feature → "increase" | "decrease"
    prior: float = 0.0                      # 0 = uniform (auto-assigned before scoring)
```

**`moirai/diagnose/ranking.py`** — evidence-weighted ranking:

```python
@dataclass
class CauseScore:
    cause: CandidateCause
    score: float                    # normalized score (0-1)
    ci_lower: float                 # bootstrap CI
    ci_upper: float
    matched_features: list[str]     # which features matched
    evidence_strength: float        # raw pre-normalization score

@dataclass
class DiagnosisResult:
    cause_scores: list[CauseScore]  # sorted by score descending
    unknown_score: float            # residual mass for unmodeled causes
    feature_shifts: list[FeatureShift]  # the evidence used
    task_breakdown: list[TaskBreakdown]
    baseline_pass_rate: float
    current_pass_rate: float
    stable_across_seeds: bool | None  # if multi-seed, were rankings stable?

def score_causes(
    comparison: VariantComparison,
    causes: list[CandidateCause],
) -> DiagnosisResult:
    """Rank candidate causes by evidence-weighted scoring.

    For each cause:
    1. Iterate expected_shifts: direction match → +|shift|, mismatch → -|shift|×0.5
    2. Unclaimed features → unknown bucket
    3. Apply priors (auto-uniform if 0.0), exp(raw), normalize to simplex
    """

def bootstrap_confidence(
    baseline_runs: list[Run],
    current_runs: list[Run],
    causes: list[CandidateCause],
    n_bootstrap: int = 200,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """Bootstrap confidence intervals on cause scores.

    Resamples runs with replacement, recomputes evidence + ranking.
    Returns (ci_lower, ci_upper) per cause.
    """
```

The scoring model (single formulation — see "Key design decisions" section 3 for full pseudocode):
1. For each cause, iterate its `expected_shifts` dictionary
2. For each expected feature: if actual shift matches expected direction, add |shift| to raw score; if opposite, subtract |shift| × 0.5
3. Features not claimed by any cause contribute to the unknown bucket
4. Apply priors (auto-uniform if prior=0.0), exponentiate raw scores, normalize to simplex

This is transparent: you can trace exactly why each cause got its score.

- [ ] `CandidateCause` schema and JSON loader in `moirai/diagnose/causes.py`
- [ ] `score_causes()` in `moirai/diagnose/ranking.py`
- [ ] `bootstrap_confidence()` for rank stability
- [ ] `DiagnosisResult` in `schema.py`
- [ ] Tests: primary scenario ranks C1 first, negative control ranks C3 first
- [ ] Test: unknown cause dominates when no candidate explains the shifts

#### Phase 4: CLI integration

**`moirai evidence` command:**
```
moirai evidence <path> --baseline variant=baseline --current variant=current [--json]
```

Extracts and displays behavioral feature shifts + task breakdown. JSON mode for piping.

**`moirai diagnose` command:**
```
moirai diagnose <path> \
    --baseline variant=baseline \
    --current variant=current \
    --causes causes.json \
    [--json] [--bootstrap N]
```

Full pipeline: evidence → ranking → output. Uses existing `apply_kv_filters()` for variant splitting (no schema changes needed).

- [ ] `evidence` CLI command in `cli.py`
- [ ] `diagnose` CLI command in `cli.py`
- [ ] Terminal output formatting in `viz/terminal.py`
- [ ] JSON output mode (`--json` flag)

#### Phase 5: End-to-end demo

```bash
#!/bin/bash
# examples/diagnosis_demo/run_demo.sh

# Generate both scenarios
python examples/diagnosis_demo/generate.py prompt_regression runs/prompt/ --seed 42
python examples/diagnosis_demo/generate.py timeout_regression runs/timeout/ --seed 42

# Scenario 1: prompt regression (invisible to pass rate)
echo "=== SCENARIO 1: Prompt Regression ==="
moirai diff runs/prompt/ --a variant=baseline --b variant=current
moirai diagnose runs/prompt/ \
    --baseline variant=baseline --current variant=current \
    --causes examples/diagnosis_demo/causes.json --bootstrap 200

# Scenario 2: timeout regression (negative control)
echo "=== SCENARIO 2: Timeout Regression (negative control) ==="
moirai diagnose runs/timeout/ \
    --baseline variant=baseline --current variant=current \
    --causes examples/diagnosis_demo/causes.json --bootstrap 200

# Robustness: repeat scenario 1 with different seeds
for seed in 42 123 456; do
    python examples/diagnosis_demo/generate.py prompt_regression runs/seed_$seed/ --seed $seed
    moirai diagnose runs/seed_$seed/ \
        --baseline variant=baseline --current variant=current \
        --causes examples/diagnosis_demo/causes.json --json \
        | python -c "import sys,json; d=json.load(sys.stdin); print(f'seed {$seed}: top cause = {d[\"cause_scores\"][0][\"cause_id\"]}')"
done
```

- [ ] `run_demo.sh` script
- [ ] Verify demo output matches expected narrative
- [ ] Verify negative control correctly ranks C3 first
- [ ] Verify ranking stable across 3 seeds
- [ ] `README.md` walkthrough

## Acceptance criteria

### Functional requirements

- [ ] `moirai evidence` produces structured variant comparison with behavioral features
- [ ] `moirai diagnose` produces ranked causes with scores and confidence intervals
- [ ] Synthetic demo runs end-to-end: same pass rate, different trajectory structure
- [ ] Primary scenario correctly ranks prompt change (C1) first
- [ ] Negative control correctly ranks timeout change (C3) first
- [ ] Unknown-cause bucket dominates when no candidate explains the shifts
- [ ] Rankings stable across 3 random seeds
- [ ] JSON output mode for programmatic consumption

### Quality gates

- [ ] Tests for evidence extraction (behavioral features, variant comparison)
- [ ] Tests for cause ranking (scoring, bootstrap, unknown bucket)
- [ ] Tests for synthetic data generator
- [ ] Demo script runs without errors
- [ ] Existing tests still pass (no schema regressions — no schema changes needed)

## Dependencies and risks

**Dependencies:**
- numpy, scipy (already in use)
- No new external dependencies

**Risks:**
- **Vocabulary drift**: if enriched step names change (e.g., `test` → `run_tests`), canonical features break. Mitigation: features are computed from step type/name patterns, not exact string matches.
- **Double-counting**: branch shifts and motif shifts can reflect the same behavioral change. Mitigation: we deduplicate at the canonical feature level — each feature contributes once.
- **Small sample instability**: 10 runs per task per variant is the minimum for Fisher's power. Bootstrap CI captures this uncertainty honestly.
- **Synthetic data overfitting**: the generator could produce data that only works with our specific feature extractor. Mitigation: build the evidence API first, then the generator.

## What this plan does NOT include

- Experiment selection (deferred — proving diagnosis works comes first)
- Persistent diagnosis state (single-shot MVP)
- Schema changes to `Run` (existing tags suffice)
- HTML visualization (CLI only)
- Automatic cause detection from git diffs
- Multi-cause interaction detection
- KL divergence cluster machinery (features are simpler and more robust)

## File inventory

### New files
| File | Purpose |
|------|---------|
| `moirai/analyze/evidence.py` | Feature extraction + `compare_variants()` |
| `moirai/diagnose/__init__.py` | Package init |
| `moirai/diagnose/causes.py` | `CandidateCause` schema + JSON loader |
| `moirai/diagnose/ranking.py` | `score_causes()` + `bootstrap_confidence()` |
| `examples/diagnosis_demo/generate.py` | Synthetic data generator |
| `examples/diagnosis_demo/causes.json` | Demo candidate causes |
| `examples/diagnosis_demo/run_demo.sh` | End-to-end demo script |
| `examples/diagnosis_demo/README.md` | Demo walkthrough |
| `tests/test_evidence.py` | Evidence extraction tests |
| `tests/test_ranking.py` | Cause ranking tests |
| `tests/test_generate_demo.py` | Generator tests |

### Modified files
| File | Change |
|------|--------|
| `moirai/schema.py` | Add `FeatureShift`, `TaskBreakdown`, `VariantComparison`, `CauseScore`, `DiagnosisResult` |
| `moirai/analyze/stats.py` | Add `cohens_h()`, `proportion_delta_ci()` |
| `moirai/cli.py` | Add `diagnose` and `evidence` commands |
| `moirai/viz/terminal.py` | Add diagnosis + evidence output formatting |

## Implementation order

1. **Phase 1** (evidence API) — defines the output contract. Build and test against hand-crafted fixtures.
2. **Phase 2** (generator) — targets Phase 1's actual outputs. Produces demo data.
3. **Phase 3** (ranking) — consumes evidence, tested against both scenarios.
4. **Phase 4** (CLI) — wires everything together.
5. **Phase 5** (demo) — validates end-to-end.

Phase 1 has no dependencies. Phase 2 requires Phase 1. Phases 3-4 require Phase 2. Phase 5 requires Phase 4.

## Self-critique (revised)

1. **Is the scoring model honest?** Yes. It's called "evidence-weighted ranking," not "Bayesian inference." The math is transparent: match features, sum shifts, normalize. No pretense of calibrated likelihoods.

2. **Does double-counting break things?** No. By extracting canonical features (each computed once), we avoid counting the same behavioral shift through multiple evidence types.

3. **Can the model distinguish confounded causes?** Only if they affect different features. If two causes both list `TEST_AFTER_EDIT_RATE`, the model can't separate them — it reports uncertainty (split score). This is correct behavior, not a bug.

4. **What fails with noisy data?** Feature extraction with <5 runs. Wald CIs blow up (can exceed [0,1]), bootstrap intervals are huge. The model is honest about this (wide CI = low confidence).

5. **Is the unknown-cause bucket real?** Yes. If observed feature shifts don't match any declared cause's expected features, the unknown bucket absorbs the mass. The model says "I don't know" when appropriate.

6. **Where does the negative control matter?** It proves the ranking isn't hardwired. Without it, a skeptic could claim the model always picks the first cause. The timeout scenario shows the model responds to actual evidence, not cause ordering.

## References

### Internal
- `moirai/analyze/compare.py:10` — existing `compare_cohorts()`, pattern to follow
- `moirai/analyze/divergence.py:13` — `find_divergence_points()`, produces branch evidence
- `moirai/analyze/motifs.py:39` — `find_motifs()`, produces pattern evidence
- `moirai/analyze/cluster.py:11` — `cluster_runs()`, produces cluster distribution
- `moirai/analyze/stats.py` — statistical primitives (Fisher's, BH, permutation FDR)
- `moirai/filters.py:51` — `apply_kv_filters()`, handles variant splitting via tags
- `moirai/cli.py:312` — `diff` command pattern for `--a`/`--b` K=V filters
