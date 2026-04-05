---
title: "feat: behavioral features pipeline"
type: feat
date: 2026-04-05
status: deepened
---

# feat: behavioral features pipeline

## Enhancement summary

**Deepened on:** 2026-04-05
**Research:** swarm reference implementations at /Volumes/mnemosyne/moirai/swarm/ (23 scripts)

### Corrections from reference implementations
1. **edit_dispersion** uses `unique_files / total_edits`, NOT `1 - HHI` as originally planned
2. **Tie handling** in median split: values `<= median` go to LOW group, `> median` to HIGH — not "exclude ties"
3. **uncertainty_density** does NOT exist in swarm scripts — it was computed from content.py's regex, not discovered by the swarm
4. **symmetry_index** uses ideal position scoring (explore@0.2, modify@0.5, verify@0.85), not transition counting

## Overview

Add `moirai/analyze/features.py` and a `moirai features` CLI command that computes behavioral features per run, runs within-task natural experiments, validates with split-half, and outputs a ranked table + JSON. This makes moirai's headline findings reproducible from a single command.

## Problem statement

The 4 robust features (test_position_centroid +6.8pp, symmetry_index +6.1pp, etc.) were discovered by 23 ad hoc swarm scripts on an SSD, not by moirai. Nobody can clone the repo and reproduce the findings. The tool can't demonstrate its own value proposition.

## Proposed solution

### Files touched

| File | Change |
|---|---|
| `moirai/schema.py` | Add `FeatureSpec`, `FeatureResult`, `ExperimentResult` |
| `moirai/analyze/features.py` | **NEW** — feature registry, computation, experiments, pipeline |
| `moirai/analyze/content.py` | Rename `_UNCERTAINTY_RE` → `UNCERTAINTY_RE` (make public) |
| `moirai/analyze/stats.py` | Add `sign_test()` |
| `moirai/cli.py` | Add `features` command |
| `moirai/viz/terminal.py` | Add `print_features()` |
| `tests/test_features.py` | **NEW** — unit + known-answer tests |
| `tests/test_stats.py` | Add sign_test tests |

### `moirai/schema.py` additions

```python
from typing import Literal

@dataclass(frozen=True)
class FeatureSpec:
    """Definition of a behavioral feature in the registry."""
    name: str
    compute: Callable[[Run], float | None]
    direction: Literal["positive", "negative"]
    description: str

class ExperimentResult(NamedTuple):
    """Result of a within-task median-split natural experiment."""
    delta_pp: float           # pass_mean - fail_mean in percentage points
    p_value: float | None     # sign test p-value (exact binomial)
    n_tasks: int              # tasks contributing (had enough data for median split)

@dataclass
class FeatureResult:
    """Ranked result for one behavioral feature."""
    name: str
    description: str
    direction: Literal["positive", "negative"]
    delta_pp: float           # percentage point difference (pass_mean - fail_mean)
    p_value: float | None     # sign test p-value (uncorrected)
    q_value: float | None     # BH-adjusted p-value
    split_half: bool          # True if significant (p<0.05) on both random halves
    pass_mean: float
    fail_mean: float
    n_tasks: int              # tasks contributing (varies per feature — some runs lack tests/reasoning)
    n_runs: int               # total runs with non-None feature values
```

**Dropped from v0:** `FeatureReport` wrapper dataclass. `rank_features()` returns `list[FeatureResult]`. The CLI builds the JSON envelope with dataset metadata.

### `moirai/analyze/features.py` structure

```python
from __future__ import annotations
from typing import Literal
from collections.abc import Callable
from moirai.schema import FeatureSpec, FeatureResult, Run
from moirai.analyze.content import compute_test_centroid, select_task_groups, UNCERTAINTY_RE
from moirai.compress import step_enriched_name, phase_sequence

# ── Feature registry ───────────────────────────────────────────────

FEATURES: list[FeatureSpec] = [
    FeatureSpec("test_position_centroid", _test_position_centroid, "positive",
                "Where in the trajectory testing is concentrated"),
    FeatureSpec("symmetry_index", _symmetry_index, "positive",
                "How well the trajectory follows explore→modify→verify"),
    FeatureSpec("uncertainty_density", _uncertainty_density, "negative",
                "Hedging language density in reasoning text"),
    FeatureSpec("reasoning_hypothesis_formation", _hypothesis_formation, "negative",
                "Explicit hypothesis formation in reasoning"),
    FeatureSpec("edit_dispersion", _edit_dispersion, "positive",
                "Whether edits spread across files (higher = more spread)"),
]


# ── Feature functions ──────────────────────────────────────────────
# Each: (Run) -> float | None. Returns None when data is insufficient.

def _test_position_centroid(run: Run) -> float | None:
    """Normalized center of mass of test step positions (0-1).
    Delegates to content.compute_test_centroid."""

def _symmetry_index(run: Run) -> float | None:
    """Mean ideal-position score across classifiable steps.
    Each explore/modify/verify step scored as 1 - |actual_pos - ideal_pos|.
    Ideal: explore@0.2, modify@0.5, verify@0.85. Returns None if < 4 classifiable."""
    # Uses phase_sequence() from compress.py, filters to explore/modify/verify

def _uncertainty_density(run: Run) -> float | None:
    """Fraction of reasoning steps matching UNCERTAINTY_RE.
    Returns None if run has no reasoning steps."""
    # Uses UNCERTAINTY_RE from content.py (now public)

def _hypothesis_formation(run: Run) -> float | None:
    """Fraction of reasoning steps with explicit hypotheses.
    Matches: "I think", "the issue is", "hypothesis", "probably because"."""
    # Returns None if run has no reasoning steps

def _edit_dispersion(run: Run) -> float | None:
    """unique_files / total_edits. 1.0 = every edit hits a different file.
    Returns None if run has < 2 edit steps."""
    # Extracts file paths from step.attrs["file_path"] on edit steps (not write)


# ── Natural experiment ─────────────────────────────────────────────

def within_task_experiment(
    task_runs: dict[str, list[Run]],
    feature_fn: Callable[[Run], float | None],
    min_group_size: int = 3,
) -> ExperimentResult:
    """Within-task median-split natural experiment.

    For each task:
      1. Compute feature value per run
      2. Split at median: HIGH = value > median, LOW = value <= median
         (ties go to LOW, matching swarm reference)
      3. Compare pass rates: HIGH group vs LOW group
      4. Record the delta for this task

    Across tasks: sign test (exact binomial) on the per-task deltas.
    Skip tasks where either group after split has < min_group_size runs.
    """


# ── Main pipeline ──────────────────────────────────────────────────

def rank_features(
    runs: list[Run],
    min_runs: int = 10,
    seed: int = 42,
) -> list[FeatureResult]:
    """Full pipeline: compute features, run natural experiments, validate, rank.

    1. select_task_groups(runs, min_runs) to get mixed-outcome tasks
    2. For each feature in FEATURES:
       a. within_task_experiment() → delta_pp, p_value, n_tasks
       b. Inline split-half: split task keys 50/50 (using seed),
          run within_task_experiment on each half,
          significant = both halves p < 0.05
    3. BH-correct p-values across all features
    4. Sort by |delta_pp| descending
    5. Return list[FeatureResult]
    """
```

### `moirai/analyze/stats.py` addition

```python
def sign_test(deltas: list[float]) -> float | None:
    """Two-sided sign test on a list of values.

    Counts positives and negatives (zeros/ties EXCLUDED).
    Returns p-value from binomial test under H0: P(positive) = 0.5.
    Returns None if fewer than 5 non-zero values.
    """
```

### CLI command (`moirai/cli.py`)

```python
@app.command()
def features(
    path: Path = typer.Argument(..., help="Data directory"),
    min_runs: int = typer.Option(10, help="Min runs per task"),
    output: Path | None = typer.Option(None, help="JSON output path"),
    seed: int = typer.Option(42, help="Random seed for split-half"),
    strict: bool = typer.Option(False, help="Strict validation"),
) -> None:
    """Compute behavioral features and rank by predictive power."""
    runs = _load_and_filter(path, strict=strict)

    from moirai.analyze.features import rank_features
    results = rank_features(runs, min_runs=min_runs, seed=seed)

    from moirai.viz.terminal import print_features
    print_features(results, runs)

    if output:
        _write_features_json(results, runs, path, output)
```

JSON envelope built by CLI (not the analysis layer):

```json
{
  "dataset": "examples/swe_rebench",
  "n_runs": 12854,
  "n_tasks_mixed": 1096,
  "n_tasks_total": 5800,
  "features": [ ... ]
}
```

### Terminal output (`print_features`)

```
Behavioral features — 1,096 mixed-outcome tasks, 12,854 runs

  Feature                          Delta    p-value   q-value   Split-half
  ──────────────────────────────   ───────  ────────  ────────  ──────────
  test_position_centroid           +6.8pp   <0.001    <0.001    ✓
  uncertainty_density              -6.5pp   <0.001    <0.001    —
  symmetry_index                   +6.1pp   <0.001    <0.001    ✓
  reasoning_hypothesis_formation   -5.7pp   <0.001    <0.001    ✓
  edit_dispersion                  +4.8pp   <0.001    <0.001    ✓
```

## Reference implementations (from swarm scripts)

### test_position_centroid
**Source:** `swarm/temporal_features_batch3.py:32-40`
- Normalize each test step position: `i / (n-1)` where n = total steps
- Return mean of normalized positions
- Returns None if < 5 steps or < 2 test steps
- TEST_NAMES: step names starting with "test("

### symmetry_index
**Source:** `swarm/interaction_structural_batch2.py:122-144`
- Classify each step into phase (explore/modify/verify)
- Ideal positions: explore@0.2, modify@0.5, verify@0.85
- Score per step: `1.0 - abs(actual_position - ideal_position)`
- Return mean score across all classifiable steps
- Returns None if < 6 steps or < 4 classifiable steps

### uncertainty_density
**Source:** `moirai/analyze/content.py` (NOT in swarm scripts)
- Uses `UNCERTAINTY_RE` regex: maybe, might, possibly, actually-comma, let me try/step back/reconsider, another approach
- Count matches across all reasoning text
- Divide by number of reasoning steps
- Returns None if no reasoning steps

### reasoning_hypothesis_formation
**Source:** `swarm/reasoning_features_batch1.py:76-85`
- 10 regex patterns: "i think the/this/it", "the cause is/might be", "this fails because", "the issue/problem/bug is", "my hypothesis/theory/guess", "i suspect/believe/hypothesize", "this is likely/probably", "the reason is/for", "what's happening is", "this suggests/indicates/means"
- Count total matches across all reasoning text (lowercased)
- Divide by number of reasoning steps
- Returns None if no reasoning steps

### edit_dispersion
**Source:** `swarm/edit_features_batch1.py:59-66`
- Count distinct file paths across 'edit' steps (NOT 'write' steps)
- Return `unique_files / total_edit_steps`
- File paths from `step.attrs.get('file_path', '')`
- Returns None if < 2 edit steps
- **Note:** Higher = more spread (positive direction). This is NOT 1-HHI.

### Median-split methodology
**Source:** `swarm/evaluate_feature.py:56-130`
- For each mixed-outcome task with >= 6 runs:
  - Compute median: `sorted(vals)[len(vals)//2]`
  - HIGH group: `value > median` (strict inequality)
  - LOW group: `value <= median` (ties go to LOW)
  - Compute pass rate delta: `high_rate - low_rate`
- Sign test across tasks: count positive vs negative deltas (exclude zeros)
- Exact binomial test: P(X >= n_pos) under H0: p=0.5, two-sided

## Technical considerations

- **Reuse `select_task_groups`** from content.py for task grouping — don't reimplement.
- **Reuse `compute_test_centroid`** from content.py — delegates via `_test_position_centroid`.
- **Rename `_UNCERTAINTY_RE` → `UNCERTAINTY_RE`** in content.py (make public, it's a shared constant).
- **Reuse `phase_sequence`** from compress.py for symmetry_index.
- **Median-split tie handling**: values `<= median` go to LOW group, `> median` to HIGH (matching swarm implementation). Skip tasks where either group has < `min_group_size` runs.
- **Split-half inlined** in `rank_features()` — 10 lines, not a standalone function. Splits task keys with `random.Random(seed).shuffle()`, runs `within_task_experiment` on each half, checks both p < 0.05.
- **BH across 5 features** barely adjusts anything (noted by all reviewers). Include it for correctness but don't oversell.
- **Performance**: 5 features × 12,854 runs is ~77K feature computations, each O(steps). Fast. Loading 12K JSON files is the bottleneck (pre-existing, not introduced here).
- **Commit pending content.py changes first** (compute_test_centroid, UNCERTAINTY_RE rename) before building features.py.

## Acceptance criteria

- [ ] `moirai features /Volumes/mnemosyne/moirai/swe_rebench_v2 --min-runs 10` produces the ranked table
- [ ] `--output results.json` writes valid JSON matching the envelope schema
- [ ] The 4 split-half survivors are marked `split_half: true`
- [ ] Effect sizes match validated numbers within rounding (test_position_centroid +6.8pp, etc.)
- [ ] Known-answer tests: hand-computable dataset where delta_pp is verified
- [ ] Edge case tests: all-pass tasks, all-fail tasks, runs with no tests, runs with no reasoning
- [ ] Each feature function tested individually with constructed runs

## Implementation phases

### Phase 1: Building blocks (45 min)
- `FeatureSpec`, `FeatureResult`, `ExperimentResult` in schema.py
- `sign_test()` in stats.py with tests (including tie handling)
- Rename `_UNCERTAINTY_RE` → `UNCERTAINTY_RE` in content.py
- 6 feature functions in features.py with unit tests for each
- `_HYPOTHESIS_RE` regex in features.py (private to this module)

### Phase 2: Pipeline (30 min)
- `within_task_experiment()` with `ExperimentResult` return type
- `rank_features()` with inline split-half, BH correction, sorting
- Known-answer integration test with constructed data
- Edge case tests

### Phase 3: CLI + output (20 min)
- `features` command in cli.py (follows existing pattern)
- `print_features()` in terminal.py
- `_write_features_json()` helper for JSON envelope
- Integration test with sample data

## Dependencies and risks

- **Data dependency**: full validation requires 12,854 runs on SSD. Tests use constructed data.
- **Number validation**: feature values MUST match documented results. Mismatch means a bug in either the library or the original swarm scripts — investigate, don't paper over.
- **content.py is modified (unstaged)**: commit pending changes before starting Phase 1.

## Review feedback incorporated

From architecture, Python quality, and simplicity reviews (2026-04-05):
- ✓ `FeatureSpec` frozen dataclass replaces 4-tuple registry
- ✓ `Literal["positive", "negative"]` for direction field
- ✓ `ExperimentResult` NamedTuple replaces bare tuple return
- ✓ `FeatureReport` dropped — CLI builds JSON envelope
- ✓ Split-half inlined in `rank_features()`, not standalone function
- ✓ `_UNCERTAINTY_RE` made public
- ✓ Reuse `select_task_groups` from content.py
- ✓ Collapsed to 3 phases, Phase 5 (blog integration) deferred
- ✓ Tie handling specified: ties go to LOW group (matching swarm reference), min_group_size threshold
- ✓ `--strict` CLI option added
- ✓ `split_half` is `bool` not `bool | None` — pipeline always computes it

## Review round 2 fixes

From deepened plan review (2026-04-05):
- ✓ Dropped `@dataclass` from `ExperimentResult` — use `NamedTuple` only
- ✓ Fixed tie handling: docstring + review feedback now say "ties go to LOW" (matching swarm)
- ✓ Fixed `symmetry_index` docstring: ideal-position scoring, not transition counting
- ✓ Fixed sign test: exact binomial throughout, removed z-test approximation
- ✓ Added `Callable` import and `step_enriched_name` import
- Note: `compute_test_centroid` uses `i/n` but swarm uses `i/(n-1)` — fix during implementation to match swarm
- Note: Add min step/test count guards to `_test_position_centroid` wrapper
- Note: Compile hypothesis regex as single `_HYPOTHESIS_RE` at module level
- Note: Extract ideal phase positions to `_IDEAL_PHASE_POSITIONS` constant
- Note: Use `step_enriched_name` for all step classification (swarm was using enriched names implicitly)
- Consider: rename `edit_dispersion` to `edit_dispersion` (current name implies opposite of what higher values mean)
- ✓ Dropped `trajectory_arc_score` (fails split-half, redundant with symmetry_index) — 5 features
- ✓ Renamed `edit_concentration` → `edit_dispersion` (higher = more spread, name now matches semantics)

## References

- Brainstorm: `docs/brainstorms/2026-04-05-features-pipeline-brainstorm.md`
- Experiment results: `docs/analysis/2026-04-04-complete-experiment-results.md`
- Swarm feature scripts: `/Volumes/mnemosyne/moirai/swarm/` (23 scripts, reference implementations)
- Feature selection report: `/Volumes/mnemosyne/moirai/swarm/FEATURE_SELECTION_REPORT.md`
