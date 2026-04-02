---
title: "Analysis taxonomy — four layers of agent trace analysis"
type: feat
date: 2026-04-02
origin: "Systematic validation across 21,466 traces showing which abstractions predict outcomes and which don't"
---

# Analysis taxonomy

## The problem

moirai has accumulated analysis features without a coherent framework for what each layer does, when to use it, and what it can and can't tell you. Structural alignment was the original foundation, content-aware LLM analysis was added to fill its gaps, and reasoning metrics were bolted on after discovering uncertainty density predicts outcomes. Transition bigrams — the most actionable finding — exist only in throwaway scripts.

This spec formalizes the four validated layers, documents what we tried and discarded, and adds the missing transition analysis to the tool.

## The four layers

### Layer 1: Structural alignment

**What it measures:** Sequence similarity of tool-call types/names across runs. Needleman-Wunsch alignment, agglomerative clustering, divergence detection at high-entropy columns.

**What it predicts:** Nothing about outcomes directly. Concordance scoring (τ between structural typicality and outcome) shows |τ| < 0.1 for most clusters across all datasets.

**What it's good for:** Organizing runs into comparable groups. Identifying WHERE runs diverge (which alignment column). Providing scaffolding for layers 2-4.

**Status:** Fully implemented. `align_runs()`, `cluster_runs()`, `compute_concordance()`, `find_divergence_points()`.

### Layer 2: Transition analysis

**What it measures:** Bigram transition frequencies on enriched step names, computed per-run and compared between passing and failing runs. Controlled for edit presence (only analyzes runs that made at least one edit).

**What it predicts:** Framework-specific failure signatures. SWE-smith fails by `test(fail) → test(fail)` looping (delta -3.89 per run). CoderForge fails by `bash(other) → bash(other)` looping (delta -1.80) and `edit(source) → edit(source)` without testing (delta -0.72). These are mirror images — the same rule helps one framework and hurts the other.

**What it's good for:** Diagnosing a specific framework's behavioral imbalance. Prescribing framework-specific interventions ("your agent tests too much without editing" vs "your agent edits too much without testing").

**Status:** Validated in ad hoc scripts. Not yet in moirai. This spec adds it.

### Layer 3: Reasoning metrics

**What it measures:** Structural features of the agent's reasoning text, extracted via regex. No LLM needed.

Three metrics:
- **Uncertainty density:** Hedging words ("maybe", "let me try", "attempt", "might", "perhaps", "not sure") per reasoning step. Higher in failing runs across 3 of 4 frameworks (10-13pp pass rate gap, Cohen's d = -0.34 to -0.73).
- **Diagnosis density:** Explicit diagnosis phrases ("the issue is", "the problem is", "root cause", "fails because") per reasoning step. Weak signal, inconsistent direction.
- **Code reference density:** Mentions of specific code elements (`.py`, `line N`, `def name`, `class name`) per reasoning step. Task-specific signal.

**What it predicts:** Uncertainty density is the strongest cross-framework predictor found. Agents that hedge more in their reasoning fail more. The dose-response is monotonic: 0-0.1 uncertainty → 47% pass, 0.5-1.0 → 24% pass (pooled across 1,631 runs with reasoning).

**What it's good for:** Flagging runs/clusters where the agent is guessing rather than diagnosing. Suggesting reasoning-level interventions ("state your hypothesis before acting").

**Status:** Implemented in `compute_reasoning_metrics()`. Displayed in `print_explanation()`. Regex patterns need tests.

### Layer 4: Content analysis

**What it measures:** LLM-assisted comparison of actual step content (reasoning text, tool inputs, tool results) at structurally divergent positions. Uses Sonnet via `claude -p`.

**What it predicts:** Task-specific failure causes. "Failing runs searched `task_executor.py` instead of globally for `[ERROR]`." "Failing runs used `search_file` instead of `search_dir`." "Failing runs adjusted test assertions to match broken output."

**What it's good for:** Producing specific, accurate, human-readable diagnoses of why runs fail on a particular task. The most accurate layer but the most expensive (~$0.05 per task group, 60-120s latency).

**Status:** Implemented in `build_prompt()`, `invoke_llm()`, `parse_response()`.

## What we tried and discarded

| Approach | Why it failed |
|---|---|
| Step type counts (search count, read count) | Doesn't predict outcomes even after controlling for confounds |
| Fine-grained structural features (search scope, read target) | Too crude — regex extraction misses the semantic distinction |
| LLM semantic step classification (DEAD_END, FIX_ATTEMPT) | FIX_ATTEMPT was 100% redundant with "has edit step." DEAD_END was outcome-leaked when Haiku saw pass/fail. Outcome-blind classification collapsed the signal. |
| Convergence curves (AUC, half-point, plateau fraction) | Near zero signal on CoderForge, mixed on swe_agent |
| Information flow (search result → next read) | File path matching too crude, most runs showed zero connected steps |
| Universal intervention rules ("break test-fail loops") | Simulation showed structural proxies don't predict outcomes. And the rule helps SWE-smith but hurts CoderForge (mirror paradox). |

## What this spec adds

### `compute_transition_bigrams()` in content.py

```python
def compute_transition_bigrams(
    runs: list[Run],
    min_count: int = 10,
) -> list[TransitionSignal]:
    """Compute transition bigram signals between passing and failing runs.

    Only analyzes runs with at least one edit/write step (controls for
    the trivial confound that runs without edits always fail).

    Returns signals sorted by absolute delta descending.
    """
```

Implementation:
1. Filter to runs with `result.success is not None` and at least one edit/write step
2. Split into pass/fail groups
3. For each run, extract enriched step name sequence using `step_enriched_name()` from compress.py, filtering None values
4. Compute bigram counts per group
5. Normalize to per-run rates (count / n_runs_in_group)
6. Compute delta (pass_rate - fail_rate) and direction
7. Filter by min_count, sort by absolute delta

### `TransitionSignal` dataclass in schema.py

```python
@dataclass
class TransitionSignal:
    """A transition bigram that differs between passing and failing runs."""
    from_step: str
    to_step: str
    pass_rate: float        # avg occurrences per passing run
    fail_rate: float        # avg occurrences per failing run
    delta: float            # pass_rate - fail_rate
    total_count: int
    direction: str          # "pass_correlated" or "fail_correlated"
```

Place after `ReasoningMetrics` in schema.py.

### Changes to ExplanationReport

Add field: `transitions: list[TransitionSignal] = field(default_factory=list)`

### Changes to run_explain

After computing reasoning metrics, call `compute_transition_bigrams(known)` and attach the top-10 results (by absolute delta) to the report.

### Changes to print_explanation

Display top-5 transition signals:

```
Transition signals
  test(fail) → test(fail)    pass: 4.6/run  fail: 8.5/run  Δ=-3.9 ■ fail
  bash(other) → bash(other)  pass: 2.4/run  fail: 4.2/run  Δ=-1.8 ■ fail
  test → test                pass: 2.7/run  fail: 1.8/run  Δ=+0.9 ■ pass
```

Red for fail-correlated, green for pass-correlated. Only show if transitions are non-empty.

### Tests

**TestReasoningMetrics** (6 tests):
- test_uncertainty_detection
- test_causal_detection
- test_diagnosis_detection
- test_code_ref_detection
- test_empty_reasoning_returns_none
- test_per_step_normalization

**TestTransitionBigrams** (5 tests):
- test_basic_bigrams
- test_controls_for_edit_presence
- test_minimum_count_filter
- test_sorted_by_delta
- test_direction_label

### Cleanup

- Remove `KNOWN_FINDING_CATEGORIES` from schema.py (unused after normalization was removed)
- Remove unused `seed` parameter from `_cluster_based_groups`

## What this spec does NOT add

- No new CLI commands. Transition analysis is part of `moirai explain`, not a separate command.
- No LLM classification layer. Haiku semantic classification didn't survive validation.
- No universal intervention rules. Interventions are framework-specific and outside moirai's scope.
- No convergence curves or information flow metrics. These didn't predict outcomes.
