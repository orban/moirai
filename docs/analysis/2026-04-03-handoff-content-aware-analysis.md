---
title: "Handoff: content-aware analysis — findings, validated methods, and next steps"
date: 2026-04-03
branch: feat/content-aware-explain
runs_analyzed: 21466
datasets: 11
frameworks: 7
---

# Handoff: content-aware analysis

## What was built

### `moirai explain` command
Replaces the old single-run explain with a cross-run content-aware analysis pipeline:
- Groups runs by task_id (falls back to structural clustering for single-run-per-task datasets)
- Aligns within groups, finds divergent columns
- Optionally pipes step content through claude/codex for LLM-assisted differential diagnosis
- `--mode auto|claude|codex|structural` controls LLM usage
- `--run <id>` preserves the old single-run behavior

### Four analysis layers
1. **Structural alignment** (existing): step type/name alignment, divergence detection, concordance scoring
2. **Transition bigrams** (new): per-run normalized bigram rates on enriched step names, controlled for edit presence, top-5 by absolute delta
3. **Reasoning metrics** (new): uncertainty/diagnosis/code-ref density extracted from reasoning text via regex
4. **Content analysis** (existing, enhanced): LLM comparison at divergent alignment positions

### New dataclasses
- `ContentFinding`, `ExplanationReport` (content layer output)
- `ReasoningMetrics` (per-group reasoning quality)
- `TransitionSignal` (bigram pass/fail rate differences)

### Tests
308 total (282 existing + 17 content + 5 reasoning + 4 transition)

## What we found across 21,466 trajectories

### Five qualitative failure patterns (from LLM content analysis)
1. **Premature narrowing** — fixating on first plausible file without broad exploration
2. **Test-fail loops without strategy change** — retrying without new information
3. **Circular validation** — writing custom tests that match broken implementation
4. **Errors observed but not acted on** — seeing tracebacks but not changing strategy
5. **Re-implementation over discovery** — building from scratch instead of finding existing code

These are consistent across frameworks and confirmed by LLM analysis on 18 tasks/clusters. Full details in `docs/analysis/2026-04-01-cross-dataset-content-analysis.md`.

### The intervention paradox (from transition bigrams)
Framework-specific failure signatures are mirror images:
- SWE-smith: `test(fail)→test(fail)` is fail-correlated (delta -3.89). Agents test too much without editing.
- CoderForge: `test→test` is pass-correlated (delta +0.92). Agents edit too much without testing.

A universal "break test-fail loops" rule helps one and hurts the other. Details in `docs/analysis/2026-04-02-transition-analysis-findings.md`.

### Reasoning uncertainty predicts outcomes (from reasoning metrics)
Uncertainty density (hedging words per reasoning step) predicts failure across 3 of 4 frameworks:

| Framework | Low uncertainty | High uncertainty | Delta |
|---|---|---|---|
| CoderForge | 69% pass | 56% pass | +12pp |
| SWE-agent | 15% pass | 5% pass | +10pp |
| OpenHands | 55% pass | 41% pass | +13pp |

**Caveat:** Confounded with task difficulty. Within-task validation needed.

### Haiku semantic classification: partially validated
Tested outcome-blind Haiku classification on 3 tasks (93 runs total):
- **FIX_ATTEMPT is NOT redundant with "has edit"** on tasks where both pass/fail runs edit. The COUNT of fix attempts matters, but direction is task-specific (more attempts = thrashing on easy tasks, persistence on hard tasks).
- **DEAD_END is noise outcome-blind.** The signal came from leakage when outcome was provided.
- Per-step semantic labels add real signal beyond structural features, but the signal is task-conditional, not universal.

### The meta-finding
Every abstraction layer we tested is conditional:
- Transition bigrams: conditional on framework
- FIX_ATTEMPT count: conditional on task difficulty
- Uncertainty density: conditional on task difficulty
- Content diagnosis: conditional on specific task

No universal predictor of agent outcomes was found from trace data alone.

## What we tried and discarded

| Approach | Why it failed |
|---|---|
| Step type counts | Doesn't predict outcomes after controlling for confounds |
| Fine-grained structural features (search scope) | Regex extraction too crude for semantic distinction |
| LLM semantic step classification (DEAD_END) | Outcome-leaked or redundant with structural features |
| Convergence curves (AUC, half-point) | Near zero signal across datasets |
| Information flow (search result → next read) | File path matching too crude |
| Universal intervention rules | Simulation showed structural proxies don't cause outcomes. Mirror paradox. |
| Trajectory length as predictor | Confounded by task difficulty. Within-task effect vanishes. |

## The reframe: progress prediction

Codex proposed (and we agree): the right question isn't "what trace feature predicts pass/fail?" but **"is this agent making progress toward solving the problem right now?"**

Latent progress states:
- **Oriented**: agent understands codebase structure, narrowing toward bug
- **Mislocalized**: agent is searching/editing the wrong part of the codebase
- **Productive iteration**: agent is making edits that move tests closer to passing
- **Thrashing**: agent is retrying without new information
- **Premature convergence**: agent committed to wrong fix too early
- **Verification gap**: agent hasn't tested its changes

These states are what the content analysis already identifies qualitatively. The transition bigrams detect some (thrashing = `test→test` loops). Reasoning metrics detect others (rising uncertainty = possible thrashing or mislocalization).

The north star metric: **given a trace prefix, can we predict whether the next 5-10 steps reduce distance to resolution?**

## Next steps

### HMM strategy exploration (in progress)
Active work in `.claude/worktrees/exp+hmm-strategies/` with 100x more data. HMMs are the natural formalism — latent states with observable emissions (step types + reasoning features) and learned transition probabilities. The HMM will discover the progress states from data rather than prescribing them.

Moirai's role: provide the observation sequences (enriched step names + reasoning metrics) as HMM input. The transition bigram layer may be replaced by the HMM's learned transition matrix.

### Intervention experiment (designed, not yet run)
Spec at `docs/specs/2026-04-02-intervention-experiment-design.md`. Three system prompt variants tested against baseline on 5 tasks. Blocked on: deciding whether the intervention should target the HMM-discovered states rather than ad hoc rules.

### Immediate improvements
- Validate uncertainty density within-task (same task, multiple runs) to control for difficulty confound
- Run Haiku semantic classification at scale (all qualifying runs across all datasets, ~$3 for 24K steps) to check if task-conditional patterns aggregate into useful clusters
- Export moirai analysis as observation sequences for HMM consumption

## File index

### Specs
- `docs/specs/2026-04-01-content-aware-diagnosis-design.md` — content-aware explain command
- `docs/specs/2026-04-02-analysis-taxonomy-design.md` — four analysis layers
- `docs/specs/2026-04-02-intervention-experiment-design.md` — intervention experiment

### Analysis
- `docs/analysis/2026-04-01-swe-smith-content-analysis.md` — 4 swe_smith clusters
- `docs/analysis/2026-04-01-cross-dataset-content-analysis.md` — 5 datasets, 5 patterns
- `docs/analysis/2026-04-01-complete-findings.md` — full findings across 21K traces
- `docs/analysis/2026-04-02-transition-analysis-findings.md` — framework-specific transitions

### Plans
- `docs/plans/2026-04-01-feat-content-aware-explain-plan.md` — implementation plan

### Code
- `moirai/analyze/content.py` — content extraction, prompt building, LLM invocation, reasoning metrics, transition bigrams
- `moirai/schema.py` — ContentFinding, ExplanationReport, ReasoningMetrics, TransitionSignal
- `moirai/viz/terminal.py` — print_explanation with reasoning + transition display
- `tests/test_content.py` — 26 tests covering all new functionality

### Data
- `examples/eval_harness/` — 483 runs
- `examples/swe_smith/` — 1000 runs (+1000 in swe_smith_extra)
- `examples/swe_agent/` — 500 runs (+5000 in swe_agent_full)
- `examples/openhands/` — 500 runs (+5000 in openhands_full)
- `examples/coderforge/` — 500 runs (+5000 in coderforge_full)
- `examples/real_data_demo/` — 2483 runs (scenario1 + scenario2)
