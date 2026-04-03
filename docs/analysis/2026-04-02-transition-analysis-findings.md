---
title: "Transition analysis — framework-specific failure signatures"
date: 2026-04-02
datasets: [swe_smith (995 editors), coderforge (494 editors)]
method: bigram transition analysis on enriched step names, controlled for edit presence
---

# The intervention paradox: opposite failure modes across frameworks

## The problem with universal rules

We spent a day trying to find a universal intervention that improves agent pass rates. We tried:

1. **Structural features** (search scope, read targets, test-fail gap counts) — didn't predict outcomes
2. **LLM semantic classification** (Haiku labeling steps as ORIENT/DEAD_END/FIX_ATTEMPT) — collapsed under scrutiny. FIX_ATTEMPT was 100% redundant with "has edit step." DEAD_END was partially data leakage.
3. **Universal rules** ("break test-fail loops," "read after 3 failures") — the simulation showed these wouldn't help because the structural patterns they target don't causally drive outcomes

Each approach failed for the same reason: we assumed there was one abstraction that works across all frameworks. There isn't.

## What actually works: transition bigrams, controlled for confounds

When we stopped looking for universal features and instead:
- Controlled for the obvious confound (only analyzed runs that actually made edits)
- Looked at **transitions** between step types (bigrams) instead of step counts
- Analyzed **per framework** instead of pooling

The signal appeared immediately.

## The mirror finding

### SWE-smith: fails by testing without editing

995 runs that made at least one edit. 746 pass (75%), 249 fail (25%).

| Transition | Pass/run | Fail/run | Delta |
|---|---|---|---|
| test(fail) → test(fail) | 4.63 | **8.52** | -3.89 |
| test(fail) → write | 2.01 | 2.72 | -0.71 |
| write → test(fail) | 1.73 | 2.39 | -0.66 |
| test(fail) → test(pass) | **0.35** | 0.21 | +0.14 |

Failing SWE-smith runs average **8.5 consecutive test-fail transitions** per run vs 4.6 for passing runs. They test, see failure, test again, see failure again — without changing the code between attempts. The transition `test(fail) → test(pass)` (breaking through) happens 67% more often in passing runs.

**The SWE-smith intervention:** Force an edit between consecutive test failures.

### CoderForge: fails by editing without testing

494 runs that made at least one edit. 316 pass (64%), 178 fail (36%).

| Transition | Pass/run | Fail/run | Delta |
|---|---|---|---|
| bash(other) → bash(other) | 2.38 | **4.17** | -1.80 |
| read(source) → read(source) | 2.84 | **3.94** | -1.11 |
| bash(python) → bash(other) | 0.76 | **1.86** | -1.10 |
| test → test | **2.72** | 1.81 | +0.92 |
| edit(source) → edit(source) | 1.33 | **2.04** | -0.72 |
| bash(python) → test | **2.51** | 1.88 | +0.63 |

Failing CoderForge runs get stuck in bash and read loops — exploring without converging. They make consecutive edits without testing between them (`edit → edit` is fail-correlated). Meanwhile, `test → test` and `bash(python) → test` are **pass-correlated** — more testing is good.

**The CoderForge intervention:** Force a test run between consecutive edits.

### The paradox

| Rule | Effect on SWE-smith | Effect on CoderForge |
|---|---|---|
| "Break test-fail loops" | Helps (addresses #1 signal) | **Hurts** (testing more is good) |
| "Test between edits" | Neutral | Helps (addresses #1 signal) |
| "Edit between tests" | Helps (addresses #1 signal) | **Hurts** (editing more is bad) |

A universal "break test-fail loops" rule would improve SWE-smith and degrade CoderForge. The frameworks have **opposite pathologies**: one tests too much, the other tests too little. The intervention must be calibrated to the framework's specific imbalance.

## Why this wasn't visible before

Three levels of analysis failed to find this:

**Level 1: Step type counts.** "52% of swe_smith runs have test-fail loops" was true but not actionable — we couldn't tell if the loops caused failure or just correlated with harder tasks.

**Level 2: Controlled feature analysis.** When we controlled for edit presence, the step-level features (test count, search count, read count) didn't separate pass from fail. Counts don't capture temporal structure.

**Level 3: LLM semantic classification.** Haiku's labels were either redundant with structural features (FIX_ATTEMPT = has edit) or leaked from the outcome (DEAD_END). The semantic layer didn't add predictive power.

**What worked: transition bigrams.** The signal isn't in how many tests you run — it's in what you do *between* tests. That's a temporal pattern, not a count. And it's framework-specific, not universal.

## Methodology notes

### Controlling for edit presence

All analysis is restricted to runs that made at least one edit. This removes the trivial confound (runs with zero edits always fail) and forces the analysis to find what separates good editors from bad editors.

### Why bigrams and not longer n-grams

We tested bigrams (2-step transitions). Trigrams would give more context but require exponentially more data for the same statistical power. With ~500-1000 runs per dataset, bigrams are the right granularity — enough runs per bigram type to detect differences, short enough to be interpretable as "what the agent does next."

### Cross-framework validity

The finding that frameworks have opposite pathologies was discovered by analyzing them separately, not by pooling. Pooling across frameworks would have canceled the signals (SWE-smith's fail-correlated `test→test` vs CoderForge's pass-correlated `test→test`). This is a form of Simpson's paradox in agent evaluation.

## Implications

### For agent developers

Don't apply generic "best practices" to agent behavior. Measure your specific framework's transition imbalance:
- If `test→test` is fail-correlated: your agent tests too much without editing. Add a forced edit after N consecutive test failures.
- If `edit→edit` is fail-correlated: your agent edits too much without testing. Add a forced test after each edit.
- If `bash→bash` is fail-correlated: your agent gets stuck in exploration loops. Add a convergence check.

### For moirai

The right analysis layer is transition bigrams on enriched step names, computed per-framework and controlled for edit presence. This should be a built-in analysis mode — `moirai transitions <path>` that outputs the framework-specific imbalance table.

### For the blog

The narrative arc is:
1. We found 5 failure patterns across 21,466 traces (descriptive)
2. We tried to turn them into universal interventions (failed)
3. We discovered the interventions are framework-specific mirror images (the real finding)
4. The right abstraction is transition bigrams, not step counts or LLM classifications (methodological contribution)
