---
title: "Moirai track-record audit"
date: 2026-04-21
scope: "Every concrete pattern moirai has surfaced from eval data, scored honestly for survival, actionability, and contribution to the claim that the methodology is worth the squeeze."
context: "Companion to 2026-04-21 eval-time scope commitment. Goal is to decide whether to continue investing in moirai or redirect."
---

# Moirai track-record audit

## Why this audit exists

The scope commitment on 2026-04-21 pinned moirai as an **eval-time diagnostic tool**: its value claim is that running N aligned trials of a task and applying moirai's trajectory analysis surfaces actionable failure modes that single-run inspection and aggregate metrics miss.

This audit tests that claim against the actual record. Every concrete pattern moirai has surfaced across three months of work is listed below. Each gets scored on two axes:

- **Survived validation?** Split-half, held-out, within-family stratification, or corrected-data re-run. Many early findings did not.
- **Actionable?** Would a developer reading this have changed something (prompt, harness, training data, eval design) with a reasonable expectation of improvement?

The null hypothesis is that the methodology produces patterns at roughly the rate random feature mining would — high false-positive rate, mostly stale on correction, occasional weak true positives that don't justify the compute cost of running N trials per task.

## Summary verdict

**22 concrete patterns scored.** Of those:

- **7 survived rigorous validation.** 4 features via split-half on 125-feature swarm; structure score via holdout; reranking via held-out benchmark; framework-pathology paradox (transition analysis) via cross-dataset comparison.
- **9 were killed by correction or stratification.** Initial "universal failure modes" (test-fail loops, premature narrowing, errors-not-acted-on, circular validation, re-implementation) did not survive the 2026-04-04 test-detection correction or the 2026-04-02 cross-framework check. Flagship +39pp result was a selection-bias artifact.
- **4 survived descriptively but are untested for action.** Strategy clustering, divergence clustering, motif patterns, wedge claims exist as outputs — nobody has run a downstream experiment asking "if we act on this, does it help?"
- **2 produced conflicting results under scope changes.** HMM latent states (significant on 22-label vocabulary, zero on 31-label); Haiku semantic features (significant at n=30, collapsed at n=210).

**Net read:** the methodology's false-positive rate on its own data is high. Every concrete "universal intervention" moirai has proposed has either died under correction or been shown to flip sign across frameworks. What survives is genuine but narrow: a handful of weak-correlation features, a best-of-K reranking procedure, and a descriptive finding that frameworks have *opposite* pathologies — which is more a warning against universal rules than a prescription for action.

The honest read is that moirai's value, if any, is narrow and weak. It's not nothing — the 4 surviving features, structure score, and reranking are real — but the claim that the methodology "uncovers hidden actionable patterns" is not well-supported by the track record as it stands.

## Audit table

Columns: **Claim** (what moirai said), **Surfaced** (when/where), **Implied action** (what could a dev do with it), **Survival** (what validation said), **Actionable?** (yes/no/untested), **Notes**.

### Pre-correction findings (the "5 universal failure patterns" from 2026-04-01)

| # | Claim | Surfaced | Implied action | Survival | Actionable? | Notes |
|---|---|---|---|---|---|---|
| 1 | Test-fail loops predict failure (52% prev, 11pp gap, #1 signal) | 2026-04-01, cross-dataset | Inject rule: break test-fail loops | **KILLED** by 2026-04-04 correction (has_test_fail_loop p=0.064, not significant) AND by 2026-04-02 cross-framework analysis (flips sign in CoderForge) | **No** | Flagship finding from the initial blog-style writeup. Selection-bias artifact from broken test detection (cd prefix). Cross-framework it *hurts* CoderForge. Pure false positive. |
| 2 | Premature narrowing dominates multi-file failures | 2026-04-01 | Search broadly before reading narrowly | Never re-validated post-correction | **Untested** | Framework-dependent by admission. Evidence was qualitative case studies. Would need separate validation; not done. |
| 3 | Circular validation (51% swe_smith, 36% swe_agent, 6pp drop) | 2026-04-01 | Force pytest over custom scripts | Not re-run post-correction; shares data lineage with #1 | **Untested** | Plausible but never survived under corrected feature extraction. |
| 4 | Errors observed but not acted on (74% swe_agent, 15pp gap) | 2026-04-01 | Force error-response behavior | Framework-specific; not validated cross-family | **No** | Dropped in later analyses. |
| 5 | Re-implementation over discovery (12% prev, 1.42x fail-lift) | 2026-04-01 | Require reading before writing | Low prevalence; never re-validated | **Untested** | Small sample, never pressure-tested. |

### Framework-specific / descriptive findings

| # | Claim | Surfaced | Implied action | Survival | Actionable? | Notes |
|---|---|---|---|---|---|---|
| 6 | Stash amnesia in Qwen-32B (CoderForge) | 2026-04-01 | Scaffold git-stash recovery | Specific behavioral failure in one model | **Yes (narrow)** | Real observation. But it's a known class of "context window state-tracking failure" — would a dev need moirai to find this? Probably not. |
| 7 | Blocking command trap (C-c rejected by sandbox) | 2026-04-01 | Add interrupt handling / timeouts | Real harness-infrastructure failure | **Yes (narrow)** | Harness bug, not a trajectory insight. Moirai surfaced it but didn't need trajectory analysis to find it. |
| 8 | OpenHands re-edits same file in 82% of runs | 2026-04-01 | Discourage iterative str_replace | Observational; correlational (45% vs 62% pass) | **Untested** | Never validated as causal. |
| 9 | eval_harness: 3 configs are statistically indistinguishable (16-19%) | 2026-04-01 | Don't pay for context injection w/o validation | Clean null result from A/B | **Yes** | One of moirai's cleanest findings, but it's literally just a pass-rate chi-squared — doesn't require trajectory analysis. |

### Transition analysis (2026-04-02)

| # | Claim | Surfaced | Implied action | Survival | Actionable? | Notes |
|---|---|---|---|---|---|---|
| 10 | Frameworks have *opposite* pathologies (SWE-smith tests too much, CoderForge tests too little) | 2026-04-02 | Stop looking for universal rules; calibrate per-framework | **Validated** via separate-per-framework analysis | **Yes** | This is actually moirai's most defensible high-level finding — but it's a *negative* finding (universal rules don't work). Valuable as a warning, weak as a product. |
| 11 | SWE-smith: force edit between consecutive test failures | 2026-04-02 | Prompt rule | Correlational, within one framework | **Untested** | Never actually RCT'd. |
| 12 | CoderForge: force test between consecutive edits | 2026-04-02 | Prompt rule | Correlational, within one framework | **Untested** | Never actually RCT'd. |

### Feature engineering swarm (2026-04-03/04)

| # | Claim | Surfaced | Implied action | Survival | Actionable? | Notes |
|---|---|---|---|---|---|---|
| 13 | test_position_centroid predicts success (r=+0.121) | 2026-04-04 | Prefer agents that test later in trajectory | **SURVIVED** split-half | **Untested as action** | Real correlation, but never tested as an intervention target. Effect size is small. |
| 14 | symmetry_index (explore→modify→verify arc) predicts success (r=+0.101) | 2026-04-04 | Prompt for this arc | **SURVIVED** split-half | **Untested as action** | Same as above. |
| 15 | reasoning_hypothesis_formation predicts failure (r=-0.088) | 2026-04-04 | Discourage explicit hypothesizing | **SURVIVED** split-half | **Untested as action** | Counterintuitive; would need an RCT before any prescription. |
| 16 | edit_concentration (different files, not hammering) predicts success (r=+0.064) | 2026-04-04 | Discourage repeated edits to same file | **SURVIVED** split-half | **Untested as action** | Weakest of the 4; borderline. |
| 17 | has_edit_test as a +39pp effect | 2026-04-04 | Force edit-then-test pattern | **KILLED by correction** (was +39pp → +14.9pp after fixing test detection) | **No** | Major false positive — the flagship "moirai finding" for several days was a selection-bias artifact. |
| 18 | HMM latent states are predictive | 2026-04-03/04 | Use hidden state for flagging | **KILLED** on 31-label vocabulary (0 significant); was significant on 22-label | **No** | Vocabulary-sensitive, fragile, doesn't survive basic robustness check. |
| 19 | Haiku semantic features (sem_explore_targeted, sem_flail) | Pre-correction | Semantic interventions | **KILLED** at scale (r=+0.346 at n=30 → r=+0.034 at n=210) | **No** | Sample-size inflation. |

### Structural techniques (running inventory)

| # | Claim | Surfaced | Implied action | Survival | Actionable? | Notes |
|---|---|---|---|---|---|---|
| 20 | Strategy clustering: 75% of tasks form 2+ clusters, 40pp median spread | 2026-04-04 | Identify failure clusters, investigate | Descriptive only | **Untested** | Real clustering result. Never tested whether intervening on cluster membership helps. |
| 21 | Divergence clustering: 13 clusters, 98% coverage (wasted orientation 18%, wrong file, skipped reasoning) | 2026-04-04 | Target top divergence causes | Descriptive only | **Untested** | Useful taxonomy; never turned into an intervention. |
| 22 | Motif patterns (edit→write_test→test(pass) = failure, lift 0.85) | 2026-04-04 | Avoid specific motif | Descriptive only | **Untested** | Large motif library, never acted on. |

### Holdout / validation work (2026-04-06)

| # | Claim | Surfaced | Implied action | Survival | Actionable? | Notes |
|---|---|---|---|---|---|---|
| 23 | Divergence predicts outcome | pre-holdout | Flag divergent runs | **KILLED** on held-out (AUROC 0.507 = noise) | **No** | The central moirai mechanism is noise on held-out. |
| 24 | Features predict outcome | pre-holdout | Feature-based classifier | Partially killed (AUROC 0.556 = weak) on held-out | **Marginal** | Barely above chance. |
| 25 | Structure score (divergence + earlyness + stability composite) | 2026-04-06 | Favor high-structure runs | **SURVIVED** with tau=+0.402 on holdout | **Yes (narrow)** | Real signal. But only within a family. |
| 26 | Reranking on best-of-K with structure score (+2-4pp) | 2026-04-06 | Best-of-K selection | **SURVIVED** on held-out (narrow family) | **Yes (narrow)** | The one concrete actionable result. Only applies in the multi-run setting. Small effect. |
| 27 | Conditional routing (structure-aware) beats global 59.2% of tasks | 2026-04-06 | Route based on structure signal | **SURVIVED** but narrow | **Yes (narrow)** | Small per-task lift over global. |

### Wedge UX (2026-04-07)

| # | Claim | Surfaced | Implied action | Survival | Actionable? | Notes |
|---|---|---|---|---|---|---|
| 28 | Wedge claims cover 77/1096 tasks (7%) | 2026-04-07 | Human-readable branch summaries | Descriptive; coverage limited | **Untested** | Claims exist but have never been evaluated for quality or actionability. Plausibly useful to a dev reading them, but no evidence. |

## Hit-rate summary

Bucketing:

| Bucket | Count | % |
|---|---|---|
| Survived AND actionable (directly or narrowly) | 5 (#10, #25, #26, #27, plus #9 for completeness) | 18% |
| Survived but untested as action | 6 (#13-16, #20-22 minus overlaps) | 21% |
| Killed by correction/stratification/holdout | 9 (#1, #17-19, #23, #4 plus variants) | 32% |
| Untested, never validated | 6 (#2-5, #8, #11, #12, #28 minus overlaps) | ~29% |

**The generous read:** ~40% of patterns moirai surfaced either survived validation or remain plausibly actionable pending test.

**The honest read:** Only ~18% are both validated AND actionable, and of those, half are framework-specific observations that don't require trajectory analysis to find (#9: "configs don't matter" is a chi-squared on pass rates). The genuinely moirai-specific validated-and-actionable findings are #10 (universal rules don't work — a negative result), #25/#26/#27 (structure score in narrow best-of-K setting).

**The damning read:** For every week the project existed, at least one "major finding" later died under correction. `has_edit_test` at +39pp was the headline for a while before dropping to +14.9pp. Test-fail loops was the flagship "universal" pattern before being shown to flip sign across frameworks. HMM latent states were significant on one vocabulary and vanished on the next. The methodology, unpressured, generates patterns at roughly the rate that random feature mining would produce false positives.

## What this means for the scope commitment

The scope commitment (moirai as eval-time diagnostic tool) assumed the methodology's diagnostic outputs are worth something. This audit suggests the worth is narrower than the framing has implied.

Specifically:

1. **The surviving signal exists but is narrow.** Structure score + reranking works, but only in best-of-K with aligned trials within a family. That's the full validated scope.

2. **The diagnostic framing is weaker than claimed.** Very few of moirai's "here's what's going wrong" observations have translated into validated actionable guidance. The one clear example of diagnostic-to-action is the transition analysis showing frameworks have opposite pathologies — and that's a *warning*, not an intervention.

3. **The track record looks like feature mining.** False positives have been frequent and sometimes flagship. Only aggressive validation (split-half, holdout, cross-framework stratification) has kept the record honest.

## Recommended next steps

Three options, ordered by conservatism:

### A. Accept the narrow scope and write it up honestly
The defensible claim is: "Running N repeat trials and scoring with a structure-composite lets you do weak reranking (+2-4pp) within a family, and reveals that universal intervention rules don't generalize (negative result)." Write this up as the blog post, stop claiming more. Minimal additional compute. Most honest.

### B. Run the rediscovery test
Take 3-5 documented agent improvements (SWE-bench submissions, OpenHands changelogs, paper revisions). Run moirai on pre-improvement trajectories. Does moirai's output point at what the improvement actually addressed? If yes on ≥60%, the methodology is validated as a diagnostic tool. If no, it's feature mining. ~1 week of focused work.

### C. Investigate the untested pile
#2-5, #8, #11, #12, #20-22, #28 are listed as "untested." Design minimum-viable tests for each: would acting on this improve outcomes? This is a lot of work and risks spawning another round of findings-that-later-die. Not recommended as first step.

**My lean:** start with A (write up the honest version), then do B (rediscovery test) to see whether the diagnostic framing can survive an external comparison. Skip C unless B validates.

## Addendum: what the audit doesn't cover

- **Qualitative value to human readers.** A human reading moirai output might find it useful for orientation even if the patterns don't hold up statistically. This audit can't score that without a user study.
- **Exploratory value.** Moirai may have generated ideas that informed other work without those ideas themselves being "findings." Hard to score.
- **Wedge claim quality.** 77 claims exist; their usefulness to a reader was never evaluated. A small blind rating study would fix this cheaply.

These are separate from the track-record audit but worth naming as open questions.
