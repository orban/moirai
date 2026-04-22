---
date: 2026-04-21
topic: eval-time-scope-commitment
---

# Eval-time scope commitment

## What we're committing to

Moirai is an **eval-time diagnostic tool**, not a production debugging tool. Its users are eval designers, researchers, and anyone running repeat-trial benchmarks. The data shape moirai operates on is N aligned trials of the same task. The value prop is: *"Before you ship, run repeat trials of your eval set; moirai shows failure modes that single-run inspection and aggregate metrics miss."*

Everything downstream of this — claims, experiments, blog framing, product copy — should assume this scope. Production-monitoring pitches, heterogeneous-log framing, real-time triggers, and Raindrop comparisons are all out of scope until further notice.

## Why this approach, and what was rejected

Three scopes were considered in the 2026-04-21 brainstorm. Only one is on the table:

### (1) Own the eval-time setting — *chosen*
Moirai's distinctive methods (sequence alignment, divergence detection, structure score, reranking across branches) all require N aligned runs of the same task. That's an eval-time data shape. Nobody else occupies the "divergence-point analysis of repeat trials" niche. The methodology produces real-but-weak signal on this data (reranking +2–4pp, structure score tau=+0.402, 4 features survive split-half). Committing to this scope is the honest product: the claims are defensible, the users exist, the competitive position is clear.

### (2) Rebuild for production heterogeneous logs — *rejected*
Production teams have M trajectories of M different invocations with no natural alignment. Moirai's alignment-based core doesn't run on that data. The parts of moirai that would transfer (per-trajectory features, LLM summaries, clustering) are not distinctive versus existing production tools like Raindrop. Going this route means competing on a weaker footing without a clear technical edge. Deferred, not erased — if the eval-time scope doesn't pan out, this is a fallback direction, but it's a rebuild.

### (3) Pivot to RL/HMM/ORM framing — *rejected for now*
Treating the underlying object as "P(success | partial trajectory, hidden state)" and building an online risk model is intellectually clean and aligns with the 2026-04-06 trajectory-learning brainstorm. But it's a bigger rewrite and moves away from the alignment-based machinery that's already built. Keep on the list as a later option if eval-time scope validation fails.

## Key decisions

- **Users are eval designers and researchers, not production ops.** Blog, docs, and experiments should reflect that framing.
- **The input assumption is N aligned trials per task.** Stop pitching moirai as useful on one-off logs.
- **Real-time intervention / runtime triggers are out of scope.** The previous brainstorm arc (Codex's RCT, partial-structure-score triggers) was pushing toward a production use case we're not claiming.
- **"Make sense of 100K heterogeneous logs" is not what moirai does.** That's Raindrop's job.
- **Diagnostic value, not predictive value, is the claim.** Moirai's output is patterns and failure modes that inform downstream changes (prompt, harness, training data, eval design). Not a runtime signal.

## Open questions for the next planning phase

- **Is the track record actually good enough to support the claim?** Pre-audit signal is mixed: test-fail loops (false positive via Simpson's paradox), HMM bash trap (confounded), structure score (partial positive, weak), wedge claims (untested), 4 surviving features (unvalidated for actionability). The track-record audit comes next to answer this honestly.
- **What's the strongest eval-time value claim we can defend?** Candidates: (a) "moirai surfaces failure modes single-run inspection misses," (b) "moirai analysis of repeat trials is more sample-efficient than running more trials with aggregate metrics," (c) "moirai rediscovers published agent improvements from pre-improvement trajectories." Audit + rediscovery test will filter these.
- **What does a null result look like, and how would we recognize it?** If the audit shows more false positives than true ones, or the hit rate of moirai-surfaced patterns leading to actionable fixes is under ~30%, the honest conclusion is that even this narrow scope doesn't support the methodology. That's a real outcome worth preparing for.

## Next steps

1. **Track-record audit** (today, 1–2 hours). Exhaustive scoring of every concrete pattern moirai has surfaced. Output: `docs/analysis/2026-04-21-moirai-track-record-audit.md`.
2. **If audit looks credible:** rediscovery test on 3–5 documented agent improvements. Check whether moirai-on-pre-change trajectories points at them.
3. **If audit looks bad:** write up the null result honestly and reconsider whether scope (3) — RL/HMM pivot — deserves the investment.

Planning handoff: none for now. This is a scope commitment, not a feature build. The next concrete artifact is the audit itself, not an implementation plan.
