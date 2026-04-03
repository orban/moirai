---
title: "Intervention experiment — breaking test-fail loops"
type: experiment
date: 2026-04-02
origin: "Cross-dataset analysis showing test-fail loops as #1 failure predictor (52% prevalence, 11pp gap)"
---

# Intervention experiment

## Hypothesis

Injecting a "break test-fail loops" rule into an agent's system prompt will improve pass rates on coding tasks. Three rule variants are tested:

- **Rule R (read):** "After 3 consecutive test failures without an intervening edit, re-read the source file you last edited before making another change."
- **Rule S (search):** "After 3 consecutive test failures without an intervening edit, search the codebase for the function or error mentioned in the test output before making another change."
- **Rule T (think):** "After 3 consecutive test failures without an intervening edit, explain in your reasoning what you believe the root cause is before making another change."

## Phase 1: Simulation

Scan all 2000 swe_smith traces. For each failing run:

1. Find the first point where 3+ consecutive test-type steps occur without an intervening edit/write step
2. Record what the agent did next
3. For each rule, determine whether the prescribed action would have exposed the information needed to fix the bug

### Detection heuristic

A "test-fail loop" is 3+ consecutive steps where:
- Step type is "tool" and step name contains "test" or step is a judge with test_result
- No intervening step has type "tool" with name containing "edit", "write", or "str_replace"

### Output

Per rule:
- `triggered`: failing runs that hit 3+ consecutive test failures
- `natural_compliance`: of those, runs where the agent already took the prescribed action next
- `intervention_candidates`: triggered minus natural compliance
- `estimated_uplift`: candidates where content analysis shows the prescribed action would have provided useful information

## Phase 2: Real experiment

### Task selection

Pick 5 tasks where simulation shows highest intervention potential.

### Variants

4 system prompt variants per task:
- **Baseline:** original prompt
- **+R:** baseline + read rule
- **+S:** baseline + search rule
- **+T:** baseline + think rule

### Runs

5 tasks × 4 variants × 3 runs = 60 agent invocations.
Estimated cost: ~2M tokens over 2-3 nights via nightshift.

### Measurement

- Pass rate per variant (primary metric)
- Compliance rate: did the agent follow the rule when it triggered?
- Effectiveness rate: when the agent followed the rule, did it change strategy?

### Fallback

If prompt compliance < 50%, build a programmatic guardrail that injects the instruction into the next user message after detecting 3 consecutive test failures. Rerun the non-compliant subset.

## Analysis plan

Report as the concluding section of the blog post:
1. Simulation estimate: "rule S would have helped X% of failing runs"
2. Real result: "pass rates improved from Y% to Z%"
3. If applicable: "the agent ignored the rule Q% of the time, so we enforced it programmatically"
