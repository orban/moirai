---
title: "Complete findings — 21,466 agent trajectories across 7 frameworks"
date: 2026-04-01
total_runs: 21466
datasets: 11
frameworks: [SWE-agent, SWE-smith, OpenHands, CoderForge/Qwen, eval_harness (flat_llm, intent_layer, none)]
method: moirai explain (structural + LLM content analysis via claude sonnet)
---

# Complete findings: what makes coding agents fail?

21,466 trajectories. 7 agent frameworks. 5 datasets. One question: why do some runs solve the task and others don't?

## The data

| Dataset | Runs | Pass rate | Framework | Model |
|---|---|---|---|---|
| eval_harness | 483 | 18% | Claude Code (3 configs) | Claude |
| swe_smith | 2,000 | 75% | SWE-agent | Claude |
| swe_agent | 5,500 | 18% | SWE-agent | Various |
| openhands | 5,500 | 49% | OpenHands (CodeAct) | Various |
| coderforge | 5,500 | 57% | CoderForge | Qwen-32B |
| scenario1 | 983 | 14% | Claude Code | Claude |
| scenario2 | 1,500 | 51% | Claude Code | Claude |

## Part 1: Five universal failure patterns

These appear across all frameworks regardless of model, harness, or task type.

### 1. Test-fail loops without strategy change

**Prevalence:** 52% of swe_smith runs. Pass rate drops 11pp when present (p<0.001).

Agents hit test failures and retry without changing their approach. The transition `test(fail) → test(fail)` occurs 4.6x per passing run but 8.5x per failing run — the single most failure-predictive signal in the data.

The mechanism: agents interpret "fail" as "try again" instead of "reconsider." Passing agents insert read/search/reason steps between failures to gather new information before the next attempt.

Evidence from content analysis:
- swe_smith C1: re-ran tests after error observations without writing code (0% success at cols 32, 36)
- swe_smith C2: Conan retried `str_replace` with slight variations until budget exhaustion
- swe_smith C8: sqlparse cycled through 7+ rewrites without reading source

### 2. Premature narrowing

**Prevalence:** Framework-dependent. Not a failure predictor in swe_smith (where the task file is usually the right starting point) but the dominant failure mode in multi-file tasks.

Agents fixate on the first plausible file or hypothesis without mapping the codebase structure. Narrow search commands (`search_file` scoped to one file) consistently underperform broad ones (`search_dir` across the repo) at early decision points.

Evidence:
- eval_harness ansible-85516: searched `task_executor.py` instead of globally, missing `display.py:956`
- swe_agent geopandas-3240: locked onto `tools/sjoin.py` without discovering `geodataframe.py` wrapper
- scenario1 ansible-85487: fixated on `apt_repository.py` instead of `deb822_repository.py`
- CoderForge sepal_ui-571: shell `find/grep` returned empty because the file needed to be created

### 3. Circular validation

**Prevalence:** 51% of swe_smith runs, 36% of swe_agent runs. Pass rate drops 6pp in swe_smith (p=0.012).

Agents write custom tests that validate their (broken) implementation, creating false confidence. The project's actual test suite catches bugs that custom scripts miss.

Evidence:
- swe_smith C8: scrapy adjusted test assertions to match broken output
- scenario2 sepal_ui-347: custom scripts exited 0 without hitting the real code path
- scenario2 lexicon-336: validated with `pytest -k 'create'` — wrong test action
- scenario1 ansible-85487: verified with `ast.parse` (syntax only) instead of pytest

### 4. Errors observed but not acted on

**Prevalence:** 74% of swe_agent runs — the dominant signal in that framework. Pass rate 6% with the pattern vs 21% without (p<0.001).

Agents surface error information (tracebacks, "No replacement was performed," error_observation signals) but don't use it to change strategy.

Evidence:
- swe_smith C2: `str_replace` silently failed repeatedly — agent never re-read the file
- swe_smith C6: chardet surfaced `system:error_observation` but didn't pivot
- scenario2 sepal_ui-347: passing run's script errored with AttributeError revealing the bug; failing runs' scripts exited 0

### 5. Re-implementation over discovery

**Prevalence:** 12% of swe_smith runs. Highest fail-lift at 1.42x (p=0.004).

Agents build from scratch instead of finding existing infrastructure. Low prevalence but high impact — the codebase already contains the answer.

Evidence:
- swe_agent cognitive_complexity-15: `tests/conftest.py` had the canonical 4-line implementation; failing runs reimplemented from scratch
- swe_agent pypistats-41: added input validation guards instead of fixing date arithmetic
- swe_smith C8: built standalone diagnostic scripts instead of reading library internals

### Dose-response

The patterns compound. Runs exhibiting 0 anti-patterns pass at 89% (swe_smith) / 21% (swe_agent). Runs with 3+ patterns pass at 56% / 4%.

## Part 2: Framework-specific failures

### CoderForge / Qwen-32B

Two patterns unique to this framework:

**Stash amnesia:** The Qwen model uses `git stash` as a workflow tool but forgets the stash exists later in the context window. It re-implements the fix from scratch instead of running `git stash pop`. This is a context-window state tracking failure specific to the 32B model.

**Blocking command trap:** OpenHands' sandbox rejects `C-c` interrupts on long-running commands (git clone, network ops), creating an unrecoverable state. The agent spends the remaining budget trying to interrupt a process that won't stop. This is a harness/infrastructure failure, not an LLM reasoning failure.

### eval_harness: agent configs don't matter

Three configurations (flat_llm, intent_layer, none) produce statistically indistinguishable pass rates (16-19%, chi² p>>0.05). 47 of 52 tasks have identical outcomes across all configs.

The `none` config (no CLAUDE.md, no context layering) is 2x faster and marginally better. On the one task where config matters (graphiti-843), CLAUDE.md reading distracts the agent from a simple fix.

## Part 3: Success patterns

What do 86-92% pass rate clusters do that 25-38% clusters don't?

### The success formula

| Metric | High-pass (≥80%) | Low-pass (≤40%) |
|---|---|---|
| Median trajectory length | 10-15 steps | 31-65 steps |
| Max consecutive test(fail) | 1.7-3.1 | 4.8-16.3 |
| First write/edit position | Step 4.6 | Step 5.6 |
| Steps from test(fail) to next edit | 2.0 | 3.0 |
| test(fail) as % of trajectory | 38% | 54% |

### Three things high-pass agents do

1. **Act quickly.** First code modification within 4-5 steps. Don't over-explore.
2. **Stay tight.** Modify-verify loops are short. When a test fails, change code within 2 steps. Don't re-run the same failing test.
3. **Finish early.** 10-15 total steps, not 30-65. Not smarter per step — each step is just more productive.

### The ending signal

| Last step type | Pass rate |
|---|---|
| test(pass) | 93% |
| edit | 80% |
| write | 78% |
| test(fail) | 61% |
| search | 43% |

### Step length vs outcome (the confound)

At the population level, failing runs are 1.26-1.47x longer. But when you control for task (same task, different runs), the effect largely vanishes. Longer trajectories correlate with failure mostly because **hard tasks produce both more failures and longer trajectories** — it's a difficulty confound, not purely a retry effect.

The "point of no return" is real though:

| Dataset | Base pass rate | Halves at step | Rate there |
|---|---|---|---|
| swe_agent | 10% | 20 | 5% |
| eval_harness | 18% | 51 | 7% |
| scenario1 | 14% | 37 | 7% |
| swe_smith | 75% | 110 | 37% |
| scenario2 | 51% | 98 | 20% |

## Part 4: The meta-pattern

All findings converge on one insight:

**Failing agents optimize for action. Passing agents optimize for understanding.**

Failing agents move fast — read one file, form a hypothesis, write a fix, run tests, retry. Passing agents move slower initially — search broadly, read multiple files, understand the call chain — then make one targeted, correct edit.

The concordance scoring (τ=-0.554 in swe_smith cluster 2) quantifies this: the structurally typical trajectory IS the failure mode. The runs that deviate from consensus by inserting search/read steps where the majority retries are the ones that succeed.

## Part 5: Actionable recommendations

### For agent developers

1. **Detect and break test-fail loops.** After 3 consecutive test failures without an intervening edit, force a strategy switch: read source, search for related code, or reason about the error. This addresses the #1 failure signal (1.8x more frequent in failures).

2. **Always validate against project tests.** Never rely on custom scripts for final verification. Use `pytest` or the project's test runner. This eliminates circular validation.

3. **Search broadly before reading narrowly.** Use repo-wide search before file-scoped search. `search_dir` before `search_file`. Build a codebase map before committing to specific files.

4. **Parse errors as structured data.** Extract file paths and line numbers from tracebacks. Use them to redirect exploration. Don't retry past errors — investigate them.

5. **Implement early termination.** Trajectories past the "point of no return" (dataset-dependent, roughly 2-3x the median length) have sharply diminished returns. Save compute by stopping early and reporting failure.

### For harness developers

6. **Add interrupt handling.** The blocking command trap (CoderForge/OpenHands) is a harness-level failure that wastes the entire remaining budget. Long-running commands need timeouts.

7. **Don't add context overhead without evidence.** The eval_harness framework comparison shows that reading CLAUDE.md adds latency without improving outcomes. Context injection should be validated empirically.

### For evaluation designers

8. **Group by cluster, not just task.** Single-run-per-task datasets (swe_smith, openhands) need cluster-based analysis to surface meaningful patterns. Task-based grouping produces nothing when n=1.

9. **Report concordance alongside pass rates.** A 75% pass rate means nothing if structural similarity doesn't predict outcomes (|τ| < 0.1). Concordance scoring tells you whether your analysis framework is capturing the real signal.

10. **Control for task difficulty.** Population-level correlations (length vs outcome, pattern prevalence vs outcome) are confounded by task difficulty. Always report within-task comparisons alongside aggregate numbers.
