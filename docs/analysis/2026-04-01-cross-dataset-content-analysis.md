---
title: "Cross-dataset content analysis — failure patterns across 5 datasets, 3983 runs"
date: 2026-04-01
datasets:
  - eval_harness (483 runs, 5 tasks analyzed)
  - swe_smith (1000 runs, 4 clusters analyzed)
  - swe_agent (500 runs, 3 tasks analyzed)
  - real_data scenario1 (983 runs, 3 tasks analyzed)
  - real_data scenario2 (1500 runs, 3 tasks analyzed)
method: moirai explain with LLM analysis via claude sonnet
total_findings: ~50 across 18 analysis units
---

# Cross-dataset content analysis

## Overview

Content-aware analysis was run across 5 datasets (3,983 total runs). 18 tasks/clusters were analyzed with LLM-assisted differential diagnosis. The structural layer (alignment + divergence detection) identified where runs diverge; the LLM compared the actual step content (reasoning, tool inputs, results) to explain why.

## Five universal failure patterns

These patterns appear across all datasets regardless of agent framework, task type, or evaluation harness.

### 1. Premature narrowing

**What it is:** Agents fixate on the first plausible file or hypothesis without building enough context about the codebase structure.

| Dataset | Task | Evidence |
|---|---|---|
| eval_harness | ansible-85516 | Failing runs searched specific files (`task_executor.py`) instead of globally, missing `display.py:956` |
| swe_agent | geopandas-3240 | Failing runs locked onto `tools/sjoin.py` without discovering the `geodataframe.py` wrapper layer |
| swe_agent | cognitive_complexity-15 | Failing runs searched for prose "cognitive complexity" instead of function name `snippet_compexity` |
| scenario1 | ansible-84999 | Failing run skipped glob searches at col 3, entered read loop on prematurely chosen files |
| scenario1 | ansible-85487 | Failing run fixated on `apt_repository.py` instead of `deb822_repository.py` — confused two related modules |

**The mechanism:** Narrow search commands (`search_file` scoped to one file) consistently underperform broad ones (`search_dir` across the repo) at early decision points. Agents that read before searching commit to files before knowing which ones matter.

### 2. Test-fail loops without strategy change

**What it is:** Agents hit test failures and retry without changing their approach — interpreting "fail" as "try again" instead of "reconsider."

| Dataset | Cluster/Task | Evidence |
|---|---|---|
| swe_smith | C1 (25% pass) | Re-run tests after errors without writing code. `test(fail)` at cols 32, 36: 0% success |
| swe_smith | C6 (38% pass) | Stuck at cols 52, 67, 87 — all 0% success. Passing runs interleave read/search between failures |
| swe_smith | C2 (55% pass) | Conan retries `str_replace` with slightly different strings until budget exhaustion |
| swe_smith | C8 (50% pass) | sqlparse cycles through 7+ rewrites without reading source to understand why each fails |

**The mechanism:** Failing agents don't update their internal model when evidence contradicts their approach. Passing agents treat each failure as information — they read source, search for context, reason about root cause, then make a targeted edit.

### 3. Circular validation

**What it is:** Agents write custom tests that validate their (broken) implementation, creating false confidence.

| Dataset | Task | Evidence |
|---|---|---|
| swe_smith | C8 scrapy | Adjusted test assertions to match broken output — "tests pass" because tests were bent to match the bug |
| scenario2 | sepal_ui-347 | Custom scripts exited 0 without hitting the real code path. Official pytest caught the actual bug |
| scenario2 | lexicon-336 | Validated with `pytest -k 'create'` — wrong test action, the bug is in `list` |
| scenario1 | ansible-85487 | Verified with `ast.parse` (syntax check only) instead of running pytest |

**The mechanism:** Agents that write their own tests tend to build tests that confirm their implementation rather than the specification. Using the project's actual test suite is the strongest predictor of correct fixes.

### 4. Errors observed but not acted on

**What it is:** The agent surfaces error information (tracebacks, "No replacement was performed", error_observation signals) but doesn't use it to change strategy.

| Dataset | Task | Evidence |
|---|---|---|
| swe_smith | C2 | Conan's `str_replace` silently fails repeatedly — agent never re-reads the file to get the exact string |
| swe_smith | C6 | chardet surfaces `system:error_observation` at col 18 but doesn't pivot |
| swe_smith | C1 | Error observations interpreted as "retry" not "change approach" |
| scenario2 | sepal_ui-347 | Passing run's script *errored* with `AttributeError` revealing the real bug. Failing runs' scripts exited 0 |

**The mechanism:** Errors are information. Agents that treat error output as diagnostic signal (reading tracebacks, identifying the failing line) outperform agents that treat errors as noise to retry past.

### 5. Re-implementation over discovery

**What it is:** Agents build from scratch instead of finding existing infrastructure in the codebase.

| Dataset | Task | Evidence |
|---|---|---|
| swe_agent | cognitive_complexity-15 | `tests/conftest.py` has the canonical 4-line implementation. Failing runs reimplemented from scratch |
| swe_agent | pypistats-41 | Failing runs added input validation guards instead of fixing the date arithmetic |
| scenario2 | sepal_ui-646 | Passing runs used git stash as workflow discipline. Failing runs had no implementation/verify separation |
| swe_smith | C8 | sqlparse built standalone diagnostic scripts instead of reading library internals |

**The mechanism:** Searching for existing implementations, test fixtures, and helper functions before writing new code consistently leads to better outcomes. The codebase already contains the answer — the agent's job is to find it, not reinvent it.

## The meta-pattern

All five failure patterns share a common root:

**Failing agents optimize for action. Passing agents optimize for understanding.**

Failing agents move fast — they read one file, form a hypothesis, write a fix, run tests, retry. Passing agents move slower initially — they search broadly, read multiple files, understand the call chain — then make one targeted, correct edit.

The structural data confirms this. Across all datasets, the consensus trajectory is dominated by `verify(fail)` loops. The runs that deviate from consensus by inserting `search`, `read`, or `reason` steps between failures are the ones that succeed. The concordance finding (τ=-0.554 in swe_smith cluster 2) quantifies this: the typical pattern is the failure mode.

## Recommendations for agent developers

### Immediate interventions

1. **Force strategy switch after 3+ consecutive test failures.** Insert a mandatory read/search step before allowing another edit. This addresses patterns 2 and 4.

2. **Always validate against the project's real test suite.** Never rely on custom scripts for final verification. This addresses pattern 3.

3. **Use broad search before narrow read.** `search_dir` before `search_file`. `grep` across the repo before reading specific files. This addresses patterns 1 and 5.

### Architectural interventions

4. **Add an exploration budget.** Reserve the first N% of the step budget for broad codebase exploration before allowing any edits. The data shows that early exploration quality is the strongest predictor of success.

5. **Treat errors as structured data.** Parse tracebacks, extract file paths and line numbers, use them to redirect exploration. Don't retry past errors — investigate them.

6. **Implement edit verification.** After `str_replace` or similar operations, re-read the file to confirm the change was applied. Silent failures waste budget.

## Concordance validation

The content-aware analysis validated the concordance scoring hypothesis across all datasets:

- **Structural analysis** correctly identified that structure alone doesn't predict outcomes (|τ| < 0.1 for most clusters)
- **Content analysis** explained *why*: the tool-call sequence (read, search, edit, test) is the same for passing and failing runs, but the *content* (which file, which search query, what the test returned) is where the signal lives
- **Negative concordance** (τ=-0.554 in swe_smith C2) was confirmed: the consensus trajectory IS the failure mode, and deviants succeed by breaking the retry loop with targeted search/read steps
