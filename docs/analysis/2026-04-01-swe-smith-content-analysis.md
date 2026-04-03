---
title: "Content-aware analysis of swe_smith failure modes"
date: 2026-04-01
dataset: swe_smith (1000 runs, 17 qualifying clusters)
method: moirai explain with cluster-based fallback, LLM analysis via claude sonnet
clusters_analyzed: [1, 2, 6, 8]
---

# SWE-smith content-aware analysis

## Context

Concordance scoring across swe_smith showed |τ| < 0.1 for most clusters — structural similarity doesn't predict outcomes. Cluster 2 was the exception: τ=-0.554 (p=0.003), meaning the structurally typical runs *fail* and the deviants succeed.

Content-aware analysis uses the structural alignment to locate divergent positions, then asks an LLM to compare the actual step content (reasoning, tool inputs, results) between passing and failing runs.

## Three failure patterns

### 1. Test-fail loops without strategy change

The most common failure mode across all four clusters. Agents hit test failures and retry without changing their approach.

| Cluster | Pass rate | Evidence |
|---|---|---|
| C1 | 25% | Re-run tests after error observations without writing code. `test(fail)` at cols 32, 36: **0% success**. |
| C6 | 38% | Stuck in test-fail at cols 52, 67, 87 — all **0% success**. Passing runs interleave read/search/reason between failures. |
| C2 | 55% | Conan hits repeated "No replacement was performed" errors, retries `str_replace` with slightly different strings until budget exhaustion. |
| C8 | 50% | sqlparse cycles through **7+ rewrites** of `_process_case` without reading source to understand why each iteration fails. Dies at cost limit. |

### 2. Broad exploration vs targeted search early

The second pattern: failing runs read broadly or write prematurely, while passing runs search for the specific target early.

| Cluster | Pass rate | Evidence |
|---|---|---|
| C2 | 55% | Col 11: `search` → **100% success**, `read` → **0% success**. Passing runs search for the target function immediately. Failing runs read multiple files without searching. |
| C6 | 38% | Col 9: `write` → **0% success** — premature modification before understanding the bug. |
| C8 | 50% | Col 40: `read` → **100% success**, `test(fail)` → **50%**. Passing runs read source to understand the API contract before editing. |

### 3. Errors observed but not acted on

Agents surface error information but don't use it to change strategy.

| Cluster | Pass rate | Evidence |
|---|---|---|
| C1 | 25% | Error observations interpreted as "retry" not "change approach." |
| C2 | 55% | Conan's `str_replace` failures silently fail. Agent keeps retrying without diagnostic information. |
| C6 | 38% | chardet surfaces `system:error_observation` at col 18 but doesn't pivot strategy. |
| C8 | 50% | scrapy adjusts test assertions to match broken output instead of fixing the code — masking the failure. |

## The meta-insight

These aren't separate problems. They're the same failure mode at different severity levels:

**Failing agents don't update their internal model when evidence contradicts their approach.** They see test failures and errors but interpret them as "try again" rather than "reconsider." Passing agents treat failures as information — they read source, search for context, reason about root cause, then make a targeted edit.

This maps directly to the concordance finding: in cluster 2 (τ=-0.554), the structurally typical pattern is the failure mode (test-fail loop). The runs that deviate from the consensus — by inserting search/read steps where the majority retries — are the ones that succeed.

## Per-cluster detail

### Cluster 1: 8 runs, 25% pass

**Summary:** Failing runs are caught in a test-fail loop: after accumulating error observations, they issue another test call rather than switching to code modification. Passing runs recognize the same error signals as a trigger to write/modify and exit the observation phase.

**Key divergence:** At columns 32 and 36, `test(fail)` has 0% success rate while `write` actions correlate with passing outcomes.

**Findings:**
- `error_ignored` col 32: Failing runs re-run tests after error observations without making code changes
- `reasoning_gap` col 36: Failing runs still cycling through error observations four steps later
- `wrong_command` col 32: Failing runs issue test commands when the correct action is write/edit

**Consensus:** verify(fail) → explore → verify(fail) → modify → verify(fail) → explore×2 → verify(fail)×2 → modify → verify(fail) → modify → verify(fail) → modify → verify(fail)×2 → explore

### Cluster 2: 22 runs, 55% pass (τ=-0.554, p=0.003)

**Summary:** Failing runs do broad early exploration (reading multiple source files) rather than targeted searching, which delays root cause identification. Once stuck in test-fail loops, they cycle between reason and write steps without converging.

**Key divergence:** At col 11, `search` → 100% success, `read` → 0% success. At col 58, `write` → 75% success, `test(fail)` → 33%.

**Findings:**
- `wrong_file` col 11: Passing runs search for the specific target function immediately. Failing runs read files broadly.
- `reasoning_gap` col 58: Passing runs commit to a write after reasoning. Failing runs re-enter another test(fail) cycle.
- `error_ignored` col 66: Conan hits silently failing `str_replace` operations — "No replacement was performed" errors that don't change strategy.
- `reasoning_gap` col 71: Conan exits at step 97 due to cost limit from edit-failure loops.
- `wrong_command` col 51: Passing runs have already achieved test(pass) by column 51. Failing runs never achieve an intermediate passing test.

**Consensus:** explore → verify(fail)×5 → modify×2 → verify(fail) → explore → verify(fail)×11 → explore → verify(fail)×3 → explore → verify(fail)×4 → think → verify(fail)×2 → modify → verify(fail)×3 → modify → verify(fail) → explore

### Cluster 6: 16 runs, 38% pass

**Summary:** Failing runs exhibit two compounding failure modes: some write too early (column 9, 0% success) before understanding the mutation, while the rest get locked into test-fail loops at columns 52, 67, and 87 without inserting search or read steps to break the cycle.

**Key divergence:** At col 9, `write` → 0% success (premature modification). At cols 52, 67, 87: `test(fail)` → 0% in all cases.

**Findings:**
- `premature_modification` col 9: Writing before adequate exploration
- `reasoning_gap` col 67: Agent fails to change strategy after repeated test failures (0% success for test(fail) at this position)
- `error_ignored` col 18: chardet surfaces error_observation but doesn't pivot
- `reasoning_gap` col 52: Failing runs exhaust retries without exploring
- `reasoning_gap` col 87: Late-stage failure to re-read code after repeated failures is terminal

**Consensus:** explore → verify(fail) → modify → verify(fail) → modify → verify(fail)×11 → modify → verify(fail)×6 → modify → verify(fail)×7 → explore → verify(fail)×30 → modify

### Cluster 8: 50 runs, 50% pass

**Summary:** Failing runs iterate by trial-and-error without reading source code to validate their model of the bug. The sqlparse run thrashes through 7+ rewrites using standalone test scripts as a feedback loop. The scrapy run builds false confidence by adjusting test assertions to match broken output.

**Key divergence:** At col 40, `read` → 100% success, `test(fail)` → 50%. At col 16, `test(pass)` → 100% (early test validation is the strongest predictor).

**Findings:**
- `reasoning_gap` col 40: Failing runs produce standalone diagnostic scripts rather than reading source code to understand the API contract
- `error_ignored` col 40: scrapy adjusts test assertions to match broken behavior, masking the failure
- `reasoning_gap` col 40: Failing runs never synthesize a root cause — they iterate through implementations by trial-and-error until budget expires
- `missing_test` col 16: Runs that achieve test(pass) early (col 16) succeed 100% of the time

**Consensus:** explore×2 → verify(fail)×2 → think → modify → verify(fail) → modify → verify(fail)×4 → modify → verify(fail)×3 → verify(pass) → modify → verify(fail)×2 → modify → verify(fail)×9 → verify(pass)

## Actionable recommendations for agent developers

1. **Force strategy switch after 3+ consecutive test failures.** Read source code or search for the target function before retrying. This single intervention addresses the dominant failure mode in 3 of 4 clusters.

2. **Prioritize targeted search over broad file reading early in the trajectory.** In cluster 2, search at step 11 has 100% success; read has 0%. The agent should identify the specific function or file to modify before reading broadly.

3. **Treat `str_replace` failures as actionable errors.** "No replacement was performed" means the target string doesn't match. The agent should re-read the file to get the exact string rather than guessing variations.

4. **Validate fixes against real tests, not standalone scripts.** Cluster 8 shows that agents using standalone test scripts as feedback can't distinguish correct fixes from overfitted ones. Early validation against the real test suite (col 16, 100% success when passing) is the strongest predictor.

5. **Never adjust test assertions to match broken output.** The scrapy failure in cluster 8 is a cautionary tale: "tests pass" is meaningless if you changed the tests to match the bug.
