---
title: "feat: trajectory biology — multi-level analysis framework"
type: feat
date: 2026-03-30
spec: docs/specs/2026-03-30-trajectory-biology-design.md
---

# Trajectory biology implementation plan

Five staged implementation adding gapped motifs, positional MI, domain architecture, entropy profiles, and a statistical foundation fix to moirai.

**Design spec:** `docs/specs/2026-03-30-trajectory-biology-design.md`

## Key decisions from spec flow review

These decisions resolve ambiguities the spec left open:

1. **BH correction scope per command.** Each command applies BH across its own test family by default. `moirai patterns` corrects across motif tests only. `moirai branch` corrects across divergence tests only. Full cross-family pooling is future work (a `moirai report` command) — it avoids the performance cliff where `moirai patterns` would need to run alignments just for statistical correction.

2. **Drop `branch --mi`.** The `branch` command groups by task_id (3-30 runs per group). MI requires N >= 500. Per-task MI can't work. MI is standalone only via `moirai mi`.

3. **Transition UX for Stage 1.** When BH correction drops all results, show: "N raw hits at p<0.2, 0 survived BH correction at q=0.05." This prevents users from thinking the tool is broken.

4. **Gapped motifs minimum sample.** Warn when N < 30: "Gapped motif discovery works best with 30+ runs (you have N)."

5. **New commands get standard filter flags.** `moirai mi`, `moirai rhythm`, `moirai strategies` all support `--model`, `--harness`, `--task-family`, `--strict`.

6. **`Motif` move migration.** Move `Motif` to `schema.py`, add re-export in `motifs.py` so `from moirai.analyze.motifs import Motif` still works. Remove re-export in a later cleanup.

7. **`_nw_align()` backward compatibility.** Add optional `score_fn: Callable[[str, str], float] | None = None` parameter. When None, use existing integer match/mismatch scoring. Stage 5 only.

8. **Permutation test output.** Supplements normal output: print motifs as usual, then append a diagnostic section with empirical FDR.

9. **`pipeline.py` scope (revised).** Start as a statistical utilities module, not a full orchestrator. Houses `benjamini_hochberg()` and permutation test logic. Each command calls it independently. Evolve into orchestrator only if/when cross-family pooling is needed.

---

## Phase 1: Statistical foundation

**Goal:** Fix the multiple testing bug and consolidate duplicated code. Every existing command improves.

### 1.1 Create `moirai/analyze/stats.py`

Extract and consolidate statistical primitives:

- [ ] `fishers_exact_2x2(a, b, c, d) -> float | None` — from `motifs.py:152-189`
- [ ] `hypergeom_pmf(k, N, K, n) -> float | None` — from `divergence.py:168-185`
- [ ] `log_comb(n, k) -> float` — from `divergence.py:188-195`
- [ ] `chi_squared_test(branches) -> float` — from `divergence.py:198-229`
- [ ] `chi2_sf(x, df) -> float` — from `divergence.py:232-245`
- [ ] `benjamini_hochberg(p_values: list[float | None], q: float = 0.05) -> list[float | None]` — new

BH implementation: sort p-values ascending, compute adjusted p_i = p_i × m / rank, enforce monotonicity, return q-values in original order. ~20 lines.

**Tests (`tests/test_stats.py`):**
- [ ] BH on empty input returns empty
- [ ] BH on single p-value returns that p-value
- [ ] BH on known reference dataset (e.g., [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216] at q=0.05 → first 4 survive)
- [ ] BH with None values (skip them, preserve positions)
- [ ] Fisher's exact matches existing implementation (characterization tests)
- [ ] Fisher's exact on known 2x2 tables

### 1.2 Update existing `Motif` and `DivergencePoint`

- [ ] Add `q_value: float | None = None` to `DivergencePoint` in `schema.py:87`
- [ ] Add `risk_difference: float = 0.0` to `Motif` (currently in `motifs.py:12`)
- [ ] Move `Motif` dataclass from `motifs.py` to `schema.py`
- [ ] Add re-export in `motifs.py`: `from moirai.schema import Motif`
- [ ] Update `recommend.py:6` import (already imports from `motifs`)

### 1.3 Wire BH into `find_motifs()`

- [ ] Replace `_fishers_exact_2x2` call in `motifs.py:129` with `stats.fishers_exact_2x2`
- [ ] Remove the duplicate Fisher's implementation from `motifs.py:152-189`
- [ ] After collecting all motifs with raw p-values, apply `benjamini_hochberg()` to get q-values
- [ ] Filter by `q_value <= q_threshold` instead of `p_value <= p_threshold`
- [ ] Keep `p_threshold` parameter for backward compat but add `q_threshold: float = 0.05`
- [ ] Compute `risk_difference = success_rate - baseline_rate` on each motif
- [ ] Change sort key from `abs(success_rate - baseline_rate)` to `(q_value, -abs(risk_difference))`

### 1.4 Wire BH into `find_divergence_points()`

- [ ] Replace Fisher's/chi-squared calls with `stats.*` equivalents
- [ ] Remove duplicate implementations from `divergence.py:138-245`
- [ ] Apply `benjamini_hochberg()` to all divergence point p-values
- [ ] Set `q_value` on each `DivergencePoint`
- [ ] Filter by `q_value` instead of `p_value`

### 1.5 Transition UX

- [ ] In `print_motifs()` (`terminal.py`): when results are empty after BH, print "N patterns found at p<0.2, 0 survived correction at q=0.05"
- [ ] In `print_divergence()` / `print_cluster_divergence()`: same pattern
- [ ] Pass raw hit count alongside filtered results (add parameter to viz functions, or return it alongside results from analysis functions)

### 1.6 Update existing tests

- [ ] `tests/test_motifs.py`: verify motifs now have `q_value` set; add test that BH correction filters stricter than raw p
- [ ] `tests/test_divergence.py`: verify `DivergencePoint` now has `q_value`; same BH filtering test
- [ ] Characterization test: run existing test datasets through old and new code, document what changed

---

## Phase 2: Gapped motifs

**Goal:** Find ordered step patterns with flexible gaps. This is the core new analysis capability.

**Depends on:** Phase 1 (BH correction, stats.py)

### 2.1 Add `GappedMotif` to `schema.py`

```python
@dataclass
class GappedMotif:
    anchors: tuple[str, ...]
    total_runs: int
    success_runs: int
    fail_runs: int
    success_rate: float
    baseline_rate: float
    lift: float
    risk_difference: float
    p_value: float | None
    q_value: float | None = None
    avg_position: float = 0.0

    @property
    def display(self) -> str:
        return " → ... → ".join(self.anchors)
```

- [ ] Add `GappedMotif` to `schema.py`
- [ ] Define `PatternResult` protocol in `schema.py` (shared interface for `Motif` and `GappedMotif`)
- [ ] Verify both `Motif` and `GappedMotif` satisfy the protocol

### 2.2 Implement `find_gapped_motifs()` in `motifs.py`

```python
def find_gapped_motifs(
    runs: list[Run],
    max_length: int = 3,
    min_count: int = 3,
    q_threshold: float = 0.05,
) -> tuple[list[GappedMotif], int]:
    """Returns (motifs, n_tests_run)."""
```

Implementation steps:

- [ ] Build frequency index: map enriched step names to integer IDs, filter by `max(min_count, ceil(0.05 * len(runs)))`, warn if N < 30
- [ ] Enumerate ordered pairs: single-pass per run with `seen_types` set, `run_pairs_seen` deduplication. Increment global `pair_counts: dict[tuple[int,int], tuple[int,int]]` (success, fail).
- [ ] Enumerate ordered triples: extend single-pass with `seen_pairs` set, `run_triples_seen` deduplication. Increment `triple_counts`.
- [ ] If `max_length >= 4`: prefix pruning — build `surviving_triples` set from triples with count >= min_count. Extend pass with `seen_triples` filtered by `surviving_triples`. Increment `quad_counts`.
- [ ] Run Fisher's exact on all candidates via `stats.fishers_exact_2x2()`. Collect raw p-values.
- [ ] Apply `stats.benjamini_hochberg()`. Filter by `q_threshold`.
- [ ] Prune: for each surviving pattern, check if it's a strict subsequence of a longer surviving pattern with same/more runs and same/better q-value. Check transitively.
- [ ] Build `GappedMotif` objects. Sort by `(q_value, -abs(risk_difference))`.
- [ ] Return motifs and total number of tests run (for transition UX).

### 2.3 Extend `patterns` CLI command

In `cli.py`, modify the existing `patterns` command:

- [ ] Add `--gapped` flag (default False)
- [ ] Add `--max-length` option (default 3, only used with `--gapped`)
- [ ] Add `--permutation-test` option (default None, integer)
- [ ] When `--gapped`: call `find_gapped_motifs()` in addition to `find_motifs()`, merge results by q-value
- [ ] When `--permutation-test N`: after computing results, run permutation loop (see 2.4)

### 2.4 Permutation test

Implement in `stats.py`:

```python
def permutation_fdr(
    membership: np.ndarray,     # (n_patterns, n_runs) boolean
    outcomes: np.ndarray,        # (n_runs,) boolean
    q_threshold: float = 0.05,
    n_permutations: int = 1000,
    seed: int = 42,
) -> tuple[float, list[int]]:
    """Returns (empirical_fdr, discoveries_per_permutation)."""
```

- [ ] Pre-compute boolean membership matrix (patterns × runs)
- [ ] Per permutation: shuffle outcome vector, compute contingency tables via matrix multiply, Fisher's exact on each, BH correction, count discoveries
- [ ] Report: empirical FDR = mean(discoveries_per_permutation) / actual_discoveries
- [ ] Use fixed seed for reproducibility

### 2.5 Viz for gapped motifs

- [ ] Extend `print_motifs()` in `terminal.py` to handle both `Motif` and `GappedMotif` via `PatternResult` protocol. Display `" → ... → "` for gapped, `" → "` for contiguous. Show q-value instead of p-value.
- [ ] Update HTML motif rendering in `html.py` if applicable

### 2.6 Tests (`tests/test_gapped_motifs.py`)

- [ ] Basic: 3 runs with known ordered subsequences, verify correct patterns found
- [ ] Greedy matching: run with repeated step types, verify earliest valid assignment
- [ ] Pruning: pattern (A, C) should be pruned when (A, B, C) exists with same/more runs and better q-value
- [ ] Pruning preservation: pattern (A, C) should NOT be pruned when it covers more runs
- [ ] Transitive pruning: (A, B) pruned by (A, B, C, D) directly
- [ ] min_count filter: patterns below threshold excluded
- [ ] Frequency filter: rare step types excluded
- [ ] Empty input: returns empty list
- [ ] Single run: returns empty list (can't discriminate)
- [ ] All-pass or all-fail: returns empty list (no variation)
- [ ] N < 30 warning message
- [ ] `PatternResult` protocol: verify `GappedMotif` satisfies it
- [ ] Permutation test with fixed seed: deterministic output
- [ ] Permutation test on null data (shuffled labels): empirical FDR near q_threshold

---

## Phase 3: Positional mutual information

**Goal:** Find long-range structural couplings between aligned positions.

**Depends on:** Phase 1 (stats.py). Independent of Phase 2.

### 3.1 Add `PositionalMI` to `schema.py`

- [ ] Add `PositionalMI` dataclass as specified in the design spec

### 3.2 Implement `moirai/analyze/coupling.py`

```python
def positional_mutual_information(
    alignment: Alignment,
    runs: list[Run],
    nmi_threshold: float = 0.3,
    min_distinct: int = 2,
) -> list[PositionalMI]:
```

- [ ] Check N >= 500. If not, return empty list (caller handles warning).
- [ ] For each column pair (i, j) where i < j:
  - Build joint distribution from non-GAP values (streaming — don't store all pairs)
  - Compute MI, H(i), H(j), NMI
  - If NMI < threshold or either column has < min_distinct values, discard and continue
  - For survivors: chi-squared test on (value_i × value_j × outcome) table
  - Store `PositionalMI` with raw p-value
- [ ] Apply BH correction across MI p-values only (per-command scope)
- [ ] Return sorted by NMI descending

### 3.3 Add `moirai mi` CLI command

- [ ] Standard filter flags: `--model`, `--harness`, `--task-family`, `--strict`
- [ ] `--nmi-threshold` option (default 0.3)
- [ ] Check known-outcome run count >= 500 before calling analysis. If below, print warning to stderr and exit 1.
- [ ] Call `_load_and_filter()` → `align_runs()` → `positional_mutual_information()`
- [ ] Call `print_mi()` (new viz function)

### 3.4 Viz

- [ ] `print_mi()` in `terminal.py`: Rich table with columns (col_i, col_j, NMI, q-value, top joint values)
- [ ] Show the step values at each column for context

### 3.5 Tests (`tests/test_coupling.py`)

- [ ] Perfect correlation: two identical columns → NMI = 1.0
- [ ] Independence: two unrelated columns → NMI ≈ 0.0
- [ ] N < 500: returns empty list
- [ ] GAP handling: rows with GAP in either column excluded
- [ ] Single distinct value in a column: filtered out
- [ ] Chi-squared outcome test: known contingency table
- [ ] Streaming: memory stays bounded (no materialization of all joint distributions)

---

## Phase 4: Entropy rate profiles

**Goal:** Measure per-step behavioral predictability and the "discipline gap" between successful and failing runs.

**Depends on:** Phase 1 (stats.py). Independent of Phases 2-3.

### 4.1 Add `EntropyProfile` to `schema.py`

- [ ] Add `EntropyProfile` dataclass as specified, including `context_k: int` field

### 4.2 Implement `moirai/analyze/rhythm.py`

```python
def entropy_profiles(
    alignment: Alignment,
    runs: list[Run],
    cluster_labels: dict[str, int],
    k: int | None = None,  # None = adaptive
) -> list[EntropyProfile]:
```

- [ ] Group runs by cluster using `cluster_labels`
- [ ] Per cluster: determine k (if adaptive: k=3 when cluster has >= 200 known-outcome runs, else k=1)
- [ ] Per column i: build empirical distribution of step values conditional on context (previous k columns). Count contexts from all runs in the cluster.
- [ ] Drop contexts with < 2 observations
- [ ] Compute conditional entropy H(column_i | context)
- [ ] Split by outcome: compute entropy for success-only and failure-only runs separately
- [ ] Compute discipline_gap = failure_entropy - success_entropy (None if either is None)
- [ ] Return one `EntropyProfile` per cluster

### 4.3 Add `moirai rhythm` CLI command

- [ ] Standard filter flags
- [ ] `--level` and `--threshold` options (passed to `cluster_runs()`, same as `clusters` command)
- [ ] `--context-k` option (default None = adaptive)
- [ ] Call `_load_and_filter()` → `cluster_runs()` → `align_runs()` per cluster → `entropy_profiles()`
- [ ] Call `print_rhythm()` (new viz function)

### 4.4 Viz

- [ ] `print_rhythm()` in `terminal.py`: per cluster, show a compact sparkline-like representation of the entropy profile. Highlight columns with discipline_gap > 0.5 (or some threshold). Show context_k used.
- [ ] When cluster is 100% pass or 100% fail, show "N/A — no outcome variation in cluster"

### 4.5 Tests (`tests/test_rhythm.py`)

- [ ] Constant sequence: all runs same step at every column → entropy = 0.0 everywhere
- [ ] Random sequence: high entropy
- [ ] Adaptive k: cluster with 300 runs uses k=3, cluster with 50 runs uses k=1
- [ ] Discipline gap: construct cluster where passing runs are consistent at column X and failing runs vary → positive gap at X
- [ ] All-pass cluster: success_entropy is populated, failure_entropy is all None
- [ ] Empty cluster: returns empty profile
- [ ] GAP handling in context windows

---

## Phase 5: Domain architecture

**Goal:** Cluster runs by motif-level strategy rather than step-level similarity.

**Depends on:** Phase 2 (needs validated gapped motifs as input).

**Gate:** Only proceed if Phase 2 motifs produce clean, stable results on real eval-harness and SWE-smith data. If motifs are too noisy or too few survive BH, defer this phase.

### 5.1 Add dataclasses to `schema.py`

- [ ] `DomainArchitecture` and `StrategyCluster` as specified

### 5.2 Parameterize `_nw_align()` in `align.py`

- [ ] Add optional `score_fn: Callable[[str, str], float] | None = None` parameter
- [ ] When None: use existing `match`/`mismatch`/`gap` integer scoring (backward compat)
- [ ] When provided: use `score_fn(a, b)` for the diagonal score, `gap` for gap penalty
- [ ] Update all existing callers to pass None explicitly (no behavior change)

### 5.3 Implement `moirai/analyze/domains.py`

```python
def domain_architectures(
    runs: list[Run],
    motifs: list[PatternResult],
) -> list[DomainArchitecture]:

def strategy_clusters(
    architectures: list[DomainArchitecture],
    step_labels: dict[str, int],
    threshold: float = 0.3,
) -> list[StrategyCluster]:
```

- [ ] Motif matching: for each run, match all motifs (greedy left-to-right). Resolve conflicts by q-value.
- [ ] Build `DomainArchitecture` per run. Compute `unmatched_fraction`.
- [ ] If no motifs survive for any run, return early with warning.
- [ ] Domain-level NW alignment using parameterized `_nw_align()` with motif-ID scoring function
- [ ] Agglomerative clustering on domain-distance matrix (reuse existing `cluster.py` pattern)
- [ ] Build `StrategyCluster` objects with `step_cluster_overlap`

### 5.4 Add `moirai strategies` CLI command

- [ ] Standard filter flags
- [ ] `--level`, `--threshold` (for step-level clustering baseline)
- [ ] `--min-count`, `--max-length` (passed through to motif discovery)
- [ ] Call `_load_and_filter()` → `find_motifs()` + `find_gapped_motifs()` → `domain_architectures()` → `strategy_clusters()`
- [ ] Call `print_strategies()` (new viz function)
- [ ] When no motifs survive: print "No significant motifs found — strategy clustering requires discriminative patterns. Try with more runs."

### 5.5 Viz

- [ ] `print_strategies()` in `terminal.py`: per strategy cluster, show prototype architecture, success rate, run count, overlap with step-level clusters

### 5.6 Tests (`tests/test_domains.py`)

- [ ] Empty motif list: returns empty architectures
- [ ] Runs with no matching motifs: `unmatched_fraction = 1.0`
- [ ] Overlapping motif conflict: lower q-value wins
- [ ] Strategy clustering: runs with same architecture → same cluster
- [ ] `step_cluster_overlap` computation: known mapping verified
- [ ] Parameterized `_nw_align()`: custom score_fn produces different alignment than default

---

## Acceptance criteria

### Functional

- [ ] `moirai patterns` applies BH correction; shows transition message when correction drops all results
- [ ] `moirai patterns --gapped` finds ordered subsequences with gaps that contiguous n-grams miss
- [ ] `moirai patterns --permutation-test 1000` reports empirical FDR
- [ ] `moirai mi` finds coupled aligned positions (N >= 500 only)
- [ ] `moirai rhythm` shows entropy profiles with discipline gap per cluster
- [ ] `moirai strategies` clusters runs by motif-level architecture (Phase 5, gated)
- [ ] All new commands support `--model`, `--harness`, `--task-family`, `--strict` filters
- [ ] All existing tests pass after each phase

### Non-functional

- [ ] `moirai patterns --gapped` runs in under 2s at N=165, V=20
- [ ] `moirai patterns --gapped --max-length 4` runs in under 10s at N=500, V=30
- [ ] `moirai mi` runs in under 1s (excluding alignment prerequisite)
- [ ] `moirai rhythm` runs in under 1s
- [ ] `moirai patterns --permutation-test 1000` runs in under 30s at N=500

### Quality gates

- [ ] All new functions have tests covering empty input, single item, normal case, and edge cases
- [ ] No duplicate Fisher's exact implementations remain after Phase 1
- [ ] `BH` correction is applied to all hypothesis tests (motifs, divergence, MI)
- [ ] Characterization test documents what existing output changed after Phase 1

---

## Risk analysis

| Risk | Impact | Mitigation |
|---|---|---|
| BH correction eliminates all existing findings on some datasets | Users think tool is broken | Transition UX: show raw hit count. Consider `--raw` escape hatch |
| Gapped motifs don't find anything contiguous motifs don't | Feature has no value | Test on real eval-harness and SWE-smith data early in Phase 2 |
| MI at N=500 is still too sparse for reliable estimation | False couplings reported | NMI threshold 0.3 is conservative; permutation test validates |
| Domain architecture amplifies noisy motifs | Strategy clusters are meaningless | Phase 5 is gated on Phase 2 validation |
| `_nw_align()` parameterization breaks existing callers | Regression | Add score_fn as optional param with None default; existing callers unchanged |

## References

- Design spec: `docs/specs/2026-03-30-trajectory-biology-design.md`
- Existing analysis modules: `moirai/analyze/*.py`
- CLI entry point: `moirai/cli.py`
- Schema: `moirai/schema.py`
- Viz: `moirai/viz/terminal.py`, `moirai/viz/html.py`
- Fisher's exact (to consolidate): `motifs.py:152-189`, `divergence.py:138-196`
- SWE-smith converter: `scripts/convert_swe_smith.py`
- Eval-harness converter: `scripts/convert_eval_harness.py`
