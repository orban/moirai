---
title: "feat: trajectory biology — multi-level analysis framework"
type: feat
date: 2026-03-30
spec: docs/specs/2026-03-30-trajectory-biology-design.md
---

# Trajectory biology implementation plan

Five staged implementation adding gapped motifs, positional MI, domain architecture, entropy profiles, and a statistical foundation fix to moirai.

**Design spec:** `docs/specs/2026-03-30-trajectory-biology-design.md`

## Key decisions

1. **BH correction scope per command.** Each command applies BH across its own test family. `moirai patterns` corrects across motif tests only. `moirai branch` corrects across divergence tests only. This means q-values from different commands aren't directly comparable (a motif at q=0.03 in `moirai patterns` might get a different q-value if pooled with divergence tests). This is the right default for a CLI tool where commands run independently — full cross-family pooling would require `moirai patterns` to run alignments just for statistical correction, turning a 1s command into a multi-minute pipeline.

2. **Drop `branch --mi`.** The `branch` command groups by task_id (3-30 runs per group). MI requires N >= 500. Per-task MI can't work. MI is standalone only via `moirai mi`.

3. **Transition UX.** When BH correction drops all results, show: "N raw hits at p<0.2, 0 survived BH correction at q=0.05."

4. **Gapped motifs minimum sample.** Warn when N < 30.

5. **New commands get standard filter flags.** `moirai mi`, `moirai rhythm`, `moirai strategies` all support `--model`, `--harness`, `--task-family`, `--strict`.

6. **No `pipeline.py`.** Per-command BH means no orchestrator is needed. `stats.py` houses `benjamini_hochberg()` and `permutation_fdr()`. Each command calls them directly.

7. **No `PatternResult` protocol.** `Motif` and `GappedMotif` have different field names (`.pattern` vs `.anchors`), so a protocol can't unify them cleanly. `print_motifs()` accepts `list[Motif | GappedMotif]` instead.

8. **`Motif` moves to `schema.py` in Phase 2** alongside `GappedMotif`. Update the one import in `recommend.py` directly. No re-export shim.

9. **Permutation test uses vectorized chi-squared**, not per-pattern Fisher's exact. Fisher's in the permutation loop would take ~800s at N=500. Vectorized chi-squared takes ~3.5s.

10. **Alignment wall-clock costs.** MI and rhythm at N=500 will wait 30-60s for alignment before the fast analysis starts. Document this in CLI help text.

---

## Phase 1: Statistical foundation + permutation test

**Goal:** Fix the multiple testing bug, consolidate duplicated code, and add permutation validation. Every existing command improves.

### 1.1 Create `moirai/analyze/stats.py`

Consolidate duplicated statistical primitives from `motifs.py:152-189` and `divergence.py:138-245` into a shared module. Add BH correction and permutation test.

- [x] `fishers_exact_2x2()`, `hypergeom_pmf()`, `log_comb()` — consolidated from duplicates
- [x] `chi_squared_test()`, `chi2_sf()` — from `divergence.py`
- [x] `benjamini_hochberg(p_values, q=0.05)` — new, ~20 lines
- [x] `permutation_fdr(membership, outcomes, q_threshold, n_permutations, seed)` — new, uses vectorized chi-squared approximation inside the loop (not Fisher's exact), numpy `argsort`-based vectorized BH

### 1.2 Wire BH into existing analysis

- [x] `find_motifs()`: replace inline Fisher's with `stats.*`, apply BH, filter by q-value, sort by `(q_value, -abs(success_rate - baseline_rate))`
- [x] `find_divergence_points()`: replace inline Fisher's/chi-squared with `stats.*`, apply BH, set `q_value` on each `DivergencePoint`
- [x] Add `q_value: float | None = None` to `DivergencePoint` in `schema.py`
- [x] Remove all duplicate implementations from `motifs.py` and `divergence.py`

### 1.3 Transition UX

- [x] When BH drops all results, print: "N raw hits at p<0.2, 0 survived correction at q=0.05"
- [x] Apply to `print_motifs()` and `print_divergence()` / `print_cluster_divergence()`

### 1.4 Add `--permutation-test` to `patterns` command

- [x] `--permutation-test N` option on `moirai patterns`
- [x] Pre-compute boolean membership matrix, run vectorized permutation loop
- [x] Print motifs as usual, then append diagnostic: "Empirical FDR: X% (N permutations, Y mean discoveries vs Z actual)"

### 1.5 Tests

**`tests/test_stats.py`:**
- [x] BH: empty, single, known reference dataset, None handling
- [x] Fisher's exact: characterization tests matching existing output
- [x] Permutation FDR: deterministic with fixed seed

**Update existing tests:**
- [x] `test_motifs.py`: verify `q_value` set, BH filters stricter than raw p
- [x] `test_divergence.py`: verify `DivergencePoint.q_value` set
- [x] Characterization test: document what changed

---

## Phase 2: Gapped motifs

**Goal:** Find ordered step patterns with flexible gaps.

**Depends on:** Phase 1

### 2.1 Add `GappedMotif` to `schema.py`, move `Motif`

- [x] Move `Motif` from `motifs.py` to `schema.py`
- [x] Update `recommend.py` import: `from moirai.schema import Motif`
- [x] Add `GappedMotif` to `schema.py`

### 2.2 Implement `find_gapped_motifs()` in `motifs.py`

- [x] Build frequency index: map to integer IDs, filter by `max(min_count, ceil(0.05 * N))`
- [x] Enumerate ordered pairs: single-pass with `seen_types`, `run_pairs_seen` dedup
- [x] Enumerate ordered triples: pairs collected AFTER triples loop to prevent same-step artifacts
- [x] Apply BH, prune subsequences (transitive), sort by `(q_value, -abs(success_rate - baseline_rate))`

### 2.3 Extend `patterns` CLI command

- [x] `--gapped` flag, `--max-length` option (default 3)
- [x] When `--gapped`: run both `find_motifs()` and `find_gapped_motifs()`, merge by q-value
- [x] Permutation test covers both contiguous and gapped when `--gapped` is set

### 2.4 Viz

- [x] `print_motifs()` handles both `Motif` and `GappedMotif` via shared attributes

### 2.5 Tests (`tests/test_gapped_motifs.py`)

- [x] Basic subsequence discovery, greedy matching with repeated types
- [x] Pruning: shorter pruned when longer covers same/more runs with better q-value
- [x] Pruning preservation: shorter kept when it covers more runs
- [x] Filters: min_count, frequency, empty input, single run, all-pass/all-fail
- [x] CLI smoke test via Typer test client: `patterns --gapped` on fixture data

---

## Phase 3: Positional mutual information

**Goal:** Find long-range structural couplings between aligned positions.

**Depends on:** Phase 1. Independent of Phase 2.

**Note:** At N=500, `align_runs()` takes ~40s. Document this in CLI help: "Computing alignment (this may take a minute for large datasets)..."

### 3.1 Add `PositionalMI` to `schema.py`

- [ ] As specified in design spec. Serialization: flatten `joint_values` tuple keys to `"val_i|val_j"` strings for JSON.

### 3.2 Implement `moirai/analyze/coupling.py`

- [ ] Check N >= 500. If not, return empty list.
- [ ] Stream MI computation per column pair: build joint distribution, compute NMI, discard if below threshold. Don't materialize all joint distributions.
- [ ] Chi-squared on survivors' (value_i × value_j × outcome) table
- [ ] Apply BH across MI p-values

### 3.3 Add `moirai mi` CLI command

- [ ] Standard filter flags
- [ ] N < 500 check: print warning to stderr, exit 1
- [ ] Call `_load_and_filter()` → `align_runs()` → `positional_mutual_information()`

### 3.4 Viz + Tests

- [ ] `print_mi()` in `terminal.py`: Rich table (col_i, col_j, NMI, q-value, top joint values)
- [ ] Tests: perfect correlation (NMI=1.0), independence (NMI≈0.0), N<500 returns empty, GAP handling, streaming memory
- [ ] CLI smoke test via Typer test client

---

## Phase 4: Entropy rate profiles

**Goal:** Measure per-step behavioral predictability and the discipline gap.

**Depends on:** Phase 1. Independent of Phases 2-3.

### 4.1 Add `EntropyProfile` to `schema.py`

- [ ] As specified, with `context_k: int` field

### 4.2 Implement `moirai/analyze/rhythm.py`

- [ ] Group runs by cluster, determine adaptive k: k=3 when cluster >= 500 known-outcome runs, k=1 otherwise. (k=3 at N<500 produces too-sparse context distributions.)
- [ ] Per column: empirical conditional entropy, split by outcome, discipline gap
- [ ] 100% pass or 100% fail clusters: set failure/success entropy to None

### 4.3 Add `moirai rhythm` CLI command

- [ ] Standard filter flags + `--level`, `--threshold` (for clustering), `--context-k`
- [ ] Call `_load_and_filter()` → `cluster_runs()` → `align_runs()` per cluster → `entropy_profiles()`
- [ ] When cluster has no outcome variation: "N/A — no outcome variation"

### 4.4 Viz + Tests

- [ ] `print_rhythm()`: compact entropy profile per cluster, highlight large discipline gaps
- [ ] Tests: constant sequence (entropy=0), adaptive k threshold, discipline gap construction, all-pass cluster, GAP handling
- [ ] CLI smoke test via Typer test client

---

## Phase 5: Domain architecture (stub — detailed planning after Phase 2 validation)

**Goal:** Cluster runs by motif-level strategy rather than step-level similarity.

**Depends on:** Phase 2 producing clean, stable motifs on real data.

**Gate:** Only proceed if Phase 2 motifs survive BH correction, differentiate outcomes meaningfully, and produce interpretable patterns on real eval-harness and SWE-smith data. If motifs are too noisy or too few survive, defer indefinitely.

**High-level approach:** Given significant motifs, scan each run for which motifs it contains and in what order. Compare runs by their "domain architecture" (ordered list of matched motifs) using NW alignment at the motif level. Cluster by domain-architecture distance. Compare strategy clusters to step-level clusters.

**Key implementation note:** `_nw_align()` in `align.py` is private. When this phase proceeds, either make it public or add a public wrapper with an optional `score_fn` parameter for motif-level scoring. Don't modify `_nw_align()` until this phase is greenlit.

**Detailed task breakdown deferred** until Phase 2 validation results are in hand.

---

## Acceptance criteria

### Functional

- [ ] `moirai patterns` applies BH correction; shows transition message when correction drops all results
- [ ] `moirai patterns --gapped` finds ordered subsequences with gaps that contiguous n-grams miss
- [ ] `moirai patterns --permutation-test 1000` reports empirical FDR
- [ ] `moirai mi` finds coupled aligned positions (N >= 500 only)
- [ ] `moirai rhythm` shows entropy profiles with discipline gap per cluster
- [ ] All new commands support `--model`, `--harness`, `--task-family`, `--strict` filters
- [ ] All existing tests pass after each phase

### Non-functional

- [ ] `moirai patterns --gapped` under 2s at N=165, V=20
- [ ] `moirai patterns --gapped --max-length 4` under 10s at N=500, V=30
- [ ] `moirai mi` under 1s (excluding alignment; document alignment cost in help text)
- [ ] `moirai rhythm` under 1s
- [ ] `moirai patterns --permutation-test 1000` under 30s at N=500

### Quality gates

- [ ] No duplicate Fisher's exact implementations remain after Phase 1
- [ ] BH correction applied to all hypothesis tests
- [ ] Characterization test documents what existing output changed after Phase 1
- [ ] CLI smoke tests for every new command

---

## Risk analysis

| Risk | Impact | Mitigation |
|---|---|---|
| BH correction eliminates all existing findings | Users think tool is broken | Transition UX shows raw hit count |
| Gapped motifs don't find anything contiguous misses | Feature has no value | Test on real eval-harness/SWE-smith data early in Phase 2 |
| MI too sparse at N=500 | False couplings | NMI threshold 0.3 is conservative; permutation test validates |
| Domain architecture amplifies noisy motifs | Meaningless clusters | Phase 5 gated on Phase 2 validation |
| Per-command BH gives different q-values than pooled | User confusion | Document in CLI help; consistent within each command |

## References

- Design spec: `docs/specs/2026-03-30-trajectory-biology-design.md`
- Analysis modules: `moirai/analyze/*.py`
- CLI: `moirai/cli.py`
- Schema: `moirai/schema.py`
- Viz: `moirai/viz/terminal.py`, `moirai/viz/html.py`
- Fisher's exact (to consolidate): `motifs.py:152-189`, `divergence.py:138-196`
- Data converters: `scripts/convert_swe_smith.py`, `scripts/convert_eval_harness.py`
