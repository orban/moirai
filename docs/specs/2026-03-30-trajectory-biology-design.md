# Trajectory biology: multi-level analysis for moirai

## Overview

Moirai currently operates almost entirely at the primary structure level — raw step sequences, Needleman-Wunsch alignment, contiguous n-gram motifs. This design introduces three additional levels of analysis borrowed from genomics and information theory, plus statistical fixes to the existing system.

The core idea: biological sequences have multiple levels of organization (primary, secondary, tertiary, quaternary), and each level has its own natural analysis tools. Agent trajectories have the same structure. Applying primary-structure tools to tertiary-structure questions (e.g., NW alignment when the real question is about strategy architecture) produces noisy results.

## Levels of organization

**Primary structure** — the raw step sequence. NW alignment, edit distance, contiguous n-grams. What moirai does today.

**Secondary structure** — local behavioral motifs with internal flexibility. Recurring patterns like "read test file, then (some steps), then edit source, then (some steps), then test." These have a recognizable shape regardless of what comes before or after. The motif has grammar, not just content.

**Tertiary structure** — global strategy architecture. Which secondary-structure motifs appear, in what order, with what pacing. Two runs might have identical local patterns but different global strategies. This is where rhythm and pacing live.

**Quaternary structure** — multi-agent interaction. Handoffs, subagent delegation. Out of scope for this design.

## Statistical foundation (applies to all sections)

### BH correction across all tests

The existing codebase runs hundreds of Fisher's exact tests (contiguous n-grams in `motifs.py`, divergence columns in `divergence.py`) with a raw p-value threshold of 0.2 and no multiple testing correction. At 400 tests, this produces ~80 false positives by chance alone.

**Fix:** Apply Benjamini-Hochberg FDR correction at q = 0.05 across all tests. This is non-negotiable and applies retroactively to existing contiguous motifs and divergence points, not just new features.

Implementation: a shared `benjamini_hochberg(p_values: list[float], q: float = 0.05) -> list[float | None]` function in a new `moirai/analyze/stats.py` module. Returns adjusted q-values. Callers filter by `q_value <= q` instead of `p_value <= p_threshold`.

The existing duplicate Fisher's exact test implementations in `motifs.py` (lines 152-189) and `divergence.py` (lines 138-196) should be consolidated into `stats.py`.

**Pooling across test families:** BH correction pools all p-values from all test types into a single correction. The pooled families are: contiguous n-gram Fisher's, gapped motif Fisher's, divergence column Fisher's/chi-squared, and MI pair chi-squared. Entropy profiles (Section 4) produce no hypothesis tests and contribute no p-values to the pool. Pooling is valid under PRDS (positive regression dependency on subsets), which holds for tests sharing anchors or runs (Benjamini & Yekutieli 2001).

### Orchestration

BH pooling requires a coordinator — no single analysis module can produce final q-values in isolation. Create `moirai/analyze/pipeline.py` to orchestrate multi-module analysis:

1. Run all test-producing analyses (motifs, gapped motifs, divergence, MI)
2. Collect all raw p-values
3. Apply BH correction
4. Distribute adjusted q-values back to results

Each analysis function returns results with raw p-values only. `pipeline.py` finalizes them. Independent callers who use analysis functions directly get valid raw p-values and can apply their own corrections.

The pipeline module is also the natural home for the permutation test, which needs to rerun all statistical tests with shuffled labels.

### Permutation test validation

`moirai patterns --permutation-test N` shuffles outcome labels (success/failure) N times (default 1000), reruns the statistical tests, and reports empirical FDR. This validates whether the pipeline produces signal or noise on a specific dataset.

**Critical: caching.** The permutation test caches all sequence-derived computations (alignments, pattern-run membership, MI column values). Only statistical tests (contingency table construction, Fisher's exact, chi-squared, BH correction) are re-run per permutation. Without caching, 1000 permutations at N=500 would take days. With caching, it takes under 10 minutes.

**Recommended optimization:** Pre-compute a boolean membership matrix (patterns × runs). Per permutation, shuffle the label vector, multiply to get all contingency tables in one operation. This brings per-iteration cost to ~5ms, making 1000 iterations feasible in under 10 seconds at any scale.

This is a diagnostic, not a filter. It's opt-in. The label-shuffle null is correct for validating Fisher's exact tests (it directly mirrors the null hypothesis). The permutation test scope matches the BH pooling scope — it simulates the same pool of tests that BH corrects.

### Ranking

All patterns (contiguous and gapped motifs, MI pairs) are ranked by BH-adjusted q-value ascending. Tiebreaker: risk difference (`success_rate - baseline_rate`), which directly answers "how much does this pattern change success probability" and is bounded and interpretable.

### Success rate convention

Throughout this design, `success_rate = success_runs / (success_runs + fail_runs)`. The denominator excludes runs with unknown outcomes. `total_runs` is the count of all matching runs including unknowns. Both fields are present on all pattern dataclasses to avoid ambiguity.

### Shared type interface

`Motif` and `GappedMotif` share most fields. Define a `PatternResult` protocol so downstream consumers (viz, narrate, recommend) handle both types uniformly without `isinstance` checks:

```python
class PatternResult(Protocol):
    total_runs: int
    success_runs: int
    fail_runs: int
    success_rate: float
    baseline_rate: float
    lift: float
    risk_difference: float
    p_value: float | None
    q_value: float | None

    @property
    def display(self) -> str: ...
```

Add `q_value: float | None = None` and `risk_difference: float` to the existing `Motif` and `DivergencePoint` dataclasses as part of Stage 1.

### Dataclass placement

All new dataclasses go in `schema.py`, following the existing convention where `Step`, `Run`, `Alignment`, `DivergencePoint`, `ClusterInfo`, etc. all live there. Move the existing `Motif` class from `motifs.py` to `schema.py` as well to resolve the current inconsistency.

---

## Section 1: Gapped motif patterns (secondary structure)

### Problem

`motifs.py` finds contiguous n-grams: `read(test_file) → edit(source) → test`. It can't find `read(test_file) → ... → edit(source) → ... → test` where the `...` varies across runs. The most important behavioral patterns — "always read the test file before editing" — have flexible gaps.

### Design

**Pattern representation:** An ordered subsequence of anchor steps with unconstrained gaps. Pattern `(A, B, C)` matches a run if there exist positions i < j < k where step[i] = A, step[j] = B, step[k] = C. No gap bounds are learned or enforced.

**Matching semantics:** Greedy left-to-right scan. For pattern `(A, B, C)`: find the first A, then the first B after that A, then the first C after that B. This is deterministic and O(L) per pattern per run. When step types repeat (e.g., `read(source)` appears 10 times), greedy matching uses the earliest valid assignment.

**Frequency filter:** Only step types appearing in >= 5% of runs are "frequent" and enter the candidate pool. Report the actual number of tests conducted alongside results.

### Discovery algorithm

1. **Build frequency index.** For each of V enriched step types, count runs containing it. Drop types below max(min_count, 5% of N). Map surviving types to integer IDs for faster hashing throughout.

2. **Enumerate ordered pairs.** All (A, B) from frequent types where A appears before B in >= min_count runs. Compute Fisher's exact p-value for each pair's association with outcome.

3. **Enumerate ordered triples.** All (A, B, C) from frequent types as ordered subsequences in >= min_count runs. Fisher's exact test.

4. **Optionally enumerate length-4** (opt-in via `--max-length 4`, default is 3). Uses prefix pruning: only track a triple in the length-4 pass if that triple survived min_count at stage 3. This reduces the inner loop by ~90%.

5. **BH correction.** Handled by `pipeline.py`: pool all raw p-values (gapped + contiguous + divergence + MI) and apply BH at q = 0.05.

6. **Prune.** After BH correction, drop patterns that are strict subsequences of longer patterns, but only if the longer pattern covers the same or more runs AND has an equal or better q-value. If the shorter pattern covers more runs (is more general), keep both. Check transitive subsequence relations (a length-2 can be pruned by a length-4 directly). Pruning is display-only — no re-computation of q-values after pruning.

7. **Rank** by q-value ascending, risk difference as tiebreaker.

### Counting

Single-pass per run using integer IDs. Walk the step sequence maintaining `seen_types`, `seen_pairs`, and per-run deduplication sets (`run_pairs_seen`, `run_triples_seen`) to avoid double-counting when step types repeat. For length-4, only track triples that survived min_count at stage 3 (`surviving_triples` set).

Total complexity:
- Pairs: O(N × L × V). Negligible.
- Triples: O(N × L × |seen_pairs|). Under 1s at current scale, ~2s at N=500/V=30.
- Length-4 with prefix pruning: O(N × L × |surviving_triples|). Typically a few seconds. Opt-in.

### Output

```python
@dataclass
class GappedMotif:
    anchors: tuple[str, ...]
    total_runs: int              # all matching runs (including unknown outcome)
    success_runs: int            # matching runs with success=True
    fail_runs: int               # matching runs with success=False
    success_rate: float          # success_runs / (success_runs + fail_runs)
    baseline_rate: float         # overall success rate among known-outcome runs
    lift: float                  # success_rate / baseline_rate
    risk_difference: float       # success_rate - baseline_rate
    p_value: float | None        # raw Fisher's exact
    q_value: float | None        # BH-adjusted (set by pipeline)
    avg_position: float          # mean normalized position of first anchor

    @property
    def display(self) -> str:
        return " → ... → ".join(self.anchors)
```

### CLI

```bash
moirai patterns path/to/runs/ --gapped              # contiguous + gapped, max length 3
moirai patterns path/to/runs/ --gapped --max-length 4  # include length-4 (slower)
moirai patterns path/to/runs/ --permutation-test 1000  # validate with permutation test
```

### Where it lives

- `find_gapped_motifs()` in `moirai/analyze/motifs.py`
- `benjamini_hochberg()`, `fishers_exact_2x2()` in new `moirai/analyze/stats.py`
- BH orchestration in new `moirai/analyze/pipeline.py`
- `GappedMotif` dataclass in `moirai/schema.py`

---

## Section 2: Positional mutual information (primary → secondary bridge)

### Problem

Divergence analysis treats each aligned column independently. But agent decisions are coupled: reading the test file at step 10 might determine whether the agent loops at step 30. These long-range couplings are invisible to column-by-column analysis.

### Minimum sample size

MI requires estimating joint probability distributions P(x, y) over column pairs. With V step types per column, reliable MI estimation needs enough runs to populate the joint distribution.

**Requirement:** N >= 500 runs with known outcomes. Below this threshold, `moirai mi` prints a warning and exits. At N=500 with V=20, the average cell in a 20×20 joint distribution has ~1.25 observations — still sparse, but NMI thresholding filters out unreliable pairs. At N=2000+ (achievable with SWE-bench/SWE-smith data), MI estimation becomes comfortable.

### Design

**What it computes:** For each pair of aligned columns (i, j), the mutual information between step values at those positions across all runs. High MI means knowing the agent's choice at position i tells you what it'll do at position j.

**Algorithm:**

1. Take an existing alignment from `align_runs()`. MI reuses this alignment — it does not recompute it. The alignment itself is the dominant cost (~7s at N=165, ~3.5 min at N=500).

2. For each column pair (i, j) where i < j, compute MI from the joint distribution of step values:

   ```
   MI(i, j) = Σ P(x,y) × log₂(P(x,y) / (P(x) × P(y)))
   ```

   where x and y are step values at columns i and j across all runs, excluding runs with GAP in either column.

3. Normalize: `NMI(i, j) = MI(i, j) / min(H(i), H(j))`, giving NMI in [0, 1].

4. Filter: keep pairs where NMI > threshold (default 0.3) and both columns have >= 2 distinct non-GAP values.

5. For each high-MI pair, test whether the *combination* of values at (i, j) predicts success better than either column alone. Chi-squared test on the (value_i × value_j × outcome) contingency table. P-value feeds into the shared BH correction pool.

**Memory:** Compute MI per column pair in a streaming loop. Discard the joint distribution immediately for pairs below the NMI threshold. Do not materialize all C²/2 joint distributions simultaneously (would peak at ~540MB at large scale).

**Computational cost:** O(C² × N) where C is alignment columns (~50-110), N is runs. Under 0.5s even at large scale. The alignment prerequisite dominates.

**Serialization note:** `joint_values` uses tuple keys `(str, str)` which aren't valid JSON keys. Flatten to `"val_i|val_j"` string keys for serialization.

### Output

```python
@dataclass
class PositionalMI:
    column_i: int
    column_j: int
    mutual_information: float         # raw MI in bits
    normalized_mi: float              # NMI in [0, 1]
    joint_values: dict[tuple[str, str], int]  # (val_i, val_j) -> count
    outcome_p_value: float | None     # chi-sq: does the combination predict outcome?
    outcome_q_value: float | None     # BH-adjusted (set by pipeline)
```

### CLI

```bash
moirai mi path/to/runs/              # show high-MI column pairs (requires N >= 500)
moirai branch path/to/runs/ --mi     # annotate divergence points with MI couplings
```

### Where it lives

- `positional_mutual_information()` in new `moirai/analyze/coupling.py`
- `PositionalMI` dataclass in `moirai/schema.py`
- MI p-values included in `pipeline.py` BH pool

### Relationship to divergence analysis

MI is complementary, not a replacement. Divergence asks "does this column matter for outcomes?" MI asks "are these columns coupled to each other?" A high-MI pair where one column is also a divergence point tells you *which earlier decision constrained the critical fork*.

---

## Section 3: Domain architecture (tertiary structure)

### Problem

Clustering uses step-level edit distance, which over-penalizes superficial differences (different file names, different search patterns) and under-penalizes structural differences (same steps in a different strategic order). Two runs with the same strategy but different file targets look dissimilar; two runs with different strategies but similar step distributions look similar.

### Prerequisite

Section 1 (gapped motifs) must be implemented and validated on real data first. Domain architecture quality depends entirely on motif quality.

### Design

**What it does:** Given significant motifs (contiguous + gapped), scan each run for which motifs it contains and in what order. This produces a "domain architecture" — a compact representation of a run's strategy as an ordered list of recognized behavioral patterns.

**Algorithm:**

1. Take the set of significant motifs from Section 1 (after BH correction and pruning).

2. For each run, match all motifs using greedy left-to-right matching. Record each motif's first anchor position.

3. Handle overlapping motifs: two motifs conflict if they require the same step position for different anchors. Keep non-conflicting motifs. When motifs conflict, prefer the one with a lower q-value (more statistically significant).

4. Sort matched motifs by first anchor position to get the domain architecture.

5. Compute pairwise domain-architecture distance using Needleman-Wunsch at the motif level. Parameterize `_nw_align()` with a scoring function rather than duplicating the algorithm — motif-level alignment may want partial credit for similar motifs.

6. Cluster runs by domain-architecture distance. These are "strategy clusters."

7. Compare strategy clusters to existing step-level clusters. Report where they agree and disagree.

**Computational cost:**
- Motif matching: O(M × L × N) where M is number of significant motifs (typically 10-50). Under 0.5s.
- Pairwise NW on domain architectures (length ~5-15): O(N² × D²) where D is domain architecture length. At N=500, D=10: ~2s. Acceptable.

### Output

```python
@dataclass
class DomainArchitecture:
    run_id: str
    domains: list[str]              # ordered motif names
    domain_positions: list[float]   # normalized first-anchor position of each
    unmatched_fraction: float       # fraction of steps not covered by any motif

@dataclass
class StrategyCluster:
    cluster_id: int
    count: int
    success_rate: float | None
    prototype_architecture: list[str]
    step_cluster_overlap: dict[int, int]  # step_cluster_id -> run count
```

### CLI

```bash
moirai strategies path/to/runs/     # show strategy clusters with domain architectures
```

### Where it lives

- New `moirai/analyze/domains.py`. Depends on `motifs.py` (Section 1) and `align.py`.
- `DomainArchitecture`, `StrategyCluster` dataclasses in `moirai/schema.py`.

---

## Section 4: Entropy rate profiles (tertiary structure)

### Problem

Moirai can tell you *what* agents do (step sequences) and *where* they diverge (divergence points). It can't tell you *how* agents behave over time — whether they're methodical, erratic, stuck in loops, or shifting strategies. The "rhythm" of a trajectory is invisible.

### Minimum sample size

Conditional entropy with context window k requires enough runs per cluster to populate context-step distributions.

**Adaptive context:** Use k=3 when the cluster has >= 200 runs with known outcomes. Fall back to k=1 (unconditional column entropy split by outcome) for smaller clusters. k=1 requires no context estimation and works reliably at any sample size — it's 90% of the value (showing where successful vs failing runs differ in predictability) without the sparsity risk.

### Design

**What it computes:** A per-column conditional entropy curve showing how predictable the agent's behavior is at each point in the aligned trajectory.

**Algorithm:**

1. Take an existing alignment (reuse, don't recompute) and group runs by cluster.

2. Within each cluster, for each aligned column i, compute the conditional entropy of the step at column i given a context window of the previous k columns:

   ```
   H(column_i | context) = -Σ P(context) × Σ P(step | context) × log₂(P(step | context))
   ```

   The conditional probabilities are estimated empirically from all runs in the cluster. Contexts with fewer than 2 observations are dropped (insufficient data).

3. Produce an entropy profile: a vector of conditional entropies, one per aligned column.

4. Split by outcome: compute separate entropy profiles for successful and failing runs within the cluster.

5. Compute the "discipline gap" at each column: `failure_entropy[i] - success_entropy[i]`. Positive values mean failing runs are less predictable at this position. Large discipline gaps identify positions where successful agents follow a consistent pattern while failing agents flail.

**Key distinction from divergence analysis:** Divergence asks "do runs split here?" Entropy rate asks "is behavior predictable here?" A position can have low divergence (everyone does roughly the same thing) but high entropy (the choice is unpredictable given context). Or high divergence (runs split) but low entropy in each branch (each branch is internally consistent).

**Computational cost:** O(C × N) for building empirical distributions, plus O(C × min(N, V^k)) for entropy computation. At C=110, N=500: under 10ms. The V^k factor is theoretical — you iterate over observed contexts (bounded by N), not all possible contexts. k=3 is trivially fast at any projected scale.

### Output

```python
@dataclass
class EntropyProfile:
    cluster_id: int
    context_k: int                        # actual k used (1 or 3, adaptive)
    columns: list[int]
    entropy_values: list[float]           # conditional entropy per column
    success_entropy: list[float | None]   # entropy among successful runs only
    failure_entropy: list[float | None]   # entropy among failing runs only
    discipline_gap: list[float | None]    # failure_entropy - success_entropy
```

### CLI

```bash
moirai rhythm path/to/runs/           # entropy profiles per cluster
```

In the HTML report (`moirai branch --html`), entropy profiles can be rendered as sparklines alongside the alignment matrix. Columns with large discipline gaps get highlighted.

### Where it lives

- New `moirai/analyze/rhythm.py`. Takes an `Alignment` and list of `Run` objects.
- `EntropyProfile` dataclass in `moirai/schema.py`.

---

## Implementation sequence

The staged approach, in dependency order:

### Stage 1: Statistical foundation
- Create `moirai/analyze/stats.py` with `benjamini_hochberg()` and consolidated `fishers_exact_2x2()`
- Create `moirai/analyze/pipeline.py` for multi-module BH orchestration
- Add `q_value` and `risk_difference` fields to existing `Motif` and `DivergencePoint`
- Move `Motif` dataclass to `schema.py`
- Retroactively apply BH correction to `find_motifs()` and `find_divergence_points()`
- Update ranking to use q-value + risk difference
- Add `--permutation-test` diagnostic (validates the full BH pool, not just motifs)
- This is a bug fix to the existing system, independent of new features

### Stage 2: Gapped motifs (Section 1)
- Add `find_gapped_motifs()` to `motifs.py`
- Add `GappedMotif` to `schema.py`, define `PatternResult` protocol
- Extend `moirai patterns` CLI with `--gapped` and `--max-length` flags
- Integrate with `pipeline.py` for BH pooling
- Validate on real eval-harness and SWE-smith data before proceeding

### Stage 3: Positional MI (Section 2)
- Create `moirai/analyze/coupling.py`
- Add `PositionalMI` to `schema.py`
- Add `moirai mi` command and `--mi` flag to `moirai branch`
- Integrate with `pipeline.py` BH pool
- Requires N >= 500 runs; independent of Stage 2

### Stage 4: Entropy rate profiles (Section 4)
- Create `moirai/analyze/rhythm.py`
- Add `EntropyProfile` to `schema.py`
- Add `moirai rhythm` command
- Adaptive k (1 or 3) based on cluster size
- Independent of Stages 2-3; can be built in parallel with Stage 3

### Stage 5: Domain architecture (Section 3)
- Create `moirai/analyze/domains.py`
- Add `DomainArchitecture`, `StrategyCluster` to `schema.py`
- Add `moirai strategies` command
- Parameterize `_nw_align()` scoring function for motif-level alignment
- Depends on Stage 2 (needs validated motifs as input)
- Build only after Stage 2 motifs are validated on real data

### Follow-up work (not in scope)
- Update `recommend.py` to consume `GappedMotif`, `PositionalMI`, `EntropyProfile`
- Update `narrate.py` to incorporate MI couplings into findings
- HTML viz renderers for MI heatmaps, entropy sparklines, domain architecture timelines

## Where the analogy holds and where it breaks

**Holds well:**
- Sequences have multiple levels of organization, each with natural tools
- Local structural motifs (secondary structure) have internal flexibility and combine into global architectures (tertiary structure)
- Conservation across successful runs reveals functional constraints
- Coupled positions (coevolution / MI) reveal structural relationships invisible at the individual-position level

**Breaks:**
- Biological sequences evolved over millions of years with natural selection. Agent trajectories are generated in minutes by a single system. There's no "evolution" between runs — each starts fresh.
- DNA has a 4-letter alphabet that's fixed. The enriched step vocabulary is open-ended and hierarchical. The "alphabet" itself is a design choice.
- Biological function is determined by 3D structure, not just sequence. Agent success might be more directly readable from the sequence (or it might not — that's an empirical question).
- No inheritance. Runs don't descend from each other (though model weights are shared, which is loosely analogous to a shared genome).

The analogy is most productive as a source of algorithms and analytical frames, not as a literal mapping. The framework earns its keep if it surfaces patterns that step-level analysis misses.
