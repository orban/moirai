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

The existing duplicate Fisher's exact test implementations in `motifs.py` and `divergence.py` should be consolidated into `stats.py` as well.

**Pooling across test families:** BH correction pools all p-values from all test types (contiguous n-grams, gapped motifs, divergence points, MI pairs) into a single correction. This is valid under PRDS (positive regression dependency on subsets), which holds for tests sharing anchors or runs (Benjamini & Yekutieli 2001). Document this assumption.

### Permutation test validation

`moirai patterns --permutation-test N` shuffles outcome labels (success/failure) N times (default 1000), reruns the full discovery pipeline each time, and reports empirical FDR. This validates whether the pipeline produces signal or noise on a specific dataset.

This is a diagnostic, not a filter. It's expensive (N × full pipeline), so it's opt-in. The label-shuffle null is correct for validating Fisher's exact tests (it directly mirrors the null hypothesis). Sequence-shuffle tests a different question ("are patterns structurally surprising?") and could be added later as a separate diagnostic.

### Ranking

All patterns (contiguous and gapped motifs, MI pairs) are ranked by BH-adjusted q-value ascending. Tiebreaker: risk difference (`success_rate - baseline_rate`), which directly answers "how much does this pattern change success probability" and is bounded and interpretable.

### Success rate convention

Throughout this design, `success_rate = success_runs / (success_runs + fail_runs)`. The denominator excludes runs with unknown outcomes. `total_runs` is the count of all matching runs including unknowns. Both fields are present on all pattern dataclasses to avoid ambiguity.

---

## Section 1: Gapped motif patterns (secondary structure)

### Problem

`motifs.py` finds contiguous n-grams: `read(test_file) → edit(source) → test`. It can't find `read(test_file) → ... → edit(source) → ... → test` where the `...` varies across runs. The most important behavioral patterns — "always read the test file before editing" — have flexible gaps.

### Design

**Pattern representation:** An ordered subsequence of anchor steps with unconstrained gaps. Pattern `(A, B, C)` matches a run if there exist positions i < j < k where step[i] = A, step[j] = B, step[k] = C. No gap bounds are learned or enforced.

Descriptive statistics about gap lengths (median gap between each anchor pair) are computed for reporting but not used for matching or filtering.

**Matching semantics:** Greedy left-to-right scan. For pattern `(A, B, C)`: find the first A, then the first B after that A, then the first C after that B. This is deterministic and O(L) per pattern per run. When step types repeat (e.g., `read(source)` appears 10 times), greedy matching uses the earliest valid assignment.

**Frequency filter:** Only step types appearing in >= 5% of runs are "frequent" and enter the candidate pool. Report the actual number of tests conducted alongside results.

### Discovery algorithm

1. **Build frequency index.** For each of V enriched step types, count runs containing it. Drop types below max(min_count, 5% of N).

2. **Enumerate ordered pairs.** All (A, B) from frequent types where A appears before B in >= min_count runs. Compute Fisher's exact p-value for each pair's association with outcome.

3. **Enumerate ordered triples.** All (A, B, C) from frequent types as ordered subsequences in >= min_count runs. Fisher's exact test.

4. **Optionally enumerate length-4** (opt-in via `--max-length 4`). Uses prefix pruning: only track a triple in the length-4 pass if that triple survived min_count at stage 3. This reduces the inner loop by ~90%.

5. **BH correction.** Pool all raw p-values (gapped + contiguous + divergence + MI) and apply BH at q = 0.05.

6. **Prune.** After BH correction, drop patterns that are strict subsequences of longer patterns, but only if the longer pattern covers the same or more runs AND has an equal or better q-value. If the shorter pattern covers more runs (is more general), keep both. Check transitive subsequence relations (a length-2 can be pruned by a length-4 directly). Pruning is display-only — no re-computation of q-values after pruning.

7. **Rank** by q-value ascending, risk difference as tiebreaker.

Default max length: 3 (runs in under 1 second at current scale). Length-4 is opt-in due to higher computational cost (~5s at N=165/V=20, ~2 min at N=500/V=30 without prefix pruning, much less with it).

### Counting optimization

**Single-pass per run.** Walk the step sequence maintaining:
- `seen_types: set[int]` — types encountered so far (for pairs)
- `seen_pairs: set[tuple[int, int]]` — completed pairs (for triples)  
- `seen_triples: set[tuple[int, int, int]]` — completed triples (for length-4, opt-in)
- `run_pairs_seen: set[tuple[int, int]]` — deduplication per run
- `run_triples_seen: set[tuple[int, int, int]]` — deduplication per run

At each step with type z:
- For each t in `seen_types`: if (t, z) not in `run_pairs_seen`, increment pair count, add to `run_pairs_seen`, add to `seen_pairs`
- For each (t, u) in `seen_pairs`: if (t, u, z) not in `run_triples_seen`, increment triple count, add to `run_triples_seen`, add to `seen_triples`
- For length-4: for each (t, u, v) in `seen_triples` where (t, u, v) is in `surviving_triples`: emit (t, u, v, z)

**Map types to integer IDs** early (in stage 1). Use integer tuples throughout for faster hashing.

Total complexity:
- Pairs: O(N × L × V). ~132K ops at current scale. Negligible.
- Triples: O(N × L × V²). ~2.6M ops. Under 1s.
- Length-4 with prefix pruning: O(N × L × |surviving_triples|). Depends on data, typically a few seconds.

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
    p_value: float | None        # raw Fisher's exact
    q_value: float | None        # BH-adjusted
    avg_first_position: float    # mean normalized position of first anchor across matching runs
    median_gaps: list[float]     # median gap between each adjacent anchor pair (descriptive)

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
- BH correction retroactively applied to existing `find_motifs()` and `find_divergence_points()`

---

## Section 2: Positional mutual information (primary → secondary bridge)

### Problem

Divergence analysis treats each aligned column independently. But agent decisions are coupled: reading the test file at step 10 might determine whether the agent loops at step 30. These long-range couplings are invisible to column-by-column analysis.

### Design

**What it computes:** For each pair of aligned columns (i, j), the mutual information between step values at those positions across all runs. High MI means knowing the agent's choice at position i tells you what it'll do at position j.

**Algorithm:**

1. Take an existing alignment from `align_runs()`.

2. For each column pair (i, j) where i < j, compute MI from the joint distribution of step values:

   ```
   MI(i, j) = Σ P(x,y) × log₂(P(x,y) / (P(x) × P(y)))
   ```

   where x and y are step values at columns i and j across all runs, excluding runs with GAP in either column.

3. Normalize: `NMI(i, j) = MI(i, j) / min(H(i), H(j))`, giving NMI in [0, 1].

4. Filter: keep pairs where NMI > threshold (default 0.3) and both columns have >= 2 distinct non-GAP values.

5. For each high-MI pair, test whether the *combination* of values at (i, j) predicts success better than either column alone. Chi-squared test on the (value_i × value_j × outcome) contingency table. P-value feeds into the shared BH correction pool.

**Computational cost:** O(C² × N) where C is alignment columns (~50-110), N is runs. At C=110, N=165: ~1M operations. Negligible.

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
    outcome_q_value: float | None     # BH-adjusted
```

### CLI

```bash
moirai mi path/to/runs/              # show high-MI column pairs
moirai branch path/to/runs/ --mi     # annotate divergence points with MI couplings
```

### Where it lives

`positional_mutual_information()` in new `moirai/analyze/coupling.py`. Takes an `Alignment` and list of `Run` objects.

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

5. Compute pairwise domain-architecture distance using Needleman-Wunsch at the motif level. Match = same motif ID, mismatch/gap scored the same way as step-level alignment but operating on motif IDs instead.

6. Cluster runs by domain-architecture distance. These are "strategy clusters."

7. Compare strategy clusters to existing step-level clusters. Report where they agree and disagree.

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

New `moirai/analyze/domains.py`. Depends on `motifs.py` (Section 1) and `align.py`.

---

## Section 4: Entropy rate profiles (tertiary structure)

### Problem

Moirai can tell you *what* agents do (step sequences) and *where* they diverge (divergence points). It can't tell you *how* agents behave over time — whether they're methodical, erratic, stuck in loops, or shifting strategies. The "rhythm" of a trajectory is invisible.

### Design

**What it computes:** A per-column conditional entropy curve showing how predictable the agent's behavior is at each point in the aligned trajectory.

**Algorithm:**

1. Take an existing alignment and group runs by cluster.

2. Within each cluster, for each aligned column i, compute the conditional entropy of the step at column i given a context window of the previous k columns (default k=3):

   ```
   H(column_i | context) = -Σ P(context) × Σ P(step | context) × log₂(P(step | context))
   ```

   The conditional probabilities are estimated empirically from all runs in the cluster. Contexts with fewer than 2 observations are dropped (insufficient data).

3. Produce an entropy profile: a vector of conditional entropies, one per aligned column.

4. Split by outcome: compute separate entropy profiles for successful and failing runs within the cluster.

5. Compute the "discipline gap" at each column: `failure_entropy[i] - success_entropy[i]`. Positive values mean failing runs are less predictable at this position. Large discipline gaps identify positions where successful agents follow a consistent pattern while failing agents flail.

**Key distinction from divergence analysis:** Divergence asks "do runs split here?" Entropy rate asks "is behavior predictable here?" A position can have low divergence (everyone does roughly the same thing) but high entropy (the choice is unpredictable given context). Or high divergence (runs split) but low entropy in each branch (each branch is internally consistent).

**Computational cost:** O(C × N × V^k). At C=110, N=165, V=20, k=3: ~145M with hash-based context lookup. A few seconds. Reducing k to 2 makes it trivially fast. k=3 is the default; k=2 available for large datasets.

### Output

```python
@dataclass
class EntropyProfile:
    cluster_id: int
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

New `moirai/analyze/rhythm.py`. Takes an `Alignment` and list of `Run` objects.

---

## Implementation sequence

The staged approach, in dependency order:

### Stage 1: Statistical foundation
- Create `moirai/analyze/stats.py` with `benjamini_hochberg()` and consolidated `fishers_exact_2x2()`
- Retroactively apply BH correction to `find_motifs()` and `find_divergence_points()`
- Update ranking to use q-value + risk difference
- This is a bug fix to the existing system, independent of new features

### Stage 2: Gapped motifs (Section 1)
- Add `find_gapped_motifs()` to `motifs.py`
- Extend `moirai patterns` CLI with `--gapped` and `--max-length` flags
- Add `--permutation-test` diagnostic
- Validate on real eval-harness and SWE-smith data before proceeding

### Stage 3: Positional MI (Section 2)
- Create `moirai/analyze/coupling.py`
- Add `moirai mi` command and `--mi` flag to `moirai branch`
- Independent of Stage 2; can be built in parallel

### Stage 4: Entropy rate profiles (Section 4)
- Create `moirai/analyze/rhythm.py`
- Add `moirai rhythm` command
- Independent of Stages 2-3; can be built in parallel with Stage 3

### Stage 5: Domain architecture (Section 3)
- Create `moirai/analyze/domains.py`
- Add `moirai strategies` command
- Depends on Stage 2 (needs validated motifs as input)
- Build only after Stage 2 motifs are validated on real data

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
