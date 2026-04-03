---
title: "Concordance scoring — structural predictiveness per cluster"
type: feat
date: 2026-03-31
origin: "Muslimani et al., 'Analyzing Reward Functions via Trajectory Alignment' (NeurIPS 2024 BML Workshop)"
---

# Concordance scoring

## The question

moirai clusters runs by structural similarity, then tests whether branches predict outcomes. But it never asks the prior question: **does structural similarity itself predict outcomes in this cluster?**

A cluster where structurally typical runs succeed and structurally atypical ones fail is a cluster where moirai's analysis is trustworthy. A cluster where structure and outcome are uncorrelated is one where tool-call sequences alone don't explain what's happening — the agent's reasoning, file content, or environment state matters more.

Concordance scoring gives moirai a self-diagnostic: a per-cluster measure of how much its structural lens actually captures.

## Origin

Muslimani et al. propose the *Trajectory Alignment Coefficient*, a Kendall's Tau-b between two rankings of trajectories: one induced by a reward function, one by human preferences. They find that 94% of human-designed reward functions are misaligned, and misalignment is worst for failing trajectories (low-return violin plots shift left in Figure 2b).

The transferable insight: two different ways of scoring trajectories can disagree, and measuring that disagreement is itself useful. In moirai's case, the two rankings are:

1. **Structural distance from consensus** — how "typical" a run looks within its cluster
2. **Outcome quality** — success/failure, or a continuous score when available

Kendall's Tau-b between these rankings = the concordance score.

## What it measures

For each cluster, concordance answers: "do runs that look structurally like the cluster's consensus also tend to have similar outcomes?"

| Concordance (τ) | Interpretation |
|---|---|
| τ > 0.4 | Structure predicts outcomes well. Divergence points found in this cluster are likely meaningful. |
| 0.1 < τ < 0.4 | Moderate. Structure explains some variance but not all. |
| -0.1 < τ < 0.1 | Structure and outcome are unrelated. Divergence analysis in this cluster may not be informative. Content-level analysis needed. |
| τ < -0.1 | Structurally typical runs *fail more*. The "normal" behavior in this cluster is the failure mode. |

Negative concordance is the most interesting case — it means the dominant structural pattern in the cluster is actually a failure pattern. The successful runs are the structural outliers.

## Design

### Statistical primitive: `kendall_tau_b` in `stats.py`

Use `scipy.stats.kendalltau` (variant='b'). scipy is already a dependency.

Wrapper in `stats.py` adds:
- Guard for degenerate inputs (n < 5, all-same values, NaN from scipy)
- Typed return: `tuple[float, float | None]` (tau, p-value)
- `None` p-value means degenerate input (insufficient data), not "test failed"

```python
# moirai/analyze/stats.py

def kendall_tau_b(x: list[float], y: list[float]) -> tuple[float, float | None]:
    """Kendall's Tau-b rank correlation.

    Returns (tau, p_value). Returns (0.0, None) for degenerate inputs
    (n < 5, no variation in x or y, or scipy returns NaN).
    """
```

### Concordance function in `cluster.py`

No new module. The function lives in `cluster.py` alongside `cluster_runs` — it's a per-cluster metric.

```python
def compute_concordance(
    runs: list[Run],
    cluster_labels: dict[str, int],
    level: str = "name",
) -> dict[int, ConcordanceScore]:
    """Compute structural concordance for each cluster.

    For each cluster with mixed outcomes and >= 5 runs:
    1. Align runs within the cluster (progressive NW at enriched-name level)
    2. Build consensus from alignment matrix
    3. Count mismatches between each run's aligned row and the consensus row
    4. Rank runs by mismatch count (fewer mismatches = more typical)
    5. Rank runs by outcome (result.score if available, else binary 0/1)
    6. Compute Kendall's Tau-b between the two rankings
    """
```

**Distance from consensus** uses the progressive alignment matrix directly. `align_runs` returns an `Alignment` with a matrix where each row is an aligned run. `_consensus` gives the consensus row. Count mismatches between each run's row and the consensus row, dividing by alignment length. This avoids the bug of re-aligning each run independently (which produces per-run alignments of different lengths and inconsistent normalization denominators).

**Outcome ranking** uses `result.score` when populated (continuous, higher = better). Falls back to binary `result.success` (True=1.0, False=0.0). Runs with `result.success is None` are excluded.

### Schema type

```python
@dataclass
class ConcordanceScore:
    tau: float                    # Kendall's Tau-b, in [-1, 1]
    p_value: float | None         # significance (normal approximation); None = degenerate input
    n_runs: int                   # runs used (after excluding unknown outcomes)
    used_continuous: bool         # True if result.score was used, False if binary
```

No `cluster_id` field — returned as `dict[int, ConcordanceScore]` where the key is the cluster ID.

### Integration: `moirai clusters --concordance`

Opt-in via `--concordance` flag. Computing concordance requires per-cluster alignment (O(N² × L²) per cluster), which `moirai clusters` doesn't currently pay. This effectively doubles runtime, so it should be explicit.

**Terminal output** adds one line per cluster:

```
Cluster 3: 47 runs, 19% success, concordance: τ=0.62 (p=0.003)
  verify(pass)×3 → explore → modify → verify(pass)
  ...
```

When concordance is low (|τ| < 0.1):

```
Cluster 2: 35 runs, 17% success, concordance: τ=0.04 (p=0.81)
  ⚠ structure alone doesn't explain outcomes — reasoning or content analysis needed
```

When concordance is negative:

```
Cluster 1: 28 runs, 25% success, concordance: τ=-0.31 (p=0.04)
  ⚠ structurally typical runs fail more — the dominant pattern is a failure mode
```

**HTML output** adds concordance to cluster panels. Color-coded: green (τ > 0.3), yellow (0.1–0.3), red (|τ| < 0.1 or negative).

### Integration: `moirai branch`

Always-on — `branch` already pays the per-task alignment cost. Report concordance for each task's run population alongside divergence points.

```
ansible_ansible-85488: 12 runs (3P/9F), aligned to 42 columns, concordance: τ=0.55
  Position 15 (q=0.012)
    read(config): 3 runs, 100% success
    bash(python): 9 runs, 0% success
```

### NOT in scope

- **Multi-metric alignment** (pairwise Tau between tokens, latency, cost). Future work.
- **Content-level concordance** (semantic embeddings). Far future.
- **Causal interpretation**. Concordance is correlational.

## Algorithm detail

```
for each cluster:
  runs_in_cluster = [r for r in runs if labels[r.run_id] == cluster_id]
  known = [r for r in runs_in_cluster if r.result.success is not None]

  if len(known) < 5:
    skip  # too few for meaningful correlation

  successes = [r for r in known if r.result.success]
  failures = [r for r in known if not r.result.success]
  if not successes or not failures:
    skip  # no variation in outcome

  # Align within cluster using progressive alignment
  alignment = align_runs(known, level="name")
  consensus = _consensus(alignment.matrix)

  # Count mismatches from consensus using alignment matrix directly
  # (NOT re-aligning — that produces inconsistent alignment lengths)
  distances = []
  n_cols = len(consensus)
  for row in alignment.matrix:
    mismatches = sum(1 for a, b in zip(row, consensus) if a != b)
    distances.append(mismatches / n_cols)

  # Check for zero variance in distances (all runs identical)
  if len(set(distances)) < 2:
    skip  # no structural variation — tau is undefined

  # Outcome scores
  if all(r.result.score is not None for r in known):
    outcomes = [r.result.score for r in known]
    used_continuous = True
  else:
    outcomes = [1.0 if r.result.success else 0.0 for r in known]
    used_continuous = False

  # Kendall's Tau-b
  # Negate distances: closer to consensus = higher value = more typical
  # Positive tau then means "typical runs succeed more"
  tau, p = kendall_tau_b([-d for d in distances], outcomes)

  yield ConcordanceScore(tau, p, len(known), used_continuous)
```

## Known limitations

1. **Consensus bias toward dominant outcome.** Consensus is majority-vote. In a cluster that's 80% failures, the consensus reflects the failure pattern. A positive tau in this case means "runs that look like the failing majority also fail" — which is true but not as useful as it sounds. The sign convention (positive = "typical runs succeed") can be misleading when the cluster is heavily skewed. Users should cross-reference tau with the cluster success rate.

2. **Unimodal assumption.** Consensus assumes the cluster has one dominant structural pattern. If a cluster has two distinct sub-patterns with similar outcomes, the consensus is a meaningless average of both, and distances from it are not informative. Sub-pattern detection (`cluster_subpatterns`) partially addresses this, but concordance doesn't account for it.

3. **Low power with binary outcomes.** When outcomes are binary (pass/fail) and skewed (e.g., 40 failures, 5 successes), Kendall's Tau-b handles ties correctly but statistical power is low. The p-value may not be meaningful even if tau looks large. The `used_continuous` flag helps downstream code decide how much to trust the result.

## File changes

| File | Change |
|---|---|
| `moirai/analyze/stats.py` | Add `kendall_tau_b()` wrapper (~15 lines) |
| `moirai/analyze/cluster.py` | Add `compute_concordance()` (~50 lines) |
| `moirai/schema.py` | Add `ConcordanceScore` dataclass (~6 lines) |
| `moirai/cli.py` | Add `--concordance` to `clusters`, always-on in `branch` (~15 lines each) |
| `moirai/viz/terminal.py` | Display concordance in `print_clusters` and branch output (~20 lines) |
| `moirai/viz/html.py` | Add concordance to HTML cluster panels (~10 lines) |
| `tests/test_cluster.py` | Add concordance tests (~80 lines) |
| `tests/test_stats.py` | Add `kendall_tau_b` tests (~30 lines) |

**Total**: ~230 lines new code, ~30 lines modified. No new modules, no new test files beyond extending existing ones.

## Acceptance criteria

- [ ] `moirai clusters runs/ --concordance` shows concordance for each cluster with mixed outcomes and >= 5 runs
- [ ] `moirai branch runs/` shows per-task concordance alongside divergence points (always-on)
- [ ] Concordance uses `result.score` when available, binary success otherwise
- [ ] Low-concordance clusters get a diagnostic message
- [ ] Negative-concordance clusters get a message about dominant failure mode
- [ ] Zero-variance distances (all runs identical) handled gracefully
- [ ] All existing tests continue to pass
- [ ] Tests cover perfect/zero/negative concordance, edge cases, binary vs continuous, and CLI integration
