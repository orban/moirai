#!/usr/bin/env python3
"""Simpson's paradox test for structural concordance.

Tests whether the whole-dataset negative concordance (tau < 0) is an artifact
of cluster composition rather than a real within-cluster behavioral signal.

Method:
  1. Marginal concordance: align ALL runs, compute distance from consensus,
     Kendall's Tau-b between negated distance and outcome.
  2. Within-cluster concordance: cluster runs (threshold=0.3), compute tau
     within each cluster that has mixed outcomes and >= 5 runs.
  3. Cluster-weighted average: weight each within-cluster tau by n_runs/total.
  4. If marginal tau is strongly negative but weighted average is near zero,
     the negative correlation is driven by cluster composition (hard tasks
     cluster together), not by within-cluster behavioral patterns.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

# Allow imports from moirai
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import functools
import time

# Force unbuffered output so progress shows up in real time
print = functools.partial(print, flush=True)  # type: ignore[assignment]

import numpy as np
from scipy.spatial.distance import squareform
from scipy.stats import kendalltau

from moirai.analyze.align import align_runs, _consensus, distance_matrix
from moirai.analyze.cluster import cluster_runs
from moirai.load import load_runs
from moirai.schema import GAP

# Progressive alignment is O(n^2) in NW calls. Cap marginal alignment
# at this many runs; use stratified sampling to preserve pass/fail ratio.
MAX_ALIGN_RUNS = 100


# ── helpers ──────────────────────────────────────────────────────────────


def distances_from_consensus(alignment) -> list[float]:
    """Compute normalized distance from consensus for each aligned run."""
    if not alignment.matrix or not alignment.matrix[0]:
        return []

    cons = _consensus(alignment.matrix)
    n_cols = len(cons)
    if n_cols == 0:
        return []

    dists = []
    for row in alignment.matrix:
        mismatches = sum(1 for a, b in zip(row, cons) if a != b)
        dists.append(mismatches / n_cols)
    return dists


def mean_pairwise_distances(runs, level="type") -> list[float]:
    """Mean pairwise trajectory distance per run (from condensed matrix).

    This is a proxy for "atypicality": runs far from the average are
    structurally unusual.  Avoids the O(n^2) progressive alignment
    but still uses the O(n^2) pairwise NW distances that cluster_runs
    also computes.
    """
    n = len(runs)
    if n < 2:
        return [0.0] * n

    condensed = distance_matrix(runs, level=level)
    full = squareform(condensed)  # (n, n) symmetric matrix
    return list(full.mean(axis=1))


def compute_tau(distances: list[float], outcomes: list[float]) -> tuple[float, float | None]:
    """Kendall tau-b between negated distance (typicality) and outcome."""
    if len(distances) < 5:
        return 0.0, None
    if len(set(distances)) < 2 or len(set(outcomes)) < 2:
        return 0.0, None

    result = kendalltau([-d for d in distances], outcomes, variant="b")
    tau_val = float(result.statistic)
    p_val = float(result.pvalue)
    if np.isnan(tau_val):
        return 0.0, None
    return tau_val, p_val


def _stratified_sample(runs, max_n, seed=42):
    """Sample up to max_n runs, preserving pass/fail ratio."""
    rng = np.random.default_rng(seed)
    passes = [r for r in runs if r.result.success]
    fails = [r for r in runs if not r.result.success]

    n_pass = int(max_n * len(passes) / len(runs))
    n_fail = max_n - n_pass

    # Clamp
    n_pass = min(n_pass, len(passes))
    n_fail = min(n_fail, len(fails))

    sampled_passes = list(rng.choice(passes, size=n_pass, replace=False)) if n_pass else []
    sampled_fails = list(rng.choice(fails, size=n_fail, replace=False)) if n_fail else []
    return sampled_passes + sampled_fails


def marginal_concordance(runs, level="type"):
    """Compute concordance across ALL runs (ignoring cluster structure).

    Uses mean pairwise distance as the atypicality measure. This avoids
    the expensive progressive alignment step while computing the same
    underlying O(n^2) pairwise NW distances.

    For datasets > MAX_ALIGN_RUNS, we stratified-sample down.
    """
    known = [r for r in runs if r.result.success is not None]
    if len(known) < 5:
        return None, None, 0

    has_success = any(r.result.success for r in known)
    has_failure = any(not r.result.success for r in known)
    if not has_success or not has_failure:
        return None, None, len(known)

    sampled = False
    used = known
    if len(known) > MAX_ALIGN_RUNS:
        used = _stratified_sample(known, MAX_ALIGN_RUNS)
        sampled = True
        print(f"    (sampled {len(used)} of {len(known)} runs for marginal computation)")

    t0 = time.time()
    dists = mean_pairwise_distances(used, level=level)
    elapsed = time.time() - t0
    print(f"    (pairwise distances: {elapsed:.1f}s for {len(used)} runs)")

    if not dists or len(set(dists)) < 2:
        return None, None, len(used)

    outcomes = [1.0 if r.result.success else 0.0 for r in used]
    tau, p = compute_tau(dists, outcomes)
    return tau, p, len(used)


def within_cluster_concordance(runs, level="type", threshold=0.3):
    """Compute per-cluster concordance and the cluster-weighted average."""
    t0 = time.time()
    cr = cluster_runs(runs, level=level, threshold=threshold)
    print(f"    (clustering: {time.time() - t0:.1f}s)")

    # Group runs by cluster
    by_cluster: dict[int, list] = defaultdict(list)
    for run in runs:
        cid = cr.labels.get(run.run_id)
        if cid is not None:
            by_cluster[cid].append(run)

    # Identify eligible clusters first
    eligible: list[tuple[int, list]] = []
    for cid, cluster_runs_list in sorted(by_cluster.items()):
        known = [r for r in cluster_runs_list if r.result.success is not None]
        if len(known) < 5:
            continue
        has_success = any(r.result.success for r in known)
        has_failure = any(not r.result.success for r in known)
        if not has_success or not has_failure:
            continue
        eligible.append((cid, known))

    print(f"    ({len(eligible)} eligible clusters to align...)")

    cluster_taus = []
    total_eligible = 0

    for i, (cid, known) in enumerate(eligible):
        t1 = time.time()
        alignment = align_runs(known, level=level)
        dists = distances_from_consensus(alignment)
        if not dists or len(set(dists)) < 2:
            continue

        outcomes = [1.0 if r.result.success else 0.0 for r in known]
        tau, p = compute_tau(dists, outcomes)

        n = len(known)
        success_rate = sum(outcomes) / n
        cluster_taus.append({
            "cluster_id": cid,
            "tau": tau,
            "p_value": p,
            "n_runs": n,
            "success_rate": success_rate,
            "prototype": cr.clusters[cid].prototype[:60] if cid < len(cr.clusters) else "?",
        })
        total_eligible += n
        elapsed = time.time() - t1
        if elapsed > 2:
            print(f"      cluster {cid}: {n} runs, {elapsed:.1f}s")

    # Cluster-weighted average tau
    if total_eligible > 0:
        weighted_tau = sum(c["tau"] * c["n_runs"] / total_eligible for c in cluster_taus)
    else:
        weighted_tau = None

    return cluster_taus, weighted_tau, total_eligible, cr


def within_family_concordance(runs, level="type"):
    """Compute per-task_family concordance (eval_harness decomposition)."""
    by_family: dict[str, list] = defaultdict(list)
    for r in runs:
        fam = r.task_family or r.task_id.rsplit("-", 1)[0]
        by_family[fam].append(r)

    # Identify eligible families first
    eligible: list[tuple[str, list]] = []
    for fam, fam_runs in sorted(by_family.items()):
        known = [r for r in fam_runs if r.result.success is not None]
        if len(known) < 5:
            continue
        has_success = any(r.result.success for r in known)
        has_failure = any(not r.result.success for r in known)
        if not has_success or not has_failure:
            continue
        eligible.append((fam, known))

    print(f"    ({len(eligible)} eligible families to align...)")

    family_taus = []
    total_eligible = 0

    for fam, known in eligible:
        t1 = time.time()
        alignment = align_runs(known, level=level)
        dists = distances_from_consensus(alignment)
        if not dists or len(set(dists)) < 2:
            continue

        outcomes = [1.0 if r.result.success else 0.0 for r in known]
        tau, p = compute_tau(dists, outcomes)

        n = len(known)
        success_rate = sum(outcomes) / n
        family_taus.append({
            "family": fam,
            "tau": tau,
            "p_value": p,
            "n_runs": n,
            "success_rate": success_rate,
        })
        total_eligible += n
        elapsed = time.time() - t1
        if elapsed > 2:
            print(f"      family {fam}: {n} runs, {elapsed:.1f}s")

    if total_eligible > 0:
        weighted_tau = sum(f["tau"] * f["n_runs"] / total_eligible for f in family_taus)
    else:
        weighted_tau = None

    return family_taus, weighted_tau, total_eligible


# ── main ─────────────────────────────────────────────────────────────────


DATASETS = {
    "eval_harness": Path("/Users/ryo/dev/moirai/examples/eval_harness"),
    "swe_smith": Path("/Users/ryo/dev/moirai/examples/swe_smith"),
    "swe_agent": Path("/Users/ryo/dev/moirai/examples/swe_agent"),
}

LEVEL = "type"


def fmt_tau(tau, p=None):
    if tau is None:
        return "  n/a"
    s = f"{tau:+.4f}"
    if p is not None and p < 0.05:
        s += " *"
    if p is not None and p < 0.01:
        s += "*"
    return s


def run_dataset(name, path):
    print(f"\n{'=' * 72}")
    print(f"  DATASET: {name}")
    print(f"  path: {path}")
    print(f"{'=' * 72}")

    runs, warnings = load_runs(path)
    if warnings:
        print(f"  (skipped {len(warnings)} files with warnings)")
    print(f"  loaded {len(runs)} runs")

    known = [r for r in runs if r.result.success is not None]
    n_success = sum(1 for r in known if r.result.success)
    n_fail = len(known) - n_success
    print(f"  outcomes: {n_success} pass / {n_fail} fail ({100*n_success/len(known):.1f}% pass rate)")

    # 1. Marginal concordance
    print(f"\n  ── Marginal concordance (all runs pooled) ──")
    m_tau, m_p, m_n = marginal_concordance(runs, level=LEVEL)
    print(f"  tau = {fmt_tau(m_tau, m_p)}   (n={m_n})")

    # 2. Within-cluster concordance
    print(f"\n  ── Within-cluster concordance (threshold=0.3) ──")
    cluster_taus, weighted_tau, total_elig, cr = within_cluster_concordance(
        runs, level=LEVEL, threshold=0.3,
    )
    print(f"  {len(cr.clusters)} clusters total, {len(cluster_taus)} with mixed outcomes & n>=5")
    print()

    if cluster_taus:
        print(f"  {'Cluster':>8} {'n':>5} {'pass%':>6} {'tau':>9} {'p':>9}")
        print(f"  {'─' * 8} {'─' * 5} {'─' * 6} {'─' * 9} {'─' * 9}")
        for c in cluster_taus:
            p_str = f"{c['p_value']:.4f}" if c["p_value"] is not None else "  n/a"
            print(f"  {c['cluster_id']:>8d} {c['n_runs']:>5d} {100*c['success_rate']:>5.1f}% {fmt_tau(c['tau'], c['p_value']):>9} {p_str:>9}")
        print()
        print(f"  cluster-weighted avg tau = {fmt_tau(weighted_tau)}   (n={total_elig})")
    else:
        print(f"  (no eligible clusters)")

    # 3. Comparison
    print(f"\n  ── Simpson's paradox test ──")
    if m_tau is not None and weighted_tau is not None:
        gap = m_tau - weighted_tau
        print(f"  marginal tau:        {fmt_tau(m_tau)}")
        print(f"  weighted-cluster tau:{fmt_tau(weighted_tau)}")
        print(f"  gap (marginal - wt): {gap:+.4f}")
        print()

        if abs(m_tau) > 0.05 and abs(gap) > abs(m_tau) * 0.5:
            print(f"  ** SIMPSON'S PARADOX DETECTED **")
            print(f"  The marginal tau ({m_tau:+.4f}) is substantially different from the")
            print(f"  cluster-weighted average ({weighted_tau:+.4f}).")
            if m_tau < -0.05 and weighted_tau > -0.03:
                print(f"  The negative marginal correlation is driven by cluster composition")
                print(f"  (hard tasks cluster together), not by within-cluster behavior.")
            elif m_tau < -0.05 and weighted_tau > m_tau:
                print(f"  Cluster composition amplifies the negative signal. The within-cluster")
                print(f"  signal is weaker than the marginal one.")
        else:
            print(f"  No strong paradox: marginal and within-cluster taus are consistent.")
    else:
        print(f"  (insufficient data for comparison)")

    return {
        "name": name,
        "n_runs": len(runs),
        "marginal_tau": m_tau,
        "marginal_p": m_p,
        "weighted_cluster_tau": weighted_tau,
        "n_clusters_eligible": len(cluster_taus),
        "cluster_taus": cluster_taus,
    }


def main():
    print("Simpson's paradox test: does cluster composition drive negative concordance?")
    print("=" * 72)

    results = {}
    for name, path in DATASETS.items():
        if not path.exists():
            print(f"\n  SKIP {name}: {path} not found")
            continue
        results[name] = run_dataset(name, path)

    # Task-family decomposition for eval_harness
    if "eval_harness" in results:
        print(f"\n\n{'=' * 72}")
        print(f"  EVAL_HARNESS: task-family decomposition")
        print(f"{'=' * 72}")

        runs, _ = load_runs(DATASETS["eval_harness"])
        family_taus, fam_weighted, fam_total = within_family_concordance(runs, level=LEVEL)
        m_tau = results["eval_harness"]["marginal_tau"]

        print(f"\n  {len(family_taus)} families with mixed outcomes & n>=5")
        print()

        if family_taus:
            print(f"  {'Family':<40} {'n':>5} {'pass%':>6} {'tau':>9} {'p':>9}")
            print(f"  {'─' * 40} {'─' * 5} {'─' * 6} {'─' * 9} {'─' * 9}")
            for f in sorted(family_taus, key=lambda x: x["tau"]):
                p_str = f"{f['p_value']:.4f}" if f["p_value"] is not None else "  n/a"
                print(f"  {f['family']:<40} {f['n_runs']:>5d} {100*f['success_rate']:>5.1f}% {fmt_tau(f['tau'], f['p_value']):>9} {p_str:>9}")
            print()
            print(f"  family-weighted avg tau = {fmt_tau(fam_weighted)}   (n={fam_total})")
            print()
            if m_tau is not None and fam_weighted is not None:
                gap = m_tau - fam_weighted
                print(f"  marginal tau:       {fmt_tau(m_tau)}")
                print(f"  family-weighted tau:{fmt_tau(fam_weighted)}")
                print(f"  gap:                {gap:+.4f}")

    # Summary table
    print(f"\n\n{'=' * 72}")
    print(f"  SUMMARY TABLE")
    print(f"{'=' * 72}\n")
    print(f"  {'Dataset':<16} {'n':>5} {'Marginal tau':>13} {'Wtd-cluster tau':>16} {'Gap':>8}")
    print(f"  {'─' * 16} {'─' * 5} {'─' * 13} {'─' * 16} {'─' * 8}")
    for name, r in results.items():
        m = fmt_tau(r["marginal_tau"])
        w = fmt_tau(r["weighted_cluster_tau"])
        gap_str = ""
        if r["marginal_tau"] is not None and r["weighted_cluster_tau"] is not None:
            gap_str = f"{r['marginal_tau'] - r['weighted_cluster_tau']:+.4f}"
        print(f"  {name:<16} {r['n_runs']:>5} {m:>13} {w:>16} {gap_str:>8}")

    # Interpretation
    print(f"\n\n{'=' * 72}")
    print(f"  INTERPRETATION")
    print(f"{'=' * 72}\n")

    any_paradox = False
    for name, r in results.items():
        m = r["marginal_tau"]
        w = r["weighted_cluster_tau"]
        if m is not None and w is not None:
            gap = m - w
            if abs(m) > 0.05 and abs(gap) > abs(m) * 0.5:
                any_paradox = True

    if any_paradox:
        print("  The marginal (whole-dataset) tau and the cluster-weighted tau diverge")
        print("  substantially for at least one dataset. This confirms Simpson's paradox:")
        print("  the negative correlation between structural distance and outcome is")
        print("  primarily an artifact of cluster composition. Runs on hard tasks tend to")
        print("  cluster together (similar failing strategies), while easy-task runs form")
        print("  separate clusters (similar succeeding strategies). When you pool them,")
        print("  you see a negative correlation, but within each cluster the signal is")
        print("  much weaker or absent.")
        print()
        print("  Practical implication: concordance should always be computed within")
        print("  clusters, never on the whole dataset. The marginal tau is misleading.")
    else:
        print("  The marginal and within-cluster taus are broadly consistent across")
        print("  datasets. Simpson's paradox is not strongly present -- the negative")
        print("  concordance reflects a genuine within-cluster pattern, not just")
        print("  cluster composition effects.")


if __name__ == "__main__":
    main()
