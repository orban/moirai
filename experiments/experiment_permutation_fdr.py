#!/usr/bin/env python3
"""Permutation FDR validation for gapped and contiguous motifs.

Tests whether BH-corrected motif discoveries are real signal or inflated
by structural correlation between tests (gapped motifs are nested —
A→B is contained in A→B→C, so their test statistics are dependent).

Approach:
  1. Run find_gapped_motifs / find_motifs to get BH discoveries
  2. Build the boolean membership matrix (patterns × runs)
  3. Run permutation_fdr (label shuffling) to estimate empirical FDR
  4. Compare BH discovery count vs mean null discoveries

If empirical FDR ≈ nominal q (0.05), BH is holding up despite correlation.
If empirical FDR >> q, BH is anti-conservative and discoveries are inflated.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow importing from the moirai package
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from moirai.analyze.motifs import (
    find_gapped_motifs,
    find_motifs,
    _filtered_names,
    _extract_ngrams,
    _is_subsequence,
)
from moirai.analyze.stats import permutation_fdr
from moirai.load import load_runs


def build_gapped_membership(runs, motifs):
    """Build boolean membership matrix for gapped motifs.

    For each motif, check which runs contain that ordered subsequence
    using greedy left-to-right matching (same logic as find_gapped_motifs).
    """
    known = [r for r in runs if r.result.success is not None]
    n_runs = len(known)
    n_patterns = len(motifs)

    membership = np.zeros((n_patterns, n_runs), dtype=bool)
    outcomes = np.array([r.result.success for r in known], dtype=bool)

    for i, motif in enumerate(motifs):
        anchors = motif.anchors
        for j, run in enumerate(known):
            names = _filtered_names(run)
            # Greedy subsequence matching
            pos = 0
            matched = 0
            while pos < len(names) and matched < len(anchors):
                if names[pos] == anchors[matched]:
                    matched += 1
                pos += 1
            membership[i, j] = matched == len(anchors)

    return membership, outcomes, known


def build_contiguous_membership(runs, motifs):
    """Build boolean membership matrix for contiguous n-gram motifs."""
    known = [r for r in runs if r.result.success is not None]
    n_runs = len(known)
    n_patterns = len(motifs)

    membership = np.zeros((n_patterns, n_runs), dtype=bool)
    outcomes = np.array([r.result.success for r in known], dtype=bool)

    for i, motif in enumerate(motifs):
        pattern = motif.pattern
        plen = len(pattern)
        for j, run in enumerate(known):
            names = _filtered_names(run)
            # Sliding window for contiguous match
            for k in range(len(names) - plen + 1):
                if tuple(names[k:k + plen]) == pattern:
                    membership[i, j] = True
                    break

    return membership, outcomes, known


def build_all_gapped_candidates_membership(runs, max_length=3, min_count=3, q_threshold=0.05):
    """Build membership for ALL gapped candidates (not just survivors).

    This is what permutation_fdr needs — the full candidate set,
    because BH correction runs over all candidates.
    """
    from moirai.compress import step_enriched_name
    import math
    from moirai.analyze.stats import fishers_exact_2x2, benjamini_hochberg

    known = [r for r in runs if r.result.success is not None]
    if not known:
        return np.zeros((0, 0), dtype=bool), np.array([], dtype=bool), [], 0

    baseline = sum(1 for r in known if r.result.success) / len(known)
    if baseline in (0.0, 1.0):
        return np.zeros((0, 0), dtype=bool), np.array([], dtype=bool), [], 0

    # Replicate the candidate enumeration from find_gapped_motifs
    # to get the full set of candidates and their membership vectors
    all_types: dict[str, int] = {}
    id_to_name: list[str] = []
    sequences: dict[str, list[int]] = {}
    type_counts: dict[str, int] = {}

    for run in runs:
        names = _filtered_names(run)
        ids = []
        seen: set[str] = set()
        for name in names:
            if name not in all_types:
                all_types[name] = len(id_to_name)
                id_to_name.append(name)
            ids.append(all_types[name])
            if name not in seen:
                seen.add(name)
                type_counts[name] = type_counts.get(name, 0) + 1
        sequences[run.run_id] = ids

    freq_threshold = max(min_count, math.ceil(0.05 * len(runs)))
    frequent_ids = {all_types[n] for n, c in type_counts.items() if c >= freq_threshold}

    if len(frequent_ids) < 2:
        return np.zeros((0, 0), dtype=bool), np.array([], dtype=bool), [], 0

    # Enumerate all candidate patterns and track per-run membership
    # pattern_key -> set of run_ids that contain it
    pattern_runs: dict[tuple[int, ...], set[str]] = {}

    for run in known:
        seq = sequences.get(run.run_id, [])
        if len(seq) < 2:
            continue

        seen_types: set[int] = set()
        seen_pairs: set[tuple[int, int]] = set()
        run_pairs_seen: set[tuple[int, int]] = set()
        run_triples_seen: set[tuple[int, int, int]] = set()

        for z in seq:
            if z not in frequent_ids:
                seen_types.add(z)
                continue

            if max_length >= 3:
                for t, u in seen_pairs:
                    triple = (t, u, z)
                    if triple not in run_triples_seen:
                        run_triples_seen.add(triple)
                        if triple not in pattern_runs:
                            pattern_runs[triple] = set()
                        pattern_runs[triple].add(run.run_id)

            for t in seen_types:
                if t not in frequent_ids:
                    continue
                pair = (t, z)
                if pair not in run_pairs_seen:
                    run_pairs_seen.add(pair)
                    if pair not in pattern_runs:
                        pattern_runs[pair] = set()
                    pattern_runs[pair].add(run.run_id)
                    seen_pairs.add(pair)

            seen_types.add(z)

    # Filter by min_count
    candidate_patterns = [
        (p, rids) for p, rids in pattern_runs.items()
        if len(rids) >= min_count
    ]

    if not candidate_patterns:
        return np.zeros((0, 0), dtype=bool), np.array([], dtype=bool), [], 0

    # Build membership matrix for all candidates
    run_index = {r.run_id: i for i, r in enumerate(known)}
    n_runs = len(known)
    n_patterns = len(candidate_patterns)

    membership = np.zeros((n_patterns, n_runs), dtype=bool)
    outcomes = np.array([r.result.success for r in known], dtype=bool)

    for i, (pattern, rids) in enumerate(candidate_patterns):
        for rid in rids:
            if rid in run_index:
                membership[i, run_index[rid]] = True

    # Count BH discoveries on the full candidate set
    total_success = int(outcomes.sum())
    total_fail = n_runs - total_success

    p_values = []
    for i in range(n_patterns):
        with_s = int(membership[i] @ outcomes.astype(float))
        with_total = int(membership[i].sum())
        with_f = with_total - with_s
        without_s = total_success - with_s
        without_f = total_fail - with_f
        pv = fishers_exact_2x2(with_s, with_f, without_s, without_f)
        p_values.append(pv)

    adjusted = benjamini_hochberg(p_values)
    bh_discoveries = sum(1 for qv in adjusted if qv is not None and qv <= q_threshold)

    return membership, outcomes, candidate_patterns, bh_discoveries


def build_all_contiguous_candidates_membership(runs, min_n=3, max_n=5, min_count=3, q_threshold=0.05):
    """Build membership for ALL contiguous candidates (not just survivors)."""
    from moirai.analyze.stats import fishers_exact_2x2, benjamini_hochberg

    known = [r for r in runs if r.result.success is not None]
    if not known:
        return np.zeros((0, 0), dtype=bool), np.array([], dtype=bool), [], 0

    baseline = sum(1 for r in known if r.result.success) / len(known)
    if baseline in (0.0, 1.0):
        return np.zeros((0, 0), dtype=bool), np.array([], dtype=bool), [], 0

    # Extract all n-grams per run
    run_grams: dict[str, set[tuple[str, ...]]] = {}
    all_grams: set[tuple[str, ...]] = set()

    for run in runs:
        names = _filtered_names(run)
        grams_with_pos = _extract_ngrams(names, min_n, max_n)
        seen: set[tuple[str, ...]] = set()
        for gram, _ in grams_with_pos:
            seen.add(gram)
        run_grams[run.run_id] = seen
        all_grams.update(seen)

    # Filter by min_count
    gram_run_count: dict[tuple[str, ...], int] = {}
    for run in known:
        for gram in run_grams.get(run.run_id, set()):
            gram_run_count[gram] = gram_run_count.get(gram, 0) + 1

    candidate_grams = [g for g, c in gram_run_count.items() if c >= min_count]
    if not candidate_grams:
        return np.zeros((0, 0), dtype=bool), np.array([], dtype=bool), [], 0

    # Build membership matrix
    run_index = {r.run_id: i for i, r in enumerate(known)}
    n_runs = len(known)
    n_patterns = len(candidate_grams)

    membership = np.zeros((n_patterns, n_runs), dtype=bool)
    outcomes = np.array([r.result.success for r in known], dtype=bool)

    for i, gram in enumerate(candidate_grams):
        for run in known:
            if gram in run_grams.get(run.run_id, set()):
                membership[i, run_index[run.run_id]] = True

    # Count BH discoveries
    total_success = int(outcomes.sum())
    total_fail = n_runs - total_success

    p_values = []
    for i in range(n_patterns):
        with_s = int(membership[i] @ outcomes.astype(float))
        with_total = int(membership[i].sum())
        with_f = with_total - with_s
        without_s = total_success - with_s
        without_f = total_fail - with_f
        pv = fishers_exact_2x2(with_s, with_f, without_s, without_f)
        p_values.append(pv)

    adjusted = benjamini_hochberg(p_values)
    bh_discoveries = sum(1 for qv in adjusted if qv is not None and qv <= q_threshold)

    return membership, outcomes, candidate_grams, bh_discoveries


def analyze_dataset(name: str, data_path: str, n_permutations: int = 500):
    """Run permutation FDR analysis on one dataset."""
    print(f"\n{'=' * 70}")
    print(f"  Dataset: {name}")
    print(f"{'=' * 70}")

    runs, warnings = load_runs(data_path)
    known = [r for r in runs if r.result.success is not None]
    n_success = sum(1 for r in known if r.result.success)
    n_fail = len(known) - n_success
    print(f"  Runs: {len(runs)} total, {len(known)} with outcomes "
          f"({n_success} pass, {n_fail} fail)")

    if len(known) < 10:
        print("  Skipping — too few runs with outcomes.")
        return

    # --- Gapped motifs ---
    print(f"\n  --- Gapped motifs ---")
    t0 = time.time()
    gapped_motifs, gapped_tested = find_gapped_motifs(runs, q_threshold=0.05)
    t_gapped = time.time() - t0
    print(f"  find_gapped_motifs: {gapped_tested} candidates, "
          f"{len(gapped_motifs)} survived BH q=0.05 "
          f"({len(gapped_motifs)}/{gapped_tested} = "
          f"{len(gapped_motifs)/max(gapped_tested,1)*100:.1f}%) "
          f"[{t_gapped:.1f}s]")

    if gapped_tested > 0:
        print(f"  Building full candidate membership matrix...")
        t0 = time.time()
        g_membership, g_outcomes, g_candidates, g_bh_disc = \
            build_all_gapped_candidates_membership(runs)
        t_build = time.time() - t0
        print(f"  Matrix: {g_membership.shape[0]} patterns × {g_membership.shape[1]} runs "
              f"(BH discoveries via Fisher: {g_bh_disc}) [{t_build:.1f}s]")

        if g_membership.shape[0] > 0:
            print(f"  Running permutation FDR (n={n_permutations})...")
            t0 = time.time()
            emp_fdr, disc_per_perm = permutation_fdr(
                g_membership, g_outcomes,
                q_threshold=0.05,
                n_permutations=n_permutations,
            )
            t_perm = time.time() - t0

            # permutation_fdr uses chi-squared internally, which may give
            # a different BH discovery count than our Fisher-based count.
            # Report both for transparency.
            actual_disc_chi2 = int(np.mean(disc_per_perm) / emp_fdr) if emp_fdr > 0 else 0
            mean_null = np.mean(disc_per_perm)
            median_null = np.median(disc_per_perm)
            max_null = max(disc_per_perm) if disc_per_perm else 0
            p95_null = np.percentile(disc_per_perm, 95) if disc_per_perm else 0

            print(f"\n  GAPPED MOTIF RESULTS:")
            print(f"    BH discoveries (Fisher's exact): {len(gapped_motifs)}")
            print(f"    Permutation null discoveries:     mean={mean_null:.1f}, "
                  f"median={median_null:.0f}, p95={p95_null:.0f}, max={max_null}")
            print(f"    Empirical FDR:                    {emp_fdr:.4f}")
            print(f"    Nominal q threshold:              0.05")
            print(f"    [{t_perm:.1f}s for {n_permutations} permutations]")

            if emp_fdr <= 0.10:
                print(f"    VERDICT: Gapped motifs are TRUSTWORTHY "
                      f"(empirical FDR {emp_fdr:.3f} ≤ 0.10)")
            elif emp_fdr <= 0.25:
                print(f"    VERDICT: Gapped motifs are MARGINAL "
                      f"(empirical FDR {emp_fdr:.3f} between 0.10 and 0.25)")
            else:
                print(f"    VERDICT: Gapped motifs are INFLATED "
                      f"(empirical FDR {emp_fdr:.3f} > 0.25)")

    # --- Contiguous motifs ---
    print(f"\n  --- Contiguous motifs ---")
    t0 = time.time()
    contig_motifs, contig_tested = find_motifs(runs, q_threshold=0.05)
    t_contig = time.time() - t0
    print(f"  find_motifs: {contig_tested} candidates, "
          f"{len(contig_motifs)} survived BH q=0.05 "
          f"({len(contig_motifs)}/{max(contig_tested,1)} = "
          f"{len(contig_motifs)/max(contig_tested,1)*100:.1f}%) "
          f"[{t_contig:.1f}s]")

    if contig_tested > 0:
        print(f"  Building full candidate membership matrix...")
        t0 = time.time()
        c_membership, c_outcomes, c_candidates, c_bh_disc = \
            build_all_contiguous_candidates_membership(runs)
        t_build = time.time() - t0
        print(f"  Matrix: {c_membership.shape[0]} patterns × {c_membership.shape[1]} runs "
              f"(BH discoveries via Fisher: {c_bh_disc}) [{t_build:.1f}s]")

        if c_membership.shape[0] > 0:
            print(f"  Running permutation FDR (n={n_permutations})...")
            t0 = time.time()
            emp_fdr, disc_per_perm = permutation_fdr(
                c_membership, c_outcomes,
                q_threshold=0.05,
                n_permutations=n_permutations,
            )
            t_perm = time.time() - t0

            mean_null = np.mean(disc_per_perm)
            median_null = np.median(disc_per_perm)
            max_null = max(disc_per_perm) if disc_per_perm else 0
            p95_null = np.percentile(disc_per_perm, 95) if disc_per_perm else 0

            print(f"\n  CONTIGUOUS MOTIF RESULTS:")
            print(f"    BH discoveries (Fisher's exact): {len(contig_motifs)}")
            print(f"    Permutation null discoveries:     mean={mean_null:.1f}, "
                  f"median={median_null:.0f}, p95={p95_null:.0f}, max={max_null}")
            print(f"    Empirical FDR:                    {emp_fdr:.4f}")
            print(f"    Nominal q threshold:              0.05")
            print(f"    [{t_perm:.1f}s for {n_permutations} permutations]")

            if emp_fdr <= 0.10:
                print(f"    VERDICT: Contiguous motifs are TRUSTWORTHY "
                      f"(empirical FDR {emp_fdr:.3f} ≤ 0.10)")
            elif emp_fdr <= 0.25:
                print(f"    VERDICT: Contiguous motifs are MARGINAL "
                      f"(empirical FDR {emp_fdr:.3f} between 0.10 and 0.25)")
            else:
                print(f"    VERDICT: Contiguous motifs are INFLATED "
                      f"(empirical FDR {emp_fdr:.3f} > 0.25)")


def main():
    print("Permutation FDR Validation for Motif Discovery")
    print("=" * 70)
    print("Question: Is the BH-corrected survival rate real signal,")
    print("or inflated by structural correlation between tests?")
    print("(Gapped motifs are nested: A→B is contained in A→B→C)")

    # Data lives in the main repo, not in the worktree
    examples_dir = Path("/Users/ryo/dev/moirai/examples")
    n_permutations = 500

    datasets = [
        ("eval_harness", examples_dir / "eval_harness"),
        ("swe_agent", examples_dir / "swe_agent"),
        ("swe_smith", examples_dir / "swe_smith"),
    ]

    for name, path in datasets:
        if path.exists():
            analyze_dataset(name, str(path), n_permutations=n_permutations)
        else:
            print(f"\n  Skipping {name} — directory not found: {path}")

    print(f"\n{'=' * 70}")
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
