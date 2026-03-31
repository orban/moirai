"""Motif discovery — find step patterns that predict success or failure.

Extracts n-gram subsequences from runs and tests which patterns
discriminate between successful and failing runs.
"""
from __future__ import annotations

import math

from moirai.analyze.stats import benjamini_hochberg, fishers_exact_2x2
from moirai.compress import step_enriched_name
from moirai.schema import GappedMotif, Motif, Run


def _filtered_names(run: Run) -> list[str]:
    """Get enriched step name sequence (no noise, with attr context)."""
    return [n for s in run.steps if (n := step_enriched_name(s)) is not None]


def _extract_ngrams(names: list[str], min_n: int = 3, max_n: int = 5) -> list[tuple[tuple[str, ...], float]]:
    """Extract all n-grams with their normalized positions.

    Returns list of (ngram, position) where position is in [0, 1].
    """
    results: list[tuple[tuple[str, ...], float]] = []
    length = len(names)
    if length == 0:
        return results

    for n in range(min_n, max_n + 1):
        for i in range(length - n + 1):
            gram = tuple(names[i:i + n])
            pos = i / max(length - 1, 1)
            results.append((gram, pos))

    return results


def find_motifs(
    runs: list[Run],
    min_n: int = 3,
    max_n: int = 5,
    min_count: int = 3,
    q_threshold: float = 0.05,
) -> tuple[list[Motif], int]:
    """Find step patterns that correlate with success or failure.

    Returns (motifs, n_candidates_tested). Motifs are sorted by q-value
    ascending with effect size as tiebreaker. BH correction is applied
    internally.
    """
    if not runs:
        return [], 0

    # Baseline success rate
    known = [r for r in runs if r.result.success is not None]
    if not known:
        return [], 0
    baseline = sum(1 for r in known if r.result.success) / len(known)
    if baseline == 0.0 or baseline == 1.0:
        return [], 0  # no variation to explain

    total_success = sum(1 for r in known if r.result.success)
    total_fail = len(known) - total_success

    # For each run, extract n-grams
    run_grams: dict[str, set[tuple[str, ...]]] = {}  # run_id -> set of unique grams
    gram_positions: dict[tuple[str, ...], list[float]] = {}  # gram -> list of positions

    for run in runs:
        names = _filtered_names(run)
        grams_with_pos = _extract_ngrams(names, min_n, max_n)

        seen: set[tuple[str, ...]] = set()
        for gram, pos in grams_with_pos:
            if gram not in seen:
                seen.add(gram)
                if gram not in gram_positions:
                    gram_positions[gram] = []
                gram_positions[gram].append(pos)

        run_grams[run.run_id] = seen

    # For each gram, count success/fail
    gram_counts: dict[tuple[str, ...], tuple[int, int]] = {}  # gram -> (success, fail)
    for run in known:
        grams = run_grams.get(run.run_id, set())
        for gram in grams:
            if gram not in gram_counts:
                gram_counts[gram] = (0, 0)
            s, f = gram_counts[gram]
            if run.result.success:
                gram_counts[gram] = (s + 1, f)
            else:
                gram_counts[gram] = (s, f + 1)

    # Build candidate motifs with raw p-values (no filtering yet)
    candidates: list[Motif] = []
    for gram, (succ, fail) in gram_counts.items():
        total = succ + fail
        if total < min_count:
            continue

        rate = succ / total
        lift = rate / baseline if baseline > 0 else 1.0

        without_s = total_success - succ
        without_f = total_fail - fail
        p_val = fishers_exact_2x2(succ, fail, without_s, without_f)

        avg_pos = sum(gram_positions.get(gram, [0.0])) / max(len(gram_positions.get(gram, [0.0])), 1)

        candidates.append(Motif(
            pattern=gram,
            total_runs=total,
            success_runs=succ,
            fail_runs=fail,
            success_rate=rate,
            baseline_rate=baseline,
            lift=lift,
            p_value=p_val,
            avg_position=avg_pos,
        ))

    # Apply BH correction
    raw_p = [m.p_value for m in candidates]
    adjusted = benjamini_hochberg(raw_p)
    for motif, qv in zip(candidates, adjusted):
        motif.q_value = qv

    # Filter by q-value
    motifs = [m for m in candidates if m.q_value is not None and m.q_value <= q_threshold]

    # Sort by q-value ascending, then by effect size as tiebreaker
    motifs.sort(key=lambda m: (m.q_value if m.q_value is not None else 1.0, -abs(m.success_rate - m.baseline_rate)))

    return motifs, len(candidates)


def find_gapped_motifs(
    runs: list[Run],
    max_length: int = 3,
    min_count: int = 3,
    q_threshold: float = 0.05,
) -> tuple[list[GappedMotif], int]:
    """Find ordered subsequence patterns with flexible gaps.

    Discovers patterns like (A, B, C) where A appears before B before C
    in a run, with any number of steps in between. Uses greedy left-to-right
    matching: first A, then first B after A, then first C after B.

    Returns (motifs, n_candidates_tested).
    """
    if not runs:
        return [], 0

    known = [r for r in runs if r.result.success is not None]
    if not known:
        return [], 0
    baseline = sum(1 for r in known if r.result.success) / len(known)
    if baseline == 0.0 or baseline == 1.0:
        return [], 0

    total_success = sum(1 for r in known if r.result.success)
    total_fail = len(known) - total_success

    # Build frequency index: enriched names → integer IDs
    sequences: dict[str, list[int]] = {}  # run_id → list of type IDs
    type_counts: dict[str, int] = {}  # type name → number of runs containing it

    all_types: dict[str, int] = {}  # name → ID
    id_to_name: list[str] = []

    for run in runs:
        names = _filtered_names(run)
        ids: list[int] = []
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

    # Frequency filter: drop types below max(min_count, 5% of N)
    freq_threshold = max(min_count, math.ceil(0.05 * len(runs)))
    frequent_ids: set[int] = set()
    for name, count in type_counts.items():
        if count >= freq_threshold:
            frequent_ids.add(all_types[name])

    if len(frequent_ids) < 2:
        return [], 0

    # Single-pass counting: enumerate ordered pairs and triples
    # pair_counts[pair] = (success, fail)
    pair_counts: dict[tuple[int, int], list[int]] = {}   # [success, fail]
    triple_counts: dict[tuple[int, int, int], list[int]] = {}
    pair_positions: dict[tuple[int, int], list[float]] = {}
    triple_positions: dict[tuple[int, int, int], list[float]] = {}

    for run in known:
        seq = sequences.get(run.run_id, [])
        length = len(seq)
        if length < 2:
            continue

        is_success = 1 if run.result.success else 0
        seen_types: set[int] = set()
        seen_pairs: set[tuple[int, int]] = set()
        run_pairs_seen: set[tuple[int, int]] = set()
        run_triples_seen: set[tuple[int, int, int]] = set()

        for pos_idx, z in enumerate(seq):
            if z not in frequent_ids:
                seen_types.add(z)
                continue

            norm_pos = pos_idx / max(length - 1, 1)

            # Triples first: use pairs from PREVIOUS steps only (not this step's new pairs)
            if max_length >= 3:
                for t, u in seen_pairs:
                    triple = (t, u, z)
                    if triple not in run_triples_seen:
                        run_triples_seen.add(triple)
                        if triple not in triple_counts:
                            triple_counts[triple] = [0, 0]
                            triple_positions[triple] = []
                        triple_counts[triple][is_success] += 1
                        triple_positions[triple].append(norm_pos)

            # Pairs: for each frequent type seen before, emit (t, z)
            # Done AFTER triples to avoid using same-step pairs for triple extension
            for t in seen_types:
                if t not in frequent_ids:
                    continue
                pair = (t, z)
                if pair not in run_pairs_seen:
                    run_pairs_seen.add(pair)
                    if pair not in pair_counts:
                        pair_counts[pair] = [0, 0]
                        pair_positions[pair] = []
                    pair_counts[pair][is_success] += 1
                    pair_positions[pair].append(norm_pos)
                    seen_pairs.add(pair)

            seen_types.add(z)

    # Collect all candidates, compute Fisher's exact
    candidates: list[GappedMotif] = []

    def _add_candidates(
        counts: dict[tuple[int, ...], list[int]],
        positions: dict[tuple[int, ...], list[float]],
    ) -> None:
        for pattern_ids, (fail, succ) in counts.items():
            total = succ + fail
            if total < min_count:
                continue
            rate = succ / total
            lift = rate / baseline if baseline > 0 else 1.0
            without_s = total_success - succ
            without_f = total_fail - fail
            p_val = fishers_exact_2x2(succ, fail, without_s, without_f)
            anchors = tuple(id_to_name[i] for i in pattern_ids)
            pos_list = positions.get(pattern_ids, [0.0])
            avg_pos = sum(pos_list) / max(len(pos_list), 1)

            candidates.append(GappedMotif(
                anchors=anchors,
                total_runs=total,
                success_runs=succ,
                fail_runs=fail,
                success_rate=rate,
                baseline_rate=baseline,
                lift=lift,
                p_value=p_val,
                avg_position=avg_pos,
            ))

    _add_candidates(pair_counts, pair_positions)
    if max_length >= 3:
        _add_candidates(triple_counts, triple_positions)

    n_tested = len(candidates)
    if not candidates:
        return [], 0

    # Apply BH correction
    raw_p = [m.p_value for m in candidates]
    adjusted = benjamini_hochberg(raw_p)
    for motif, qv in zip(candidates, adjusted):
        motif.q_value = qv

    # Filter by q-value
    surviving = [m for m in candidates if m.q_value is not None and m.q_value <= q_threshold]

    # Prune: drop strict subsequences of longer patterns when the longer
    # pattern covers same/more runs and has equal/better q-value.
    # Check transitively.
    pruned: set[int] = set()
    for i, short in enumerate(surviving):
        if i in pruned:
            continue
        for j, long in enumerate(surviving):
            if i == j or j in pruned:
                continue
            if len(long.anchors) <= len(short.anchors):
                continue
            if not _is_subsequence(short.anchors, long.anchors):
                continue
            # Long is a supersequence of short
            long_q = long.q_value if long.q_value is not None else 1.0
            short_q = short.q_value if short.q_value is not None else 1.0
            if long.total_runs >= short.total_runs and long_q <= short_q:
                pruned.add(i)
                break

    motifs = [m for i, m in enumerate(surviving) if i not in pruned]

    # Sort by q-value ascending, effect size as tiebreaker
    motifs.sort(key=lambda m: (m.q_value if m.q_value is not None else 1.0, -abs(m.success_rate - m.baseline_rate)))

    return motifs, n_tested


def _is_subsequence(short: tuple[str, ...], long: tuple[str, ...]) -> bool:
    """Check if short is a strict subsequence of long."""
    it = iter(long)
    return all(s in it for s in short)
