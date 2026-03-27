from __future__ import annotations

import numpy as np

from moirai.schema import Alignment, GAP, Run, step_type_sequence


def _nw_align(
    seq_a: list[str],
    seq_b: list[str],
    match: int = 1,
    mismatch: int = -1,
    gap: int = -1,
) -> tuple[list[str], list[str]]:
    """Needleman-Wunsch global alignment on string sequences.

    Returns two aligned sequences with GAP markers for insertions/deletions.
    """
    n, m = len(seq_a), len(seq_b)

    # Score matrix
    score = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = score[i - 1][0] + gap
    for j in range(1, m + 1):
        score[0][j] = score[0][j - 1] + gap

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = score[i - 1][j - 1] + (match if seq_a[i - 1] == seq_b[j - 1] else mismatch)
            up = score[i - 1][j] + gap
            left = score[i][j - 1] + gap
            score[i][j] = max(diag, up, left)

    # Traceback
    aligned_a: list[str] = []
    aligned_b: list[str] = []
    i, j = n, m

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            diag = score[i - 1][j - 1] + (match if seq_a[i - 1] == seq_b[j - 1] else mismatch)
            if score[i][j] == diag:
                aligned_a.append(seq_a[i - 1])
                aligned_b.append(seq_b[j - 1])
                i -= 1
                j -= 1
                continue
        if i > 0 and score[i][j] == score[i - 1][j] + gap:
            aligned_a.append(seq_a[i - 1])
            aligned_b.append(GAP)
            i -= 1
        else:
            aligned_a.append(GAP)
            aligned_b.append(seq_b[j - 1])
            j -= 1

    aligned_a.reverse()
    aligned_b.reverse()
    return aligned_a, aligned_b


def _get_sequence(run: Run, level: str) -> list[str]:
    if level == "name":
        from moirai.compress import step_enriched_name
        return [n for s in run.steps if (n := step_enriched_name(s)) is not None]
    if level == "type":
        return step_type_sequence(run)
    raise ValueError(f"invalid level '{level}', expected 'type' or 'name'")


def trajectory_distance(run_a: Run, run_b: Run, level: str = "type") -> float:
    """Normalized edit distance between two runs' step sequences.

    Uses NW alignment, counts non-match columns, divides by alignment length.
    Returns a float in [0.0, 1.0].
    """
    seq_a = _get_sequence(run_a, level)
    seq_b = _get_sequence(run_b, level)

    if not seq_a and not seq_b:
        return 0.0

    aligned_a, aligned_b = _nw_align(seq_a, seq_b)
    mismatches = sum(1 for a, b in zip(aligned_a, aligned_b) if a != b)
    return mismatches / len(aligned_a)


def distance_matrix(runs: list[Run], level: str = "type") -> np.ndarray:
    """Compute condensed pairwise distance matrix for scipy."""
    n = len(runs)
    if n < 2:
        return np.array([])

    # Precompute sequences
    sequences = [_get_sequence(r, level) for r in runs]

    # Build condensed distance matrix
    dists: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            seq_a, seq_b = sequences[i], sequences[j]
            if not seq_a and not seq_b:
                dists.append(0.0)
            else:
                aligned_a, aligned_b = _nw_align(seq_a, seq_b)
                mismatches = sum(1 for a, b in zip(aligned_a, aligned_b) if a != b)
                dists.append(mismatches / len(aligned_a))

    return np.array(dists)


def _consensus(aligned_sequences: list[list[str]]) -> list[str]:
    """Build consensus sequence from multiple aligned sequences.

    At each column, take the majority non-GAP value. Ties broken by first occurrence.
    """
    if not aligned_sequences:
        return []

    n_cols = len(aligned_sequences[0])
    result: list[str] = []

    for col in range(n_cols):
        counts: dict[str, int] = {}
        first_seen: dict[str, int] = {}
        for seq_idx, seq in enumerate(aligned_sequences):
            val = seq[col]
            if val == GAP:
                continue
            if val not in counts:
                counts[val] = 0
                first_seen[val] = seq_idx
            counts[val] += 1

        if not counts:
            result.append(GAP)
        else:
            # Sort by count desc, then first occurrence asc
            best = max(counts.keys(), key=lambda v: (counts[v], -first_seen[v]))
            result.append(best)

    return result


def align_runs(runs: list[Run], level: str = "type") -> Alignment:
    """Progressive multi-run alignment.

    1. Compute pairwise distances
    2. Find closest pair, align with NW
    3. Build consensus, align next closest against it
    4. Repeat until all runs incorporated
    5. Map all runs back through consensus
    """
    if not runs:
        return Alignment(run_ids=[], matrix=[], level=level)

    if len(runs) == 1:
        seq = _get_sequence(runs[0], level)
        return Alignment(run_ids=[runs[0].run_id], matrix=[seq], level=level)

    sequences = [_get_sequence(r, level) for r in runs]
    n = len(runs)

    # Compute full pairwise distance matrix
    full_dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            seq_a, seq_b = sequences[i], sequences[j]
            if not seq_a and not seq_b:
                d = 0.0
            else:
                al_a, al_b = _nw_align(seq_a, seq_b)
                mismatches = sum(1 for a, b in zip(al_a, al_b) if a != b)
                d = mismatches / len(al_a) if al_a else 0.0
            full_dist[i][j] = d
            full_dist[j][i] = d

    # Progressive alignment
    remaining = set(range(n))

    # Find closest pair to start
    best_i, best_j = 0, 1
    best_d = full_dist[0][1]
    for i in range(n):
        for j in range(i + 1, n):
            if full_dist[i][j] < best_d:
                best_d = full_dist[i][j]
                best_i, best_j = i, j

    # Align first pair
    al_a, al_b = _nw_align(sequences[best_i], sequences[best_j])

    # Track aligned versions of each run
    aligned: dict[int, list[str]] = {best_i: al_a, best_j: al_b}
    remaining.discard(best_i)
    remaining.discard(best_j)

    # Build initial consensus
    current_consensus = _consensus([al_a, al_b])

    # Progressively add remaining runs
    while remaining:
        # Find closest remaining run to the consensus
        best_idx = -1
        best_d = float("inf")
        for idx in remaining:
            # Approximate distance: align against consensus and measure
            al_seq, al_con = _nw_align(sequences[idx], current_consensus)
            mismatches = sum(1 for a, b in zip(al_seq, al_con) if a != b)
            d = mismatches / len(al_seq) if al_seq else 0.0
            if d < best_d:
                best_d = d
                best_idx = idx

        # Align this run against the consensus
        al_new, al_con = _nw_align(sequences[best_idx], current_consensus)

        # If alignment introduced new gaps in the consensus, propagate to all existing alignments
        if len(al_con) != len(current_consensus):
            # Build gap mapping: where in the new alignment does each old column land?
            old_col = 0
            col_mapping: list[int | None] = []  # new_col -> old_col or None (new gap)
            for new_col in range(len(al_con)):
                if al_con[new_col] != GAP or (old_col < len(current_consensus) and current_consensus[old_col] == GAP):
                    col_mapping.append(old_col)
                    old_col += 1
                else:
                    col_mapping.append(None)

            # Remap existing aligned sequences
            for idx in aligned:
                old_aligned = aligned[idx]
                new_aligned: list[str] = []
                for mapping in col_mapping:
                    if mapping is not None and mapping < len(old_aligned):
                        new_aligned.append(old_aligned[mapping])
                    else:
                        new_aligned.append(GAP)
                aligned[idx] = new_aligned

        aligned[best_idx] = al_new
        remaining.discard(best_idx)

        # Update consensus
        current_consensus = _consensus(list(aligned.values()))

    # Build final matrix in original run order
    run_ids = [r.run_id for r in runs]
    matrix = [aligned[i] for i in range(n)]

    # Ensure all rows have the same length
    max_len = max(len(row) for row in matrix)
    for row in matrix:
        while len(row) < max_len:
            row.append(GAP)

    return Alignment(run_ids=run_ids, matrix=matrix, level=level)
