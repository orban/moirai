from __future__ import annotations

import numpy as np

from moirai.schema import Run, step_type_sequence, step_name_sequence

GAP = "-"


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
        return step_name_sequence(run)
    return step_type_sequence(run)


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
