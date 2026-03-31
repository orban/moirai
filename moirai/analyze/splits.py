"""Derive divergence points from dendrogram splits.

Instead of testing every alignment column independently (Fisher's test),
this derives divergence from the hierarchical clustering structure:
each internal node in the dendrogram splits runs into two subtrees,
and we find the alignment column that best discriminates them.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram, linkage

from moirai.analyze.align import distance_matrix
from moirai.analyze.divergence import _compute_significance
from moirai.schema import Alignment, GAP, Run, SplitDivergence


def find_split_divergences(
    alignment: Alignment,
    runs: list[Run],
) -> tuple[list[SplitDivergence], np.ndarray, dict]:
    """Derive divergence points from dendrogram splits.

    Returns:
        splits: list of SplitDivergence, one per internal node (sorted by merge distance desc)
        Z: linkage matrix (for rendering)
        dendro: scipy dendrogram result dict (for rendering)
    """
    n_runs = len(runs)
    if n_runs < 2 or not alignment.matrix or not alignment.matrix[0]:
        return [], np.array([]), {}

    n_cols = len(alignment.matrix[0])
    success_map = {r.run_id: r.result.success for r in runs}

    # Compute linkage
    condensed = distance_matrix(runs, level=alignment.level if alignment.level == "name" else "name")
    if len(condensed) == 0:
        return [], np.array([]), {}

    Z = linkage(condensed, method="average")
    dendro = scipy_dendrogram(Z, no_plot=True)

    # Walk linkage to get subtree leaves at each internal node
    leaves_of: dict[int, set[int]] = {}
    for i in range(n_runs):
        leaves_of[i] = {i}

    splits: list[SplitDivergence] = []

    for i, (left_node, right_node, dist, count) in enumerate(Z):
        left_node, right_node = int(left_node), int(right_node)
        left_indices = leaves_of[left_node]
        right_indices = leaves_of[right_node]
        node_id = n_runs + i
        leaves_of[node_id] = left_indices | right_indices

        # Find most discriminating column
        col, separation = _discriminating_column(
            left_indices, right_indices, alignment.matrix, n_cols,
        )

        if col < 0 or separation == 0:
            # No discriminating column — runs are identical at all positions
            continue

        # Value distributions at the discriminating column
        left_vals = Counter(
            alignment.matrix[j][col] for j in left_indices
            if col < len(alignment.matrix[j]) and alignment.matrix[j][col] != GAP
        )
        right_vals = Counter(
            alignment.matrix[j][col] for j in right_indices
            if col < len(alignment.matrix[j]) and alignment.matrix[j][col] != GAP
        )

        # Success rates
        left_rids = [alignment.run_ids[j] for j in left_indices]
        right_rids = [alignment.run_ids[j] for j in right_indices]

        left_known = [success_map[rid] for rid in left_rids if success_map.get(rid) is not None]
        right_known = [success_map[rid] for rid in right_rids if success_map.get(rid) is not None]

        left_rate = sum(1 for x in left_known if x) / len(left_known) if left_known else None
        right_rate = sum(1 for x in right_known if x) / len(right_known) if right_known else None

        # Fisher's test: does this split predict outcome?
        values_for_test = {
            "left": left_rids,
            "right": right_rids,
        }
        p_val = _compute_significance(values_for_test, success_map)

        splits.append(SplitDivergence(
            node_id=node_id,
            left_runs=left_rids,
            right_runs=right_rids,
            merge_distance=dist,
            column=col,
            separation=separation,
            left_values=dict(left_vals),
            right_values=dict(right_vals),
            left_success_rate=left_rate,
            right_success_rate=right_rate,
            p_value=p_val,
        ))

    # Sort by merge distance descending (biggest splits first = top of tree)
    splits.sort(key=lambda s: -s.merge_distance)

    return splits, Z, dendro


def _discriminating_column(
    left_indices: set[int],
    right_indices: set[int],
    matrix: list[list[str]],
    n_cols: int,
) -> tuple[int, float]:
    """Find the alignment column that best separates left from right subtree.

    Score = fraction of (left, right) cross-pairs with different values.
    """
    best_col = -1
    best_score = 0.0

    left_list = sorted(left_indices)
    right_list = sorted(right_indices)
    total_pairs = len(left_list) * len(right_list)
    if total_pairs == 0:
        return -1, 0.0

    for col in range(n_cols):
        left_vals = [matrix[i][col] for i in left_list if col < len(matrix[i]) and matrix[i][col] != GAP]
        right_vals = [matrix[i][col] for i in right_list if col < len(matrix[i]) and matrix[i][col] != GAP]

        if not left_vals or not right_vals:
            continue

        disagree = sum(1 for lv in left_vals for rv in right_vals if lv != rv)
        score = disagree / (len(left_vals) * len(right_vals))

        if score > best_score:
            best_score = score
            best_col = col

    return best_col, best_score
