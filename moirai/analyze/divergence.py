from __future__ import annotations

import math

from moirai.schema import Alignment, DivergencePoint, GAP, Run


def find_divergence_points(alignment: Alignment, runs: list[Run]) -> list[DivergencePoint]:
    """Find columns where runs diverge and correlate with outcome.

    A column is a divergence point if it contains more than one distinct non-GAP value.
    Points are sorted by entropy descending.
    """
    if not alignment.matrix or not alignment.matrix[0]:
        return []

    # Build run_id -> success mapping
    success_map = {r.run_id: r.result.success for r in runs}
    n_cols = len(alignment.matrix[0])

    points: list[DivergencePoint] = []

    for col in range(n_cols):
        # Collect values at this column (excluding gaps)
        values: dict[str, list[str]] = {}  # value -> list of run_ids
        for run_idx, run_id in enumerate(alignment.run_ids):
            if run_idx < len(alignment.matrix):
                val = alignment.matrix[run_idx][col] if col < len(alignment.matrix[run_idx]) else GAP
                if val != GAP:
                    if val not in values:
                        values[val] = []
                    values[val].append(run_id)

        # Only a divergence point if >1 distinct value
        if len(values) <= 1:
            continue

        # Value counts
        value_counts = {v: len(ids) for v, ids in values.items()}

        # Entropy
        total = sum(value_counts.values())
        entropy = 0.0
        for count in value_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        # Success rates per value
        success_by_value: dict[str, float | None] = {}
        for val, run_ids in values.items():
            successes = [success_map.get(rid) for rid in run_ids]
            known = [s for s in successes if s is not None]
            if known:
                success_by_value[val] = sum(1 for s in known if s) / len(known)
            else:
                success_by_value[val] = None

        points.append(DivergencePoint(
            column=col,
            value_counts=value_counts,
            entropy=entropy,
            success_by_value=success_by_value,
        ))

    # Sort by entropy descending
    points.sort(key=lambda p: -p.entropy)
    return points
