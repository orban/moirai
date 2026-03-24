from __future__ import annotations

from collections import Counter

from moirai.analyze.cluster import cluster_runs
from moirai.analyze.summary import summarize_runs
from moirai.schema import CohortDiff, Run, signature


def compare_cohorts(
    a_runs: list[Run],
    b_runs: list[Run],
    level: str = "type",
    threshold: float = 0.3,
) -> CohortDiff:
    """Compare two cohorts of runs.

    Clusters each cohort independently, matches by prototype label,
    and computes shifts. Raises ValueError if either cohort is empty.
    """
    if not a_runs:
        raise ValueError("cohort A is empty")
    if not b_runs:
        raise ValueError("cohort B is empty")

    a_summary = summarize_runs(a_runs)
    b_summary = summarize_runs(b_runs)

    # Pre-clustering: unique signatures
    a_sigs = Counter(signature(r) for r in a_runs)
    b_sigs = Counter(signature(r) for r in b_runs)

    a_only = [(sig, count) for sig, count in a_sigs.most_common() if sig not in b_sigs]
    b_only = [(sig, count) for sig, count in b_sigs.most_common() if sig not in a_sigs]

    # Cluster each cohort
    a_clusters = cluster_runs(a_runs, level=level, threshold=threshold)
    b_clusters = cluster_runs(b_runs, level=level, threshold=threshold)

    # Match clusters by prototype and compute shifts
    a_proto_counts: dict[str, int] = {}
    for c in a_clusters.clusters:
        a_proto_counts[c.prototype] = c.count

    b_proto_counts: dict[str, int] = {}
    for c in b_clusters.clusters:
        b_proto_counts[c.prototype] = c.count

    all_protos = set(a_proto_counts.keys()) | set(b_proto_counts.keys())
    shifts: list[tuple[str, int]] = []
    for proto in all_protos:
        a_count = a_proto_counts.get(proto, 0)
        b_count = b_proto_counts.get(proto, 0)
        delta = b_count - a_count
        if delta != 0:
            shifts.append((proto, delta))

    # Sort by absolute delta descending
    shifts.sort(key=lambda x: -abs(x[1]))

    return CohortDiff(
        a_summary=a_summary,
        b_summary=b_summary,
        a_only_signatures=a_only,
        b_only_signatures=b_only,
        cluster_shifts=shifts,
    )
