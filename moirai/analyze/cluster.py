from __future__ import annotations

from collections import Counter

from scipy.cluster.hierarchy import fcluster, linkage

from moirai.analyze.align import distance_matrix
from moirai.schema import ClusterInfo, ClusterResult, Run, signature


def cluster_runs(
    runs: list[Run],
    level: str = "type",
    threshold: float = 0.3,
) -> ClusterResult:
    """Cluster runs by trajectory structural similarity.

    Uses agglomerative clustering with average linkage.
    Threshold is the maximum inter-cluster distance.
    """
    if not runs:
        return ClusterResult(clusters=[], labels={})

    if len(runs) == 1:
        run = runs[0]
        success_vals = [run.result.success] if run.result.success is not None else []
        rate = (sum(1 for s in success_vals if s) / len(success_vals)) if success_vals else None
        error_types: dict[str, int] = {}
        if run.result.error_type:
            error_types[run.result.error_type] = 1
        info = ClusterInfo(
            cluster_id=0,
            count=1,
            success_rate=rate,
            prototype=signature(run),
            avg_length=float(len(run.steps)),
            error_types=error_types,
        )
        return ClusterResult(clusters=[info], labels={run.run_id: 0})

    # Compute pairwise distances
    condensed = distance_matrix(runs, level=level)

    # Agglomerative clustering with average linkage
    Z = linkage(condensed, method="average")
    cluster_labels = fcluster(Z, t=threshold, criterion="distance")

    # Build run_id -> cluster_id mapping (0-indexed)
    unique_labels = sorted(set(cluster_labels))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = {run.run_id: label_map[cl] for run, cl in zip(runs, cluster_labels)}

    # Build ClusterInfo for each cluster
    clusters: list[ClusterInfo] = []
    for cluster_id in range(len(unique_labels)):
        cluster_runs_list = [r for r in runs if labels[r.run_id] == cluster_id]
        n = len(cluster_runs_list)

        # Success rate
        success_vals = [r.result.success for r in cluster_runs_list if r.result.success is not None]
        if success_vals:
            rate = sum(1 for s in success_vals if s) / len(success_vals)
        else:
            rate = None

        # Prototype: most common signature
        sig_counter = Counter(signature(r) for r in cluster_runs_list)
        proto = sig_counter.most_common(1)[0][0]

        # Avg length
        avg_len = sum(len(r.steps) for r in cluster_runs_list) / n

        # Error types
        err_counter: Counter[str] = Counter()
        for r in cluster_runs_list:
            if r.result.error_type:
                err_counter[r.result.error_type] += 1

        clusters.append(ClusterInfo(
            cluster_id=cluster_id,
            count=n,
            success_rate=rate,
            prototype=proto,
            avg_length=avg_len,
            error_types=dict(err_counter),
        ))

    return ClusterResult(clusters=clusters, labels=labels)
