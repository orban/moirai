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


def compute_concordance(
    runs: list[Run],
    cluster_labels: dict[str, int],
    level: str = "name",
) -> dict:
    """Compute structural concordance for each cluster.

    Measures whether structural typicality (distance from consensus)
    predicts outcome quality within each cluster. Uses Kendall's Tau-b.

    Only computed for clusters with mixed outcomes and >= 5 known-outcome runs.
    """
    from moirai.analyze.align import consensus, align_runs
    from moirai.analyze.stats import kendall_tau_b
    from moirai.schema import ConcordanceScore

    # Group runs by cluster
    by_cluster: dict[int, list[Run]] = {}
    for run in runs:
        cid = cluster_labels.get(run.run_id)
        if cid is None:
            continue
        if cid not in by_cluster:
            by_cluster[cid] = []
        by_cluster[cid].append(run)

    results: dict[int, ConcordanceScore] = {}

    for cid, cluster_runs_list in by_cluster.items():
        # Filter to known outcomes
        known = [r for r in cluster_runs_list if r.result.success is not None]
        if len(known) < 5:
            continue

        # Need mixed outcomes
        has_success = any(r.result.success for r in known)
        has_failure = any(not r.result.success for r in known)
        if not has_success or not has_failure:
            continue

        # Align within cluster
        alignment = align_runs(known, level=level)
        if not alignment.matrix or not alignment.matrix[0]:
            continue

        cons = consensus(alignment.matrix)
        n_cols = len(cons)

        # Distance from consensus: count mismatches in alignment matrix
        distances: list[float] = []
        for row in alignment.matrix:
            mismatches = sum(1 for a, b in zip(row, cons) if a != b)
            distances.append(mismatches / n_cols if n_cols > 0 else 0.0)

        # Need variance in distances
        if len(set(distances)) < 2:
            continue

        # Outcome scores
        if all(r.result.score is not None for r in known):
            outcomes = [r.result.score for r in known]
            used_continuous = True
        else:
            outcomes = [1.0 if r.result.success else 0.0 for r in known]
            used_continuous = False

        # Negate distances: closer to consensus = higher = more typical
        # Positive tau then means "typical runs succeed more"
        tau, p_value = kendall_tau_b([-d for d in distances], outcomes)

        results[cid] = ConcordanceScore(
            tau=tau,
            p_value=p_value,
            n_runs=len(known),
            used_continuous=used_continuous,
        )

    return results
