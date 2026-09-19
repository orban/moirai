"""Per-task structure score — predicts when branch-aware analysis helps."""
from __future__ import annotations

import random
from dataclasses import dataclass

from moirai.analyze.align import align_runs
from moirai.analyze.divergence import find_divergence_points
from moirai.schema import Run


@dataclass
class StructureScore:
    """Structure score for one task."""
    task_id: str
    branch_gap: float       # mean success-rate gap at top divergence points (0-1)
    earlyness: float        # how early decisive forks appear (0=late, 1=early)
    stability: float        # reproducibility of divergence map across resamples (0-1)
    composite: float        # equal-weight mean of the three components
    n_runs: int
    n_divergence_points: int


def _compute_branch_gap(
    points: list,
    n_cols: int,
    top_k: int = 5,
) -> tuple[float, float, int]:
    """Compute branch success-rate gap and earlyness from divergence points.

    Returns (branch_gap, earlyness, n_divergence_points).
    """
    if not points:
        return 0.0, 0.5, 0

    top = points[:top_k]

    gaps = []
    positions = []
    for dp in top:
        rates = [r for r in dp.success_by_value.values() if r is not None]
        if len(rates) >= 2:
            gaps.append(max(rates) - min(rates))
        positions.append(dp.column / max(n_cols - 1, 1))

    branch_gap = sum(gaps) / len(gaps) if gaps else 0.0
    earlyness = 1.0 - (sum(positions) / len(positions)) if positions else 0.5

    return branch_gap, earlyness, len(points)


def _compute_stability(
    runs: list[Run],
    alignment_full,
    full_points: list,
    top_k: int = 5,
    n_resamples: int = 20,
    subsample_frac: float = 0.7,
    seed: int = 42,
) -> float:
    """Measure how stable the divergence map is across resamples.

    Repeatedly subsample runs, find divergence points, measure overlap
    with the full-data divergence points. Returns fraction of resamples
    that recover at least one of the top-K divergence columns (Jaccard-like).
    """
    if len(runs) < 5 or not full_points:
        return 0.0

    full_cols = {dp.column for dp in full_points[:top_k]}

    rng = random.Random(seed)
    n_sub = max(4, int(len(runs) * subsample_frac))
    recoveries = 0

    for _ in range(n_resamples):
        sub = rng.sample(runs, min(n_sub, len(runs)))
        # Need both pass and fail in subsample
        has_pass = any(r.result.success for r in sub)
        has_fail = any(not r.result.success for r in sub)
        if not has_pass or not has_fail:
            continue

        sub_alignment = align_runs(sub, level="name")
        if not sub_alignment.matrix or not sub_alignment.matrix[0]:
            continue

        sub_points, _ = find_divergence_points(
            sub_alignment, sub, min_branch_size=1, q_threshold=0.30,
        )
        sub_cols = {dp.column for dp in sub_points[:top_k]}

        # Check overlap — any shared column counts as recovery
        # Use relative position matching (within 10% of alignment length)
        # since subsample alignment may have different column count
        sub_n_cols = len(sub_alignment.matrix[0])
        full_n_cols = len(alignment_full.matrix[0])

        sub_positions = {c / max(sub_n_cols - 1, 1) for c in sub_cols}
        full_positions = {c / max(full_n_cols - 1, 1) for c in full_cols}

        # Match if any sub position is within 10% of a full position
        matched = False
        for sp in sub_positions:
            for fp in full_positions:
                if abs(sp - fp) < 0.10:
                    matched = True
                    break
            if matched:
                break

        if matched:
            recoveries += 1

    return recoveries / n_resamples if n_resamples > 0 else 0.0


def compute_structure_score(
    runs: list[Run],
    task_id: str,
    top_k: int = 5,
    n_resamples: int = 20,
    seed: int = 42,
) -> StructureScore:
    """Compute the full structure score for one task."""
    alignment = align_runs(runs, level="name")
    if not alignment.matrix or not alignment.matrix[0]:
        return StructureScore(
            task_id=task_id, branch_gap=0.0, earlyness=0.5,
            stability=0.0, composite=0.167, n_runs=len(runs), n_divergence_points=0,
        )

    points, _ = find_divergence_points(
        alignment, runs, min_branch_size=2, q_threshold=0.20,
    )
    n_cols = len(alignment.matrix[0])

    branch_gap, earlyness, n_dp = _compute_branch_gap(points, n_cols, top_k=top_k)
    stability = _compute_stability(
        runs, alignment, points,
        top_k=top_k, n_resamples=n_resamples, seed=seed,
    )

    # Equal-weight composite, each component in [0, 1]
    composite = (branch_gap + earlyness + stability) / 3.0

    return StructureScore(
        task_id=task_id,
        branch_gap=branch_gap,
        earlyness=earlyness,
        stability=stability,
        composite=composite,
        n_runs=len(runs),
        n_divergence_points=n_dp,
    )


def _compute_one(args: tuple) -> StructureScore | None:
    """Picklable wrapper for multiprocessing."""
    task_id, runs, top_k, n_resamples, seed = args
    return compute_structure_score(runs, task_id, top_k=top_k, n_resamples=n_resamples, seed=seed)


def compute_all_structure_scores(
    task_runs: dict[str, list[Run]],
    min_runs: int = 4,
    top_k: int = 5,
    n_resamples: int = 20,
    seed: int = 42,
    n_workers: int | None = None,
) -> list[StructureScore]:
    """Compute structure scores for all qualifying tasks.

    Uses multiprocessing for parallelism. Set n_workers=1 to disable.
    """
    import multiprocessing as mp

    work_items = []
    for task_id, runs in task_runs.items():
        known = [r for r in runs if r.result.success is not None]
        has_pass = any(r.result.success for r in known)
        has_fail = any(not r.result.success for r in known)
        if len(known) < min_runs or not has_pass or not has_fail:
            continue
        work_items.append((task_id, known, top_k, n_resamples, seed))

    if n_workers == 1 or len(work_items) <= 4:
        scores = [_compute_one(item) for item in work_items]
    else:
        workers = n_workers or min(mp.cpu_count() or 4, len(work_items))
        with mp.Pool(workers) as pool:
            scores = pool.map(_compute_one, work_items)

    result = [s for s in scores if s is not None]
    result.sort(key=lambda s: -s.composite)
    return result
