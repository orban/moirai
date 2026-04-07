"""Reranking experiment — test whether moirai-derived scoring improves best-of-K selection."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from moirai.analyze.features import FEATURES
from moirai.analyze.holdout import (
    score_divergence,
    score_features,
    train_divergence_model,
)
from moirai.schema import Run


@dataclass
class RerankMethodResult:
    """Results for one scoring method across all tasks."""
    name: str
    selection_accuracy: float       # P(selected run passes)
    lift_over_random: float         # selection_accuracy - random_accuracy
    ci_lower: float                 # bootstrap 95% CI lower
    ci_upper: float                 # bootstrap 95% CI upper
    n_samples: int


@dataclass
class FamilyResult:
    """Reranking results for one task family."""
    family: str
    n_tasks: int
    random_acc: float
    oracle_acc: float
    method_accs: dict[str, float]           # method -> selection accuracy
    captured_oracle_gap: dict[str, float]   # method -> (acc - random) / (oracle - random)


@dataclass
class RerankResults:
    """Full results of the reranking experiment."""
    k: int
    n_tasks: int
    n_samples_per_task: int
    random_pass_at_1: float         # overall pass@1 (single random run)
    random_best_of_k: float         # expected pass rate from random selection of K
    oracle_best_of_k: float         # upper bound: always pick a passing run if one exists
    method_results: list[RerankMethodResult]
    per_task: dict[str, dict[str, float]]  # task_id -> method -> selection accuracy
    per_family: list[FamilyResult] | None = None


def _build_scorers(
    task_runs: list[Run],
    task_id: str,
) -> dict[str, object]:
    """Build all scoring functions for a task, including divergence model."""
    # Train divergence model on ALL runs for this task
    # (reranking is not a held-out study — the model sees all available data)
    model = train_divergence_model(task_runs, task_id)

    def divergence_scorer(run: Run) -> float:
        if model is None:
            return 0.5
        return score_divergence(run, model)

    def feature_scorer(run: Run) -> float:
        return score_features(run)

    unc_spec = next((s for s in FEATURES if s.name == "uncertainty_density"), None)

    def uncertainty_scorer(run: Run) -> float:
        """Single-feature scorer: inverted uncertainty density."""
        if unc_spec is None:
            return 0.5
        val = unc_spec.compute(run)
        if val is None:
            return 0.5
        return 1.0 - max(0.0, min(1.0, val))

    return {
        "divergence": divergence_scorer,
        "features": feature_scorer,
        "uncertainty_inv": uncertainty_scorer,
        "longest": lambda r: float(len(r.steps)),
    }


def _wilson_ci(
    outcomes: list[float],
) -> tuple[float, float]:
    """Wilson score interval (95%) for binary outcomes — O(1).

    More accurate than normal approximation for small n or extreme proportions.
    outcomes should be 0.0/1.0 values.
    """
    n = len(outcomes)
    if n == 0:
        return 0.0, 0.0

    p = sum(outcomes) / n
    z = 1.96
    denominator = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


_METHOD_NAMES = ["divergence", "features", "uncertainty_inv", "longest", "random"]
_SCORED_METHODS = ["divergence", "features", "uncertainty_inv", "longest"]


@dataclass
class _TaskResult:
    """Per-task reranking result (for multiprocessing)."""
    task_id: str
    family: str
    per_method_acc: dict[str, float]
    oracle_rate: float
    n_pass: int
    n_runs: int


def _run_one_task(args: tuple) -> _TaskResult:
    """Process one task's reranking experiment. Picklable for multiprocessing."""
    task_id, runs, k, n_samples, task_seed = args

    rng = random.Random(task_seed)
    family = runs[0].task_family if runs and runs[0].task_family else "_unknown_"

    scorers = _build_scorers(runs, task_id)
    precomputed: dict[str, dict[str, float]] = {}
    for method, scorer in scorers.items():
        precomputed[method] = {r.run_id: scorer(r) for r in runs}

    task_outcomes: dict[str, list[bool]] = {m: [] for m in _METHOD_NAMES}
    oracle_hits = 0

    for _ in range(n_samples):
        sample = rng.sample(runs, k)

        if any(r.result.success for r in sample):
            oracle_hits += 1

        random_pick = rng.choice(sample)
        task_outcomes["random"].append(random_pick.result.success is True)

        for method in _SCORED_METHODS:
            scores = precomputed[method]
            best_run = max(sample, key=lambda r: scores[r.run_id])
            task_outcomes[method].append(best_run.result.success is True)

    per_method_acc = {}
    for m in _METHOD_NAMES:
        per_method_acc[m] = sum(task_outcomes[m]) / len(task_outcomes[m]) if task_outcomes[m] else 0.0

    return _TaskResult(
        task_id=task_id,
        family=family,
        per_method_acc=per_method_acc,
        oracle_rate=oracle_hits / n_samples,
        n_pass=sum(1 for r in runs if r.result.success),
        n_runs=len(runs),
    )


def rerank_experiment(
    task_runs: dict[str, list[Run]],
    k: int = 3,
    n_samples: int = 1000,
    seed: int = 42,
    min_runs: int = 4,
    n_workers: int | None = None,
) -> RerankResults:
    """Run the reranking experiment.

    For each task with mixed outcomes:
    1. Build scoring models from all runs
    2. Repeatedly sample K runs
    3. Select the top-scored run
    4. Measure whether it passes

    Uses multiprocessing for parallelism. Set n_workers=1 to disable.
    """
    import multiprocessing as mp

    # Filter to mixed-outcome tasks
    work_items = []
    for tid, runs in task_runs.items():
        known = [r for r in runs if r.result.success is not None]
        if len(known) < max(k, min_runs):
            continue
        has_pass = any(r.result.success for r in known)
        has_fail = any(not r.result.success for r in known)
        if has_pass and has_fail:
            # Each task gets its own deterministic seed (not hash() which is randomized per process)
            task_seed = seed + sum(ord(c) * (i + 1) for i, c in enumerate(tid)) % (2**31)
            work_items.append((tid, known, k, n_samples, task_seed))

    if n_workers == 1 or len(work_items) <= 4:
        task_results = [_run_one_task(item) for item in work_items]
    else:
        workers = n_workers or min(mp.cpu_count() or 4, len(work_items))
        with mp.Pool(workers) as pool:
            task_results = pool.map(_run_one_task, work_items)

    # Aggregate from per-task results
    per_task: dict[str, dict[str, float]] = {}
    per_task_oracle: dict[str, float] = {}
    task_family_map: dict[str, str] = {}
    all_outcomes: dict[str, list[float]] = {m: [] for m in _METHOD_NAMES}
    total_pass_at_1 = 0
    total_runs_for_pass_at_1 = 0

    for tr in task_results:
        per_task[tr.task_id] = tr.per_method_acc
        per_task_oracle[tr.task_id] = tr.oracle_rate
        task_family_map[tr.task_id] = tr.family
        total_pass_at_1 += tr.n_pass
        total_runs_for_pass_at_1 += tr.n_runs
        for m in _METHOD_NAMES:
            # Expand back to per-sample outcomes for aggregate CI
            acc = tr.per_method_acc[m]
            n_success = round(acc * n_samples)
            all_outcomes[m].extend([1.0] * n_success + [0.0] * (n_samples - n_success))

    random_pass_at_1 = total_pass_at_1 / total_runs_for_pass_at_1 if total_runs_for_pass_at_1 else 0.0
    random_acc = sum(all_outcomes["random"]) / len(all_outcomes["random"]) if all_outcomes["random"] else 0.0
    oracle_rate = sum(per_task_oracle.values()) / len(per_task_oracle) if per_task_oracle else 0.0

    method_results = []
    for m in _METHOD_NAMES:
        outcomes = all_outcomes[m]
        if not outcomes:
            continue
        acc = sum(outcomes) / len(outcomes)
        ci_lo, ci_hi = _wilson_ci(outcomes)
        method_results.append(RerankMethodResult(
            name=m,
            selection_accuracy=acc,
            lift_over_random=acc - random_acc,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            n_samples=len(outcomes),
        ))

    # Sort by selection accuracy descending
    method_results.sort(key=lambda r: -r.selection_accuracy)

    # Per-family aggregation: captured oracle gap by task family
    families: dict[str, list[str]] = {}  # family -> [task_ids]
    for tid, fam in task_family_map.items():
        families.setdefault(fam, []).append(tid)

    per_family: list[FamilyResult] = []
    for fam, task_ids in sorted(families.items()):
        if len(task_ids) < 2:
            continue
        # Family-level random and oracle rates (averaged across tasks, not samples)
        fam_random = sum(per_task[t].get("random", 0) for t in task_ids) / len(task_ids)
        fam_oracle = sum(
            per_task_oracle.get(t, 0.5) for t in task_ids
        ) / len(task_ids)

        fam_method_accs: dict[str, float] = {}
        fam_captured: dict[str, float] = {}
        oracle_gap = fam_oracle - fam_random
        for m in _METHOD_NAMES:
            fam_acc = sum(per_task[t].get(m, 0) for t in task_ids) / len(task_ids)
            fam_method_accs[m] = fam_acc
            if oracle_gap > 0.01:
                fam_captured[m] = (fam_acc - fam_random) / oracle_gap
            else:
                fam_captured[m] = 0.0

        per_family.append(FamilyResult(
            family=fam,
            n_tasks=len(task_ids),
            random_acc=fam_random,
            oracle_acc=fam_oracle,
            method_accs=fam_method_accs,
            captured_oracle_gap=fam_captured,
        ))

    # Sort families by number of tasks descending
    per_family.sort(key=lambda f: -f.n_tasks)

    return RerankResults(
        k=k,
        n_tasks=len(task_results),
        n_samples_per_task=n_samples,
        random_pass_at_1=random_pass_at_1,
        random_best_of_k=random_acc,
        oracle_best_of_k=oracle_rate,
        method_results=method_results,
        per_task=per_task,
        per_family=per_family,
    )
