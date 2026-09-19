"""Held-out prediction study — test whether divergence points predict outcomes on unseen runs."""
from __future__ import annotations

import random
from dataclasses import dataclass

from moirai.analyze.align import _nw_align, _get_sequence, align_runs, consensus
from moirai.analyze.divergence import find_divergence_points
from moirai.analyze.features import FEATURES
from moirai.schema import Alignment, DivergencePoint, GAP, Run


# ── Data structures ───────────────────────────────────────────────


@dataclass
class DivergenceModel:
    """Trained divergence model for one task — stores branch success rates."""
    task_id: str
    consensus: list[str]
    divergence_points: list[DivergencePoint]
    train_pass_rate: float
    n_train_runs: int


@dataclass
class MethodResult:
    """AUROC and other metrics for one scoring method."""
    name: str
    auroc: float
    mean_auroc: float             # mean of per-task AUROCs (Simpson's-resistant)
    accuracy_at_3: float | None   # accuracy when selecting top-3 scored runs
    n_scored: int
    win_rate_vs: dict[str, float] | None = None  # method -> fraction of tasks this method wins


@dataclass
class HoldoutResults:
    """Full results of the held-out prediction study."""
    n_tasks: int
    n_train_runs: int
    n_test_runs: int
    method_results: list[MethodResult]
    per_task: dict[str, dict[str, float]]  # task_id -> method -> auroc


# ── Splitting ─────────────────────────────────────────────────────


def split_runs(
    task_runs: dict[str, list[Run]],
    train_frac: float = 0.7,
    seed: int = 42,
    min_train_pass: int = 2,
    min_train_fail: int = 2,
    min_test: int = 2,
) -> dict[str, tuple[list[Run], list[Run]]]:
    """Split runs per task into train/test, stratified by outcome.

    Returns {task_id: (train_runs, test_runs)} for tasks with enough data.
    """
    rng = random.Random(seed)
    result: dict[str, tuple[list[Run], list[Run]]] = {}

    for task_id, runs in task_runs.items():
        pass_runs = [r for r in runs if r.result.success is True]
        fail_runs = [r for r in runs if r.result.success is False]

        if len(pass_runs) < min_train_pass + 1 or len(fail_runs) < min_train_fail + 1:
            continue

        # Stratified split: split pass and fail independently
        rng.shuffle(pass_runs)
        rng.shuffle(fail_runs)

        n_train_pass = max(min_train_pass, int(len(pass_runs) * train_frac))
        n_train_fail = max(min_train_fail, int(len(fail_runs) * train_frac))

        # Ensure at least 1 of each in test
        n_train_pass = min(n_train_pass, len(pass_runs) - 1)
        n_train_fail = min(n_train_fail, len(fail_runs) - 1)

        train = pass_runs[:n_train_pass] + fail_runs[:n_train_fail]
        test = pass_runs[n_train_pass:] + fail_runs[n_train_fail:]

        if len(test) < min_test:
            continue

        rng.shuffle(train)
        rng.shuffle(test)
        result[task_id] = (train, test)

    return result


# ── Divergence model ──────────────────────────────────────────────


def train_divergence_model(
    train_runs: list[Run],
    task_id: str,
    level: str = "name",
    q_threshold: float = 0.10,
    min_branch_size: int = 2,
) -> DivergenceModel | None:
    """Build a divergence model from training runs.

    Aligns runs, finds significant divergence points, stores branch success rates.
    Returns None if alignment fails or no divergence points found.
    """
    if len(train_runs) < 4:
        return None

    alignment = align_runs(train_runs, level=level)
    if not alignment.matrix or not alignment.matrix[0]:
        return None

    points, _ = find_divergence_points(
        alignment, train_runs,
        min_branch_size=min_branch_size,
        q_threshold=q_threshold,
    )

    if not points:
        return None

    consensus_seq = consensus(alignment.matrix)

    n_pass = sum(1 for r in train_runs if r.result.success is True)
    pass_rate = n_pass / len(train_runs) if train_runs else 0.5

    return DivergenceModel(
        task_id=task_id,
        consensus=consensus_seq,
        divergence_points=points,
        train_pass_rate=pass_rate,
        n_train_runs=len(train_runs),
    )


# ── Scoring methods ───────────────────────────────────────────────


def _align_run_to_consensus(
    run: Run,
    consensus_seq: list[str],
    level: str = "name",
) -> dict[int, str]:
    """Align a single run against a consensus, return {consensus_col: run_value}.

    Maps each consensus column to the value the run has at the
    corresponding position in the alignment.
    """
    run_seq = _get_sequence(run, level)
    if not run_seq or not consensus_seq:
        return {}

    aligned_run, aligned_con = _nw_align(run_seq, consensus_seq)

    # Walk through alignment, tracking which consensus column each position maps to
    mapping: dict[int, str] = {}
    con_col = 0
    for i in range(len(aligned_con)):
        if aligned_con[i] != GAP:
            # This position corresponds to consensus column con_col
            mapping[con_col] = aligned_run[i]
            con_col += 1
        # If aligned_con[i] is GAP, it's an insertion in the run — skip

    return mapping


def score_divergence(
    run: Run,
    model: DivergenceModel,
    top_k: int | None = None,
    level: str = "name",
) -> float:
    """Score a run using divergence-point branch matching.

    For each divergence point, checks which branch the run takes.
    Returns the mean branch success rate across matched divergence points.
    Falls back to 0.5 (neutral) for unmatched points.
    """
    if not model.divergence_points:
        return 0.5

    mapping = _align_run_to_consensus(run, model.consensus, level=level)
    if not mapping:
        return 0.5

    points = model.divergence_points
    if top_k is not None:
        points = points[:top_k]

    scores: list[float] = []
    for dp in points:
        run_value = mapping.get(dp.column, GAP)

        if run_value == GAP:
            scores.append(0.5)
            continue

        branch_rate = dp.success_by_value.get(run_value)
        if branch_rate is not None:
            scores.append(branch_rate)
        else:
            # Run has a value not seen in training — neutral
            scores.append(0.5)

    return sum(scores) / len(scores) if scores else 0.5


def score_features(run: Run) -> float:
    """Score a run using moirai's behavioral features.

    Computes each feature, normalizes by direction (positive = higher is better),
    and returns the mean. Missing features get 0.5.
    """
    values: list[float] = []
    for spec in FEATURES:
        val = spec.compute(run)
        if val is None:
            values.append(0.5)
            continue
        # Clamp to [0, 1] — most features are naturally in this range
        clamped = max(0.0, min(1.0, val))
        if spec.direction == "negative":
            clamped = 1.0 - clamped
        values.append(clamped)

    return sum(values) / len(values) if values else 0.5


def score_step_count(run: Run) -> float:
    """Score a run by inverse step count. Shorter = higher score."""
    n = len(run.steps)
    if n == 0:
        return 0.5
    # Sigmoid-ish: map step counts to [0, 1], centered around 50 steps
    return 1.0 / (1.0 + n / 50.0)


def score_random(run: Run, rng: random.Random) -> float:
    """Random baseline score."""
    return rng.random()


# ── AUROC computation ─────────────────────────────────────────────


def compute_auroc(scored: list[tuple[float, bool]]) -> float:
    """Compute AUROC from (score, is_positive) pairs.

    Uses the Mann-Whitney U statistic formulation.
    Returns 0.5 if degenerate (all same class or empty).
    """
    positives = [s for s, label in scored if label]
    negatives = [s for s, label in scored if not label]

    if not positives or not negatives:
        return 0.5

    # Count concordant pairs: P(score_pos > score_neg)
    concordant = 0
    tied = 0
    for p in positives:
        for n in negatives:
            if p > n:
                concordant += 1
            elif p == n:
                tied += 1

    total = len(positives) * len(negatives)
    return (concordant + 0.5 * tied) / total


def compute_accuracy_at_k(scored: list[tuple[float, bool]], k: int = 3) -> float | None:
    """Among the top-K scored runs, what fraction actually passed?"""
    if len(scored) < k:
        return None
    sorted_runs = sorted(scored, key=lambda x: -x[0])
    top_k = sorted_runs[:k]
    return sum(1 for _, label in top_k if label) / k


# ── Main study ────────────────────────────────────────────────────


def run_holdout_study(
    task_runs: dict[str, list[Run]],
    train_frac: float = 0.7,
    seed: int = 42,
    min_runs: int = 6,
    top_k_divergence: int | None = 10,
) -> HoldoutResults:
    """Run the full held-out prediction study.

    For each task:
    1. Split runs into train/test
    2. Build divergence model from train runs
    3. Score test runs with each method
    4. Compute AUROC per task per method

    Aggregates across tasks.
    """
    # Filter to mixed-outcome tasks with enough runs
    qualifying = {
        tid: runs for tid, runs in task_runs.items()
        if len([r for r in runs if r.result.success is not None]) >= min_runs
    }

    splits = split_runs(qualifying, train_frac=train_frac, seed=seed)

    rng = random.Random(seed + 1)  # separate RNG for random scorer
    methods = ["divergence", "features", "step_count", "random"]

    # Collect all scored runs across tasks
    all_scored: dict[str, list[tuple[float, bool]]] = {m: [] for m in methods}
    per_task_auroc: dict[str, dict[str, float]] = {}
    total_train = 0
    total_test = 0

    for task_id, (train, test) in splits.items():
        total_train += len(train)
        total_test += len(test)

        # Build divergence model
        model = train_divergence_model(train, task_id)

        task_scores: dict[str, list[tuple[float, bool]]] = {m: [] for m in methods}

        for run in test:
            success = run.result.success is True

            # Divergence score
            if model is not None:
                div_score = score_divergence(run, model, top_k=top_k_divergence)
            else:
                div_score = 0.5
            task_scores["divergence"].append((div_score, success))

            # Feature score
            feat_score = score_features(run)
            task_scores["features"].append((feat_score, success))

            # Step count score
            sc_score = score_step_count(run)
            task_scores["step_count"].append((sc_score, success))

            # Random score
            rand_score = score_random(run, rng)
            task_scores["random"].append((rand_score, success))

        # Per-task AUROC
        per_task_auroc[task_id] = {}
        for m in methods:
            pairs = task_scores[m]
            has_both = (any(label for _, label in pairs) and any(not label for _, label in pairs))
            if has_both:
                per_task_auroc[task_id][m] = compute_auroc(pairs)
            all_scored[m].extend(pairs)

    # Per-task win rates: for each method pair, on what fraction of tasks
    # does method A beat method B (by per-task AUROC)?
    # This prevents high-run-count tasks from dominating the aggregate.
    win_rates: dict[str, dict[str, float]] = {m: {} for m in methods}
    for m_a in methods:
        for m_b in methods:
            if m_a == m_b:
                continue
            wins = 0
            comparable = 0
            for task_id, task_aurocs in per_task_auroc.items():
                if m_a in task_aurocs and m_b in task_aurocs:
                    comparable += 1
                    if task_aurocs[m_a] > task_aurocs[m_b]:
                        wins += 1
            win_rates[m_a][m_b] = wins / comparable if comparable > 0 else 0.5

    # Aggregate AUROC and accuracy@3
    method_results = []
    for m in methods:
        pairs = all_scored[m]
        auroc = compute_auroc(pairs)
        acc_at_3 = compute_accuracy_at_k(pairs, k=3)
        # Mean per-task AUROC (resistant to task-size imbalance)
        task_aurocs = [per_task_auroc[t][m] for t in per_task_auroc if m in per_task_auroc[t]]
        m_auroc = sum(task_aurocs) / len(task_aurocs) if task_aurocs else 0.5
        method_results.append(MethodResult(
            name=m, auroc=auroc, mean_auroc=m_auroc,
            accuracy_at_3=acc_at_3, n_scored=len(pairs),
            win_rate_vs=win_rates[m],
        ))

    # Sort by AUROC descending
    method_results.sort(key=lambda r: -r.auroc)

    return HoldoutResults(
        n_tasks=len(splits),
        n_train_runs=total_train,
        n_test_runs=total_test,
        method_results=method_results,
        per_task=per_task_auroc,
    )


