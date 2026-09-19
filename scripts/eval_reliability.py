"""Eval reliability experiment — does structure-aware resampling improve eval accuracy?

Uses existing multi-run data to simulate what happens when you evaluate with
different allocation strategies under a fixed compute budget.

Key metrics:
1. Absolute error in estimated pass rate vs "true" rate (mean of all runs)
2. False signal rate: how often do two independent n=1 evals of the SAME agent
   disagree by more than a threshold?
3. Budget-matched comparison: n=1 everywhere vs structure-aware allocation

Usage:
    python -m scripts.eval_reliability /Volumes/mnemosyne/moirai/swe_rebench_v2/
"""
from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass

from moirai.load import load_runs
from moirai.analyze.content import select_task_groups
from moirai.analyze.structure import compute_all_structure_scores

# ── Frozen parameters ─────────────────────────────────────────────
SEED = 42
MIN_RUNS = 6        # need enough runs to have a meaningful "true" rate
N_SIMULATIONS = 2000
STRUCTURE_THRESHOLD = 0.20
FALSE_SIGNAL_THRESHOLD = 0.03  # 3pp — would you ship a different model for 3pp?
# ──────────────────────────────────────────────────────────────────


@dataclass
class TaskInfo:
    task_id: str
    true_pass_rate: float
    n_runs: int
    structure_score: float
    is_high_structure: bool
    outcomes: list[bool]  # True=pass, False=fail


@dataclass
class PolicyResult:
    name: str
    mean_absolute_error: float
    median_absolute_error: float
    p90_absolute_error: float
    false_signal_rate: float   # fraction of paired evals that disagree by > threshold
    mean_budget_per_task: float


def compute_true_rates(
    task_groups: dict[str, list],
) -> dict[str, tuple[float, list[bool]]]:
    """Compute empirical pass rate per task from all available runs."""
    result = {}
    for task_id, runs in task_groups.items():
        outcomes = [r.result.success is True for r in runs if r.result.success is not None]
        if len(outcomes) < MIN_RUNS:
            continue
        result[task_id] = (sum(outcomes) / len(outcomes), outcomes)
    return result


def simulate_eval(
    tasks: list[TaskInfo],
    runs_per_task: dict[str, int],
    rng: random.Random,
) -> float:
    """Simulate one evaluation: sample runs_per_task[tid] runs per task, return aggregate pass rate."""
    total_pass = 0
    total_tasks = 0
    for task in tasks:
        k = runs_per_task[task.task_id]
        sampled = rng.choices(task.outcomes, k=k)
        total_pass += sum(sampled) / k  # per-task pass rate estimate
        total_tasks += 1
    return total_pass / total_tasks if total_tasks > 0 else 0.0


def run_policy(
    tasks: list[TaskInfo],
    policy_name: str,
    budget_per_task: float,
    n_simulations: int,
    seed: int,
) -> PolicyResult:
    """Run a resampling policy and measure eval accuracy."""
    rng = random.Random(seed)
    n_tasks = len(tasks)
    total_budget = int(budget_per_task * n_tasks)

    # Compute allocation
    if policy_name == "n=1":
        allocation = {t.task_id: 1 for t in tasks}

    elif policy_name == "uniform":
        k = max(1, int(budget_per_task))
        allocation = {t.task_id: min(k, t.n_runs) for t in tasks}

    elif policy_name == "structure-aware":
        # Spend 1 run on low-structure, remainder goes to high-structure
        high_tasks = [t for t in tasks if t.is_high_structure]
        low_tasks = [t for t in tasks if not t.is_high_structure]
        budget_for_high = total_budget - len(low_tasks)  # 1 run each for low
        if high_tasks and budget_for_high > 0:
            k_high = max(1, budget_for_high // len(high_tasks))
        else:
            k_high = 1
        allocation = {}
        for t in low_tasks:
            allocation[t.task_id] = 1
        for t in high_tasks:
            allocation[t.task_id] = min(k_high, t.n_runs)

    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    actual_budget = sum(allocation.values()) / n_tasks

    # True aggregate pass rate
    true_rate = sum(t.true_pass_rate for t in tasks) / n_tasks

    # Simulate
    errors = []
    paired_diffs = []
    for i in range(0, n_simulations, 2):
        # Run two independent evals (for false signal rate)
        est_a = simulate_eval(tasks, allocation, rng)
        est_b = simulate_eval(tasks, allocation, rng)
        errors.append(abs(est_a - true_rate))
        errors.append(abs(est_b - true_rate))
        paired_diffs.append(abs(est_a - est_b))

    errors.sort()
    mae = sum(errors) / len(errors)
    median_ae = errors[len(errors) // 2]
    p90_ae = errors[int(len(errors) * 0.90)]
    false_signal = sum(1 for d in paired_diffs if d > FALSE_SIGNAL_THRESHOLD) / len(paired_diffs)

    return PolicyResult(
        name=policy_name,
        mean_absolute_error=mae,
        median_absolute_error=median_ae,
        p90_absolute_error=p90_ae,
        false_signal_rate=false_signal,
        mean_budget_per_task=actual_budget,
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.eval_reliability <data_path>")
        sys.exit(1)

    data_path = sys.argv[1]

    print(f"Loading runs from {data_path}...")
    runs, _ = load_runs(data_path)
    print(f"  {len(runs)} runs loaded")

    task_groups, _ = select_task_groups(runs, min_runs=MIN_RUNS)
    print(f"  {len(task_groups)} mixed-outcome tasks with >={MIN_RUNS} runs")

    # Compute true rates
    true_rates = compute_true_rates(task_groups)
    print(f"  {len(true_rates)} tasks with enough data for ground truth")

    # Compute structure scores
    print(f"\nComputing structure scores...")
    t0 = time.time()
    scores = compute_all_structure_scores(task_groups, min_runs=MIN_RUNS, seed=SEED)
    score_map = {s.task_id: s.composite for s in scores}
    print(f"  Done in {time.time() - t0:.0f}s")

    # Build task info
    tasks = []
    for task_id, (true_rate, outcomes) in true_rates.items():
        s = score_map.get(task_id, 0.0)
        tasks.append(TaskInfo(
            task_id=task_id,
            true_pass_rate=true_rate,
            n_runs=len(outcomes),
            structure_score=s,
            is_high_structure=s >= STRUCTURE_THRESHOLD,
            outcomes=outcomes,
        ))

    n_high = sum(1 for t in tasks if t.is_high_structure)
    n_low = len(tasks) - n_high
    true_agg = sum(t.true_pass_rate for t in tasks) / len(tasks)
    print(f"\n  {len(tasks)} tasks: {n_high} high-structure, {n_low} low-structure")
    print(f"  True aggregate pass rate: {true_agg:.1%}")

    # Per-task volatility vs structure score
    from moirai.analyze.stats import kendall_tau_b

    volatilities = []
    struct_scores = []
    for t in tasks:
        p = t.true_pass_rate
        # Bernoulli variance: p*(1-p). Max volatility at p=0.5
        volatilities.append(p * (1 - p))
        struct_scores.append(t.structure_score)

    tau_vol, p_vol = kendall_tau_b(struct_scores, volatilities)
    print(f"\n  Structure score vs task volatility: tau={tau_vol:+.3f}", end="")
    print(f"  p={p_vol:.4f}" if p_vol is not None else "")

    # Run policies at budget = 1.5 runs/task (50% more than n=1)
    budgets = [1.0, 1.5, 2.0, 3.0]

    for budget in budgets:
        print(f"\n{'=' * 65}")
        print(f"  Budget: {budget:.1f} runs/task ({int(budget * len(tasks))} total runs)")
        print(f"{'=' * 65}\n")

        policies = ["n=1", "uniform", "structure-aware"]
        if budget == 1.0:
            policies = ["n=1"]  # uniform and structure-aware collapse to n=1

        results = []
        for policy in policies:
            r = run_policy(tasks, policy, budget, N_SIMULATIONS, SEED)
            results.append(r)

        print(f"  {'Policy':<22s} {'MAE':>7s}  {'Median':>7s}  {'P90':>7s}  {'False signal':>12s}  {'Actual budget':>13s}")
        print(f"  {'─' * 22} {'─' * 7}  {'─' * 7}  {'─' * 7}  {'─' * 12}  {'─' * 13}")
        for r in results:
            fs_str = f"{r.false_signal_rate:.1%}"
            print(
                f"  {r.name:<22s} {r.mean_absolute_error:>6.1%}"
                f"  {r.median_absolute_error:>6.1%}"
                f"  {r.p90_absolute_error:>6.1%}"
                f"  {fs_str:>12s}"
                f"  {r.mean_budget_per_task:>12.1f}"
            )

    print()


if __name__ == "__main__":
    main()
