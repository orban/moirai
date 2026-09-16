"""Exact-procedure power simulation: generates ledgers, runs the real estimator."""
from __future__ import annotations

import random
from dataclasses import dataclass

from moirai.intervention.analyze import cell_table, contrast
from moirai.intervention.schema import TrialRecord


@dataclass(frozen=True)
class Design:
    n_tasks: int
    checkpoints_per_task: int
    reps_per_arm: int
    branch_gap: float           # p(favored) - p(disfavored), population mean
    native_offset: float = 0.0  # p(native) - midpoint(favored, disfavored)
    baseline_low: float = 0.25
    baseline_high: float = 0.65
    task_gap_jitter: float = 0.15
    missing_rate: float = 0.0


@dataclass
class PowerResult:
    design: Design
    n_sims: int
    branch_rejection_rate: float
    branch_coverage: float
    branch_mean_ci_width: float
    uplift_rejection_rate: float
    uplift_coverage: float
    uplift_mean_ci_width: float
    uplift_truth: float


def _synthetic_records(d: Design, rng: random.Random, selector: str = "sim") -> tuple[list[TrialRecord], float, float]:
    recs: list[TrialRecord] = []
    true_branch = 0.0
    true_uplift = 0.0
    n = 0
    for t in range(d.n_tasks):
        base = rng.uniform(d.baseline_low, d.baseline_high)
        gap = d.branch_gap + rng.uniform(-d.task_gap_jitter, d.task_gap_jitter)
        for c in range(d.checkpoints_per_task):
            b = min(0.98, max(0.02, base + rng.uniform(-0.04, 0.04)))
            g = gap + rng.uniform(-0.03, 0.03)
            p = {"favored": min(1.0, max(0.0, b + g / 2)), "disfavored": min(1.0, max(0.0, b - g / 2)),
                 "native": min(1.0, max(0.0, b + d.native_offset))}
            true_branch += p["favored"] - p["disfavored"]
            true_uplift += p["favored"] - p["native"]
            for arm, pa in p.items():
                for r in range(d.reps_per_arm):
                    missing = rng.random() < d.missing_rate
                    outcome = None if missing else (rng.random() < pa)
                    recs.append(TrialRecord(
                        trial_id=f"{t}-{c}-{arm}-{r}", task_id=f"t{t}", checkpoint_id=f"c{c}", candidate_id=f"k{t}{c}",
                        selector=selector, arm=arm, block=r, seed=n, status="infra_failure" if missing else ("success" if outcome else "failure"),
                        outcome=outcome, failure_class=None, grade=None, usage={}, model_config_hash="", prompt_hash="",
                        tools_hash="", environment_hash="", workspace_hash_before="", workspace_hash_after=None,
                        injected_tool_call_id=None, injected_payload_hash=None, trajectory_ref=None, retry_of=None,
                        started_at="", finished_at=None,
                    ))
                    n += 1
    k = d.n_tasks * d.checkpoints_per_task
    return recs, true_branch / k, true_uplift / k


def simulate(design: Design, n_sims: int = 200, seed: int = 20260905, n_boot: int = 400) -> PowerResult:
    rng = random.Random(seed)
    b_rej = b_cov = b_w = 0.0
    u_rej = u_cov = u_w = 0.0
    u_truths = []
    for i in range(n_sims):
        recs, tb, tu = _synthetic_records(design, rng)
        cells = cell_table(recs)
        b = contrast(cells, "sim", "favored", "disfavored", n_boot=n_boot, seed=seed + i)
        u = contrast(cells, "sim", "favored", "native", n_boot=n_boot, seed=seed + i)
        assert b is not None and u is not None
        b_rej += b.excludes_zero
        b_cov += b.ci_low <= tb <= b.ci_high
        b_w += b.ci_high - b.ci_low
        u_rej += u.excludes_zero
        u_cov += u.ci_low <= tu <= u.ci_high
        u_w += u.ci_high - u.ci_low
        u_truths.append(tu)
    return PowerResult(
        design=design, n_sims=n_sims,
        branch_rejection_rate=b_rej / n_sims, branch_coverage=b_cov / n_sims, branch_mean_ci_width=b_w / n_sims,
        uplift_rejection_rate=u_rej / n_sims, uplift_coverage=u_cov / n_sims, uplift_mean_ci_width=u_w / n_sims,
        uplift_truth=sum(u_truths) / len(u_truths),
    )
