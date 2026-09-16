"""Fixed-weight contrasts with nested task/checkpoint weights and missingness bounds.

Estimand for selector s and arms (a, b):

    Delta = (1/T) * sum_t (1/C_t) * sum_c [ p(t,c,a) - p(t,c,b) ]

Uncertainty for the frozen-cohort target is a parametric bootstrap over
continuations within each (task, checkpoint, arm) cell using Jeffreys-smoothed
cell probabilities, so all-pass or all-fail cells never collapse to zero
variance. A task-level cluster bootstrap is reported separately for the
optional population target. Missing assigned outcomes are bounded, not dropped.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

from moirai.analyze.stats import kendall_tau_b
from moirai.intervention.schema import CandidateIntervention, Exclusion, TrialRecord

CellKey = tuple[str, str, str, str]   # selector, task, checkpoint, arm


@dataclass
class Cell:
    n_assigned: int = 0
    n_observed: int = 0
    n_success: int = 0

    @property
    def rate(self) -> float | None:
        return self.n_success / self.n_observed if self.n_observed else None

    @property
    def smoothed(self) -> float:
        return (self.n_success + 0.5) / (self.n_observed + 1.0)


def cell_table(records: list[TrialRecord]) -> dict[CellKey, Cell]:
    cells: dict[CellKey, Cell] = {}
    for r in records:
        c = cells.setdefault((r.selector, r.task_id, r.checkpoint_id, r.arm), Cell())
        c.n_assigned += 1
        if r.outcome is not None:
            c.n_observed += 1
            c.n_success += int(r.outcome)
    return cells


@dataclass
class ContrastEstimate:
    selector: str
    arm_a: str
    arm_b: str
    estimate: float
    ci_low: float
    ci_high: float
    bound_low: float          # all missing in a = fail, in b = success
    bound_high: float
    n_tasks: int
    n_checkpoints: int
    n_continuations: int
    n_missing: int
    per_task: dict[str, float]
    task_ci_low: float | None = None      # cluster bootstrap over tasks (population target)
    task_ci_high: float | None = None
    method: str = "jeffreys parametric bootstrap within cells; equal task weight; equal checkpoint weight within task"

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0


def _pairs(cells: dict[CellKey, Cell], selector: str, arm_a: str, arm_b: str) -> dict[str, dict[str, tuple[Cell, Cell]]]:
    """task -> checkpoint -> (cell_a, cell_b) where both arms have observed outcomes."""
    out: dict[str, dict[str, tuple[Cell, Cell]]] = {}
    for (s, t, c, arm), cell in cells.items():
        if s != selector or arm != arm_a:
            continue
        other = cells.get((s, t, c, arm_b))
        if other is None:
            continue
        if cell.n_observed == 0 or other.n_observed == 0:
            continue
        out.setdefault(t, {})[c] = (cell, other)
    return out


def _weighted_delta(pairs: dict[str, dict[str, tuple[float, float]]]) -> tuple[float, dict[str, float]]:
    per_task: dict[str, float] = {}
    for t, cps in pairs.items():
        per_task[t] = sum(pa - pb for pa, pb in cps.values()) / len(cps)
    return (sum(per_task.values()) / len(per_task)) if per_task else 0.0, per_task


def contrast(
    cells: dict[CellKey, Cell],
    selector: str,
    arm_a: str,
    arm_b: str,
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> ContrastEstimate | None:
    pairs = _pairs(cells, selector, arm_a, arm_b)
    if not pairs:
        return None

    point, per_task = _weighted_delta({t: {c: (a.rate, b.rate) for c, (a, b) in cps.items()} for t, cps in pairs.items()})  # type: ignore[misc]

    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        sim: dict[str, dict[str, tuple[float, float]]] = {}
        for t, cps in pairs.items():
            sim[t] = {}
            for c, (a, b) in cps.items():
                sa = sum(1 for _ in range(a.n_observed) if rng.random() < a.smoothed) / a.n_observed
                sb = sum(1 for _ in range(b.n_observed) if rng.random() < b.smoothed) / b.n_observed
                sim[t][c] = (sa, sb)
        boots.append(_weighted_delta(sim)[0])
    boots.sort()
    lo_i = int((1 - level) / 2 * n_boot)
    hi_i = min(n_boot - 1, int((1 + level) / 2 * n_boot))
    ci_low, ci_high = boots[lo_i], boots[hi_i]

    # Missing-outcome identification bounds under fixed weights.
    low_pairs = {}
    high_pairs = {}
    n_missing = 0
    n_cont = 0
    n_ckpt = 0
    for t, cps in pairs.items():
        low_pairs[t] = {}
        high_pairs[t] = {}
        for c, (a, b) in cps.items():
            n_ckpt += 1
            n_cont += a.n_assigned + b.n_assigned
            miss_a, miss_b = a.n_assigned - a.n_observed, b.n_assigned - b.n_observed
            n_missing += miss_a + miss_b
            low_pairs[t][c] = (a.n_success / a.n_assigned, (b.n_success + miss_b) / b.n_assigned)
            high_pairs[t][c] = ((a.n_success + miss_a) / a.n_assigned, b.n_success / b.n_assigned)
    bound_low = _weighted_delta(low_pairs)[0]
    bound_high = _weighted_delta(high_pairs)[0]

    task_lo = task_hi = None
    tasks = sorted(per_task)
    if len(tasks) >= 2:
        trng = random.Random(seed + 1)
        tb = []
        for _ in range(n_boot):
            sample = [per_task[trng.choice(tasks)] for _ in tasks]
            tb.append(sum(sample) / len(sample))
        tb.sort()
        task_lo, task_hi = tb[lo_i], tb[hi_i]

    return ContrastEstimate(
        selector=selector, arm_a=arm_a, arm_b=arm_b, estimate=point,
        ci_low=ci_low, ci_high=ci_high, bound_low=bound_low, bound_high=bound_high,
        n_tasks=len(pairs), n_checkpoints=n_ckpt, n_continuations=n_cont, n_missing=n_missing,
        per_task=per_task, task_ci_low=task_lo, task_ci_high=task_hi,
    )


# ── Selector-level report ─────────────────────────────────────────


@dataclass
class SelectorReport:
    selector: str
    n_tasks_considered: int
    n_candidates: int
    n_exclusions: int
    exclusion_reasons: dict[str, int]
    coverage: float
    selection_wall_seconds: float
    selection_cost_usd: float
    n_trials: int
    n_outcomes: int
    n_infra_failures: int
    execution_cost_usd: float
    cost_per_actionable_checkpoint_usd: float | None
    uplift: ContrastEstimate | None
    branch: ContrastEstimate | None
    structure_tau: float | None = None
    structure_tau_p: float | None = None
    per_task_structure: dict[str, float | None] = field(default_factory=dict)


def selector_report(
    selector: str,
    records: list[TrialRecord],
    candidates: list[CandidateIntervention],
    exclusions: list[Exclusion],
    n_boot: int = 2000,
    seed: int = 0,
) -> SelectorReport:
    cands = [c for c in candidates if c.selector == selector]
    excls = [e for e in exclusions if e.selector == selector]
    recs = [r for r in records if r.selector == selector]
    cells = cell_table(recs)
    uplift = contrast(cells, selector, "favored", "native", n_boot=n_boot, seed=seed)
    branch = contrast(cells, selector, "favored", "disfavored", n_boot=n_boot, seed=seed)

    n_considered = len(cands) + len(excls)
    sel_wall = sum(c.selection_cost.wall_seconds for c in cands) + sum(e.selection_cost.wall_seconds for e in excls)
    sel_cost = sum(c.selection_cost.cost_usd for c in cands) + sum(e.selection_cost.cost_usd for e in excls)
    exec_cost = sum(float(r.usage.get("cost_usd", 0.0)) for r in recs)
    n_ckpt = len({(r.task_id, r.checkpoint_id) for r in recs})
    per_ckpt = (sel_cost + exec_cost) / n_ckpt if n_ckpt else None

    tau = tau_p = None
    per_struct = {c.task_id: c.structure_score for c in cands}
    if uplift is not None:
        xs, ys = [], []
        for t, eff in uplift.per_task.items():
            s = per_struct.get(t)
            if s is not None:
                xs.append(s)
                ys.append(eff)
        if len(xs) >= 3:
            tau, tau_p = kendall_tau_b(xs, ys)

    return SelectorReport(
        selector=selector, n_tasks_considered=n_considered, n_candidates=len(cands), n_exclusions=len(excls),
        exclusion_reasons=dict(Counter(e.reason for e in excls)),
        coverage=(len(cands) / n_considered) if n_considered else 0.0,
        selection_wall_seconds=sel_wall, selection_cost_usd=sel_cost,
        n_trials=len(recs), n_outcomes=sum(1 for r in recs if r.outcome is not None),
        n_infra_failures=sum(1 for r in recs if r.status == "infra_failure"),
        execution_cost_usd=exec_cost, cost_per_actionable_checkpoint_usd=per_ckpt,
        uplift=uplift, branch=branch, structure_tau=tau, structure_tau_p=tau_p, per_task_structure=per_struct,
    )


@dataclass
class SelectorComparison:
    selector_a: str
    selector_b: str
    n_shared_tasks: int
    mean_uplift_difference: float | None      # mean over shared tasks of uplift_a - uplift_b
    n_tasks_a_better: int
    n_tasks_b_better: int
    n_tasks_tied: int


def compare_selectors(reports: list[SelectorReport]) -> list[SelectorComparison]:
    out: list[SelectorComparison] = []
    for i, ra in enumerate(reports):
        for rb in reports[i + 1:]:
            if ra.uplift is None or rb.uplift is None:
                out.append(SelectorComparison(ra.selector, rb.selector, 0, None, 0, 0, 0))
                continue
            shared = sorted(set(ra.uplift.per_task) & set(rb.uplift.per_task))
            diffs = [ra.uplift.per_task[t] - rb.uplift.per_task[t] for t in shared]
            out.append(SelectorComparison(
                ra.selector, rb.selector, len(shared),
                (sum(diffs) / len(diffs)) if diffs else None,
                sum(1 for d in diffs if d > 0), sum(1 for d in diffs if d < 0), sum(1 for d in diffs if d == 0),
            ))
    return out


def neutral_summary(records: list[TrialRecord]) -> dict[str, dict[str, int]]:
    """Descriptive counts with arm labels replaced by opaque codes.

    For execution monitoring before orientation is revealed.
    """
    import hashlib

    codes = {}
    out: dict[str, dict[str, int]] = {}
    for r in records:
        code = codes.setdefault(r.arm, "arm-" + hashlib.sha256(f"neutral:{r.arm}".encode()).hexdigest()[:6])
        d = out.setdefault(code, {"assigned": 0, "observed": 0, "infra_failure": 0})
        d["assigned"] += 1
        d["observed"] += int(r.outcome is not None)
        d["infra_failure"] += int(r.status == "infra_failure")
    return out
