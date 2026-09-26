"""Diagnosability of a run set — can these runs support divergence analysis at all?

Every branch-point method in moirai compares runs against each other at some
position and asks which choice predicted the outcome. That question is only
answerable if the runs actually differ in *which* components they touch. A set of
runs that all do the same thing, or a set where every run is unique from step one,
carries no information to localize with, however many runs you collect.

Spectrum-based fault localization hit this twenty years ago and measures it with
DDU (Perez, Abreu & van Deursen, "A Test-Suite Diagnosability Metric for
Spectrum-Based Fault Localization Approaches", ICSE 2017, doi:10.1109/ICSE.2017.66).
DDU scores a coverage matrix on three axes and multiplies them. It needs no
pass/fail labels and no ground truth about where the fault is, which is what makes
it usable *before* spending compute rather than after.

The port here is a straight one. Their transactions (tests) are our runs; their
components (statements, branches) are our step signatures. A_ij = 1 when run i
executed component j at least once.

    density     rho  = sum(A) / (N*M), renormalized as rho' = 1 - |1 - 2*rho|
                Their ideal is rho = 0.5, not 1.0: a component involved in half the
                transactions carries the most information about which one failed.
                rho' maps that onto [0,1] with 1.0 best, so the product below behaves.
    diversity   Gini-Simpson over *distinct rows*. 1.0 when every run touches a
                different set of components, 0.0 when they are all identical.
    uniqueness  distinct columns / M. Components with identical columns form an
                "ambiguity group" -- nothing can tell them apart, because they are
                always hit together.

    DDU = rho' * diversity * uniqueness

Any term at zero collapses the product, which is the intended behaviour: a run set
that fails on any one axis is undiagnosable regardless of the other two.

What this does NOT tell you: whether a signal exists. It tells you whether your run
set could express one. A high DDU with no predictive signal is a real negative
result; a low DDU with no predictive signal means you never ran the experiment.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable

from moirai.schema import Run


@dataclass
class Diagnosability:
    """DDU and its three components for one run set."""
    task_id: str
    ddu: float
    density: float           # rho', renormalized so 1.0 is ideal
    raw_density: float       # rho as measured, ideal is 0.5
    diversity: float         # Gini-Simpson over distinct activity rows
    uniqueness: float        # distinct columns / total components
    n_runs: int              # N, rows
    n_components: int        # M, columns
    n_distinct_rows: int
    n_ambiguity_groups: int


def step_name_signature(step) -> str:
    """Coarse alphabet: the step name alone."""
    return step.name


def step_target_signature(step) -> str:
    """Finer alphabet: the step name plus what it touched."""
    attrs = step.attrs or {}
    target = attrs.get("file_path") or attrs.get("command") or ""
    return f"{step.name}|{str(target)[:60]}" if target else step.name


def build_activity_matrix(
    runs: list[Run],
    signature: Callable = step_name_signature,
) -> tuple[list[frozenset[str]], list[str]]:
    """Return (one row per run as a set of component ids, sorted component list).

    Sparse on purpose. The dense matrix is N*M booleans and every term below is
    computable from row-sets and column-sets without materializing it.
    """
    rows = [frozenset(signature(s) for s in r.steps) for r in runs]
    components = sorted({c for row in rows for c in row})
    return rows, components


def _density(rows: list[frozenset[str]], n_components: int) -> tuple[float, float]:
    """Return (raw rho, renormalized rho'). Ideal rho is 0.5; ideal rho' is 1.0."""
    if not rows or n_components == 0:
        return 0.0, 0.0
    hits = sum(len(row) for row in rows)
    rho = hits / (len(rows) * n_components)
    return rho, 1.0 - abs(1.0 - 2.0 * rho)


def _diversity(rows: list[frozenset[str]]) -> tuple[float, int]:
    """Gini-Simpson over distinct rows. Returns (diversity, n_distinct_rows).

    Undefined for a single run -- one row cannot be more or less diverse than
    itself -- so N < 2 returns 0.0 rather than a made-up value.
    """
    n = len(rows)
    if n < 2:
        return 0.0, n
    groups = Counter(rows)
    collisions = sum(c * (c - 1) for c in groups.values())
    return 1.0 - collisions / (n * (n - 1)), len(groups)


def _uniqueness(rows: list[frozenset[str]], components: list[str]) -> tuple[float, int]:
    """Distinct columns / M. Returns (uniqueness, n_ambiguity_groups).

    Two components sharing a column are indistinguishable to any spectrum method:
    they are hit by exactly the same runs, so no evidence separates them.
    """
    if not components:
        return 0.0, 0
    columns: dict[str, frozenset[int]] = {
        c: frozenset(i for i, row in enumerate(rows) if c in row) for c in components
    }
    groups = len(set(columns.values()))
    return groups / len(components), groups


def compute_diagnosability(
    runs: list[Run],
    task_id: str = "",
    signature: Callable = step_name_signature,
) -> Diagnosability:
    """Score one task's run set. No pass/fail labels are read."""
    rows, components = build_activity_matrix(runs, signature)
    raw_density, density = _density(rows, len(components))
    diversity, n_distinct = _diversity(rows)
    uniqueness, n_groups = _uniqueness(rows, components)
    return Diagnosability(
        task_id=task_id or (runs[0].task_id if runs else ""),
        ddu=density * diversity * uniqueness,
        density=density,
        raw_density=raw_density,
        diversity=diversity,
        uniqueness=uniqueness,
        n_runs=len(runs),
        n_components=len(components),
        n_distinct_rows=n_distinct,
        n_ambiguity_groups=n_groups,
    )


def compute_all_diagnosability(
    task_groups: dict[str, list[Run]],
    signature: Callable = step_name_signature,
    min_runs: int = 2,
) -> list[Diagnosability]:
    """Score every task with at least `min_runs` runs, worst DDU first.

    Tasks below `min_runs` are dropped rather than scored, because diversity is
    undefined on a single row and a zero there would read as "undiagnosable"
    when the truth is "not measured".
    """
    scored = [
        compute_diagnosability(runs, task_id=task_id, signature=signature)
        for task_id, runs in task_groups.items()
        if len(runs) >= min_runs
    ]
    return sorted(scored, key=lambda d: d.ddu)
