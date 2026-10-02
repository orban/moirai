#!/usr/bin/env python3
"""Does the trace-divergence negative result survive a better matcher?

The published negative result is held-out AUROC 0.507 — a coin flip — from
Needleman-Wunsch alignment over a 13-symbol alphabet of step names, with branch
points gated by Fisher's exact + Benjamini-Hochberg at q <= 0.05.

Three things about that pipeline are suspect, and this measures each:

  1. The gate is a hypothesis test. At a median of 11 runs per task a 2-vs-9
     split cannot clear significance after correcting across every column, so
     812 of 1,096 tasks return no branch points at all and the scorer falls to
     a constant. Spectrum-based fault localization has used Ochiai as a
     *ranking* statistic for twenty years; it is always defined.

  2. Global alignment aliases states. NW can align step i of run A to step j of
     run B when their histories differ entirely, so a "divergence column" pools
     runs that are in different states. A prefix tree cannot do this: at a node
     every run shares a byte-identical history.

  3. The alphabet is 13 symbols. `read` matches `read` whether the run read the
     test file or setup.py. The converted traces already carry attrs.file_path,
     attrs.command and output.action — the aligner just ignores them.

Ablation ladder, each rung adding one change:

    M0  NW on step names     + Fisher/BH     (current; must reproduce ~0.507)
    M1  NW on step names     + Ochiai
    M2  prefix tree on names + Ochiai
    M3  prefix tree on content signatures + Ochiai with shrinkage

crossed with per-task run budget, because the whole question is whether the
signal is absent or merely undetected at 11 runs.

Pre-registered decision rule, fixed before looking at any output:

  * every rung <= 0.55 AUROC at every budget
        -> the negative result stands and is now robust to method
  * any rung rising with budget and clearing 0.60 at n >= 15
        -> the original conclusion was a power artifact, and the finding
           becomes "detectable, and here is the rollout budget it takes"

Every cell is run twice: once real, once with pass/fail labels shuffled within
task. The shuffled arm must sit at 0.50. Anything else means the estimator is
leaking and the real numbers are uninterpretable.

Usage:
    python scripts/exp_matching_ablation.py /Volumes/mnemosyne/moirai/swe_rebench_v2/ \
        --out scripts/blog_output/matching_ablation.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import math
import os
import random
import re
import sys

BUDGETS = [4, 6, 8, 11, 15, 20, 0]  # 0 = use every run the task has
MIN_TEST = 2                        # held-out runs needed on each side to score
SEED = 42


# ── signatures ───────────────────────────────────────────────────────────────

def sig_name(step: dict) -> str:
    """The current alphabet: 13 symbols across the whole dataset."""
    return step["name"]


def sig_content(step: dict) -> str:
    """Name plus what the step actually touched.

    attrs carries file_path for reads/edits and command for bash. Workspace
    prefixes are stripped because they embed the repo name, which is constant
    within a task and would only pad the symbol.
    """
    attrs = step.get("attrs") or {}
    target = attrs.get("file_path") or attrs.get("command") or ""
    if not target:
        action = (step.get("output") or {}).get("action") or ""
        target = hashlib.md5(str(action).encode()).hexdigest()[:8]
    target = re.sub(r"^/workspace/[^/]+/?", "", str(target))[:60]
    return f'{step["name"]}|{target}'


# ── statistics ───────────────────────────────────────────────────────────────

def ochiai(n_pass: int, n_fail: int, total_pass: int) -> float:
    """Pass-oriented Ochiai. Always defined; no significance threshold.

    The SBFL original scores suspiciousness toward failure. Orientation is
    flipped here because the downstream use is ranking runs by how good their
    choices look, not localizing a fault.
    """
    denom = math.sqrt(total_pass * (n_pass + n_fail))
    return n_pass / denom if denom > 0 else 0.0


def shrunk_rate(n_pass: int, n_total: int, prior: float, strength: float = 2.0) -> float:
    """Empirical-Bayes shrinkage toward the task's own pass rate.

    RTMC's prior-based value smoothing, in one line. A branch taken by two runs
    should not read as 100% just because both happened to pass.
    """
    return (n_pass + strength * prior) / (n_total + strength)


def auroc(scores_pos: list[float], scores_neg: list[float]) -> float | None:
    """Mann-Whitney U. Ties contribute 0.5, which is what makes a constant
    scorer land on exactly 0.500 rather than something noisy."""
    if not scores_pos or not scores_neg:
        return None
    wins = 0.0
    for a in scores_pos:
        for b in scores_neg:
            wins += 1.0 if a > b else (0.5 if a == b else 0.0)
    return wins / (len(scores_pos) * len(scores_neg))


# ── matchers ─────────────────────────────────────────────────────────────────

def node_key(sigs: list[str], d: int, context: int | None) -> tuple:
    """The state a run is in just before its step at depth d.

    `context=None` is the full prefix, which is the strictest equivalence there
    is: two runs share a state only when their entire histories match. It
    forbids reconvergence, so once two runs differ they are never compared
    again. `context=w` keeps only the last w steps, which is a Markov-order-w
    state and lets runs that wandered apart come back together.

    Which one is right is measurable rather than a matter of taste. Over 1,382
    tasks at 8 runs each, DDU on the name alphabet reads 0.008 for the full
    prefix and 0.523 for w=2 -- the full prefix fragments the run set into
    ambiguity groups, while a two-step context sits above the band Perez et al.
    measured for real-fault suites.
    """
    return tuple(sigs[:d]) if context is None else tuple(sigs[max(0, d - context):d])


def fit_tree(train, sigfn, prior: float, shrink: bool, context: int | None = None):
    """Branch model over train runs. Returns {state: {sig: score}} for branch nodes.

    A node is a decision point when the runs sharing that state went on to do
    two or more different things. No significance test, no minimum branch size:
    every branch node contributes, and thin branches are handled by shrinkage
    rather than by exclusion.
    """
    by_prefix: dict[tuple, dict[str, list[bool]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for steps, ok in train:
        sigs = [sigfn(s) for s in steps]
        for d in range(len(sigs)):
            by_prefix[node_key(sigs, d, context)][sigs[d]].append(ok)

    model: dict[tuple, dict[str, float]] = {}
    for prefix, branches in by_prefix.items():
        if len(branches) < 2:
            continue
        total_pass = sum(sum(v) for v in branches.values())
        if total_pass == 0:
            continue
        scored = {}
        for sig, outcomes in branches.items():
            n_pass, n = sum(outcomes), len(outcomes)
            scored[sig] = (
                shrunk_rate(n_pass, n, prior) if shrink
                else ochiai(n_pass, n - n_pass, total_pass)
            )
        model[prefix] = scored
    return model


def score_tree(steps, model, sigfn, context: int | None = None) -> float:
    """Walk the run through the fitted model, averaging the scores of the
    branches it took. A run whose states are all unseen is scored on nothing and
    falls back to the constant; under a bounded context that is much rarer,
    because a state can recur instead of being spent once."""
    sigs = [sigfn(s) for s in steps]
    vals = []
    for d in range(len(sigs)):
        node = model.get(node_key(sigs, d, context))
        if node and sigs[d] in node:
            vals.append(node[sigs[d]])
    return sum(vals) / len(vals) if vals else 0.5


def fit_columns(train, sigfn, prior: float, use_ochiai: bool):
    """Column model standing in for the NW pipeline.

    Runs are compared position by position, which is what NW degenerates to on
    sequences of similar length over a 13-symbol alphabet. This deliberately
    keeps the aliasing defect so that M1 isolates the change of statistic: same
    (mis)matching, ranking instead of a significance gate.
    """
    cols: dict[int, dict[str, list[bool]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for steps, ok in train:
        for i, s in enumerate(steps):
            cols[i][sigfn(s)].append(ok)

    model: dict[int, dict[str, float]] = {}
    for i, branches in cols.items():
        if len(branches) < 2:
            continue
        total_pass = sum(sum(v) for v in branches.values())
        if total_pass == 0:
            continue
        scored = {}
        for sig, outcomes in branches.items():
            n_pass, n = sum(outcomes), len(outcomes)
            if use_ochiai:
                scored[sig] = ochiai(n_pass, n - n_pass, total_pass)
            else:
                # M0's gate: a branch must have >=2 runs and clear an
                # uncorrected 0.05 two-tailed proportion difference. Generous
                # to M0 — the real pipeline also applies BH across columns,
                # which only removes more.
                if n < 2:
                    continue
                rate = n_pass / n
                se = math.sqrt(max(prior * (1 - prior), 1e-9) / n)
                if abs(rate - prior) < 1.96 * se:
                    continue
                scored[sig] = rate
        if scored:
            model[i] = scored
    return model


def score_columns(steps, model, sigfn) -> float:
    vals = []
    for i, s in enumerate(steps):
        node = model.get(i)
        if node:
            sig = sigfn(s)
            if sig in node:
                vals.append(node[sig])
    return sum(vals) / len(vals) if vals else 0.5


METHODS = {
    "M0_nw_names_fisher":   dict(tree=False, sig=sig_name,    ochiai=False, shrink=False, context=None),
    "M1_nw_names_ochiai":   dict(tree=False, sig=sig_name,    ochiai=True,  shrink=False, context=None),
    "M2_tree_names_ochiai": dict(tree=True,  sig=sig_name,    ochiai=True,  shrink=False, context=None),
    "M3_tree_content_shrunk": dict(tree=True, sig=sig_content, ochiai=True, shrink=True,  context=None),
    # M4/M5 change only the state equivalence: last two steps instead of the
    # whole history. Same statistic, same absent gate, same scoring walk as
    # M2/M3, so any difference is the abstraction and nothing else.
    "M4_markov2_names_ochiai":   dict(tree=True, sig=sig_name,    ochiai=True, shrink=False, context=2),
    "M5_markov2_content_shrunk": dict(tree=True, sig=sig_content, ochiai=True, shrink=True,  context=2),
}


def evaluate(task_runs, method: dict, budget: int, rng: random.Random, shuffle: bool):
    """Held-out AUROC for one task. Returns (auroc, had_model) or None."""
    runs = list(task_runs)
    if budget and len(runs) > budget:
        runs = rng.sample(runs, budget)

    outcomes = [bool(r["result"].get("success")) for r in runs]
    if len(set(outcomes)) < 2:
        return None
    if shuffle:
        rng.shuffle(outcomes)

    paired = list(zip([r["steps"] for r in runs], outcomes))
    rng.shuffle(paired)
    half = len(paired) // 2
    train, test = paired[:half], paired[half:]
    if len(train) < 2:
        return None
    pos = [s for s, ok in test if ok]
    neg = [s for s, ok in test if not ok]
    if len(pos) < 1 or len(neg) < 1:
        return None

    prior = sum(ok for _, ok in train) / len(train)
    sigfn = method["sig"]
    context = method.get("context")
    if method["tree"]:
        model = fit_tree(train, sigfn, prior, method["shrink"], context)
        def scorer(s):
            return score_tree(s, model, sigfn, context)
    else:
        model = fit_columns(train, sigfn, prior, method["ochiai"])
        def scorer(s):
            return score_columns(s, model, sigfn)

    a = auroc([scorer(s) for s in pos], [scorer(s) for s in neg])
    return None if a is None else (a, bool(model))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap tasks, for a smoke run")
    args = ap.parse_args()

    by_task: dict[str, list[dict]] = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(args.data_path, "*.json"))):
        d = json.load(open(f))
        by_task[d["task_id"]].append(d)

    tasks = {
        t: rs for t, rs in by_task.items()
        if len(rs) >= 4 and len({bool(r["result"].get("success")) for r in rs}) == 2
    }
    if args.limit:
        tasks = dict(list(tasks.items())[: args.limit])
    print(f"{len(by_task)} tasks loaded, {len(tasks)} mixed-outcome with >=4 runs\n")

    results = {}
    for name, method in METHODS.items():
        results[name] = {}
        for budget in BUDGETS:
            for shuffle in (False, True):
                rng = random.Random(SEED)
                aurocs, with_model = [], 0
                for t, rs in tasks.items():
                    if budget and len(rs) < budget:
                        continue
                    got = evaluate(rs, method, budget, rng, shuffle)
                    if got is None:
                        continue
                    a, had = got
                    aurocs.append(a)
                    with_model += int(had)
                if not aurocs:
                    continue
                key = f"{budget or 'all'}{'_shuffled' if shuffle else ''}"
                results[name][key] = {
                    "auroc": sum(aurocs) / len(aurocs),
                    "n_tasks": len(aurocs),
                    "detection_rate": with_model / len(aurocs),
                }
                tag = "shuffled" if shuffle else "real    "
                r = results[name][key]
                print(f"  {name:24s} n={budget or 'all':>3}  {tag}  "
                      f"AUROC {r['auroc']:.3f}  detect {r['detection_rate']:5.1%}  "
                      f"({r['n_tasks']} tasks)")
        print()

    with open(args.out, "w") as f:
        json.dump({"results": results, "budgets": BUDGETS, "seed": SEED}, f, indent=1)
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
