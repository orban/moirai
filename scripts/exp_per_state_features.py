#!/usr/bin/env python3
"""Per-state behavioral features, which is what we never tried.

Every scorer in this project reduces a run to branch choices: at some state the
run picked action A over B, and A carries a suspiciousness score. Ochiai, Fisher,
shrunk rates, Liblit's Increase(P) -- all of them are per-branch. The ceiling is
0.58 held-out across unseen tasks, and it does not move when the alphabet
changes: step names, step+target, Markov-2, hashed path buckets, 3,596 generated
predicates and a 20,000-token TF-IDF model all land between 0.554 and 0.578.

Cho et al., "Automata from Agent Traces: Failure and Next-Step Prediction"
(arXiv:2608.23670, AIWILD @ ICML 2026) score the same data differently. Their FSM
state is the last activity performed -- their Theorem 3 gives |Q| = |A| + 1, one
state per activity plus the initial state, which is exactly the `win-1`
abstraction measured here at 0.559. What they extract is not which branch a run
took but *per-state behavioral features*, which lift MLP/GRU/Transformer
baselines on 20 of 21 pairs and reach rank-AUROC 0.66 on SWE-agent at the 25%
checkpoint against 0.5 for flagging everything.

So the alphabet was never the variable. This tests the featurization.

For each state q (= last activity) a run contributes:

    visits            how often the run was in q
    share             visits / total steps
    entered_early     fraction of visits in the first quarter of the run
    error_rate        fraction of steps in q whose status was error
    out_entropy       entropy of the distribution over next activities from q
    self_loop         fraction of transitions out of q that return to q
    mean_gap          mean steps between consecutive visits to q

Plus run-level terms the FSM view makes natural: number of distinct states
visited, transition entropy over the whole run, and whether the run terminated
in an accepting activity.

Same protocol as every other cross-task experiment here: tasks split in half,
vocabulary and model fit on train tasks only, AUROC computed within each
held-out task and averaged, shuffled arm permuting outcomes within task.

    python scripts/exp_per_state_features.py /path/to/swe_rebench_v2/ --out out.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import random
import statistics
import sys

import numpy as np

SEED = 42
MIN_RUNS = 4
L2 = 1.0
EPOCHS = 200
LR = 1.0

PER_STATE = ("visits", "share", "early", "error_rate", "out_entropy",
             "self_loop", "mean_gap")


def activities(steps: list[dict]) -> list[str]:
    """The FSM alphabet: one symbol per step, bounded and small.

    Cho et al. extract the tool-call function name where one exists and fall back
    to a role:content_type label. The converted traces here already carry that
    distinction in `name`, so this is the same alphabet by a different route.
    """
    return [s.get("name", "?") for s in steps]


def state_features(steps: list[dict], alphabet: list[str]) -> np.ndarray:
    acts = activities(steps)
    n = len(acts)
    if n == 0:
        return np.zeros(len(alphabet) * len(PER_STATE) + 3, dtype=np.float32)

    idx = {a: i for i, a in enumerate(alphabet)}
    visits = collections.Counter(acts)
    early = collections.Counter(a for a in acts[: max(1, n // 4)])
    errors = collections.Counter(
        s.get("name", "?") for s in steps if s.get("status") == "error")
    nxt: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for a, b in zip(acts, acts[1:]):
        nxt[a][b] += 1
    positions: dict[str, list[int]] = collections.defaultdict(list)
    for i, a in enumerate(acts):
        positions[a].append(i)

    feats = np.zeros((len(alphabet), len(PER_STATE)), dtype=np.float32)
    for a, i in idx.items():
        v = visits.get(a, 0)
        if not v:
            continue
        out = nxt.get(a) or collections.Counter()
        total_out = sum(out.values())
        ent = 0.0
        if total_out:
            for c in out.values():
                p = c / total_out
                ent -= p * math.log(p + 1e-12)
        gaps = [b - a_ for a_, b in zip(positions[a], positions[a][1:])]
        feats[i] = (
            math.log1p(v),
            v / n,
            early.get(a, 0) / max(1, v),
            errors.get(a, 0) / v,
            ent,
            (out.get(a, 0) / total_out) if total_out else 0.0,
            (statistics.mean(gaps) / n) if gaps else 0.0,
        )

    run_ent = 0.0
    trans = collections.Counter(zip(acts, acts[1:]))
    tot = sum(trans.values())
    for c in trans.values():
        p = c / tot if tot else 0.0
        if p:
            run_ent -= p * math.log(p)
    tail = np.array([len(visits) / max(1, len(alphabet)), run_ent,
                     math.log1p(n)], dtype=np.float32)
    return np.concatenate([feats.ravel(), tail])


def fit_logreg(X, y):
    w = np.zeros(X.shape[1], dtype=np.float32)
    b = np.float32(0.0)
    yy = y.astype(np.float32)
    for _ in range(EPOCHS):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        err = p - yy
        w -= LR * ((X.T @ err) / len(yy) + L2 / len(yy) * w)
        b -= LR * err.mean()
    return w, b


def auroc(pos, neg):
    if not len(pos) or not len(neg):
        return None
    wins = 0.0
    for a in pos:
        for bb in neg:
            wins += 1.0 if a > bb else (0.5 if a == bb else 0.0)
    return wins / (len(pos) * len(neg))


def clip(steps, frac):
    """First `frac` of a run. Prefix prediction is the deployable
    question and the one Cho et al. report at the 25% checkpoint;
    the full trace leaks its own ending."""
    if frac >= 1.0:
        return steps
    return steps[: max(1, int(len(steps) * frac))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prefix-frac", type=float, default=1.0,
                    help="score only the first fraction of each trace")
    ap.add_argument("--protocol", choices=("task", "trace"), default="task",
                    help="task: hold out whole tasks, score within each. "
                         "trace: 80/20 over runs with tasks spanning both "
                         "sides, scored pooled -- the setup Cho et al. use.")
    ap.add_argument("--all-tasks", action="store_true",
                    help="keep every task, not just mixed-outcome ones. Cho et "
                         "al. split over all traces, which includes always-pass "
                         "and always-fail tasks that are far easier to separate.")
    ap.add_argument("--length-only", action="store_true",
                    help="ablate to the length feature alone, their 0.659 baseline")
    args = ap.parse_args()

    by_task = collections.defaultdict(list)
    for fp in sorted(glob.glob(os.path.join(args.data_path, "*.json"))):
        d = json.load(open(fp))
        by_task[d["task_id"]].append(d)
    tasks = {t: rs for t, rs in by_task.items()
             if len(rs) >= MIN_RUNS
             and (args.all_tasks
                  or len({bool(r["result"].get("success")) for r in rs}) == 2)}
    if args.all_tasks:
        mixed = sum(1 for rs in tasks.values()
                    if len({bool(r["result"].get("success")) for r in rs}) == 2)
        print(f"all-tasks mode: {len(tasks)} tasks, {mixed} of them mixed-outcome",
              file=sys.stderr)
    if args.limit:
        tasks = dict(list(tasks.items())[: args.limit])

    alphabet = sorted({a for rs in tasks.values() for r in rs
                       for a in activities(clip(r["steps"], args.prefix_frac))})
    print(f"{len(tasks)} mixed-outcome tasks, "
          f"{sum(len(v) for v in tasks.values())} runs, "
          f"|A| = {len(alphabet)} activities -> |Q| = {len(alphabet) + 1} states",
          file=sys.stderr)

    prepped = {t: [(state_features(clip(r["steps"], args.prefix_frac), alphabet),
                    bool(r["result"].get("success"))) for r in rs]
               for t, rs in tasks.items()}

    out = {}
    for arm in ("real", "shuffled"):
        rng = random.Random(SEED)
        data = {}
        for t, rs in prepped.items():
            ys = [ok for _, ok in rs]
            if arm == "shuffled":
                rng.shuffle(ys)
            data[t] = [(x, ok) for (x, _), ok in zip(rs, ys)]

        ids = sorted(data)
        rng.shuffle(ids)
        if args.protocol == "trace":
            # Every run pooled, then split 80/20 regardless of task. A task's
            # runs land on both sides, so the model can memorize task
            # difficulty -- which is why the shuffled arm matters here.
            allruns = [(x, ok) for t in ids for x, ok in data[t]]
            rng.shuffle(allruns)
            cut = int(0.8 * len(allruns))
            tr, te = allruns[:cut], allruns[cut:]
            train_ids, test_ids = ids, []
            Xtr = np.vstack([x for x, _ in tr])
            ytr = np.array([ok for _, ok in tr])
        else:
            half = len(ids) // 2
            train_ids, test_ids = ids[:half], ids[half:]
            Xtr = np.vstack([x for t in train_ids for x, _ in data[t]])
            ytr = np.array([ok for t in train_ids for _, ok in data[t]])
        if args.length_only:
            Xtr = Xtr[:, -1:]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        if args.length_only:
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        w, b = fit_logreg((Xtr - mu) / sd, ytr)

        aurocs = []
        pooled_pos, pooled_neg = [], []
        if args.protocol == "trace":
            Xte = np.vstack([x for x, _ in te])
            if args.length_only:
                Xte = Xte[:, -1:]
            sc = ((Xte - mu) / sd) @ w + b
            pooled_pos = [v for v, (_, ok) in zip(sc, te) if ok]
            pooled_neg = [v for v, (_, ok) in zip(sc, te) if not ok]
            aurocs = [auroc(pooled_pos, pooled_neg) or float("nan")]
        for t in test_ids:
            X = (np.vstack([x for x, _ in data[t]]) - mu) / sd
            oks = [ok for _, ok in data[t]]
            scores = X @ w + b
            pos = [s for s, ok in zip(scores, oks) if ok]
            neg = [s for s, ok in zip(scores, oks) if not ok]
            pooled_pos.extend(pos)
            pooled_neg.extend(neg)
            a = auroc(pos, neg)
            if a is not None:
                aurocs.append(a)

        # Pooled across held-out tasks, i.e. rank every run against every other
        # run regardless of task. This is the easier metric: it can exploit the
        # fact that some tasks fail more often, which within-task ranking cannot.
        # Reported so the number is comparable to papers that rank globally.
        pooled = auroc(pooled_pos[::3], pooled_neg[::3])

        names = [f"{a}:{f}" for a in alphabet for f in PER_STATE] + \
                ["n_states", "transition_entropy", "log_len"]
        top = sorted(range(len(w)), key=lambda j: -abs(w[j]))[:15]
        out[arm] = {
            "n_features": int(Xtr.shape[1]),
            "train_tasks": len(train_ids),
            "test_tasks": len(aurocs),
            "auroc": statistics.mean(aurocs) if aurocs else float("nan"),
            "auroc_pooled": pooled,
            "top_features": [{"feature": names[j], "weight": round(float(w[j]), 4)}
                             for j in top],
        }
        m = out[arm]
        print(f"[{arm}] within-task AUROC {m['auroc']:.4f} | "
              f"pooled AUROC {m['auroc_pooled']:.4f} | "
              f"{m['test_tasks']} tasks", file=sys.stderr)

    out["margin"] = out["real"]["auroc"] - out["shuffled"]["auroc"]
    print(f"\nmargin over shuffled: {out['margin']:+.4f}", file=sys.stderr)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
