#!/usr/bin/env python3
"""The bag-of-words baseline that should have run first.

Everything else in this project describes a trajectory by its structure: step
names, step+target, prefix-tree nodes, Markov contexts, hashed path buckets, and
finally 3,596 generated predicates with Liblit's Increase(P). The best of those
reaches 0.577 held-out AUROC across unseen tasks.

None of them looked at the text.

Feng et al., "From Confident Closing to Silent Failure" (arXiv:2606.09863) report
TF-IDF detectors at task-disjoint AUROC 0.83 on tau2-bench against 0.65 for the
best LLM judge, at 3,300x lower latency. Their target is false success rather
than resolution, and the agent's own closing message is diagnostic there, so the
number will not transfer. That is not the point. The point is that a bag of words
over raw trajectory text is the cheapest possible representation and nobody here
ran it, so none of the structural results have a baseline to be measured against.

This runs it under the identical protocol as the predicate experiment: tasks
split in half, vocabulary and model fit on train tasks only, AUROC computed
within each held-out task and averaged, and a shuffled arm permuting outcomes
within task.

Implemented against numpy rather than scikit-learn because sklearn is not a
declared dependency of this repo and adding one for a baseline is not worth it.
TF-IDF and regularized logistic regression are forty lines.

    python scripts/exp_tfidf_baseline.py /path/to/swe_rebench_v2/ --out out.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import random
import re
import statistics
import sys

import numpy as np

SEED = 42
MIN_RUNS = 4
MIN_DF = 10          # a token must appear in this many train runs
MAX_DF_FRAC = 0.90   # and be absent from at least this fraction
MAX_FEATURES = 20000
MAX_CHARS_PER_STEP = 400
L2 = 1.0
EPOCHS = 30
LR = 0.5

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,30}")


def run_text(steps: list[dict]) -> str:
    """Flatten a run into text: step names plus what each step actually said.

    Truncated per step because a single traceback can be longer than the rest of
    the trajectory combined, and the repo prefix is stripped for the same reason
    it is stripped everywhere else -- it names the task, not the behaviour.
    """
    parts = []
    for s in steps:
        parts.append(s.get("name", "?"))
        if s.get("status") == "error":
            parts.append("STATUS_ERROR")
        attrs = s.get("attrs") or {}
        for key in ("file_path", "command"):
            if attrs.get(key):
                parts.append(str(attrs[key])[:MAX_CHARS_PER_STEP])
        out = s.get("output") or {}
        for key in ("action", "result", "reasoning"):
            if out.get(key):
                parts.append(str(out[key])[:MAX_CHARS_PER_STEP])
    text = " ".join(parts)
    return re.sub(r"/workspace/[^/\s]+/?", " ", text)


def tokenize(text: str) -> collections.Counter:
    return collections.Counter(t.lower() for t in TOKEN_RE.findall(text))


def fit_vocab(train_counts) -> dict[str, int]:
    df = collections.Counter()
    for c in train_counts:
        df.update(c.keys())
    n = len(train_counts)
    keep = [(t, d) for t, d in df.items() if MIN_DF <= d <= MAX_DF_FRAC * n]
    keep.sort(key=lambda kv: -kv[1])
    return {t: i for i, (t, _) in enumerate(keep[:MAX_FEATURES])}


def vectorize(counts, vocab, idf):
    """L2-normalized TF-IDF rows, log term frequency."""
    X = np.zeros((len(counts), len(vocab)), dtype=np.float32)
    for r, c in enumerate(counts):
        for t, n in c.items():
            j = vocab.get(t)
            if j is not None:
                X[r, j] = (1.0 + math.log(n)) * idf[j]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-9)


def fit_logreg(X, y):
    """Plain full-batch gradient descent on L2-regularized logistic loss.

    The feature matrix is small enough that nothing fancier earns its keep, and
    a hand-rolled solver is auditable in a way an imported one is not.
    """
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    by_task = collections.defaultdict(list)
    for fp in sorted(glob.glob(os.path.join(args.data_path, "*.json"))):
        d = json.load(open(fp))
        by_task[d["task_id"]].append(d)
    tasks = {t: rs for t, rs in by_task.items()
             if len(rs) >= MIN_RUNS
             and len({bool(r["result"].get("success")) for r in rs}) == 2}
    if args.limit:
        tasks = dict(list(tasks.items())[: args.limit])
    print(f"{len(tasks)} mixed-outcome tasks, "
          f"{sum(len(v) for v in tasks.values())} runs", file=sys.stderr)

    prepped = {t: [(tokenize(run_text(r["steps"])), bool(r["result"].get("success")))
                   for r in rs]
               for t, rs in tasks.items()}

    out = {}
    for arm in ("real", "shuffled"):
        rng = random.Random(SEED)
        data = {}
        for t, rs in prepped.items():
            ys = [ok for _, ok in rs]
            if arm == "shuffled":
                rng.shuffle(ys)
            data[t] = [(c, ok) for (c, _), ok in zip(rs, ys)]

        ids = sorted(data)
        rng.shuffle(ids)
        half = len(ids) // 2
        train_ids, test_ids = ids[:half], ids[half:]

        train = [r for t in train_ids for r in data[t]]
        vocab = fit_vocab([c for c, _ in train])
        df = np.zeros(len(vocab), dtype=np.float32)
        for c, _ in train:
            for t_ in c:
                j = vocab.get(t_)
                if j is not None:
                    df[j] += 1
        idf = np.log((1 + len(train)) / (1 + df)) + 1.0

        Xtr = vectorize([c for c, _ in train], vocab, idf)
        ytr = np.array([ok for _, ok in train])
        w, b = fit_logreg(Xtr, ytr)

        aurocs = []
        for t in test_ids:
            counts = [c for c, _ in data[t]]
            oks = [ok for _, ok in data[t]]
            scores = vectorize(counts, vocab, idf) @ w + b
            pos = [s for s, ok in zip(scores, oks) if ok]
            neg = [s for s, ok in zip(scores, oks) if not ok]
            a = auroc(pos, neg)
            if a is not None:
                aurocs.append(a)

        top = sorted(range(len(w)), key=lambda j: -abs(w[j]))[:20]
        inv = {j: t_ for t_, j in vocab.items()}
        out[arm] = {
            "vocab": len(vocab),
            "train_tasks": len(train_ids),
            "test_tasks": len(aurocs),
            "auroc": statistics.mean(aurocs) if aurocs else float("nan"),
            "top_features": [{"token": inv[j], "weight": round(float(w[j]), 4)}
                             for j in top],
        }
        m = out[arm]
        print(f"[{arm}] vocab {m['vocab']}, held-out AUROC {m['auroc']:.4f} "
              f"over {m['test_tasks']} tasks", file=sys.stderr)

    out["margin"] = out["real"]["auroc"] - out["shuffled"]["auroc"]
    print(f"\nmargin over shuffled: {out['margin']:+.4f}", file=sys.stderr)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
