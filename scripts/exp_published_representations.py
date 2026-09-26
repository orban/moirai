#!/usr/bin/env python3
"""Six published trajectory representations, one corpus, one protocol, one control.

Every representation tried here so far was invented on the spot: step names,
step+target, prefix trees, Markov windows, hashed path buckets, templated
predicates. They all land between 0.554 and 0.578 held-out AUROC across unseen
tasks, which says more about the scoring than the encodings. The obvious missing
comparison is against representations somebody published and validated.

Four are specified precisely enough to reimplement, and none of them need
anything our logs lack -- no LLM hidden states (which rules out OAT,
arXiv:2607.12747, whose step vector is a mean-pooled layer activation), and no
resumable rollouts (which rules out Math-Shepherd, arXiv:2312.08935).

  bpe      Byte-pair encoding over action sequences, merging frequently
           co-occurring adjacent actions. Cho Seonglae et al. call this emergent
           vocabulary induction (arXiv:2606.16988) and report the V-measure
           plateauing at K=192 actions, so that is the vocabulary size used here.

  outcome  A 13-symbol alphabet encoding action *and* its environment response --
           syntax error, import error, re-patch, test failure, runtime error
           (arXiv:2604.02547). Our traces carry this in `status`, and every
           experiment so far spent it as a scalar error-rate feature instead of
           folding it into the alphabet.

  canon    Jaccard similarity between a run's step set and the task's canonical
           set, defined as steps appearing in more than half of that task's
           *successful* runs (arXiv:2602.19008, d=0.48, roughly 0.63 AUROC
           equivalent). Computed leave-one-run-out so a run never contributes to
           the reference it is scored against. Note this is not a pure predictor:
           it reads the outcomes of a task's other runs. Reported separately for
           that reason.

  tfidf2   Bigram TF-IDF, 30k features, min_df=2, sublinear TF, L2 logistic
           regression with class-balanced weights -- the exact configuration
           that reaches task-disjoint AUROC 0.83 on tau2-bench false-success
           detection (arXiv:2606.09863). Their label differs from ours; the
           representation does not.

Protocol is the one used throughout: tasks split in half, everything fit on the
train half, AUROC computed within each held-out task and averaged, and a shuffled
arm permuting outcomes within task. A representation only counts if it beats its
own shuffled arm.

    python scripts/exp_published_representations.py /path/to/swe_rebench_v2/ --out out.json
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
BPE_VOCAB = 192          # arXiv:2606.16988, where their V-measure plateaus
TFIDF_FEATURES = 30000   # arXiv:2606.09863
TFIDF_MIN_DF = 2
L2 = 1.0
EPOCHS = 150
LR = 1.0


# ── representations ──────────────────────────────────────────────────────────

def acts_plain(steps):
    return [s.get("name", "?") for s in steps]


def acts_outcome(steps):
    """Action crossed with its environment response, not just the action."""
    out = []
    for s in steps:
        st = s.get("status")
        out.append(f'{s.get("name", "?")}!{"e" if st == "error" else "o"}')
    return out


def learn_bpe(corpus: list[list[str]], target_vocab: int):
    """Merge the most frequent adjacent pair until the vocabulary hits target.

    Straight BPE with actions as the base tokens. Merges are learned on train
    tasks only and then applied verbatim to held-out runs, so the vocabulary
    never sees the evaluation set.
    """
    seqs = [list(s) for s in corpus]
    vocab = {t for s in seqs for t in s}
    merges: list[tuple[str, str]] = []
    while len(vocab) < target_vocab:
        pairs = collections.Counter()
        for s in seqs:
            for a, b in zip(s, s[1:]):
                pairs[(a, b)] += 1
        if not pairs:
            break
        (a, b), n = pairs.most_common(1)[0]
        if n < 5:
            break
        merged = f"{a}+{b}"
        merges.append((a, b))
        vocab.add(merged)
        for i, s in enumerate(seqs):
            out, j = [], 0
            while j < len(s):
                if j + 1 < len(s) and s[j] == a and s[j + 1] == b:
                    out.append(merged)
                    j += 2
                else:
                    out.append(s[j])
                    j += 1
            seqs[i] = out
    return merges


def apply_bpe(seq: list[str], merges) -> list[str]:
    s = list(seq)
    for a, b in merges:
        out, j = [], 0
        while j < len(s):
            if j + 1 < len(s) and s[j] == a and s[j + 1] == b:
                out.append(f"{a}+{b}")
                j += 2
            else:
                out.append(s[j])
                j += 1
        s = out
    return s


def bigrams(tokens: list[str]) -> collections.Counter:
    c = collections.Counter(tokens)
    c.update(f"{a}|{b}" for a, b in zip(tokens, tokens[1:]))
    return c


def text_tokens(steps) -> list[str]:
    """Trajectory text, for the TF-IDF arm."""
    import re
    parts = []
    for s in steps:
        parts.append(s.get("name", "?"))
        if s.get("status") == "error":
            parts.append("STATUS_ERROR")
        attrs = s.get("attrs") or {}
        for k in ("file_path", "command"):
            if attrs.get(k):
                parts.append(str(attrs[k])[:400])
        o = s.get("output") or {}
        for k in ("action", "result", "reasoning"):
            if o.get(k):
                parts.append(str(o[k])[:400])
    txt = re.sub(r"/workspace/[^/\s]+/?", " ", " ".join(parts))
    return [t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,30}", txt)]


# ── shared machinery ─────────────────────────────────────────────────────────

def fit_logreg(X, y, class_balanced=False):
    w = np.zeros(X.shape[1], dtype=np.float32)
    b = np.float32(0.0)
    yy = y.astype(np.float32)
    sw = np.ones_like(yy)
    if class_balanced:
        pos, neg = yy.sum(), len(yy) - yy.sum()
        if pos and neg:
            sw = np.where(yy > 0, len(yy) / (2 * pos), len(yy) / (2 * neg))
    sw = sw / sw.mean()
    for _ in range(EPOCHS):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        err = (p - yy) * sw
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


def counts_to_matrix(counts, vocab, idf=None):
    X = np.zeros((len(counts), len(vocab)), dtype=np.float32)
    for r, c in enumerate(counts):
        for t, n in c.items():
            j = vocab.get(t)
            if j is not None:
                X[r, j] = (1.0 + math.log(n)) * (idf[j] if idf is not None else 1.0)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-9)


def build_vocab(counts, max_features, min_df):
    df = collections.Counter()
    for c in counts:
        df.update(c.keys())
    n = len(counts)
    keep = [(t, d) for t, d in df.items() if min_df <= d <= 0.95 * n]
    keep.sort(key=lambda kv: -kv[1])
    return {t: i for i, (t, _) in enumerate(keep[:max_features])}


def acts_target(steps):
    """Name plus target: the richer alphabet, for testing whether the canonical
    method's weakness here is cardinality rather than the method."""
    out = []
    for st in steps:
        a = st.get("attrs") or {}
        t = a.get("file_path") or a.get("command") or ""
        import re as _re
        t = _re.sub(r"^/workspace/[^/]+/?", "", str(t))[:60]
        out.append(f'{st.get("name", "?")}|{t}' if t else st.get("name", "?"))
    return out


def canonical_scores(task_runs, actfn=None):
    """Leave-one-run-out Jaccard against the task's canonical step set.

    Not a pure predictor: the reference is built from the outcomes of the other
    runs on the same task. That is the published design (arXiv:2602.19008),
    where the contrast is the object of study rather than a deployable signal.
    """
    actfn = actfn or acts_plain
    sets = [set(actfn(steps)) for steps, _ in task_runs]
    oks = [ok for _, ok in task_runs]
    # Equalize the reference size across labels. A successful run has one fewer
    # sibling success available than a failed run does, and that difference alone
    # shifts the ">half" threshold in a direction correlated with the label. The
    # shuffled arm sat at 0.436 with the naive version, which is what surfaced it.
    n_succ = sum(1 for ok in oks if ok)
    ref_size = max(1, n_succ - 1)
    out = []
    for i in range(len(sets)):
        others = [sets[j] for j in range(len(sets)) if j != i and oks[j]][:ref_size]
        if not others:
            out.append(0.0)
            continue
        freq = collections.Counter(t for s in others for t in s)
        canon = {t for t, c in freq.items() if c > len(others) / 2}
        if not canon:
            out.append(0.0)
            continue
        inter = len(sets[i] & canon)
        union = len(sets[i] | canon)
        out.append(inter / union if union else 0.0)
    return out


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
    print(f"{len(tasks)} tasks, {sum(len(v) for v in tasks.values())} runs",
          file=sys.stderr)

    raw = {t: [(r["steps"], bool(r["result"].get("success"))) for r in rs]
           for t, rs in tasks.items()}

    out = {}
    for arm in ("real", "shuffled"):
        rng = random.Random(SEED)
        data = {}
        for t, rs in raw.items():
            ys = [ok for _, ok in rs]
            if arm == "shuffled":
                rng.shuffle(ys)
            data[t] = [(steps, ok) for (steps, _), ok in zip(rs, ys)]

        ids = sorted(data)
        rng.shuffle(ids)
        half = len(ids) // 2
        train_ids, test_ids = ids[:half], ids[half:]

        # canon needs no training; score it directly per task
        canon_aurocs = []
        for t in test_ids:
            sc = canonical_scores(data[t])
            oks = [ok for _, ok in data[t]]
            a = auroc([s for s, ok in zip(sc, oks) if ok],
                      [s for s, ok in zip(sc, oks) if not ok])
            if a is not None:
                canon_aurocs.append(a)
        out.setdefault("canon", {})[arm] = statistics.mean(canon_aurocs)

        rich = []
        for t in test_ids:
            sc = canonical_scores(data[t], acts_target)
            oks = [ok for _, ok in data[t]]
            a = auroc([s_ for s_, ok in zip(sc, oks) if ok],
                      [s_ for s_, ok in zip(sc, oks) if not ok])
            if a is not None:
                rich.append(a)
        out.setdefault("canon_rich", {})[arm] = statistics.mean(rich)

        merges = learn_bpe(
            [acts_plain(s) for t in train_ids for s, _ in data[t]], BPE_VOCAB)
        print(f"[{arm}] learned {len(merges)} BPE merges", file=sys.stderr)

        featurizers = {
            "bpe": lambda s: collections.Counter(apply_bpe(acts_plain(s), merges)),
            "outcome": lambda s: bigrams(acts_outcome(s)),
            "tfidf2": lambda s: bigrams(text_tokens(s)),
        }
        for name, fx in featurizers.items():
            cnt = {t: [fx(s) for s, _ in data[t]] for t in ids}
            tr = [c for t in train_ids for c in cnt[t]]
            ytr = np.array([ok for t in train_ids for _, ok in data[t]])
            mx = TFIDF_FEATURES if name == "tfidf2" else 5000
            mdf = TFIDF_MIN_DF if name == "tfidf2" else 5
            vocab = build_vocab(tr, mx, mdf)
            df = np.zeros(len(vocab), dtype=np.float32)
            for c in tr:
                for t_ in c:
                    j = vocab.get(t_)
                    if j is not None:
                        df[j] += 1
            idf = np.log((1 + len(tr)) / (1 + df)) + 1.0
            w, b = fit_logreg(counts_to_matrix(tr, vocab, idf), ytr,
                              class_balanced=(name == "tfidf2"))
            aur = []
            for t in test_ids:
                sc = counts_to_matrix(cnt[t], vocab, idf) @ w + b
                oks = [ok for _, ok in data[t]]
                a = auroc([s for s, ok in zip(sc, oks) if ok],
                          [s for s, ok in zip(sc, oks) if not ok])
                if a is not None:
                    aur.append(a)
            out.setdefault(name, {})[arm] = statistics.mean(aur)
            out[name]["vocab"] = len(vocab)

    print()
    for name in ("bpe", "outcome", "canon", "canon_rich", "tfidf2"):
        r, s = out[name]["real"], out[name]["shuffled"]
        v = out[name].get("vocab", "-")
        print(f"{name:9s} vocab={str(v):>6s}  real {r:.4f}  shuffled {s:.4f}  "
              f"margin {r - s:+.4f}", file=sys.stderr)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
