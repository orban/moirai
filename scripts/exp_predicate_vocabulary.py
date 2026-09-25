#!/usr/bin/env python3
"""Does a generated predicate vocabulary find signal that step labels cannot?

Every rung of the matching ablation holds one thing fixed without saying so: the
alphabet. M0 through M5 all describe a step by its name, optionally with the file
it touched, and the best of them reaches 0.554 held-out AUROC. Hashing file paths
into 26 arbitrary buckets does better than any of those hand-built alphabets,
which is a strong hint that the alphabet was never carrying the information and
nobody had looked at that axis.

Spectrum-based fault localization hit this in 2005. Liblit, Naik, Zheng, Aiken &
Jordan, "Scalable Statistical Bug Isolation" (PLDI 2005, doi:10.1145/1065010.1065014)
stopped choosing a vocabulary. Templates emit every predicate the program admits
-- 857,384 of them on Rhythmbox -- and the outcome labels prune the set: keep P
only when the 95% confidence interval on Increase(P) sits strictly above zero,
which left 537, then eliminate redundancy, which left 15.

Their statistic is what matters here:

    Failure(P) = F(P) / (S(P) + F(P))
    Context(P) = F(P observed) / (S(P observed) + F(P observed))
    Increase(P) = Failure(P) - Context(P)

Context is the counterfactual baseline. A run that has already thrown five
tracebacks by step 30 is doomed whatever it does next, so every predicate true at
that point scores high on raw failure rate -- Liblit calls these innocent
bystanders. Subtracting the failure rate of merely *reaching* the site removes
them. The published pipeline's Fisher test had no such correction, and the essay
currently claims fixing that needs a POMDP. It needs arithmetic.

The port keeps the (site, condition) structure rather than flattening predicates
to run-level booleans. Flattening would make every predicate always-observed,
Context would collapse to the global failure rate, and the whole method would
reduce to ranking by raw failure rate. So a predicate is "run did action A" as
the site, refined by a condition on how it did it.

Protocol. Tasks are split in half. Predicates are selected on the train tasks
only and scored on held-out *tasks*, because selecting from tens of thousands of
predicates on ~11 runs within a task would manufacture signal from nothing. AUROC
is computed within each held-out task and averaged, which matches the matching
ablation and keeps task difficulty from confounding the pooled ranking.

The shuffled arm runs the identical pipeline with outcomes permuted within task.
It is the load-bearing control at this vocabulary size: if selection leaks, the
shuffled arm rises above 0.50 and every real number is void.

    python scripts/exp_predicate_vocabulary.py /path/to/swe_rebench_v2/ --out out.json
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

SEED = 42
MIN_RUNS = 4
MIN_SUPPORT = 20        # a predicate must be true in this many train runs
MAX_SUPPORT_FRAC = 0.95  # and false in at least this fraction, or it says nothing
COUNT_THRESHOLDS = (0, 1, 2, 5, 10)
Z = 1.96


# ── feature extraction ───────────────────────────────────────────────────────

def path_tokens(path: str) -> set[str]:
    """Tokens of a workspace-relative path. The repo prefix is stripped because
    it is constant within a task and would only pad the vocabulary."""
    path = re.sub(r"^/workspace/[^/]+/?", "", str(path))
    return {t.lower() for t in re.split(r"[/_.\-]+", path) if len(t) >= 3}


def extract(steps: list[dict]) -> dict:
    """Everything the templates below read, in one pass over the run."""
    names = [s.get("name", "?") for s in steps]
    reads, edits, cmds, exts, toks = [], [], [], set(), set()
    errors = 0
    per_action_targets: dict[str, set[str]] = collections.defaultdict(set)

    for s in steps:
        name = s.get("name", "?")
        if s.get("status") == "error":
            errors += 1
        attrs = s.get("attrs") or {}
        fp, cmd = attrs.get("file_path"), attrs.get("command")
        if fp:
            rel = re.sub(r"^/workspace/[^/]+/?", "", str(fp))
            toks |= path_tokens(fp)
            if "." in rel.rsplit("/", 1)[-1]:
                exts.add(rel.rsplit(".", 1)[-1].lower()[:8])
            per_action_targets[name].add(rel)
            (edits if name in ("edit", "write") else reads).append(rel)
        if cmd:
            # OpenHands prefixes almost every call with `cd /workspace/<repo> &&`,
            # so taking the first token of the whole string yields `cd` for the
            # entire corpus. Split the shell line and take the head of each
            # segment, then drop the navigation verbs that carry no intent.
            for seg in re.split(r"&&|\|\||[;|]", str(cmd)):
                seg = seg.strip()
                if not seg:
                    continue
                head = seg.split()[0].rsplit("/", 1)[-1][:20]
                if head and head not in ("cd", "source", "export", "."):
                    cmds.append(head)
            toks |= {t.lower() for t in re.findall(r"[A-Za-z_]{3,}", str(cmd)[:200])}

    seen_reads: set[str] = set()
    reread = any(r in seen_reads or seen_reads.add(r) for r in reads)
    edited_unread = any(e not in set(reads) for e in edits)

    return {
        "names": names,
        "counts": collections.Counter(names),
        "cmds": collections.Counter(cmds),
        "exts": exts,
        "toks": toks,
        "errors": errors,
        "n_steps": len(steps),
        "n_files": len(set(reads) | set(edits)),
        "reread": reread,
        "edited_unread": edited_unread,
        "per_action_targets": per_action_targets,
    }


def predicates(f: dict, vocab: dict) -> tuple[set[str], set[str]]:
    """Return (sites observed, predicates true) for one run.

    A site is a coarse gate the run either entered or did not. A predicate is a
    condition refining that site. Context(P) is measured over runs that entered
    the site, which is what keeps it a counterfactual baseline rather than the
    global failure rate.
    """
    sites: set[str] = {"any"}
    true: set[str] = set()

    for name, n in f["counts"].items():
        sites.add(f"act:{name}")
        true.add(f"any|did:{name}")
        for k in COUNT_THRESHOLDS:
            if n > k:
                true.add(f"act:{name}|n>{k}")

    # Adjacent-action pairs, gated on having performed the first action at all.
    for a, b in zip(f["names"], f["names"][1:]):
        true.add(f"act:{a}|then:{b}")

    # Ordered pairs: A appears before B anywhere. Liblit's dissertation proposes
    # exactly this family over traces.
    first_at: dict[str, int] = {}
    for i, n in enumerate(f["names"]):
        first_at.setdefault(n, i)
    for a, ia in first_at.items():
        for b, ib in first_at.items():
            if a != b and ia < ib:
                true.add(f"act:{a}|before:{b}")

    for head, n in f["cmds"].items():
        if head in vocab["cmds"]:
            sites.add("act:bash")
            true.add(f"act:bash|cmd:{head}")

    for ext in f["exts"]:
        if ext in vocab["exts"]:
            true.add(f"any|ext:{ext}")

    for tok in f["toks"] & vocab["toks"]:
        true.add(f"any|tok:{tok}")

    for k in (5, 10, 20, 30, 50):
        if f["n_steps"] > k:
            true.add(f"any|steps>{k}")
        if f["n_files"] > k // 5:
            true.add(f"any|files>{k // 5}")
    for k in (0, 1, 3, 5):
        if f["errors"] > k:
            true.add(f"any|errors>{k}")

    if f["reread"]:
        true.add("act:read|reread")
    if f["edited_unread"]:
        true.add("act:edit|edited_unread")
    if f["names"]:
        true.add(f"any|first:{f['names'][0]}")
        true.add(f"any|last:{f['names'][-1]}")

    return sites, {p for p in true if p.split("|", 1)[0] in sites}


# ── Liblit statistics ────────────────────────────────────────────────────────

def increase_ci(s_true, f_true, s_obs, f_obs) -> tuple[float, float]:
    """Increase(P) and the lower bound of its 95% interval.

    Both terms are Bernoulli proportions, so the difference gets the usual
    normal approximation. Liblit retains P only when this lower bound is
    strictly above zero; that test alone removed 99.9% of their predicates.
    """
    n_true, n_obs = s_true + f_true, s_obs + f_obs
    if n_true == 0 or n_obs == 0:
        return 0.0, -1.0
    failure = f_true / n_true
    context = f_obs / n_obs
    inc = failure - context
    se = math.sqrt(
        failure * (1 - failure) / n_true + context * (1 - context) / n_obs
    )
    return inc, inc - Z * se


def importance(inc: float, f_true: int, num_f: int) -> float:
    """Harmonic mean of Increase and a log-scaled failure count, per the paper:
    Importance(P) = 2 / (1/Increase(P) + 1/(log F(P) / log NumF))."""
    if inc <= 0 or f_true <= 0 or num_f <= 1:
        return 0.0
    sens = math.log(f_true) / math.log(num_f)
    if sens <= 0:
        return 0.0
    return 2.0 / (1.0 / inc + 1.0 / sens)


def select(train_tasks, max_keep=200):
    """Generate, prune by Increase, then eliminate redundancy -- stratified.

    Liblit's setting is one program, so every predicate varies over the run set
    by construction. Here there are 1,096 tasks, and a path token like
    `python__gql__` is true for every run of one task and false everywhere else.
    Pooled across the corpus such a predicate looks strongly predictive, because
    it is really reporting which task the run belongs to and some tasks fail more
    than others. It contributes nothing to the evaluation, which ranks runs
    *within* a held-out task: a predicate constant across that task's runs adds
    the same constant to every score and cancels.

    Pooling the two sources of variance and evaluating on only one of them is
    Simpson's paradox, and it put ~190 inert repository names into a 200-slot
    model. So counts come only from tasks where the predicate actually varies,
    which is the stratified estimator and the only variance the scorer can use.

    Step 3 of the elimination is what stops a single dominant predicate and its
    near-duplicates from filling the list: once P is chosen, every run where P
    holds is discarded, so the next pick is scored on what remains.
    """
    s_true, f_true = collections.Counter(), collections.Counter()
    s_obs, f_obs = collections.Counter(), collections.Counter()
    for runs in train_tasks:
        # Only predicates that split this task's runs carry usable information.
        counts = collections.Counter(p for _, preds, _ in runs for p in preds)
        varying = {p for p, c in counts.items() if 0 < c < len(runs)}
        if not varying:
            continue
        # Per-task, per-site observation counts. The baseline for P has to come
        # from the same tasks its failure rate does, or the stratification is
        # undone on the Context side.
        site_s, site_f = collections.Counter(), collections.Counter()
        for sites, _, ok in runs:
            for site in sites:
                (site_s if ok else site_f)[site] += 1
        for sites, preds, ok in runs:
            for p in preds & varying:
                (s_true if ok else f_true)[p] += 1
        for p in varying:
            site = p.split("|", 1)[0]
            s_obs[p] += site_s[site]
            f_obs[p] += site_f[site]

    train_runs = [r for runs in train_tasks for r in runs]
    num_f = sum(1 for _, _, ok in train_runs if not ok)
    n_train = len(train_runs)
    scored = {}
    for p in set(s_true) | set(f_true):
        st, ft = s_true[p], f_true[p]
        if ft + st < MIN_SUPPORT or (ft + st) > MAX_SUPPORT_FRAC * n_train:
            continue
        inc, lo = increase_ci(st, ft, s_obs[p], f_obs[p])
        if lo <= 0:
            continue
        imp = importance(inc, ft, num_f)
        if imp > 0:
            scored[p] = (imp, inc)

    generated = len(set(s_true) | set(f_true))
    survived = len(scored)

    # Redundancy elimination.
    remaining = [(set(preds), ok) for _, preds, ok in train_runs if not ok]
    kept = []
    pool = dict(scored)
    while pool and remaining and len(kept) < max_keep:
        best = max(pool, key=lambda p: pool[p][0])
        kept.append((best, pool[best][1]))
        del pool[best]
        remaining = [(pr, ok) for pr, ok in remaining if best not in pr]
    return kept, generated, survived


def score_run(preds: set[str], model: list[tuple[str, float]]) -> float:
    """Higher means more likely to succeed. Increase predicts failure, so the
    run's score is the negated sum of the Increase of every selected predicate
    it satisfies."""
    return -sum(inc for p, inc in model if p in preds)


def auroc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = 0.0
    for a in pos:
        for b in neg:
            wins += 1.0 if a > b else (0.5 if a == b else 0.0)
    return wins / (len(pos) * len(neg))


# ── experiment ───────────────────────────────────────────────────────────────

def build_vocab(all_feats) -> dict:
    """Frequency-filtered open vocabularies. A token in three runs is noise; a
    token in every run separates nothing."""
    n = len(all_feats)
    cmd_df, ext_df, tok_df = collections.Counter(), collections.Counter(), collections.Counter()
    for f in all_feats:
        cmd_df.update(set(f["cmds"]))
        ext_df.update(f["exts"])
        tok_df.update(f["toks"])
    keep = lambda c: {k for k, v in c.items() if MIN_SUPPORT <= v <= MAX_SUPPORT_FRAC * n}
    return {"cmds": keep(cmd_df), "exts": keep(ext_df), "toks": keep(tok_df)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--exclude", default="", help="regex; drop matching predicates")
    ap.add_argument("--only", default="", help="regex; keep only matching predicates")
    args = ap.parse_args()

    by_task = collections.defaultdict(list)
    for fp in sorted(glob.glob(os.path.join(args.data_path, "*.json"))):
        d = json.load(open(fp))
        by_task[d["task_id"]].append(d)
    tasks = {
        t: rs for t, rs in by_task.items()
        if len(rs) >= MIN_RUNS and len({bool(r["result"].get("success")) for r in rs}) == 2
    }
    if args.limit:
        tasks = dict(list(tasks.items())[: args.limit])
    print(f"{len(tasks)} mixed-outcome tasks, "
          f"{sum(len(v) for v in tasks.values())} runs", file=sys.stderr)

    feats = {t: [(extract(r["steps"]), bool(r["result"].get("success"))) for r in rs]
             for t, rs in tasks.items()}
    vocab = build_vocab([f for rs in feats.values() for f, _ in rs])
    print(f"vocabulary: {len(vocab['cmds'])} commands, {len(vocab['exts'])} "
          f"extensions, {len(vocab['toks'])} path tokens", file=sys.stderr)

    prepped = {t: [(*predicates(f, vocab), ok) for f, ok in rs] for t, rs in feats.items()}
    if args.exclude or args.only:
        ex = re.compile(args.exclude) if args.exclude else None
        on = re.compile(args.only) if args.only else None
        def filt(ps):
            return {p for p in ps
                    if (not ex or not ex.search(p)) and (not on or on.search(p))}
        prepped = {t: [(s_, filt(p_), ok) for s_, p_, ok in rs] for t, rs in prepped.items()}
        print(f"filtered: exclude={args.exclude!r} only={args.only!r}", file=sys.stderr)

    out = {}
    for arm in ("real", "shuffled"):
        rng = random.Random(SEED)
        data = {}
        for t, rs in prepped.items():
            outcomes = [ok for *_, ok in rs]
            if arm == "shuffled":
                rng.shuffle(outcomes)
            data[t] = [(s, p, ok) for (s, p, _), ok in zip(rs, outcomes)]

        task_ids = sorted(data)
        rng.shuffle(task_ids)
        half = len(task_ids) // 2
        train_ids, test_ids = task_ids[:half], task_ids[half:]

        train_tasks = [data[t] for t in train_ids]
        model, generated, survived = select(train_tasks)

        aurocs = []
        for t in test_ids:
            pos = [score_run(p, model) for _, p, ok in data[t] if ok]
            neg = [score_run(p, model) for _, p, ok in data[t] if not ok]
            a = auroc(pos, neg)
            if a is not None:
                aurocs.append(a)

        out[arm] = {
            "generated": generated,
            "survived_increase_ci": survived,
            "kept_after_elimination": len(model),
            "train_tasks": len(train_ids),
            "test_tasks": len(aurocs),
            "auroc": statistics.mean(aurocs) if aurocs else float("nan"),
            "top": [{"predicate": p, "increase": round(i, 4)} for p, i in model[:15]],
        }
        m = out[arm]
        print(f"\n[{arm}] {m['generated']} generated -> {m['survived_increase_ci']} "
              f"survive Increase CI -> {m['kept_after_elimination']} after elimination",
              file=sys.stderr)
        print(f"[{arm}] held-out AUROC {m['auroc']:.4f} over {m['test_tasks']} tasks",
              file=sys.stderr)

    out["margin"] = out["real"]["auroc"] - out["shuffled"]["auroc"]
    print(f"\nmargin over shuffled: {out['margin']:+.4f}", file=sys.stderr)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
