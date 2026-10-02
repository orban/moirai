#!/usr/bin/env python3
"""Does the mixed-outcome filter explain the gap to published numbers?

Every experiment in this project scores tasks where the same agent both passes
and fails. That filter was applied during conversion, before any of this work
started, and it was never questioned. It selects the maximally ambiguous subset
of the corpus on purpose: if a task always passes or always fails, there is no
within-task contrast to find a divergence point in.

Cho et al. (arXiv:2608.23670) do not filter. Their failure-prediction split is
80/20 over *traces* across nine labeled datasets, which includes tasks that
always pass and tasks that always fail. Those are close to trivially separable,
and on SWE-agent they report held-out AUROC 0.790 for structural features
against 0.659 for trace length alone. Running their protocol on our filtered
corpus gives 0.584 and 0.563.

So the gap is either the corpus or the filter, and the converted dataset on the
NAS cannot tell us which -- it contains only mixed-outcome tasks. The raw parquet
still has all 67,074 rows. This reads from there and scores both populations
under the identical protocol.

If the unfiltered population jumps toward 0.79, our ceiling is an artifact of
studying only the ambiguous tasks, and every negative result in this project is
scoped to that subset rather than to agent trajectories in general.

    python scripts/exp_mixed_outcome_filter.py /tmp/rebench/trajectories.parquet --out out.json
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from convert_swe_rebench import parse_openhands_trajectory  # noqa: E402
from exp_per_state_features import (  # noqa: E402
    activities, auroc, fit_logreg, state_features,
)

SEED = 42
MIN_RUNS = 4


def load(path: Path):
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    by_task: dict[str, list] = collections.defaultdict(list)
    seen = 0
    for batch in pf.iter_batches(batch_size=64, columns=[
            "trajectory_id", "instance_id", "resolved", "trajectory"]):
        for row in batch.to_pylist():
            seen += 1
            steps = parse_openhands_trajectory(row.get("trajectory") or [])
            if not steps:
                continue
            ok = row.get("resolved")
            if ok not in (0, 1, True, False):
                continue
            by_task[row["instance_id"]].append((steps, bool(ok)))
        if seen % 20000 < 64:
            print(f"  {seen}/{total}", file=sys.stderr)
    return by_task


def evaluate(tasks, alphabet, length_only: bool, arm: str, rng: random.Random):
    """80/20 over runs, tasks spanning both sides, AUROC pooled. Their setup."""
    runs = []
    for rs in tasks.values():
        ys = [ok for _, ok in rs]
        if arm == "shuffled":
            rng.shuffle(ys)
        for (steps, _), ok in zip(rs, ys):
            runs.append((state_features(steps, alphabet), ok))
    rng.shuffle(runs)
    cut = int(0.8 * len(runs))
    tr, te = runs[:cut], runs[cut:]

    Xtr = np.vstack([x for x, _ in tr])
    Xte = np.vstack([x for x, _ in te])
    if length_only:
        Xtr, Xte = Xtr[:, -1:], Xte[:, -1:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    w, b = fit_logreg((Xtr - mu) / sd, np.array([ok for _, ok in tr]))
    sc = ((Xte - mu) / sd) @ w + b
    pos = [v for v, (_, ok) in zip(sc, te) if ok]
    neg = [v for v, (_, ok) in zip(sc, te) if not ok]
    return auroc(pos[::2], neg[::2]), len(runs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("parquet")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    by_task = load(Path(args.parquet))
    enough = {t: rs for t, rs in by_task.items() if len(rs) >= MIN_RUNS}
    mixed = {t: rs for t, rs in enough.items()
             if len({ok for _, ok in rs}) == 2}
    always = len(enough) - len(mixed)
    print(f"\n{len(by_task)} tasks parsed, {len(enough)} with >={MIN_RUNS} runs, "
          f"{len(mixed)} mixed-outcome, {always} always-same-outcome\n",
          file=sys.stderr)

    alphabet = sorted({a for rs in enough.values() for steps, _ in rs
                       for a in activities(steps)})
    print(f"|A| = {len(alphabet)}\n", file=sys.stderr)

    out = {}
    for pop_name, pop in (("all_tasks", enough), ("mixed_only", mixed)):
        for length_only in (False, True):
            key = f"{pop_name}/{'length' if length_only else 'structural'}"
            row = {}
            for arm in ("real", "shuffled"):
                a, n = evaluate(pop, alphabet, length_only, arm,
                                random.Random(SEED))
                row[arm] = a
                row["n_runs"] = n
            row["margin"] = row["real"] - row["shuffled"]
            out[key] = row
            print(f"{key:28s} real {row['real']:.4f}  "
                  f"shuffled {row['shuffled']:.4f}  "
                  f"margin {row['margin']:+.4f}  ({row['n_runs']} runs)",
                  file=sys.stderr)

    out["_meta"] = {"tasks_total": len(by_task), "tasks_enough": len(enough),
                    "tasks_mixed": len(mixed), "alphabet": len(alphabet)}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
