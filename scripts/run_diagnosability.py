#!/usr/bin/env python3
"""Measure DDU on the full SWE-rebench OpenHands corpus.

The trace-divergence result says branch-point analysis finds nothing on this data.
That leaves a question the experiment itself cannot answer: was there no signal, or
could this run set never have expressed one? DDU separates those, and reads no
pass/fail labels to do it.

Reads trajectories.parquet directly rather than through `datasets`, streaming by
row group -- the file is 2.1 GB and the decoded trajectories are an order of
magnitude larger, so nothing here holds more than one batch at a time. Only the
classified step list survives each row; the raw message content is dropped
immediately.

Step classification is imported from convert_swe_rebench rather than reimplemented,
so the component alphabet is byte-identical to the one the published analysis used.

    python scripts/run_diagnosability.py /path/to/trajectories.parquet --out ddu.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pyarrow.parquet as pq

from convert_swe_rebench import parse_openhands_trajectory  # noqa: E402

from moirai.analyze.diagnosability import (  # noqa: E402
    compute_all_diagnosability,
    step_name_signature,
    step_target_signature,
)
from moirai.schema import Result, Run, Step  # noqa: E402

# Matches the published analysis: a task counts only when it has enough runs to
# compare and both outcomes actually occur.
MIN_RUNS = 4


def _to_run(run_id: str, task_id: str, raw_steps: list[dict], resolved) -> Run:
    """Build a Run carrying only what the signatures read."""
    steps = [
        Step(
            idx=s["idx"],
            type=s["type"],
            name=s["name"],
            attrs={
                k: v
                for k, v in (s.get("attrs") or {}).items()
                if k in ("file_path", "command")
            },
        )
        for s in raw_steps
    ]
    success = True if resolved in (1, True) else False if resolved in (0, False) else None
    return Run(run_id=run_id, task_id=task_id, steps=steps, result=Result(success=success))


def load_corpus(parquet_path: Path, batch_size: int = 64) -> dict[str, list[Run]]:
    pf = pq.ParquetFile(parquet_path)
    total = pf.metadata.num_rows
    by_task: dict[str, list[Run]] = collections.defaultdict(list)
    seen = 0
    for batch in pf.iter_batches(
        batch_size=batch_size,
        columns=["trajectory_id", "instance_id", "resolved", "trajectory"],
    ):
        for row in batch.to_pylist():
            seen += 1
            traj = row.get("trajectory") or []
            raw_steps = parse_openhands_trajectory(traj)
            if not raw_steps:
                continue
            by_task[row["instance_id"]].append(
                _to_run(row["trajectory_id"], row["instance_id"], raw_steps, row.get("resolved"))
            )
        if seen % 5000 < batch_size:
            print(f"  {seen}/{total} rows, {len(by_task)} tasks", file=sys.stderr)
    print(f"  {seen}/{total} rows, {len(by_task)} tasks", file=sys.stderr)
    return by_task


def select_mixed_outcome(by_task: dict[str, list[Run]], min_runs: int) -> dict[str, list[Run]]:
    return {
        t: rs
        for t, rs in by_task.items()
        if len(rs) >= min_runs
        and len({r.result.success for r in rs if r.result.success is not None}) == 2
    }


def summarize(scores) -> dict:
    vals = sorted(d.ddu for d in scores)
    n = len(vals)

    def pct(p):
        return vals[min(n - 1, int(p * n))]

    return {
        "n_tasks": n,
        "median_ddu": vals[n // 2],
        "mean_ddu": sum(vals) / n,
        "min_ddu": vals[0],
        "max_ddu": vals[-1],
        "p25": pct(0.25),
        "p75": pct(0.75),
        "p95": pct(0.95),
        "frac_below_0.10": sum(1 for v in vals if v < 0.10) / n,
        "mean_density": sum(d.density for d in scores) / n,
        "mean_raw_density": sum(d.raw_density for d in scores) / n,
        "mean_diversity": sum(d.diversity for d in scores) / n,
        "mean_uniqueness": sum(d.uniqueness for d in scores) / n,
        "mean_runs": sum(d.n_runs for d in scores) / n,
        "mean_components": sum(d.n_components for d in scores) / n,
        "mean_distinct_rows": sum(d.n_distinct_rows for d in scores) / n,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-runs", type=int, default=MIN_RUNS)
    args = ap.parse_args()

    print("Loading corpus...", file=sys.stderr)
    by_task = load_corpus(Path(args.parquet))
    tasks = select_mixed_outcome(by_task, args.min_runs)
    print(
        f"\n{len(by_task)} tasks total, "
        f"{len(tasks)} mixed-outcome with >={args.min_runs} runs "
        f"({sum(len(v) for v in tasks.values())} runs)\n",
        file=sys.stderr,
    )
    if not tasks:
        print("No qualifying tasks.", file=sys.stderr)
        return 1

    out = {
        "n_tasks_all": len(by_task),
        "n_tasks_mixed_outcome": len(tasks),
        "n_runs_scored": sum(len(v) for v in tasks.values()),
        "min_runs": args.min_runs,
        "granularities": {},
    }
    for label, sig in (("name", step_name_signature), ("target", step_target_signature)):
        scores = compute_all_diagnosability(tasks, signature=sig, min_runs=args.min_runs)
        s = summarize(scores)
        out["granularities"][label] = s
        print(f"[{label}] median DDU {s['median_ddu']:.4f}  mean {s['mean_ddu']:.4f}  "
              f"density {s['mean_density']:.3f}  diversity {s['mean_diversity']:.3f}  "
              f"uniqueness {s['mean_uniqueness']:.3f}", file=sys.stderr)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWritten to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
