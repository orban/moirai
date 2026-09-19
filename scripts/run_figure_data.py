"""Dump the per-task data the blog figures plot.

run_structure_routing.py prints the headline number. This writes the observations
underneath it, so a figure can show 1,096 tasks rather than five summary bars.

The frozen parameters are copied from run_structure_routing.py by value and asserted
against the number it prints. If a figure and the reproduction script ever disagree,
that assertion is what catches it.

Usage:
    python scripts/run_figure_data.py /Volumes/mnemosyne/moirai/swe_rebench_v2/ \
        --out scripts/blog_output/figure_data.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time

# ── Frozen parameters: must match scripts/run_structure_routing.py ────────────
SEED = 42
MIN_RUNS = 4
K = 3
N_SAMPLES = 500
TOP_K = 5
N_RESAMPLES = 20
STRUCTURE_THRESHOLD = 0.20
# ─────────────────────────────────────────────────────────────────────────────

# What run_structure_routing.py prints on SWE-rebench v2. A figure built from data
# that no longer reproduces this is a figure that lies.
EXPECTED_CONDITIONAL = 0.592
EXPECTED_N_TASKS = 1096


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", help="Directory of run JSON files")
    parser.add_argument("--out", required=True, help="Where to write the JSON")
    args = parser.parse_args()

    from moirai.load import load_runs
    from moirai.analyze.content import select_task_groups
    from moirai.analyze.rerank import rerank_experiment
    from moirai.analyze.structure import compute_all_structure_scores

    print(f"Loading runs from {args.data_path}...")
    runs, _ = load_runs(args.data_path)
    task_groups, _ = select_task_groups(runs, min_runs=MIN_RUNS)
    print(f"  {len(runs)} runs, {len(task_groups)} mixed-outcome tasks")

    t0 = time.time()
    scores = compute_all_structure_scores(
        task_groups, min_runs=MIN_RUNS, top_k=TOP_K,
        n_resamples=N_RESAMPLES, seed=SEED,
    )
    print(f"  structure scored in {time.time() - t0:.0f}s")

    t0 = time.time()
    results = rerank_experiment(
        task_groups, k=K, n_samples=N_SAMPLES, seed=SEED, min_runs=MIN_RUNS,
    )
    print(f"  reranking done in {time.time() - t0:.0f}s")

    score_map = {s.task_id: s for s in scores}
    per_task = results.per_task
    shared = sorted(tid for tid in per_task if tid in score_map)

    tasks = []
    for tid in shared:
        s = score_map[tid]
        acc = per_task[tid]
        tasks.append({
            "task_id": tid,
            "structure": s.composite,
            "branch_gap": s.branch_gap,
            "earlyness": s.earlyness,
            "stability": s.stability,
            "n_runs": s.n_runs,
            "n_divergence_points": s.n_divergence_points,
            "acc_random": acc["random"],
            "acc_divergence": acc["divergence"],
            "acc_features": acc["features"],
        })

    n = len(tasks)
    high = [t for t in tasks if t["structure"] >= STRUCTURE_THRESHOLD]
    conditional = sum(
        t["acc_divergence"] if t["structure"] >= STRUCTURE_THRESHOLD else t["acc_features"]
        for t in tasks
    ) / n

    summary = {
        "n_tasks": n,
        "n_runs": len(runs),
        "n_high_structure": len(high),
        "threshold": STRUCTURE_THRESHOLD,
        "acc_random": sum(t["acc_random"] for t in tasks) / n,
        "acc_divergence": sum(t["acc_divergence"] for t in tasks) / n,
        "acc_features": sum(t["acc_features"] for t in tasks) / n,
        "acc_conditional": conditional,
        # The §4 oracle: best of the two heuristics per task. Not the §2 oracle,
        # which is "pick a passing run if one exists" and is much higher.
        "acc_oracle_best_method": sum(
            max(t["acc_divergence"], t["acc_features"]) for t in tasks
        ) / n,
        "params": {
            "seed": SEED, "min_runs": MIN_RUNS, "k": K,
            "n_samples": N_SAMPLES, "top_k": TOP_K, "n_resamples": N_RESAMPLES,
        },
    }

    if abs(conditional - EXPECTED_CONDITIONAL) > 0.01:
        print(
            f"ERROR: conditional accuracy {conditional:.1%} is more than 1pp from the "
            f"{EXPECTED_CONDITIONAL:.1%} run_structure_routing.py reports. The figures "
            f"and the reproduction script would disagree; not writing.",
            file=sys.stderr,
        )
        return 1
    if n != EXPECTED_N_TASKS:
        print(
            f"ERROR: {n} tasks, expected {EXPECTED_N_TASKS}. The dataset or the "
            f"filter changed; the post's stated counts would be wrong.",
            file=sys.stderr,
        )
        return 1

    with open(args.out, "w") as f:
        json.dump({"summary": summary, "tasks": tasks}, f)

    print(f"\n  {n} tasks, {len(high)} high-structure ({100*len(high)/n:.0f}%)")
    print(f"  conditional {conditional:.1%}  (matches run_structure_routing.py)")
    print(f"  written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
