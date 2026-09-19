"""Reproduce the structure-routing result.

One command, one result. No flags, no tuning knobs.

Usage:
    python scripts/run_structure_routing.py /Volumes/mnemosyne/moirai/swe_rebench_v2/

Expected output:
    Conditional (threshold=0.20): 59.3% accuracy
    Beats features-only (57.3%) and divergence-only (56.1%)
"""
from __future__ import annotations

import sys
import time

# ── Frozen parameters ─────────────────────────────────────────────
SEED = 42
MIN_RUNS = 4
K = 3
N_SAMPLES = 500
TOP_K = 5
N_RESAMPLES = 20
STRUCTURE_THRESHOLD = 0.20
# ──────────────────────────────────────────────────────────────────


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_structure_routing.py <data_path>")
        sys.exit(1)

    data_path = sys.argv[1]

    from moirai.load import load_runs
    from moirai.analyze.content import select_task_groups
    from moirai.analyze.rerank import rerank_experiment
    from moirai.analyze.structure import compute_all_structure_scores

    # Step 1: Load
    print(f"Loading runs from {data_path}...")
    runs, warnings = load_runs(data_path)
    print(f"  {len(runs)} runs loaded")

    task_groups, skip = select_task_groups(runs, min_runs=MIN_RUNS)
    print(f"  {len(task_groups)} mixed-outcome tasks")

    # Step 2: Structure scores
    print(f"\nComputing structure scores (top_k={TOP_K}, resamples={N_RESAMPLES})...")
    t0 = time.time()
    scores = compute_all_structure_scores(
        task_groups, min_runs=MIN_RUNS, top_k=TOP_K,
        n_resamples=N_RESAMPLES, seed=SEED,
    )
    score_map = {s.task_id: s.composite for s in scores}
    print(f"  {len(scores)} tasks scored in {time.time() - t0:.0f}s")

    # Step 3: Reranking
    print(f"\nRunning reranking experiment (K={K}, samples={N_SAMPLES})...")
    t0 = time.time()
    results = rerank_experiment(
        task_groups, k=K, n_samples=N_SAMPLES, seed=SEED, min_runs=MIN_RUNS,
    )
    print(f"  {results.n_tasks} tasks evaluated in {time.time() - t0:.0f}s")

    # Step 4: Conditional routing
    per_task = results.per_task
    tasks_with_both = [tid for tid in per_task if tid in score_map]
    n = len(tasks_with_both)

    conditional_acc = 0.0
    n_high = 0
    for tid in tasks_with_both:
        if score_map[tid] >= STRUCTURE_THRESHOLD:
            conditional_acc += per_task[tid]["divergence"]
            n_high += 1
        else:
            conditional_acc += per_task[tid]["features"]
    conditional_acc /= n

    feat_global = sum(per_task[t]["features"] for t in tasks_with_both) / n
    div_global = sum(per_task[t]["divergence"] for t in tasks_with_both) / n
    rand_global = sum(per_task[t]["random"] for t in tasks_with_both) / n
    oracle = sum(
        max(per_task[t]["divergence"], per_task[t]["features"])
        for t in tasks_with_both
    ) / n

    # Step 5: Print result
    print(f"\n{'=' * 60}")
    print(f"  Structure-Conditional Routing Result")
    print(f"  {n} tasks, K={K}, threshold={STRUCTURE_THRESHOLD}")
    print(f"  {n_high} high-structure tasks ({100*n_high/n:.0f}%)")
    print(f"{'=' * 60}\n")

    print(f"  {'Strategy':<30s} {'Accuracy':>8s}  {'Lift':>8s}")
    print(f"  {'─' * 30} {'─' * 8}  {'─' * 8}")
    for name, acc in [
        ("Random", rand_global),
        ("Divergence (global)", div_global),
        ("Features (global)", feat_global),
        ("Conditional (struct≥0.20)", conditional_acc),
        ("Oracle (best per task)", oracle),
    ]:
        lift = acc - rand_global
        marker = " ◀" if name.startswith("Conditional") else ""
        print(f"  {name:<30s} {acc:>7.1%}  {lift:>+7.1%}{marker}")

    print()

    # Step 6: Assertion — fail if result drifts
    assert abs(conditional_acc - 0.593) < 0.01, (
        f"Result drifted: expected ~59.3%, got {conditional_acc:.1%}. "
        f"Check data path, seed, or parameters."
    )
    print("  ✓ Result stable (within 1pp of 59.3%)")


if __name__ == "__main__":
    main()
