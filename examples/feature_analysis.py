#!/usr/bin/env python3
"""End-to-end behavioral feature analysis on SWE-rebench data.

This script reproduces the headline findings from the moirai project:
5 behavioral features that predict AI agent success/failure on software
engineering tasks, validated with within-task natural experiments and
split-half validation.

Usage:
    # With SWE-rebench data (12,854 runs):
    python examples/feature_analysis.py /Volumes/mnemosyne/moirai/swe_rebench_v2

    # Or any moirai-formatted run directory:
    python examples/feature_analysis.py path/to/runs

    # With JSON export:
    python examples/feature_analysis.py path/to/runs --output results.json

Prerequisites:
    pip install -e .

    # To get the SWE-rebench data:
    python scripts/convert_swe_rebench.py <path-to-raw-trajectories> examples/swe_rebench

Data source:
    Nebius SWE-rebench (huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories)
    12,854 same-agent runs (OpenHands v0.54.0 + Qwen3-Coder-480B)
    1,096 mixed-outcome tasks (same agent sometimes passes, sometimes fails)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path if running from examples/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moirai.load import load_runs
from moirai.analyze.features import FEATURES, rank_features, within_task_experiment
from moirai.analyze.content import select_task_groups, compute_test_centroid


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="Path to run directory")
    parser.add_argument("--min-runs", type=int, default=10, help="Min runs per task (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split-half")
    parser.add_argument("--output", type=Path, default=None, help="JSON output path")
    parser.add_argument("--verbose", action="store_true", help="Show per-feature details")
    args = parser.parse_args()

    # ── Step 1: Load runs ──────────────────────────────────────────────
    print(f"Loading runs from {args.path}...")
    runs, warnings = load_runs(args.path)
    if warnings:
        for w in warnings[:5]:
            print(f"  warning: {w}")
    print(f"  {len(runs):,} runs loaded")

    n_total_tasks = len({r.task_id for r in runs})
    print(f"  {n_total_tasks:,} total tasks")

    # ── Step 2: Identify mixed-outcome tasks ───────────────────────────
    task_groups, skip_reasons = select_task_groups(runs, min_runs=args.min_runs)
    print(f"  {len(task_groups):,} mixed-outcome tasks with >= {args.min_runs} runs")

    if not task_groups:
        print("No qualifying tasks. Try lowering --min-runs.")
        sys.exit(1)

    # ── Step 3: Compute and rank features ──────────────────────────────
    print(f"\nComputing {len(FEATURES)} behavioral features...")
    results = rank_features(runs, min_runs=args.min_runs, seed=args.seed)

    # ── Step 4: Display results ────────────────────────────────────────
    print(f"\n{'Feature':<36s} {'Delta':>8s}  {'p-value':>8s}  {'q-value':>8s}  {'Split-half':>10s}")
    print(f"{'─' * 36} {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 10}")

    for r in results:
        sign = "+" if r.delta_pp >= 0 else ""
        delta = f"{sign}{r.delta_pp:.1f}pp"
        p = "<0.001" if r.p_value is not None and r.p_value < 0.001 else (f"{r.p_value:.3f}" if r.p_value else "—")
        q = "<0.001" if r.q_value is not None and r.q_value < 0.001 else (f"{r.q_value:.3f}" if r.q_value else "—")
        sh = "✓" if r.split_half else "—"
        print(f"  {r.name:<36s} {delta:>8s}  {p:>8s}  {q:>8s}  {sh:>10s}")

    # ── Step 5: Interpretation ─────────────────────────────────────────
    survivors = [r for r in results if r.split_half]
    print(f"\n{len(survivors)} features survive split-half validation:")
    for r in survivors:
        direction = "higher" if r.direction == "positive" else "lower"
        print(f"  {r.name}: {direction} values predict success ({r.delta_pp:+.1f}pp)")

    if args.verbose:
        print("\nPer-feature details:")
        for r in results:
            print(f"\n  {r.name}:")
            print(f"    pass mean: {r.pass_mean:.4f}")
            print(f"    fail mean: {r.fail_mean:.4f}")
            print(f"    tasks contributing: {r.n_tasks}")
            print(f"    runs with data: {r.n_runs}")
            print(f"    description: {r.description}")

    # ── Step 6: Export JSON ────────────────────────────────────────────
    if args.output:
        envelope = {
            "dataset": str(args.path),
            "n_runs": len(runs),
            "n_tasks_mixed": len(task_groups),
            "n_tasks_total": n_total_tasks,
            "min_runs": args.min_runs,
            "seed": args.seed,
            "features": [
                {
                    "name": r.name,
                    "description": r.description,
                    "direction": r.direction,
                    "delta_pp": round(r.delta_pp, 2),
                    "p_value": r.p_value,
                    "q_value": r.q_value,
                    "split_half": r.split_half,
                    "pass_mean": round(r.pass_mean, 4),
                    "fail_mean": round(r.fail_mean, 4),
                    "n_tasks": r.n_tasks,
                    "n_runs": r.n_runs,
                }
                for r in results
            ],
        }
        with open(args.output, "w") as f:
            json.dump(envelope, f, indent=2)
        print(f"\nJSON written to {args.output}")

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\nMethodology:")
    print(f"  Within-task median-split natural experiment (controls for task difficulty)")
    print(f"  Sign test across tasks (exact binomial, two-sided)")
    print(f"  Benjamini-Hochberg correction for multiple testing")
    print(f"  Split-half validation: random 50/50 task split, significant on both halves")
    print(f"  Agent: OpenHands v0.54.0 + Qwen3-Coder-480B (same agent, stochastic reruns)")


if __name__ == "__main__":
    main()
