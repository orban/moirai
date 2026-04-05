#!/usr/bin/env python3
"""Experiment: Haiku semantic step classification on same-agent data.

Tests whether LLM-based semantic labels (outcome-blind) predict
success/failure beyond what structural step types capture.

Design:
- Sample 200 runs from mixed-outcome tasks (same agent: OpenHands+Qwen3)
- Have Haiku classify each step into semantic categories WITHOUT seeing outcome
- Test if semantic label counts predict outcomes within-task
- Compare to structural feature baselines (has_edit_test, step_count, etc.)
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import subprocess
from moirai.normalize import normalize_run
from moirai.compress import step_enriched_name

# Semantic categories — designed to be outcome-blind
CATEGORIES = [
    "ORIENT",           # Getting bearings: pwd, ls, version checks
    "EXPLORE_BROAD",    # Broad exploration: searching across the codebase
    "EXPLORE_TARGETED", # Targeted exploration: reading a specific relevant file
    "DIAGNOSE",         # Analyzing the problem: reasoning about root cause
    "REPRODUCE",        # Writing/running reproduction scripts
    "FIX_ATTEMPT",      # Editing code to fix the issue
    "TEST_VERIFY",      # Running tests to verify a fix
    "TEST_EXPLORE",     # Running tests to understand current behavior
    "BACKTRACK",        # Undoing previous changes, reverting
    "FLAIL",            # Repeating similar actions without progress
    "SETUP",            # Installing deps, configuring environment
    "OTHER",            # Doesn't fit above categories
]

CLASSIFICATION_PROMPT = """Classify each agent step into exactly one category. You are analyzing an AI coding agent's trajectory on a software task. You do NOT know whether this run succeeded or failed — classify based only on what the agent is doing.

Categories:
- ORIENT: Getting bearings (pwd, ls, checking versions, confirming working directory)
- EXPLORE_BROAD: Searching broadly across the codebase (find, grep, directory listings)
- EXPLORE_TARGETED: Reading a specific file that's likely relevant to the issue
- DIAGNOSE: Reasoning about the root cause, forming hypotheses about the bug
- REPRODUCE: Writing or running a script to reproduce the issue
- FIX_ATTEMPT: Editing source code to fix the issue
- TEST_VERIFY: Running the test suite or specific tests to check if a fix works
- TEST_EXPLORE: Running tests to understand current behavior (before fixing)
- BACKTRACK: Undoing changes, reverting edits, starting over on an approach
- FLAIL: Repeating similar actions without apparent new information or progress
- SETUP: Installing dependencies, configuring the environment
- OTHER: Doesn't fit the above

For each step, output ONLY the category name, one per line. No explanations.

Steps to classify:
"""


def format_step_for_classification(step: dict, idx: int) -> str:
    """Format a step for the classification prompt."""
    out = step.get("output", {})
    reasoning = out.get("reasoning", "")[:200]
    action = out.get("action", "")[:200]
    result = str(out.get("result", ""))[:150]
    name = f"{step['type']}:{step['name']}"

    parts = [f"[Step {idx}] {name}"]
    if reasoning:
        parts.append(f"  Thinking: {reasoning}")
    if action:
        parts.append(f"  Action: {action}")
    if result:
        parts.append(f"  Result: {result}")
    return "\n".join(parts)


def _classify_one_batch(args: tuple[int, list[dict]]) -> tuple[int, list[str]]:
    """Classify one batch — designed for concurrent.futures."""
    batch_idx, batch = args
    step_texts = []
    for j, step in enumerate(batch):
        step_texts.append(format_step_for_classification(step, j))

    prompt = CLASSIFICATION_PROMPT + "\n\n".join(step_texts)

    try:
        result = subprocess.run(
            ["claude", "-p", "--model", "haiku", "--max-turns", "1"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=45,
        )
        text = result.stdout.strip()
        labels = []
        for line in text.split("\n"):
            line = line.strip().upper()
            for cat in CATEGORIES:
                if cat in line:
                    labels.append(cat)
                    break
            else:
                if line and not line.startswith(("#", "-", "[")):
                    labels.append("OTHER")

        while len(labels) < len(batch):
            labels.append("OTHER")
        return batch_idx, labels[:len(batch)]

    except Exception:
        return batch_idx, ["OTHER"] * len(batch)


def classify_steps_batch(
    steps: list[dict],
    batch_size: int = 50,
    max_workers: int = 10,
) -> list[str]:
    """Classify steps using Haiku via claude CLI, parallelized."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Split into batches
    batches = []
    for i in range(0, len(steps), batch_size):
        batches.append((i // batch_size, steps[i:i + batch_size]))

    # Run in parallel
    results: dict[int, list[str]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_classify_one_batch, b): b[0] for b in batches}
        for future in as_completed(futures):
            batch_idx, labels = future.result()
            results[batch_idx] = labels

    # Reassemble in order
    all_labels = []
    for i in range(len(batches)):
        all_labels.extend(results.get(i, ["OTHER"] * len(batches[i][1])))

    return all_labels


def main():
    data_dir = Path("/Volumes/mnemosyne/moirai/swe_rebench_full")

    print("Loading runs...")
    task_runs: dict[str, list] = defaultdict(list)
    for f in sorted(data_dir.iterdir()):
        if f.suffix != ".json":
            continue
        raw = json.loads(f.read_text())
        run, _ = normalize_run(raw)
        if run.steps and run.result.success is not None:
            task_runs[run.task_id].append((run, raw))

    # Select mixed-outcome tasks with enough runs
    balanced = [
        (tid, runs) for tid, runs in task_runs.items()
        if sum(1 for r, _ in runs if r.result.success) >= 3
        and sum(1 for r, _ in runs if not r.result.success) >= 3
    ]
    print(f"Balanced tasks: {len(balanced)}")

    # Sample 35 tasks, 3 pass + 3 fail per task = 210 runs
    random.seed(42)
    sample_tasks = random.sample(balanced, min(35, len(balanced)))

    sample_runs = []
    for tid, runs in sample_tasks:
        pass_runs = [(r, raw) for r, raw in runs if r.result.success][:3]
        fail_runs = [(r, raw) for r, raw in runs if not r.result.success][:3]
        sample_runs.extend([(tid, r, raw) for r, raw in pass_runs])
        sample_runs.extend([(tid, r, raw) for r, raw in fail_runs])

    print(f"Sample: {len(sample_runs)} runs from {len(sample_tasks)} tasks")
    total_steps = sum(len(r.steps) for _, r, _ in sample_runs)
    print(f"Total steps to classify: {total_steps}")
    estimated_cost = total_steps / 15 * 0.0002  # ~200 input tokens per batch, haiku pricing
    print(f"Estimated cost: ~${estimated_cost:.2f}")

    # Classify all steps
    results = []

    for i, (tid, run, raw) in enumerate(sample_runs):
        if i % 5 == 0:
            print(f"  Classifying run {i}/{len(sample_runs)}...", flush=True)

        # Prepare steps as dicts
        step_dicts = []
        for s in run.steps:
            step_dicts.append({
                "idx": s.idx,
                "type": s.type,
                "name": s.name,
                "output": {
                    "reasoning": s.output.get("reasoning", ""),
                    "action": s.output.get("action", ""),
                    "result": str(s.output.get("result", ""))[:200],
                },
            })

        labels = classify_steps_batch(step_dicts)
        structural = [step_enriched_name(s) or "unknown" for s in run.steps]

        results.append({
            "task_id": tid,
            "run_id": run.run_id,
            "success": run.result.success,
            "structural": structural,
            "semantic": labels,
            "n_steps": len(run.steps),
        })

    # Save raw results
    output_path = Path("/Volumes/mnemosyne/moirai/haiku_classification_results.json")
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {output_path}")

    # ============================================================
    # ANALYSIS
    # ============================================================
    print(f"\n{'=' * 80}")
    print("ANALYSIS")
    print(f"{'=' * 80}")

    # Aggregate semantic label distribution
    all_labels = []
    for r in results:
        all_labels.extend(r["semantic"])
    label_dist = Counter(all_labels)
    print(f"\nSemantic label distribution:")
    for label, count in label_dist.most_common():
        print(f"  {label:<20s} {count:>6d} ({count * 100 // len(all_labels)}%)")

    # Per-run semantic features
    def point_biserial(fv, bv):
        n = len(fv)
        if n < 10:
            return 0, 1
        g0 = [fv[i] for i in range(n) if not bv[i]]
        g1 = [fv[i] for i in range(n) if bv[i]]
        n0, n1 = len(g0), len(g1)
        if n0 < 3 or n1 < 3:
            return 0, 1
        m0, m1 = sum(g0) / n0, sum(g1) / n1
        om = sum(fv) / n
        ov = sum((x - om) ** 2 for x in fv) / n
        if ov == 0:
            return 0, 1
        rpb = (m1 - m0) * math.sqrt(n0 * n1 / (n * n)) / math.sqrt(ov)
        if abs(rpb) >= 1:
            return rpb, 0
        t = rpb * math.sqrt((n - 2) / (1 - rpb ** 2))
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
        return rpb, p

    # Build per-run feature vectors
    semantic_features = {}
    for cat in CATEGORIES:
        feat_name = f"sem_{cat.lower()}_frac"
        semantic_features[feat_name] = []

    task_ids = []
    successes = []
    for r in results:
        task_ids.append(r["task_id"])
        successes.append(r["success"])
        label_counts = Counter(r["semantic"])
        total = max(len(r["semantic"]), 1)
        for cat in CATEGORIES:
            feat_name = f"sem_{cat.lower()}_frac"
            semantic_features[feat_name].append(label_counts.get(cat, 0) / total)

    # Within-task centered analysis
    print(f"\n{'=' * 80}")
    print("SEMANTIC FEATURES — WITHIN-TASK CORRELATION")
    print(f"{'=' * 80}")

    task_groups: dict[str, list[int]] = defaultdict(list)
    for i, tid in enumerate(task_ids):
        task_groups[tid].append(i)

    print(f"\n  {'Feature':<30s} {'Pooled r':>10s} {'Within r':>10s} {'Within p':>12s} {'Sig':>4s}")
    print(f"  {'-' * 70}")

    for feat_name in sorted(semantic_features):
        vals = semantic_features[feat_name]
        pr, _ = point_biserial(vals, successes)

        cv, cs = [], []
        for tid, indices in task_groups.items():
            if len(indices) < 4:
                continue
            outcomes = [successes[i] for i in indices]
            if not any(outcomes) or all(outcomes):
                continue
            fv = [vals[i] for i in indices]
            tm = sum(fv) / len(fv)
            for v, o in zip(fv, outcomes):
                cv.append(v - tm)
                cs.append(o)

        wr, wp = point_biserial(cv, cs)
        sig = "***" if wp < 0.001 else "**" if wp < 0.01 else "*" if wp < 0.05 else "ns"
        print(f"  {feat_name:<30s} {pr:>+10.4f} {wr:>+10.4f} {wp:>12.2e} {sig:>4s}")

    # Compare: structural features on the same runs
    print(f"\n{'=' * 80}")
    print("STRUCTURAL FEATURES (same runs, for comparison)")
    print(f"{'=' * 80}")

    structural_features = {
        "step_count": [r["n_steps"] for r in results],
        "edit_frac": [
            sum(1 for s in r["structural"] if s.startswith(("edit", "write"))) / max(len(r["structural"]), 1)
            for r in results
        ],
        "explore_frac": [
            sum(1 for s in r["structural"] if s.startswith(("read", "search"))) / max(len(r["structural"]), 1)
            for r in results
        ],
    }

    print(f"\n  {'Feature':<30s} {'Pooled r':>10s} {'Within r':>10s} {'Within p':>12s} {'Sig':>4s}")
    print(f"  {'-' * 70}")

    for feat_name, vals in structural_features.items():
        pr, _ = point_biserial(vals, successes)
        cv, cs = [], []
        for tid, indices in task_groups.items():
            if len(indices) < 4:
                continue
            outcomes = [successes[i] for i in indices]
            if not any(outcomes) or all(outcomes):
                continue
            fv = [vals[i] for i in indices]
            tm = sum(fv) / len(fv)
            for v, o in zip(fv, outcomes):
                cv.append(v - tm)
                cs.append(o)
        wr, wp = point_biserial(cv, cs)
        sig = "***" if wp < 0.001 else "**" if wp < 0.01 else "*" if wp < 0.05 else "ns"
        print(f"  {feat_name:<30s} {pr:>+10.4f} {wr:>+10.4f} {wp:>12.2e} {sig:>4s}")


if __name__ == "__main__":
    main()
