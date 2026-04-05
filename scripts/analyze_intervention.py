#!/usr/bin/env python3
"""Analyze the test-after-edit intervention study results.

Usage:
    python scripts/analyze_intervention.py /path/to/eval-harness/results-intervention-*/

Computes:
    - Per-task pass rate: baseline vs test_after_edit
    - Within-task delta and sign test
    - Compliance rate: did the agent actually test after editing?
    - Effect size (Cohen's h) with bootstrap confidence interval
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import random


def load_trials(results_dir: Path) -> list[dict]:
    """Load all trial JSONs from results/trials/."""
    trials_dir = results_dir / "trials"
    if not trials_dir.exists():
        print(f"error: {trials_dir} not found", file=sys.stderr)
        sys.exit(1)

    trials = []
    for p in sorted(trials_dir.glob("*.json")):
        try:
            trials.append(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError) as e:
            print(f"warning: skipping {p.name}: {e}", file=sys.stderr)
    return trials


def load_logs(results_dir: Path) -> dict[str, list[str]]:
    """Load log files keyed by stem (task_id-condition-rN)."""
    log_dir = results_dir.parent / "logs"
    if not log_dir.exists():
        return {}
    logs = {}
    for p in log_dir.glob("*.log"):
        logs[p.stem] = p.read_text(encoding="utf-8", errors="replace").splitlines()
    return logs


def measure_compliance(log_lines: list[str]) -> dict[str, Any]:
    """Check whether agent tested after each edit.

    Scans [tool] lines in order. After every Edit or Write to a non-test file,
    checks whether the next tool call is a test (Bash containing pytest/test).
    Returns counts and per-edit details.
    """
    tool_lines = [l.strip() for l in log_lines if l.strip().startswith("[tool] ")]

    edits = 0
    edits_followed_by_test = 0
    edit_positions: list[dict] = []

    for i, line in enumerate(tool_lines):
        action = line[7:]  # strip "[tool] "

        is_edit = action.startswith("Edit ") or action.startswith("Write ")
        if not is_edit:
            continue

        # Skip edits to test files
        target = action.split(" ", 1)[1] if " " in action else ""
        if "test" in target.lower():
            continue

        edits += 1

        # Look at next tool action
        next_is_test = False
        if i + 1 < len(tool_lines):
            next_action = tool_lines[i + 1][7:]
            if next_action.startswith("Bash:"):
                cmd = next_action[5:].strip().lower()
                if any(kw in cmd for kw in ["pytest", "test", "npm test", "cargo test", "go test"]):
                    next_is_test = True

        if next_is_test:
            edits_followed_by_test += 1

        edit_positions.append({
            "position": i,
            "file": target[:80],
            "followed_by_test": next_is_test,
        })

    compliance_rate = edits_followed_by_test / edits if edits > 0 else None

    return {
        "total_source_edits": edits,
        "edits_followed_by_test": edits_followed_by_test,
        "compliance_rate": compliance_rate,
    }


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def bootstrap_ci(
    baseline_outcomes: list[bool],
    treatment_outcomes: list[bool],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap CI for difference in pass rates (treatment - baseline)."""
    rng = random.Random(seed)
    n_base = len(baseline_outcomes)
    n_treat = len(treatment_outcomes)

    if n_base == 0 or n_treat == 0:
        return {"delta": 0, "ci_lower": 0, "ci_upper": 0}

    observed_delta = sum(treatment_outcomes) / n_treat - sum(baseline_outcomes) / n_base

    deltas = []
    for _ in range(n_bootstrap):
        b_sample = rng.choices(baseline_outcomes, k=n_base)
        t_sample = rng.choices(treatment_outcomes, k=n_treat)
        d = sum(t_sample) / n_treat - sum(b_sample) / n_base
        deltas.append(d)

    deltas.sort()
    alpha = 1 - ci
    lo = deltas[int(alpha / 2 * n_bootstrap)]
    hi = deltas[int((1 - alpha / 2) * n_bootstrap)]

    return {"delta": observed_delta, "ci_lower": lo, "ci_upper": hi}


def sign_test_p(deltas: list[float]) -> float:
    """Two-sided sign test on per-task deltas.

    Returns p-value. Null hypothesis: median delta = 0.
    Uses exact binomial probability.
    """
    positive = sum(1 for d in deltas if d > 0)
    negative = sum(1 for d in deltas if d < 0)
    n = positive + negative  # ties excluded

    if n == 0:
        return 1.0

    # Exact binomial: P(X >= max(positive, negative)) under p=0.5
    k = max(positive, negative)
    # Sum binomial tail
    from math import comb
    p_tail = sum(comb(n, i) * 0.5**n for i in range(k, n + 1))
    return min(2 * p_tail, 1.0)  # two-sided


def main():
    parser = argparse.ArgumentParser(description="Analyze test-after-edit intervention results")
    parser.add_argument("results_dir", type=Path, help="Path to eval-harness results directory")
    parser.add_argument("--ci", type=float, default=0.95, help="Confidence interval level (default: 0.95)")
    args = parser.parse_args()

    results_dir = args.results_dir
    trials = load_trials(results_dir)
    logs = load_logs(results_dir)

    if not trials:
        print("No trials found.", file=sys.stderr)
        sys.exit(1)

    # Filter to just baseline and treatment conditions
    BASELINE = "none"
    TREATMENT = "test_after_edit"

    relevant = [t for t in trials if t.get("condition") in (BASELINE, TREATMENT)]
    if not relevant:
        print(f"No trials with conditions '{BASELINE}' or '{TREATMENT}' found.", file=sys.stderr)
        print(f"Conditions present: {sorted(set(t.get('condition', '?') for t in trials))}", file=sys.stderr)
        sys.exit(1)

    # Group by task_id
    by_task: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for t in relevant:
        task_id = t["task_id"]
        cond = t["condition"]
        by_task[task_id][cond].append(t)

    # --- Per-task pass rates ---
    print("=" * 80)
    print("PER-TASK PASS RATES")
    print("=" * 80)
    print(f"{'task_id':<50} {'baseline':>10} {'treatment':>10} {'delta':>8}")
    print("-" * 80)

    all_baseline_outcomes: list[bool] = []
    all_treatment_outcomes: list[bool] = []
    task_deltas: list[float] = []

    for task_id in sorted(by_task.keys()):
        base_runs = by_task[task_id].get(BASELINE, [])
        treat_runs = by_task[task_id].get(TREATMENT, [])

        # Filter out infra errors
        base_valid = [r for r in base_runs if not r.get("error", "").startswith("[")]
        treat_valid = [r for r in treat_runs if not r.get("error", "").startswith("[")]

        if not base_valid and not treat_valid:
            continue

        base_pass = sum(1 for r in base_valid if r.get("success")) / len(base_valid) if base_valid else 0
        treat_pass = sum(1 for r in treat_valid if r.get("success")) / len(treat_valid) if treat_valid else 0
        delta = treat_pass - base_pass

        task_deltas.append(delta)

        # Truncate long task IDs
        display_id = task_id[:48] + ".." if len(task_id) > 50 else task_id
        print(f"{display_id:<50} {base_pass:>9.0%} {treat_pass:>9.0%} {delta:>+7.0%}")

        all_baseline_outcomes.extend(r.get("success", False) for r in base_valid)
        all_treatment_outcomes.extend(r.get("success", False) for r in treat_valid)

    # --- Aggregate stats ---
    print("")
    print("=" * 80)
    print("AGGREGATE")
    print("=" * 80)

    n_base = len(all_baseline_outcomes)
    n_treat = len(all_treatment_outcomes)
    agg_base = sum(all_baseline_outcomes) / n_base if n_base else 0
    agg_treat = sum(all_treatment_outcomes) / n_treat if n_treat else 0

    print(f"Baseline pass rate:     {agg_base:.1%}  ({sum(all_baseline_outcomes)}/{n_base})")
    print(f"Treatment pass rate:    {agg_treat:.1%}  ({sum(all_treatment_outcomes)}/{n_treat})")
    print(f"Aggregate delta:        {agg_treat - agg_base:+.1%}")

    # Effect size
    h = cohens_h(agg_treat, agg_base)
    print(f"Cohen's h:              {h:+.3f}  ", end="")
    if abs(h) < 0.2:
        print("(negligible)")
    elif abs(h) < 0.5:
        print("(small)")
    elif abs(h) < 0.8:
        print("(medium)")
    else:
        print("(large)")

    # Bootstrap CI on aggregate delta
    bs = bootstrap_ci(all_baseline_outcomes, all_treatment_outcomes, ci=args.ci)
    print(f"Bootstrap {args.ci:.0%} CI:       [{bs['ci_lower']:+.1%}, {bs['ci_upper']:+.1%}]")

    # Sign test on per-task deltas
    p = sign_test_p(task_deltas)
    pos = sum(1 for d in task_deltas if d > 0)
    neg = sum(1 for d in task_deltas if d < 0)
    tied = sum(1 for d in task_deltas if d == 0)
    print(f"Sign test:              {pos} positive, {neg} negative, {tied} tied  (p={p:.4f})")

    # --- Compliance ---
    print("")
    print("=" * 80)
    print("COMPLIANCE (treatment condition only)")
    print("=" * 80)
    print(f"{'task_id':<50} {'edits':>6} {'tested':>7} {'rate':>8}")
    print("-" * 80)

    total_edits = 0
    total_tested = 0
    compliance_by_task: dict[str, list[float]] = defaultdict(list)

    for task_id in sorted(by_task.keys()):
        treat_runs = by_task[task_id].get(TREATMENT, [])
        for run in treat_runs:
            rep = run.get("rep", 0)
            log_key = f"{task_id}-{TREATMENT}-r{rep}"
            log_lines = logs.get(log_key, [])
            if not log_lines:
                continue

            comp = measure_compliance(log_lines)
            edits = comp["total_source_edits"]
            tested = comp["edits_followed_by_test"]
            total_edits += edits
            total_tested += tested
            if comp["compliance_rate"] is not None:
                compliance_by_task[task_id].append(comp["compliance_rate"])

    for task_id in sorted(compliance_by_task.keys()):
        rates = compliance_by_task[task_id]
        avg_rate = sum(rates) / len(rates)
        display_id = task_id[:48] + ".." if len(task_id) > 50 else task_id
        # Use total edits/tested for this task from the runs
        print(f"{display_id:<50} {'':>6} {'':>7} {avg_rate:>7.0%}")

    print("-" * 80)
    overall_compliance = total_tested / total_edits if total_edits > 0 else 0
    print(f"{'OVERALL':<50} {total_edits:>6} {total_tested:>7} {overall_compliance:>7.0%}")

    # --- Correlation: compliance vs success ---
    print("")
    print("=" * 80)
    print("COMPLIANCE vs SUCCESS CORRELATION")
    print("=" * 80)

    # Per-run: did the run comply well AND pass?
    compliant_pass = 0
    compliant_fail = 0
    noncompliant_pass = 0
    noncompliant_fail = 0
    compliance_threshold = 0.5

    for task_id in sorted(by_task.keys()):
        treat_runs = by_task[task_id].get(TREATMENT, [])
        for run in treat_runs:
            if run.get("error", "").startswith("["):
                continue  # skip infra errors
            rep = run.get("rep", 0)
            log_key = f"{task_id}-{TREATMENT}-r{rep}"
            log_lines = logs.get(log_key, [])
            if not log_lines:
                continue

            comp = measure_compliance(log_lines)
            if comp["compliance_rate"] is None:
                continue

            passed = run.get("success", False)
            compliant = comp["compliance_rate"] >= compliance_threshold

            if compliant and passed:
                compliant_pass += 1
            elif compliant and not passed:
                compliant_fail += 1
            elif not compliant and passed:
                noncompliant_pass += 1
            else:
                noncompliant_fail += 1

    total_comp = compliant_pass + compliant_fail + noncompliant_pass + noncompliant_fail
    if total_comp > 0:
        print(f"Compliance threshold: {compliance_threshold:.0%} of edits followed by test")
        print(f"")
        print(f"                        Pass    Fail")
        print(f"  Compliant runs:      {compliant_pass:>5}   {compliant_fail:>5}   "
              f"({compliant_pass / max(1, compliant_pass + compliant_fail):.0%} pass rate)")
        print(f"  Non-compliant runs:  {noncompliant_pass:>5}   {noncompliant_fail:>5}   "
              f"({noncompliant_pass / max(1, noncompliant_pass + noncompliant_fail):.0%} pass rate)")
    else:
        print("No compliance data available (logs missing?)")

    # --- JSON output ---
    summary = {
        "n_tasks": len(task_deltas),
        "n_baseline_runs": n_base,
        "n_treatment_runs": n_treat,
        "baseline_pass_rate": round(agg_base, 4),
        "treatment_pass_rate": round(agg_treat, 4),
        "aggregate_delta": round(agg_treat - agg_base, 4),
        "cohens_h": round(h, 4),
        f"bootstrap_ci_{args.ci:.0%}": {
            "lower": round(bs["ci_lower"], 4),
            "upper": round(bs["ci_upper"], 4),
        },
        "sign_test": {
            "positive": pos,
            "negative": neg,
            "tied": tied,
            "p_value": round(p, 6),
        },
        "compliance": {
            "total_edits": total_edits,
            "edits_followed_by_test": total_tested,
            "overall_rate": round(overall_compliance, 4),
        },
    }

    summary_path = results_dir / "intervention_analysis.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    main()
