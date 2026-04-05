#!/usr/bin/env python3
"""Experiment: which behavioral features predict success/failure?

For each dataset, computes the 8 canonical behavioral features per-run
(not per-cohort), then tests correlation with outcomes using:
  1. Point-biserial correlation (feature value vs binary outcome)
  2. Mann-Whitney U (feature distribution in pass vs fail runs)
  3. Feature value distributions (mean/std for pass vs fail)

Also adds features beyond the existing 8:
  - Test-fail loop presence (3+ consecutive test(fail) without edit)
  - Exploration ratio (search + read steps / total steps)
  - Edit density (edit steps / total steps)
  - Late-test ratio (fraction of test steps in second half of trajectory)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moirai.compress import step_enriched_name
from moirai.load import load_runs
from moirai.schema import Run


# --- Per-run feature extractors ---
# Each returns a float for a single run.

def _enriched(run: Run) -> list[str]:
    return [n for s in run.steps if (n := step_enriched_name(s)) is not None]


def feat_test_after_edit(run: Run) -> float:
    """1.0 if any edit is followed by a test, else 0.0."""
    names = _enriched(run)
    for i in range(len(names) - 1):
        if names[i].startswith("edit") and names[i + 1].startswith("test"):
            return 1.0
    return 0.0


def feat_iterative_fix(run: Run) -> float:
    """1.0 if contains edit→test→edit pattern."""
    names = _enriched(run)
    for i in range(len(names) - 2):
        if names[i].startswith("edit") and names[i + 1].startswith("test") and names[i + 2].startswith("edit"):
            return 1.0
    return 0.0


def feat_search_before_edit(run: Run) -> float:
    """1.0 if any search precedes any edit."""
    names = _enriched(run)
    for i in range(len(names) - 1):
        if names[i].startswith("search") and names[i + 1].startswith("edit"):
            return 1.0
    return 0.0


def feat_step_count(run: Run) -> float:
    """Number of steps."""
    return float(len(run.steps))


def feat_step_failure_rate(run: Run) -> float:
    """Fraction of steps with non-ok status."""
    if not run.steps:
        return 0.0
    return sum(1 for s in run.steps if s.status != "ok") / len(run.steps)


def feat_reasoning_density(run: Run) -> float:
    """Fraction of steps with reasoning."""
    if not run.steps:
        return 0.0
    return sum(1 for s in run.steps if s.output and s.output.get("reasoning")) / len(run.steps)


def feat_test_fail_loop(run: Run) -> float:
    """1.0 if 3+ consecutive test(fail) steps without intervening edit."""
    names = _enriched(run)
    consec = 0
    for name in names:
        if name == "test(fail)":
            consec += 1
            if consec >= 3:
                return 1.0
        elif name.startswith("edit"):
            consec = 0
        # Other steps don't reset the counter
    return 0.0


def feat_exploration_ratio(run: Run) -> float:
    """Fraction of steps that are search or read."""
    names = _enriched(run)
    if not names:
        return 0.0
    explore = sum(1 for n in names if n.startswith("read") or n.startswith("search"))
    return explore / len(names)


def feat_edit_density(run: Run) -> float:
    """Fraction of steps that are edit or write."""
    names = _enriched(run)
    if not names:
        return 0.0
    edits = sum(1 for n in names if n.startswith("edit") or n.startswith("write"))
    return edits / len(names)


def feat_late_test_ratio(run: Run) -> float:
    """Fraction of test steps that appear in second half of trajectory."""
    names = _enriched(run)
    if not names:
        return 0.0
    mid = len(names) // 2
    total_tests = sum(1 for n in names if n.startswith("test"))
    if total_tests == 0:
        return 0.0
    late_tests = sum(1 for i, n in enumerate(names) if n.startswith("test") and i >= mid)
    return late_tests / total_tests


def feat_bash_ratio(run: Run) -> float:
    """Fraction of steps that are bash commands."""
    names = _enriched(run)
    if not names:
        return 0.0
    return sum(1 for n in names if n.startswith("bash")) / len(names)


def feat_subagent_used(run: Run) -> float:
    """1.0 if subagent was invoked."""
    names = _enriched(run)
    return 1.0 if "subagent" in names else 0.0


def feat_unique_files_read(run: Run) -> float:
    """Number of distinct files read."""
    files = set()
    for s in run.steps:
        if s.name == "read" and s.attrs.get("file_path"):
            files.add(s.attrs["file_path"])
    return float(len(files))


def feat_unique_files_edited(run: Run) -> float:
    """Number of distinct files edited."""
    files = set()
    for s in run.steps:
        if s.name in ("edit", "write") and s.attrs.get("file_path"):
            files.add(s.attrs["file_path"])
    return float(len(files))


ALL_FEATURES: dict[str, callable] = {
    "test_after_edit": feat_test_after_edit,
    "iterative_fix": feat_iterative_fix,
    "search_before_edit": feat_search_before_edit,
    "step_count": feat_step_count,
    "step_failure_rate": feat_step_failure_rate,
    "reasoning_density": feat_reasoning_density,
    "test_fail_loop": feat_test_fail_loop,
    "exploration_ratio": feat_exploration_ratio,
    "edit_density": feat_edit_density,
    "late_test_ratio": feat_late_test_ratio,
    "bash_ratio": feat_bash_ratio,
    "subagent_used": feat_subagent_used,
    "unique_files_read": feat_unique_files_read,
    "unique_files_edited": feat_unique_files_edited,
}


def analyze_dataset(name: str, runs: list[Run]) -> None:
    known = [r for r in runs if r.result.success is not None]
    n_pass = sum(1 for r in known if r.result.success)
    n_fail = len(known) - n_pass
    pass_rate = n_pass / len(known) if known else 0

    print(f"\n{'='*75}")
    print(f"{name}: {len(known)} runs, {n_pass}P/{n_fail}F ({pass_rate:.0%} pass)")
    print(f"{'='*75}")

    pass_runs = [r for r in known if r.result.success]
    fail_runs = [r for r in known if not r.result.success]

    results = []

    for feat_name, feat_fn in ALL_FEATURES.items():
        # Compute per-run
        pass_vals = np.array([feat_fn(r) for r in pass_runs])
        fail_vals = np.array([feat_fn(r) for r in fail_runs])
        all_vals = np.concatenate([pass_vals, fail_vals])
        all_outcomes = np.concatenate([np.ones(len(pass_vals)), np.zeros(len(fail_vals))])

        # Skip if no variance
        if np.std(all_vals) < 1e-10:
            results.append((feat_name, 0.0, 1.0, np.mean(pass_vals), np.mean(fail_vals), 0.0, "no variance"))
            continue

        # Point-biserial correlation
        r_pb, p_pb = scipy_stats.pointbiserialr(all_outcomes, all_vals)

        # Mann-Whitney U
        if len(pass_vals) > 0 and len(fail_vals) > 0:
            u_stat, p_mw = scipy_stats.mannwhitneyu(pass_vals, fail_vals, alternative="two-sided")
        else:
            p_mw = 1.0

        # Effect size: rank-biserial correlation from U
        n1, n2 = len(pass_vals), len(fail_vals)
        rank_biserial = 1 - 2 * u_stat / (n1 * n2) if n1 > 0 and n2 > 0 else 0.0

        delta = np.mean(pass_vals) - np.mean(fail_vals)
        results.append((feat_name, r_pb, p_pb, np.mean(pass_vals), np.mean(fail_vals), rank_biserial, "ok"))

    # Sort by absolute correlation
    results.sort(key=lambda x: -abs(x[1]))

    # Print
    print(f"\n{'Feature':<25} {'r_pb':>8} {'p-value':>10} {'Pass mean':>10} {'Fail mean':>10} {'Delta':>8} {'rank_bis':>9} {'Sig':>5}")
    print("-" * 95)

    for feat_name, r_pb, p_val, pass_mean, fail_mean, rank_bis, status in results:
        if status == "no variance":
            print(f"{feat_name:<25} {'---':>8} {'---':>10} {pass_mean:>10.3f} {fail_mean:>10.3f} {'---':>8} {'---':>9} {'---':>5}")
            continue

        delta = pass_mean - fail_mean
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
        print(f"{feat_name:<25} {r_pb:>8.3f} {p_val:>10.4f} {pass_mean:>10.3f} {fail_mean:>10.3f} {delta:>+8.3f} {rank_bis:>+9.3f} {sig:>5}")

    # Summary: which features are significant?
    sig_features = [(name, r, p) for name, r, p, _, _, _, status in results if status == "ok" and p < 0.05]
    print(f"\n{len(sig_features)} / {len(ALL_FEATURES)} features significant at p<0.05")

    if sig_features:
        print("\nTop predictors of success:")
        for name, r, p in sig_features[:5]:
            direction = "more → success" if r > 0 else "more → failure"
            print(f"  {name}: r={r:.3f} ({direction})")


def main():
    examples_dir = Path("/Users/ryo/dev/moirai/examples")
    datasets = [
        ("eval_harness", examples_dir / "eval_harness"),
        ("swe_smith", examples_dir / "swe_smith"),
        ("swe_agent", examples_dir / "swe_agent"),
    ]

    for name, path in datasets:
        if not path.exists():
            print(f"Skipping {name}: {path} not found")
            continue
        runs, _ = load_runs(path)
        analyze_dataset(name, runs)


if __name__ == "__main__":
    main()
