#!/usr/bin/env python3
"""Experiment: do behavioral-feature correlations survive within task_family?

Tests the Simpson's paradox hypothesis: if ansible tasks pass at ~72% while
other families pass at ~0%, features that happen to concentrate in ansible
(e.g. subagent_used) will show spurious correlation with success overall
but vanish when you control for task_family.

Steps:
  1. Load eval_harness runs, group by task_family
  2. Print task_family breakdown (count, pass rate)
  3. Compute point-biserial correlations overall
  4. Recompute within ansible-only and non-ansible subsets
  5. Compare: which features survive within-family?
  6. Run motif discovery (find_motifs, find_gapped_motifs) on ansible-only
     vs whole dataset and compare significant pattern counts
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moirai.analyze.motifs import find_gapped_motifs, find_motifs
from moirai.compress import step_enriched_name
from moirai.load import load_runs
from moirai.schema import Run


# ---------------------------------------------------------------------------
# Feature extractors (same as experiment_features.py)
# ---------------------------------------------------------------------------

def _enriched(run: Run) -> list[str]:
    return [n for s in run.steps if (n := step_enriched_name(s)) is not None]


def feat_test_after_edit(run: Run) -> float:
    names = _enriched(run)
    for i in range(len(names) - 1):
        if names[i].startswith("edit") and names[i + 1].startswith("test"):
            return 1.0
    return 0.0


def feat_iterative_fix(run: Run) -> float:
    names = _enriched(run)
    for i in range(len(names) - 2):
        if names[i].startswith("edit") and names[i + 1].startswith("test") and names[i + 2].startswith("edit"):
            return 1.0
    return 0.0


def feat_search_before_edit(run: Run) -> float:
    names = _enriched(run)
    for i in range(len(names) - 1):
        if names[i].startswith("search") and names[i + 1].startswith("edit"):
            return 1.0
    return 0.0


def feat_step_count(run: Run) -> float:
    return float(len(run.steps))


def feat_step_failure_rate(run: Run) -> float:
    if not run.steps:
        return 0.0
    return sum(1 for s in run.steps if s.status != "ok") / len(run.steps)


def feat_reasoning_density(run: Run) -> float:
    if not run.steps:
        return 0.0
    return sum(1 for s in run.steps if s.output and s.output.get("reasoning")) / len(run.steps)


def feat_test_fail_loop(run: Run) -> float:
    names = _enriched(run)
    consec = 0
    for name in names:
        if name == "test(fail)":
            consec += 1
            if consec >= 3:
                return 1.0
        elif name.startswith("edit"):
            consec = 0
    return 0.0


def feat_exploration_ratio(run: Run) -> float:
    names = _enriched(run)
    if not names:
        return 0.0
    explore = sum(1 for n in names if n.startswith("read") or n.startswith("search"))
    return explore / len(names)


def feat_edit_density(run: Run) -> float:
    names = _enriched(run)
    if not names:
        return 0.0
    edits = sum(1 for n in names if n.startswith("edit") or n.startswith("write"))
    return edits / len(names)


def feat_late_test_ratio(run: Run) -> float:
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
    names = _enriched(run)
    if not names:
        return 0.0
    return sum(1 for n in names if n.startswith("bash")) / len(names)


def feat_subagent_used(run: Run) -> float:
    names = _enriched(run)
    return 1.0 if "subagent" in names else 0.0


def feat_unique_files_read(run: Run) -> float:
    files = set()
    for s in run.steps:
        if s.name == "read" and s.attrs.get("file_path"):
            files.add(s.attrs["file_path"])
    return float(len(files))


def feat_unique_files_edited(run: Run) -> float:
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


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def compute_correlations(
    runs: list[Run],
    label: str,
) -> dict[str, tuple[float, float]]:
    """Compute point-biserial r for each feature vs success.

    Returns {feature_name: (r_pb, p_value)}.
    Skips features with no variance.
    """
    known = [r for r in runs if r.result.success is not None]
    pass_runs = [r for r in known if r.result.success]
    fail_runs = [r for r in known if not r.result.success]

    if not pass_runs or not fail_runs:
        return {}

    results: dict[str, tuple[float, float]] = {}
    for feat_name, feat_fn in ALL_FEATURES.items():
        pass_vals = np.array([feat_fn(r) for r in pass_runs])
        fail_vals = np.array([feat_fn(r) for r in fail_runs])
        all_vals = np.concatenate([pass_vals, fail_vals])
        all_outcomes = np.concatenate([np.ones(len(pass_vals)), np.zeros(len(fail_vals))])

        if np.std(all_vals) < 1e-10:
            results[feat_name] = (0.0, 1.0)
            continue

        r_pb, p_pb = scipy_stats.pointbiserialr(all_outcomes, all_vals)
        results[feat_name] = (r_pb, p_pb)

    return results


def print_correlation_table(
    corrs: dict[str, tuple[float, float]],
    label: str,
    n_pass: int,
    n_fail: int,
) -> None:
    """Print a sorted correlation table."""
    items = sorted(corrs.items(), key=lambda x: -abs(x[1][0]))

    print(f"\n{'Feature':<25} {'r_pb':>8} {'p-value':>10} {'Sig':>5}")
    print("-" * 52)

    sig_count = 0
    for feat_name, (r_pb, p_val) in items:
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
        if p_val < 0.05:
            sig_count += 1
        print(f"{feat_name:<25} {r_pb:>8.3f} {p_val:>10.4f} {sig:>5}")

    print(f"\n  {sig_count} / {len(corrs)} features significant at p<0.05")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_dir = Path("/Users/ryo/dev/moirai/examples/eval_harness")
    runs, warnings = load_runs(data_dir)
    known = [r for r in runs if r.result.success is not None]

    print("=" * 75)
    print("TASK FAMILY ABLATION EXPERIMENT")
    print("=" * 75)
    print(f"\nLoaded {len(runs)} runs ({len(known)} with known outcomes)")

    # -----------------------------------------------------------------------
    # 1. Task family breakdown
    # -----------------------------------------------------------------------
    families: dict[str, list[Run]] = defaultdict(list)
    for r in known:
        fam = r.task_family or "unknown"
        families[fam].append(r)

    print(f"\n{'─' * 60}")
    print("TASK FAMILY BREAKDOWN")
    print(f"{'─' * 60}")
    print(f"\n{'Family':<25} {'Count':>6} {'Pass':>6} {'Fail':>6} {'Rate':>8}")
    print("-" * 55)

    for fam in sorted(families.keys(), key=lambda f: -len(families[f])):
        fam_runs = families[fam]
        n_pass = sum(1 for r in fam_runs if r.result.success)
        n_fail = len(fam_runs) - n_pass
        rate = n_pass / len(fam_runs) if fam_runs else 0
        print(f"{fam:<25} {len(fam_runs):>6} {n_pass:>6} {n_fail:>6} {rate:>7.1%}")

    total_pass = sum(1 for r in known if r.result.success)
    total_fail = len(known) - total_pass
    print("-" * 55)
    print(f"{'TOTAL':<25} {len(known):>6} {total_pass:>6} {total_fail:>6} {total_pass / len(known):>7.1%}")

    # -----------------------------------------------------------------------
    # 2. Overall correlations
    # -----------------------------------------------------------------------
    print(f"\n{'─' * 60}")
    print(f"OVERALL CORRELATIONS ({len(known)} runs, {total_pass}P/{total_fail}F)")
    print(f"{'─' * 60}")

    overall_corrs = compute_correlations(known, "overall")
    print_correlation_table(overall_corrs, "overall", total_pass, total_fail)

    # -----------------------------------------------------------------------
    # 3. Ansible-only correlations
    # -----------------------------------------------------------------------
    ansible_runs = families.get("ansible_ansible", [])
    ansible_pass = sum(1 for r in ansible_runs if r.result.success)
    ansible_fail = len(ansible_runs) - ansible_pass

    print(f"\n{'─' * 60}")
    print(f"ANSIBLE-ONLY CORRELATIONS ({len(ansible_runs)} runs, {ansible_pass}P/{ansible_fail}F)")
    print(f"{'─' * 60}")

    if ansible_pass > 0 and ansible_fail > 0:
        ansible_corrs = compute_correlations(ansible_runs, "ansible")
        print_correlation_table(ansible_corrs, "ansible", ansible_pass, ansible_fail)
    else:
        print("\n  SKIPPED: no outcome variance within ansible")
        ansible_corrs = {}

    # -----------------------------------------------------------------------
    # 4. Non-ansible correlations
    # -----------------------------------------------------------------------
    non_ansible_runs = [r for r in known if (r.task_family or "unknown") != "ansible_ansible"]
    non_ansible_pass = sum(1 for r in non_ansible_runs if r.result.success)
    non_ansible_fail = len(non_ansible_runs) - non_ansible_pass

    print(f"\n{'─' * 60}")
    print(f"NON-ANSIBLE CORRELATIONS ({len(non_ansible_runs)} runs, {non_ansible_pass}P/{non_ansible_fail}F)")
    print(f"{'─' * 60}")

    if non_ansible_pass > 0 and non_ansible_fail > 0:
        non_ansible_corrs = compute_correlations(non_ansible_runs, "non-ansible")
        print_correlation_table(non_ansible_corrs, "non-ansible", non_ansible_pass, non_ansible_fail)
    else:
        print("\n  SKIPPED: no outcome variance in non-ansible runs")
        non_ansible_corrs = {}

    # -----------------------------------------------------------------------
    # 5. Per-family correlations (any family with outcome variance)
    # -----------------------------------------------------------------------
    print(f"\n{'─' * 60}")
    print("PER-FAMILY CORRELATIONS (families with outcome variance)")
    print(f"{'─' * 60}")

    family_corrs: dict[str, dict[str, tuple[float, float]]] = {}
    for fam, fam_runs in sorted(families.items()):
        fam_pass = sum(1 for r in fam_runs if r.result.success)
        fam_fail = len(fam_runs) - fam_pass
        if fam_pass == 0 or fam_fail == 0:
            print(f"\n  {fam}: SKIPPED (all {'pass' if fam_pass > 0 else 'fail'}, n={len(fam_runs)})")
            continue

        fam_corr = compute_correlations(fam_runs, fam)
        family_corrs[fam] = fam_corr
        print(f"\n  {fam} ({len(fam_runs)} runs, {fam_pass}P/{fam_fail}F, "
              f"rate={fam_pass / len(fam_runs):.1%}):")

        sig_in_family = [
            (name, r, p)
            for name, (r, p) in sorted(fam_corr.items(), key=lambda x: -abs(x[1][0]))
            if p < 0.05
        ]
        if sig_in_family:
            for name, r, p in sig_in_family:
                direction = "+" if r > 0 else "-"
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else "*")
                print(f"    {name:<25} r={r:+.3f} p={p:.4f} {sig}")
        else:
            print("    (no features significant at p<0.05)")

    # -----------------------------------------------------------------------
    # 6. Comparison: which features survive within-family?
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 75}")
    print("SIMPSON'S PARADOX CHECK: CONFOUNDED vs SURVIVING FEATURES")
    print(f"{'=' * 75}")

    overall_sig = {name for name, (r, p) in overall_corrs.items() if p < 0.05}
    ansible_sig = {name for name, (r, p) in ansible_corrs.items() if p < 0.05} if ansible_corrs else set()

    # A feature survives if it's significant in at least one within-family test
    within_family_sig: set[str] = set()
    for fam, fam_corr in family_corrs.items():
        for name, (r, p) in fam_corr.items():
            if p < 0.05:
                within_family_sig.add(name)

    confounded = overall_sig - within_family_sig
    surviving = overall_sig & within_family_sig
    family_only = within_family_sig - overall_sig

    print(f"\nFeatures significant overall:         {len(overall_sig)}")
    print(f"Features significant within-family:   {len(within_family_sig)}")
    print(f"Features significant in ansible-only: {len(ansible_sig)}")

    print(f"\nCONFOUNDED (significant overall but NOT within any family):")
    if confounded:
        for name in sorted(confounded):
            r_overall, p_overall = overall_corrs[name]
            print(f"  {name:<25} overall r={r_overall:+.3f} p={p_overall:.4f}")
            # Show per-family values for context
            for fam, fam_corr in family_corrs.items():
                if name in fam_corr:
                    r_fam, p_fam = fam_corr[name]
                    print(f"    within {fam}: r={r_fam:+.3f} p={p_fam:.4f}")
    else:
        print("  (none)")

    print(f"\nSURVIVING (significant overall AND within at least one family):")
    if surviving:
        for name in sorted(surviving):
            r_overall, p_overall = overall_corrs[name]
            print(f"  {name:<25} overall r={r_overall:+.3f} p={p_overall:.4f}")
            for fam, fam_corr in family_corrs.items():
                if name in fam_corr:
                    r_fam, p_fam = fam_corr[name]
                    marker = " <-- sig" if p_fam < 0.05 else ""
                    print(f"    within {fam}: r={r_fam:+.3f} p={p_fam:.4f}{marker}")
    else:
        print("  (none)")

    print(f"\nFAMILY-ONLY (significant within a family but NOT overall):")
    if family_only:
        for name in sorted(family_only):
            r_overall, p_overall = overall_corrs[name]
            print(f"  {name:<25} overall r={r_overall:+.3f} p={p_overall:.4f}")
            for fam, fam_corr in family_corrs.items():
                if name in fam_corr:
                    r_fam, p_fam = fam_corr[name]
                    marker = " <-- sig" if p_fam < 0.05 else ""
                    print(f"    within {fam}: r={r_fam:+.3f} p={p_fam:.4f}{marker}")
    else:
        print("  (none)")

    # -----------------------------------------------------------------------
    # 7. Direction flips (sign reversal between overall and within-family)
    # -----------------------------------------------------------------------
    print(f"\n{'─' * 60}")
    print("DIRECTION FLIPS (sign of r reverses within a family)")
    print(f"{'─' * 60}")

    flip_count = 0
    for name in sorted(ALL_FEATURES.keys()):
        if name not in overall_corrs:
            continue
        r_overall, _ = overall_corrs[name]
        if abs(r_overall) < 0.01:
            continue
        for fam, fam_corr in family_corrs.items():
            if name in fam_corr:
                r_fam, p_fam = fam_corr[name]
                if abs(r_fam) > 0.01 and (r_overall > 0) != (r_fam > 0):
                    flip_count += 1
                    print(f"  {name:<25} overall r={r_overall:+.3f}  "
                          f"within {fam} r={r_fam:+.3f} (p={p_fam:.4f})")

    if flip_count == 0:
        print("  (no direction flips detected)")

    # -----------------------------------------------------------------------
    # 8. Motif discovery comparison: whole dataset vs ansible-only
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 75}")
    print("MOTIF DISCOVERY: WHOLE DATASET vs ANSIBLE-ONLY")
    print(f"{'=' * 75}")

    # Whole dataset
    motifs_all, n_cand_all = find_motifs(known)
    gapped_all, n_gcand_all = find_gapped_motifs(known)

    print(f"\nWhole dataset ({len(known)} runs):")
    print(f"  Contiguous motifs: {n_cand_all} candidates tested, {len(motifs_all)} significant (q<0.05)")
    print(f"  Gapped motifs:     {n_gcand_all} candidates tested, {len(gapped_all)} significant (q<0.05)")

    if motifs_all:
        print(f"\n  Top 5 contiguous motifs (whole dataset):")
        for m in motifs_all[:5]:
            print(f"    {m.display:<50} lift={m.lift:.2f} q={m.q_value:.4f} "
                  f"({m.success_runs}P/{m.fail_runs}F)")

    if gapped_all:
        print(f"\n  Top 5 gapped motifs (whole dataset):")
        for m in gapped_all[:5]:
            print(f"    {m.display:<50} lift={m.lift:.2f} q={m.q_value:.4f} "
                  f"({m.success_runs}P/{m.fail_runs}F)")

    # Ansible-only
    if ansible_pass > 0 and ansible_fail > 0:
        motifs_ansible, n_cand_ansible = find_motifs(ansible_runs)
        gapped_ansible, n_gcand_ansible = find_gapped_motifs(ansible_runs)

        print(f"\nAnsible-only ({len(ansible_runs)} runs):")
        print(f"  Contiguous motifs: {n_cand_ansible} candidates tested, "
              f"{len(motifs_ansible)} significant (q<0.05)")
        print(f"  Gapped motifs:     {n_gcand_ansible} candidates tested, "
              f"{len(gapped_ansible)} significant (q<0.05)")

        if motifs_ansible:
            print(f"\n  Top 5 contiguous motifs (ansible-only):")
            for m in motifs_ansible[:5]:
                print(f"    {m.display:<50} lift={m.lift:.2f} q={m.q_value:.4f} "
                      f"({m.success_runs}P/{m.fail_runs}F)")

        if gapped_ansible:
            print(f"\n  Top 5 gapped motifs (ansible-only):")
            for m in gapped_ansible[:5]:
                print(f"    {m.display:<50} lift={m.lift:.2f} q={m.q_value:.4f} "
                      f"({m.success_runs}P/{m.fail_runs}F)")

        # Comparison
        print(f"\n{'─' * 60}")
        print("MOTIF COMPARISON SUMMARY")
        print(f"{'─' * 60}")
        print(f"  {'Metric':<35} {'Whole':>10} {'Ansible':>10}")
        print(f"  {'-' * 55}")
        print(f"  {'Contiguous candidates':<35} {n_cand_all:>10} {n_cand_ansible:>10}")
        print(f"  {'Contiguous significant':<35} {len(motifs_all):>10} {len(motifs_ansible):>10}")
        print(f"  {'Gapped candidates':<35} {n_gcand_all:>10} {n_gcand_ansible:>10}")
        print(f"  {'Gapped significant':<35} {len(gapped_all):>10} {len(gapped_ansible):>10}")

        # Check overlap: which whole-dataset motifs also appear in ansible-only
        all_patterns = {m.pattern for m in motifs_all}
        ansible_patterns = {m.pattern for m in motifs_ansible}
        shared = all_patterns & ansible_patterns
        all_only = all_patterns - ansible_patterns
        ansible_only_patterns = ansible_patterns - all_patterns

        print(f"\n  Contiguous pattern overlap:")
        print(f"    Shared:              {len(shared)}")
        print(f"    Whole-only:          {len(all_only)} (likely confounded by family)")
        print(f"    Ansible-only:        {len(ansible_only_patterns)} (real within-family signal)")

        all_gapped_patterns = {m.anchors for m in gapped_all}
        ansible_gapped_patterns = {m.anchors for m in gapped_ansible}
        g_shared = all_gapped_patterns & ansible_gapped_patterns
        g_all_only = all_gapped_patterns - ansible_gapped_patterns
        g_ansible_only = ansible_gapped_patterns - all_gapped_patterns

        print(f"\n  Gapped pattern overlap:")
        print(f"    Shared:              {len(g_shared)}")
        print(f"    Whole-only:          {len(g_all_only)} (likely confounded by family)")
        print(f"    Ansible-only:        {len(g_ansible_only)} (real within-family signal)")
    else:
        print("\n  Ansible-only: SKIPPED (no outcome variance)")

    # -----------------------------------------------------------------------
    # 9. Verdict
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 75}")
    print("VERDICT")
    print(f"{'=' * 75}")

    total_overall_sig = len(overall_sig)
    total_surviving = len(surviving)
    total_confounded = len(confounded)
    pct_confounded = total_confounded / total_overall_sig * 100 if total_overall_sig > 0 else 0

    print(f"\n  Of {total_overall_sig} features significant in the overall dataset:")
    print(f"    {total_surviving} survive within-family analysis (real signal)")
    print(f"    {total_confounded} are confounded by task_family ({pct_confounded:.0f}% Simpson's paradox)")
    if family_only:
        print(f"    {len(family_only)} are significant only within a family (masked overall)")

    if pct_confounded > 50:
        print(f"\n  HYPOTHESIS SUPPORTED: majority of correlations ({pct_confounded:.0f}%) are")
        print(f"  Simpson's paradox artifacts driven by task_family composition.")
    elif pct_confounded > 0:
        print(f"\n  HYPOTHESIS PARTIALLY SUPPORTED: {pct_confounded:.0f}% of correlations are confounded,")
        print(f"  but some real signal survives within-family.")
    else:
        print(f"\n  HYPOTHESIS REJECTED: no confounded features found. Correlations")
        print(f"  appear genuine within task families.")


if __name__ == "__main__":
    main()
