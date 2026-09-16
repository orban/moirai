"""Estimator boundaries, missingness bounds, positive control, null calibration, power."""
from __future__ import annotations

import pytest

from moirai.intervention.analyze import cell_table, compare_selectors, contrast, neutral_summary, selector_report
from moirai.intervention.dryrun import run_dry_run, synthetic_candidate
from moirai.intervention.power import Design, simulate
from moirai.intervention.schema import TrialRecord


def rec(task, ckpt, arm, outcome, sel="s", i=0) -> TrialRecord:
    return TrialRecord(trial_id=f"{task}-{ckpt}-{arm}-{i}", task_id=task, checkpoint_id=ckpt, candidate_id=task, selector=sel,
                       arm=arm, block=i, seed=i, status="success" if outcome else ("failure" if outcome is False else "infra_failure"),
                       outcome=outcome, failure_class=None, grade=None, usage={"cost_usd": 0.1}, model_config_hash="",
                       prompt_hash="", tools_hash="", environment_hash="", workspace_hash_before="", workspace_hash_after=None,
                       injected_tool_call_id=None, injected_payload_hash=None, trajectory_ref=None, retry_of=None, started_at="", finished_at="")


def cells_for(spec: dict[tuple[str, str, str], list], sel="s"):
    recs = []
    for (t, c, arm), outcomes in spec.items():
        recs += [rec(t, c, arm, o, sel, i) for i, o in enumerate(outcomes)]
    return cell_table(recs), recs


class TestContrast:
    def test_all_pass_and_all_fail_cells_have_finite_nonzero_intervals(self):
        cells, _ = cells_for({("t", "c", "favored"): [True] * 5, ("t", "c", "disfavored"): [False] * 5})
        e = contrast(cells, "s", "favored", "disfavored", n_boot=300, seed=1)
        assert e.estimate == 1.0
        assert 0.0 < e.ci_low < e.ci_high <= 1.0
        assert e.bound_low == e.bound_high == 1.0

    def test_equal_task_weights(self):
        cells, _ = cells_for({
            ("a", "c", "favored"): [True] * 10, ("a", "c", "native"): [False] * 10,        # +1.0
            ("b", "c", "favored"): [False] * 2, ("b", "c", "native"): [False] * 2,          # 0.0
        })
        e = contrast(cells, "s", "favored", "native", n_boot=100)
        assert e.estimate == pytest.approx(0.5) and e.per_task == {"a": 1.0, "b": 0.0}

    def test_checkpoints_nested_in_task(self):
        cells, _ = cells_for({
            ("a", "c1", "favored"): [True] * 4, ("a", "c1", "native"): [False] * 4,
            ("a", "c2", "favored"): [False] * 4, ("a", "c2", "native"): [False] * 4,
        })
        e = contrast(cells, "s", "favored", "native", n_boot=100)
        assert e.estimate == pytest.approx(0.5) and e.n_checkpoints == 2 and e.n_tasks == 1

    def test_missing_outcome_bounds(self):
        cells, _ = cells_for({("t", "c", "favored"): [True, True, None, None], ("t", "c", "native"): [False, False, None]})
        e = contrast(cells, "s", "favored", "native", n_boot=100)
        assert e.estimate == 1.0 and e.n_missing == 3
        assert e.bound_low == pytest.approx(2 / 4 - 1 / 3)
        assert e.bound_high == pytest.approx(1.0)

    def test_none_when_arm_missing(self):
        cells, _ = cells_for({("t", "c", "favored"): [True]})
        assert contrast(cells, "s", "favored", "native") is None

    def test_neutral_summary_hides_arms(self):
        _, recs = cells_for({("t", "c", "favored"): [True], ("t", "c", "native"): [None]})
        s = neutral_summary(recs)
        assert all(k.startswith("arm-") for k in s) and "favored" not in s


class TestPositiveAndNullControls:
    def test_positive_control_recovers_sign(self, tmp_path):
        s = run_dry_run(tmp_path / "pc", n_tasks=3, reps=6, seed=11, p_favored=0.95, p_disfavored=0.05, p_native=0.5, n_boot=300)
        br = s["selector_reports"][0]["branch"]
        up = s["selector_reports"][0]["uplift"]
        assert br["estimate"] > 0.5 and br["ci_low"] > 0
        assert up["estimate"] > 0.2 and up["ci_low"] > 0

    def test_null_fixture_does_not_reject(self, tmp_path):
        s = run_dry_run(tmp_path / "null", n_tasks=3, reps=6, seed=12, null=True, n_boot=300)
        br = s["selector_reports"][0]["branch"]
        assert br["ci_low"] <= 0 <= br["ci_high"]

    def test_null_calibration_by_simulation(self):
        r = simulate(Design(n_tasks=4, checkpoints_per_task=2, reps_per_arm=6, branch_gap=0.0), n_sims=40, seed=3, n_boot=120)
        assert r.branch_rejection_rate <= 0.20
        assert r.branch_coverage >= 0.80
        assert r.uplift_coverage >= 0.80

    def test_power_rises_with_effect(self):
        small = simulate(Design(4, 2, 6, branch_gap=0.0), n_sims=25, seed=4, n_boot=100)
        large = simulate(Design(4, 2, 6, branch_gap=0.6, task_gap_jitter=0.05), n_sims=25, seed=4, n_boot=100)
        assert large.branch_rejection_rate > small.branch_rejection_rate


class TestSelectorReport:
    def test_coverage_costs_and_comparison(self):
        _, recs = cells_for({("t1", "c", "favored"): [True] * 3, ("t1", "c", "native"): [False] * 3}, sel="moirai")
        _, recs2 = cells_for({("t1", "c", "favored"): [False] * 3, ("t1", "c", "native"): [False] * 3}, sel="heuristic")
        from moirai.intervention.schema import Exclusion
        cands = [synthetic_candidate("t1", "moirai"), synthetic_candidate("t1", "heuristic", with_disfavored=False)]
        excl = [Exclusion("t2", "moirai", "0.1", "no_divergence_points")]
        rm = selector_report("moirai", recs + recs2, cands, excl, n_boot=50)
        rh = selector_report("heuristic", recs + recs2, cands, excl, n_boot=50)
        assert rm.coverage == 0.5 and rm.exclusion_reasons == {"no_divergence_points": 1}
        assert rh.coverage == 1.0 and rh.branch is None
        assert rm.execution_cost_usd == pytest.approx(0.6) and rm.cost_per_actionable_checkpoint_usd == pytest.approx(0.6)
        cmp = compare_selectors([rm, rh])
        assert cmp[0].n_shared_tasks == 1 and cmp[0].mean_uplift_difference == pytest.approx(1.0)
