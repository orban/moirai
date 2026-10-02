"""Tests for run-set diagnosability (DDU).

Every expected value here is hand-computable from the definitions in
diagnosability.py, so a regression shows up as a wrong number rather than a
vague failure. Formulas are from Perez, Abreu & van Deursen, ICSE 2017.
"""

import pytest

from moirai.schema import Step, Result, Run
from moirai.analyze.diagnosability import (
    build_activity_matrix,
    compute_all_diagnosability,
    compute_diagnosability,
    step_name_signature,
    step_target_signature,
)


def _run(run_id, task_id, names, success=True, targets=None):
    steps = [
        Step(idx=i, type="tool", name=n, attrs={"file_path": targets[i]} if targets else {})
        for i, n in enumerate(names)
    ]
    return Run(run_id=run_id, task_id=task_id, steps=steps, result=Result(success=success))


class TestActivityMatrix:
    def test_rows_are_component_sets_and_repeats_collapse(self):
        runs = [_run("r0", "t", ["read", "read", "edit"]), _run("r1", "t", ["edit", "test"])]
        rows, components = build_activity_matrix(runs)
        assert rows == [frozenset({"read", "edit"}), frozenset({"edit", "test"})]
        assert components == ["edit", "read", "test"]

    def test_target_signature_splits_a_name_by_what_it_touched(self):
        runs = [_run("r0", "t", ["read", "read"], targets=["a.py", "b.py"])]
        rows, components = build_activity_matrix(runs, signature=step_target_signature)
        assert components == ["read|a.py", "read|b.py"]
        # The same two steps are one component under the coarse alphabet.
        assert len(build_activity_matrix(runs, signature=step_name_signature)[1]) == 1


class TestDiversity:
    def test_identical_runs_have_zero_diversity_and_zero_ddu(self):
        runs = [_run(f"r{i}", "t", ["read", "edit"]) for i in range(4)]
        d = compute_diagnosability(runs)
        assert d.diversity == 0.0
        assert d.n_distinct_rows == 1
        assert d.ddu == 0.0

    def test_all_distinct_runs_have_diversity_one(self):
        runs = [_run("r0", "t", ["a"]), _run("r1", "t", ["b"]), _run("r2", "t", ["c"])]
        assert compute_diagnosability(runs).diversity == 1.0

    def test_half_colliding_matches_gini_simpson_by_hand(self):
        # 4 runs, two groups of 2. collisions = 2*1 + 2*1 = 4; N(N-1) = 12.
        runs = [
            _run("r0", "t", ["a"]), _run("r1", "t", ["a"]),
            _run("r2", "t", ["b"]), _run("r3", "t", ["b"]),
        ]
        d = compute_diagnosability(runs)
        assert d.diversity == pytest.approx(1 - 4 / 12)
        assert d.n_distinct_rows == 2

    def test_single_run_is_not_scored_as_zero_diversity_by_accident(self):
        # Diversity is undefined for N=1; the guard returns 0.0 and min_runs
        # keeps such tasks out of aggregate reporting entirely.
        d = compute_diagnosability([_run("r0", "t", ["a"])])
        assert d.diversity == 0.0
        assert compute_all_diagnosability({"t": [_run("r0", "t", ["a"])]}) == []


class TestDensity:
    def test_raw_density_of_one_half_renormalizes_to_ideal(self):
        # 2 runs, 2 components, each run touches exactly one => rho = 2/4 = 0.5
        runs = [_run("r0", "t", ["a"]), _run("r1", "t", ["b"])]
        d = compute_diagnosability(runs)
        assert d.raw_density == pytest.approx(0.5)
        assert d.density == pytest.approx(1.0)

    def test_saturated_matrix_is_worst_case_density(self):
        # Every run touches every component => rho = 1.0 => rho' = 0.0
        runs = [_run(f"r{i}", "t", ["a", "b"]) for i in range(3)]
        d = compute_diagnosability(runs)
        assert d.raw_density == pytest.approx(1.0)
        assert d.density == pytest.approx(0.0)
        assert d.ddu == 0.0

    def test_renormalization_is_symmetric_about_one_half(self):
        sparse = compute_diagnosability(
            [_run("r0", "t", ["a"]), _run("r1", "t", ["b"]), _run("r2", "t", ["c"]),
             _run("r3", "t", ["d"])]
        )
        # rho = 4/16 = 0.25 => rho' = 1 - |1 - 0.5| = 0.5
        assert sparse.raw_density == pytest.approx(0.25)
        assert sparse.density == pytest.approx(0.5)


class TestUniqueness:
    def test_components_always_hit_together_form_one_ambiguity_group(self):
        # a and b are hit by exactly the same runs; nothing can separate them.
        runs = [_run("r0", "t", ["a", "b"]), _run("r1", "t", ["c"])]
        d = compute_diagnosability(runs)
        assert d.n_components == 3
        assert d.n_ambiguity_groups == 2          # {a,b} and {c}
        assert d.uniqueness == pytest.approx(2 / 3)

    def test_fully_separable_components_have_uniqueness_one(self):
        runs = [_run("r0", "t", ["a"]), _run("r1", "t", ["b"]), _run("r2", "t", ["c"])]
        assert compute_diagnosability(runs).uniqueness == pytest.approx(1.0)


class TestDDU:
    def test_ddu_is_the_product_of_its_three_terms(self):
        runs = [
            _run("r0", "t", ["a", "b"]), _run("r1", "t", ["b", "c"]),
            _run("r2", "t", ["c"]), _run("r3", "t", ["a", "c"]),
        ]
        d = compute_diagnosability(runs)
        assert d.ddu == pytest.approx(d.density * d.diversity * d.uniqueness)
        assert 0.0 <= d.ddu <= 1.0

    def test_any_term_at_zero_collapses_the_product(self):
        identical = [_run(f"r{i}", "t", ["a", "b"]) for i in range(3)]
        assert compute_diagnosability(identical).ddu == 0.0

    def test_no_pass_fail_label_changes_the_score(self):
        """DDU is structural. Flipping every outcome must not move it."""
        names = [["a", "b"], ["b", "c"], ["c"], ["a", "c"]]
        passing = [_run(f"r{i}", "t", n, success=True) for i, n in enumerate(names)]
        failing = [_run(f"r{i}", "t", n, success=False) for i, n in enumerate(names)]
        assert compute_diagnosability(passing).ddu == compute_diagnosability(failing).ddu

    def test_empty_run_set_does_not_raise(self):
        d = compute_diagnosability([], task_id="t")
        assert d.ddu == 0.0
        assert d.n_runs == 0


class TestAggregate:
    def test_results_are_sorted_worst_first(self):
        groups = {
            "good": [_run("g0", "good", ["a"]), _run("g1", "good", ["b"])],
            "bad": [_run("b0", "bad", ["a", "b"]), _run("b1", "bad", ["a", "b"])],
        }
        out = compute_all_diagnosability(groups)
        assert [d.task_id for d in out] == ["bad", "good"]

    def test_min_runs_filter_drops_undermeasured_tasks(self):
        groups = {"one": [_run("x", "one", ["a"])],
                  "four": [_run(f"y{i}", "four", ["a", "b"][: i % 2 + 1]) for i in range(4)]}
        assert [d.task_id for d in compute_all_diagnosability(groups, min_runs=2)] == ["four"]
