"""Selector abstraction tests on a synthetic discovery dataset with a known fork."""
from __future__ import annotations

import json
from dataclasses import replace

from moirai.intervention.schema import CostRecord

from moirai.converters.swe_rebench import convert_row
from moirai.intervention.raw import load_raw_row
from moirai.intervention.schema import ActionPayload, CandidateIntervention, Exclusion
from moirai.intervention.selectors import (
    CriticSelector,
    DiscoveryDataset,
    HeuristicSelector,
    MoiraiSelector,
    RandomSelector,
    TaskSpec,
    build_selector,
    payload_from_event,
    select_all,
)
from moirai.normalize import normalize_run
from tests.test_intervention_provenance import EDIT, GREP, TEST, VIEW, make_row

README = ("str_replace_editor", {"command": "view", "path": "/w/README.md"}, "readme")
GREP_TEST = ("execute_bash", {"command": "grep -rn test_foo /w/tests"}, "/w/tests/t.py:3")
PASS_ACTIONS = [VIEW, GREP, EDIT, TEST]
FAIL_ACTIONS = [VIEW, README, EDIT, TEST]


def build_dataset(n_pass: int = 3, n_fail: int = 3, task: str = "task-1"):
    rows = []
    for i in range(n_pass):
        rows.append(make_row(f"p{i}", task, PASS_ACTIONS, resolved=1))
    for i in range(n_fail):
        rows.append(make_row(f"f{i}", task, FAIL_ACTIONS, resolved=0))
    raw = {r["trajectory_id"]: load_raw_row(r) for r in rows}
    runs = [normalize_run(convert_row(r, r["trajectory_id"]))[0] for r in rows]
    return DiscoveryDataset.build(runs, raw), rows


class TestMoiraiSelector:
    def test_finds_known_fork_with_exact_payloads(self):
        ds, _ = build_dataset()
        c = MoiraiSelector(min_runs=4, min_support=2).select(ds, TaskSpec("task-1", 0.4))
        assert isinstance(c, CandidateIntervention), c
        assert c.favored.tool_name == "execute_bash" and json.loads(c.favored.arguments_json)["command"] == "grep -rn foo /w"
        assert c.disfavored is not None and json.loads(c.disfavored.arguments_json)["path"] == "/w/README.md"
        assert c.anchor.message_idx == 4          # after the shared first view + observation
        assert c.evidence["favored_rate"] == 1.0 and c.evidence["disfavored_rate"] == 0.0
        assert c.structure_score == 0.4
        assert c.selection_cost.wall_seconds >= 0

    def test_deterministic_and_outcome_blind_anchor(self):
        ds, _ = build_dataset()
        a = MoiraiSelector().select(ds, TaskSpec("task-1"))
        b = MoiraiSelector().select(ds, TaskSpec("task-1"))
        assert replace(a, selection_cost=CostRecord()) == replace(b, selection_cost=CostRecord())
        # Anchor is the lowest-hash run among those active at the column, regardless of outcome.
        import hashlib
        ranks = sorted(ds.runs["task-1"], key=lambda r: hashlib.sha256(r.run_id.encode()).hexdigest())
        assert a.anchor.anchor_run_id == ranks[0].run_id

    def test_exclusion_when_no_divergence(self):
        ds, _ = build_dataset(n_pass=0, n_fail=4)
        r = MoiraiSelector().select(ds, TaskSpec("task-1"))
        assert isinstance(r, Exclusion) and r.reason == "single_outcome_task"

    def test_exclusion_when_branch_actions_ineligible(self):
        # Fork is edit vs create: both mutating, so no eligible pair.
        create = ("str_replace_editor", {"command": "create", "path": "/w/new.py", "file_text": "x"}, "created")
        rows = [make_row(f"p{i}", "t", [VIEW, EDIT, TEST], 1) for i in range(3)] + \
               [make_row(f"f{i}", "t", [VIEW, create, TEST], 0) for i in range(3)]
        raw = {r["trajectory_id"]: load_raw_row(r) for r in rows}
        runs = [normalize_run(convert_row(r, r["trajectory_id"]))[0] for r in rows]
        ds = DiscoveryDataset.build(runs, raw)
        r = MoiraiSelector().select(ds, TaskSpec("t"))
        assert isinstance(r, Exclusion) and r.reason == "no_eligible_candidate_pair"
        assert "branch_action_ineligible" in r.detail

    def test_unresolved_provenance_excluded(self):
        ds, _ = build_dataset()
        ds.provenance = {k: type(v)(v.run_id, v.trajectory_id, "failed", [], ["forced"]) for k, v in ds.provenance.items()}
        r = MoiraiSelector().select(ds, TaskSpec("task-1"))
        assert isinstance(r, Exclusion) and r.reason == "insufficient_discovery_runs"


class TestOtherSelectors:
    def test_heuristic_search_before_edit(self):
        ds, _ = build_dataset()
        c = HeuristicSelector().select(ds, TaskSpec("task-1"))
        assert isinstance(c, CandidateIntervention), c
        assert c.disfavored is None
        assert c.favored.action_class == "search"
        assert c.evidence["trigger_step_idx"] == 2

    def test_random_is_seed_deterministic(self):
        ds, _ = build_dataset()
        a = RandomSelector(seed=7).select(ds, TaskSpec("task-1"))
        b = RandomSelector(seed=7).select(ds, TaskSpec("task-1"))
        c = RandomSelector(seed=8).select(ds, TaskSpec("task-1"))
        assert isinstance(a, CandidateIntervention)
        assert replace(a, selection_cost=CostRecord()) == replace(b, selection_cost=CostRecord())
        assert a.favored.payload_hash != a.disfavored.payload_hash
        assert isinstance(c, CandidateIntervention)

    def test_critic_contract_enforced(self):
        ds, _ = build_dataset()
        bad = ActionPayload("str_replace_editor", json.dumps({"command": "str_replace", "path": "/w/a.py"}), "edit", "x", 0, "x")

        def critic_mutating(task, run_id, messages, pool):
            return 4, bad, None

        r = CriticSelector(critic_mutating, "test").select(ds, TaskSpec("task-1"))
        assert isinstance(r, Exclusion) and r.reason == "critic_action_ineligible"

        def critic_ok(task, run_id, messages, pool):
            run, ev = pool[0]
            return ev.call.message_idx, payload_from_event(run, ev), None

        c = CriticSelector(critic_ok, "test").select(ds, TaskSpec("task-1"))
        assert isinstance(c, CandidateIntervention) and c.selector == "critic"

        def critic_late(task, run_id, messages, pool):
            run, ev = pool[0]
            return 10, payload_from_event(run, ev), None    # after the test run: not replayable

        r = CriticSelector(critic_late, "test").select(ds, TaskSpec("task-1"))
        assert isinstance(r, Exclusion) and r.reason == "critic_location_not_replayable"

    def test_registry_and_select_all(self):
        ds, _ = build_dataset()
        for name in ("moirai", "heuristic", "random"):
            sel = build_selector(name)
            cands, excls = select_all(sel, ds, [TaskSpec("task-1"), TaskSpec("missing-task")])
            assert len(cands) == 1 and len(excls) == 1
            assert cands[0].selector == name
        try:
            build_selector("critic")
        except ValueError as e:
            assert "critic callable" in str(e)

    def test_candidate_carries_no_solution_leakage(self):
        ds, rows = build_dataset()
        c = MoiraiSelector().select(ds, TaskSpec("task-1"))
        blob = json.dumps(c.evidence) + c.favored.arguments_json + c.disfavored.arguments_json
        assert "diff --git" not in blob
        assert "resolved" not in c.favored.arguments_json
