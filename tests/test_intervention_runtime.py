"""Checkpoint isolation, injection protocol, randomisation, ledger, budget and runner tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from moirai.intervention.budget import BudgetExceeded, BudgetGuard, PriceTable
from moirai.intervention.checkpoint import (
    LocalDirWorkspaceProvider,
    ManifestDrift,
    build_manifest,
    contamination_probe,
    verify_clone,
    verify_independent,
)
from moirai.intervention.dryrun import SYNTHETIC_ENV, SYNTHETIC_MODEL, SYNTHETIC_TOOLS, run_dry_run, synthetic_candidate, synthetic_prefix
from moirai.intervention.grade import FixtureGrader, grade_with_retry
from moirai.intervention.inject import AtMostOnce, build_forced_events, new_tool_call_id, validate_history, validate_prefix_boundary
from moirai.intervention.ledger import Ledger, LedgerConflict, LedgerCorrupt
from moirai.intervention.randomize import ManifestDrift as AssignDrift
from moirai.intervention.randomize import assign, balance_report, load_manifest, write_manifest
from moirai.intervention.runner import CheckpointBundle, run_trials
from moirai.intervention.schema import CostRecord, TrialRecord
from moirai.intervention.synthetic import SyntheticRuntime, synthetic_budget


def _base(tmp_path: Path, provider: LocalDirWorkspaceProvider, task="t0"):
    d = tmp_path / "base" / task
    d.mkdir(parents=True)
    (d / "a.py").write_text("x=1\n")
    (d / "sub").mkdir()
    (d / "sub" / "b.txt").write_text("b\n")
    return provider.register_base(d, f"base-{task}")


class TestCheckpoint:
    def test_clone_identity_and_independence(self, tmp_path):
        p = LocalDirWorkspaceProvider(tmp_path / "ws")
        base = _base(tmp_path, p)
        a, b = p.clone(base), p.clone(base)
        assert a.hash() == b.hash() == base.hash()
        assert a.path != b.path
        assert verify_independent(a, b) == []

    def test_contamination_probe(self, tmp_path):
        p = LocalDirWorkspaceProvider(tmp_path / "ws")
        base = _base(tmp_path, p)
        a, b, c = p.clone(base), p.clone(base), p.clone(base)
        assert contamination_probe(a, [b, c], base) == []
        assert a.hash() != b.hash()

    def test_manifest_and_drift(self, tmp_path):
        p = LocalDirWorkspaceProvider(tmp_path / "ws")
        base = _base(tmp_path, p)
        cand = synthetic_candidate("t0")
        prefix = synthetic_prefix("t0")
        m = build_manifest("t0", cand.anchor, prefix, SYNTHETIC_TOOLS, base, SYNTHETIC_ENV, SYNTHETIC_MODEL, synthetic_budget(), "replayable")
        clone = p.clone(base)
        assert verify_clone(m, clone, prefix, SYNTHETIC_TOOLS, SYNTHETIC_MODEL) == []
        (clone.path / "a.py").write_text("x=2\n")
        assert any("workspace hash" in s for s in verify_clone(m, clone, prefix, SYNTHETIC_TOOLS, SYNTHETIC_MODEL))
        assert any("canonical request" in s for s in verify_clone(m, p.clone(base), prefix, SYNTHETIC_TOOLS, {"model": "other"}))
        with pytest.raises(ManifestDrift):
            build_manifest("t0", cand.anchor, prefix[:-1], SYNTHETIC_TOOLS, base, SYNTHETIC_ENV, SYNTHETIC_MODEL, synthetic_budget(), "replayable")


class TestInjection:
    def test_forced_events_share_id_and_validate(self):
        cand = synthetic_candidate("t0")
        tcid = new_tool_call_id("trial", cand.favored.payload_hash)
        a, t = build_forced_events(cand.favored, tcid, "obs")
        assert a["tool_calls"][0]["id"] == t["tool_call_id"] == tcid
        assert a["tool_calls"][0]["function"]["arguments"] == cand.favored.arguments_json
        assert a["content"] == ""
        msgs = synthetic_prefix("t0") + [a, t]
        assert validate_history(msgs) == []
        assert validate_prefix_boundary(msgs) == []
        assert validate_prefix_boundary(msgs[:-1])          # ends with assistant
        assert any("orphan" in e for e in validate_history(msgs[:-2] + [t]))
        assert any("duplicate" in e for e in validate_history(msgs + [a, t]))
        assert new_tool_call_id("trial2", cand.favored.payload_hash) != tcid

    def test_at_most_once(self):
        g = AtMostOnce()
        g.claim("t", "c1")
        with pytest.raises(RuntimeError):
            g.claim("t", "c1")
        g.claim("t2", "c1")


class TestRandomization:
    def test_balance_and_determinism(self):
        cands = [synthetic_candidate(f"t{i}") for i in range(3)] + [synthetic_candidate("t9", with_disfavored=False)]
        cps = {c.candidate_id: f"cp-{c.task_id}" for c in cands}
        m1 = assign(cands, cps, reps=4, seed=1)
        m2 = assign(cands, cps, reps=4, seed=1)
        m3 = assign(cands, cps, reps=4, seed=2)
        assert m1.manifest_hash == m2.manifest_hash != m3.manifest_hash
        for cid, counts in balance_report(m1).items():
            n_arms = 2 if cid == [c for c in cands if c.disfavored is None][0].candidate_id else 3
            assert len(counts) == n_arms and set(counts.values()) == {4}
        # Every block contains each arm once per candidate, interleaved across candidates.
        for b in range(4):
            block = [a for a in m1.assignments if a.block == b]
            assert len(block) == 3 * 3 + 2
        assert len({a.trial_id for a in m1.assignments}) == len(m1.assignments)
        assert len({a.seed for a in m1.assignments}) == len(m1.assignments)
        assert not any(a.arm in a.trial_id for a in m1.assignments)

    def test_manifest_drift_rejected(self, tmp_path):
        cands = [synthetic_candidate("t0")]
        m = assign(cands, {cands[0].candidate_id: "cp"}, reps=2, seed=3)
        write_manifest(tmp_path / "m.json", m)
        assert load_manifest(tmp_path / "m.json").manifest_hash == m.manifest_hash
        data = json.loads((tmp_path / "m.json").read_text())
        data["assignments"][0]["arm"] = "native"
        (tmp_path / "m.json").write_text(json.dumps(data))
        with pytest.raises(AssignDrift):
            load_manifest(tmp_path / "m.json")


def _record(trial_id: str, arm: str = "favored", outcome=True, cost=0.5) -> TrialRecord:
    return TrialRecord(trial_id=trial_id, task_id="t", checkpoint_id="c", candidate_id="k", selector="s", arm=arm,
                       block=0, seed=1, status="success" if outcome else "failure", outcome=outcome, failure_class=None,
                       grade=None, usage={"cost_usd": cost}, model_config_hash="", prompt_hash="", tools_hash="",
                       environment_hash="", workspace_hash_before="", workspace_hash_after=None, injected_tool_call_id=None,
                       injected_payload_hash=None, trajectory_ref=None, retry_of=None, started_at="", finished_at="")


class TestLedger:
    def test_idempotent_conflict_and_chain(self, tmp_path):
        led = Ledger(tmp_path / "l.jsonl")
        assert led.append(_record("a")) is True
        assert led.append(_record("a")) is False
        with pytest.raises(LedgerConflict):
            led.append(_record("a", outcome=False))
        led.append(_record("b", cost=0.25))
        again = Ledger(tmp_path / "l.jsonl")
        assert again.spent_usd() == 0.75 and len(again.records()) == 2
        assert again.summary()["by_arm"]["favored"]["success"] == 2
        lines = (tmp_path / "l.jsonl").read_text().splitlines()
        tampered = json.loads(lines[0])
        tampered["record"]["outcome"] = False
        (tmp_path / "l.jsonl").write_text(json.dumps(tampered) + "\n" + lines[1] + "\n")
        with pytest.raises(LedgerCorrupt):
            Ledger(tmp_path / "l.jsonl")


class TestBudget:
    def test_fails_closed_before_spending(self):
        g = BudgetGuard(cap_usd=3.0, reserve_per_trial_usd=1.0)
        g.authorize(2)
        with pytest.raises(BudgetExceeded):
            g.authorize(2)
        assert g.in_flight == 2
        g.commit(CostRecord(cost_usd=0.4))
        g.commit(CostRecord(input_tokens=1_000_000))
        assert g.spent_usd == pytest.approx(0.7)
        g.authorize(1)
        with pytest.raises(BudgetExceeded):
            g.authorize(2)

    def test_default_reserve_is_high_scenario(self):
        g = BudgetGuard(cap_usd=200.0)
        assert g.reserve_per_trial_usd == pytest.approx(PriceTable().cost(BudgetGuard.__dataclass_fields__ and __import__("moirai.intervention.budget", fromlist=["HIGH_SCENARIO"]).HIGH_SCENARIO))
        assert g.reserve_per_trial_usd > 1.0
        with pytest.raises(BudgetExceeded):
            BudgetGuard(cap_usd=1.0).authorize(1)


class TestGrader:
    def test_retry_only_on_grader_error(self, tmp_path):
        p = LocalDirWorkspaceProvider(tmp_path / "ws")
        base = _base(tmp_path, p)
        (base.path / "outcome.json").write_text(json.dumps({"task_id": "t0", "passed": True, "grader_flaky_once": True}))
        r = grade_with_retry(FixtureGrader(), base, "t0", max_attempts=3)
        assert r.status == "ok" and r.passed is True and r.attempts == 2
        (base.path / "outcome.json").unlink()
        r = grade_with_retry(FixtureGrader(), base, "t0", max_attempts=2)
        assert r.status == "grader_error" and r.attempts == 2


class TestRunnerDryRun:
    def test_end_to_end_hand_audit(self, tmp_path):
        s = run_dry_run(tmp_path / "dr", n_tasks=2, reps=2, seed=5, n_boot=100)
        assert s["instrument_checks"]["clone_identity"] is True
        assert s["instrument_checks"]["clone_independence_problems"] == []
        assert s["instrument_checks"]["contaminated_by_probe"] == []
        assert s["native_untouched"] and s["forced_arms_injected"] and s["distinct_workspaces"] and s["at_most_once"]
        run = s["run"]
        assert run["executed"] == 2 * 2 * 3 and run["infra_failure"] == 1
        led = Ledger(tmp_path / "dr" / "ledger.jsonl")
        recs = {r.trial_id: r for r in led.records()}
        infra = [r for r in recs.values() if r.status == "infra_failure"]
        assert len(infra) == 1 and infra[0].outcome is None and infra[0].failure_class == "infra:restore"
        agent_err = [r for r in recs.values() if r.failure_class == "agent_error"]
        assert len(agent_err) == 1 and agent_err[0].status == "failure" and agent_err[0].outcome is False
        flaky = [r for r in recs.values() if r.grade and r.grade["attempts"] == 2]
        assert len(flaky) == 1
        for r in recs.values():
            assert r.usage["cost_usd"] > 0 or r.status == "infra_failure"
            if r.arm != "native" and r.status != "infra_failure":
                assert r.injected_tool_call_id and r.injected_payload_hash
        assert s["ledger_regenerated"]["spent_usd"] == pytest.approx(s["budget"]["spent_usd"])
        # Rerunning resumes: nothing is executed twice.
        s2 = run_dry_run(tmp_path / "dr", n_tasks=2, reps=2, seed=5, n_boot=100)
        assert s2["run"]["executed"] == 0 and s2["run"]["skipped_existing"] == 12

    def test_budget_stops_run_before_paid_step(self, tmp_path):
        s = run_dry_run(tmp_path / "dr", n_tasks=2, reps=1, seed=5, cap_usd=1.0, n_boot=50)
        assert s["run"]["executed"] == 0 and s["run"]["stopped_reason"].startswith("budget")
        assert not (tmp_path / "dr" / "ledger.jsonl").exists()

    def test_drift_stops_run(self, tmp_path):
        p = LocalDirWorkspaceProvider(tmp_path / "ws")
        base = _base(tmp_path, p)
        cand = synthetic_candidate("t0")
        prefix = synthetic_prefix("t0")
        m = build_manifest("t0", cand.anchor, prefix, SYNTHETIC_TOOLS, base, SYNTHETIC_ENV, SYNTHETIC_MODEL, synthetic_budget(), "replayable")
        (base.path / "a.py").write_text("drifted\n")          # base changes after the manifest froze
        manifest = assign([cand], {cand.candidate_id: m.checkpoint_id}, reps=1, seed=1)
        rt = SyntheticRuntime({cand.favored.payload_hash: 0.9, cand.disfavored.payload_hash: 0.1}, 0.5)
        led = Ledger(tmp_path / "l.jsonl")
        rep = run_trials(manifest, {cand.candidate_id: cand}, {m.checkpoint_id: CheckpointBundle(m, prefix, SYNTHETIC_TOOLS, base)},
                         rt, FixtureGrader(), led, BudgetGuard(100.0, reserve_per_trial_usd=1.0), p)
        assert rep.executed == 0 and "drift" in rep.stopped_reason
        assert led.records() == []
