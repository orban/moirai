"""Policy v2: verified full replay. Static admissibility, certificates, enrolment, selector wiring."""
from __future__ import annotations

import json

from moirai.converters.swe_rebench import convert_row
from moirai.intervention.eligibility import (
    POLICY_V2,
    POLICY_VERSION,
    anchor_admitted,
    certificate_from_report,
    classify_action,
    load_certificates,
    network_command,
    prefix_admissibility,
    screen_events_v2,
)
from moirai.intervention.raw import index_events, load_raw_row
from moirai.intervention.schema import CandidateIntervention, Exclusion, ReplayCertificate
from moirai.intervention.selectors import DiscoveryDataset, HeuristicSelector, MoiraiSelector, RandomSelector, TaskSpec, fork_coverage
from moirai.normalize import normalize_run
from tests.test_intervention_provenance import EDIT, GREP, TEST, VIEW, make_row
from tests.test_intervention_selectors import FAIL_ACTIONS, PASS_ACTIONS

CURL = ("execute_bash", {"command": "curl -s https://example.com/x"}, "ok")
STDIN = ("execute_bash", {"command": "y", "is_input": True}, "done")
SCRIPT = ("execute_bash", {"command": "cd /w && python repro.py"}, "Traceback")


def _events(actions, tid="t1"):
    row = make_row(tid, "task-1", actions, resolved=1)
    return index_events(load_raw_row(row))[0]


def _report(tid="t1", boundary=10, first_rejected=None, baseline_ok=True, accepted=None):
    return {"trajectory_id": tid, "anchor_message_idx": boundary, "baseline": {"ok": baseline_ok},
            "first_rejected_message_idx": first_rejected, "n_steps": 5, "instrument_hash": "abc", "judged_with": "def",
            "anchor_accepted": (first_rejected is None and baseline_ok) if accepted is None else accepted}


def _cert(tid="t1", through=1000, baseline_ok=True):
    return ReplayCertificate(tid, "", "abc", "def", baseline_ok, through, None, through if baseline_ok else None, True, 5)


class TestStaticScreen:
    def test_scripts_and_tests_are_admissible(self):
        ev = _events([VIEW, SCRIPT, TEST, GREP])
        pr = prefix_admissibility(ev, ev[-1].call.message_idx)
        assert pr.status == "admissible" and pr.n_prefix_events == 3

    def test_stdin_and_network_block(self):
        ev = _events([VIEW, STDIN, GREP])
        pr = prefix_admissibility(ev, ev[-1].call.message_idx)
        assert pr.status == "inadmissible" and pr.blocking_classes == ("stdin_to_process",)
        assert pr.first_blocking_message_idx == ev[1].call.message_idx
        ev = _events([CURL, GREP])
        assert prefix_admissibility(ev, ev[-1].call.message_idx).blocking_classes == ("network_command",)

    def test_network_patterns(self):
        assert network_command("pip install requests")
        assert network_command("git clone https://github.com/x/y")
        assert network_command("python -c \"import requests; requests.get('http://x')\"")
        assert not network_command("cd /w && python -m pytest tests/ -q")
        assert not network_command("git status")


class TestCertificates:
    def test_no_rejection_certifies_to_boundary(self):
        c = certificate_from_report(_report(boundary=40))
        assert c.certified_through == 40 and c.complete and c.baseline_ok

    def test_rejection_certifies_up_to_rejected_step(self):
        c = certificate_from_report(_report(boundary=40, first_rejected=22))
        assert c.certified_through == 22 and not c.complete

    def test_baseline_failure_certifies_nothing(self):
        c = certificate_from_report(_report(baseline_ok=False))
        assert c.certified_through is None and not c.complete

    def test_load_picks_widest(self, tmp_path):
        (tmp_path / "a-auto.json").write_text(json.dumps(_report(boundary=16)))
        (tmp_path / "a-deep.json").write_text(json.dumps(_report(boundary=80, first_rejected=54)))
        (tmp_path / "b-full.json").write_text(json.dumps(_report(tid="t2", boundary=90)))
        (tmp_path / "junk.json").write_text("{}")
        certs = load_certificates([tmp_path])
        assert certs["t1"].certified_through == 54 and certs["t2"].certified_through == 90


class TestEnrolment:
    def test_enrols_post_test_reads_when_certified(self):
        ev = _events([VIEW, SCRIPT, TEST, GREP, VIEW])
        rows = screen_events_v2("t1", "task-1", ev, _cert(through=ev[3].call.message_idx))
        by = {r.message_idx: r for r in rows}
        assert by[ev[3].call.message_idx].enrolled and by[ev[3].call.message_idx].prefix_status == "certified"
        assert by[ev[4].call.message_idx].exclusion_reason == "replay:rejected_before_anchor"
        assert by[ev[2].call.message_idx].exclusion_reason.startswith("action:")
        assert all(r.policy_version == POLICY_V2 for r in rows)

    def test_no_certificate_or_failed_baseline(self):
        ev = _events([VIEW, GREP])
        assert screen_events_v2("t1", "task-1", ev, None)[1].exclusion_reason == "replay:no_certificate"
        assert screen_events_v2("t1", "task-1", ev, _cert(baseline_ok=False))[1].exclusion_reason == "replay:baseline_failed"

    def test_static_block_beats_certificate(self):
        ev = _events([VIEW, STDIN, GREP])
        rows = screen_events_v2("t1", "task-1", ev, _cert())
        assert rows[2].exclusion_reason == "prefix:stdin_to_process"

    def test_anchor_admitted_dispatch(self):
        ev = _events([VIEW, TEST, GREP])
        m = ev[2].call.message_idx
        assert not anchor_admitted(POLICY_VERSION, ev, m, None)          # test in prefix blocks v1
        assert anchor_admitted(POLICY_V2, ev, m, _cert())
        assert not anchor_admitted(POLICY_V2, ev, m, _cert(through=m - 1))


def _dataset(policy, certs):
    rows = [make_row(f"p{i}", "task-1", PASS_ACTIONS, resolved=1) for i in range(3)]
    rows += [make_row(f"f{i}", "task-1", FAIL_ACTIONS, resolved=0) for i in range(3)]
    raw = {r["trajectory_id"]: load_raw_row(r) for r in rows}
    runs = [normalize_run(convert_row(r, r["trajectory_id"]))[0] for r in rows]
    return DiscoveryDataset.build(runs, raw, policy=policy, certificates=certs)


class TestSelectorsUnderV2:
    def test_moirai_needs_certificates_under_v2(self):
        ds = _dataset(POLICY_V2, {})
        r = MoiraiSelector(min_runs=4, min_support=2).select(ds, TaskSpec("task-1"))
        assert isinstance(r, Exclusion) and "no_replayable_anchor" in r.detail

    def test_moirai_finds_same_fork_with_certificates(self):
        ids = [f"p{i}" for i in range(3)] + [f"f{i}" for i in range(3)]
        ds = _dataset(POLICY_V2, {i: _cert(i) for i in ids})
        c = MoiraiSelector(min_runs=4, min_support=2).select(ds, TaskSpec("task-1"))
        assert isinstance(c, CandidateIntervention) and c.anchor.message_idx == 4

    def test_random_pool_shrinks_to_certified_prefixes(self):
        ids = [f"p{i}" for i in range(3)] + [f"f{i}" for i in range(3)]
        full = len(_dataset(POLICY_V2, {i: _cert(i) for i in ids}).enrolled_events("task-1"))
        partial = len(_dataset(POLICY_V2, {i: _cert(i, through=2) for i in ids}).enrolled_events("task-1"))
        assert full > partial > 0
        assert isinstance(RandomSelector(seed=1).select(_dataset(POLICY_V2, {i: _cert(i) for i in ids}), TaskSpec("task-1")), CandidateIntervention)


CD_GREP = ("execute_bash", {"command": "cd /w && grep -rn foo ."}, "./a.py:1:foo")
LATE_PASS = [VIEW, TEST, CD_GREP, EDIT]
LATE_FAIL = [VIEW, TEST, ("str_replace_editor", {"command": "view", "path": "/w/README.md"}, "readme"), EDIT]


class TestActionRuleV2:
    def test_cd_prefixed_read_admitted_only_under_v2(self):
        a = json.dumps({"command": "cd /w && grep -rn foo ."})
        assert not classify_action("execute_bash", a).eligible
        v = classify_action("execute_bash", a, POLICY_V2)
        assert v.eligible and v.action_class == "search" and v.reasons == ("cd_prefix_changes_persistent_cwd",)
        assert not classify_action("execute_bash", json.dumps({"command": "cd /w && python x.py"}), POLICY_V2).eligible
        assert not classify_action("execute_bash", json.dumps({"command": "cd /w"}), POLICY_V2).eligible


def _late_dataset(policy, certs):
    rows = [make_row(f"p{i}", "task-1", LATE_PASS, resolved=1) for i in range(3)]
    rows += [make_row(f"f{i}", "task-1", LATE_FAIL, resolved=0) for i in range(3)]
    raw = {r["trajectory_id"]: load_raw_row(r) for r in rows}
    runs = [normalize_run(convert_row(r, r["trajectory_id"]))[0] for r in rows]
    return DiscoveryDataset.build(runs, raw, policy=policy, certificates=certs)


class TestForkCoverage:
    def test_post_test_fork_admitted_only_under_v2(self):
        ids = [f"p{i}" for i in range(3)] + [f"f{i}" for i in range(3)]
        v1 = fork_coverage(_late_dataset(POLICY_VERSION, {}), TaskSpec("task-1"))
        v2 = fork_coverage(_late_dataset(POLICY_V2, {i: _cert(i) for i in ids}), TaskSpec("task-1"))
        assert v1["forks"] and v2["forks"]
        f1 = next(f for f in v1["forks"] if f["n_active"] == 6)
        f2 = next(f for f in v2["forks"] if f["n_active"] == 6)
        assert f1["n_admitted"] == 0 and not f1["candidate_possible"]
        assert f2["n_admitted"] == 6 and f2["pair_eligible"] and f2["candidate_possible"]
        assert all(a["message_idx"] == 6 for a in f2["anchors"])

    def test_selectors_emit_candidates_at_post_test_fork_under_v2(self):
        ids = [f"p{i}" for i in range(3)] + [f"f{i}" for i in range(3)]
        ds = _late_dataset(POLICY_V2, {i: _cert(i) for i in ids})
        m = MoiraiSelector(min_runs=4, min_support=2).select(ds, TaskSpec("task-1"))
        assert isinstance(m, CandidateIntervention) and m.anchor.message_idx == 6
        assert json.loads(m.favored.arguments_json)["command"].startswith("cd /w && grep")
        assert isinstance(MoiraiSelector(min_runs=4, min_support=2).select(_late_dataset(POLICY_VERSION, {}), TaskSpec("task-1")), Exclusion)
        h = HeuristicSelector().select(ds, TaskSpec("task-1"))
        assert isinstance(h, CandidateIntervention), h
