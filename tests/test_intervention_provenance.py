"""Provenance, raw audit and eligibility tests for the intervention harness."""
from __future__ import annotations

import json

from moirai.converters.swe_rebench import convert_row
from moirai.intervention.eligibility import classify_action, prefix_restorability, screen_events
from moirai.intervention.provenance import resolve_provenance, truncated_fields
from moirai.intervention.raw import index_events, load_raw_row
from moirai.normalize import normalize_run


def tool_call(tcid: str, name: str, args: dict) -> dict:
    return {"id": tcid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def assistant(content: str, *calls: dict) -> dict:
    return {"role": "assistant", "content": content, "name": None, "tool_call_id": None, "tool_calls": list(calls) or None}


def observation(tcid: str, name: str, content: str) -> dict:
    return {"role": "tool", "content": content, "name": name, "tool_call_id": tcid, "tool_calls": None}


def make_row(trajectory_id: str, instance_id: str, actions: list[tuple[str, dict, str]], resolved: int = 1) -> dict:
    """actions: (tool_name, args, observation_text) executed one per assistant message."""
    msgs = [
        {"role": "system", "content": "sys", "name": None, "tool_call_id": None, "tool_calls": None},
        {"role": "user", "content": "task", "name": None, "tool_call_id": None, "tool_calls": None},
    ]
    for i, (name, args, obs) in enumerate(actions):
        tcid = f"{trajectory_id}-call-{i}"
        msgs.append(assistant("thinking" if i == 0 else "", tool_call(tcid, name, args)))
        msgs.append(observation(tcid, name, obs))
    return {"trajectory_id": trajectory_id, "instance_id": instance_id, "repo": "o/r", "trajectory": msgs,
            "tools": [{"function": {"name": "execute_bash"}}], "model_patch": "diff --git a b", "exit_status": "submit",
            "resolved": resolved, "gen_tests_correct": None, "pred_passes_gen_tests": None}


VIEW = ("str_replace_editor", {"command": "view", "path": "/w/a.py"}, "1\tprint()")
GREP = ("execute_bash", {"command": "grep -rn foo /w"}, "/w/a.py:1:foo")
EDIT = ("str_replace_editor", {"command": "str_replace", "path": "/w/a.py", "old_str": "a", "new_str": "b"}, "edited")
TEST = ("execute_bash", {"command": "cd /w && python -m pytest -q"}, "1 passed exit code 0")


def _run_from_row(row: dict):
    run, _ = normalize_run(convert_row(row, row["trajectory_id"]))
    return run


class TestRawIndex:
    def test_observations_matched_by_id_not_position(self):
        row = make_row("t1", "task-1", [VIEW, GREP])
        # Swap the two observations' positions; ids still identify them.
        msgs = row["trajectory"]
        msgs[3], msgs[5] = msgs[5], msgs[3]
        events, audit = index_events(load_raw_row(row))
        assert [e.observation for e in events] == ["1\tprint()", "/w/a.py:1:foo"]
        assert audit.n_orphan_observations == 0 and audit.n_missing_observations == 0

    def test_audit_counts(self):
        row = make_row("t1", "task-1", [VIEW])
        row["trajectory"].append(assistant("", tool_call("m1", "execute_bash", {"command": "ls"}), tool_call("m2", "execute_bash", {"command": "pwd"})))
        row["trajectory"].append(observation("m1", "execute_bash", "a"))
        row["trajectory"].append(observation("zzz", "execute_bash", "orphan"))
        _, audit = index_events(load_raw_row(row))
        assert audit.n_multi_call_messages == 1
        assert audit.n_missing_observations == 1
        assert audit.n_orphan_observations == 1
        assert audit.n_tool_calls == 3

    def test_model_patch_is_hashed_not_stored(self):
        traj = load_raw_row(make_row("t1", "task-1", [VIEW]))
        assert traj.model_patch_hash and "diff" not in traj.model_patch_hash


class TestProvenance:
    def test_round_trip_resolves_every_step(self):
        row = make_row("t1", "task-1", [VIEW, GREP, EDIT, TEST])
        traj = load_raw_row(row)
        events, _ = index_events(traj)
        pm = resolve_provenance(_run_from_row(row), traj, events)
        assert pm.status == "resolved", pm.reasons
        assert [e.tool_call_id for e in pm.entries] == [f"t1-call-{i}" for i in range(4)]
        assert [e.message_idx for e in pm.entries] == [2, 4, 6, 8]
        assert all(e.observation_message_idx == e.message_idx + 1 for e in pm.entries)
        assert all(e.arguments_hash for e in pm.entries)

    def test_multi_tool_call_misassociation_fails_closed(self):
        row = make_row("t1", "task-1", [VIEW])
        row["trajectory"].append(assistant("", tool_call("m1", "execute_bash", {"command": "ls"}), tool_call("m2", "execute_bash", {"command": "pwd"})))
        row["trajectory"].append(observation("m2", "execute_bash", "second first"))
        row["trajectory"].append(observation("m1", "execute_bash", "first second"))
        traj = load_raw_row(row)
        pm = resolve_provenance(_run_from_row(row), traj, index_events(traj)[0])
        assert pm.status == "failed"
        assert any("attached to call" in r for r in pm.reasons)

    def test_stored_run_drift_fails_closed(self):
        row = make_row("t1", "task-1", [VIEW, GREP])
        run = _run_from_row(row)
        run.steps[1].name = "read"     # tamper with the stored run
        traj = load_raw_row(row)
        pm = resolve_provenance(run, traj, index_events(traj)[0])
        assert pm.status == "failed" and any("stored != regenerated" in r for r in pm.reasons)
        assert pm.entries == []

    def test_step_count_mismatch_fails_closed(self):
        row = make_row("t1", "task-1", [VIEW, GREP])
        run = _run_from_row(row)
        run.steps.pop()
        traj = load_raw_row(row)
        pm = resolve_provenance(run, traj, index_events(traj)[0])
        assert pm.status == "failed" and "step count mismatch" in pm.reasons[0]

    def test_id_mismatch_fails_closed(self):
        row = make_row("t1", "task-1", [VIEW])
        run = _run_from_row(row)
        traj = load_raw_row(row)
        traj.trajectory_id = "other"
        assert resolve_provenance(run, traj).status == "failed"

    def test_truncation_detected(self):
        long_obs = "x" * 5000
        row = make_row("t1", "task-1", [("execute_bash", {"command": "cat " + "a" * 600}, long_obs)])
        traj = load_raw_row(row)
        pm = resolve_provenance(_run_from_row(row), traj, index_events(traj)[0])
        assert pm.status == "resolved"
        assert set(pm.entries[0].truncated_fields) >= {"result", "command"}
        assert truncated_fields({"output": {"action": "a" * 500}, "attrs": {}}) == ("action",)


class TestEligibility:
    def test_allowlisted_reads(self):
        assert classify_action("str_replace_editor", json.dumps({"command": "view", "path": "/w"})).eligible
        assert classify_action("execute_bash", json.dumps({"command": "grep -rn foo /w | head -20"})).eligible
        assert classify_action("execute_bash", json.dumps({"command": "find /w -name '*.py'"})).eligible
        assert classify_action("execute_bash", json.dumps({"command": "cat /w/a.py"})).eligible
        assert classify_action("execute_bash", json.dumps({"command": "ls -la /w"})).action_class == "list"

    def test_mutations_and_shell_state_excluded(self):
        cases = [
            (("str_replace_editor", {"command": "str_replace", "path": "/w"}), "file_mutation"),
            (("str_replace_editor", {"command": "create", "path": "/w"}), "file_creation"),
            (("execute_bash", {"command": "cd /w && cat a.py"}), "cd_prefix_changes_persistent_cwd"),
            (("execute_bash", {"command": "cat a.py > b.py"}), "forbidden_token:>"),
            (("execute_bash", {"command": "grep foo $(ls)"}), "forbidden_token:$("),
            (("execute_bash", {"command": "find . -name x -delete"}), "find_with_side_effect_flag"),
            (("execute_bash", {"command": "grep foo a | xargs rm"}), "pipe_target_not_allowlisted:xargs"),
            (("execute_bash", {"command": "python -m pytest"}), "test_execution"),
            (("execute_bash", {"command": "pip install x"}), "setup_command:pip"),
            (("execute_bash", {"command": "python a.py"}), "command_not_allowlisted:python"),
            (("execute_bash", {"command": "export X=1"}), "shell_environment_mutation"),
            (("execute_bash", {"command": "sleep 100 &"}), "background_process"),
            (("execute_bash", {"command": "", "is_input": True}), "stdin_input_to_running_process"),
            (("think", {"thought": "hmm"}), "reasoning_not_an_environment_action"),
            (("browser", {"code": "x"}), "browser_action"),
        ]
        for (tool, args), reason in cases:
            v = classify_action(tool, json.dumps(args))
            assert not v.eligible, (tool, args)
            assert reason in v.reasons, (tool, args, v.reasons)

    def test_undecodable_arguments_excluded(self):
        v = classify_action("execute_bash", "{not json")
        assert not v.eligible and "undecodable_arguments" in v.reasons

    def test_prefix_restorability(self):
        row = make_row("t1", "task-1", [VIEW, EDIT, GREP, TEST, VIEW])
        events, _ = index_events(load_raw_row(row))
        assert prefix_restorability(events, 6).status == "replayable"          # before GREP: view, edit
        blocked = prefix_restorability(events, 10)                                # after TEST
        assert blocked.status == "requires_full_replay" and "test" in blocked.blocking_classes
        assert blocked.first_blocking_message_idx == 8

    def test_screen_events_reasons(self):
        row = make_row("t1", "task-1", [VIEW, TEST, GREP])
        events, _ = index_events(load_raw_row(row))
        rows = screen_events("t1", "task-1", events)
        assert [r.enrolled for r in rows] == [True, False, False]
        assert rows[1].exclusion_reason == "action:test_execution"
        assert rows[2].exclusion_reason == "prefix:test"
