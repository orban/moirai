"""Docker-free tests for replay helpers and the CoderForge raw loader."""
from __future__ import annotations

import json

from moirai.intervention.raw import index_events, load_raw_row
from moirai.intervention.replay import _cat_n, choose_post_test_anchor, normalize_observation, similarity
from tests.test_intervention_provenance import EDIT, GREP, TEST, VIEW, make_row


class TestSimilarity:
    def test_noise_is_normalised(self):
        a = "ran in 0.12s at 2026-09-05 10:00:00 obj 0xdeadbeef00\n[The command completed with exit code 0.]"
        b = "ran in 3.40s at 2026-09-06 11:11:11 obj 0x0123456789\n[The command completed with exit code 0.]"
        assert similarity(a, b) == 1.0
        assert normalize_observation("x  \n\n\n\ny") == "x\n\ny"

    def test_divergent_output_scores_low(self):
        assert similarity("1 passed", "3 failed, 1 error\nTraceback ...") < 0.8

    def test_cat_n_format(self):
        assert _cat_n("a\nb\n") == "     1\ta\n     2\tb\n     3\t"
        assert _cat_n("a\nb", start=10).startswith("    10\ta")


class TestAnchorChoice:
    def test_first_eligible_after_test(self):
        row = make_row("t1", "task-1", [VIEW, EDIT, TEST, GREP, VIEW])
        events, _ = index_events(load_raw_row(row))
        assert choose_post_test_anchor(events) == 8      # GREP, right after TEST at message 6

    def test_none_without_test(self):
        row = make_row("t1", "task-1", [VIEW, EDIT, GREP])
        events, _ = index_events(load_raw_row(row))
        assert choose_post_test_anchor(events) is None


class TestCoderForgeLoader:
    def test_row_shape(self):
        msgs = make_row("x", "x", [VIEW])["trajectory"]
        row = {"trajectory_id": "Owner__repo-12_run3", "finish_reason": "tool_calls",
               "image": "qingyangwu/sweb.eval.x86_64.owner_1776_repo-12", "messages": json.dumps(msgs),
               "reward": 1.0, "tools": json.dumps([{"function": {"name": "execute_bash"}}]), "license": "MIT"}
        t = load_raw_row(row)
        assert t.instance_id == "Owner__repo-12" and t.resolved is True and t.source == "coderforge"
        assert t.image.startswith("qingyangwu/") and t.extra["run_number"] == 3
        assert len(t.messages) == len(msgs) and t.tools[0]["function"]["name"] == "execute_bash"
        events, audit = index_events(t)
        assert audit.n_tool_calls == 1 and events[0].observation == "1\tprint()"


from moirai.intervention.replay import (
    choose_deep_anchor,
    group_timeout_steps,
    hidden_footer,
    strip_command_echo,
    render_terminal,
    openhands_bash_rejection,
    numbered_lines,
    truncated_parts_match,
    judge_step,
    line_diff_empty,
    parse_exit_code,
    pytest_outcome,
    recorded_flags,
    strip_ansi,
)


class TestObservationParsing:
    def test_exit_code_both_formats(self):
        assert parse_exit_code("out\n[Command finished with exit code 1]") == 1
        assert parse_exit_code("out\n[The command completed with exit code 130. CTRL+C was sent.]") == 130
        assert parse_exit_code("no footer") is None

    def test_flags(self):
        f = recorded_flags("[The command has no new output after 10 seconds. You may wait longer...]")
        assert f["timeout_notice"] and not f["truncated"]
        assert recorded_flags('[Your command "ls" is NOT executed. The previous command is still running - x]')["not_executed"]
        assert recorded_flags("abc\n[... Observation truncated due to length ...]\ndef")["truncated"]

    def test_pytest_outcome(self):
        obs = "tests/test_a.py::test_x PASSED\ntests/test_a.py::test_y FAILED\n===== 1 failed, 1 passed, 2 warnings in 0.31s =====\n[Command finished with exit code 1]"
        o = pytest_outcome(obs)
        assert o["counts"] == {"failed": 1, "passed": 1}
        assert pytest_outcome("no summary here") is None
        assert pytest_outcome("\x1b[32m===== 3 passed in 1.2s =====\x1b[0m")["counts"] == {"passed": 3}

    def test_pytest_sugar_outcome(self):
        obs = ("Test session starts (platform: linux, Python 3.9.21, pytest 8.3.4, pytest-sugar 1.0.0)\n"
               " tests/test_Alert.py::TestAlert.test_init ✓                 14% █▍\n"
               " tests/test_Alert.py::TestAlert.test_update_progress ⨯     100% ██████████\n"
               "\nResults (0.55s):\n       1 passed\n       1 failed\n         - tests/test_Alert.py:59 TestAlert.test_update_progress\n"
               "[The command completed with exit code 1.]")
        o = pytest_outcome(obs)
        assert o["counts"] == {"passed": 1, "failed": 1}
        assert o["verdicts"]["tests/test_Alert.py::TestAlert.test_init"] == "PASSED"
        assert o["verdicts"]["tests/test_Alert.py::TestAlert.test_update_progress"] == "FAILED"
        verbose = "tests/test_a.py::test_x PASSED   [ 50%]\ntests/test_a.py::test_y FAILED   [100%]\n==== 1 failed, 1 passed in 0.2s ===="
        assert pytest_outcome(verbose)["verdicts"] == {"tests/test_a.py::test_x": "PASSED", "tests/test_a.py::test_y": "FAILED"}

    def test_order_insensitive_and_line_diff(self):
        a, b = "b.py\na.py\nc.py", "a.py\nb.py\nc.py"
        assert similarity(a, b) < 1.0 and similarity(a, b, order_insensitive=True) == 1.0
        assert line_diff_empty(a, b, order_insensitive=True) and not line_diff_empty(a, b)
        assert strip_ansi("\x1b[1mbold\x1b[0m") == "bold"


class TestJudgeStep:
    def test_exit_code_disagreement_rejects(self):
        ok, why = judge_step("read_shell", "execute_bash", {"command": "cat x"}, "x\n[Command finished with exit code 0]", "x", "match", 1, 0.95)
        assert not ok and why.startswith("exit_code")

    def test_pytest_counts_govern(self):
        rec = "==== 2 passed in 0.1s ====\n[Command finished with exit code 0]"
        rep = "==== 2 passed in 9.9s ===="
        ok, why = judge_step("test", "execute_bash", {"command": "pytest"}, rec, rep, "near", 0, 0.95)
        assert ok and why == "pytest_outcome_equal"
        ok, why = judge_step("test", "execute_bash", {"command": "pytest"}, rec, "==== 1 failed, 1 passed in 0.1s ====", "near", 0, 0.95)
        assert not ok and why.startswith("pytest_counts")

    def test_collect_only_falls_back_to_content(self):
        rec = "collected 5 items\n<Module a.py>\n[Command finished with exit code 0]"
        assert judge_step("test", "execute_bash", {"command": "pytest --collect-only"}, rec, "collected 5 items\n<Module a.py>", "match", 0, 0.95)[0]
        assert judge_step("test", "execute_bash", {"command": "pytest --collect-only"}, rec, "collected 6 items\n<Module a.py>", "near", 0, 0.95) == (False, "content_differs")
        assert judge_step("test", "execute_bash", {"command": "pytest"}, "==== 1 passed in 0.1s ====", "no summary", "mismatch", 0, 0.95)[1] == "pytest_summary_missing_on_one_side"
        # A test run whose traceback text differs but whose pass/fail set matches is accepted.
        assert judge_step("test", "execute_bash", {"command": "pytest"}, "long traceback A\n==== 1 failed, 6 passed in 0.1s ====\n[The command completed with exit code 1.]", "long traceback B at 0xdeadbeef\n==== 1 failed, 6 passed in 3.1s ====", "mismatch", 1, 0.95) == (True, "pytest_outcome_equal")

    def test_hung_process_and_is_input_reject(self):
        assert not judge_step("exec", "execute_bash", {"command": "python x.py"}, "[The command has no new output after 10 seconds.]", "done", "mismatch", 0, 0.95)[0]
        assert judge_step("input", "execute_bash", {"command": "", "is_input": True}, "", "", "skipped", None, 0.95) == (False, "undeliverable:is_input")

    def test_near_with_real_content_difference_rejects(self):
        rec = "     1\timport re\n     2\tx = 1\n"
        rep = "     1\timport re\n     2\tx = 2\n"
        ok, why = judge_step("view", "str_replace_editor", {"command": "view", "path": "/a"}, rec, rep, "near", None, 0.95)
        assert not ok and why == "content_differs"

    def test_truncated_recording_compares_prefix(self):
        rec = "line1\nline2\n[... Observation truncated due to length ...]"
        rep = "line1\nline2\nline3\nline4"
        assert judge_step("read_shell", "execute_bash", {"command": "cat f"}, rec, rep, "near", 0, 0.95)[0]

    def test_skipped_think_accepted(self):
        assert judge_step("reason", "think", {}, "Your thought has been logged.", "", "skipped", None, 0.95) == (True, "no_environment_effect")


class TestDeepAnchor:
    def test_cap_binds(self):
        row = make_row("t1", "task-1", [VIEW] * 50 + [GREP])
        events, _ = index_events(load_raw_row(row))
        assert choose_deep_anchor(events, max_prefix_calls=40) == events[40].call.message_idx
        assert choose_deep_anchor(events, max_prefix_calls=1000) == events[-1].call.message_idx


class TestTimeoutGroups:
    def _events(self, actions):
        row = make_row("t1", "task-1", actions)
        return index_events(load_raw_row(row))[0]

    def test_group_detected_and_merged_exit(self):
        notice = "partial\n[The command has no new output after 10 seconds. You may wait longer to see additional output by sending empty command '', send other commands to interact with the current process, or send keys to interrupt/kill the command.]"
        acts = [
            ("execute_bash", {"command": "python -m pytest -q"}, notice),
            ("execute_bash", {"command": "ls", "is_input": False}, '[Your command "ls" is NOT executed. The previous command is still running - You CANNOT send new commands until the previous command is completed. By setting `is_input` to `true`, you can interact with the current process.]'),
            ("execute_bash", {"command": "", "is_input": True}, "[Below is the output of the previous command.]\n2 passed in 3.0s\n[The command completed with exit code 0.]"),
            ("execute_bash", {"command": "echo done"}, "done\n[The command completed with exit code 0.]"),
        ]
        ev = self._events(acts)
        groups = group_timeout_steps(ev)
        assert groups == {0: [1, 2]}
        merged = "\n".join(ev[k].observation for k in [0, 1, 2])
        assert parse_exit_code(merged) == 0
        # merged text carries the pytest summary; equal counts accept the test step regardless of similarity
        assert judge_step("test", "execute_bash", {"command": "python -m pytest -q"}, merged, "partial\n2 passed in 9.9s", "mismatch", 0, 0.95)[1] == "pytest_outcome_equal"
        ok, why = judge_step("read_shell", "execute_bash", {"command": "cat x"}, merged, "partial\n2 passed in 9.9s", "near", 0, 0.95)
        assert ok, why

    def test_interrupted_group_rejected(self):
        notice = "[The command has no new output after 10 seconds. You may wait longer to see additional output by sending empty command '', send other commands to interact with the current process, or send keys to interrupt/kill the command.]"
        acts = [
            ("execute_bash", {"command": "python serve.py"}, notice),
            ("execute_bash", {"command": "C-c", "is_input": True}, "[The command completed with exit code 130. CTRL+C was sent.]"),
            ("execute_bash", {"command": "echo done"}, "done\n[The command completed with exit code 0.]"),
        ]
        ev = self._events(acts)
        assert group_timeout_steps(ev) == {0: [1]}
        merged = "\n".join(ev[k].observation for k in [0, 1])
        assert judge_step("exec", "execute_bash", {"command": "python serve.py"}, merged, "server output", "mismatch", 0, 0.95) == (False, "recorded_process_interrupted")
        assert judge_step("exec", "execute_bash", {"command": "python serve.py"}, merged, "", "near", 130, 0.95) == (False, "recorded_process_interrupted")

    def test_hidden_footer(self):
        assert hidden_footer("/testbed", 0) == ""
        assert hidden_footer("/testbed", 10) == "\n10 hidden files/directories in this directory are excluded. You can use 'ls -la /testbed' to see them.\n"


class TestRoundThreeFixes:
    def test_strip_multiline_echo(self):
        cmd = 'cd /testbed && python -c "\nimport sys\nprint(1)\n"'
        rec = cmd + "\n1\n[The command completed with exit code 0.]"
        assert strip_command_echo(rec, cmd) == "1\n[The command completed with exit code 0.]"
        assert strip_command_echo("other\n1", cmd) == "other\n1"

    def test_response_clipped_compares_prefix(self):
        rec = "     1\tline\n     2\tmore<response clipped><NOTE>Due to the max output limit, only part of this file has been shown to you."
        rep = "     1\tline\n     2\tmore stuff here\n     3\tend"
        assert recorded_flags(rec)["truncated"]
        ok, why = judge_step("view", "str_replace_editor", {"command": "view", "path": "/f"}, rec, rep, "near", None, 0.95)
        assert ok and why == "equal_outside_recorded_truncation"

    def test_interrupt_after_pytest_finished_accepted(self):
        rec = ("==== 1 failed, 6 passed in 0.5s ====\n[The command has no new output after 10 seconds. You may wait longer to see additional output by sending empty command '', send other commands to interact with the current process, or send keys to interrupt/kill the command.]\n"
               "[The command completed with exit code 130. CTRL+C was sent.]")
        rep = "==== 1 failed, 6 passed in 0.9s ===="
        assert judge_step("test", "execute_bash", {"command": "pytest"}, rec, rep, "near", 1, 0.95) == (True, "pytest_outcome_equal_before_interrupt")
        assert judge_step("test", "execute_bash", {"command": "pytest"}, rec, "==== 7 passed in 0.9s ====", "mismatch", 0, 0.95) == (False, "recorded_process_interrupted")


class TestFailingSetRule:
    def test_passed_line_parse_glitch_does_not_reject(self):
        rec = "tests/a.py::test_x PASSED\ntests/a.py::test_y[a b] PASSED\ntests/a.py::test_z FAILED\n==== 1 failed, 2 passed in 0.1s ====\n[The command completed with exit code 1.]"
        rep = "tests/a.py::test_x PASSED\ntests/a.py::test_y[a\nb] PASSED\ntests/a.py::test_z FAILED\n==== 1 failed, 2 passed in 0.3s ===="
        assert judge_step("test", "execute_bash", {"command": "pytest -v"}, rec, rep, "near", 1, 0.95) == (True, "pytest_outcome_equal")
        rep2 = "tests/a.py::test_x FAILED\ntests/a.py::test_y[a b] PASSED\ntests/a.py::test_z PASSED\n==== 1 failed, 2 passed in 0.3s ===="
        assert judge_step("test", "execute_bash", {"command": "pytest -v"}, rec, rep2, "near", 1, 0.95) == (False, "pytest_failing_set_differs")


class TestRoundFourFixes:
    def test_render_terminal_resolves_carriage_returns(self):
        assert render_terminal("collecting ... \rcollected 7 items\nnext") == "collected 7 items\nnext"
        assert render_terminal("  0%|          | 0/100\r100%|xxxxxxxxxx| 100/100") == "100%|xxxxxxxxxx| 100/100"
        assert render_terminal("abcdef\rXY") == "XYcdef"

    def test_echo_strip_ignores_blank_lines(self):
        cmd = 'cd /t && python -c "\nimport sys\n\nprint(1)\n"'
        rec = 'cd /t && python -c "\nimport sys\nprint(1)\n"\n1\n[The command completed with exit code 0.]'
        assert strip_command_echo(rec, cmd) == "1\n[The command completed with exit code 0.]"

    def test_openhands_rejects_exec_pipe(self):
        assert openhands_bash_rejection('find . -name "*.py" -exec grep -l x {} \\; | head -5') == "bash: syntax error near unexpected token `|'"
        assert openhands_bash_rejection("find . -name '*.py' | head") is None

    def test_edit_snippet_consistency_rule(self):
        head = "The file /a.py has been edited. Here's the result of running `cat -n` on a snippet of /a.py:\n"
        rec = head + "    78\tx = 1\n    79\ty = 2\n    80\t\nReview the changes"
        rep = head + "    77\timport os\n    78\tx = 1\n    79\ty = 2\nReview the changes"
        assert numbered_lines(rec) == {78: "x = 1", 79: "y = 2", 80: ""}
        args = {"command": "str_replace", "path": "/a.py"}
        assert judge_step("edit", "str_replace_editor", args, rec, rep, "near", None, 0.95) == (True, "edit_snippet_consistent")
        rep2 = rep.replace("y = 2", "y = 3")
        assert judge_step("edit", "str_replace_editor", args, rec, rep2, "near", None, 0.95) == (False, "edit_snippet_line_differs:79")

    def test_noise_dates_sizes_hashes(self):
        a = "drwxr-xr-x 1 root root 1714 Sep 18 07:36 ..\n7a33380 (HEAD -> master) Initial commit"
        b = "drwxr-xr-x 1 root root  194 Sep  6 02:22 ..\n9ac84b9 (HEAD -> master) Initial commit"
        assert normalize_observation(a) == normalize_observation(b)

    def test_tqdm_bar_width_is_noise(self):
        a = "1. Testing:\nProgress:   0%|" + " " * 150 + "\n   Success: n = 50.0"
        b = "1. Testing:\nProgress:   0%|" + " " * 60 + "0/100\n   Success: n = 50.0"
        assert normalize_observation(a) == normalize_observation(b)

    def test_edit_snippet_trailing_empty_artifact_ignored(self):
        head = "The file /a.py has been edited. Here's the result of running `cat -n` on a snippet of /a.py:\n"
        rec = head + "   146\ta\n   147\tb\n   148\t\nReview the changes"
        rep = head + "   146\ta\n   147\tb\n   148\tThe color can also be changed\nReview the changes"
        args = {"command": "str_replace", "path": "/a.py"}
        assert judge_step("edit", "str_replace_editor", args, rec, rep, "near", None, 0.95) == (True, "edit_snippet_consistent")

    def test_truncated_recording_compares_head_and_tail(self):
        marker = "[... Observation truncated due to length ...]"
        full = "\n".join(f"line {i}" for i in range(40))
        rec = full[:60] + "\n" + marker + "\n" + full[-70:]
        assert truncated_parts_match(rec, full) is True
        assert truncated_parts_match(rec, full.replace("line 38", "line X")) is False
        assert truncated_parts_match(rec, full.replace("line 2\n", "line Y\n")) is False
        ok, reason = judge_step("search", "execute_bash", {"command": "grep -r x ."}, rec, full, "match", 0, 0.95)
        assert (ok, reason) == (True, "equal_outside_recorded_truncation")

    def test_ls_column_padding_is_noise(self):
        a = "-rw-r--r-- 1 root root   41 Sep 18 07:36 nomenclature.yaml"
        b = "-rw-r--r-- 1 root root  41 Sep  6 02:22 nomenclature.yaml"
        assert normalize_observation(a) == normalize_observation(b)

    def test_tqdm_percentage_and_tabs_are_render_noise(self):
        assert normalize_observation("Progress:  50%|#####     | 1/2") == normalize_observation("Progress:   0%|          | 0/2")
        assert normalize_observation("\tmodified:   a.py\n\tnew.py") == normalize_observation("        modified:   a.py\n        new.py")
        # editor snippets keep their tab-separated numbering for the line-consistency rule
        assert numbered_lines("    78\tx = 1\n    79\ty") == {78: "x = 1", 79: "y"}

    def test_quiet_pytest_summary_and_noop_replace(self):
        from moirai.intervention.replay import pytest_outcome
        assert pytest_outcome("................ [100%]\n16 passed in 1.40s\n[The command completed with exit code 0.]")["counts"] == {"passed": 16}
        assert pytest_outcome("....F. [100%]\n5 passed, 1 failed in 0.3s")["counts"] == {"passed": 5, "failed": 1}
        assert pytest_outcome("=========== 3 passed in 0.1s ===========")["counts"] == {"passed": 3}
        rec = "................" + " " * 200 + "[100%]\n16 passed in 1.40s\n[The command completed with exit code 0.]"
        rep = "................" + " " * 57 + "[100%]\n16 passed in 1.20s\n"
        assert judge_step("test", "execute_bash", {"command": "pytest -q"}, rec, rep, "mismatch", 0, 0.95) == (True, "pytest_outcome_equal")

    def test_log_timestamp_millis_are_noise(self):
        a = "2025-09-18 07:36:12,515 - serpentTools - DEBUG - msg"
        b = "2026-09-06 03:10:44,877 - serpentTools - DEBUG - msg"
        assert normalize_observation(a) == normalize_observation(b)
