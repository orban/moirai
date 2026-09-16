"""Frozen action-family and prefix-restorability rules (policy v1).

The first pilot delivers only non-mutating read/search/navigation actions. A
tool label is not a safety guarantee, so ``execute_bash`` commands are parsed
conservatively: only a single allow-listed read command (optionally preceded by
one ``cd``), with pipes permitted solely between allow-listed commands, and no
redirection, substitution, backgrounding, or control operators. Anything not
positively recognised is excluded with a reason.

Prefix restorability is a replay-coverage rule, not a promise: a prefix whose
actions are all file views/edits, allow-listed reads, and ``cd`` can be
reconstructed by replaying tool history on a fresh workspace. Any other prefix
action (script execution, installs, tests, background processes) makes the
checkpoint depend on process or shell state we cannot prove we restored.
"""
from __future__ import annotations

import json
import re
import shlex

from moirai.intervention.schema import (
    NON_MUTATING_CLASSES,
    REPLAYABLE_PREFIX_CLASSES,
    ActionVerdict,
    EligibilityRecord,
    PrefixRestorability,
    RawEvent,
    ReplayCertificate,
)

POLICY_VERSION = "non_mutating_v1"
POLICY_V2 = "verified_replay_v2"
POLICIES = (POLICY_VERSION, POLICY_V2)

READ_COMMANDS = frozenset({"cat", "head", "tail", "wc", "stat", "file", "nl"})
LIST_COMMANDS = frozenset({"ls", "pwd", "tree"})
SEARCH_COMMANDS = frozenset({"grep", "rg", "ag", "ack", "find", "egrep", "fgrep"})
PIPE_TAIL_COMMANDS = frozenset({"head", "tail", "wc", "sort", "uniq", "grep", "cat", "cut", "tr", "nl"})

_FORBIDDEN_TOKENS = ("$(", "`", ">", "<", ";", "&&", "||", "&", "\n")
_FIND_FORBIDDEN_FLAGS = frozenset({"-exec", "-execdir", "-delete", "-ok", "-okdir", "-fprint", "-fprintf", "-fls"})
_TEST_RE = re.compile(r"(pytest|unittest|nosetests|tox\b|make\s+(test|check)|cargo\s+test|go\s+test|npm\s+test)")
_TEST_SCRIPT_RE = re.compile(r"\bpython3?\s+\S*test\S*\.py\b")
_ENV_RE = re.compile(r"^(export|source|\.|unset|alias|set)\b")


def _decode(args_json: str) -> dict | None:
    try:
        obj = json.loads(args_json)
    except (json.JSONDecodeError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def classify_bash(command: str, is_input: bool = False) -> ActionVerdict:
    cmd = command.strip()
    if is_input:
        return ActionVerdict("input", False, ("stdin_input_to_running_process",))
    if not cmd:
        return ActionVerdict("unknown", False, ("empty_command",))
    if "\n" in cmd:
        return ActionVerdict("exec", False, ("multi_line_command",))

    low = cmd.lower()
    if _TEST_RE.search(low) or _TEST_SCRIPT_RE.search(low):
        return ActionVerdict("test", False, ("test_execution",))
    if cmd.rstrip().endswith("&") and not cmd.rstrip().endswith("&&"):
        return ActionVerdict("background", False, ("background_process",))

    # Split off one leading `cd <dir> &&`.
    cwd_change = False
    m = re.match(r"^cd\s+(\S+)\s*(?:&&|;)\s*(.*)$", cmd)
    if m:
        cwd_change = True
        rest = m.group(2).strip()
        if not rest:
            return ActionVerdict("shell_cwd", False, ("cwd_change_only",))
        cmd = rest
    elif re.match(r"^cd(\s|$)", cmd):
        return ActionVerdict("shell_cwd", False, ("cwd_change_only",))

    if _ENV_RE.match(cmd):
        return ActionVerdict("shell_env", False, ("shell_environment_mutation",))

    for tok in _FORBIDDEN_TOKENS:
        if tok in cmd:
            return ActionVerdict("exec", False, (f"forbidden_token:{tok.strip() or 'newline'}",))

    segments = [s.strip() for s in cmd.split("|")]
    try:
        parsed = [shlex.split(s) for s in segments]
    except ValueError:
        return ActionVerdict("exec", False, ("unparseable_shell",))
    if any(not p for p in parsed):
        return ActionVerdict("exec", False, ("empty_pipe_segment",))

    head_cmd = parsed[0][0]
    head_class: str | None = None
    if head_cmd in SEARCH_COMMANDS:
        head_class = "search"
        if head_cmd == "find" and any(a in _FIND_FORBIDDEN_FLAGS for a in parsed[0][1:]):
            return ActionVerdict("exec", False, ("find_with_side_effect_flag",))
    elif head_cmd in READ_COMMANDS:
        head_class = "read_shell"
    elif head_cmd in LIST_COMMANDS:
        head_class = "list"
    if head_class is None:
        if head_cmd in ("pip", "pip3", "apt", "apt-get", "conda", "uv", "npm", "cargo", "make"):
            return ActionVerdict("setup", False, (f"setup_command:{head_cmd}",))
        return ActionVerdict("exec", False, (f"command_not_allowlisted:{head_cmd}",))

    for seg in parsed[1:]:
        if seg[0] not in PIPE_TAIL_COMMANDS:
            return ActionVerdict("exec", False, (f"pipe_target_not_allowlisted:{seg[0]}",))
        if seg[0] == "sort" and any(a in ("-o", "--output") for a in seg[1:]):
            return ActionVerdict("exec", False, ("sort_with_output_file",))

    reasons = ("cd_prefix_changes_persistent_cwd",) if cwd_change else ()
    # A persistent-shell `cd` is a state mutation even when the read is pure.
    return ActionVerdict(head_class, not cwd_change, reasons)


def classify_action(tool_name: str, arguments_json: str, policy: str = POLICY_VERSION) -> ActionVerdict:
    """Frozen policy: which raw tool calls may be injected in the pilot.

    Under policy v2 a read preceded by one ``cd <dir> &&`` is admitted: the shell's
    working directory is replayed state, and the cwd effect is part of the delivered
    action packet (recorded in the verdict reasons, applied identically to the arm
    that receives it).
    """
    v = _classify_action_v1(tool_name, arguments_json)
    if policy == POLICY_V2 and not v.eligible and v.reasons == ("cd_prefix_changes_persistent_cwd",) \
            and v.action_class in NON_MUTATING_CLASSES:
        return ActionVerdict(v.action_class, True, v.reasons)
    return v


def _classify_action_v1(tool_name: str, arguments_json: str) -> ActionVerdict:
    args = _decode(arguments_json)
    if args is None:
        return ActionVerdict("unknown", False, ("undecodable_arguments",))

    if tool_name == "think":
        return ActionVerdict("reason", False, ("reasoning_not_an_environment_action",))

    if tool_name == "str_replace_editor":
        command = args.get("command")
        if command == "view":
            return ActionVerdict("view", True)
        if command == "create":
            return ActionVerdict("create", False, ("file_creation",))
        if command in ("str_replace", "insert", "undo_edit"):
            return ActionVerdict("edit", False, ("file_mutation",))
        return ActionVerdict("unknown", False, (f"unknown_editor_command:{command}",))

    if tool_name == "execute_bash":
        return classify_bash(str(args.get("command", "")), bool(args.get("is_input", False)))

    if tool_name == "browser":
        return ActionVerdict("browse", False, ("browser_action",))

    if tool_name == "finish":
        return ActionVerdict("unknown", False, ("terminal_action",))

    return ActionVerdict("unknown", False, (f"unknown_tool:{tool_name}",))


def prefix_restorability(events: list[RawEvent], message_idx: int) -> PrefixRestorability:
    """Can messages[:message_idx] be reconstructed by replaying tool history?"""
    blocking: list[str] = []
    first_block: int | None = None
    n = 0
    for ev in events:
        if ev.call.message_idx >= message_idx:
            break
        n += 1
        v = classify_action(ev.call.tool_name, ev.call.arguments_json)
        cls = v.action_class
        # `cd` alone or as a prefix is a deterministic, replayable shell mutation.
        if v.reasons and "cd_prefix_changes_persistent_cwd" in v.reasons:
            cls = "shell_cwd"
        if cls not in REPLAYABLE_PREFIX_CLASSES:
            blocking.append(cls)
            if first_block is None:
                first_block = ev.call.message_idx
        if ev.observation is None:
            blocking.append("missing_observation")
            if first_block is None:
                first_block = ev.call.message_idx
    status = "replayable" if not blocking else "requires_full_replay"
    return PrefixRestorability(status, tuple(sorted(set(blocking))), n, first_block)


def screen_events(run_id: str, task_id: str, events: list[RawEvent]) -> list[EligibilityRecord]:
    """One eligibility row per raw tool call; exclusion reasons are explicit."""
    rows: list[EligibilityRecord] = []
    for ev in events:
        v = classify_action(ev.call.tool_name, ev.call.arguments_json)
        pr = prefix_restorability(events, ev.call.message_idx)
        exclusion: str | None = None
        if not v.eligible:
            exclusion = "action:" + (v.reasons[0] if v.reasons else v.action_class)
        elif pr.status != "replayable":
            exclusion = "prefix:" + ",".join(pr.blocking_classes)
        elif ev.observation is None:
            exclusion = "action:missing_observation"
        rows.append(EligibilityRecord(
            run_id=run_id, task_id=task_id,
            message_idx=ev.call.message_idx, tool_call_id=ev.tool_call_id,
            tool_name=ev.call.tool_name, action_class=v.action_class,
            action_eligible=v.eligible, action_reasons=v.reasons,
            prefix_status=pr.status, prefix_blocking=pr.blocking_classes,
            enrolled=exclusion is None, exclusion_reason=exclusion,
            policy_version=POLICY_VERSION,
        ))
    return rows


# ── Policy v2: verified full replay ───────────────────────────────
#
# The action rule is unchanged (non-mutating reads only). The prefix rule is no
# longer static: a prefix is admitted when a stored replay of it inside the pinned
# task image accepted every step under the fidelity rule. A static screen still
# excludes prefixes that a replay can never certify: stdin fed to a running
# process, commands that reach the network, and tools the replay does not emulate.

_NETWORK_RE = re.compile(
    r"(\b(curl|wget|ssh|scp|rsync|nc|telnet)\b"
    r"|\bpip3?\s+(install|download)\b"
    r"|\bgit\s+(clone|fetch|pull|push|ls-remote|submodule)\b"
    r"|\b(apt|apt-get|yum|dnf|brew)\s+"
    r"|\bconda\s+(install|create|update)\b"
    r"|\b(npm|yarn|pnpm)\s+(install|add|i)\b"
    r"|https?://)"
)
_ADMISSIBLE_TOOLS = frozenset({"execute_bash", "str_replace_editor", "think", "finish"})


def network_command(command: str) -> bool:
    return bool(_NETWORK_RE.search(command or ""))


def prefix_admissibility(events: list[RawEvent], message_idx: int) -> PrefixRestorability:
    """Static screen for policy v2: can a replay in principle certify messages[:message_idx]?"""
    blocking: list[str] = []
    first_block: int | None = None
    n = 0

    def block(reason: str, idx: int) -> None:
        nonlocal first_block
        blocking.append(reason)
        if first_block is None:
            first_block = idx

    for ev in events:
        if ev.call.message_idx >= message_idx:
            break
        n += 1
        args = _decode(ev.call.arguments_json) or {}
        idx = ev.call.message_idx
        if ev.call.tool_name not in _ADMISSIBLE_TOOLS:
            block(f"tool_not_emulated:{ev.call.tool_name}", idx)
        elif ev.call.tool_name == "execute_bash":
            if args.get("is_input"):
                block("stdin_to_process", idx)
            elif network_command(str(args.get("command", ""))):
                block("network_command", idx)
        if ev.observation is None and ev.call.tool_name not in ("think", "finish"):
            block("missing_observation", idx)
    status = "admissible" if not blocking else "inadmissible"
    return PrefixRestorability(status, tuple(sorted(set(blocking))), n, first_block)


def certificate_from_report(report: dict, path: str = "") -> ReplayCertificate:
    """Derive what a replay report certifies. Never widens beyond what was replayed."""
    baseline_ok = bool((report.get("baseline") or {}).get("ok"))
    boundary = int(report["anchor_message_idx"])
    first_rej = report.get("first_rejected_message_idx")
    n_steps = int(report.get("n_steps", len(report.get("steps", []))))
    if not baseline_ok:
        certified = None
    elif first_rej is None:
        certified = boundary            # every replayed step accepted: anchors up to the boundary
    else:
        certified = int(first_rej)      # prefix of an anchor AT the rejected step is still intact
    return ReplayCertificate(
        run_id=report["trajectory_id"], report_path=path,
        instrument_hash=str(report.get("instrument_hash", "")), judged_with=str(report.get("judged_with") or report.get("instrument_hash", "")),
        baseline_ok=baseline_ok, replayed_through=boundary, first_rejected_message_idx=first_rej,
        certified_through=certified, complete=bool(baseline_ok and first_rej is None and report.get("anchor_accepted")),
        n_steps=n_steps,
    )


def load_certificates(replay_dirs: list) -> dict[str, ReplayCertificate]:
    """Best certificate per trajectory across replay report directories (widest certified prefix)."""
    from pathlib import Path

    best: dict[str, ReplayCertificate] = {}
    for d in replay_dirs:
        for p in sorted(Path(d).glob("*.json")):
            try:
                r = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if "anchor_accepted" not in r or "trajectory_id" not in r:
                continue
            c = certificate_from_report(r, str(p))
            cur = best.get(c.run_id)
            if cur is None or (c.certified_through or -1) > (cur.certified_through or -1):
                best[c.run_id] = c
    return best


def screen_events_v2(run_id: str, task_id: str, events: list[RawEvent],
                     cert: ReplayCertificate | None) -> list[EligibilityRecord]:
    """One eligibility row per raw tool call under policy v2."""
    rows: list[EligibilityRecord] = []
    for ev in events:
        v = classify_action(ev.call.tool_name, ev.call.arguments_json, POLICY_V2)
        pr = prefix_admissibility(events, ev.call.message_idx)
        certified = (pr.status == "admissible" and cert is not None and cert.certified_through is not None
                     and ev.call.message_idx <= cert.certified_through)
        exclusion: str | None = None
        if not v.eligible:
            exclusion = "action:" + (v.reasons[0] if v.reasons else v.action_class)
        elif ev.observation is None:
            exclusion = "action:missing_observation"
        elif pr.status != "admissible":
            exclusion = "prefix:" + ",".join(pr.blocking_classes)
        elif cert is None:
            exclusion = "replay:no_certificate"
        elif cert.certified_through is None:
            exclusion = "replay:baseline_failed"
        elif not certified:
            exclusion = "replay:rejected_before_anchor"
        rows.append(EligibilityRecord(
            run_id=run_id, task_id=task_id,
            message_idx=ev.call.message_idx, tool_call_id=ev.tool_call_id,
            tool_name=ev.call.tool_name, action_class=v.action_class,
            action_eligible=v.eligible, action_reasons=v.reasons,
            prefix_status="certified" if certified else pr.status,
            prefix_blocking=pr.blocking_classes,
            enrolled=exclusion is None, exclusion_reason=exclusion,
            policy_version=POLICY_V2,
        ))
    return rows


def anchor_admitted(policy: str, events: list[RawEvent], message_idx: int,
                    cert: ReplayCertificate | None) -> bool:
    """Prefix rule shared by the ledger and the selectors."""
    if policy == POLICY_VERSION:
        return prefix_restorability(events, message_idx).status == "replayable"
    if policy == POLICY_V2:
        if prefix_admissibility(events, message_idx).status != "admissible":
            return False
        return cert is not None and cert.certified_through is not None and message_idx <= cert.certified_through
    raise ValueError(f"unknown policy {policy!r}")


def is_non_mutating(verdict: ActionVerdict) -> bool:
    return verdict.eligible and verdict.action_class in NON_MUTATING_CLASSES
