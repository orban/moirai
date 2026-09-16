"""Raw OpenHands/SWE-rebench trajectory loading, indexing and audit.

Raw rows are the HuggingFace dataset rows saved verbatim as JSON
(``scripts/fetch_swe_rebench_raw.py``). Observations are matched to calls by
``tool_call_id`` here, never by position, so the audit can detect the
converter's positional misassociation risk.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from moirai.intervention.schema import RawAudit, RawEvent, RawToolCall, RawTrajectory


_CODERFORGE_ID = re.compile(r"^(.+?)_run(\d+)$")


def load_raw_row(data: dict) -> RawTrajectory:
    """Accept a Nebius SWE-rebench row or a Together CoderForge row."""
    if "trajectory" not in data and "messages" in data:
        return _load_coderforge_row(data)
    patch = data.get("model_patch")
    patch_hash = hashlib.sha256(patch.encode("utf-8")).hexdigest() if isinstance(patch, str) else None
    resolved = data.get("resolved")
    if resolved in (1, True):
        resolved_b: bool | None = True
    elif resolved in (0, False):
        resolved_b = False
    else:
        resolved_b = None
    return RawTrajectory(
        trajectory_id=str(data["trajectory_id"]),
        instance_id=str(data["instance_id"]),
        repo=str(data.get("repo", "")),
        messages=list(data.get("trajectory") or []),
        tools=list(data.get("tools") or []),
        resolved=resolved_b,
        exit_status=data.get("exit_status"),
        model_patch_hash=patch_hash,
    )


def _load_coderforge_row(data: dict) -> RawTrajectory:
    msgs = data.get("messages")
    if isinstance(msgs, str):
        msgs = json.loads(msgs)
    tools = data.get("tools") or []
    if isinstance(tools, str):
        tools = json.loads(tools)
    tid = str(data["trajectory_id"])
    m = _CODERFORGE_ID.match(tid)
    instance_id = m.group(1) if m else tid
    reward = data.get("reward")
    resolved = None if reward is None else float(reward) == 1.0
    return RawTrajectory(
        trajectory_id=tid, instance_id=instance_id,
        repo=instance_id.split("__")[0] if "__" in instance_id else "",
        messages=list(msgs or []), tools=list(tools), resolved=resolved,
        exit_status=data.get("finish_reason"), model_patch_hash=None,
        image=data.get("image") or None, source="coderforge",
        extra={"run_number": int(m.group(2)) if m else None, "license": data.get("license"), "reward": reward},
    )


def load_raw_file(path: Path) -> RawTrajectory:
    return load_raw_row(json.loads(Path(path).read_text(encoding="utf-8")))


def load_raw_dir(path: Path) -> dict[str, RawTrajectory]:
    """Load every raw row under ``path`` (``*.json`` rows or ``*.jsonl`` files)."""
    path = Path(path)
    out: dict[str, RawTrajectory] = {}
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for p in files:
        if p.suffix == ".json":
            t = load_raw_file(p)
            out[t.trajectory_id] = t
        elif p.suffix == ".jsonl":
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    t = load_raw_row(json.loads(line))
                    out[t.trajectory_id] = t
    return out


def index_events(traj: RawTrajectory) -> tuple[list[RawEvent], RawAudit]:
    """Pair every tool call with its observation by tool_call_id."""
    calls: list[RawToolCall] = []
    n_assistant = 0
    n_multi = 0
    n_undecodable = 0
    for mi, msg in enumerate(traj.messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        n_assistant += 1
        tcs = msg.get("tool_calls") or []
        if len(tcs) > 1:
            n_multi += 1
        for ci, tc in enumerate(tcs):
            fn = tc.get("function") or {}
            args_raw = fn.get("arguments", "")
            args_json = args_raw if isinstance(args_raw, str) else json.dumps(args_raw, sort_keys=True)
            decoded: dict | None
            try:
                obj = json.loads(args_json)
                decoded = obj if isinstance(obj, dict) else None
            except (json.JSONDecodeError, TypeError):
                decoded = None
            if decoded is None:
                n_undecodable += 1
            calls.append(RawToolCall(
                message_idx=mi, call_idx=ci, tool_call_id=str(tc.get("id")),
                tool_name=str(fn.get("name", "unknown")), arguments_json=args_json,
                arguments=decoded, assistant_content=str(msg.get("content") or ""),
            ))

    id_counts = Counter(c.tool_call_id for c in calls)
    n_dup = sum(1 for c in id_counts.values() if c > 1)

    observations: dict[str, tuple[int, str]] = {}
    n_orphan = 0
    for mi, msg in enumerate(traj.messages):
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        tcid = msg.get("tool_call_id")
        if tcid is None or tcid not in id_counts:
            n_orphan += 1
            continue
        observations.setdefault(str(tcid), (mi, str(msg.get("content") or "")))

    events: list[RawEvent] = []
    n_missing = 0
    for c in calls:
        obs = observations.get(c.tool_call_id)
        if obs is None:
            n_missing += 1
            events.append(RawEvent(call=c, observation_message_idx=None, observation=None))
        else:
            events.append(RawEvent(call=c, observation_message_idx=obs[0], observation=obs[1]))

    audit = RawAudit(
        trajectory_id=traj.trajectory_id,
        n_messages=len(traj.messages),
        n_assistant=n_assistant,
        n_tool_calls=len(calls),
        n_multi_call_messages=n_multi,
        n_orphan_observations=n_orphan,
        n_missing_observations=n_missing,
        n_duplicate_ids=n_dup,
        n_undecodable_arguments=n_undecodable,
    )
    return events, audit


def model_visible_prefix(traj: RawTrajectory, message_idx: int) -> list[dict]:
    """Messages the model saw before producing messages[message_idx]."""
    if message_idx < 0 or message_idx > len(traj.messages):
        raise ValueError(f"message_idx {message_idx} out of range for {traj.trajectory_id}")
    return [dict(m) for m in traj.messages[:message_idx]]
