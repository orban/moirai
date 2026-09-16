"""Forced-action event construction and history protocol checks."""
from __future__ import annotations

import hashlib

from moirai.intervention.schema import ActionPayload


def new_tool_call_id(trial_id: str, payload_hash: str) -> str:
    return "chatcmpl-tool-" + hashlib.sha256(f"{trial_id}:{payload_hash}".encode("utf-8")).hexdigest()[:32]


def build_forced_events(action: ActionPayload, tool_call_id: str, observation: str, content: str = "") -> tuple[dict, dict]:
    """Assistant tool-call event and its matching observation, in raw message shape.

    ``content`` is the fixed neutral assistant text policy (empty by default).
    No historical reasoning is copied in.
    """
    assistant = {
        "role": "assistant",
        "content": content,
        "name": None,
        "tool_call_id": None,
        "tool_calls": [{
            "id": tool_call_id,
            "type": "function",
            "function": {"name": action.tool_name, "arguments": action.arguments_json},
        }],
    }
    tool = {
        "role": "tool",
        "content": observation,
        "name": action.tool_name,
        "tool_call_id": tool_call_id,
        "tool_calls": None,
    }
    return assistant, tool


def validate_history(messages: list[dict]) -> list[str]:
    """Protocol errors in a message list; empty when the history is well formed."""
    errors: list[str] = []
    pending: dict[str, int] = {}
    seen: set[str] = set()
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "assistant":
            if pending:
                errors.append(f"message {i}: assistant event while calls {sorted(pending)} await observations")
                pending.clear()
            for tc in m.get("tool_calls") or []:
                tcid = tc.get("id")
                if not tcid:
                    errors.append(f"message {i}: tool call without id")
                    continue
                if tcid in seen:
                    errors.append(f"message {i}: duplicate tool_call_id {tcid}")
                seen.add(tcid)
                pending[tcid] = i
        elif role == "tool":
            tcid = m.get("tool_call_id")
            if tcid not in pending:
                errors.append(f"message {i}: orphan observation for {tcid}")
            else:
                del pending[tcid]
    return errors


def validate_prefix_boundary(messages: list[dict]) -> list[str]:
    errors = validate_history(messages)
    if messages and messages[-1].get("role") == "assistant":
        errors.append("prefix ends with an assistant event; boundary must precede the model's next decision")
    return errors


class AtMostOnce:
    """Guards against delivering the same forced action twice in one trial."""

    def __init__(self) -> None:
        self._done: set[tuple[str, str]] = set()

    def claim(self, trial_id: str, tool_call_id: str) -> None:
        key = (trial_id, tool_call_id)
        if key in self._done:
            raise RuntimeError(f"forced action {tool_call_id} already executed in trial {trial_id}")
        self._done.add(key)
