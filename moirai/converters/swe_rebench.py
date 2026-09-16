"""Convert Nebius SWE-rebench OpenHands trajectories to moirai runs.

This is the single source of truth for the SWE-rebench step classification.
``scripts/convert_swe_rebench.py`` wraps it with dataset download logic, and
``moirai.intervention.provenance`` re-runs it with provenance tracking to map
normalized steps back to raw messages.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

ACTION_TRUNCATION = 500
ARG_VALUE_TRUNCATION = 100
REASONING_TRUNCATION = 2000
RESULT_TRUNCATION = 2000
COMMAND_TRUNCATION = 200


@dataclass(frozen=True)
class StepOrigin:
    """Where a converted step came from in the raw message list."""
    step_idx: int
    message_idx: int
    call_idx: int | None            # None for pure-reasoning assistant messages
    tool_call_id: str | None
    tool_name: str | None
    observation_message_idx: int | None = None
    observation_tool_call_id: str | None = None

    @property
    def observation_misassociated(self) -> bool:
        """True when the attached observation's id differs from the call's id."""
        if self.tool_call_id is None or self.observation_tool_call_id is None:
            return False
        return self.tool_call_id != self.observation_tool_call_id


def format_action(fn_name: str, fn_args: dict) -> str:
    """The truncated action string stored in converted runs."""
    inner = ", ".join(f"{k}={str(v)[:ARG_VALUE_TRUNCATION]}" for k, v in fn_args.items())
    return f"{fn_name}({inner})"[:ACTION_TRUNCATION]


def parse_tool_arguments(fn_args_raw) -> dict:
    """Decode tool-call arguments the same way the converter does."""
    try:
        fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
    except (json.JSONDecodeError, TypeError):
        fn_args = {"raw": str(fn_args_raw)[:500]}
    if not isinstance(fn_args, dict):
        fn_args = {"raw": str(fn_args)[:500]}
    return fn_args


def parse_openhands_trajectory(
    messages: list[dict],
    origins: list[StepOrigin] | None = None,
) -> list[dict]:
    """Convert an OpenHands message list to moirai steps.

    If ``origins`` is given, one StepOrigin per produced step is appended to it.
    Observation attachment deliberately preserves the historical behaviour
    (attach to the most recently appended step) so that converted runs on disk
    remain reproducible; the origin records whether that attachment matched the
    tool_call_id so provenance resolution can fail closed on misassociation.
    """
    steps: list[dict] = []
    idx = 0
    step_origin_index: dict[int, int] = {}  # step position -> origins index

    for message_idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")

        if role in ("system", "user"):
            continue

        if role == "assistant":
            content = msg.get("content", "") or ""
            tool_calls = msg.get("tool_calls") or []

            if not tool_calls:
                if content.strip():
                    steps.append({
                        "idx": idx,
                        "type": "llm",
                        "name": "reason",
                        "status": "ok",
                        "input": {},
                        "output": {"reasoning": content[:REASONING_TRUNCATION]},
                        "attrs": {},
                    })
                    if origins is not None:
                        step_origin_index[len(steps) - 1] = len(origins)
                        origins.append(StepOrigin(
                            step_idx=idx, message_idx=message_idx, call_idx=None,
                            tool_call_id=None, tool_name=None,
                        ))
                    idx += 1
                continue

            for call_idx, tc in enumerate(tool_calls):
                fn = tc.get("function", {})
                fn_name = fn.get("name", "unknown")
                fn_args = parse_tool_arguments(fn.get("arguments", ""))

                step_type, step_name, attrs = _classify_tool(fn_name, fn_args)

                step = {
                    "idx": idx,
                    "type": step_type,
                    "name": step_name,
                    "status": "ok",
                    "input": {},
                    "output": {},
                    "attrs": attrs,
                }

                if content:
                    step["output"]["reasoning"] = content[:REASONING_TRUNCATION]

                if fn_name == "think":
                    step["output"]["reasoning"] = fn_args.get("thought", "")[:REASONING_TRUNCATION]
                else:
                    step["output"]["action"] = format_action(fn_name, fn_args)

                steps.append(step)
                if origins is not None:
                    step_origin_index[len(steps) - 1] = len(origins)
                    origins.append(StepOrigin(
                        step_idx=idx, message_idx=message_idx, call_idx=call_idx,
                        tool_call_id=tc.get("id"), tool_name=fn_name,
                    ))
                idx += 1

                # Only use content for the first tool call
                content = ""

        elif role == "tool":
            tool_content = msg.get("content", "") or ""
            if steps and tool_content:
                _attach_observation(steps[-1], tool_content)
                if origins is not None:
                    oi = step_origin_index.get(len(steps) - 1)
                    if oi is not None:
                        o = origins[oi]
                        origins[oi] = StepOrigin(
                            step_idx=o.step_idx, message_idx=o.message_idx,
                            call_idx=o.call_idx, tool_call_id=o.tool_call_id,
                            tool_name=o.tool_name,
                            observation_message_idx=message_idx,
                            observation_tool_call_id=msg.get("tool_call_id"),
                        )

    return steps


def _attach_observation(step: dict, tool_content: str) -> None:
    step["output"]["result"] = tool_content[:RESULT_TRUNCATION]

    ec_match = re.search(r'exit code (\d+)', tool_content)
    exit_code = int(ec_match.group(1)) if ec_match else None

    if step["name"] == "test":
        if exit_code is not None and exit_code != 0:
            step["status"] = "error"
        elif exit_code == 0:
            step["status"] = "ok"
        elif any(err in tool_content.lower() for err in ("traceback", "assert", "failed")):
            step["status"] = "error"
    elif any(err in tool_content.lower() for err in (
        "error:", "traceback", "exception", "failed",
        "command not found", "no such file",
    )):
        step["status"] = "error"


def _classify_tool(fn_name: str, fn_args: dict) -> tuple[str, str, dict]:
    """Map OpenHands tool calls to moirai step types.

    Uses substring matching for test detection (handles cd && pytest)
    and strips cd prefix before classifying other commands.
    """
    attrs: dict = {}

    if fn_name == "think":
        return "llm", "reason", attrs

    if fn_name == "execute_bash":
        cmd = fn_args.get("command", "")
        attrs["command"] = cmd[:COMMAND_TRUNCATION]

        cmd_lower = cmd.lower().strip()

        test_keywords = (
            "pytest", "python -m pytest", "python3 -m pytest",
            "tox ", "python -m tox", "make test", "make check",
            "unittest", "python -m unittest", "cargo test",
            "go test", "npm test", "nosetests",
        )
        if any(kw in cmd_lower for kw in test_keywords):
            return "tool", "test", attrs
        if re.search(r'\bpython3?\s+\S*test\S*\.py\b', cmd_lower):
            return "tool", "test", attrs

        effective = re.sub(r'^cd\s+\S+\s*(?:&&|;)\s*', '', cmd_lower, count=1)

        if any(effective.startswith(p) for p in ("grep ", "rg ", "ag ", "ack ")):
            return "tool", "search", attrs
        if effective.startswith("find "):
            return "tool", "search", attrs
        if any(effective.startswith(p) for p in ("python ", "python3 ")):
            return "tool", "bash(python)", attrs
        if any(effective.startswith(p) for p in ("cat ", "head ", "tail ", "less ")):
            return "tool", "read", attrs
        if any(effective.startswith(p) for p in ("ls ", "pwd", "tree ")):
            return "tool", "bash(explore)", attrs
        if any(effective.startswith(p) for p in ("pip ", "pip3 ", "apt ", "conda ", "uv ")):
            return "tool", "bash(setup)", attrs
        if effective.startswith("git "):
            return "tool", "bash(git)", attrs
        return "tool", "bash(other)", attrs

    if fn_name == "str_replace_editor":
        command = fn_args.get("command", "")
        path = fn_args.get("path", "")
        attrs["file_path"] = path

        if command == "view":
            return "tool", "read", attrs
        if command == "create":
            return "tool", "write", attrs
        if command in ("str_replace", "insert"):
            return "tool", "edit", attrs
        return "tool", "edit", attrs

    if fn_name == "browser":
        return "tool", "browse", attrs

    return "tool", fn_name, attrs


def task_family_of(instance_id: str) -> str:
    parts = instance_id.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return instance_id


def resolved_to_success(resolved) -> bool | None:
    if resolved == 1 or resolved is True:
        return True
    if resolved == 0 or resolved is False:
        return False
    return None


def convert_messages(
    trajectory_id: str,
    instance_id: str,
    messages: list[dict],
    success: bool | None,
    repo: str = "",
    harness: str = "swe-rebench",
    model: str = "Qwen3-Coder-480B",
    tags: dict | None = None,
) -> dict:
    """Build a moirai run from an OpenHands message list (any corpus using the scaffold)."""
    steps = parse_openhands_trajectory(messages)
    return {
        "run_id": trajectory_id,
        "task_id": instance_id,
        "task_family": task_family_of(instance_id),
        "agent": "openhands",
        "model": model,
        "harness": harness,
        "tags": dict(tags or {}),
        "steps": steps,
        "result": {
            "success": success,
            "score": None,
            "label": "resolved" if success else ("failed" if success is False else None),
        },
    }


def convert_row(row: dict, trajectory_id: str) -> dict:
    """Convert a single HF dataset row to a moirai run."""
    instance_id = row["instance_id"]
    trajectory = row.get("trajectory", [])
    repo = row.get("repo", "")

    steps = parse_openhands_trajectory(trajectory)
    success = resolved_to_success(row.get("resolved"))

    return {
        "run_id": trajectory_id,
        "task_id": instance_id,
        "task_family": task_family_of(instance_id),
        "agent": "openhands",
        "model": "Qwen3-Coder-480B",
        "harness": "swe-rebench",
        "tags": {
            "repo": repo,
            "submission": "nebius-swe-rebench",
            "exit_status": row.get("exit_status", ""),
        },
        "steps": steps,
        "result": {
            "success": success,
            "score": None,
            "label": "resolved" if success else ("failed" if success is False else None),
        },
    }
