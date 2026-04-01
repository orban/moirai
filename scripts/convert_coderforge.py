#!/usr/bin/env python3
"""Convert CoderForge trajectories from HuggingFace into moirai run format.

Usage:
    python scripts/convert_coderforge.py <output-dir> [--count N] [--split SPLIT]

Downloads from togethercomputer/CoderForge-Preview on HuggingFace.
Each trajectory uses OpenHands-style tool calls (execute_bash, str_replace_editor,
think, finish). The `think` tool is extracted as reasoning for the following step.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# All available splits in the dataset
SPLITS = ["SWE_Rebench", "SWE_Smith", "R2E_Gym", "filtered_reward1"]


def _parse_args(args_raw: dict | str) -> dict:
    """Ensure tool call arguments are a dict."""
    if isinstance(args_raw, str):
        try:
            return json.loads(args_raw)
        except json.JSONDecodeError:
            return {"raw": args_raw}
    return args_raw


def _classify_bash_command(command: str) -> str:
    """Return step name for a bash command."""
    cmd_lower = command.lower()
    test_keywords = ["pytest", "python -m test", "npm test", "cargo test",
                     "go test", "make test", "unittest", "python -m pytest"]
    if any(kw in cmd_lower for kw in test_keywords):
        return "test"
    # Only classify as search if the command *starts* with a search tool,
    # not when grep/find appear mid-pipeline (e.g. "ps aux | grep python")
    first_cmd = cmd_lower.split("|")[0].strip().split("&&")[0].strip()
    # Strip leading "cd ... &&" or "cd ... ;"
    if first_cmd.startswith("cd "):
        parts = re.split(r"[;&]", cmd_lower, maxsplit=1)
        first_cmd = parts[-1].strip() if len(parts) > 1 else first_cmd
    search_starters = ["find ", "ls ", "grep ", "rg ", "cat -n", "head ", "tail "]
    if any(first_cmd.startswith(kw) for kw in search_starters):
        return "search"
    return "bash"


def _classify_editor_command(args: dict) -> tuple[str, dict]:
    """Return (step_name, attrs) for a str_replace_editor call."""
    command = args.get("command", "")
    path = args.get("path", "")
    attrs: dict = {}
    if path:
        attrs["file_path"] = path

    if command == "view":
        return "read", attrs
    if command == "create":
        return "write", attrs
    if command == "str_replace":
        return "edit", attrs
    if command == "insert":
        return "write", attrs
    # Fallback
    return "edit", attrs


def _detect_error(step_name: str, content: str) -> str:
    """Return 'error' or 'ok' based on step type and result content.

    For read/search steps, the result is file or grep content — words like
    'error' appearing inside source code aren't real errors. For bash/test
    steps, we look for actual failure signals.
    """
    if step_name in ("read", "search", "write"):
        # Only flag these if the tool itself reported a failure
        content_lower = content.lower()
        if any(kw in content_lower for kw in [
            "command failed", "is not executed", "file not found",
            "no such file", "permission denied",
        ]):
            return "error"
        return "ok"

    # For bash/test steps, check for actual error indicators
    content_lower = content.lower()
    if any(kw in content_lower for kw in [
        "is not executed", "command failed",
    ]):
        return "error"
    # Traceback at the start of output is a real crash
    if "traceback (most recent call last)" in content_lower:
        return "error"
    # Test failures
    if step_name == "test" and any(kw in content_lower for kw in [
        "failed", "error", "failures=",
    ]):
        return "error"
    return "ok"


def convert_trajectory(row: dict) -> dict | None:
    """Convert a single CoderForge trajectory row into a moirai run."""
    trajectory_id = row.get("trajectory_id", "unknown")
    reward = row.get("reward", 0.0)
    finish_reason = row.get("finish_reason", "unknown")

    # Parse task_id and run number from trajectory_id
    # Format: "owner__repo-issue_runN" e.g. "0b01001001__spectree-64_run3"
    match = re.match(r"^(.+?)_run(\d+)$", trajectory_id)
    if match:
        task_id = match.group(1)
        run_number = int(match.group(2))
    else:
        task_id = trajectory_id
        run_number = 1

    # Derive task_family from task_id (owner part)
    parts = task_id.split("__")
    task_family = parts[0] if len(parts) > 1 else task_id

    # Parse messages
    raw_messages = row.get("messages", "[]")
    if isinstance(raw_messages, str):
        try:
            messages = json.loads(raw_messages)
        except json.JSONDecodeError:
            print(f"  warning: failed to parse messages for {trajectory_id}", file=sys.stderr)
            return None
    else:
        messages = raw_messages

    if not messages:
        return None

    # Convert messages to steps.
    # Pattern: assistant message has tool_calls, next message is tool result.
    # The `think` tool contains reasoning that should attach to the *next* step.
    steps: list[dict] = []
    idx = 0
    pending_reasoning: str | None = None

    for msg in messages:
        role = msg.get("role", "")

        if role == "assistant":
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                # Pure text assistant message (rare in this dataset)
                content = str(msg.get("content", ""))
                if content.strip():
                    pending_reasoning = content
                continue

            for call in tool_calls:
                func = call.get("function", {})
                tool_name = func.get("name", "unknown")
                args = _parse_args(func.get("arguments", {}))
                tool_call_id = call.get("id", "")

                # Handle think tool: capture reasoning, don't emit a step
                if tool_name == "think":
                    thought = args.get("thought", "")
                    if thought:
                        # Accumulate reasoning (multiple thinks can precede a tool)
                        if pending_reasoning:
                            pending_reasoning += "\n\n" + thought
                        else:
                            pending_reasoning = thought
                    continue

                # Handle finish tool: emit a final step
                if tool_name == "finish":
                    finish_msg = args.get("message", "")
                    step: dict = {
                        "idx": idx,
                        "type": "system",
                        "name": "finish",
                        "status": "ok",
                        "attrs": {},
                        "output": {},
                    }
                    if pending_reasoning:
                        step["output"]["reasoning"] = pending_reasoning
                        pending_reasoning = None
                    if finish_msg:
                        step["output"]["result"] = finish_msg
                    steps.append(step)
                    idx += 1
                    continue

                # Classify the tool call
                step_type = "tool"
                attrs: dict = {}

                if tool_name == "execute_bash":
                    command = args.get("command", "")
                    name = _classify_bash_command(command)
                    attrs["command"] = command
                elif tool_name == "str_replace_editor":
                    name, attrs = _classify_editor_command(args)
                else:
                    name = tool_name

                step = {
                    "idx": idx,
                    "type": step_type,
                    "name": name,
                    "status": "ok",
                    "attrs": attrs,
                    "output": {},
                }
                if pending_reasoning:
                    step["output"]["reasoning"] = pending_reasoning
                    pending_reasoning = None

                # Stash tool_call_id so we can match the result later
                step["_tool_call_id"] = tool_call_id
                steps.append(step)
                idx += 1

        elif role == "tool":
            # Tool result message — attach to the matching step
            tool_call_id = msg.get("tool_call_id", "")
            content = str(msg.get("content", ""))

            # Find the step with this tool_call_id
            matched = False
            for step in reversed(steps):
                if step.get("_tool_call_id") == tool_call_id:
                    step["output"]["result"] = content
                    step["status"] = _detect_error(step["name"], content)
                    matched = True
                    break

            if not matched and content:
                # Orphan tool result — can happen if think result
                # ("Your thought has been logged.") — just skip it
                pass

    if not steps:
        return None

    # Clean up internal tracking fields
    for step in steps:
        step.pop("_tool_call_id", None)

    run = {
        "run_id": trajectory_id,
        "task_id": task_id,
        "task_family": task_family,
        "agent": "openhands",
        "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "harness": "coderforge",
        "tags": {
            "run_number": run_number,
            "reward": reward,
            "finish_reason": finish_reason,
            "image": row.get("image", ""),
        },
        "steps": steps,
        "result": {
            "success": reward == 1.0,
            "reward": reward,
        },
    }

    return run


def main():
    parser = argparse.ArgumentParser(
        description="Convert CoderForge trajectories to moirai format"
    )
    parser.add_argument("output_dir", type=Path, help="Output directory for moirai runs")
    parser.add_argument("--count", type=int, default=100, help="Number of trajectories to convert")
    parser.add_argument(
        "--split", default="all",
        help=f"Dataset split ({', '.join(SPLITS)}, or 'all'). Default: all",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = SPLITS if args.split == "all" else [args.split]

    print(f"Loading CoderForge trajectories (splits={splits}, count={args.count})...")

    from datasets import load_dataset
    ds = load_dataset("togethercomputer/CoderForge-Preview", "trajectories", streaming=True)

    converted = 0
    skipped = 0

    for split_name in splits:
        if split_name not in ds:
            print(f"  warning: split '{split_name}' not found, skipping", file=sys.stderr)
            continue

        for row in ds[split_name]:
            if converted >= args.count:
                break

            run = convert_trajectory(row)
            if run is None:
                skipped += 1
                continue

            safe_id = run["run_id"][:120].replace("/", "_").replace(" ", "_")
            out_path = output_dir / f"{safe_id}.json"
            out_path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
            converted += 1

            if converted % 25 == 0:
                print(f"  {converted} converted ({split_name})...")

        if converted >= args.count:
            break

    print(f"\nconverted {converted} runs, skipped {skipped}")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
