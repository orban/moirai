#!/usr/bin/env python3
"""Convert OpenHands SWE-rebench trajectories from HuggingFace into moirai run format.

Usage:
    python scripts/convert_openhands.py <output-dir> [--count N]

Downloads from nebius/SWE-rebench-openhands-trajectories on HuggingFace.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_tool_args(raw_arguments: str | dict) -> dict:
    """Parse tool call arguments from string or dict."""
    if isinstance(raw_arguments, dict):
        return raw_arguments
    try:
        return json.loads(raw_arguments)
    except (json.JSONDecodeError, TypeError):
        return {}


def classify_tool_call(tool_name: str, args: dict) -> tuple[str, str, dict]:
    """Classify an OpenHands tool call into (type, name, attrs).

    Returns (step_type, step_name, attrs).
    """
    if tool_name == "think":
        return "llm", "reason", {}

    if tool_name == "finish":
        return "system", "finish", {}

    if tool_name == "task_tracker":
        return "system", "plan", {}

    if tool_name == "str_replace_editor":
        command = args.get("command", "")
        path = args.get("path", "")
        attrs: dict = {}
        if path:
            attrs["file_path"] = path

        if command == "view":
            return "tool", "read", attrs
        if command == "str_replace":
            return "tool", "edit", attrs
        if command == "create":
            return "tool", "write", attrs
        if command == "insert":
            return "tool", "write", attrs
        return "tool", "edit", attrs

    if tool_name == "execute_bash":
        cmd = args.get("command", "")
        attrs = {"command": cmd}
        cmd_lower = cmd.lower()
        if any(kw in cmd_lower for kw in [
            "pytest", "python -m test", "npm test", "cargo test",
            "go test", "make test", "unittest", "tox",
        ]):
            return "tool", "test", attrs
        if any(kw in cmd_lower for kw in ["find ", "ls ", "grep ", "rg ", "cat ", "head ", "tail "]):
            return "tool", "search", attrs
        return "tool", "bash", attrs

    # Fallback
    return "tool", tool_name[:50], {}


def convert_trajectory(row: dict) -> dict | None:
    """Convert a single OpenHands trajectory row into a moirai run."""
    instance_id = row.get("instance_id", "unknown")
    trajectory_id = row.get("trajectory_id", "unknown")
    repo = row.get("repo", "unknown")
    resolved = row.get("resolved", 0)
    exit_status = row.get("exit_status", "unknown")

    messages = row.get("trajectory", [])
    if not messages:
        return None

    # Build a map of tool_call_id -> tool result content for matching
    tool_results: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") == "tool" and msg.get("tool_call_id"):
            tool_results[msg["tool_call_id"]] = msg.get("content", "") or ""

    steps: list[dict] = []
    idx = 0

    for msg in messages:
        role = msg.get("role", "")

        if role != "assistant":
            continue

        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            # Pure text assistant message (rare in OpenHands)
            if content.strip():
                steps.append({
                    "idx": idx,
                    "type": "llm",
                    "name": "reason",
                    "status": "ok",
                    "output": {"reasoning": content},
                })
                idx += 1
            continue

        for tc in tool_calls:
            func = tc.get("function", {})
            tool_name = func.get("name", "unknown")
            tc_id = tc.get("id", "")
            args = parse_tool_args(func.get("arguments", "{}"))

            step_type, step_name, attrs = classify_tool_call(tool_name, args)

            output: dict = {}

            # Reasoning: the assistant's content that precedes the tool call
            if content:
                output["reasoning"] = content

            # Tool input: the full arguments (no truncation)
            if tool_name == "think":
                # For think, the thought IS the reasoning
                thought = args.get("thought", "")
                if thought:
                    output["reasoning"] = thought
            elif tool_name == "finish":
                finish_msg = args.get("message", "")
                if finish_msg:
                    output["result"] = finish_msg
            elif tool_name == "str_replace_editor":
                # Store the full tool input
                output["tool_input"] = args
            elif tool_name == "execute_bash":
                output["tool_input"] = args
            elif tool_name == "task_tracker":
                output["tool_input"] = args

            # Tool result: the response from the environment
            result_content = tool_results.get(tc_id, "")
            if result_content and tool_name not in ("think",):
                output["result"] = result_content

            # Determine status from result
            status = "ok"
            if result_content:
                result_lower = result_content.lower()
                if any(kw in result_lower for kw in [
                    "traceback", "error:", "exception:", "command not found",
                    "no such file", "permission denied",
                ]):
                    status = "error"

            step = {
                "idx": idx,
                "type": step_type,
                "name": step_name,
                "status": status,
                "output": output,
            }
            if attrs:
                step["attrs"] = attrs

            steps.append(step)
            idx += 1

    if not steps:
        return None

    # task_family: the repo slug (owner/name -> owner__name)
    task_family = repo.replace("/", "__") if repo else instance_id.split("-")[0]

    run = {
        "run_id": trajectory_id,
        "task_id": instance_id,
        "task_family": task_family,
        "agent": "openhands",
        "model": "unknown",
        "harness": "swe-rebench",
        "tags": {
            "dataset": "SWE-rebench-openhands",
            "repo": repo,
            "exit_status": exit_status,
            "resolved": bool(resolved),
            "gen_tests_correct": row.get("gen_tests_correct", None),
            "pred_passes_gen_tests": row.get("pred_passes_gen_tests", None),
        },
        "steps": steps,
        "result": {
            "success": bool(resolved),
        },
    }

    return run


def main():
    parser = argparse.ArgumentParser(
        description="Convert OpenHands SWE-rebench trajectories to moirai format",
    )
    parser.add_argument("output_dir", help="Output directory for moirai runs")
    parser.add_argument("--count", type=int, default=100, help="Number of trajectories to convert")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading OpenHands trajectories (count={args.count})...")

    from datasets import load_dataset
    ds = load_dataset(
        "nebius/SWE-rebench-openhands-trajectories",
        split="train",
        streaming=True,
    )

    converted = 0
    skipped = 0

    for i, row in enumerate(ds):
        if converted >= args.count:
            break

        run = convert_trajectory(row)
        if run is None:
            skipped += 1
            continue

        safe_id = run["run_id"][:80].replace("/", "_").replace(" ", "_")
        out_path = output_dir / f"{safe_id}.json"
        out_path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
        converted += 1

        if converted % 25 == 0:
            print(f"  {converted} converted...")

    print(f"\nconverted {converted} runs, skipped {skipped}")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
