#!/usr/bin/env python3
"""Convert nebius/SWE-agent-trajectories from HuggingFace into moirai run format.

Usage:
    python scripts/convert_swe_agent.py <output-dir> [--count N] [--model MODEL]

Downloads from nebius/SWE-agent-trajectories on HuggingFace (streaming).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


def _extract_command(text: str) -> str | None:
    """Extract the command from a SWE-agent AI message.

    SWE-agent uses the DISCUSSION + ```command``` format.
    The command is inside the last fenced code block.
    """
    # Find all fenced code blocks
    blocks = re.findall(r"```(?:\w*\n)?(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    return None


def _extract_reasoning(text: str) -> str:
    """Extract the reasoning/discussion text before any code block.

    Returns the full text before the last ``` block, untruncated.
    """
    # Find the position of the last opening ```
    last_fence = text.rfind("```")
    if last_fence > 0:
        return text[:last_fence].strip()
    return text.strip()


def classify_command(command: str) -> tuple[str, str, dict]:
    """Classify a SWE-agent command into (type, name, attrs).

    SWE-agent commands include both custom editor commands and bash commands.
    """
    if not command:
        return "tool", "action", {}

    cmd_lower = command.lower().strip()
    first_word = cmd_lower.split()[0] if cmd_lower.split() else ""

    # SWE-agent built-in editor commands
    if first_word == "open":
        parts = command.strip().split()
        attrs = {}
        if len(parts) >= 2:
            attrs["file_path"] = parts[1]
        return "tool", "read", attrs

    if first_word == "goto" or first_word == "scroll_down" or first_word == "scroll_up":
        return "tool", "read", {}

    if first_word == "edit":
        return "tool", "edit", {}

    if first_word == "create":
        parts = command.strip().split()
        attrs = {}
        if len(parts) >= 2:
            attrs["file_path"] = parts[1]
        return "tool", "write", attrs

    if first_word == "submit":
        return "system", "submit", {}

    if first_word in ("search_dir", "search_file", "find_file"):
        return "tool", "search", {}

    # Bash commands
    if any(kw in cmd_lower for kw in [
        "pytest", "python -m test", "npm test", "cargo test",
        "go test", "make test", "unittest", "python test",
        "python reproduce", "python run_test",
    ]):
        return "tool", "test", {}

    if first_word in ("find", "ls", "grep", "rg", "cat", "head", "tail", "wc"):
        return "tool", "search", {}

    if first_word == "cd":
        return "tool", "bash", {}

    if first_word in ("sed", "awk", "patch"):
        return "tool", "edit", {}

    if first_word in ("touch", "mkdir", "cp", "mv"):
        return "tool", "write", {}

    # Generic bash
    return "tool", "bash", {}


def convert_trajectory(row: dict, run_index: int) -> dict | None:
    """Convert a single SWE-agent trajectory row into a moirai run."""
    instance_id = row.get("instance_id", "unknown")
    model_name = row.get("model_name", "unknown")
    target = row.get("target", False)
    exit_status = row.get("exit_status", "")
    trajectory = row.get("trajectory", [])

    if not trajectory:
        return None

    # Build steps by pairing AI messages (action) with following user messages (observation)
    steps: list[dict] = []
    idx = 0

    msgs = trajectory
    i = 0
    while i < len(msgs):
        msg = msgs[i]
        role = msg.get("role", "")
        text = msg.get("text", "") or ""

        if role == "system":
            i += 1
            continue

        if role == "ai":
            # Extract reasoning and command
            reasoning = _extract_reasoning(text)
            command = _extract_command(text)

            step_type, name, attrs = classify_command(command or "")

            # Look ahead for the user observation (tool result)
            result_text = None
            if i + 1 < len(msgs) and msgs[i + 1].get("role") == "user":
                result_text = msgs[i + 1].get("text", "") or ""
                i += 1  # consume the user message

            output: dict = {}
            if reasoning:
                output["reasoning"] = reasoning
            if command:
                output["tool_input"] = command
            if result_text:
                output["result"] = result_text

            steps.append({
                "idx": idx,
                "type": step_type,
                "name": name,
                "status": "ok",
                "attrs": attrs,
                "output": output,
            })
            idx += 1

        elif role == "user":
            # Standalone user message (the initial issue prompt) -- skip
            # These are the initial problem description or orphan observations
            pass

        i += 1

    if not steps:
        return None

    # Derive task_family from instance_id
    # Format: "owner__repo-number" e.g. "AnalogJ__lexicon-336"
    parts = instance_id.split("__")
    if len(parts) >= 2:
        task_family = parts[0] + "__" + parts[1].rsplit("-", 1)[0]
    else:
        task_family = instance_id

    # Create a stable run_id from instance_id + run_index
    run_id = f"{instance_id}-r{run_index}"

    run = {
        "run_id": run_id,
        "task_id": instance_id,
        "task_family": task_family,
        "agent": "swe-agent",
        "model": model_name,
        "harness": "swe-bench",
        "tags": {
            "dataset": "nebius/SWE-agent-trajectories",
            "exit_status": exit_status,
            "resolved": target,
        },
        "steps": steps,
        "result": {
            "success": target,
        },
    }

    return run


def main():
    parser = argparse.ArgumentParser(
        description="Convert nebius/SWE-agent-trajectories to moirai format"
    )
    parser.add_argument("output_dir", help="Output directory for moirai runs")
    parser.add_argument(
        "--count", type=int, default=100,
        help="Number of trajectories to convert (default: 100)"
    )
    parser.add_argument(
        "--model", default=None,
        help="Filter to a specific model_name (e.g. swe-agent-llama-70b)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading nebius/SWE-agent-trajectories (count={args.count})...")

    from datasets import load_dataset
    ds = load_dataset(
        "nebius/SWE-agent-trajectories", split="train", streaming=True
    )

    converted = 0
    skipped = 0
    # Track how many runs we've seen per instance_id for run indexing
    instance_counts: dict[str, int] = {}

    for row in ds:
        if converted >= args.count:
            break

        if args.model and row.get("model_name") != args.model:
            skipped += 1
            continue

        instance_id = row.get("instance_id", "unknown")
        run_index = instance_counts.get(instance_id, 0)
        instance_counts[instance_id] = run_index + 1

        run = convert_trajectory(row, run_index)
        if run is None:
            skipped += 1
            continue

        safe_id = run["run_id"][:120].replace("/", "_").replace(" ", "_")
        out_path = output_dir / f"{safe_id}.json"
        out_path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
        converted += 1

        if converted % 25 == 0:
            print(f"  {converted} converted...")

    print(f"\nconverted {converted} runs, skipped {skipped}")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
