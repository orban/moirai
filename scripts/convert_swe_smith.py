#!/usr/bin/env python3
"""Convert SWE-smith trajectories from HuggingFace into moirai run format.

Usage:
    python scripts/convert_swe_smith.py [--count N] [--split tool] <output-dir>

Downloads from SWE-bench/SWE-smith-trajectories on HuggingFace.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def classify_action(content: str) -> tuple[str, str, dict]:
    """Classify an assistant action message into (type, name, attrs).

    SWE-agent tool-calling format uses function calls like:
    {"name": "str_replace_editor", "arguments": {...}}
    or bash commands.
    """
    # Try to parse as tool call JSON
    try:
        # Sometimes the content has a tool_calls structure
        if "str_replace_editor" in content or "view" in content.lower():
            if "command" in content:
                if '"view"' in content:
                    return "tool", "read", {}
                if '"str_replace"' in content:
                    return "tool", "edit", {}
                if '"create"' in content:
                    return "tool", "write", {}
                if '"insert"' in content:
                    return "tool", "write", {}
    except Exception:
        pass

    content_lower = content.lower()

    # Bash / execution commands
    if "bash" in content_lower or "execute" in content_lower:
        if any(kw in content_lower for kw in ["pytest", "python -m test", "npm test", "cargo test", "go test", "make test", "unittest"]):
            return "tool", "test", {}
        if any(kw in content_lower for kw in ["find ", "ls ", "grep ", "rg ", "cat ", "head "]):
            return "tool", "search", {}
        return "tool", "bash", {}

    # File operations
    if any(kw in content_lower for kw in ["str_replace", "edit", "replace"]):
        return "tool", "edit", {}
    if any(kw in content_lower for kw in ["view", "read", "open_file", "cat "]):
        return "tool", "read", {}
    if any(kw in content_lower for kw in ["create", "write"]):
        return "tool", "write", {}
    if any(kw in content_lower for kw in ["find", "search", "grep", "glob"]):
        return "tool", "search", {}

    # Thinking / reasoning (if the agent outputs reasoning before acting)
    if content.strip().startswith("I ") or content.strip().startswith("Let me"):
        return "llm", "reason", {}

    return "tool", "action", {}


def convert_trajectory(row: dict) -> dict | None:
    """Convert a single SWE-smith trajectory row into a moirai run."""
    instance_id = row.get("instance_id", "unknown")
    resolved = row.get("resolved", False)
    model = row.get("model", "unknown")
    traj_id = row.get("traj_id", instance_id)

    # Parse messages
    raw_messages = row.get("messages", "[]")
    if isinstance(raw_messages, str):
        try:
            messages = json.loads(raw_messages)
        except json.JSONDecodeError:
            return None
    else:
        messages = raw_messages

    if not messages:
        return None

    # Convert messages to steps
    steps = []
    idx = 0

    for msg in messages:
        role = msg.get("role", "")
        msg_type = msg.get("message_type", "")
        content = str(msg.get("content", ""))

        if role == "system":
            continue  # skip system prompt

        if role == "assistant" and msg_type == "action":
            step_type, name, attrs = classify_action(content)
            steps.append({
                "idx": idx,
                "type": step_type,
                "name": name,
                "status": "ok",
                "output": {"summary": content[:150]},
                "attrs": attrs,
            })
            idx += 1

        elif role in ("user", "tool") and msg_type == "observation":
            # Environment observations — we track these as system steps
            # but only if they indicate something notable (errors, test results)
            if "error" in content.lower() or "traceback" in content.lower():
                steps.append({
                    "idx": idx,
                    "type": "system",
                    "name": "error_observation",
                    "status": "error",
                    "output": {"summary": content[:150]},
                })
                idx += 1
            elif any(kw in content.lower() for kw in ["passed", "failed", "test"]):
                steps.append({
                    "idx": idx,
                    "type": "judge",
                    "name": "test_result",
                    "status": "ok" if "passed" in content.lower() else "error",
                    "output": {"summary": content[:150]},
                })
                idx += 1
            # Otherwise skip verbose observations to keep trajectories manageable

    if not steps:
        return None

    # Derive task_family from instance_id (repo name)
    # Format: "django-money__django-money.835c1ab8.func_pm_ctrl_shuffle__viqnyl9u"
    parts = instance_id.split(".")
    task_family = parts[0] if parts else instance_id

    run = {
        "run_id": traj_id,
        "task_id": instance_id,
        "task_family": task_family,
        "agent": "swe-agent",
        "model": model,
        "harness": "swe-smith",
        "tags": {
            "dataset": "SWE-smith",
            "resolved": resolved,
        },
        "steps": steps,
        "result": {
            "success": resolved,
        },
    }

    return run


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert SWE-smith trajectories to moirai format")
    parser.add_argument("output_dir", help="Output directory for moirai runs")
    parser.add_argument("--count", type=int, default=100, help="Number of trajectories to convert")
    parser.add_argument("--split", default="tool", help="Dataset split (tool, xml, ticks)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SWE-smith trajectories (split={args.split}, count={args.count})...")

    from datasets import load_dataset
    ds = load_dataset("SWE-bench/SWE-smith-trajectories", split=args.split, streaming=True)

    converted = 0
    skipped = 0

    for i, row in enumerate(ds):
        if converted >= args.count:
            break

        run = convert_trajectory(row)
        if run is None:
            skipped += 1
            continue

        # Use a clean filename
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
