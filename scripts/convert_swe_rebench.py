#!/usr/bin/env python3
"""Convert Nebius SWE-rebench OpenHands trajectories to moirai format.

Downloads from HuggingFace and converts to moirai JSON runs.
Only downloads mixed-outcome instances (both pass and fail for same task).

Usage:
    python scripts/convert_swe_rebench.py OUTPUT_DIR [--min-pass N] [--min-fail N]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def parse_openhands_trajectory(messages: list[dict]) -> list[dict]:
    """Convert OpenHands message list to moirai steps."""
    steps = []
    idx = 0

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")

        if role == "system" or role == "user":
            continue

        if role == "assistant":
            content = msg.get("content", "") or ""
            tool_calls = msg.get("tool_calls") or []

            if not tool_calls:
                # Pure reasoning message
                if content.strip():
                    steps.append({
                        "idx": idx,
                        "type": "llm",
                        "name": "reason",
                        "status": "ok",
                        "input": {},
                        "output": {"reasoning": content[:2000]},
                        "attrs": {},
                    })
                    idx += 1
                continue

            for tc in tool_calls:
                fn = tc.get("function", {})
                fn_name = fn.get("name", "unknown")
                fn_args_raw = fn.get("arguments", "")

                # Parse arguments
                try:
                    fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
                except (json.JSONDecodeError, TypeError):
                    fn_args = {"raw": str(fn_args_raw)[:500]}

                if not isinstance(fn_args, dict):
                    fn_args = {"raw": str(fn_args)[:500]}

                # Map tool names to moirai types
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
                    step["output"]["reasoning"] = content[:2000]

                if fn_name == "think":
                    step["output"]["reasoning"] = fn_args.get("thought", "")[:2000]
                else:
                    step["output"]["action"] = f"{fn_name}({', '.join(f'{k}={str(v)[:100]}' for k, v in fn_args.items())})"[:500]

                steps.append(step)
                idx += 1

                # Only use content for the first tool call
                content = ""

        elif role == "tool":
            # Attach tool result to the last step
            tool_content = msg.get("content", "") or ""
            if steps and tool_content:
                steps[-1]["output"]["result"] = tool_content[:2000]

                # Extract exit code (most reliable signal)
                ec_match = re.search(r'exit code (\d+)', tool_content)
                exit_code = int(ec_match.group(1)) if ec_match else None

                if steps[-1]["name"] == "test":
                    # For test steps, use exit code as primary signal
                    if exit_code is not None and exit_code != 0:
                        steps[-1]["status"] = "error"
                    elif exit_code == 0:
                        steps[-1]["status"] = "ok"
                    else:
                        # No exit code — fall back to keyword heuristic
                        if any(err in tool_content.lower() for err in [
                            "traceback", "assert", "failed",
                        ]):
                            steps[-1]["status"] = "error"
                else:
                    # For non-test steps, use keyword heuristic
                    if any(err in tool_content.lower() for err in [
                        "error:", "traceback", "exception", "failed",
                        "command not found", "no such file",
                    ]):
                        steps[-1]["status"] = "error"

    return steps


def _classify_tool(fn_name: str, fn_args: dict) -> tuple[str, str, dict]:
    """Map OpenHands tool calls to moirai step types.

    Uses substring matching for test detection (handles cd && pytest)
    and strips cd prefix before classifying other commands.
    """
    attrs = {}

    if fn_name == "think":
        return "llm", "reason", attrs

    if fn_name == "execute_bash":
        cmd = fn_args.get("command", "")
        attrs["command"] = cmd[:200]

        cmd_lower = cmd.lower().strip()

        # Test detection FIRST — substring match handles cd && pytest
        test_keywords = (
            "pytest", "python -m pytest", "python3 -m pytest",
            "tox ", "python -m tox", "make test", "make check",
            "unittest", "python -m unittest", "cargo test",
            "go test", "npm test", "nosetests",
        )
        if any(kw in cmd_lower for kw in test_keywords):
            return "tool", "test", attrs
        # Custom test scripts: python test_foo.py
        if re.search(r'\bpython3?\s+\S*test\S*\.py\b', cmd_lower):
            return "tool", "test", attrs

        # Strip cd prefix for remaining classification
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


def convert_row(row: dict, trajectory_id: str) -> dict:
    """Convert a single HF dataset row to a moirai run."""
    instance_id = row["instance_id"]
    resolved = row.get("resolved")
    trajectory = row.get("trajectory", [])
    repo = row.get("repo", "")

    steps = parse_openhands_trajectory(trajectory)

    # Extract task family from instance_id (e.g., "owner__repo-123" -> "owner__repo")
    task_family = instance_id
    parts = instance_id.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        task_family = parts[0]

    success = None
    if resolved == 1 or resolved is True:
        success = True
    elif resolved == 0 or resolved is False:
        success = False

    return {
        "run_id": trajectory_id,
        "task_id": instance_id,
        "task_family": task_family,
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


def main():
    parser = argparse.ArgumentParser(description="Convert SWE-rebench trajectories")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--min-pass", type=int, default=1, help="Min pass runs per instance")
    parser.add_argument("--min-fail", type=int, default=1, help="Min fail runs per instance")
    parser.add_argument("--limit", type=int, default=0, help="Max instances to process (0=all)")
    args = parser.parse_args()

    from datasets import load_dataset

    print("Loading dataset (streaming)...", file=sys.stderr)
    ds = load_dataset("nebius/SWE-rebench-openhands-trajectories", split="train", streaming=True)

    # First pass: identify mixed-outcome instances
    print("Pass 1: identifying mixed-outcome instances...", file=sys.stderr)
    instance_outcomes: dict[str, dict] = defaultdict(lambda: {"pass": 0, "fail": 0})
    total = 0
    for row in ds:
        total += 1
        iid = row["instance_id"]
        resolved = row.get("resolved")
        if resolved == 1 or resolved is True:
            instance_outcomes[iid]["pass"] += 1
        elif resolved == 0 or resolved is False:
            instance_outcomes[iid]["fail"] += 1
        if total % 10000 == 0:
            print(f"  scanned {total}...", file=sys.stderr)

    mixed = {
        iid for iid, v in instance_outcomes.items()
        if v["pass"] >= args.min_pass and v["fail"] >= args.min_fail
    }
    print(f"  {len(mixed)} instances with >= {args.min_pass} pass + {args.min_fail} fail", file=sys.stderr)

    if args.limit > 0:
        mixed = set(sorted(mixed)[:args.limit])
        print(f"  limited to {len(mixed)} instances", file=sys.stderr)

    # Second pass: convert mixed-outcome runs
    print("Pass 2: converting trajectories...", file=sys.stderr)
    ds = load_dataset("nebius/SWE-rebench-openhands-trajectories", split="train", streaming=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    converted = 0
    skipped = 0

    for row in ds:
        iid = row["instance_id"]
        if iid not in mixed:
            skipped += 1
            continue

        traj_id = row.get("trajectory_id", f"{iid}_{converted}")
        run = convert_row(row, traj_id)

        if not run["steps"]:
            skipped += 1
            continue

        # Write one file per run
        safe_id = traj_id.replace("/", "_").replace(" ", "_")[:200]
        out_path = args.output_dir / f"{safe_id}.json"
        out_path.write_text(json.dumps(run, indent=2) + "\n")
        converted += 1

        if converted % 1000 == 0:
            print(f"  converted {converted}...", file=sys.stderr)

    print(f"\nDone: {converted} runs converted, {skipped} skipped", file=sys.stderr)
    print(f"Output: {args.output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
