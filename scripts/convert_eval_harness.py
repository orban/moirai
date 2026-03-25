#!/usr/bin/env python3
"""Convert eval-harness trial results + logs into moirai run format.

Usage:
    python scripts/convert_eval_harness.py /path/to/eval-harness /path/to/output/
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def parse_log_steps(log_path: Path) -> list[dict]:
    """Parse a [tool] ... log file into moirai steps."""
    if not log_path.exists():
        return []

    steps = []
    idx = 0
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("[tool] "):
            action = line[7:]
            step = _parse_tool_action(action, idx)
            steps.append(step)
            idx += 1
        elif line.startswith("[result] "):
            # Final result line, e.g. "[result] 9 turns, $0.2505"
            meta = line[9:]
            steps.append({
                "idx": idx,
                "type": "system",
                "name": "result",
                "status": "ok",
                "output": {"summary": meta},
            })
            idx += 1

    return steps


def _parse_tool_action(action: str, idx: int) -> dict:
    """Parse a tool action string into a step dict."""
    # Common patterns:
    # "Read /path/to/file"
    # "Edit /path/to/file"
    # "Write /path/to/file"
    # "Glob: pattern"
    # "Grep: pattern"
    # "Bash: command"
    # "TodoWrite"
    # "Agent ..."

    # Classify into semantic step types for richer trajectory analysis:
    # - tool/read, tool/search (glob, grep) = exploration
    # - tool/edit, tool/write = modification
    # - tool/bash = execution (often test runs)
    # - llm/subagent = delegation
    # - system/todo = planning
    step_type = "tool"
    name = "unknown"
    status = "ok"
    attrs: dict = {}

    if action.startswith("Read "):
        name = "read"
        attrs["file_path"] = action[5:].strip()
    elif action.startswith("Edit "):
        name = "edit"
        attrs["file_path"] = action[5:].strip()
    elif action.startswith("Write "):
        name = "write"
        attrs["file_path"] = action[6:].strip()
    elif action.startswith("Glob:"):
        name = "search"
        attrs["pattern"] = action[5:].strip()
    elif action.startswith("Grep:"):
        name = "search"
        attrs["pattern"] = action[5:].strip()
    elif action.startswith("Bash:"):
        cmd = action[5:].strip()
        attrs["command"] = cmd[:200]
        # Classify bash commands
        if any(kw in cmd.lower() for kw in ["pytest", "test", "npm test", "cargo test", "go test"]):
            name = "test"
        else:
            name = "bash"
    elif action.startswith("TodoWrite"):
        step_type = "system"
        name = "plan"
    elif action.startswith("Agent"):
        step_type = "llm"
        name = "subagent"
    else:
        name = action[:50].lower().replace(" ", "_")

    return {
        "idx": idx,
        "type": step_type,
        "name": name,
        "status": status,
        "attrs": attrs,
    }


def convert_trial(trial_path: Path, log_dir: Path) -> dict | None:
    """Convert a single trial JSON + matching log into a moirai run."""
    data = json.loads(trial_path.read_text(encoding="utf-8"))

    task_id = data.get("task_id", "unknown")
    condition = data.get("condition", "unknown")
    rep = data.get("rep", 0)

    run_id = f"{task_id}-{condition}-r{rep}"

    # Find matching log file
    log_stem = trial_path.stem  # e.g. "ansible_ansible-83217-flat_llm-r0"
    log_path = log_dir / f"{log_stem}.log"
    steps = parse_log_steps(log_path)

    if not steps:
        # No log = no trace data, skip
        return None

    # Build result
    success = data.get("success", None)
    error = data.get("error")
    error_class = data.get("error_class")

    result: dict = {"success": success}
    if error:
        result["summary"] = error[:200]
    if error_class:
        result["error_type"] = error_class

    # Derive task_family from task_id (repo name)
    parts = task_id.split("-", 1)
    task_family = parts[0] if parts else task_id

    run = {
        "run_id": run_id,
        "task_id": task_id,
        "task_family": task_family,
        "agent": "claude-code",
        "model": "claude-sonnet",  # eval harness default
        "harness": condition,
        "tags": {
            "rep": rep,
            "tool_calls": data.get("tool_calls", 0),
            "wall_clock_seconds": data.get("wall_clock_seconds", 0),
            "input_tokens": data.get("input_tokens", 0),
            "output_tokens": data.get("output_tokens", 0),
            "lines_changed": data.get("lines_changed", 0),
        },
        "steps": steps,
        "result": result,
    }

    return run


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/convert_eval_harness.py <eval-harness-dir> <output-dir>")
        sys.exit(1)

    eval_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    trials_dir = eval_dir / "results" / "trials"
    log_dir = eval_dir / "logs"

    if not trials_dir.exists():
        print(f"error: {trials_dir} not found")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    trial_files = sorted(trials_dir.glob("*.json"))
    converted = 0
    skipped = 0

    for trial_path in trial_files:
        run = convert_trial(trial_path, log_dir)
        if run is None:
            skipped += 1
            continue

        out_path = output_dir / f"{trial_path.stem}.json"
        out_path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
        converted += 1

    print(f"converted {converted} runs, skipped {skipped} (no log data)")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
