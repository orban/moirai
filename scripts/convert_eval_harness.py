#!/usr/bin/env python3
"""Convert eval-harness trial results + logs into moirai run format.

Usage:
    python scripts/convert_eval_harness.py /path/to/eval-harness /path/to/output/
    python scripts/convert_eval_harness.py /path/to/eval-harness /path/to/output/ --transcripts ~/.claude/projects/
"""
from __future__ import annotations

import argparse
import json
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


# --- Transcript parsing ---

# Workspace path patterns that appear in step file_paths
_WORKSPACE_PATTERNS = [
    "/eval-workspaces/",
    "/eval-harness/workspaces/",
]


def find_transcript(steps: list[dict], transcripts_dir: Path) -> Path | None:
    """Find the Claude Code transcript JSONL for a run by matching workspace paths.

    Extracts the workspace directory name from step file_path attrs,
    converts to Claude project directory format, and looks for JSONL files.
    """
    # Extract workspace dir from step file_paths
    workspace = None
    claude_dir = None
    for step in steps:
        fp = step.get("attrs", {}).get("file_path", "")
        for pattern in _WORKSPACE_PATTERNS:
            if pattern in fp:
                workspace = fp.split(pattern)[1].split("/")[0]
                prefix = fp.split(pattern)[0] + pattern.rstrip("/")
                claude_dir = prefix.replace("/", "-").replace("_", "-") + "-" + workspace.replace("_", "-")
                break
        if workspace:
            break

    if not workspace or not claude_dir:
        return None

    project_path = transcripts_dir / claude_dir
    if not project_path.exists():
        return None

    # Find JSONL files (may be in root or subagents/)
    jsonl_files = list(project_path.rglob("*.jsonl"))
    if not jsonl_files:
        return None

    if len(jsonl_files) > 1:
        # Pick the one with the most lines
        best = max(jsonl_files, key=lambda p: sum(1 for _ in p.open()))
        print(f"  warning: {len(jsonl_files)} transcripts found for {workspace}, using largest", file=sys.stderr)
        return best

    return jsonl_files[0]


def parse_transcript(jsonl_path: Path) -> list[dict]:
    """Parse a Claude Code JSONL transcript into reasoning+result dicts.

    Each assistant JSONL line has exactly one content block (Claude Code streams them).
    Text/thinking blocks precede tool_use blocks.

    Returns a list of dicts, one per tool call:
      {"reasoning": str, "result": str, "tool_name": str, "tool_use_id": str}
    """
    lines = []
    for raw in jsonl_path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            lines.append(json.loads(raw))
        except json.JSONDecodeError:
            print(f"  warning: skipping malformed JSONL line in {jsonl_path.name}", file=sys.stderr)
            continue

    tool_calls: list[dict] = []
    pending_reasoning: str | None = None
    # Map tool_use_id → index in tool_calls for result matching
    id_to_idx: dict[str, int] = {}

    for obj in lines:
        msg_type = obj.get("type")
        if msg_type not in ("assistant", "user"):
            continue

        content = obj.get("message", {}).get("content", [])
        if not isinstance(content, list):
            continue

        for block in content:
            bt = block.get("type")

            if bt == "thinking":
                pending_reasoning = block.get("thinking", "")[:500]
            elif bt == "text":
                # Only use text as reasoning if we don't already have thinking
                if pending_reasoning is None:
                    pending_reasoning = block.get("text", "")[:500]
            elif bt == "tool_use":
                tool_use_id = block.get("id", "")
                entry = {
                    "reasoning": pending_reasoning,
                    "result": None,
                    "tool_name": block.get("name", ""),
                    "tool_use_id": tool_use_id,
                }
                id_to_idx[tool_use_id] = len(tool_calls)
                tool_calls.append(entry)
                pending_reasoning = None
            elif bt == "tool_result":
                tool_use_id = block.get("tool_use_id", "")
                idx = id_to_idx.get(tool_use_id)
                if idx is not None:
                    result_content = block.get("content", "")
                    if isinstance(result_content, list):
                        # Extract text from content blocks
                        parts = [p.get("text", "") for p in result_content if isinstance(p, dict)]
                        result_content = "\n".join(parts)
                    tool_calls[idx]["result"] = str(result_content)[:300]

    return tool_calls


def _merge_transcript(steps: list[dict], transcript: list[dict]) -> None:
    """Merge transcript reasoning/result into log-parsed steps by sequential position.

    Filters out [result] system steps before matching since they have no
    transcript counterpart. Stops merging if tool names diverge.
    """
    # Tool steps only (filter out system/result steps)
    tool_steps = [s for s in steps if not (s.get("type") == "system" and s.get("name") == "result")]

    for i, tc in enumerate(transcript):
        if i >= len(tool_steps):
            break

        step = tool_steps[i]

        # Sanity check: do tool names roughly match?
        # Log uses lowercase (read, edit, search), transcript uses Claude Code names (Read, Edit, Glob)
        log_name = step.get("name", "")
        transcript_name = tc.get("tool_name", "").lower()
        # Map Claude Code names to our names
        name_map = {
            "read": "read", "edit": "edit", "write": "write",
            "bash": {"bash", "test"}, "glob": "search", "grep": "search",
            "agent": "subagent", "todowrite": "plan",
        }
        expected = name_map.get(transcript_name, transcript_name)
        if isinstance(expected, set):
            match = log_name in expected
        else:
            match = log_name == expected

        if not match:
            print(f"  warning: tool name mismatch at position {i}: log={log_name} transcript={transcript_name}, stopping merge", file=sys.stderr)
            break

        # Merge reasoning and result
        if "output" not in step:
            step["output"] = {}
        if tc["reasoning"]:
            step["output"]["reasoning"] = tc["reasoning"]
        if tc["result"]:
            step["output"]["result"] = tc["result"]


def convert_trial(trial_path: Path, log_dir: Path, transcripts_dir: Path | None = None) -> dict | None:
    """Convert a single trial JSON + matching log into a moirai run."""
    data = json.loads(trial_path.read_text(encoding="utf-8"))

    task_id = data.get("task_id", "unknown")
    condition = data.get("condition", "unknown")
    rep = data.get("rep", 0)

    run_id = f"{task_id}-{condition}-r{rep}"

    # Find matching log file
    log_stem = trial_path.stem
    log_path = log_dir / f"{log_stem}.log"
    steps = parse_log_steps(log_path)

    if not steps:
        return None

    # Merge transcript reasoning if available
    if transcripts_dir:
        jsonl_path = find_transcript(steps, transcripts_dir)
        if jsonl_path:
            transcript = parse_transcript(jsonl_path)
            _merge_transcript(steps, transcript)

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
        "model": "claude-sonnet",
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
    parser = argparse.ArgumentParser(description="Convert eval-harness trial results + logs into moirai run format")
    parser.add_argument("eval_dir", type=Path, help="Path to eval-harness directory")
    parser.add_argument("output_dir", type=Path, help="Output directory for moirai runs")
    parser.add_argument("--transcripts", type=Path, default=None,
                        help="Path to Claude Code projects directory for reasoning extraction (default: ~/.claude/projects/)")
    args = parser.parse_args()

    eval_dir = args.eval_dir
    output_dir = args.output_dir
    transcripts_dir = args.transcripts
    if transcripts_dir is None:
        default_path = Path.home() / ".claude" / "projects"
        if default_path.exists():
            transcripts_dir = default_path

    trials_dir = eval_dir / "results" / "trials"
    log_dir = eval_dir / "logs"

    if not trials_dir.exists():
        print(f"error: {trials_dir} not found")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    trial_files = sorted(trials_dir.glob("*.json"))
    converted = 0
    skipped = 0
    enriched = 0

    for trial_path in trial_files:
        run = convert_trial(trial_path, log_dir, transcripts_dir)
        if run is None:
            skipped += 1
            continue

        # Check if any step got reasoning
        has_reasoning = any(
            s.get("output", {}).get("reasoning")
            for s in run["steps"]
        )
        if has_reasoning:
            enriched += 1

        out_path = output_dir / f"{trial_path.stem}.json"
        out_path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
        converted += 1

    print(f"converted {converted} runs, skipped {skipped} (no log data)")
    if transcripts_dir:
        print(f"enriched {enriched}/{converted} with reasoning from transcripts")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
