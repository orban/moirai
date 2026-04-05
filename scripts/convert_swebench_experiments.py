#!/usr/bin/env python3
"""Convert SWE-bench/experiments trajectories into moirai run format.

Expects a local clone/checkout of https://github.com/SWE-bench/experiments
with logs and trajectories downloaded via `python -m analysis.download_logs`.

Usage:
    python scripts/convert_swebench_experiments.py /path/to/experiments /path/to/output/
    python scripts/convert_swebench_experiments.py /path/to/experiments /path/to/output/ --split verified
    python scripts/convert_swebench_experiments.py /path/to/experiments /path/to/output/ --agents "sweagent,OpenHands"
    python scripts/convert_swebench_experiments.py /path/to/experiments /path/to/output/ --submissions 20240620_sweagent_claude3.5sonnet
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Metadata parsing
# ---------------------------------------------------------------------------

def parse_metadata(meta_path: Path) -> dict:
    """Parse metadata.yaml for a submission, returning normalized fields."""
    if not meta_path.exists():
        return {}
    try:
        data = yaml.safe_load(meta_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}

    info = data.get("info", {}) or {}
    tags = data.get("tags", {}) or {}

    models = tags.get("model", [])
    if isinstance(models, str):
        models = [models]

    return {
        "agent_name": info.get("name", ""),
        "org": tags.get("org", ""),
        "models": models,
        "open_source_system": tags.get("os_system", False),
        "open_source_model": tags.get("os_model", False),
        "checked": tags.get("checked", False),
        "site": info.get("site", ""),
        "report": info.get("report", ""),
        "attempts": (tags.get("system", {}) or {}).get("attempts", "1"),
    }


def parse_results(results_path: Path) -> dict[str, str]:
    """Parse results/results.json, returning instance_id -> status mapping.

    Status is one of: "resolved", "no_generation", "no_logs", "unresolved".
    """
    if not results_path.exists():
        return {}
    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

    mapping: dict[str, str] = {}
    for status_key in ("resolved", "no_generation", "no_logs"):
        for instance_id in data.get(status_key, []):
            mapping[instance_id] = status_key
    return mapping


def collect_all_instance_ids(submission_dir: Path) -> set[str]:
    """Collect all instance IDs from logs/ and trajs/ directories."""
    ids: set[str] = set()

    logs_dir = submission_dir / "logs"
    if logs_dir.is_dir():
        for child in logs_dir.iterdir():
            if child.is_dir():
                ids.add(child.name)

    trajs_dir = submission_dir / "trajs"
    if trajs_dir.is_dir():
        for child in trajs_dir.iterdir():
            if child.is_file():
                ids.add(child.stem)

    return ids


# ---------------------------------------------------------------------------
# Report parsing (from logs/<instance_id>/report.json)
# ---------------------------------------------------------------------------

def parse_report(report_path: Path, instance_id: str) -> dict:
    """Parse an instance-level report.json for test results."""
    if not report_path.exists():
        return {}
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

    # The report is keyed by instance_id
    inst = data.get(instance_id, data)
    if not isinstance(inst, dict):
        return {}

    result: dict = {
        "resolved": inst.get("resolved", False),
        "patch_exists": inst.get("patch_exists", False),
        "patch_applied": inst.get("patch_successfully_applied", False),
    }

    tests_status = inst.get("tests_status", {})
    if tests_status:
        f2p = tests_status.get("FAIL_TO_PASS", {})
        p2p = tests_status.get("PASS_TO_PASS", {})
        result["fail_to_pass_success"] = len(f2p.get("success", []))
        result["fail_to_pass_failure"] = len(f2p.get("failure", []))
        result["pass_to_pass_success"] = len(p2p.get("success", []))
        result["pass_to_pass_failure"] = len(p2p.get("failure", []))

    return result


# ---------------------------------------------------------------------------
# Step classification helpers
# ---------------------------------------------------------------------------

def _classify_bash_command(cmd: str) -> str:
    """Return a step name for a bash command."""
    cmd_lower = cmd.lower()
    test_keywords = ["pytest", "python -m test", "npm test", "cargo test",
                     "go test", "make test", "unittest", "tox"]
    if any(kw in cmd_lower for kw in test_keywords):
        return "test"
    search_keywords = ["find ", "ls ", "grep ", "rg ", "cat ", "head ",
                       "tail ", "tree ", "wc "]
    if any(kw in cmd_lower for kw in search_keywords):
        return "search"
    return "bash"


def _classify_str_replace_command(arguments: dict) -> tuple[str, str, dict]:
    """Classify a str_replace_editor command."""
    command = arguments.get("command", "")
    file_path = arguments.get("path", "")
    attrs: dict = {}
    if file_path:
        attrs["file_path"] = file_path

    if command == "view":
        return "tool", "read", attrs
    if command == "str_replace":
        return "tool", "edit", attrs
    if command in ("create", "insert"):
        return "tool", "write", attrs
    return "tool", "edit", attrs


# ---------------------------------------------------------------------------
# SWE-agent trajectory parser (.traj files)
# ---------------------------------------------------------------------------

def _sweagent_action_type(action: str) -> tuple[str, str, dict]:
    """Classify a SWE-agent action string."""
    action_stripped = action.strip()
    attrs: dict = {}

    # SWE-agent uses shell commands directly
    if action_stripped.startswith("open "):
        target = action_stripped[5:].strip().split()[0]
        attrs["file_path"] = target
        return "tool", "read", attrs

    if action_stripped.startswith("edit "):
        return "tool", "edit", attrs

    if action_stripped.startswith("create "):
        target = action_stripped[7:].strip().split()[0]
        attrs["file_path"] = target
        return "tool", "write", attrs

    if action_stripped.startswith("scroll_up") or action_stripped.startswith("scroll_down"):
        return "tool", "read", attrs

    if action_stripped.startswith("goto "):
        return "tool", "read", attrs

    if action_stripped.startswith("search_dir") or action_stripped.startswith("search_file"):
        return "tool", "search", attrs

    if action_stripped.startswith("find_file"):
        return "tool", "search", attrs

    if action_stripped.startswith("submit"):
        return "system", "submit", attrs

    # Default: treat as bash command
    name = _classify_bash_command(action_stripped)
    if len(action_stripped) <= 300:
        attrs["command"] = action_stripped
    else:
        attrs["command"] = action_stripped[:300]
    return "tool", name, attrs


def parse_sweagent_traj(traj_path: Path) -> tuple[list[dict], dict]:
    """Parse a SWE-agent .traj file into (steps, extra_info)."""
    try:
        data = json.loads(traj_path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        return [], {}

    trajectory = data.get("trajectory", [])
    steps: list[dict] = []

    for idx, entry in enumerate(trajectory):
        if not isinstance(entry, dict):
            continue
        thought = entry.get("thought", "")
        action = entry.get("action", "")
        observation = entry.get("observation", "")

        step_type, name, attrs = _sweagent_action_type(action)

        step: dict = {
            "idx": idx,
            "type": step_type,
            "name": name,
            "status": "ok",
            "attrs": attrs,
        }

        output: dict = {}
        if thought:
            output["reasoning"] = thought
        if action:
            output["action"] = action[:500]
        if observation:
            # Keep observations short to avoid bloat
            output["result"] = observation[:500]
        if output:
            step["output"] = output

        steps.append(step)

    # Extract extra info
    info = data.get("info", {})
    extra: dict = {}
    model_stats = info.get("model_stats", {})
    if model_stats:
        extra["total_cost"] = model_stats.get("instance_cost", model_stats.get("total_cost", 0))
        extra["tokens_sent"] = model_stats.get("tokens_sent", 0)
        extra["tokens_received"] = model_stats.get("tokens_received", 0)
        extra["api_calls"] = model_stats.get("api_calls", 0)
    extra["exit_status"] = info.get("exit_status", "")

    return steps, extra


# ---------------------------------------------------------------------------
# OpenHands trajectory parser (.json files, OpenAI chat format)
# ---------------------------------------------------------------------------

def _extract_text_from_content(content) -> str:
    """Extract text from OpenHands content (can be string or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content) if content else ""


def parse_openhands_traj(traj_path: Path) -> tuple[list[dict], dict]:
    """Parse an OpenHands trajectory (OpenAI chat message array)."""
    try:
        data = json.loads(traj_path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        return [], {}

    # OpenHands format: array of {role, content, tool_calls?, tool_call_id?, name?}
    if not isinstance(data, list):
        return [], {}

    steps: list[dict] = []
    idx = 0

    # Build tool_call_id -> step index map for matching results
    tc_id_to_step: dict[str, int] = {}

    for msg in data:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")

        if role == "system":
            continue

        if role == "assistant":
            reasoning_text = _extract_text_from_content(msg.get("content"))
            tool_calls = msg.get("tool_calls", [])

            if not tool_calls:
                # Pure reasoning message (no tool call)
                if reasoning_text.strip():
                    steps.append({
                        "idx": idx,
                        "type": "llm",
                        "name": "reason",
                        "status": "ok",
                        "output": {"reasoning": reasoning_text[:500]},
                    })
                    idx += 1
                continue

            for tc in tool_calls:
                fn = tc.get("function", {})
                fn_name = fn.get("name", "unknown")
                raw_args = fn.get("arguments", "{}")
                tc_id = tc.get("id", "")

                try:
                    arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    arguments = {"raw": raw_args[:300]}

                # Classify the tool call
                step_type = "tool"
                attrs: dict = {}

                if fn_name == "execute_bash":
                    cmd = arguments.get("command", "")
                    name = _classify_bash_command(cmd)
                    if len(cmd) <= 300:
                        attrs["command"] = cmd
                    else:
                        attrs["command"] = cmd[:300]
                elif fn_name == "str_replace_editor":
                    step_type, name, attrs = _classify_str_replace_command(arguments)
                elif fn_name in ("browser", "browser_action"):
                    name = "browser"
                else:
                    name = fn_name[:50]

                step: dict = {
                    "idx": idx,
                    "type": step_type,
                    "name": name,
                    "status": "ok",
                    "attrs": attrs,
                }
                output: dict = {}
                if reasoning_text.strip():
                    output["reasoning"] = reasoning_text[:500]
                    # Only attach reasoning to first tool call in a group
                    reasoning_text = ""
                if output:
                    step["output"] = output

                tc_id_to_step[tc_id] = len(steps)
                steps.append(step)
                idx += 1

        elif role == "tool":
            # Tool result message
            tc_id = msg.get("tool_call_id", "")
            result_text = _extract_text_from_content(msg.get("content"))
            step_idx = tc_id_to_step.get(tc_id)

            if step_idx is not None and step_idx < len(steps):
                step = steps[step_idx]
                if "output" not in step:
                    step["output"] = {}
                if result_text:
                    step["output"]["result"] = result_text[:500]
                # Detect errors in output
                if "error" in result_text.lower()[:200] or "traceback" in result_text.lower()[:200]:
                    step["status"] = "error"

        elif role == "user":
            # Skip user messages (typically the task prompt)
            continue

    return steps, {}


# ---------------------------------------------------------------------------
# Moatless trajectory parser (.json with transitions)
# ---------------------------------------------------------------------------

def parse_moatless_traj(traj_path: Path) -> tuple[list[dict], dict]:
    """Parse a Moatless trajectory (JSON with transitions array)."""
    try:
        data = json.loads(traj_path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        return [], {}

    transitions = data.get("transitions", [])
    if not transitions:
        return [], {}

    steps: list[dict] = []
    idx = 0

    _name_map = {
        "SearchCode": ("tool", "search"),
        "IdentifyCode": ("tool", "read"),
        "DecideRelevance": ("llm", "reason"),
        "RequestMoreContext": ("tool", "read"),
        "PlanToCode": ("llm", "plan"),
        "EditCode": ("tool", "edit"),
        "ClarifyCodeChange": ("llm", "reason"),
        "Finished": ("system", "submit"),
        "Rejected": ("system", "reject"),
    }

    for transition in transitions:
        t_name = transition.get("name", "unknown")
        step_type, step_name = _name_map.get(t_name, ("tool", t_name.lower()))

        actions = transition.get("actions", [])
        for action_entry in actions:
            output: dict = {}

            action_data = action_entry.get("action", {})
            if isinstance(action_data, dict):
                # Extract meaningful fields
                thoughts = action_data.get("thoughts", "")
                if thoughts:
                    output["reasoning"] = str(thoughts)[:500]
                scratch_pad = action_data.get("scratch_pad", "")
                if scratch_pad:
                    output["reasoning"] = str(scratch_pad)[:500]

            output_data = action_entry.get("output", {})
            if isinstance(output_data, dict):
                # Summarize output
                summary_parts = []
                for k, v in output_data.items():
                    if isinstance(v, str) and v:
                        summary_parts.append(f"{k}: {v[:100]}")
                    elif isinstance(v, list):
                        summary_parts.append(f"{k}: [{len(v)} items]")
                if summary_parts:
                    output["result"] = "; ".join(summary_parts)[:500]

            cost = action_entry.get("completion_cost", 0)
            attrs: dict = {}
            if cost:
                attrs["cost"] = cost

            step: dict = {
                "idx": idx,
                "type": step_type,
                "name": step_name,
                "status": "ok",
                "attrs": attrs,
            }
            if output:
                step["output"] = output
            steps.append(step)
            idx += 1

    # Extract info
    extra: dict = {}
    info = data.get("info", {})
    if info:
        extra["total_cost"] = info.get("total_cost", 0)
        extra["duration"] = info.get("duration", 0)

    return steps, extra


# ---------------------------------------------------------------------------
# Agentless log parser (.log plain text)
# ---------------------------------------------------------------------------

_AGENTLESS_SECTION_RE = re.compile(r"^###\s+(.+)")
_AGENTLESS_LOG_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ - \w+ - (.+)")


def parse_agentless_log(log_path: Path) -> tuple[list[dict], dict]:
    """Parse an Agentless plain-text log into steps.

    Agentless logs are structured with ### section headers and timestamped
    log lines. We extract the major phases as steps.
    """
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return [], {}

    lines = content.split("\n")
    steps: list[dict] = []
    idx = 0
    current_section = ""
    section_content: list[str] = []

    def flush_section():
        nonlocal idx
        if not current_section:
            return
        # Classify sections
        section_lower = current_section.lower()
        if "localize" in section_lower or "suspicious" in section_lower:
            step_type, name = "tool", "search"
        elif "edit" in section_lower or "patch" in section_lower or "repair" in section_lower:
            step_type, name = "tool", "edit"
        elif "predict" in section_lower or "model" in section_lower:
            step_type, name = "llm", "reason"
        elif "retrieval" in section_lower or "embedding" in section_lower:
            step_type, name = "tool", "search"
        elif "skeleton" in section_lower or "relevant" in section_lower:
            step_type, name = "tool", "read"
        elif "test" in section_lower or "verify" in section_lower:
            step_type, name = "tool", "test"
        else:
            step_type, name = "system", "phase"

        # Extract a summary from log lines
        summary_lines = []
        for line in section_content:
            m = _AGENTLESS_LOG_RE.match(line)
            if m:
                summary_lines.append(m.group(1))
            elif line.strip() and not line.startswith("#"):
                summary_lines.append(line.strip())

        output: dict = {}
        if summary_lines:
            output["result"] = "\n".join(summary_lines[:10])[:500]

        step: dict = {
            "idx": idx,
            "type": step_type,
            "name": name,
            "status": "ok",
            "output": output,
            "attrs": {"section": current_section},
        }
        steps.append(step)
        idx += 1

    for line in lines:
        m = _AGENTLESS_SECTION_RE.match(line)
        if m:
            flush_section()
            current_section = m.group(1).strip()
            section_content = []
        else:
            section_content.append(line)

    flush_section()
    return steps, {}


# ---------------------------------------------------------------------------
# Generic fallback for unknown formats
# ---------------------------------------------------------------------------

def parse_generic_json_traj(traj_path: Path) -> tuple[list[dict], dict]:
    """Attempt to parse an unknown JSON trajectory format.

    Tries common patterns: array of messages, dict with 'messages' or
    'trajectory' key, etc.
    """
    try:
        data = json.loads(traj_path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        return [], {}

    # Case 1: Has 'trajectory' key (SWE-agent variant)
    if isinstance(data, dict) and "trajectory" in data:
        return parse_sweagent_traj(traj_path)

    # Case 2: Has 'transitions' key (Moatless variant)
    if isinstance(data, dict) and "transitions" in data:
        return parse_moatless_traj(traj_path)

    # Case 3: Array of messages (OpenHands-like)
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and "role" in first:
            return parse_openhands_traj(traj_path)

    # Case 4: Dict with 'messages' key
    if isinstance(data, dict) and "messages" in data:
        messages = data["messages"]
        if isinstance(messages, list):
            # Write a temporary-like structure for OpenHands parser
            # Actually just inline the logic
            steps: list[dict] = []
            idx = 0
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", "")
                content = str(msg.get("content", ""))[:500]
                if role == "assistant":
                    steps.append({
                        "idx": idx,
                        "type": "llm",
                        "name": "reason",
                        "status": "ok",
                        "output": {"reasoning": content},
                    })
                    idx += 1
            return steps, {}

    # Case 5: Unknown dict - just extract top-level info
    if isinstance(data, dict):
        steps = [{
            "idx": 0,
            "type": "system",
            "name": "unknown_format",
            "status": "ok",
            "output": {"summary": f"keys: {list(data.keys())[:10]}"},
        }]
        return steps, {}

    return [], {}


def parse_generic_text_traj(traj_path: Path) -> tuple[list[dict], dict]:
    """Parse a plain text trajectory (markdown or log format)."""
    suffix = traj_path.suffix.lower()
    if suffix == ".log":
        return parse_agentless_log(traj_path)

    # For .md and other text files, extract sections
    try:
        content = traj_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return [], {}

    # Split by markdown headers or blank lines
    sections = re.split(r"\n(?=#{1,3} )", content)
    steps: list[dict] = []
    for idx, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
        # Extract header
        header_match = re.match(r"^#{1,3}\s+(.+)", section)
        header = header_match.group(1) if header_match else f"section_{idx}"

        steps.append({
            "idx": idx,
            "type": "system",
            "name": "phase",
            "status": "ok",
            "output": {"result": section[:500]},
            "attrs": {"section": header},
        })

    return steps, {}


# ---------------------------------------------------------------------------
# Trajectory dispatch
# ---------------------------------------------------------------------------

def detect_traj_format(traj_path: Path) -> str:
    """Detect the trajectory format from file extension and content."""
    suffix = traj_path.suffix.lower()

    if suffix == ".traj":
        return "sweagent"

    if suffix == ".json":
        # Peek at the file to distinguish formats
        try:
            raw = traj_path.read_text(encoding="utf-8", errors="replace")[:200]
        except OSError:
            return "unknown_json"

        raw_stripped = raw.strip()
        if raw_stripped.startswith("["):
            return "openhands"
        if raw_stripped.startswith("{"):
            if '"trajectory"' in raw[:500]:
                return "sweagent"
            if '"transitions"' in raw[:500]:
                return "moatless"
            if '"messages"' in raw[:1000]:
                return "generic_json"
            return "generic_json"
        return "unknown_json"

    if suffix == ".log":
        return "agentless"

    if suffix in (".md", ".txt", ".yaml", ".yml"):
        return "text"

    return "unknown"


def parse_trajectory(traj_path: Path) -> tuple[list[dict], dict, str]:
    """Parse a trajectory file, auto-detecting format.

    Returns (steps, extra_info, format_name).
    """
    fmt = detect_traj_format(traj_path)

    if fmt == "sweagent":
        steps, extra = parse_sweagent_traj(traj_path)
    elif fmt == "openhands":
        steps, extra = parse_openhands_traj(traj_path)
    elif fmt == "moatless":
        steps, extra = parse_moatless_traj(traj_path)
    elif fmt == "agentless":
        steps, extra = parse_agentless_log(traj_path)
    elif fmt == "generic_json" or fmt == "unknown_json":
        steps, extra = parse_generic_json_traj(traj_path)
    elif fmt == "text":
        steps, extra = parse_generic_text_traj(traj_path)
    else:
        steps, extra = [], {}

    return steps, extra, fmt


# ---------------------------------------------------------------------------
# Submission-level conversion
# ---------------------------------------------------------------------------

def derive_agent_name(submission_name: str, metadata: dict) -> str:
    """Derive a short agent name from submission directory name and metadata."""
    # Prefer metadata-provided name
    org = metadata.get("org", "")
    if org:
        if isinstance(org, list):
            org = org[0] if org else ""
        if org:
            return org.lower().replace(" ", "-")

    # Parse from submission name: <date>_<agent>_<model> or <date>_<agent-name>
    parts = submission_name.split("_", 1)
    if len(parts) < 2:
        return submission_name

    remainder = parts[1]
    # Strip trailing model identifiers
    for model_suffix in ["_gpt4o", "_gpt4", "_gpt35", "_claude3opus",
                         "_claude3.5sonnet", "_sonnet", "_claude",
                         "_swellama13b", "_swellama7b"]:
        if remainder.lower().endswith(model_suffix.lower()):
            remainder = remainder[:-len(model_suffix)]
            break

    return remainder.lower().replace(" ", "-")


def derive_model(submission_name: str, metadata: dict) -> str:
    """Derive model name from metadata or submission directory name."""
    models = metadata.get("models", [])
    if models:
        return models[0]

    # Try to extract from submission name
    name_lower = submission_name.lower()
    for pattern, model in [
        ("gpt4o", "gpt-4o"),
        ("gpt4", "gpt-4"),
        ("gpt35", "gpt-3.5"),
        ("claude3opus", "claude-3-opus"),
        ("claude3.5sonnet", "claude-3.5-sonnet"),
        ("claude-3-5-sonnet", "claude-3.5-sonnet"),
        ("claude-3-5-haiku", "claude-3.5-haiku"),
        ("swellama13b", "swe-llama-13b"),
        ("swellama7b", "swe-llama-7b"),
        ("sonnet-20241022", "claude-3.5-sonnet-20241022"),
    ]:
        if pattern in name_lower:
            return model

    return "unknown"


def _task_family(instance_id: str) -> str:
    """Extract repo/task family from instance_id like 'django__django-12345'."""
    # Format: <owner>__<repo>-<number>
    # Task family = owner__repo (e.g., "django__django")
    match = re.match(r"^(.+?)-\d+$", instance_id)
    if match:
        return match.group(1)
    return instance_id


def convert_submission(
    submission_dir: Path,
    split: str,
    submission_name: str,
    instance_filter: set[str] | None = None,
) -> list[dict]:
    """Convert all instances in a submission to moirai runs."""
    metadata = parse_metadata(submission_dir / "metadata.yaml")
    results = parse_results(submission_dir / "results" / "results.json")

    agent = derive_agent_name(submission_name, metadata)
    model = derive_model(submission_name, metadata)

    # Collect instance IDs from all available sources
    instance_ids = collect_all_instance_ids(submission_dir)

    # Also add instance IDs from results.json
    instance_ids.update(results.keys())

    if instance_filter:
        instance_ids = instance_ids & instance_filter

    runs: list[dict] = []
    logs_dir = submission_dir / "logs"
    trajs_dir = submission_dir / "trajs"

    for instance_id in sorted(instance_ids):
        # Parse trajectory if available
        traj_path = _find_traj_file(trajs_dir, instance_id)
        steps: list[dict] = []
        extra: dict = {}
        traj_format = "none"

        if traj_path and traj_path.exists():
            steps, extra, traj_format = parse_trajectory(traj_path)

        # Parse report.json from logs
        report_path = logs_dir / instance_id / "report.json"
        report = parse_report(report_path, instance_id)

        # Determine success from report or results.json
        status = results.get(instance_id, "unresolved")
        if report.get("resolved") is True:
            resolved = True
        elif status == "resolved":
            resolved = True
        else:
            resolved = False

        # Skip instances with no data at all
        if not steps and not report and status == "unresolved":
            continue

        run_id = f"{submission_name}/{instance_id}"

        result: dict = {"success": resolved}
        if status == "no_generation":
            result["summary"] = "no patch generated"
        elif status == "no_logs":
            result["summary"] = "no evaluation logs"
        elif report:
            parts = []
            f2p_s = report.get("fail_to_pass_success", 0)
            f2p_f = report.get("fail_to_pass_failure", 0)
            p2p_s = report.get("pass_to_pass_success", 0)
            p2p_f = report.get("pass_to_pass_failure", 0)
            if f2p_s + f2p_f > 0:
                parts.append(f"fail_to_pass: {f2p_s}/{f2p_s + f2p_f}")
            if p2p_f > 0:
                parts.append(f"regressions: {p2p_f}")
            if parts:
                result["summary"] = ", ".join(parts)

        tags: dict = {
            "split": split,
            "submission": submission_name,
            "traj_format": traj_format,
        }

        # Add metadata tags
        if metadata.get("agent_name"):
            tags["agent_display_name"] = metadata["agent_name"]
        if metadata.get("open_source_system"):
            tags["open_source"] = True
        if metadata.get("checked"):
            tags["verified"] = True
        attempts = metadata.get("attempts", "1")
        if attempts != "1":
            tags["attempts"] = attempts

        # Add extra info from trajectory parsing
        if extra.get("total_cost"):
            tags["cost"] = extra["total_cost"]
        if extra.get("tokens_sent"):
            tags["tokens_sent"] = extra["tokens_sent"]
        if extra.get("tokens_received"):
            tags["tokens_received"] = extra["tokens_received"]
        if extra.get("api_calls"):
            tags["api_calls"] = extra["api_calls"]
        if extra.get("exit_status"):
            tags["exit_status"] = extra["exit_status"]

        run = {
            "run_id": run_id,
            "task_id": instance_id,
            "task_family": _task_family(instance_id),
            "agent": agent,
            "model": model,
            "harness": f"swe-bench-{split}",
            "tags": tags,
            "steps": steps,
            "result": result,
        }

        runs.append(run)

    return runs


def _find_traj_file(trajs_dir: Path, instance_id: str) -> Path | None:
    """Find the trajectory file for an instance, trying various extensions."""
    if not trajs_dir.is_dir():
        return None

    # Try common extensions in order of preference
    for ext in (".traj", ".json", ".log", ".md", ".txt", ".yaml", ".yml"):
        candidate = trajs_dir / f"{instance_id}{ext}"
        if candidate.exists():
            return candidate

    # Fallback: glob for any file matching the instance_id
    matches = list(trajs_dir.glob(f"{instance_id}.*"))
    if matches:
        return matches[0]

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_submissions(experiments_dir: Path, split: str) -> list[tuple[Path, str]]:
    """Find all submission directories under a split."""
    split_dir = experiments_dir / "evaluation" / split
    if not split_dir.is_dir():
        print(f"error: split directory not found: {split_dir}", file=sys.stderr)
        return []

    submissions = []
    for child in sorted(split_dir.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            submissions.append((child, child.name))

    return submissions


def filter_by_agent(submissions: list[tuple[Path, str]], agent_filter: list[str]) -> list[tuple[Path, str]]:
    """Filter submissions by agent name (case-insensitive substring match)."""
    agent_lower = [a.lower() for a in agent_filter]
    filtered = []
    for path, name in submissions:
        name_lower = name.lower()
        if any(a in name_lower for a in agent_lower):
            filtered.append((path, name))
    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Convert SWE-bench/experiments trajectories into moirai run format"
    )
    parser.add_argument(
        "experiments_dir", type=Path,
        help="Path to local clone of SWE-bench/experiments"
    )
    parser.add_argument(
        "output_dir", type=Path,
        help="Output directory for moirai runs"
    )
    parser.add_argument(
        "--split", default="verified",
        help="SWE-bench split to convert (lite, verified, test, multimodal). Default: verified"
    )
    parser.add_argument(
        "--agents", default=None,
        help="Comma-separated agent name substrings to filter (e.g. 'sweagent,OpenHands')"
    )
    parser.add_argument(
        "--submissions", default=None,
        help="Comma-separated exact submission directory names to convert"
    )
    parser.add_argument(
        "--require-trajs", action="store_true",
        help="Only convert instances that have trajectory files (skip results-only)"
    )
    args = parser.parse_args()

    experiments_dir = args.experiments_dir
    output_dir = args.output_dir

    if not experiments_dir.is_dir():
        print(f"error: experiments directory not found: {experiments_dir}", file=sys.stderr)
        sys.exit(1)

    # Find submissions
    if args.submissions:
        # Explicit submission list
        submission_names = [s.strip() for s in args.submissions.split(",")]
        submissions = []
        for name in submission_names:
            path = experiments_dir / "evaluation" / args.split / name
            if path.is_dir():
                submissions.append((path, name))
            else:
                print(f"warning: submission not found: {path}", file=sys.stderr)
    else:
        submissions = find_submissions(experiments_dir, args.split)

    if args.agents:
        agent_filter = [a.strip() for a in args.agents.split(",")]
        submissions = filter_by_agent(submissions, agent_filter)

    if not submissions:
        print("error: no submissions found matching criteria", file=sys.stderr)
        sys.exit(1)

    print(f"found {len(submissions)} submissions in {args.split}", file=sys.stderr)

    output_dir.mkdir(parents=True, exist_ok=True)
    total_runs = 0
    total_with_steps = 0
    total_resolved = 0

    errors: list[str] = []
    for sub_path, sub_name in submissions:
        print(f"\nconverting {sub_name}...", file=sys.stderr)

        try:
            runs = convert_submission(sub_path, args.split, sub_name)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            errors.append(f"{sub_name}: {e}")
            continue

        if args.require_trajs:
            before = len(runs)
            runs = [r for r in runs if r["steps"]]
            skipped = before - len(runs)
            if skipped:
                print(f"  skipped {skipped} instances without trajectories", file=sys.stderr)

        if not runs:
            print(f"  no runs produced (missing logs/trajs?)", file=sys.stderr)
            continue

        # Write output: one file per run (moirai expects single Run objects)
        safe_sub = sub_name.replace("/", "_").replace(" ", "_")
        sub_out_dir = output_dir / safe_sub
        sub_out_dir.mkdir(parents=True, exist_ok=True)

        for run in runs:
            safe_id = run["task_id"].replace("/", "_").replace(" ", "_")
            out_path = sub_out_dir / f"{safe_id}.json"
            out_path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")

        with_steps = sum(1 for r in runs if r["steps"])
        resolved = sum(1 for r in runs if r["result"]["success"])

        print(
            f"  {len(runs)} runs ({with_steps} with steps, {resolved} resolved) -> {safe_sub}/",
            file=sys.stderr,
        )

        total_runs += len(runs)
        total_with_steps += with_steps
        total_resolved += resolved

    print(f"\ntotal: {total_runs} runs, {total_with_steps} with steps, {total_resolved} resolved", file=sys.stderr)
    print(f"output: {output_dir}", file=sys.stderr)
    if errors:
        print(f"\n{len(errors)} submissions failed:", file=sys.stderr)
        for err in errors:
            print(f"  {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
