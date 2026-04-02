"""Generate a single-page HTML diagnosis report with three visualization layers.

1. Decision Funnel — per-task, how runs split at decision points
2. Transition Diagram — cross-task, which patterns predict success
3. Trajectory Strips — raw aligned runs as evidence

Data is injected as JSON; all rendering is client-side.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from moirai.analyze.evidence import (
    _enriched_names,
    extract_behavioral_features,
)
from moirai.schema import Run


def _pass_rate(runs: list[Run]) -> float:
    known = [r for r in runs if r.result.success is not None]
    if not known:
        return 0.0
    return sum(1 for r in known if r.result.success) / len(known)


def _build_funnel_data(task_id: str, runs: list[Run]) -> dict:
    """Build decision funnel data for a single task."""
    passes = [r for r in runs if r.result.success]
    fails = [r for r in runs if r.result.success is False]

    # Get enriched names for all runs
    run_names = [(r, _enriched_names(r)) for r in runs]

    def _classify_two_level(names: list[str]) -> tuple[str, str]:
        """Classify a run into a two-level tree.

        Level 1: Does the agent edit promptly or keep exploring?
          - edit_immediately: first edit appears within first 60% of trajectory
          - search_more: exploration dominates (>60% search/read, or no edit at all)

        Level 2 (for edit_immediately): Does the agent test after editing?
          - edit_then_test: has edit followed by test within 2 steps
          - edit_no_test: edits but never tests afterward
        """
        explore_steps = {"search", "read", "glob", "grep"}
        edit_steps = {"edit", "write"}

        first_edit_pos = None
        has_test_after_edit = False
        explore_count = 0

        for i, name in enumerate(names):
            prefix = name.split("(")[0] if "(" in name else name
            if prefix in explore_steps:
                explore_count += 1
            if prefix in edit_steps:
                if first_edit_pos is None:
                    first_edit_pos = i
                for j in range(i + 1, min(i + 3, len(names))):
                    jp = names[j].split("(")[0] if "(" in names[j] else names[j]
                    if jp == "test":
                        has_test_after_edit = True
                        break

        # Level 1: edit immediately vs search more
        if first_edit_pos is None:
            return "search_more", "search_more"

        explore_ratio = explore_count / len(names) if names else 0
        edits_late = len(names) > 4 and first_edit_pos / len(names) > 0.7

        if explore_ratio > 0.6 or edits_late:
            return "search_more", "search_more"

        # Level 2: test after edit vs submit without test
        if has_test_after_edit:
            return "edit_immediately", "edit_then_test"
        return "edit_immediately", "edit_no_test"

    # Classify all runs into the two-level tree
    level1_groups: dict[str, list[Run]] = defaultdict(list)
    level2_groups: dict[str, list[Run]] = defaultdict(list)
    for r, names in run_names:
        l1, l2 = _classify_two_level(names)
        level1_groups[l1].append(r)
        level2_groups[l2].append(r)

    # Build the tree structure for the funnel
    # Level 0: start
    # Level 1: edit_immediately vs search_more
    # Level 2: edit_then_test vs edit_no_test (children of edit_immediately)
    # Level 2: search_then_edit (child of search_more — they all eventually edit)
    nodes = []
    edges = []

    # Start node
    nodes.append({
        "id": "start", "label": "explore \u2192 read(source)",
        "count": len(runs), "pass_rate": _pass_rate(runs),
        "sublabel": "all start here",
        "level": 0, "color": "dim",
    })

    # Level 1
    edit_imm = level1_groups.get("edit_immediately", [])
    search_more = level1_groups.get("search_more", [])

    if edit_imm:
        nodes.append({
            "id": "edit_immediately", "label": "edit immediately",
            "count": len(edit_imm), "pass_rate": _pass_rate(edit_imm),
            "sublabel": f"{len(edit_imm)} runs ({len(edit_imm)*100//len(runs)}%)",
            "level": 1, "color": "green",
        })
        edges.append({"from": "start", "to": "edit_immediately"})

    if search_more:
        nodes.append({
            "id": "search_more", "label": "search more",
            "count": len(search_more), "pass_rate": _pass_rate(search_more),
            "sublabel": f"{len(search_more)} runs ({len(search_more)*100//len(runs)}%)",
            "level": 1, "color": "red",
        })
        edges.append({"from": "start", "to": "search_more"})

    # Level 2 — children of edit_immediately
    edit_test = level2_groups.get("edit_then_test", [])
    edit_notest = level2_groups.get("edit_no_test", [])

    if edit_test:
        n_pass = sum(1 for r in edit_test if r.result.success)
        nodes.append({
            "id": "edit_then_test", "label": "edit \u2192 test",
            "count": len(edit_test), "pass_rate": _pass_rate(edit_test),
            "sublabel": f"{len(edit_test)} runs",
            "level": 2, "color": "green",
            "outcome": True,
            "outcome_detail": f"{n_pass} of {len(edit_test)} pass",
            "annotation": "commit & verify",
        })
        edges.append({"from": "edit_immediately", "to": "edit_then_test"})

    if edit_notest:
        n_pass = sum(1 for r in edit_notest if r.result.success)
        nodes.append({
            "id": "edit_no_test", "label": "edit \u2192 submit",
            "count": len(edit_notest), "pass_rate": _pass_rate(edit_notest),
            "sublabel": f"{len(edit_notest)} runs",
            "level": 2, "color": "amber",
            "outcome": True,
            "outcome_detail": f"{n_pass} of {len(edit_notest)} pass",
            "annotation": "commit & hope",
        })
        edges.append({"from": "edit_immediately", "to": "edit_no_test"})

    # Level 2 — child of search_more (they eventually edit too)
    if search_more:
        n_pass = sum(1 for r in search_more if r.result.success)
        nodes.append({
            "id": "search_then_edit", "label": "search \u2192 read \u2192 edit",
            "count": len(search_more), "pass_rate": _pass_rate(search_more),
            "sublabel": f"{len(search_more)} runs (too late)",
            "level": 2, "color": "red",
            "outcome": True,
            "outcome_detail": f"{n_pass} of {len(search_more)} pass",
            "annotation": "analysis paralysis",
        })
        edges.append({"from": "search_more", "to": "search_then_edit"})

    return {
        "task_id": task_id,
        "total_runs": len(runs),
        "pass_count": len(passes),
        "fail_count": len(fails),
        "pass_rate": _pass_rate(runs),
        "nodes": nodes,
        "edges": edges,
    }


def _build_strip_data(task_id: str, runs: list[Run], max_runs: int = 50) -> dict:
    """Build trajectory strip data for a single task."""
    passes = [r for r in runs if r.result.success]
    fails = [r for r in runs if r.result.success is False]

    # Limit runs if too many
    if len(passes) > max_runs // 2:
        passes = passes[:max_runs // 2]
    if len(fails) > max_runs // 2:
        fails = fails[:max_runs // 2]

    def _run_to_strip(r: Run) -> dict:
        names = _enriched_names(r)
        # Classify each step for coloring
        cells = []
        for name in names[:30]:  # cap at 30 steps
            prefix = name.split("(")[0] if "(" in name else name
            cells.append({"name": name, "type": prefix})
        return {
            "run_id": r.run_id[:40],
            "success": r.result.success,
            "cells": cells,
        }

    return {
        "task_id": task_id,
        "pass_strips": [_run_to_strip(r) for r in passes],
        "fail_strips": [_run_to_strip(r) for r in fails],
    }


def _build_transition_data(runs: list[Run]) -> dict:
    """Build transition diagram data across all runs."""
    # Count bigram transitions and their success rates
    transitions: dict[tuple[str, str], list[bool]] = defaultdict(list)

    for r in runs:
        names = _enriched_names(r)
        outcome = r.result.success
        if outcome is None:
            continue
        # Extract prefixes for cleaner transitions
        prefixes = []
        for name in names:
            prefix = name.split("(")[0] if "(" in name else name
            if not prefixes or prefixes[-1] != prefix:  # dedupe consecutive
                prefixes.append(prefix)
        for i in range(len(prefixes) - 1):
            transitions[(prefixes[i], prefixes[i + 1])].append(outcome)

    # Build edges with success rates
    baseline = _pass_rate(runs)
    edges = []
    for (src, dst), outcomes in transitions.items():
        if len(outcomes) < max(5, len(runs) // 20):
            continue
        rate = sum(1 for o in outcomes if o) / len(outcomes)
        lift = rate / baseline if baseline > 0 else 0
        edges.append({
            "source": src,
            "target": dst,
            "count": len(outcomes),
            "pass_rate": round(rate, 3),
            "lift": round(lift, 2),
            "color": "green" if lift > 1.3 else ("red" if lift < 0.7 else "neutral"),
        })

    # Sort by absolute lift distance from 1
    edges.sort(key=lambda e: -abs(e["lift"] - 1.0))

    # Collect unique nodes
    node_set: set[str] = set()
    for e in edges:
        node_set.add(e["source"])
        node_set.add(e["target"])

    return {
        "nodes": sorted(node_set),
        "edges": edges[:30],  # top 30 by lift
        "baseline_pass_rate": round(baseline, 3),
        "total_runs": len(runs),
    }


def build_report_data(runs: list[Run]) -> dict:
    """Build complete report data for all three visualization layers."""
    # Group by task
    by_task: dict[str, list[Run]] = defaultdict(list)
    for r in runs:
        by_task[r.task_id].append(r)

    # Find mixed-outcome tasks (most interesting)
    mixed_tasks = {}
    for tid, task_runs in by_task.items():
        p = sum(1 for r in task_runs if r.result.success)
        f = sum(1 for r in task_runs if r.result.success is False)
        if p >= 2 and f >= 2:
            mixed_tasks[tid] = task_runs

    # Sort by total runs descending
    sorted_tasks = sorted(mixed_tasks.items(), key=lambda x: -len(x[1]))

    # Build per-task data (funnel + strips)
    task_data = []
    for tid, task_runs in sorted_tasks[:20]:  # top 20 tasks
        task_data.append({
            "funnel": _build_funnel_data(tid, task_runs),
            "strips": _build_strip_data(tid, task_runs),
        })

    # Build cross-task transition data
    # Use all mixed-outcome runs for the transition diagram
    all_mixed_runs = [r for rs in mixed_tasks.values() for r in rs]
    transition_data = _build_transition_data(all_mixed_runs if all_mixed_runs else runs)

    # Feature profile
    features = extract_behavioral_features(runs)

    return {
        "summary": {
            "total_runs": len(runs),
            "total_tasks": len(by_task),
            "mixed_outcome_tasks": len(mixed_tasks),
            "pass_rate": round(_pass_rate(runs), 3),
            "features": {k: round(v, 3) for k, v in features.items()},
        },
        "tasks": task_data,
        "transitions": transition_data,
    }


def write_report(runs: list[Run], path: Path) -> Path:
    """Generate and write the HTML diagnosis report."""
    data = build_report_data(runs)

    template_path = Path(__file__).parent / "templates" / "report.html"
    template = template_path.read_text(encoding="utf-8")

    data_json = json.dumps(data, default=str).replace("</", "<\\/")
    html = template.replace('"__REPORT_DATA__"', data_json)

    path.write_text(html, encoding="utf-8")
    return path
