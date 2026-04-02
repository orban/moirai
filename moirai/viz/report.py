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

    # Find the key decision point: after initial exploration, what does the agent do?
    # Look for the transition after the first read/search phase
    def _classify_strategy(names: list[str]) -> str:
        """Classify a run's strategy at the decision point."""
        saw_read_source = False
        for i, name in enumerate(names):
            if name.startswith("read") and "source" in name:
                saw_read_source = True
                # What comes next?
                if i + 1 < len(names):
                    nxt = names[i + 1]
                    if nxt.startswith("edit"):
                        # Check if test follows
                        if i + 2 < len(names) and names[i + 2].startswith("test"):
                            return "edit_then_test"
                        return "edit_no_test"
                    elif nxt.startswith("search") or nxt.startswith("read"):
                        return "keep_exploring"
            elif name.startswith("edit") and not saw_read_source:
                return "edit_without_reading"

        # Fallback: check for edit→test anywhere
        for i in range(len(names) - 1):
            if names[i].startswith("edit") and names[i + 1].startswith("test"):
                return "edit_then_test"
            if names[i].startswith("edit"):
                return "edit_no_test"

        return "other"

    # Classify all runs
    strategies: dict[str, list[Run]] = defaultdict(list)
    for r, names in run_names:
        strat = _classify_strategy(names)
        strategies[strat].append(r)

    # Build funnel nodes
    nodes = []
    nodes.append({
        "id": "start",
        "label": "all runs",
        "count": len(runs),
        "pass_rate": _pass_rate(runs),
        "level": 0,
    })

    strategy_labels = {
        "edit_then_test": "edit → test",
        "edit_no_test": "edit → submit",
        "keep_exploring": "search more",
        "edit_without_reading": "edit blind",
        "other": "other",
    }

    strategy_colors = {
        "edit_then_test": "green",
        "edit_no_test": "amber",
        "keep_exploring": "red",
        "edit_without_reading": "red",
        "other": "dim",
    }

    for strat, strat_runs in sorted(strategies.items(), key=lambda x: -_pass_rate(x[1])):
        if not strat_runs:
            continue
        nodes.append({
            "id": strat,
            "label": strategy_labels.get(strat, strat),
            "count": len(strat_runs),
            "pass_rate": _pass_rate(strat_runs),
            "color": strategy_colors.get(strat, "dim"),
            "level": 1,
        })

    return {
        "task_id": task_id,
        "total_runs": len(runs),
        "pass_count": len(passes),
        "fail_count": len(fails),
        "pass_rate": _pass_rate(runs),
        "nodes": nodes,
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
