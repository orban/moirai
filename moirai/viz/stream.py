"""Stream geometry computation from dendrogram linkage.

Walks a scipy linkage matrix (Z) and builds a stream tree -- a hierarchical
structure describing how runs split into branches. Each branch tracks its
runs, success rate, and per-column step type proportions.

Also includes alignment matrix data for the interactive heatmap viewer.
"""
from __future__ import annotations

import html
import os
import numpy as np
from scipy.stats import fisher_exact

from moirai.compress import (
    compress_phases,
    compress_steps,
    step_enriched_name,
    step_phase,
)
from moirai.schema import Alignment, GAP, Run, Step
from moirai.viz.html import STEP_COLORS


def build_stream_tree(
    alignment: Alignment,
    runs: list[Run],
    Z: np.ndarray,
    dendro: dict | None = None,
    max_splits: int = 4,
) -> dict:
    """Build a stream tree from alignment + linkage matrix.

    Only the top ``max_splits`` most meaningful splits (by merge distance)
    are rendered as bifurcations. Everything below that threshold is
    collapsed into a single leaf branch.

    Returns a dict matching the StreamTree TypeScript interface.
    """
    n_runs = len(runs)
    n_cols = len(alignment.matrix[0]) if alignment.matrix and alignment.matrix[0] else 0
    run_map = {r.run_id: r for r in runs}
    success_map = {r.run_id: r.result.success for r in runs}

    n_pass = sum(1 for r in runs if r.result.success)
    n_fail = n_runs - n_pass
    task_id = runs[0].task_id if runs else ""

    # Build leaf sets for each node by walking Z bottom-up
    leaves_of: dict[int, set[int]] = {}
    for i in range(n_runs):
        leaves_of[i] = {i}

    for i, row in enumerate(Z):
        left_node, right_node = int(row[0]), int(row[1])
        node_id = n_runs + i
        leaves_of[node_id] = leaves_of[int(left_node)] | leaves_of[int(right_node)]

    root_node = n_runs + len(Z) - 1 if len(Z) > 0 else 0

    # Determine which internal nodes are worth splitting.
    split_nodes: set[int] = set()
    if len(Z) > 0:
        distances = [(n_runs + i, float(Z[i][2])) for i in range(len(Z))]
        distances.sort(key=lambda x: -x[1])
        for node_id, _dist in distances[:max_splits]:
            split_nodes.add(node_id)

    branches: dict[str, dict] = {}
    bifurcations: dict[str, dict] = {}
    tree: dict[str, list[str]] = {}

    def _run_ids_for(indices: set[int]) -> list[str]:
        return [alignment.run_ids[j] for j in sorted(indices)]

    def _success_rate(rids: list[str]) -> float:
        known = [success_map[rid] for rid in rids if success_map.get(rid) is not None]
        if not known:
            return 0.0
        return sum(1 for x in known if x) / len(known)

    def _step_proportions(indices: set[int]) -> dict[int, dict[str, float]]:
        props: dict[int, dict[str, float]] = {}
        for col in range(n_cols):
            counts: dict[str, int] = {}
            total = 0
            for j in indices:
                val = alignment.matrix[j][col] if col < len(alignment.matrix[j]) else GAP
                if val != GAP:
                    counts[val] = counts.get(val, 0) + 1
                    total += 1
            if total > 0:
                props[col] = {k: v / total for k, v in counts.items()}
        return props

    def _representative_trajectory(indices: set[int]) -> str:
        rids = _run_ids_for(indices)
        branch_runs = [run_map[rid] for rid in rids if rid in run_map]
        if not branch_runs:
            return "(empty)"
        branch_runs.sort(key=lambda r: len(r.steps))
        median_run = branch_runs[len(branch_runs) // 2]
        return compress_steps(median_run.steps)

    def _phase_mix(indices: set[int]) -> dict[str, float]:
        phase_counts: dict[str, int] = {}
        total = 0
        rids = _run_ids_for(indices)
        for rid in rids:
            run = run_map.get(rid)
            if not run:
                continue
            for step in run.steps:
                phase = step_phase(step)
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
                total += 1
        if total == 0:
            return {}
        return {k: v / total for k, v in phase_counts.items()}

    def _discriminating_column(left_indices: set[int], right_indices: set[int]) -> tuple[int, float]:
        best_col = -1
        best_score = 0.0
        left_list = sorted(left_indices)
        right_list = sorted(right_indices)

        for col in range(n_cols):
            left_vals = [
                alignment.matrix[i][col] for i in left_list
                if col < len(alignment.matrix[i]) and alignment.matrix[i][col] != GAP
            ]
            right_vals = [
                alignment.matrix[i][col] for i in right_list
                if col < len(alignment.matrix[i]) and alignment.matrix[i][col] != GAP
            ]
            if not left_vals or not right_vals:
                continue
            disagree = sum(1 for lv in left_vals for rv in right_vals if lv != rv)
            score = disagree / (len(left_vals) * len(right_vals))
            if score > best_score:
                best_score = score
                best_col = col
        return best_col, best_score

    def _fisher_p_value(left_rids: list[str], right_rids: list[str]) -> float | None:
        left_pass = sum(1 for rid in left_rids if success_map.get(rid) is True)
        left_fail = sum(1 for rid in left_rids if success_map.get(rid) is False)
        right_pass = sum(1 for rid in right_rids if success_map.get(rid) is True)
        right_fail = sum(1 for rid in right_rids if success_map.get(rid) is False)
        total_known = left_pass + left_fail + right_pass + right_fail
        if total_known < 4:
            return None
        table = [[left_pass, left_fail], [right_pass, right_fail]]
        result = fisher_exact(table)
        return float(result.pvalue)

    def _build_branch(node_id: int, indices: set[int]) -> str:
        branch_id = f"branch_{node_id}"
        rids = _run_ids_for(indices)

        col_start = n_cols
        col_end = 0
        for col in range(n_cols):
            for j in indices:
                val = alignment.matrix[j][col] if col < len(alignment.matrix[j]) else GAP
                if val != GAP:
                    col_start = min(col_start, col)
                    col_end = max(col_end, col)
                    break
        if col_start > col_end:
            col_start = 0
            col_end = max(0, n_cols - 1)

        branches[branch_id] = {
            "id": branch_id,
            "run_ids": rids,
            "success_rate": _success_rate(rids),
            "step_proportions": _step_proportions(indices),
            "trajectory": _representative_trajectory(indices),
            "phase_mix": _phase_mix(indices),
            "col_start": col_start,
            "col_end": col_end,
        }
        return branch_id

    def _walk(node_id: int) -> str:
        indices = leaves_of[node_id]
        branch_id = _build_branch(node_id, indices)

        if node_id < n_runs or node_id not in split_nodes:
            return branch_id

        z_row = node_id - n_runs
        if z_row < 0 or z_row >= len(Z):
            return branch_id

        left_node = int(Z[z_row][0])
        right_node = int(Z[z_row][1])
        left_indices = leaves_of[left_node]
        right_indices = leaves_of[right_node]

        col, separation = _discriminating_column(left_indices, right_indices)

        left_rids = _run_ids_for(left_indices)
        right_rids = _run_ids_for(right_indices)

        left_branch_id = _walk(left_node)
        right_branch_id = _walk(right_node)

        p_value = _fisher_p_value(left_rids, right_rids)

        bif_id = f"bif_{node_id}"
        bifurcations[bif_id] = {
            "id": bif_id,
            "column": col,
            "x_position": col / n_cols if n_cols > 0 else 0.0,
            "separation": separation,
            "p_value": p_value,
            "significant": p_value is not None and p_value < 0.2,
            "left_branch_id": left_branch_id,
            "right_branch_id": right_branch_id,
            "left_success_rate": _success_rate(left_rids),
            "right_success_rate": _success_rate(right_rids),
        }

        tree[bif_id] = [left_branch_id, right_branch_id]
        return branch_id

    root_branch_id = _walk(root_node)

    # Build alignment matrix in dendrogram leaf order
    leaf_order: list[int] = list(range(n_runs))
    if dendro and "leaves" in dendro:
        leaf_order = list(dendro["leaves"])

    alignment_run_ids = [alignment.run_ids[i] for i in leaf_order]
    alignment_matrix = [list(alignment.matrix[i]) for i in leaf_order]

    return {
        "task_id": task_id,
        "n_runs": n_runs,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_cols": n_cols,
        "root_branch_id": root_branch_id,
        "branches": branches,
        "bifurcations": bifurcations,
        "tree": tree,
        "alignment_matrix": alignment_matrix,
        "alignment_run_ids": alignment_run_ids,
    }


# ---------------------------------------------------------------------------
# Run metadata and detail serialization
# ---------------------------------------------------------------------------

def _step_detail_str(step: Step) -> str:
    """Extract a human-readable detail string from step attrs."""
    if not step.attrs:
        return ""

    fp = step.attrs.get("file_path", "")
    if fp:
        return html.escape(os.path.basename(fp))

    cmd = step.attrs.get("command", "")
    if cmd:
        return html.escape(cmd[:80])

    pat = step.attrs.get("pattern", "")
    if pat:
        return html.escape(pat)

    return ""


def build_run_meta(run: Run) -> dict:
    """Build tier-1 run metadata (small, always loaded)."""
    return {
        "run_id": run.run_id,
        "success": run.result.success,
        "step_count": len(run.steps),
        "trajectory": compress_steps(run.steps),
        "phase_mix": compress_phases(run),
        "harness": run.harness,
        "model": run.model,
    }


def build_run_detail(run: Run) -> dict:
    """Build tier-2 run detail (loaded on expand)."""
    steps = []
    for step in run.steps:
        enriched = step_enriched_name(step)
        if enriched is None:
            continue
        phase = step_phase(step)
        color = STEP_COLORS.get(enriched, "#6e7681")
        detail = _step_detail_str(step)

        steps.append({
            "idx": step.idx,
            "enriched": enriched,
            "detail": detail,
            "status": step.status,
            "phase": phase,
            "color": color,
            "metrics": step.metrics,
        })

    return {
        "run_id": run.run_id,
        "steps": steps,
    }
