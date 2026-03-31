"""HTML report generation for moirai branch analysis.

Architecture:
- Python prepares a JSON data payload + pre-rendered SVG strings
- The HTML template (branch.html) renders everything using Alpine.js
- No HTML generation in Python except SVG (computational layout)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from moirai.compress import compress_phases, NOISE_STEPS
from moirai.schema import (
    Alignment,
    ClusterResult,
    CohortDiff,
    GAP,
    Run,
)

_TEMPLATE_DIR = Path(__file__).parent / "templates"

# Step colors — for dark backgrounds
STEP_COLORS: dict[str, str] = {
    "read(source)": "#5b8fbe", "read(config)": "#7eb0d5", "read(test_file)": "#4a8ab5",
    "read(other)": "#90c0de", "read": "#5b8fbe",
    "search(glob)": "#6dc4be", "search(specific)": "#4aada7", "search": "#6dc4be",
    "edit(source)": "#e8943a", "edit(config)": "#d4a56a", "edit": "#e8943a",
    "write(source)": "#e8d44d", "write(config)": "#d4b84a", "write(other)": "#c9b96a",
    "write(test_file)": "#b8a840", "write": "#e8d44d",
    "test": "#5cb85c", "test(pass)": "#3fb950", "test(fail)": "#b5cf6b",
    "bash(python)": "#e8585a", "bash(explore)": "#ff9da7", "bash(setup)": "#d4a5a7",
    "bash(read)": "#c9827a", "bash(other)": "#d48a8c", "bash": "#e8585a",
    "reason": "#b893c4", "subagent": "#a87bb0", "plan": "#b89a7a",
    "result": "#8b949e", "task": "#6e7681", "taskoutput": "#6e7681",
    GAP: "#21262d",
}

STEP_LEGEND = [
    ("read", "#5b8fbe"), ("search", "#6dc4be"), ("edit", "#e8943a"),
    ("write", "#e8d44d"), ("test", "#5cb85c"), ("bash", "#e8585a"),
    ("subagent", "#a87bb0"), ("plan", "#b89a7a"),
]

DEFAULT_COLOR = "#6e7681"
TYPE_COLORS = {"tool": "#5b8fbe", "llm": "#b893c4", "system": "#b89a7a", "judge": "#5cb85c", "error": "#e8585a"}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def write_branch_html(
    alignment: Alignment | None,
    points: list,
    runs: list[Run],
    path: Path,
    task_results: list | None = None,
    analyze: bool = False,
) -> Path:
    path = Path(path)
    if not runs:
        _write_empty(path, "No runs to display.")
        return path

    # Build data payload
    data = _build_data(runs, task_results, analyze=analyze)

    # Load template and inject data
    # SVG strings contain Alpine.js attributes with quotes that break JSON embedding.
    # Extract SVGs from data and inject them as separate hidden divs.
    svgs = {}
    for task in data.get("tasks", []):
        svg_key = task["safe_id"]
        svgs[svg_key] = task.pop("svg", "")

    template_path = _TEMPLATE_DIR / "branch.html"
    template = template_path.read_text(encoding="utf-8")

    # Inject JSON data (now without SVG strings)
    html = template.replace('"__DATA_PLACEHOLDER__"', json.dumps(data, default=str))

    # Inject SVGs as hidden divs before </body>, referenced by task id
    svg_divs = "\n".join(
        f'<div id="svg-{key}" style="display:none">{svg}</div>'
        for key, svg in svgs.items()
    )
    html = html.replace("</body>", f"{svg_divs}\n</body>")

    path.write_text(html, encoding="utf-8")
    return path


def _build_data(runs: list[Run], task_results: list | None, analyze: bool = False) -> dict:
    """Build the complete data payload for the template."""
    n_pass = sum(1 for r in runs if r.result.success)

    tasks = []
    total_splits = 0

    if task_results:
        from moirai.analyze.splits import find_split_divergences

        for tid, task_runs, task_alignment, _task_points in task_results:
            if not task_alignment.matrix or not task_alignment.matrix[0]:
                continue

            splits, Z, dendro = find_split_divergences(task_alignment, task_runs)
            significant = [s for s in splits if s.separation > 0.3]
            total_splits += len(significant)

            task_data = _build_task_data(tid, task_runs, task_alignment, significant, Z, dendro)

            # LLM analysis
            if analyze and task_data["comparisons"]:
                from moirai.analyze.explain import explain_task
                analysis = _run_llm_analysis(tid, task_runs)
                if analysis:
                    task_data["analysis"] = analysis

            tasks.append(task_data)

    patterns = _build_patterns_data(runs)

    return {
        "stats": {
            "total_runs": len(runs),
            "pass_rate": f"{n_pass / len(runs):.0%}" if runs else "0%",
            "n_tasks": len(tasks),
            "n_div": total_splits,
        },
        "legend": STEP_LEGEND,
        "tasks": tasks,
        "patterns": patterns,
    }


# ---------------------------------------------------------------------------
# Per-task data
# ---------------------------------------------------------------------------

def _build_task_data(
    tid: str,
    runs: list[Run],
    alignment: Alignment,
    splits: list,
    Z,
    dendro: dict,
) -> dict:
    n_pass = sum(1 for r in runs if r.result.success)
    n_fail = len(runs) - n_pass
    n_cols = len(alignment.matrix[0])

    interpretation = _clustering_interpretation(splits, runs)
    svg = _build_dendrogram_heatmap_svg(alignment, runs, splits, dendro, tid)

    run_map = {r.run_id: r for r in runs}
    comparisons = []
    for i, split in enumerate(splits[:3], 1):
        comp = _build_comparison_data(split, run_map, number=i, task_id=tid)
        if comp:
            comparisons.append(comp)

    return {
        "id": tid,
        "safe_id": tid.replace(" ", "_").replace("/", "_"),
        "n_runs": len(runs),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_cols": n_cols,
        "interpretation": interpretation,
        "svg": svg,
        "comparisons": comparisons,
    }


def _clustering_interpretation(splits: list, runs: list[Run]) -> dict | None:
    if len(runs) < 3:
        return None

    n_pass = sum(1 for r in runs if r.result.success)
    n_fail = len(runs) - n_pass
    if n_pass == 0 or n_fail == 0:
        return None

    significant = [s for s in splits if s.separation > 0.3]
    if not significant:
        return {"text": f"No significant structural divergence across {len(runs)} runs.", "level": "muted"}

    predictive = [s for s in significant if s.p_value is not None and s.p_value < 0.2]
    if predictive:
        best = min(predictive, key=lambda s: s.p_value or 1.0)
        left_top = max(best.left_values, key=best.left_values.get) if best.left_values else "?"
        right_top = max(best.right_values, key=best.right_values.get) if best.right_values else "?"
        lr = f"{best.left_success_rate:.0%}" if best.left_success_rate is not None else "?"
        rr = f"{best.right_success_rate:.0%}" if best.right_success_rate is not None else "?"
        return {
            "text": f"Key split at column {best.column}: {left_top} ({lr} pass) vs {right_top} ({rr} pass), p={best.p_value:.3f}",
            "level": "success" if best.p_value < 0.05 else "accent",
        }

    return {
        "text": f"{len(significant)} structural split{'s' if len(significant) != 1 else ''}, none significantly predict outcome.",
        "level": "muted",
    }


# ---------------------------------------------------------------------------
# Pass/fail comparison data
# ---------------------------------------------------------------------------

def _build_comparison_data(split, run_map: dict[str, Run], number: int, task_id: str) -> dict | None:
    """Build data for a pass/fail comparison at a split point."""
    from moirai.analyze.align import _nw_align
    from moirai.compress import step_enriched_name

    all_rids = split.left_runs + split.right_runs
    pass_runs = [run_map[rid] for rid in all_rids if run_map.get(rid) and run_map[rid].result.success is True]
    fail_runs = [run_map[rid] for rid in all_rids if run_map.get(rid) and run_map[rid].result.success is False]

    if not pass_runs or not fail_runs:
        return None

    pass_run, fail_run = _pick_best_pair(pass_runs, fail_runs)

    # Build enriched step lists
    p_steps = _enrich_steps(pass_run)
    f_steps = _enrich_steps(fail_run)

    # Align
    p_names = [s["enriched"] for s in p_steps]
    f_names = [s["enriched"] for s in f_steps]
    aligned_p, aligned_f = _nw_align(p_names, f_names)

    # Find first divergence
    first_div = None
    for idx, (a, b) in enumerate(zip(aligned_p, aligned_f)):
        if a != b:
            first_div = idx
            break

    # Build aligned rows for the template
    rows = []
    p_idx, f_idx = 0, 0
    for i, (a, b) in enumerate(zip(aligned_p, aligned_f)):
        is_match = (a == b)
        is_first_div = (i == first_div)
        is_diff = not is_match

        p_cell = {"value": a, "detail": "", "gap": a == GAP}
        if a != GAP and p_idx < len(p_steps):
            p_cell["detail"] = p_steps[p_idx]["detail"]
            p_cell["color"] = STEP_COLORS.get(a, DEFAULT_COLOR)
            p_idx += 1

        f_cell = {"value": b, "detail": "", "gap": b == GAP}
        if b != GAP and f_idx < len(f_steps):
            f_cell["detail"] = f_steps[f_idx]["detail"]
            f_cell["color"] = STEP_COLORS.get(b, DEFAULT_COLOR)
            f_idx += 1

        rows.append({
            "pass": p_cell,
            "fail": f_cell,
            "match": is_match,
            "first_div": is_first_div,
            "diff": is_diff,
        })

    # Narrative
    narrative = None
    pass_reasoning = None
    fail_reasoning = None
    if first_div is not None:
        pa, fa = aligned_p[first_div], aligned_f[first_div]
        p_detail = _detail_at(p_steps, aligned_p, first_div)
        f_detail = _detail_at(f_steps, aligned_f, first_div)

        p_desc = f"{pa} ({p_detail})" if pa != GAP and p_detail else (pa if pa != GAP else "skipped this step")
        f_desc = f"{fa} ({f_detail})" if fa != GAP and f_detail else (fa if fa != GAP else "skipped this step")

        same_harness = pass_run.harness == fail_run.harness
        narrative = {
            "n_same": first_div,
            "pass_action": p_desc,
            "fail_action": f_desc,
            "same_harness": same_harness,
            "pass_harness": pass_run.harness,
            "fail_harness": fail_run.harness,
        }

        pass_reasoning = _reasoning_at(p_steps, aligned_p, first_div)
        fail_reasoning = _reasoning_at(f_steps, aligned_f, first_div)

    return {
        "number": number,
        "task_id": task_id,
        "column": split.column,
        "p_value": f"{split.p_value:.3f}" if split.p_value is not None else None,
        "separation": f"{split.separation:.0%}",
        "pass_run_id": pass_run.run_id,
        "fail_run_id": fail_run.run_id,
        "narrative": narrative,
        "pass_reasoning": pass_reasoning,
        "fail_reasoning": fail_reasoning,
        "rows": rows,
    }


def _enrich_steps(run: Run) -> list[dict]:
    from moirai.compress import step_enriched_name
    steps = []
    for s in run.steps:
        enriched = step_enriched_name(s)
        if enriched is None:
            continue
        detail = ""
        if s.attrs:
            fp = s.attrs.get("file_path", "")
            if fp:
                detail = os.path.basename(fp)
            cmd = s.attrs.get("command", "")
            if cmd:
                detail = cmd[:80]
            pat = s.attrs.get("pattern", "")
            if pat:
                detail = pat[:60]
        steps.append({
            "enriched": enriched,
            "detail": detail,
            "reasoning": s.output.get("reasoning", "") if s.output else "",
        })
    return steps


def _detail_at(steps, aligned, idx):
    orig = sum(1 for i in range(idx) if aligned[i] != GAP)
    if aligned[idx] == GAP or orig >= len(steps):
        return ""
    return steps[orig]["detail"]


def _reasoning_at(steps, aligned, idx):
    orig = sum(1 for i in range(idx) if aligned[i] != GAP)
    if aligned[idx] == GAP or orig >= len(steps):
        return None
    r = steps[orig]["reasoning"]
    return r if r else None


def _pick_best_pair(pass_runs: list[Run], fail_runs: list[Run]) -> tuple[Run, Run]:
    def _has_reasoning(r):
        return any(s.output.get("reasoning") for s in r.steps if s.output)
    def _harness(r):
        return r.harness or ""

    for p in pass_runs:
        for f in fail_runs:
            if _harness(p) == _harness(f) and _has_reasoning(p) and _has_reasoning(f):
                return p, f
    for p in pass_runs:
        for f in fail_runs:
            if _harness(p) == _harness(f):
                return p, f
    for p in pass_runs:
        for f in fail_runs:
            if _has_reasoning(p) or _has_reasoning(f):
                return p, f
    return pass_runs[0], fail_runs[0]


# ---------------------------------------------------------------------------
# Dendrogram + heatmap SVG (computational — stays in Python)
# ---------------------------------------------------------------------------

def _scipy_coords_to_svg(icoord, dcoord, cell_h, dendro_w):
    if not icoord:
        return []
    max_d = max(max(d) for d in dcoord) if dcoord else 1.0
    if max_d == 0:
        max_d = 1.0
    paths = []
    for ic, dc in zip(icoord, dcoord):
        pts = []
        for y_sc, x_sc in zip(ic, dc):
            svg_y = (y_sc - 5) / 10 * cell_h + cell_h / 2
            svg_x = dendro_w * (1 - x_sc / max_d)
            pts.append((svg_x, svg_y))
        d = (f"M {pts[0][0]:.1f},{pts[0][1]:.1f} "
             f"L {pts[1][0]:.1f},{pts[1][1]:.1f} "
             f"L {pts[2][0]:.1f},{pts[2][1]:.1f} "
             f"L {pts[3][0]:.1f},{pts[3][1]:.1f}")
        paths.append(f'<path d="{d}" fill="none" stroke="#484f58" stroke-width="1"/>')
    return paths


def _build_dendrogram_heatmap_svg(
    alignment: Alignment,
    runs: list[Run],
    splits: list,
    dendro: dict,
    task_id: str = "",
) -> str:
    """Build the dendrogram + heatmap SVG. This is computational layout that stays in Python."""
    if not alignment.matrix or not alignment.matrix[0]:
        return ""

    n_runs = len(alignment.run_ids)
    n_cols = len(alignment.matrix[0])
    success_map = {r.run_id: r.result.success for r in runs}

    # Step detail lookup
    step_details: dict[str, list[str]] = {}
    for run in runs:
        details = []
        for st in run.steps:
            if st.name in NOISE_STEPS:
                continue
            detail = ""
            if st.attrs:
                fp = st.attrs.get("file_path", "")
                if fp:
                    detail = os.path.basename(fp)
                cmd = st.attrs.get("command", "")
                if cmd:
                    detail = cmd[:50]
                pat = st.attrs.get("pattern", "")
                if pat:
                    detail = pat[:50]
            details.append(detail)
        step_details[run.run_id] = details

    # Dendrogram ordering
    dendro_paths: list[str] = []
    leaf_order: list[int] = list(range(n_runs))
    dendro_w = 0

    if dendro and "leaves" in dendro:
        leaf_order = list(dendro["leaves"])
        cell_h_tmp = max(8, min(16, 500 // max(n_runs, 1)))
        dendro_w = 100
        dendro_paths = _scipy_coords_to_svg(dendro["icoord"], dendro["dcoord"], cell_h_tmp, dendro_w)

    # Layout
    cell_w = max(8, min(16, 900 // max(n_cols, 1)))
    cell_h = max(8, min(16, 500 // max(n_runs, 1)))
    outcome_w = 6
    label_w = 140
    gap = 4
    heatmap_x = dendro_w + gap + outcome_w + gap + label_w
    svg_w = heatmap_x + n_cols * cell_w + 10
    marker_h = 20
    svg_h = n_runs * cell_h + marker_h + 4
    y_off = marker_h

    safe_tid = task_id.replace(" ", "_").replace("/", "_")
    parts = [f'<svg id="heatmap-{safe_tid}" width="{svg_w}" height="{svg_h}" xmlns="http://www.w3.org/2000/svg" '
             f'style="font-family:\'IBM Plex Mono\',monospace;font-size:9px">']

    # Dendrogram
    if dendro_paths:
        parts.append(f'<g transform="translate(0,{y_off})">')
        parts.extend(dendro_paths)
        parts.append('</g>')

    # Numbered divergence markers + connectors
    div_columns: dict[int, int] = {}
    max_d = max(max(d) for d in dendro["dcoord"]) if dendro.get("dcoord") else 1.0
    if max_d == 0:
        max_d = 1.0

    for i, sp in enumerate(splits):
        if sp.column < n_cols:
            num = i + 1
            div_columns[sp.column] = num
            col_x = heatmap_x + sp.column * cell_w + cell_w // 2

            if dendro_w > 0:
                branch_x = dendro_w * (1 - sp.merge_distance / max_d)
                left_rows = [leaf_order.index(alignment.run_ids.index(rid)) for rid in sp.left_runs if rid in alignment.run_ids]
                right_rows = [leaf_order.index(alignment.run_ids.index(rid)) for rid in sp.right_runs if rid in alignment.run_ids]
                if left_rows and right_rows:
                    branch_y = y_off + ((sum(left_rows) / len(left_rows) + sum(right_rows) / len(right_rows)) / 2) * cell_h + cell_h / 2
                    is_sig = sp.p_value is not None and sp.p_value < 0.2
                    stroke_color = "#d29922" if is_sig else "#484f58"
                    stroke_opacity = "0.5" if is_sig else "0.2"
                    parts.append(
                        f'<path d="M {branch_x:.1f},{branch_y:.1f} '
                        f'C {(branch_x + col_x) / 2:.1f},{branch_y:.1f} '
                        f'{(branch_x + col_x) / 2:.1f},{marker_h // 2 + 1:.1f} '
                        f'{col_x:.1f},{marker_h // 2 + 1:.1f}" '
                        f'fill="none" stroke="{stroke_color}" stroke-width="1" '
                        f'stroke-dasharray="3,3" opacity="{stroke_opacity}"/>'
                    )

    for col, num in div_columns.items():
        x = heatmap_x + col * cell_w + cell_w // 2
        parts.append(f'<line x1="{x}" y1="{marker_h}" x2="{x}" y2="{svg_h}" stroke="#d29922" stroke-width="0.5" opacity="0.15" data-col="{col}"/>')
        parts.append(f'<circle cx="{x}" cy="{marker_h // 2 + 1}" r="7" fill="#d29922" data-div-num="{num}" '
                     f'@click="scrollToFork(\'{safe_tid}\', {num})" style="cursor:pointer"/>')
        parts.append(f'<text x="{x}" y="{marker_h // 2 + 4}" text-anchor="middle" fill="#0d1117" font-size="9" font-weight="700" style="pointer-events:none">{num}</text>')

    # Rows
    for row_idx, orig_idx in enumerate(leaf_order):
        rid = alignment.run_ids[orig_idx]
        y = y_off + row_idx * cell_h
        succ = success_map.get(rid)

        parts.append('<g class="heatmap-row">')

        oc = "#3fb950" if succ is True else ("#f85149" if succ is False else "#30363d")
        parts.append(f'<rect x="{dendro_w + gap}" y="{y}" width="{outcome_w}" height="{cell_h - 1}" fill="{oc}" rx="1"/>')

        tag = "P" if succ is True else ("F" if succ is False else "?")
        tc = "#3fb950" if succ else "#f85149"
        short_id = rid[:18] + ".." if len(rid) > 20 else rid
        lx = dendro_w + gap + outcome_w + gap
        parts.append(f'<text x="{lx}" y="{y + cell_h - 3}" fill="#8b949e" font-size="8">{short_id}</text>')
        parts.append(f'<text x="{heatmap_x - 4}" y="{y + cell_h - 3}" text-anchor="end" fill="{tc}" font-size="9" font-weight="600">{tag}</text>')

        row = alignment.matrix[orig_idx]
        details = step_details.get(rid, [])
        non_gap_idx = 0
        for col in range(n_cols):
            val = row[col] if col < len(row) else GAP
            x = heatmap_x + col * cell_w
            if val == GAP:
                parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="#21262d"/>')
            else:
                color = STEP_COLORS.get(val, TYPE_COLORS.get(val, DEFAULT_COLOR))
                detail = details[non_gap_idx] if non_gap_idx < len(details) else ""
                tip_detail = detail.replace('"', '&quot;').replace("'", "&#39;") if detail else ""
                tip_step = val.replace('"', '&quot;').replace("'", "&#39;")
                is_div = col in div_columns
                stroke = ' stroke="#d29922" stroke-width="1.5"' if is_div else ""
                parts.append(
                    f'<rect x="{x}" y="{y}" width="{cell_w - 1}" height="{cell_h - 1}" '
                    f'fill="{color}" rx="1" opacity="0.85" data-col="{col}"{stroke} '
                    f'@mouseenter="showTip($event, \'{tip_step}\', \'{tip_detail}\')" '
                    f'@mousemove="moveTip($event)" '
                    f'@mouseleave="hideTip()"/>'
                )
                non_gap_idx += 1

        parts.append('</g>')

    parts.append('</svg>')
    return '\n'.join(parts)


# ---------------------------------------------------------------------------
# Patterns data
# ---------------------------------------------------------------------------

def _run_llm_analysis(task_id: str, runs: list[Run]) -> dict | None:
    """Run LLM analysis via claude CLI. Returns structured dict or None."""
    import re
    import shutil
    import subprocess
    import sys

    if not shutil.which("claude"):
        print("  skipping analysis: claude CLI not found", file=sys.stderr)
        return None

    from moirai.analyze.explain import explain_task

    prompt = explain_task(task_id, runs)
    if not prompt or "no mixed outcomes" in prompt:
        return None

    print(f"  analyzing {task_id}...", file=sys.stderr)
    try:
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0 or not result.stdout.strip():
            if result.stderr:
                print(f"  warning: {result.stderr[:200]}", file=sys.stderr)
            return None

        raw = result.stdout.strip()

        # Try to parse structured JSON from the response
        json_match = re.search(r'\{[^{}]*"finding"[^{}]*\}', raw, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: return as unstructured text
        return {"finding": raw[:500], "recommendation": "", "confidence": "low", "confidence_reason": "Unstructured response"}

    except subprocess.TimeoutExpired:
        print(f"  warning: analysis timed out for {task_id}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  warning: analysis failed for {task_id}: {e}", file=sys.stderr)
        return None


def _build_patterns_data(runs: list[Run]) -> dict:
    from moirai.analyze.motifs import find_motifs

    motifs, _ = find_motifs(runs, min_n=3, max_n=5, min_count=3)
    known = [r for r in runs if r.result.success is not None]
    if not known:
        return {"baseline": None, "positive": [], "negative": []}

    baseline = sum(1 for r in known if r.result.success) / len(known)
    positive = [{"pattern": m.display, "success_rate": f"{m.success_rate:.0%}", "runs": m.total_runs,
                 "p": f"{m.p_value:.3f}" if m.p_value is not None else ""}
                for m in motifs if m.lift > 1.05][:6]
    negative = [{"pattern": m.display, "success_rate": f"{m.success_rate:.0%}", "runs": m.total_runs,
                 "p": f"{m.p_value:.3f}" if m.p_value is not None else ""}
                for m in motifs if m.lift < 0.95][:6]

    return {"baseline": f"{baseline:.0%}", "n_known": len(known), "positive": positive, "negative": negative}


# ---------------------------------------------------------------------------
# Clusters and Diff (unchanged — still use plotly)
# ---------------------------------------------------------------------------

def write_clusters_html(result: ClusterResult, path: Path, runs: list[Run] | None = None) -> Path:
    import plotly.graph_objects as go

    path = Path(path)
    if not result.clusters:
        _write_empty(path, "No clusters.")
        return path

    sorted_clusters = sorted(result.clusters, key=lambda c: -c.count)
    run_map = {r.run_id: r for r in runs} if runs else {}
    labels, counts, rates, hovers = [], [], [], []

    for info in sorted_clusters:
        labels.append(f"C{info.cluster_id} ({info.count})")
        counts.append(info.count)
        rates.append(info.success_rate * 100 if info.success_rate is not None else 0)
        if run_map:
            members = [run_map[rid] for rid, cid in result.labels.items() if cid == info.cluster_id and rid in run_map]
            if members:
                rep = sorted(members, key=lambda r: len(r.steps))[len(members) // 2]
                hovers.append(compress_phases(rep)[:100])
            else:
                hovers.append("")
        else:
            hovers.append("")

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Count", x=labels, y=counts, marker_color="steelblue",
                         hovertext=hovers, hovertemplate="%{x}<br>%{y} runs<br>%{hovertext}<extra></extra>"))
    fig.add_trace(go.Bar(name="Success %", x=labels, y=rates, marker_color="seagreen"))
    fig.update_layout(title="Cluster Distribution", barmode="group", height=450)
    fig.write_html(str(path), include_plotlyjs=True)
    return path


def write_diff_html(diff: CohortDiff, a_label: str, b_label: str, path: Path) -> Path:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    path = Path(path)
    a_rate, b_rate = diff.a_summary.success_rate, diff.b_summary.success_rate
    if a_rate is None and b_rate is None:
        _write_empty(path, "No data.")
        return path

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Metrics", "Cluster Shifts"))
    fig.add_trace(go.Bar(name=f"A: {a_label}", x=["Success Rate", "Steps"], y=[(a_rate or 0) * 100, diff.a_summary.avg_steps], marker_color="steelblue"), row=1, col=1)
    fig.add_trace(go.Bar(name=f"B: {b_label}", x=["Success Rate", "Steps"], y=[(b_rate or 0) * 100, diff.b_summary.avg_steps], marker_color="coral"), row=1, col=1)

    if diff.cluster_shifts:
        from moirai.viz.terminal import _compress_prototype
        sl, sv, sc = [], [], []
        for proto, delta in diff.cluster_shifts[:10]:
            c = _compress_prototype(proto)
            sl.append(c[:27] + "..." if len(c) > 30 else c)
            sv.append(delta)
            sc.append("seagreen" if delta > 0 else "indianred")
        fig.add_trace(go.Bar(x=sv, y=sl, orientation="h", marker_color=sc, showlegend=False), row=1, col=2)

    fig.update_layout(title=f"{a_label} vs {b_label}", barmode="group", height=450)
    fig.write_html(str(path), include_plotlyjs=True)
    return path


def _write_empty(path: Path, message: str) -> None:
    path.write_text(
        f'<html><body style="background:#0d1117;color:#c9d1d9;font-family:monospace;padding:40px">'
        f'<h1>moirai</h1><p>{message}</p></body></html>',
        encoding="utf-8",
    )
