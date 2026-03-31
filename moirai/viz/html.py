from __future__ import annotations

import os
from pathlib import Path

from moirai.compress import compress_phases, NOISE_STEPS
from moirai.schema import (
    Alignment,
    ClusterResult,
    CohortDiff,
    DivergencePoint,
    GAP,
    Run,
)

# Colors for enriched step names
STEP_COLORS: dict[str, str] = {
    "read(source)": "#4e79a7", "read(config)": "#7eb0d5", "read(test_file)": "#3a6b96",
    "read(other)": "#a0c4e0", "read": "#4e79a7",
    "search(glob)": "#76b7b2", "search(specific)": "#4a9a95", "search": "#76b7b2",
    "edit(source)": "#f28e2b", "edit(config)": "#d4a56a", "edit": "#f28e2b",
    "write(source)": "#edc948", "write(config)": "#d4b84a", "write(other)": "#c9b96a",
    "write(test_file)": "#b8a840", "write": "#edc948",
    "test": "#59a14f", "test(pass)": "#2d7d2d", "test(fail)": "#b5cf6b",
    "bash(python)": "#e15759", "bash(explore)": "#ff9da7", "bash(setup)": "#d4a5a7",
    "bash(read)": "#c9827a", "bash(other)": "#d48a8c", "bash": "#e15759",
    "reason": "#b07aa1", "subagent": "#9c5e8a", "plan": "#9c755f",
    "result": "#bab0ac", "task": "#888888", "taskoutput": "#999999",
    GAP: "#2a2a2e",
}

DEFAULT_COLOR = "#bab0ac"
TYPE_COLORS = {"tool": "#4e79a7", "llm": "#b07aa1", "system": "#9c755f", "judge": "#59a14f", "error": "#e15759"}


def write_branch_html(
    alignment: Alignment | None,
    points: list[DivergencePoint],
    runs: list[Run],
    path: Path,
    task_results: list | None = None,
) -> Path:
    """Write branch analysis dashboard with fork cards and dendrogram+heatmap."""
    path = Path(path)

    if not runs:
        _write_empty(path, "No runs to display.")
        return path

    task_sections: list[str] = []

    if task_results:
        from moirai.analyze.narrate import narrate_task

        for tid, task_runs, task_alignment, task_points in task_results:
            if not task_alignment.matrix or not task_alignment.matrix[0]:
                continue

            n_pass = sum(1 for r in task_runs if r.result.success)
            n_fail = len(task_runs) - n_pass
            n_cols = len(task_alignment.matrix[0])

            findings = narrate_task(tid, task_runs, task_alignment, task_points)

            section = f'<div class="panel" style="margin-bottom:24px">'
            section += f'<h2 style="margin-bottom:2px">{tid}</h2>'
            section += f'<div style="font-size:12px;color:#888;margin-bottom:12px">{len(task_runs)} runs ({n_pass}P/{n_fail}F), {n_cols} aligned positions</div>'

            # Fork cards
            if findings:
                for finding in findings:
                    section += _build_fork_card(finding)
            else:
                section += '<div style="color:#888;font-size:13px">No significant divergence points in this task.</div>'

            # Dendrogram + heatmap as supporting evidence
            section += f'<details style="margin-top:12px"><summary style="cursor:pointer;font-size:12px;color:#888">Show aligned trajectories</summary>'
            section += f'<div style="margin-top:8px">{_build_dendrogram_heatmap(task_alignment, task_runs, task_points)}</div>'
            section += '</details>'

            section += '</div>'
            task_sections.append(section)

    patterns_html = _build_patterns_table(runs)

    n_pass = sum(1 for r in runs if r.result.success)
    n_tasks = len(task_results) if task_results else 0
    n_div = sum(len(pts) for _, _, _, pts in task_results) if task_results else len(points)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>moirai</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #f7f7f8; color: #1a1a1a; }}
  h1 {{ font-size: 22px; margin-bottom: 2px; }}
  h2 {{ font-size: 16px; color: #333; margin-top: 0; margin-bottom: 12px; }}
  .subtitle {{ color: #888; font-size: 13px; margin-bottom: 16px; }}
  .stats {{ display: flex; gap: 32px; margin: 12px 0 20px 0; }}
  .stat-value {{ font-size: 24px; font-weight: 700; }}
  .stat-label {{ font-size: 11px; color: #999; text-transform: uppercase; letter-spacing: 0.5px; }}
  .panel {{ background: white; border-radius: 6px; padding: 20px; margin: 16px 0; border: 1px solid #e5e5e5; }}
  .success {{ color: #2d7d2d; }}
  .failure {{ color: #c0392b; }}
  .mixed {{ color: #b07800; }}
</style>
</head>
<body>
<h1>moirai</h1>
<div class="subtitle">Trajectory divergence analysis</div>
<div class="stats">
  <div><div class="stat-value">{len(runs)}</div><div class="stat-label">runs</div></div>
  <div><div class="stat-value">{n_pass}/{len(runs)}</div><div class="stat-label">pass / total</div></div>
  <div><div class="stat-value">{n_tasks}</div><div class="stat-label">tasks with mixed outcomes</div></div>
  <div><div class="stat-value">{n_div}</div><div class="stat-label">divergence points</div></div>
</div>

{''.join(task_sections)}

<div class="panel">
<h2>Patterns across all tasks</h2>
<p style="font-size:13px;color:#666;margin-top:0">Step sequences significantly correlated with success or failure across all runs.</p>
{patterns_html}
</div>

</body>
</html>"""

    path.write_text(html, encoding="utf-8")
    return path


# --- Fork cards ---

def _build_fork_card(finding) -> str:
    """Render a fork card for one divergence point."""
    p_str = f"p={finding.p_value:.3f}" if finding.p_value is not None else ""

    card = f'<div style="border-left:3px solid #4e79a7;padding:8px 14px;margin:10px 0;background:#f8fafc">'
    card += f'<div style="font-size:14px;font-weight:600;color:#222;margin-bottom:4px">{finding.summary}</div>'
    if p_str:
        card += f'<div style="font-size:11px;color:#aaa;margin-bottom:8px">{p_str}</div>'

    for branch in finding.branches:
        rate = branch.success_rate
        if rate is not None and rate >= 0.6:
            tag_color = "#2d7d2d"
            tag_bg = "#e8f5e9"
        elif rate is not None and rate <= 0.2:
            tag_color = "#c0392b"
            tag_bg = "#fce4ec"
        else:
            tag_color = "#b07800"
            tag_bg = "#fff8e1"

        rate_str = f"{rate:.0%}" if rate is not None else "?"
        card += f'<div style="margin:8px 0;padding:6px 10px;background:{tag_bg};border-radius:4px;font-size:12px">'
        card += f'<span style="font-weight:700;color:{tag_color}">{rate_str} success</span>'
        card += f' — {branch.run_count} runs chose <code>{branch.value}</code>'

        # Windowed trajectory around the fork
        fp = branch.fork_position
        window_start = max(0, fp - 4)
        window_end = min(len(branch.steps), fp + 5)
        windowed = branch.steps[window_start:window_end]

        step_parts = []
        for s in windowed:
            if s.is_fork:
                label = f'<strong style="background:{tag_color};color:white;padding:1px 4px;border-radius:2px">{s.enriched_name}</strong>'
            else:
                label = f'<span style="color:#666">{s.enriched_name}</span>'
            if s.detail:
                label += f'<span style="color:#bbb;font-size:10px"> {s.detail}</span>'
            step_parts.append(label)

        if window_start > 0:
            step_parts.insert(0, '<span style="color:#ccc">...</span>')
        if window_end < len(branch.steps):
            step_parts.append('<span style="color:#ccc">...</span>')

        card += f'<div style="font-family:SF Mono,Menlo,monospace;font-size:11px;margin-top:4px;line-height:1.8">'
        card += ' → '.join(step_parts)
        card += '</div>'

        # Reasoning excerpt
        if branch.reasoning:
            card += f'<div style="font-style:italic;color:#555;font-size:11px;margin-top:4px;padding:4px 8px;background:rgba(0,0,0,0.03);border-radius:3px">'
            card += f'"{branch.reasoning[:200]}"'
            card += '</div>'

        # Run ID
        card += f'<div style="font-size:10px;color:#bbb;margin-top:2px">Run: {branch.representative_run_id}</div>'
        card += '</div>'

    card += '</div>'
    return card


# --- Dendrogram + heatmap ---

def _scipy_coords_to_svg(
    icoord: list[list[float]],
    dcoord: list[list[float]],
    cell_h: float,
    dendro_w: float,
) -> list[str]:
    """Map scipy dendrogram coordinates to SVG path elements.

    scipy dendrogram(no_plot=True) returns icoord/dcoord as lists of 4-element
    lists representing U-shaped brackets. Default leaf spacing: 5, 15, 25...
    """
    if not icoord:
        return []

    # Find max merge distance for x-scaling
    max_d = max(max(d) for d in dcoord) if dcoord else 1.0
    if max_d == 0:
        max_d = 1.0  # all identical trajectories

    paths = []
    for ic, dc in zip(icoord, dcoord):
        # ic = [y1, y1, y2, y2] (the U-bracket y positions)
        # dc = [x1, x2, x2, x1] wait no — scipy uses:
        # icoord = y-axis (leaf positions), dcoord = x-axis (merge distance)
        # Each bracket: 4 points forming a U shape
        # (ic[0],dc[0]) -> (ic[1],dc[1]) -> (ic[2],dc[2]) -> (ic[3],dc[3])
        # which is: go up from left child, across at merge height, down to right child

        svg_points = []
        for y_sc, x_sc in zip(ic, dc):
            # Map y: scipy leaf positions are 5, 15, 25... (spacing=10, offset=5)
            svg_y = (y_sc - 5) / 10 * cell_h + cell_h / 2
            # Map x: 0 (leaves) on right, max distance on left
            svg_x = dendro_w * (1 - x_sc / max_d)
            svg_points.append((svg_x, svg_y))

        # Draw as polyline: down, across, up
        d = (f"M {svg_points[0][0]:.1f},{svg_points[0][1]:.1f} "
             f"L {svg_points[1][0]:.1f},{svg_points[1][1]:.1f} "
             f"L {svg_points[2][0]:.1f},{svg_points[2][1]:.1f} "
             f"L {svg_points[3][0]:.1f},{svg_points[3][1]:.1f}")
        paths.append(f'<path d="{d}" fill="none" stroke="#666" stroke-width="1"/>')

    return paths


def _build_dendrogram_heatmap(
    alignment: Alignment,
    runs: list[Run],
    points: list[DivergencePoint],
) -> str:
    """Build dendrogram + heatmap SVG.

    Dendrogram on left (hierarchical clustering), outcome strip, run labels,
    heatmap on right. Rows ordered by dendrogram leaf order.
    """
    if not alignment.matrix or not alignment.matrix[0]:
        return "<p>No alignment data.</p>"

    n_runs = len(alignment.run_ids)
    n_cols = len(alignment.matrix[0])
    success_map = {r.run_id: r.result.success for r in runs}

    # Build step detail lookup
    step_details: dict[str, list[str]] = {}
    for run in runs:
        details = []
        for s in run.steps:
            if s.name in NOISE_STEPS:
                continue
            detail = ""
            if s.attrs:
                fp = s.attrs.get("file_path", "")
                if fp:
                    detail = os.path.basename(fp)
                cmd = s.attrs.get("command", "")
                if cmd:
                    detail = cmd[:50]
                pat = s.attrs.get("pattern", "")
                if pat:
                    detail = pat[:50]
            details.append(detail)
        step_details[run.run_id] = details

    # Compute dendrogram ordering (need 2+ runs)
    dendro_paths: list[str] = []
    leaf_order: list[int] = list(range(n_runs))
    dendro_w = 0

    if n_runs >= 2:
        from moirai.analyze.align import distance_matrix
        from scipy.cluster.hierarchy import linkage, dendrogram as scipy_dendrogram

        level = alignment.level if alignment.level == "name" else "name"
        condensed = distance_matrix(runs, level=level)

        if len(condensed) > 0:
            Z = linkage(condensed, method="average")
            result = scipy_dendrogram(Z, no_plot=True)
            leaf_order = list(result["leaves"])

            cell_h = max(6, min(14, 400 // max(n_runs, 1)))
            dendro_w = 100
            dendro_paths = _scipy_coords_to_svg(
                result["icoord"], result["dcoord"],
                cell_h, dendro_w,
            )

    # Layout dimensions
    cell_w = max(8, min(16, 800 // max(n_cols, 1)))
    cell_h = max(6, min(14, 400 // max(n_runs, 1)))
    outcome_w = 10
    label_w = 160
    gap = 4
    heatmap_x = dendro_w + gap + outcome_w + gap + label_w
    svg_w = heatmap_x + n_cols * cell_w + 10
    svg_h = n_runs * cell_h + 20  # extra for tick marks

    parts = [f'<svg width="{svg_w}" height="{svg_h}" xmlns="http://www.w3.org/2000/svg" '
             f'style="font-family:SF Mono,Menlo,monospace;font-size:9px">']

    # Dendrogram paths
    if dendro_paths:
        parts.append(f'<g class="dendrogram">')
        parts.extend(dendro_paths)
        parts.append('</g>')

    # Divergence tick marks on column header
    div_columns = {p.column for p in points}
    for col in div_columns:
        if col < n_cols:
            x = heatmap_x + col * cell_w + cell_w // 2
            parts.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{svg_h}" stroke="#e15759" stroke-width="0.5" opacity="0.3"/>')

    # Rows in dendrogram leaf order
    for row_idx, orig_idx in enumerate(leaf_order):
        rid = alignment.run_ids[orig_idx]
        y = row_idx * cell_h

        # Outcome strip
        s = success_map.get(rid)
        outcome_color = "#2d7d2d" if s is True else ("#c0392b" if s is False else "#ccc")
        ox = dendro_w + gap
        parts.append(f'<rect x="{ox}" y="{y}" width="{outcome_w}" height="{cell_h - 1}" fill="{outcome_color}" rx="1"/>')

        # Run label
        tag = "P" if s is True else ("F" if s is False else "?")
        tag_color = "#2d7d2d" if s else "#c0392b"
        short_id = rid[:20] + ".." if len(rid) > 22 else rid
        lx = dendro_w + gap + outcome_w + gap
        parts.append(f'<text x="{lx}" y="{y + cell_h - 2}" fill="#888" font-size="8">{short_id}</text>')
        parts.append(f'<text x="{heatmap_x - 4}" y="{y + cell_h - 2}" text-anchor="end" fill="{tag_color}" font-size="9" font-weight="bold">{tag}</text>')

        # Heatmap cells
        row = alignment.matrix[orig_idx]
        details = step_details.get(rid, [])
        non_gap_idx = 0
        for col in range(n_cols):
            val = row[col] if col < len(row) else GAP
            x = heatmap_x + col * cell_w

            if val == GAP:
                parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="#f0f0f0"/>')
            else:
                color = STEP_COLORS.get(val, TYPE_COLORS.get(val, DEFAULT_COLOR))
                detail = details[non_gap_idx] if non_gap_idx < len(details) else ""
                tooltip = f"{val}"
                if detail:
                    tooltip += f" — {detail}"
                is_div = col in div_columns
                stroke = ' stroke="#e15759" stroke-width="1"' if is_div else ""
                parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 1}" height="{cell_h - 1}" fill="{color}" rx="1"{stroke}>'
                             f'<title>{tooltip}</title></rect>')
                non_gap_idx += 1

    parts.append('</svg>')
    return '\n'.join(parts)


# --- Patterns table ---

def _build_patterns_table(runs: list[Run]) -> str:
    """Build patterns table."""
    from moirai.analyze.motifs import find_motifs

    motifs = find_motifs(runs, min_n=3, max_n=5, min_count=3)
    known = [r for r in runs if r.result.success is not None]
    if not known:
        return "<p>No outcome data.</p>"
    baseline = sum(1 for r in known if r.result.success) / len(known)

    positive = [m for m in motifs if m.lift > 1.05][:6]
    negative = [m for m in motifs if m.lift < 0.95][:6]

    if not positive and not negative:
        return f"<p>No significant patterns found (baseline: {baseline:.0%}).</p>"

    html = f'<p style="font-size:12px;color:#888;margin-top:0">Baseline: {baseline:.0%} across {len(known)} runs</p>'
    html += '<table style="width:100%;border-collapse:collapse;font-size:13px">'
    html += '<tr style="border-bottom:2px solid #e5e5e5"><th style="text-align:left;padding:6px">Pattern</th>'
    html += '<th style="text-align:right;padding:6px">Success</th>'
    html += '<th style="text-align:right;padding:6px">Runs</th>'
    html += '<th style="text-align:right;padding:6px">p</th></tr>'

    for m in positive:
        p_str = f"{m.p_value:.3f}" if m.p_value is not None else ""
        html += f'<tr style="border-bottom:1px solid #f0f0f0;background:#f6fbf6">'
        html += f'<td style="padding:6px;font-family:SF Mono,Menlo,monospace;font-size:12px">{m.display}</td>'
        html += f'<td style="text-align:right;padding:6px" class="success"><strong>{m.success_rate:.0%}</strong></td>'
        html += f'<td style="text-align:right;padding:6px;color:#888">{m.total_runs}</td>'
        html += f'<td style="text-align:right;padding:6px;color:#bbb">{p_str}</td></tr>'

    if positive and negative:
        html += '<tr><td colspan="4" style="padding:3px"></td></tr>'

    for m in negative:
        p_str = f"{m.p_value:.3f}" if m.p_value is not None else ""
        html += f'<tr style="border-bottom:1px solid #f0f0f0;background:#fcf6f6">'
        html += f'<td style="padding:6px;font-family:SF Mono,Menlo,monospace;font-size:12px">{m.display}</td>'
        html += f'<td style="text-align:right;padding:6px" class="failure"><strong>{m.success_rate:.0%}</strong></td>'
        html += f'<td style="text-align:right;padding:6px;color:#888">{m.total_runs}</td>'
        html += f'<td style="text-align:right;padding:6px;color:#bbb">{p_str}</td></tr>'

    html += '</table>'
    return html


# --- Clusters and Diff (unchanged) ---

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
    path.write_text(f"<html><body><h1>moirai</h1><p>{message}</p></body></html>", encoding="utf-8")
