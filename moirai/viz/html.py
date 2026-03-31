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

# Step colors — designed for dark backgrounds
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

# Grouped for legend display
STEP_LEGEND = [
    ("read", "#5b8fbe"), ("search", "#6dc4be"), ("edit", "#e8943a"),
    ("write", "#e8d44d"), ("test", "#5cb85c"), ("bash", "#e8585a"),
    ("subagent", "#a87bb0"), ("plan", "#b89a7a"),
]

DEFAULT_COLOR = "#6e7681"
TYPE_COLORS = {"tool": "#5b8fbe", "llm": "#b893c4", "system": "#b89a7a", "judge": "#5cb85c", "error": "#e8585a"}


def write_branch_html(
    alignment: Alignment | None,
    points: list[DivergencePoint],
    runs: list[Run],
    path: Path,
    task_results: list | None = None,
) -> Path:
    """Write branch analysis dashboard."""
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

            section = '<div class="task-panel">'
            section += f'<div class="task-header"><span class="task-name">{tid}</span>'
            section += f'<span class="task-meta">{len(task_runs)} runs'
            section += f'<span class="pass-count">{n_pass}P</span>'
            section += f'<span class="fail-count">{n_fail}F</span>'
            section += f'{n_cols} cols</span></div>'

            # Clustering interpretation
            section += _clustering_interpretation(task_alignment, task_runs)

            # Dendrogram + heatmap
            section += f'<div class="heatmap-container">{_build_dendrogram_heatmap(task_alignment, task_runs, task_points)}</div>'

            # Fork cards
            if findings:
                section += '<div class="fork-cards">'
                for i, finding in enumerate(findings, 1):
                    section += _build_fork_card(finding, number=i)
                section += '</div>'

            section += '</div>'
            task_sections.append(section)

    patterns_html = _build_patterns_table(runs)

    n_pass = sum(1 for r in runs if r.result.success)
    n_tasks = len(task_results) if task_results else 0
    n_div = sum(len(pts) for _, _, _, pts in task_results) if task_results else len(points)
    pass_rate = f"{n_pass / len(runs):.0%}" if runs else "0%"

    # Step legend
    legend_items = "".join(
        f'<span class="legend-item"><span class="legend-swatch" style="background:{color}"></span>{name}</span>'
        for name, color in STEP_LEGEND
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>moirai</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #0d1117;
    --surface: #161b22;
    --surface-raised: #1c2128;
    --border: #30363d;
    --text: #c9d1d9;
    --text-muted: #8b949e;
    --text-bright: #e6edf3;
    --accent: #d29922;
    --success: #3fb950;
    --failure: #f85149;
    --info: #58a6ff;
    --mono: 'IBM Plex Mono', 'SF Mono', Menlo, monospace;
    --sans: 'Outfit', -apple-system, sans-serif;
  }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: var(--mono); margin: 0; padding: 0; background: var(--bg); color: var(--text); }}

  .header {{ padding: 32px 40px 24px; border-bottom: 1px solid var(--border); }}
  .header h1 {{ font-family: var(--sans); font-size: 20px; font-weight: 700; color: var(--text-bright); margin: 0 0 2px; letter-spacing: -0.3px; }}
  .header .subtitle {{ font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1.5px; }}

  .stats-bar {{ display: flex; gap: 40px; padding: 16px 40px; border-bottom: 1px solid var(--border); background: var(--surface); }}
  .stat {{ display: flex; align-items: baseline; gap: 6px; }}
  .stat-value {{ font-family: var(--sans); font-size: 22px; font-weight: 700; color: var(--text-bright); }}
  .stat-label {{ font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.8px; }}

  .legend-bar {{ display: flex; flex-wrap: wrap; gap: 12px; padding: 10px 40px; border-bottom: 1px solid var(--border); background: var(--surface); }}
  .legend-item {{ display: flex; align-items: center; gap: 4px; font-size: 10px; color: var(--text-muted); }}
  .legend-swatch {{ width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }}

  .content {{ padding: 20px 40px; }}

  .task-panel {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 20px; overflow: hidden; }}
  .task-header {{ padding: 14px 20px; border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; }}
  .task-name {{ font-family: var(--sans); font-size: 14px; font-weight: 600; color: var(--text-bright); }}
  .task-meta {{ font-size: 11px; color: var(--text-muted); display: flex; gap: 8px; align-items: center; }}
  .pass-count {{ color: var(--success); font-weight: 600; }}
  .fail-count {{ color: var(--failure); font-weight: 600; }}

  .heatmap-container {{ padding: 16px 20px; overflow-x: auto; }}
  .cluster-interpretation {{ padding: 8px 20px 0; font-size: 11px; font-style: italic; }}

  .fork-cards {{ padding: 0 20px 16px; }}
  .fork-card {{ background: var(--surface-raised); border: 1px solid var(--border); border-radius: 6px; padding: 12px 16px; margin-top: 12px; }}
  .fork-card-header {{ display: flex; align-items: flex-start; gap: 8px; margin-bottom: 8px; }}
  .fork-number {{ display: inline-flex; align-items: center; justify-content: center; width: 20px; height: 20px; background: var(--accent); color: var(--bg); border-radius: 50%; font-size: 11px; font-weight: 700; flex-shrink: 0; margin-top: 1px; }}
  .fork-summary {{ font-size: 12px; color: var(--text); line-height: 1.4; }}
  .fork-pvalue {{ font-size: 10px; color: var(--text-muted); margin-left: 28px; }}

  .branch {{ margin: 8px 0; padding: 8px 12px; border-radius: 4px; border-left: 3px solid; }}
  .branch-success {{ border-color: var(--success); background: rgba(63, 185, 80, 0.06); }}
  .branch-failure {{ border-color: var(--failure); background: rgba(248, 81, 73, 0.06); }}
  .branch-mixed {{ border-color: var(--accent); background: rgba(210, 153, 34, 0.06); }}
  .branch-header {{ font-size: 12px; margin-bottom: 4px; }}
  .branch-rate {{ font-weight: 600; }}
  .branch-trajectory {{ font-size: 10px; color: var(--text-muted); line-height: 1.7; margin-top: 4px; }}
  .branch-trajectory .fork-step {{ background: var(--accent); color: var(--bg); padding: 1px 4px; border-radius: 2px; font-weight: 600; }}
  .branch-trajectory .ellipsis {{ color: var(--border); }}
  .branch-reasoning {{ font-size: 10px; color: var(--text-muted); font-style: italic; margin-top: 4px; padding: 4px 8px; background: rgba(255,255,255,0.03); border-radius: 3px; }}
  .branch-run-id {{ font-size: 9px; color: #484f58; margin-top: 3px; }}

  .patterns-panel {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 20px; overflow: hidden; }}
  .patterns-panel h2 {{ font-family: var(--sans); font-size: 14px; font-weight: 600; color: var(--text-bright); padding: 14px 20px; margin: 0; border-bottom: 1px solid var(--border); }}
  .patterns-panel .patterns-body {{ padding: 12px 20px; }}

  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th {{ text-align: left; padding: 6px 8px; color: var(--text-muted); font-weight: 500; border-bottom: 1px solid var(--border); font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; }}
  td {{ padding: 6px 8px; border-bottom: 1px solid rgba(48,54,61,0.5); }}
  tr.positive td {{ background: rgba(63,185,80,0.04); }}
  tr.negative td {{ background: rgba(248,81,73,0.04); }}
  .success {{ color: var(--success); }}
  .failure {{ color: var(--failure); }}
</style>
</head>
<body>

<div class="header">
  <h1>moirai</h1>
  <div class="subtitle">Trajectory divergence analysis</div>
</div>

<div class="stats-bar">
  <div class="stat"><span class="stat-value">{len(runs)}</span><span class="stat-label">runs</span></div>
  <div class="stat"><span class="stat-value">{pass_rate}</span><span class="stat-label">pass rate</span></div>
  <div class="stat"><span class="stat-value">{n_tasks}</span><span class="stat-label">mixed tasks</span></div>
  <div class="stat"><span class="stat-value">{n_div}</span><span class="stat-label">divergence pts</span></div>
</div>

<div class="legend-bar">
  {legend_items}
</div>

<div class="content">
{''.join(task_sections)}

<div class="patterns-panel">
<h2>Patterns</h2>
<div class="patterns-body">
{patterns_html}
</div>
</div>
</div>

</body>
</html>"""

    path.write_text(html, encoding="utf-8")
    return path


# --- Clustering interpretation ---

def _clustering_interpretation(alignment: Alignment, runs: list[Run]) -> str:
    if len(runs) < 3:
        return ""

    success_map = {r.run_id: r.result.success for r in runs}
    n_pass = sum(1 for r in runs if r.result.success)
    n_fail = len(runs) - n_pass

    if n_pass == 0 or n_fail == 0:
        return ""

    try:
        from moirai.analyze.align import distance_matrix
        from scipy.cluster.hierarchy import linkage, dendrogram as scipy_dendrogram

        condensed = distance_matrix(runs, level="name")
        if len(condensed) == 0:
            return ""
        Z = linkage(condensed, method="average")
        result = scipy_dendrogram(Z, no_plot=True)
        leaves = result["leaves"]
    except Exception:
        return ""

    outcomes = [success_map.get(alignment.run_ids[i]) for i in leaves]
    best_purity = 0.0
    for split in range(1, len(outcomes)):
        left = outcomes[:split]
        right = outcomes[split:]
        left_pass = sum(1 for o in left if o is True)
        right_pass = sum(1 for o in right if o is True)
        left_purity = max(left_pass, len(left) - left_pass) / len(left)
        right_purity = max(right_pass, len(right) - right_pass) / len(right)
        avg_purity = (left_purity * len(left) + right_purity * len(right)) / len(outcomes)
        if avg_purity > best_purity:
            best_purity = avg_purity

    if best_purity >= 0.9:
        msg = "Pass and fail runs cluster separately — trajectory structure predicts outcome."
        color = "var(--success)"
    elif best_purity >= 0.75:
        msg = "Partial clustering by outcome — similar trajectories tend toward similar results."
        color = "var(--accent)"
    else:
        msg = "No outcome clustering — divergence is at individual decision points, not overall structure."
        color = "var(--text-muted)"

    return f'<div class="cluster-interpretation" style="color:{color}">{msg}</div>'


# --- Fork cards ---

def _build_fork_card(finding, number: int | None = None) -> str:
    p_str = f"p={finding.p_value:.3f}" if finding.p_value is not None else ""

    card = '<div class="fork-card">'
    card += '<div class="fork-card-header">'
    if number is not None:
        card += f'<span class="fork-number">{number}</span>'
    card += f'<span class="fork-summary">{finding.summary}</span>'
    card += '</div>'
    if p_str:
        card += f'<div class="fork-pvalue">{p_str}</div>'

    for branch in finding.branches:
        rate = branch.success_rate
        if rate is not None and rate >= 0.6:
            cls = "branch-success"
            rate_cls = "success"
        elif rate is not None and rate <= 0.2:
            cls = "branch-failure"
            rate_cls = "failure"
        else:
            cls = "branch-mixed"
            rate_cls = ""

        rate_str = f"{rate:.0%}" if rate is not None else "?"
        card += f'<div class="branch {cls}">'
        card += f'<div class="branch-header"><span class="branch-rate {rate_cls}">{rate_str}</span> — {branch.run_count} runs chose <code>{branch.value}</code></div>'

        # Windowed trajectory
        fp = branch.fork_position
        window_start = max(0, fp - 4)
        window_end = min(len(branch.steps), fp + 5)
        windowed = branch.steps[window_start:window_end]

        step_parts = []
        for s in windowed:
            if s.is_fork:
                label = f'<span class="fork-step">{s.enriched_name}</span>'
            else:
                label = s.enriched_name
            if s.detail:
                detail = os.path.basename(s.detail) if "/" in s.detail else s.detail
                if len(detail) > 30:
                    detail = detail[:27] + "..."
                label += f' <span style="color:#484f58">{detail}</span>'
            step_parts.append(label)

        if window_start > 0:
            step_parts.insert(0, '<span class="ellipsis">...</span>')
        if window_end < len(branch.steps):
            step_parts.append('<span class="ellipsis">...</span>')

        card += f'<div class="branch-trajectory">{" → ".join(step_parts)}</div>'

        if branch.reasoning:
            card += f'<div class="branch-reasoning">"{branch.reasoning[:200]}"</div>'

        card += f'<div class="branch-run-id">{branch.representative_run_id}</div>'
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
    """Map scipy dendrogram coordinates to SVG path elements."""
    if not icoord:
        return []

    max_d = max(max(d) for d in dcoord) if dcoord else 1.0
    if max_d == 0:
        max_d = 1.0

    paths = []
    for ic, dc in zip(icoord, dcoord):
        svg_points = []
        for y_sc, x_sc in zip(ic, dc):
            svg_y = (y_sc - 5) / 10 * cell_h + cell_h / 2
            svg_x = dendro_w * (1 - x_sc / max_d)
            svg_points.append((svg_x, svg_y))

        d = (f"M {svg_points[0][0]:.1f},{svg_points[0][1]:.1f} "
             f"L {svg_points[1][0]:.1f},{svg_points[1][1]:.1f} "
             f"L {svg_points[2][0]:.1f},{svg_points[2][1]:.1f} "
             f"L {svg_points[3][0]:.1f},{svg_points[3][1]:.1f}")
        paths.append(f'<path d="{d}" fill="none" stroke="#484f58" stroke-width="1"/>')

    return paths


def _build_dendrogram_heatmap(
    alignment: Alignment,
    runs: list[Run],
    points: list[DivergencePoint],
) -> str:
    if not alignment.matrix or not alignment.matrix[0]:
        return '<p style="color:var(--text-muted)">No alignment data.</p>'

    n_runs = len(alignment.run_ids)
    n_cols = len(alignment.matrix[0])
    success_map = {r.run_id: r.result.success for r in runs}

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

    # Dendrogram ordering
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

            cell_h = max(8, min(16, 500 // max(n_runs, 1)))
            dendro_w = 100
            dendro_paths = _scipy_coords_to_svg(
                result["icoord"], result["dcoord"],
                cell_h, dendro_w,
            )

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

    parts = [f'<svg width="{svg_w}" height="{svg_h}" xmlns="http://www.w3.org/2000/svg" '
             f'style="font-family:\'IBM Plex Mono\',monospace;font-size:9px">']

    heatmap_y_offset = marker_h

    # Dendrogram
    if dendro_paths:
        parts.append(f'<g transform="translate(0,{heatmap_y_offset})">')
        parts.extend(dendro_paths)
        parts.append('</g>')

    # Numbered divergence markers
    div_columns: dict[int, int] = {}
    for i, p in enumerate(points):
        if p.column < n_cols:
            div_columns[p.column] = i + 1
    for col, num in div_columns.items():
        x = heatmap_x + col * cell_w + cell_w // 2
        parts.append(f'<line x1="{x}" y1="{marker_h}" x2="{x}" y2="{svg_h}" stroke="{STEP_COLORS.get("bash(python)", "#d29922")}" stroke-width="0.5" opacity="0.2"/>')
        parts.append(f'<circle cx="{x}" cy="{marker_h // 2 + 1}" r="7" fill="#d29922"/>')
        parts.append(f'<text x="{x}" y="{marker_h // 2 + 4}" text-anchor="middle" fill="#0d1117" font-size="9" font-weight="700">{num}</text>')

    # Rows
    for row_idx, orig_idx in enumerate(leaf_order):
        rid = alignment.run_ids[orig_idx]
        y = heatmap_y_offset + row_idx * cell_h

        s = success_map.get(rid)
        outcome_color = "#3fb950" if s is True else ("#f85149" if s is False else "#30363d")
        ox = dendro_w + gap
        parts.append(f'<rect x="{ox}" y="{y}" width="{outcome_w}" height="{cell_h - 1}" fill="{outcome_color}" rx="1"/>')

        tag = "P" if s is True else ("F" if s is False else "?")
        tag_color = "#3fb950" if s else "#f85149"
        short_id = rid[:18] + ".." if len(rid) > 20 else rid
        lx = dendro_w + gap + outcome_w + gap
        parts.append(f'<text x="{lx}" y="{y + cell_h - 3}" fill="#8b949e" font-size="8">{short_id}</text>')
        parts.append(f'<text x="{heatmap_x - 4}" y="{y + cell_h - 3}" text-anchor="end" fill="{tag_color}" font-size="9" font-weight="600">{tag}</text>')

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
                tooltip = f"{val}"
                if detail:
                    tooltip += f" — {detail}"
                is_div = col in div_columns
                stroke = f' stroke="#d29922" stroke-width="1.5"' if is_div else ""
                parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 1}" height="{cell_h - 1}" fill="{color}" rx="1" opacity="0.85"{stroke}>'
                             f'<title>{tooltip}</title></rect>')
                non_gap_idx += 1

    parts.append('</svg>')
    return '\n'.join(parts)


# --- Patterns table ---

def _build_patterns_table(runs: list[Run]) -> str:
    from moirai.analyze.motifs import find_motifs

    motifs = find_motifs(runs, min_n=3, max_n=5, min_count=3)
    known = [r for r in runs if r.result.success is not None]
    if not known:
        return '<p style="color:var(--text-muted)">No outcome data.</p>'
    baseline = sum(1 for r in known if r.result.success) / len(known)

    positive = [m for m in motifs if m.lift > 1.05][:6]
    negative = [m for m in motifs if m.lift < 0.95][:6]

    if not positive and not negative:
        return f'<p style="color:var(--text-muted)">No significant patterns (baseline: {baseline:.0%}).</p>'

    html = f'<p style="font-size:11px;color:var(--text-muted);margin:0 0 8px">Baseline: {baseline:.0%} across {len(known)} runs</p>'
    html += '<table>'
    html += '<tr><th style="text-align:left">Pattern</th><th style="text-align:right">Success</th><th style="text-align:right">Runs</th><th style="text-align:right">p</th></tr>'

    for m in positive:
        p_str = f"{m.p_value:.3f}" if m.p_value is not None else ""
        html += f'<tr class="positive"><td style="font-size:11px">{m.display}</td>'
        html += f'<td style="text-align:right" class="success"><strong>{m.success_rate:.0%}</strong></td>'
        html += f'<td style="text-align:right;color:var(--text-muted)">{m.total_runs}</td>'
        html += f'<td style="text-align:right;color:#484f58">{p_str}</td></tr>'

    if positive and negative:
        html += '<tr><td colspan="4" style="padding:4px"></td></tr>'

    for m in negative:
        p_str = f"{m.p_value:.3f}" if m.p_value is not None else ""
        html += f'<tr class="negative"><td style="font-size:11px">{m.display}</td>'
        html += f'<td style="text-align:right" class="failure"><strong>{m.success_rate:.0%}</strong></td>'
        html += f'<td style="text-align:right;color:var(--text-muted)">{m.total_runs}</td>'
        html += f'<td style="text-align:right;color:#484f58">{p_str}</td></tr>'

    html += '</table>'
    return html


# --- Clusters and Diff ---

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
    path.write_text(f"<html><body style='background:#0d1117;color:#c9d1d9;font-family:monospace;padding:40px'><h1>moirai</h1><p>{message}</p></body></html>", encoding="utf-8")
