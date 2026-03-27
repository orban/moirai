from __future__ import annotations

from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from moirai.compress import compress_phases
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


def write_branch_html(
    alignment: Alignment,
    points: list[DivergencePoint],
    runs: list[Run],
    path: Path,
) -> Path:
    """Write branch analysis dashboard with trajectory matrix and divergence tree."""
    path = Path(path)

    if not runs:
        _write_empty(path, "No runs to display.")
        return path

    from moirai.analyze.cluster import cluster_runs
    from moirai.analyze.align import align_runs
    from moirai.analyze.divergence import find_divergence_points
    cluster_result = cluster_runs(runs, level="type", threshold=0.3)
    run_map = {r.run_id: r for r in runs}

    # Per-cluster: type-level alignment for matrix, name-level for divergence
    matrix_parts: list[str] = []
    tree_parts: list[str] = []

    for info in sorted(cluster_result.clusters, key=lambda c: -c.count):
        members = [run_map[rid] for rid, cid in cluster_result.labels.items()
                   if cid == info.cluster_id and rid in run_map]
        if len(members) < 3:
            continue

        # Type-level alignment for the visual matrix
        type_alignment = align_runs(members, level="type")
        rate_str = f"{info.success_rate:.0%}" if info.success_rate is not None else "?"
        matrix_parts.append(
            f'<div style="margin-bottom:16px">'
            f'<div style="font-size:13px;font-weight:600;margin-bottom:4px">Cluster {info.cluster_id}: {info.count} runs, {rate_str} success</div>'
            + _build_trajectory_matrix(type_alignment, members)
            + '</div>'
        )

        # Name-level alignment for divergence analysis
        name_alignment = align_runs(members, level="name")
        name_points = find_divergence_points(name_alignment, members)
        if name_points:
            tree_parts.append(
                f'<div style="margin-bottom:20px">'
                f'<div style="font-size:13px;font-weight:600;margin-bottom:4px">Cluster {info.cluster_id} ({info.count} runs, {rate_str} success)</div>'
                + _build_divergence_tree(name_points, name_alignment, members)
                + '</div>'
            )

    matrix_html = '\n'.join(matrix_parts) if matrix_parts else '<p>No clusters with enough runs.</p>'
    tree_html = '\n'.join(tree_parts) if tree_parts else '<p>No significant divergence points found.</p>'
    patterns_html = _build_patterns_table(runs)

    n_pass = sum(1 for r in runs if r.result.success)

    # Type-level legend (simpler, cleaner)
    type_colors = {"tool": "#4e79a7", "llm": "#b07aa1", "system": "#9c755f", "judge": "#59a14f", "error": "#e15759"}
    legend_items = []
    for name, color in type_colors.items():
        legend_items.append(f'<span style="display:inline-flex;align-items:center;gap:3px;margin-right:16px">'
                           f'<span style="width:10px;height:10px;background:{color};border-radius:2px;display:inline-block"></span>'
                           f'<span style="font-size:12px">{name}</span></span>')
    legend_html = '<div style="line-height:2">' + ''.join(legend_items) + '</div>'

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
  .legend {{ margin-bottom: 12px; }}
  .tree-node {{ border: 1px solid #ddd; border-radius: 6px; padding: 10px 14px; margin: 6px 0; display: inline-block; }}
  .tree-branch {{ margin-left: 24px; border-left: 2px solid #ddd; padding-left: 16px; }}
  .tree-label {{ font-family: 'SF Mono', Menlo, monospace; font-size: 13px; font-weight: 600; }}
  .tree-stats {{ font-size: 12px; color: #666; }}
  .tree-context {{ font-family: 'SF Mono', Menlo, monospace; font-size: 11px; color: #999; margin-top: 2px; }}
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
  <div><div class="stat-value">{len(points)}</div><div class="stat-label">divergence points</div></div>
</div>

<div class="panel">
<h2>Trajectory alignment</h2>
<p style="font-size:13px;color:#666;margin-top:0">Every run as a row, aligned by sequence structure. Sorted by cluster, then outcome. Columns with red markers are statistically significant divergence points.</p>
<div class="legend">{legend_html}</div>
{matrix_html}
</div>

<div class="panel">
<h2>Where trajectories diverge</h2>
<p style="font-size:13px;color:#666;margin-top:0">At each divergence point, which choice did agents make and what happened?</p>
{tree_html}
</div>

<div class="panel">
<h2>Patterns that predict failure</h2>
<p style="font-size:13px;color:#666;margin-top:0">Step sequences significantly correlated with success or failure.</p>
{patterns_html}
</div>

</body>
</html>"""

    path.write_text(html, encoding="utf-8")
    return path


TYPE_COLORS = {"tool": "#4e79a7", "llm": "#b07aa1", "system": "#9c755f", "judge": "#59a14f", "error": "#e15759"}


def _build_trajectory_matrix(alignment: Alignment, runs: list[Run]) -> str:
    """Build an SVG trajectory alignment matrix for one cluster.

    Each row = one run. Each column = one aligned position.
    Cells colored by step type. Sorted by pass first, then by gap count.
    """
    if not alignment.matrix or not alignment.matrix[0]:
        return "<p>No alignment data.</p>"

    success_map = {r.run_id: r.result.success for r in runs}
    n_runs = len(alignment.run_ids)
    n_cols = len(alignment.matrix[0])

    # Sort: pass first, then by gap count (densest first)
    indices = list(range(n_runs))
    indices.sort(key=lambda i: (
        not (success_map.get(alignment.run_ids[i]) is True),
        sum(1 for v in alignment.matrix[i] if v == GAP),
    ))

    cell_w = max(8, min(16, 800 // max(n_cols, 1)))
    cell_h = max(6, min(14, 400 // max(n_runs, 1)))
    label_w = 180
    svg_w = label_w + n_cols * cell_w + 10
    svg_h = n_runs * cell_h + 4

    parts = [f'<svg width="{svg_w}" height="{svg_h}" xmlns="http://www.w3.org/2000/svg" '
             f'style="font-family:SF Mono,Menlo,monospace;font-size:9px">']

    for row_idx, orig_idx in enumerate(indices):
        rid = alignment.run_ids[orig_idx]
        y = row_idx * cell_h
        s = success_map.get(rid)
        tag = "P" if s is True else ("F" if s is False else "?")
        tag_color = "#2d7d2d" if s else "#c0392b"

        short_id = rid[:22] + ".." if len(rid) > 24 else rid
        parts.append(f'<text x="{label_w - 20}" y="{y + cell_h - 2}" text-anchor="end" fill="#888" font-size="8">{short_id}</text>')
        parts.append(f'<text x="{label_w - 6}" y="{y + cell_h - 2}" text-anchor="end" fill="{tag_color}" font-size="9" font-weight="bold">{tag}</text>')

        row = alignment.matrix[orig_idx]
        for col in range(n_cols):
            val = row[col] if col < len(row) else GAP
            x = label_w + col * cell_w

            if val == GAP:
                parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="#f0f0f0"/>')
            else:
                color = TYPE_COLORS.get(val, DEFAULT_COLOR)
                parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 1}" height="{cell_h - 1}" fill="{color}" rx="1">'
                             f'<title>col {col}: {val} ({rid[:30]})</title></rect>')

    parts.append('</svg>')
    return '\n'.join(parts)


def _build_divergence_tree(
    points: list[DivergencePoint],
    alignment: Alignment,
    runs: list[Run],
) -> str:
    """Build a divergence tree showing branching at each significant point."""
    if not points:
        return "<p>No significant divergence points found.</p>"

    html = ""
    for point in points[:5]:
        p_str = f"p={point.p_value:.3f}" if point.p_value is not None else ""
        phase = point.phase_context or ""

        total_runs = sum(point.value_counts.values())

        html += f'<div style="margin-bottom:20px">'
        html += f'<div style="font-size:14px;font-weight:600;margin-bottom:4px">Position {point.column} <span style="color:#888;font-weight:normal;font-size:12px">({p_str})</span></div>'
        if phase:
            html += f'<div style="font-size:11px;color:#888;margin-bottom:8px">{phase}</div>'

        # Sort branches: best success rate first
        sorted_branches = sorted(
            point.value_counts.items(),
            key=lambda x: -(point.success_by_value.get(x[0]) or 0)
        )

        for value, count in sorted_branches:
            rate = point.success_by_value.get(value)
            pct = count / total_runs * 100

            if rate is not None and rate >= 0.6:
                bar_color = "#2d7d2d"
                rate_class = "success"
            elif rate is not None and rate <= 0.2:
                bar_color = "#c0392b"
                rate_class = "failure"
            else:
                bar_color = "#b07800"
                rate_class = "mixed"

            rate_str = f"{rate:.0%}" if rate is not None else "?"
            color = STEP_COLORS.get(value, DEFAULT_COLOR)

            # Context from alignment
            context = _get_context_str(point.column, value, alignment)

            html += '<div class="tree-branch">'
            html += f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0">'
            html += f'<span style="width:10px;height:10px;background:{color};border-radius:2px;display:inline-block"></span>'
            html += f'<span class="tree-label">{value}</span>'

            # Proportional bar
            bar_w = max(pct * 3, 8)
            html += f'<svg width="{bar_w + 2}" height="16"><rect x="0" y="2" width="{bar_w}" height="12" fill="{bar_color}" rx="2" opacity="0.8"/></svg>'

            html += f'<span class="tree-stats">{count} runs, <span class="{rate_class}">{rate_str} success</span></span>'
            html += '</div>'

            if context:
                html += f'<div class="tree-context">{context}</div>'
            html += '</div>'

        html += '</div>'

    return html


def _get_context_str(col: int, value: str, alignment: Alignment) -> str:
    """Get context window string for a branch."""
    if not alignment.matrix or not alignment.run_ids:
        return ""
    for run_idx, rid in enumerate(alignment.run_ids):
        if run_idx < len(alignment.matrix) and col < len(alignment.matrix[run_idx]):
            if alignment.matrix[run_idx][col] == value:
                row = alignment.matrix[run_idx]
                start = max(0, col - 3)
                end = min(len(row), col + 4)
                parts = []
                for i in range(start, end):
                    v = row[i] if i < len(row) else GAP
                    if v == GAP:
                        continue
                    if i == col:
                        parts.append(f"[{v}]")
                    else:
                        parts.append(v)
                return " → ".join(parts)
    return ""


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
