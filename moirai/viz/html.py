from __future__ import annotations

import os
from pathlib import Path
from string import Template

from moirai.compress import compress_phases, NOISE_STEPS
from moirai.schema import (
    Alignment,
    ClusterResult,
    CohortDiff,
    DivergencePoint,
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
# Branch dashboard (main entry point)
# ---------------------------------------------------------------------------

def write_branch_html(
    alignment: Alignment | None,
    points: list[DivergencePoint],
    runs: list[Run],
    path: Path,
    task_results: list | None = None,
) -> Path:
    path = Path(path)
    if not runs:
        _write_empty(path, "No runs to display.")
        return path

    # Build task sections
    task_sections_html = ""
    if task_results:
        from moirai.analyze.narrate import narrate_task

        for tid, task_runs, task_alignment, task_points in task_results:
            if not task_alignment.matrix or not task_alignment.matrix[0]:
                continue
            safe_tid = tid.replace(" ", "_").replace("/", "_")
            findings = narrate_task(tid, task_runs, task_alignment, task_points)
            task_sections_html += _render_task_section(
                tid, safe_tid, task_runs, task_alignment, task_points, findings,
            )

    # Stats
    n_pass = sum(1 for r in runs if r.result.success)
    n_tasks = len(task_results) if task_results else 0
    n_div = sum(len(pts) for _, _, _, pts in task_results) if task_results else len(points)
    pass_rate = f"{n_pass / len(runs):.0%}" if runs else "0%"

    # Legend
    legend_items = "".join(
        f'<span class="legend-item"><span class="legend-swatch" style="background:{color}"></span>{name}</span>'
        for name, color in STEP_LEGEND
    )

    # Patterns
    patterns_html = _build_patterns_table(runs)

    # Load template and substitute
    template_path = _TEMPLATE_DIR / "branch.html"
    tpl = Template(template_path.read_text(encoding="utf-8"))
    html = tpl.safe_substitute(
        total_runs=len(runs),
        pass_rate=pass_rate,
        n_tasks=n_tasks,
        n_div=n_div,
        legend_items=legend_items,
        task_sections=task_sections_html,
        patterns_html=patterns_html,
    )

    path.write_text(html, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Per-task section
# ---------------------------------------------------------------------------

def _render_task_section(
    tid: str,
    safe_tid: str,
    runs: list[Run],
    alignment: Alignment,
    points: list[DivergencePoint],
    findings: list,
) -> str:
    n_pass = sum(1 for r in runs if r.result.success)
    n_fail = len(runs) - n_pass
    n_cols = len(alignment.matrix[0])

    s = f'<div class="task-panel" id="task-{safe_tid}">'
    s += f'<div class="task-header"><span class="task-name">{tid}</span>'
    s += f'<span class="task-meta">{len(runs)} runs '
    s += f'<span class="pass-count">{n_pass}P</span> '
    s += f'<span class="fail-count">{n_fail}F</span> '
    s += f'{n_cols} cols</span></div>'

    s += _clustering_interpretation(alignment, runs)
    s += f'<div class="heatmap-container">{_build_dendrogram_heatmap(alignment, runs, points, safe_tid)}</div>'

    if findings:
        s += '<div class="fork-cards">'
        for i, finding in enumerate(findings, 1):
            s += _build_fork_card(finding, number=i, task_id=safe_tid)
        s += '</div>'

    s += '</div>'
    return s


# ---------------------------------------------------------------------------
# Clustering interpretation
# ---------------------------------------------------------------------------

def _clustering_interpretation(alignment: Alignment, runs: list[Run]) -> str:
    if len(runs) < 3:
        return ""

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

    success_map = {r.run_id: r.result.success for r in runs}
    outcomes = [success_map.get(alignment.run_ids[i]) for i in leaves]
    best_purity = 0.0
    for split in range(1, len(outcomes)):
        left = outcomes[:split]
        right = outcomes[split:]
        lp = sum(1 for o in left if o is True)
        rp = sum(1 for o in right if o is True)
        left_purity = max(lp, len(left) - lp) / len(left)
        right_purity = max(rp, len(right) - rp) / len(right)
        avg = (left_purity * len(left) + right_purity * len(right)) / len(outcomes)
        if avg > best_purity:
            best_purity = avg

    if best_purity >= 0.9:
        msg, color = "Pass and fail runs cluster separately — trajectory structure predicts outcome.", "var(--success)"
    elif best_purity >= 0.75:
        msg, color = "Partial clustering by outcome — similar trajectories tend toward similar results.", "var(--accent)"
    else:
        msg, color = "No outcome clustering — divergence is at individual decision points, not overall structure.", "var(--text-muted)"

    return f'<div class="cluster-interpretation" style="color:{color}">{msg}</div>'


# ---------------------------------------------------------------------------
# Fork cards
# ---------------------------------------------------------------------------

def _build_fork_card(finding, number: int | None = None, task_id: str = "") -> str:
    p_str = f"p={finding.p_value:.3f}" if finding.p_value is not None else ""
    div_col = finding.fork_column

    card_id = f'fork-{task_id}-{number}' if number and task_id else ""
    hover = f'@mouseenter="highlightCol(\'{task_id}\', {div_col})" @mouseleave="clearHighlight()"' if task_id else ""

    s = f'<div class="fork-card" id="{card_id}" {hover}>'
    s += '<div class="fork-card-header">'
    if number is not None and task_id:
        s += f'<span class="fork-number" @click="scrollToHeatmap(\'{task_id}\', {number})">{number}</span>'
    elif number is not None:
        s += f'<span class="fork-number">{number}</span>'
    s += f'<span class="fork-summary">{finding.summary}</span></div>'
    if p_str:
        s += f'<div class="fork-pvalue">{p_str}</div>'

    for branch in finding.branches:
        rate = branch.success_rate
        if rate is not None and rate >= 0.6:
            cls, rate_cls = "branch-success", "success"
        elif rate is not None and rate <= 0.2:
            cls, rate_cls = "branch-failure", "failure"
        else:
            cls, rate_cls = "branch-mixed", ""

        rate_str = f"{rate:.0%}" if rate is not None else "?"
        s += f'<div class="branch {cls}">'
        s += f'<div class="branch-header"><span class="branch-rate {rate_cls}">{rate_str}</span> — {branch.run_count} runs chose <code>{branch.value}</code></div>'

        # Windowed trajectory
        fp = branch.fork_position
        w_start = max(0, fp - 4)
        w_end = min(len(branch.steps), fp + 5)
        windowed = branch.steps[w_start:w_end]

        parts = []
        for step in windowed:
            if step.is_fork:
                label = f'<span class="fork-step">{step.enriched_name}</span>'
            else:
                label = step.enriched_name
            if step.detail:
                detail = os.path.basename(step.detail) if "/" in step.detail else step.detail
                if len(detail) > 30:
                    detail = detail[:27] + "..."
                label += f' <span style="color:#484f58">{detail}</span>'
            parts.append(label)

        if w_start > 0:
            parts.insert(0, '<span style="color:var(--border)">...</span>')
        if w_end < len(branch.steps):
            parts.append('<span style="color:var(--border)">...</span>')

        s += f'<div class="branch-trajectory">{" → ".join(parts)}</div>'

        if branch.reasoning:
            s += f'<div class="branch-reasoning">"{branch.reasoning[:200]}"</div>'

        s += f'<div class="branch-run-id">{branch.representative_run_id}</div>'
        s += '</div>'

    s += '</div>'
    return s


# ---------------------------------------------------------------------------
# Dendrogram + heatmap SVG
# ---------------------------------------------------------------------------

def _scipy_coords_to_svg(
    icoord: list[list[float]],
    dcoord: list[list[float]],
    cell_h: float,
    dendro_w: float,
) -> list[str]:
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


def _build_dendrogram_heatmap(
    alignment: Alignment,
    runs: list[Run],
    points: list[DivergencePoint],
    task_id: str = "",
) -> str:
    if not alignment.matrix or not alignment.matrix[0]:
        return '<p style="color:var(--text-muted)">No alignment data.</p>'

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

    if n_runs >= 2:
        from moirai.analyze.align import distance_matrix
        from scipy.cluster.hierarchy import linkage, dendrogram as scipy_dendrogram

        condensed = distance_matrix(runs, level=alignment.level if alignment.level == "name" else "name")
        if len(condensed) > 0:
            Z = linkage(condensed, method="average")
            result = scipy_dendrogram(Z, no_plot=True)
            leaf_order = list(result["leaves"])
            cell_h_tmp = max(8, min(16, 500 // max(n_runs, 1)))
            dendro_w = 100
            dendro_paths = _scipy_coords_to_svg(result["icoord"], result["dcoord"], cell_h_tmp, dendro_w)

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

    svg_id = f'id="heatmap-{task_id}"' if task_id else ""
    parts = [f'<svg {svg_id} width="{svg_w}" height="{svg_h}" xmlns="http://www.w3.org/2000/svg" '
             f'style="font-family:\'IBM Plex Mono\',monospace;font-size:9px">']

    y_off = marker_h

    # Dendrogram
    if dendro_paths:
        parts.append(f'<g transform="translate(0,{y_off})">')
        parts.extend(dendro_paths)
        parts.append('</g>')

    # Numbered divergence markers
    div_columns: dict[int, int] = {}
    for i, p in enumerate(points):
        if p.column < n_cols:
            div_columns[p.column] = i + 1

    for col, num in div_columns.items():
        x = heatmap_x + col * cell_w + cell_w // 2
        parts.append(f'<line x1="{x}" y1="{marker_h}" x2="{x}" y2="{svg_h}" stroke="#d29922" stroke-width="0.5" opacity="0.2" data-col="{col}"/>')
        click = f' @click="scrollToFork(\'{task_id}\', {num})" style="cursor:pointer"' if task_id else ""
        parts.append(f'<circle cx="{x}" cy="{marker_h // 2 + 1}" r="7" fill="#d29922" data-div-num="{num}"{click}/>')
        parts.append(f'<text x="{x}" y="{marker_h // 2 + 4}" text-anchor="middle" fill="#0d1117" font-size="9" font-weight="700" style="pointer-events:none">{num}</text>')

    # Rows
    for row_idx, orig_idx in enumerate(leaf_order):
        rid = alignment.run_ids[orig_idx]
        y = y_off + row_idx * cell_h
        succ = success_map.get(rid)

        parts.append(f'<g class="heatmap-row">')

        # Outcome strip
        oc = "#3fb950" if succ is True else ("#f85149" if succ is False else "#30363d")
        parts.append(f'<rect x="{dendro_w + gap}" y="{y}" width="{outcome_w}" height="{cell_h - 1}" fill="{oc}" rx="1"/>')

        # Label
        tag = "P" if succ is True else ("F" if succ is False else "?")
        tc = "#3fb950" if succ else "#f85149"
        short_id = rid[:18] + ".." if len(rid) > 20 else rid
        lx = dendro_w + gap + outcome_w + gap
        parts.append(f'<text x="{lx}" y="{y + cell_h - 3}" fill="#8b949e" font-size="8">{short_id}</text>')
        parts.append(f'<text x="{heatmap_x - 4}" y="{y + cell_h - 3}" text-anchor="end" fill="{tc}" font-size="9" font-weight="600">{tag}</text>')

        # Heatmap cells
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
# Patterns table
# ---------------------------------------------------------------------------

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
    html += '<table><tr><th style="text-align:left">Pattern</th><th style="text-align:right">Success</th><th style="text-align:right">Runs</th><th style="text-align:right">p</th></tr>'

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


# ---------------------------------------------------------------------------
# Clusters and Diff (unchanged)
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
