from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from moirai.compress import compress_phases, phase_summary, NOISE_STEPS
from moirai.schema import (
    Alignment,
    ClusterResult,
    CohortDiff,
    DivergencePoint,
    GAP,
    Run,
)

# Phase colors
PHASE_COLORS = {
    "explore": "#4e79a7",
    "modify": "#f28e2b",
    "verify": "#59a14f",
    "execute": "#e15759",
    "think": "#b07aa1",
    "system": "#9c755f",
    "error": "#ff4136",
    "act": "#bab0ac",
    "other": "#76b7b2",
}

STEP_COLORS = {
    "read": "#4e79a7",
    "search": "#76b7b2",
    "edit": "#f28e2b",
    "write": "#edc948",
    "test_result": "#59a14f",
    "test": "#59a14f",
    "bash": "#e15759",
    "reason": "#b07aa1",
    "subagent": "#ff9da7",
    "plan": "#9c755f",
    "result": "#bab0ac",
}


def write_branch_html(
    alignment: Alignment,
    points: list[DivergencePoint],
    runs: list[Run],
    path: Path,
) -> Path:
    """Write a multi-panel branch analysis dashboard.

    Panel 1: Sankey diagram — step transitions colored by success rate
    Panel 2: Alignment heatmap — genome-browser-style alignment viewer
    Panel 3: Phase timeline — horizontal bars colored by phase
    """
    path = Path(path)

    if not runs:
        _write_empty(path, "No runs to display.")
        return path

    sankey_html = _build_sankey(runs)
    heatmap_html = _build_alignment_heatmap(alignment, points, runs)
    timeline_html = _build_phase_timeline(runs)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>moirai — branch analysis</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 20px; background: #fafafa; }}
  h1 {{ color: #333; }}
  h2 {{ color: #555; margin-top: 30px; }}
  .panel {{ background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .stats {{ display: flex; gap: 30px; margin: 15px 0; }}
  .stat {{ text-align: center; }}
  .stat-value {{ font-size: 24px; font-weight: bold; color: #333; }}
  .stat-label {{ font-size: 12px; color: #888; }}
  .legend {{ display: flex; gap: 15px; flex-wrap: wrap; margin: 10px 0; }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 12px; }}
  .legend-color {{ width: 12px; height: 12px; border-radius: 2px; }}
</style>
</head>
<body>
<h1>moirai branch analysis</h1>
<div class="stats">
  <div class="stat"><div class="stat-value">{len(runs)}</div><div class="stat-label">runs</div></div>
  <div class="stat"><div class="stat-value">{sum(1 for r in runs if r.result.success)}/{len(runs)}</div><div class="stat-label">pass/total</div></div>
  <div class="stat"><div class="stat-value">{len(points)}</div><div class="stat-label">divergence points</div></div>
</div>

<div class="panel">
<h2>Phase Timeline</h2>
<p>Each row is a run, colored by phase. Sorted by outcome then length. Vertical red lines mark divergence points.</p>
{timeline_html}
</div>

<div class="panel">
<h2>Step Flow</h2>
<p>How runs transition between steps. Width = frequency, color = success rate of runs taking that path.</p>
{sankey_html}
</div>

<div class="panel">
<h2>Alignment</h2>
<p>Aligned step sequences. Each row is a run, each column is an aligned position. * marks significant divergence.</p>
{heatmap_html}
</div>

</body>
</html>"""

    path.write_text(html, encoding="utf-8")
    return path


def _build_phase_timeline(runs: list[Run]) -> str:
    """Build a phase timeline: horizontal bars colored by phase, sorted by outcome."""
    from moirai.compress import step_phase

    sorted_runs = sorted(runs, key=lambda r: (not r.result.success, len(r.steps)))

    fig = go.Figure()

    for run in sorted_runs:
        phases: list[tuple[str, int, int]] = []  # (phase, start, end)
        filtered_idx = 0
        for step in run.steps:
            if step.name in NOISE_STEPS:
                continue
            phase = step_phase(step)
            if phases and phases[-1][0] == phase:
                phases[-1] = (phase, phases[-1][1], filtered_idx + 1)
            else:
                phases.append((phase, filtered_idx, filtered_idx + 1))
            filtered_idx += 1

        success = run.result.success
        tag = "PASS" if success else ("FAIL" if success is False else "?")
        short_id = run.run_id[:25] + "..." if len(run.run_id) > 25 else run.run_id

        for phase, start, end in phases:
            color = PHASE_COLORS.get(phase, "#bab0ac")
            fig.add_trace(go.Bar(
                y=[f"{short_id} [{tag}]"],
                x=[end - start],
                base=start,
                orientation="h",
                marker_color=color,
                showlegend=False,
                hovertemplate=f"{phase} (steps {start}-{end})<br>{short_id}<extra></extra>",
            ))

    # Divergence points live on the alignment coordinate system, not the per-run
    # step position system. Don't plot them here — they're shown on the heatmap.

    fig.update_layout(
        barmode="stack",
        height=max(300, len(sorted_runs) * 22 + 100),
        xaxis_title="Step position (filtered)",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=200),
        showlegend=False,
    )

    # Add a manual legend for phases
    legend_html = '<div class="legend">'
    for phase, color in PHASE_COLORS.items():
        if phase in ("act", "other"):
            continue
        legend_html += f'<div class="legend-item"><div class="legend-color" style="background:{color}"></div>{phase}</div>'
    legend_html += '</div>'

    return legend_html + fig.to_html(include_plotlyjs=True, full_html=False)


def _build_sankey(runs: list[Run]) -> str:
    """Build a Sankey diagram of step transitions."""
    # Count transitions between consecutive filtered steps
    transitions: dict[tuple[str, str], tuple[int, int]] = {}  # (from, to) -> (total, successes)

    for run in runs:
        filtered = [s.name for s in run.steps if s.name not in NOISE_STEPS]
        success = run.result.success is True

        for i in range(len(filtered) - 1):
            key = (filtered[i], filtered[i + 1])
            t, s = transitions.get(key, (0, 0))
            transitions[key] = (t + 1, s + (1 if success else 0))

    if not transitions:
        return "<p>No transitions to display.</p>"

    # Build node list
    all_steps = set()
    for (a, b) in transitions:
        all_steps.add(a)
        all_steps.add(b)
    node_list = sorted(all_steps)
    node_idx = {name: i for i, name in enumerate(node_list)}

    # Build links
    sources = []
    targets = []
    values = []
    colors = []
    hover = []

    for (src, tgt), (total, successes) in sorted(transitions.items(), key=lambda x: -x[1][0]):
        if total < 2:
            continue  # skip rare transitions
        rate = successes / total if total > 0 else 0
        sources.append(node_idx[src])
        targets.append(node_idx[tgt])
        values.append(total)

        # Color: green for high success, red for low
        r = int(255 * (1 - rate))
        g = int(255 * rate)
        colors.append(f"rgba({r},{g},100,0.4)")
        hover.append(f"{src} → {tgt}: {total} runs, {rate:.0%} success")

    node_colors = [STEP_COLORS.get(name, "#bab0ac") for name in node_list]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20,
            label=node_list,
            color=node_colors,
        ),
        link=dict(
            source=sources, target=targets, value=values,
            color=colors,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover,
        ),
    )])

    fig.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20))
    return fig.to_html(include_plotlyjs=False, full_html=False)


def _build_alignment_heatmap(
    alignment: Alignment,
    points: list[DivergencePoint],
    runs: list[Run],
) -> str:
    """Build alignment heatmap — genome-browser-style."""
    if not alignment.matrix or not alignment.matrix[0]:
        return "<p>No alignment data.</p>"

    success_map = {r.run_id: r.result.success for r in runs}
    n_runs = len(alignment.run_ids)
    n_cols = len(alignment.matrix[0])

    div_cols = {p.column for p in points}

    # Assign colors by step name/type
    all_values = sorted(set(v for row in alignment.matrix for v in row))
    value_to_num = {v: i for i, v in enumerate(all_values)}

    z = []
    text = []
    for run_idx in range(n_runs):
        row_z = []
        row_text = []
        for col in range(n_cols):
            val = alignment.matrix[run_idx][col] if col < len(alignment.matrix[run_idx]) else GAP
            row_z.append(value_to_num.get(val, 0))
            marker = " *" if col in div_cols else ""
            row_text.append(f"{val}{marker}")
        z.append(row_z)
        text.append(row_text)

    # Sort by success then length for readability
    indices = list(range(n_runs))
    indices.sort(key=lambda i: (
        not (success_map.get(alignment.run_ids[i]) is True),
        sum(1 for v in alignment.matrix[i] if v == GAP),
    ))

    z = [z[i] for i in indices]
    text = [text[i] for i in indices]
    sorted_ids = [alignment.run_ids[i] for i in indices]

    run_labels = []
    for rid in sorted_ids:
        s = success_map.get(rid)
        tag = "PASS" if s is True else ("FAIL" if s is False else "?")
        short = rid[:20] + "..." if len(rid) > 20 else rid
        run_labels.append(f"{short} [{tag}]")

    fig = go.Figure(data=go.Heatmap(
        z=z, text=text, texttemplate="%{text}",
        x=[f"{i}{'*' if i in div_cols else ''}" for i in range(n_cols)],
        y=run_labels,
        colorscale="Viridis",
        showscale=False,
        hovertemplate="Run: %{y}<br>Position: %{x}<br>Step: %{text}<extra></extra>",
    ))

    fig.update_layout(
        height=max(300, n_runs * 20 + 100),
        xaxis_title="Aligned position (* = divergence)",
        yaxis=dict(autorange="reversed"),
        font=dict(size=9),
        margin=dict(l=180),
    )

    return fig.to_html(include_plotlyjs=False, full_html=False)


def write_clusters_html(result: ClusterResult, path: Path, runs: list[Run] | None = None) -> Path:
    """Write cluster visualization with compressed labels."""
    path = Path(path)

    if not result.clusters:
        _write_empty(path, "No clusters to display.")
        return path

    sorted_clusters = sorted(result.clusters, key=lambda c: -c.count)
    run_map = {r.run_id: r for r in runs} if runs else {}

    labels = []
    counts = []
    rates = []
    hover_texts = []

    for info in sorted_clusters:
        labels.append(f"C{info.cluster_id} ({info.count})")
        counts.append(info.count)
        rates.append(info.success_rate * 100 if info.success_rate is not None else 0)

        if run_map:
            cluster_runs = [run_map[rid] for rid, cid in result.labels.items()
                           if cid == info.cluster_id and rid in run_map]
            if cluster_runs:
                rep = sorted(cluster_runs, key=lambda r: len(r.steps))[len(cluster_runs) // 2]
                hover = compress_phases(rep)
                hover_texts.append(hover[:100])
            else:
                hover_texts.append("")
        else:
            hover_texts.append("")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Run Count", x=labels, y=counts, marker_color="steelblue",
        hovertext=hover_texts,
        hovertemplate="%{x}<br>Count: %{y}<br>%{hovertext}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Success %", x=labels, y=rates, marker_color="seagreen",
        hovertemplate="%{x}<br>Success: %{y:.0f}%<extra></extra>",
    ))

    fig.update_layout(title="Cluster Distribution", barmode="group",
                      xaxis_title="Cluster", yaxis_title="Value", height=500)
    fig.write_html(str(path), include_plotlyjs=True)
    return path


def write_diff_html(diff: CohortDiff, a_label: str, b_label: str, path: Path) -> Path:
    """Write cohort comparison with two panels."""
    path = Path(path)

    a_rate = diff.a_summary.success_rate
    b_rate = diff.b_summary.success_rate

    if a_rate is None and b_rate is None:
        _write_empty(path, "No success rate data to compare.")
        return path

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Success Rate & Steps", "Cluster Shifts"))

    metrics = ["Success Rate", "Avg Steps"]
    a_vals = [(a_rate or 0) * 100, diff.a_summary.avg_steps]
    b_vals = [(b_rate or 0) * 100, diff.b_summary.avg_steps]

    fig.add_trace(go.Bar(name=f"A: {a_label}", x=metrics, y=a_vals, marker_color="steelblue"), row=1, col=1)
    fig.add_trace(go.Bar(name=f"B: {b_label}", x=metrics, y=b_vals, marker_color="coral"), row=1, col=1)

    if diff.cluster_shifts:
        from moirai.viz.terminal import _compress_prototype
        shift_labels = []
        shift_values = []
        shift_colors = []
        for proto, delta in diff.cluster_shifts[:10]:
            compressed = _compress_prototype(proto)
            if len(compressed) > 30:
                compressed = compressed[:27] + "..."
            shift_labels.append(compressed)
            shift_values.append(delta)
            shift_colors.append("seagreen" if delta > 0 else "indianred")

        fig.add_trace(go.Bar(
            x=shift_values, y=shift_labels, orientation="h",
            marker_color=shift_colors, showlegend=False,
            hovertemplate="%{y}<br>Shift: %{x} runs<extra></extra>",
        ), row=1, col=2)

    fig.update_layout(title=f"Cohort Comparison: {a_label} vs {b_label}", barmode="group", height=500)
    fig.write_html(str(path), include_plotlyjs=True)
    return path


def _write_empty(path: Path, message: str) -> None:
    path.write_text(f"<html><body><h1>moirai</h1><p>{message}</p></body></html>", encoding="utf-8")
