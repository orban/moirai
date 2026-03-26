from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from moirai.compress import compress_phases, NOISE_STEPS
from moirai.schema import (
    Alignment,
    ClusterResult,
    CohortDiff,
    DivergencePoint,
    GAP,
    Run,
)

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


def write_branch_html(
    alignment: Alignment,
    points: list[DivergencePoint],
    runs: list[Run],
    path: Path,
) -> Path:
    """Write insight-driven branch analysis dashboard.

    Panel 1: Cluster profiles — one bar per behavioral mode
    Panel 2: Discriminative patterns — step sequences that predict outcome
    Panel 3: Divergence cards — key decision points with branches and success rates
    """
    path = Path(path)

    if not runs:
        _write_empty(path, "No runs to display.")
        return path

    # Cluster for the profiles panel
    from moirai.analyze.cluster import cluster_runs
    cluster_result = cluster_runs(runs, level="type", threshold=0.3)

    profiles_html = _build_cluster_profiles(cluster_result, runs)
    patterns_html = _build_patterns_table(runs)
    divergence_html = _build_divergence_cards(points, alignment, runs)
    recs_html = _build_recommendations(runs, points)

    n_pass = sum(1 for r in runs if r.result.success)
    n_div = len(points)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>moirai — branch analysis</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 20px; background: #fafafa; color: #333; }}
  h1 {{ color: #222; margin-bottom: 5px; }}
  h2 {{ color: #444; margin-top: 0; }}
  .panel {{ background: white; border-radius: 8px; padding: 24px; margin: 20px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .stats {{ display: flex; gap: 40px; margin: 15px 0 25px 0; }}
  .stat {{ text-align: center; }}
  .stat-value {{ font-size: 28px; font-weight: bold; color: #222; }}
  .stat-label {{ font-size: 12px; color: #999; text-transform: uppercase; letter-spacing: 0.5px; }}
  .card {{ border: 1px solid #e0e0e0; border-radius: 6px; padding: 16px; margin: 12px 0; }}
  .card-header {{ font-weight: 600; margin-bottom: 8px; }}
  .branch {{ display: flex; align-items: center; gap: 12px; padding: 6px 0; }}
  .branch-bar {{ height: 24px; border-radius: 3px; min-width: 4px; }}
  .branch-label {{ font-size: 13px; min-width: 100px; }}
  .branch-stats {{ font-size: 12px; color: #888; }}
  .context {{ font-family: monospace; font-size: 12px; color: #666; margin-top: 4px; padding: 4px 8px; background: #f8f8f8; border-radius: 3px; }}
  .phase-context {{ font-size: 12px; color: #888; margin-bottom: 8px; }}
  .rec {{ border-left: 4px solid #4e79a7; padding: 12px 16px; margin: 12px 0; background: #f8fafc; }}
  .rec-action {{ font-size: 15px; font-weight: 600; color: #222; margin-bottom: 4px; }}
  .rec-impact {{ font-size: 13px; color: #555; margin-bottom: 6px; }}
  .rec-evidence {{ font-size: 12px; color: #888; }}
  .rec-evidence code {{ background: #eef; padding: 1px 4px; border-radius: 2px; }}
</style>
</head>
<body>
<h1>moirai branch analysis</h1>
<div class="stats">
  <div class="stat"><div class="stat-value">{len(runs)}</div><div class="stat-label">runs</div></div>
  <div class="stat"><div class="stat-value">{n_pass}/{len(runs)}</div><div class="stat-label">pass / total</div></div>
  <div class="stat"><div class="stat-value">{n_div}</div><div class="stat-label">divergence points</div></div>
</div>

<div class="panel">
<h2>Recommendations</h2>
<p>What to change based on this analysis, ranked by expected impact.</p>
{recs_html}
</div>

<div class="panel">
<h2>Behavioral Modes</h2>
<p>Each bar is a cluster of runs that follow a similar trajectory. Width = run count, color = success rate.</p>
{profiles_html}
</div>

<div class="panel">
<h2>Discriminative Patterns</h2>
<p>Step sequences that predict success or failure, sorted by deviation from baseline.</p>
{patterns_html}
</div>

<div class="panel">
<h2>Key Decision Points</h2>
<p>Positions in the trajectory where the agent's choice significantly affects outcome.</p>
{divergence_html}
</div>

</body>
</html>"""

    path.write_text(html, encoding="utf-8")
    return path


def _build_cluster_profiles(cluster_result: ClusterResult, runs: list[Run]) -> str:
    """Build cluster profile bars — one per behavioral mode."""
    from moirai.compress import step_phase, compress_run

    if not cluster_result.clusters:
        return "<p>No clusters found.</p>"

    run_map = {r.run_id: r for r in runs}
    sorted_clusters = sorted(cluster_result.clusters, key=lambda c: -c.count)

    # Build a horizontal bar chart: each bar is a cluster
    labels = []
    widths = []
    colors = []
    hover_texts = []

    for info in sorted_clusters:
        rate = info.success_rate or 0
        members = [run_map[rid] for rid, cid in cluster_result.labels.items()
                    if cid == info.cluster_id and rid in run_map]

        # Representative compressed trajectory
        if members:
            rep = sorted(members, key=lambda r: len(r.steps))[len(members) // 2]
            compressed = compress_run(rep)
            if len(compressed) > 60:
                compressed = compressed[:57] + "..."
            phases = compress_phases(rep)
            if len(phases) > 60:
                phases = phases[:57] + "..."
        else:
            compressed = ""
            phases = ""

        labels.append(f"{info.count} runs, {rate:.0%} success")
        widths.append(info.count)
        # Color by success rate: green for high, red for low
        r_val = int(200 * (1 - rate)) + 55
        g_val = int(200 * rate) + 55
        colors.append(f"rgb({r_val},{g_val},100)")
        hover_texts.append(f"{compressed}<br>{phases}<br>{info.count} runs, {rate:.0%} success")

    fig = go.Figure(go.Bar(
        y=labels,
        x=widths,
        orientation="h",
        marker_color=colors,
        hovertext=hover_texts,
        hovertemplate="%{hovertext}<extra></extra>",
    ))

    fig.update_layout(
        height=max(200, len(sorted_clusters) * 45 + 80),
        xaxis_title="Number of runs",
        margin=dict(l=200, r=20, t=10, b=40),
        showlegend=False,
    )

    return fig.to_html(include_plotlyjs=True, full_html=False)


def _build_patterns_table(runs: list[Run]) -> str:
    """Build an HTML table of discriminative step patterns."""
    from moirai.analyze.motifs import find_motifs

    motifs = find_motifs(runs, min_n=3, max_n=5, min_count=3)

    known = [r for r in runs if r.result.success is not None]
    if not known:
        return "<p>No outcome data for pattern analysis.</p>"
    baseline = sum(1 for r in known if r.result.success) / len(known)

    positive = [m for m in motifs if m.lift > 1.05][:8]
    negative = [m for m in motifs if m.lift < 0.95][:8]

    if not positive and not negative:
        return f"<p>No significant patterns found (baseline: {baseline:.0%} success).</p>"

    html = f'<p style="color:#888">Baseline success rate: {baseline:.0%} across {len(known)} runs</p>'
    html += '<table style="width:100%; border-collapse:collapse; font-size:14px">'
    html += '<tr style="border-bottom:2px solid #ddd"><th style="text-align:left;padding:8px">Pattern</th>'
    html += '<th style="text-align:right;padding:8px">Success</th>'
    html += '<th style="text-align:right;padding:8px">Runs</th>'
    html += '<th style="text-align:right;padding:8px">p-value</th>'
    html += '<th style="text-align:left;padding:8px">Position</th></tr>'

    for m in positive:
        pos = "early" if m.avg_position < 0.3 else ("late" if m.avg_position > 0.7 else "mid")
        p_str = f"{m.p_value:.3f}" if m.p_value is not None else ""
        html += f'<tr style="border-bottom:1px solid #eee;background:#f0faf0">'
        html += f'<td style="padding:8px;font-family:monospace">{m.display}</td>'
        html += f'<td style="text-align:right;padding:8px;color:#2d7d2d;font-weight:bold">{m.success_rate:.0%}</td>'
        html += f'<td style="text-align:right;padding:8px">{m.total_runs}</td>'
        html += f'<td style="text-align:right;padding:8px;color:#888">{p_str}</td>'
        html += f'<td style="padding:8px;color:#888">{pos}</td></tr>'

    if positive and negative:
        html += '<tr><td colspan="5" style="padding:4px"></td></tr>'

    for m in negative:
        pos = "early" if m.avg_position < 0.3 else ("late" if m.avg_position > 0.7 else "mid")
        p_str = f"{m.p_value:.3f}" if m.p_value is not None else ""
        html += f'<tr style="border-bottom:1px solid #eee;background:#faf0f0">'
        html += f'<td style="padding:8px;font-family:monospace">{m.display}</td>'
        html += f'<td style="text-align:right;padding:8px;color:#c0392b;font-weight:bold">{m.success_rate:.0%}</td>'
        html += f'<td style="text-align:right;padding:8px">{m.total_runs}</td>'
        html += f'<td style="text-align:right;padding:8px;color:#888">{p_str}</td>'
        html += f'<td style="padding:8px;color:#888">{pos}</td></tr>'

    html += '</table>'
    return html


def _build_divergence_cards(
    points: list[DivergencePoint],
    alignment: Alignment,
    runs: list[Run],
) -> str:
    """Build divergence decision point cards."""
    if not points:
        return "<p>No significant divergence points found.</p>"

    run_map = {r.run_id: r for r in runs}
    html = ""

    for point in points[:5]:
        p_str = f"p={point.p_value:.3f}" if point.p_value is not None else ""
        phase_str = point.phase_context or ""

        html += f'<div class="card">'
        html += f'<div class="card-header">Position {point.column} <span style="color:#888;font-weight:normal">({p_str})</span></div>'
        if phase_str:
            html += f'<div class="phase-context">{phase_str}</div>'

        # Show branches as proportional bars
        total = sum(point.value_counts.values())
        sorted_branches = sorted(point.value_counts.items(), key=lambda x: -x[1])

        for value, count in sorted_branches:
            rate = point.success_by_value.get(value)
            pct = count / total * 100
            rate_str = f"{rate:.0%}" if rate is not None else "N/A"

            if rate is not None and rate >= 0.7:
                bar_color = "#59a14f"
            elif rate is not None and rate <= 0.3:
                bar_color = "#e15759"
            else:
                bar_color = "#bab0ac"

            # Context window from alignment
            context = _get_context_str(point.column, value, alignment)

            html += f'<div class="branch">'
            html += f'<div class="branch-label"><strong>{value}</strong></div>'
            html += f'<div class="branch-bar" style="width:{max(pct, 3):.0f}%;background:{bar_color}"></div>'
            html += f'<div class="branch-stats">{count} runs, {rate_str} success</div>'
            html += f'</div>'
            if context:
                html += f'<div class="context">{context}</div>'

        html += '</div>'

    return html


def _get_context_str(col: int, value: str, alignment: Alignment) -> str:
    """Get a context window string for a branch at a divergence point."""
    if not alignment.matrix or not alignment.run_ids:
        return ""

    # Find a run that has this value at this column
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


def _build_recommendations(runs: list[Run], points: list[DivergencePoint]) -> str:
    """Build recommendations section from analysis synthesis."""
    from moirai.analyze.motifs import find_motifs
    from moirai.analyze.recommend import synthesize

    motifs = find_motifs(runs, min_n=3, max_n=5, min_count=3)
    known = [r for r in runs if r.result.success is not None]
    if not known:
        return "<p>No outcome data for recommendations.</p>"
    baseline = sum(1 for r in known if r.result.success) / len(known)

    recs = synthesize(motifs, points, len(runs), baseline)
    if not recs:
        return "<p>Not enough data to generate recommendations.</p>"

    html = ""
    for i, rec in enumerate(recs, 1):
        html += f'<div class="rec">'
        html += f'<div class="rec-action">{i}. {rec.action}</div>'
        html += f'<div class="rec-impact">{rec.impact}</div>'
        html += f'<div class="rec-evidence">Evidence: {"; ".join(rec.evidence)}</div>'
        html += f'</div>'

    return html


# --- Clusters and Diff HTML ---

def write_clusters_html(result: ClusterResult, path: Path, runs: list[Run] | None = None) -> Path:
    """Write cluster visualization."""
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
                hover_texts.append(compress_phases(rep)[:100])
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
    """Write cohort comparison."""
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
