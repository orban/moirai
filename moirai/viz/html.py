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


def write_branch_html(
    alignment: Alignment,
    points: list[DivergencePoint],
    runs: list[Run],
    path: Path,
) -> Path:
    """Write alignment heatmap with divergence markers and compressed labels."""
    path = Path(path)

    if not alignment.matrix or not alignment.matrix[0]:
        _write_empty(path, "No alignment data to display.")
        return path

    success_map = {r.run_id: r.result.success for r in runs}
    n_runs = len(alignment.run_ids)
    n_cols = len(alignment.matrix[0])

    div_cols = {p.column for p in points}

    all_values = set()
    for row in alignment.matrix:
        for v in row:
            all_values.add(v)
    value_to_num = {v: i for i, v in enumerate(sorted(all_values))}

    z = []
    text = []
    for run_idx in range(n_runs):
        row_z = []
        row_text = []
        for col in range(n_cols):
            val = alignment.matrix[run_idx][col] if col < len(alignment.matrix[run_idx]) else GAP
            row_z.append(value_to_num.get(val, 0))
            div_marker = " *" if col in div_cols else ""
            row_text.append(f"{val}{div_marker}")
        z.append(row_z)
        text.append(row_text)

    run_labels = []
    for rid in alignment.run_ids:
        s = success_map.get(rid)
        tag = "PASS" if s is True else ("FAIL" if s is False else "?")
        short_id = rid[:30] + "..." if len(rid) > 30 else rid
        run_labels.append(f"{short_id} [{tag}]")

    fig = go.Figure(data=go.Heatmap(
        z=z,
        text=text,
        texttemplate="%{text}",
        x=[f"{i}" for i in range(n_cols)],
        y=run_labels,
        colorscale="Viridis",
        showscale=False,
        hovertemplate="Run: %{y}<br>Position: %{x}<br>Step: %{text}<extra></extra>",
    ))

    fig.update_layout(
        title="Trajectory Alignment Heatmap (* = significant divergence point)",
        xaxis_title="Aligned Position",
        yaxis_title="Run",
        yaxis=dict(autorange="reversed"),
        height=max(400, n_runs * 25 + 200),
        font=dict(size=10),
    )

    fig.write_html(str(path), include_plotlyjs=True)
    return path


def write_clusters_html(result: ClusterResult, path: Path, runs: list[Run] | None = None) -> Path:
    """Write cluster visualization with compressed labels."""
    path = Path(path)

    if not result.clusters:
        _write_empty(path, "No clusters to display.")
        return path

    sorted_clusters = sorted(result.clusters, key=lambda c: -c.count)

    labels = []
    counts = []
    rates = []
    hover_texts = []

    run_map = {r.run_id: r for r in runs} if runs else {}

    for info in sorted_clusters:
        label = f"C{info.cluster_id} ({info.count})"
        labels.append(label)
        counts.append(info.count)
        rates.append(info.success_rate * 100 if info.success_rate is not None else 0)

        # Build hover with compressed prototype
        if run_map:
            cluster_runs = [run_map[rid] for rid, cid in result.labels.items()
                           if cid == info.cluster_id and rid in run_map]
            if cluster_runs:
                rep = sorted(cluster_runs, key=lambda r: len(r.steps))[len(cluster_runs) // 2]
                hover = compress_phases(rep)
                if len(hover) > 100:
                    hover = hover[:97] + "..."
                hover_texts.append(hover)
            else:
                hover_texts.append("")
        else:
            hover_texts.append("")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Run Count",
        x=labels, y=counts,
        marker_color="steelblue",
        hovertext=hover_texts,
        hovertemplate="%{x}<br>Count: %{y}<br>%{hovertext}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Success %",
        x=labels, y=rates,
        marker_color="seagreen",
        hovertemplate="%{x}<br>Success: %{y:.0f}%<extra></extra>",
    ))

    fig.update_layout(
        title="Cluster Distribution",
        barmode="group",
        xaxis_title="Cluster",
        yaxis_title="Value",
        height=500,
    )

    fig.write_html(str(path), include_plotlyjs=True)
    return path


def write_diff_html(
    diff: CohortDiff,
    a_label: str,
    b_label: str,
    path: Path,
) -> Path:
    """Write cohort comparison with multiple panels."""
    path = Path(path)

    a_rate = diff.a_summary.success_rate
    b_rate = diff.b_summary.success_rate

    if a_rate is None and b_rate is None:
        _write_empty(path, "No success rate data to compare.")
        return path

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Success Rate & Steps", "Cluster Shifts"),
    )

    # Panel 1: metrics comparison
    metrics = ["Success Rate", "Avg Steps"]
    a_vals = [(a_rate or 0) * 100, diff.a_summary.avg_steps]
    b_vals = [(b_rate or 0) * 100, diff.b_summary.avg_steps]

    fig.add_trace(go.Bar(name=f"A: {a_label}", x=metrics, y=a_vals, marker_color="steelblue"), row=1, col=1)
    fig.add_trace(go.Bar(name=f"B: {b_label}", x=metrics, y=b_vals, marker_color="coral"), row=1, col=1)

    # Panel 2: cluster shifts
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
            x=shift_values, y=shift_labels,
            orientation="h",
            marker_color=shift_colors,
            showlegend=False,
            hovertemplate="%{y}<br>Shift: %{x} runs<extra></extra>",
        ), row=1, col=2)

    fig.update_layout(
        title=f"Cohort Comparison: {a_label} vs {b_label}",
        barmode="group",
        height=500,
    )

    fig.write_html(str(path), include_plotlyjs=True)
    return path


def _write_empty(path: Path, message: str) -> None:
    path.write_text(
        f"<html><body><h1>moirai</h1><p>{message}</p></body></html>",
        encoding="utf-8",
    )
