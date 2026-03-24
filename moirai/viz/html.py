from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

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
    """Write a colored alignment heatmap with divergence annotations."""
    path = Path(path)

    if not alignment.matrix or not alignment.matrix[0]:
        _write_empty(path, "No alignment data to display.")
        return path

    # Build success map
    success_map = {r.run_id: r.result.success for r in runs}
    n_runs = len(alignment.run_ids)
    n_cols = len(alignment.matrix[0])

    # Build divergence column set for annotation
    div_cols = {p.column for p in points}

    # Assign numeric codes: each unique value gets a number
    all_values = set()
    for row in alignment.matrix:
        for v in row:
            all_values.add(v)
    value_to_num = {v: i for i, v in enumerate(sorted(all_values))}

    # Build z-matrix and text-matrix
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

    # Labels
    run_labels = []
    for rid in alignment.run_ids:
        s = success_map.get(rid)
        tag = "PASS" if s is True else ("FAIL" if s is False else "?")
        run_labels.append(f"{rid} [{tag}]")

    fig = go.Figure(data=go.Heatmap(
        z=z,
        text=text,
        texttemplate="%{text}",
        x=[f"col {i}" for i in range(n_cols)],
        y=run_labels,
        colorscale="Viridis",
        showscale=False,
    ))

    fig.update_layout(
        title="Trajectory Alignment Heatmap",
        xaxis_title="Aligned Position",
        yaxis_title="Run",
        yaxis=dict(autorange="reversed"),
        height=max(400, n_runs * 30 + 200),
    )

    fig.write_html(str(path), include_plotlyjs=True)
    return path


def write_clusters_html(result: ClusterResult, path: Path) -> Path:
    """Write a grouped bar chart of cluster sizes and success rates."""
    path = Path(path)

    if not result.clusters:
        _write_empty(path, "No clusters to display.")
        return path

    labels = []
    counts = []
    rates = []

    for info in result.clusters:
        # Type-only label
        types_only = " > ".join(
            part.split(":")[0] for part in info.prototype.split(" > ")
        ) if info.prototype else f"Cluster {info.cluster_id}"
        labels.append(types_only)
        counts.append(info.count)
        rates.append(info.success_rate * 100 if info.success_rate is not None else 0)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Count", x=labels, y=counts, marker_color="steelblue"))
    fig.add_trace(go.Bar(name="Success %", x=labels, y=rates, marker_color="seagreen"))

    fig.update_layout(
        title="Cluster Distribution",
        barmode="group",
        xaxis_title="Cluster Prototype",
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
    """Write a success rate comparison bar chart."""
    path = Path(path)

    a_rate = diff.a_summary.success_rate
    b_rate = diff.b_summary.success_rate

    if a_rate is None and b_rate is None:
        _write_empty(path, "No success rate data to compare.")
        return path

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=f"A: {a_label}",
        x=["Success Rate", "Avg Steps"],
        y=[
            (a_rate or 0) * 100,
            diff.a_summary.avg_steps,
        ],
        marker_color="steelblue",
    ))
    fig.add_trace(go.Bar(
        name=f"B: {b_label}",
        x=["Success Rate", "Avg Steps"],
        y=[
            (b_rate or 0) * 100,
            diff.b_summary.avg_steps,
        ],
        marker_color="coral",
    ))

    fig.update_layout(
        title=f"Cohort Comparison: {a_label} vs {b_label}",
        barmode="group",
        yaxis_title="Value",
        height=500,
    )

    fig.write_html(str(path), include_plotlyjs=True)
    return path


def _write_empty(path: Path, message: str) -> None:
    path.write_text(
        f"<html><body><h1>moirai</h1><p>{message}</p></body></html>",
        encoding="utf-8",
    )
