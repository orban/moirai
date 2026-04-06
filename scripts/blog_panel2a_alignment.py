"""Panel 2a: The stochastic story — same agent, same task, different test timing.

Two parts:
  Top: NW alignment with test markers (teaches the technique)
  Bottom: dot strip showing each run's test centroid colored by outcome
    (the punchline — stochastic variation predicts outcome)

Uses vyperlang/vyper-4385 (selected by find_exemplar_task for test_position_centroid).
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from moirai.schema import Run, Step, Result, GAP
from moirai.compress import step_enriched_name
from moirai.analyze.align import align_runs
from moirai.analyze.content import compute_test_centroid
from blog_design import (
    step_color, BG, TEXT, TEXT_MID, TEXT_MUTED, PASS_COLOR, FAIL_COLOR,
    BORDER, SURFACE, FONT_BODY, svg_header, title_block, MARGIN,
)


DATA_DIR = "/Volumes/mnemosyne/moirai/swe_rebench_v2/"
TASK_ID = "vyperlang__vyper-4385"


def load_runs():
    runs = []
    for f in sorted(os.listdir(DATA_DIR)):
        if not f.endswith(".json"):
            continue
        with open(os.path.join(DATA_DIR, f)) as fp:
            data = json.load(fp)
        if data.get("task_id") != TASK_ID:
            continue
        steps = [
            Step(
                idx=s.get("idx", 0), type=s.get("type", ""),
                name=s.get("name", ""), status=s.get("status", "ok"),
                input=s.get("input", {}), output=s.get("output", {}),
                metrics=s.get("metrics", {}), attrs=s.get("attrs", {}),
            )
            for s in data.get("steps", [])
        ]
        result = data.get("result", {})
        runs.append(Run(
            run_id=data.get("run_id", f), task_id=TASK_ID,
            task_family=data.get("task_family"), agent=data.get("agent"),
            model=data.get("model"), harness=data.get("harness"),
            timestamp=data.get("timestamp"), tags=data.get("tags", {}),
            steps=steps, result=Result(
                success=result.get("success"), score=result.get("score"),
                label=result.get("label"), error_type=result.get("error_type"),
                summary=result.get("summary"),
            ),
        ))
    pass_runs = [r for r in runs if r.result.success][:5]
    fail_runs = [r for r in runs if not r.result.success][:5]
    return pass_runs + fail_runs


def generate_panel(selected_runs: list[Run]) -> str:
    alignment = align_runs(selected_runs, level="name")
    matrix = alignment.matrix
    n_runs = len(matrix)
    n_cols = len(matrix[0])

    # Compute test centroids and find test columns
    centroids = [compute_test_centroid(r) for r in selected_runs]
    test_cols_per_run = [
        [col for col in range(n_cols) if matrix[i][col].startswith("test(")]
        for i in range(n_runs)
    ]

    # Sort runs by centroid for the dot strip (NOT by outcome)
    run_order = sorted(range(n_runs), key=lambda i: centroids[i] or 0)

    # ── Layout ─────────────────────────────────────────────────────────
    W = 700
    margin_x = 14
    gradient_w = 10   # wide gradient strip on left
    gradient_gap = 4  # clear gap between gradient and alignment
    bar_x = margin_x + gradient_w + gradient_gap
    bar_area_w = W - bar_x - margin_x
    cell_w = bar_area_w / n_cols

    # Section 1: alignment (sorted by centroid, not by outcome)
    align_top = 55
    cell_h = 16
    cell_gap = 3
    align_h = n_runs * (cell_h + cell_gap)

    # Section 2: dot strip
    strip_top = align_top + align_h + 35
    strip_h = 70

    total_h = strip_top + strip_h + 50

    lines = []
    lines.append(svg_header(W, total_h, min_height=280))

    # Header
    lines.append(title_block(
        W / 2,
        'Align them. See where the tests fall.',
        '10 runs of the same agent on the same task,'
        ' sorted by test timing \u00b7 \u25b2 = test step',
    ))

    # ── Section 1: Alignment sorted by centroid ────────────────────────
    for row_idx, i in enumerate(run_order):
        is_pass = selected_runs[i].result.success
        y = align_top + row_idx * (cell_h + cell_gap)
        c = centroids[i] or 0

        # Gradient strip encodes centroid position (continuous, not binary outcome)
        # Low centroid (early testing) = warm red, high centroid = cool green
        r_val = int(200 - 150 * c)
        g_val = int(80 + 100 * c)
        b_val = int(60 + 60 * c)
        bc = f"rgb({r_val},{g_val},{b_val})"
        lines.append(f'  <rect x="{margin_x}" y="{y}"'
                     f' width="{gradient_w}" height="{cell_h}" fill="{bc}" rx="2"/>')

        # Step cells
        for col in range(n_cols):
            val = matrix[i][col]
            color = step_color(val)
            x = bar_x + col * cell_w
            lines.append(f'  <rect x="{x:.1f}" y="{y}"'
                         f' width="{max(cell_w - 0.3, 0.5):.1f}"'
                         f' height="{cell_h}" fill="{color}"/>')

        # Test markers
        for col in test_cols_per_run[i]:
            tx = bar_x + col * cell_w + cell_w / 2
            ty = y - 1
            mc = PASS_COLOR if matrix[i][col] == "test(pass)" else FAIL_COLOR
            lines.append(f'  <polygon points="{tx-2.5},{ty} {tx+2.5},{ty} {tx},{ty-4}"'
                         f' fill="{mc}"/>')

        # Right label: centroid value + outcome indicator
        c_val = centroids[i]
        c_str = f"{c_val:.2f}" if c_val is not None else "—"
        outcome_dot = "\u25cf" if is_pass else "\u25cb"  # filled/hollow circle
        outcome_color = PASS_COLOR if is_pass else FAIL_COLOR
        lines.append(f'  <text x="{W - margin_x}" y="{y + cell_h/2 + 4}"'
                     f' text-anchor="end" font-size="9" fill="{TEXT_MID}">'
                     f'{c_str} </text>')
        lines.append(f'  <text x="{W - margin_x + 1}" y="{y + cell_h/2 + 4}"'
                     f' font-size="9" fill="{outcome_color}">{outcome_dot}</text>')

    # ── Section 2: Dot strip — the punchline ───────────────────────────
    # Number line from 0 to 1, each run is a dot colored by outcome
    strip_label_y = strip_top - 8
    lines.append(f'  <text x="{W/2}" y="{strip_label_y}" text-anchor="middle"'
                 f' font-size="13" font-weight="600" fill="{TEXT}">'
                 f'Test centroid per run</text>')

    # Axis
    axis_y = strip_top + 30
    axis_x1 = bar_x + 20
    axis_x2 = W - margin_x - 20
    axis_w = axis_x2 - axis_x1

    lines.append(f'  <line x1="{axis_x1}" y1="{axis_y}" x2="{axis_x2}" y2="{axis_y}"'
                 f' stroke="{BORDER}" stroke-width="1"/>')

    # Axis labels
    for tick_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        tx = axis_x1 + tick_val * axis_w
        lines.append(f'  <line x1="{tx}" y1="{axis_y - 3}" x2="{tx}" y2="{axis_y + 3}"'
                     f' stroke="{BORDER}" stroke-width="1"/>')
        lines.append(f'  <text x="{tx}" y="{axis_y + 16}" text-anchor="middle"'
                     f' font-size="9" fill="{TEXT_MUTED}">{tick_val:.2f}</text>')

    # Axis label
    lines.append(f'  <text x="{axis_x1 - 8}" y="{axis_y + 4}" text-anchor="end"'
                 f' font-size="9" fill="{TEXT_MUTED}">early</text>')
    lines.append(f'  <text x="{axis_x2 + 8}" y="{axis_y + 4}" text-anchor="start"'
                 f' font-size="9" fill="{TEXT_MUTED}">late</text>')

    # Dots — each run is a circle, colored by outcome
    dot_r = 7
    for i in range(n_runs):
        c = centroids[i]
        if c is None:
            continue
        is_pass = selected_runs[i].result.success
        color = PASS_COLOR if is_pass else FAIL_COLOR
        dx = axis_x1 + c * axis_w

        # Slight vertical jitter to avoid overlap
        # Pass dots above axis, fail dots below
        dy = axis_y - 12 if is_pass else axis_y + 12

        lines.append(f'  <circle cx="{dx:.1f}" cy="{dy}" r="{dot_r}"'
                     f' fill="{color}" opacity="0.8" stroke="{BG}" stroke-width="1.5"/>')

    # Legend
    legend_y = axis_y + 36
    lines.append(f'  <circle cx="{W/2 - 60}" cy="{legend_y}" r="5" fill="{PASS_COLOR}" opacity="0.8"/>')
    lines.append(f'  <text x="{W/2 - 50}" y="{legend_y + 4}" font-size="10" fill="{TEXT_MID}">pass</text>')
    lines.append(f'  <circle cx="{W/2 + 20}" cy="{legend_y}" r="5" fill="{FAIL_COLOR}" opacity="0.8"/>')
    lines.append(f'  <text x="{W/2 + 30}" y="{legend_y + 4}" font-size="10" fill="{TEXT_MID}">fail</text>')

    # Caption
    lines.append(f'  <text x="{W/2}" y="{legend_y + 20}" text-anchor="middle"'
                 f' font-size="11" fill="{TEXT_MID}">'
                 f'Same agent. Same task. The stochastic choice of when to test predicts the outcome.</text>')

    lines.append("</svg>")
    return "\n".join(lines)


if __name__ == "__main__":
    runs = load_runs()
    print(f"Loaded {len(runs)} runs for {TASK_ID}")
    for i, r in enumerate(runs):
        c = compute_test_centroid(r)
        outcome = "pass" if r.result.success else "fail"
        print(f"  {outcome}: centroid={c:.2f}" if c else f"  {outcome}: no tests")

    svg = generate_panel(runs)
    out_dir = os.path.join(os.path.dirname(__file__), "blog_output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "panel2a_alignment.svg")
    with open(out_path, "w") as f:
        f.write(svg)
    print(f"Written to {out_path} ({len(svg)} bytes)")
