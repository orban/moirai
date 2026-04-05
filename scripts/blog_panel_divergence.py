"""Panel: Raw alignment with divergence detection.

Shows NW alignment of 10 runs grouped by outcome (pass/fail),
with Fisher's exact test p-value strip and ranked divergence points.
NO test timing annotations — this is the discovery step before features.

Uses vyperlang/vyper-4385.
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from moirai.schema import Run, Step, Result, GAP
from moirai.compress import step_enriched_name
from moirai.analyze.align import align_runs
from moirai.analyze.divergence import find_activity_divergences
from blog_design import (
    step_color, svg_header, title_block,
    BG, TEXT, TEXT_MID, TEXT_MUTED,
    PASS_COLOR, FAIL_COLOR,
    MARGIN, CELL_H, CELL_GAP, GROUP_GAP, BORDER_W,
    FONT_BODY, STEP_COLORS,
    SURFACE, DIVERGENCE_HOT, DIVERGENCE_COLD,
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
            Step(idx=s.get("idx", 0), type=s.get("type", ""),
                 name=s.get("name", ""), status=s.get("status", "ok"),
                 input=s.get("input", {}), output=s.get("output", {}),
                 metrics=s.get("metrics", {}), attrs=s.get("attrs", {}))
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
                summary=result.get("summary")),
        ))
    pass_runs = [r for r in runs if r.result.success][:5]
    fail_runs = [r for r in runs if not r.result.success][:5]
    return pass_runs + fail_runs


def generate(selected_runs: list[Run]) -> str:
    alignment = align_runs(selected_runs, level="name")
    matrix = alignment.matrix
    n_runs = len(matrix)
    n_cols = len(matrix[0])

    # Activity divergences via Fisher's exact test
    activity_divs = find_activity_divergences(alignment, selected_runs)
    div_by_col = {d.column: d for d in activity_divs}

    # ── Layout ─────────────────────────────────────────────────────────
    W = 700
    margin_x = 14
    bar_x = margin_x
    bar_area_w = W - 2 * margin_x
    cell_w = bar_area_w / n_cols
    cell_h = 16
    cell_gap = 3
    group_gap = 12
    border_w = 3

    top_offset = 55
    align_h = 5 * (cell_h + cell_gap) + group_gap + 5 * (cell_h + cell_gap)

    # Divergence strip
    strip_gap = 16
    strip_h = 18
    rank_h = 14

    total_h = top_offset + align_h + strip_gap + strip_h + rank_h + 30

    lines = []
    lines.append(svg_header(W, total_h))

    # Header
    title = "Align them. Find where they diverge."
    subtitle = ("Needleman-Wunsch alignment \u00b7"
                " Fisher\u2019s exact test per column \u00b7 brighter = more significant")
    lines.append(title_block(W / 2, title, subtitle))

    # ── Alignment rows (grouped by outcome) ────────────────────────────
    for i in range(n_runs):
        is_pass = i < 5
        extra_gap = group_gap if i >= 5 else 0
        y = top_offset + i * (cell_h + cell_gap) + extra_gap

        bc = PASS_COLOR if is_pass else FAIL_COLOR
        lines.append(f'  <rect x="{bar_x - border_w - 1}" y="{y}"'
                     f' width="{border_w}" height="{cell_h}" fill="{bc}" rx="1"/>')

        for col in range(n_cols):
            val = matrix[i][col]
            color = step_color(val)
            x = bar_x + col * cell_w
            lines.append(f'  <rect x="{x:.1f}" y="{y}"'
                         f' width="{max(cell_w - 0.3, 0.5):.1f}"'
                         f' height="{cell_h}" fill="{color}"/>')

    # Pass/fail labels
    lines.append(f'  <text x="{W - 4}" y="{top_offset + 10}" text-anchor="end"'
                 f' font-size="9" fill="{PASS_COLOR}" font-weight="500">pass</text>')
    fail_y = top_offset + 5 * (cell_h + cell_gap) + group_gap + 10
    lines.append(f'  <text x="{W - 4}" y="{fail_y}" text-anchor="end"'
                 f' font-size="9" fill="{FAIL_COLOR}" font-weight="500">fail</text>')

    # ── Divergence strip (Fisher p-values) ─────────────────────────────
    strip_y = top_offset + align_h + strip_gap

    neg_log_p = []
    directions = []
    for col in range(n_cols):
        d = div_by_col.get(col)
        if d:
            neg_log_p.append(-math.log10(max(d.p_value, 1e-10)))
            directions.append(d.direction)
        else:
            neg_log_p.append(0.0)
            directions.append(0.0)
    max_nlp = max(neg_log_p) or 1

    # Strip label
    lines.append(f'  <text x="{W - margin_x}" y="{strip_y + strip_h / 2 + 3}"'
                 f' text-anchor="end" font-size="9" fill="{TEXT_MUTED}">Fisher p-value</text>')

    for j in range(n_cols):
        x = margin_x + j * cell_w
        intensity = neg_log_p[j] / max_nlp
        if intensity < 0.05:
            lines.append(f'  <rect x="{x:.1f}" y="{strip_y}"'
                         f' width="{max(cell_w - 0.3, 0.5):.1f}"'
                         f' height="{strip_h}" fill="{SURFACE}"/>')
        else:
            color = DIVERGENCE_HOT if directions[j] > 0 else DIVERGENCE_COLD
            lines.append(f'  <rect x="{x:.1f}" y="{strip_y}"'
                         f' width="{max(cell_w - 0.3, 0.5):.1f}"'
                         f' height="{strip_h}" fill="{color}"'
                         f' opacity="{0.15 + 0.85 * intensity:.2f}"/>')

    # Rank markers
    marker_y = strip_y + strip_h + 3
    top_divs = activity_divs[:4]
    for rank, d in enumerate(top_divs):
        mx = margin_x + d.column * cell_w + cell_w / 2
        is_top = rank == 0
        mc = DIVERGENCE_HOT if is_top else TEXT_MUTED
        fw = "700" if is_top else "500"
        label = f"#{rank + 1}" if not is_top else f"#{rank + 1} \u2190"
        lines.append(f'  <text x="{mx}" y="{marker_y + 9}" text-anchor="middle"'
                     f' font-size="8" font-weight="{fw}" fill="{mc}">{label}</text>')

    # Bottom annotation
    if top_divs:
        d = top_divs[0]
        dir_label = "pass-biased" if d.direction > 0 else "fail-biased"
        lines.append(f'  <text x="{W/2}" y="{total_h - 6}" text-anchor="middle"'
                     f' font-size="10" fill="{TEXT_MID}">'
                     f'Top divergence: column {d.column},'
                     f' {d.pass_active}P/{d.fail_active}F active,'
                     f' p={d.p_value:.3f} ({dir_label})</text>')

    lines.append("</svg>")
    return "\n".join(lines)


if __name__ == "__main__":
    runs = load_runs()
    print(f"Loaded {len(runs)} runs for {TASK_ID}")
    svg = generate(runs)
    out_dir = os.path.join(os.path.dirname(__file__), "blog_output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "panel_divergence.svg")
    with open(out_path, "w") as f:
        f.write(svg)
    print(f"Written to {out_path} ({len(svg)} bytes)")
