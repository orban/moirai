"""Panel 2: The Fork — Where pass and fail runs diverge.

Shows NW alignment of 10 runs with test step markers and test centroids.
Illustrates the test_position_centroid finding: pass runs test late, fail runs test early.
Uses vyperlang/vyper-4385 as the exemplar task (selected by find_exemplar_task).
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from moirai.schema import Run, Step, Result, GAP
from moirai.compress import step_enriched_name
from moirai.analyze.align import align_runs
from moirai.analyze.content import compute_test_centroid
from moirai.analyze.divergence import find_activity_divergences


# ── Color palette ──────────────────────────────────────────────────────

def step_color(label: str) -> str:
    if label == GAP:
        return "#f0eee9"
    if label.startswith("read(source"):
        return "#4a7fae"
    if label.startswith("read("):
        return "#7aaed4"
    if label.startswith("search("):
        return "#3a9e96"
    if label.startswith("edit("):
        return "#d4802a"
    if label.startswith("write("):
        return "#c4a820"
    if label == "test(pass)":
        return "#2d9a3e"
    if label == "test(fail)":
        return "#cc3e40"
    if label.startswith("bash("):
        return "#b85450"
    if label == "reason":
        return "#9070a0"
    return "#d5d2cd"


# ── Load task data ─────────────────────────────────────────────────────

DATA_DIR = "/Volumes/mnemosyne/moirai/swe_rebench_v2/"
TASK_ID = "vyperlang__vyper-4385"  # selected by find_exemplar_task(compute_test_centroid)


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


def _esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def generate_panel2(selected_runs: list[Run]) -> str:
    alignment = align_runs(selected_runs, level="name")
    matrix = alignment.matrix
    n_runs = len(matrix)
    n_cols = len(matrix[0])

    # Compute test centroids per run
    centroids = [compute_test_centroid(r) for r in selected_runs]

    # Find test column positions in the alignment for each run
    test_cols_per_run: list[list[int]] = []
    for i in range(n_runs):
        test_cols = [
            col for col in range(n_cols)
            if matrix[i][col].startswith("test(")
        ]
        test_cols_per_run.append(test_cols)

    # Compute centroid in alignment space (for the marker)
    centroid_cols = []
    for i in range(n_runs):
        tc = test_cols_per_run[i]
        if tc:
            centroid_cols.append(sum(tc) / len(tc))
        else:
            centroid_cols.append(None)

    # ── Layout ─────────────────────────────────────────────────────────
    W = 700
    margin_x = 14
    cell_h = 16
    cell_gap = 3
    group_gap = 12
    top_offset = 55
    border_w = 3
    centroid_col_w = 50  # space on right for centroid value

    bar_area_w = W - 2 * margin_x - centroid_col_w
    cell_w = bar_area_w / n_cols

    alignment_h = n_runs * (cell_h + cell_gap) + group_gap
    centroid_summary_h = 50
    total_h = top_offset + alignment_h + centroid_summary_h + 20

    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {total_h}"'
                 f' style="max-width:700px;width:100%;height:auto;min-height:280px;'
                 f'font-family:system-ui,-apple-system,sans-serif">')
    lines.append(f'  <rect width="{W}" height="{total_h}" fill="#fff"/>')

    # Header
    lines.append(f'  <text x="{W/2}" y="22" text-anchor="middle" font-size="18"'
                 f' font-weight="600" fill="#222">Align them. Find the pattern.</text>')
    lines.append(f'  <text x="{W/2}" y="42" text-anchor="middle" font-size="12"'
                 f' fill="#888">Needleman-Wunsch alignment \u00b7'
                 f' triangles mark test steps \u00b7 diamonds show test centroid</text>')

    # Column header for centroid
    cx_label = margin_x + bar_area_w + 8
    lines.append(f'  <text x="{cx_label}" y="{top_offset - 8}" font-size="9"'
                 f' fill="#888">test</text>')
    lines.append(f'  <text x="{cx_label}" y="{top_offset}" font-size="9"'
                 f' fill="#888">centroid</text>')

    # ── Draw alignment with test markers ───────────────────────────────
    for i in range(n_runs):
        is_pass = i < 5
        extra_gap = group_gap if i >= 5 else 0
        y = top_offset + i * (cell_h + cell_gap) + extra_gap

        # Outcome border
        bc = "#2d9a3e" if is_pass else "#cc3e40"
        lines.append(f'  <rect x="{margin_x - border_w - 1}" y="{y}"'
                     f' width="{border_w}" height="{cell_h}" fill="{bc}" rx="1"/>')

        # Row label
        label = f"P{i+1}" if is_pass else f"F{i-4}"
        lines.append(f'  <text x="{margin_x - border_w - 5}" y="{y + cell_h/2 + 3}"'
                     f' text-anchor="end" font-size="8" font-weight="600"'
                     f' fill="{bc}">{label}</text>')

        # Step cells
        for col in range(n_cols):
            val = matrix[i][col]
            color = step_color(val)
            x = margin_x + col * cell_w
            lines.append(f'  <rect x="{x:.1f}" y="{y}"'
                         f' width="{max(cell_w - 0.3, 0.5):.1f}"'
                         f' height="{cell_h}" fill="{color}"/>')

        # Test markers: small triangles above test cells
        for col in test_cols_per_run[i]:
            tx = margin_x + col * cell_w + cell_w / 2
            ty = y - 1
            marker_color = "#2d9a3e" if matrix[i][col] == "test(pass)" else "#cc3e40"
            lines.append(f'  <polygon points="{tx-2.5},{ty} {tx+2.5},{ty} {tx},{ty-4}"'
                         f' fill="{marker_color}"/>')

        # Centroid diamond marker on the bar
        cc = centroid_cols[i]
        if cc is not None:
            dx = margin_x + cc * cell_w
            dy = y + cell_h / 2
            lines.append(f'  <polygon points="{dx},{dy-5} {dx+4},{dy} {dx},{dy+5} {dx-4},{dy}"'
                         f' fill="none" stroke="#333" stroke-width="1.5"/>')

            # Centroid value on the right
            c_val = centroids[i]
            if c_val is not None:
                c_color = "#2d9a3e" if is_pass else "#cc3e40"
                lines.append(f'  <text x="{cx_label}" y="{y + cell_h/2 + 3}"'
                             f' font-size="10" font-weight="600"'
                             f' fill="{c_color}">{c_val:.2f}</text>')

    # Pass/fail labels
    lines.append(f'  <text x="{margin_x + bar_area_w + centroid_col_w - 2}"'
                 f' y="{top_offset + 10}" text-anchor="end"'
                 f' font-size="9" fill="#2d9a3e" font-weight="500">pass</text>')
    fail_label_y = top_offset + 5 * (cell_h + cell_gap) + group_gap + 10
    lines.append(f'  <text x="{margin_x + bar_area_w + centroid_col_w - 2}"'
                 f' y="{fail_label_y}" text-anchor="end"'
                 f' font-size="9" fill="#cc3e40" font-weight="500">fail</text>')

    # ── Centroid summary at bottom ─────────────────────────────────────
    summary_y = top_offset + alignment_h + 15

    pass_centroids = [c for c in centroids[:5] if c is not None]
    fail_centroids = [c for c in centroids[5:] if c is not None]
    pass_avg = sum(pass_centroids) / len(pass_centroids) if pass_centroids else 0
    fail_avg = sum(fail_centroids) / len(fail_centroids) if fail_centroids else 0
    delta = pass_avg - fail_avg

    # Visual bar showing average centroid positions
    bar_y = summary_y
    bar_h = 14

    # Background bar
    lines.append(f'  <rect x="{margin_x}" y="{bar_y}" width="{bar_area_w}"'
                 f' height="{bar_h}" fill="#f5f3f0" rx="3"/>')

    # Pass centroid marker
    px = margin_x + pass_avg * bar_area_w
    lines.append(f'  <rect x="{px-1}" y="{bar_y-2}" width="3"'
                 f' height="{bar_h+4}" fill="#2d9a3e" rx="1"/>')
    lines.append(f'  <text x="{px}" y="{bar_y - 6}" text-anchor="middle"'
                 f' font-size="10" font-weight="600" fill="#2d9a3e">'
                 f'pass avg: {pass_avg:.2f}</text>')

    # Fail centroid marker
    fx = margin_x + fail_avg * bar_area_w
    lines.append(f'  <rect x="{fx-1}" y="{bar_y-2}" width="3"'
                 f' height="{bar_h+4}" fill="#cc3e40" rx="1"/>')
    lines.append(f'  <text x="{fx}" y="{bar_y + bar_h + 12}" text-anchor="middle"'
                 f' font-size="10" font-weight="600" fill="#cc3e40">'
                 f'fail avg: {fail_avg:.2f}</text>')

    # Delta annotation
    mid_x = (px + fx) / 2
    lines.append(f'  <text x="{mid_x}" y="{bar_y + bar_h + 28}" text-anchor="middle"'
                 f' font-size="11" fill="#555">'
                 f'Pass runs test {delta:.0%} later in the trajectory</text>')

    lines.append("</svg>")
    return "\n".join(lines)


if __name__ == "__main__":
    runs = load_runs()
    svg = generate_panel2(runs)
    out_dir = os.path.join(os.path.dirname(__file__), "blog_output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "panel2_fork.svg")
    with open(out_path, "w") as f:
        f.write(svg)
    print(f"Written to {out_path}")
    print(f"SVG size: {len(svg)} bytes")
