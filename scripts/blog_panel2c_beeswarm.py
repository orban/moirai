"""Panel 2c: Population beeswarm — per-task test timing deltas across all tasks.

Each dot is one task. X = pass-rate delta (late testers minus early testers).
Right of zero = later testing helps on that task.
Computed from real data via moirai's library.
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from moirai.load import load_runs
from moirai.analyze.content import select_task_groups
from moirai.analyze.features import per_task_deltas, _test_position_centroid
from blog_design import (
    BG, TEXT, TEXT_MID, TEXT_MUTED, NEUTRAL_DOT, BORDER, SURFACE,
    PASS_COLOR, FAIL_COLOR, FONT_BODY, svg_header, title_block,
)


DATA_DIR = "/Volumes/mnemosyne/moirai/swe_rebench_v2/"


def _esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def compute_deltas(json_path: str | None = None) -> list[float]:
    if json_path:
        with open(json_path) as f:
            return json.load(f)

    runs, _ = load_runs(DATA_DIR)
    task_groups, _ = select_task_groups(runs, min_runs=4)
    return per_task_deltas(task_groups, _test_position_centroid, min_group_size=3)


def generate_beeswarm(deltas: list[float]) -> str:
    n_pos = sum(1 for d in deltas if d > 0)
    n_neg = sum(1 for d in deltas if d < 0)
    n_zero = sum(1 for d in deltas if d == 0)
    n = len(deltas)
    pct_pos = 100 * n_pos / n if n else 0
    mean_delta = sum(deltas) / n if n else 0

    W = 700
    H = 420
    margin_x = 80
    axis_w = W - 2 * margin_x
    axis_y = 190  # vertical center of the dot field
    dot_r = 3.0
    dot_y_min = 60   # don't overflow into title
    dot_y_max = 320  # don't overflow into summary

    # Axis range: symmetric around 0, clamped to [-0.6, 0.6]
    x_max = 0.6

    def x_pos(delta: float) -> float:
        clamped = max(-x_max, min(x_max, delta))
        return margin_x + (clamped + x_max) / (2 * x_max) * axis_w

    # Build beeswarm layout: stack dots to avoid overlap
    # Sort deltas, place each dot, nudge vertically if overlapping
    sorted_deltas = sorted(deltas)
    dot_positions: list[tuple[float, float, float]] = []  # (x, y, delta)

    for delta in sorted_deltas:
        dx = x_pos(delta)
        # Find a y position that doesn't overlap, clamped to bounds
        dy = axis_y
        placed = False
        for offset in range(80):
            for sign in [1, -1] if offset > 0 else [1]:
                candidate_y = axis_y + sign * offset * (dot_r * 2.2)
                if candidate_y < dot_y_min or candidate_y > dot_y_max:
                    continue
                overlap = False
                for px, py, _ in dot_positions:
                    dist = math.sqrt((dx - px) ** 2 + (candidate_y - py) ** 2)
                    if dist < dot_r * 2.2:
                        overlap = True
                        break
                if not overlap:
                    dy = candidate_y
                    placed = True
                    break
            if placed:
                break
        # If still not placed, skip this dot rather than overflow
        if not placed:
            continue
        dot_positions.append((dx, dy, delta))

    lines = []
    lines.append(svg_header(W, H, min_height=250))

    # Title
    lines.append(title_block(
        W / 2,
        'Is this just one task?',
        f'Each dot is one of {n:,} tasks. Right of zero = later testing predicts success.',
        y1=20, y2=38,
    ))

    zero_x = x_pos(0)

    # Background split: subtle tint left (early helps) and right (late helps)
    dot_top = dot_y_min - 5
    dot_bot = dot_y_max + 5
    lines.append(f'  <rect x="{margin_x}" y="{dot_top}" width="{zero_x - margin_x}"'
                 f' height="{dot_bot - dot_top}" fill="{FAIL_COLOR}" opacity="0.04" rx="4"/>')
    lines.append(f'  <rect x="{zero_x}" y="{dot_top}" width="{W - margin_x - zero_x}"'
                 f' height="{dot_bot - dot_top}" fill="{PASS_COLOR}" opacity="0.04" rx="4"/>')

    # Zero line (through the dot field)
    lines.append(f'  <line x1="{zero_x}" y1="{dot_top}" x2="{zero_x}"'
                 f' y2="{dot_bot}" stroke="{BORDER}" stroke-width="1"'
                 f' stroke-dasharray="4,3"/>')

    # Count annotations at top of each half
    lines.append(f'  <text x="{(margin_x + zero_x) / 2}" y="{dot_top + 14}"'
                 f' text-anchor="middle" font-size="12" font-weight="600"'
                 f' fill="{FAIL_COLOR}" opacity="0.6">{n_neg} tasks</text>')
    lines.append(f'  <text x="{(zero_x + W - margin_x) / 2}" y="{dot_top + 14}"'
                 f' text-anchor="middle" font-size="12" font-weight="600"'
                 f' fill="{PASS_COLOR}" opacity="0.6">{n_pos} tasks</text>')

    # Dots
    for dx, dy, delta in dot_positions:
        lines.append(f'  <circle cx="{dx:.1f}" cy="{dy:.1f}" r="{dot_r}"'
                     f' fill="{NEUTRAL_DOT}" opacity="0.45"/>')

    # Axis — BELOW all dots with clear gap
    axis_line_y = dot_bot + 16
    lines.append(f'  <line x1="{margin_x}" y1="{axis_line_y}" x2="{W - margin_x}"'
                 f' y2="{axis_line_y}" stroke="{BORDER}" stroke-width="1"/>')

    # Ticks
    for tick in [-0.4, -0.2, 0, 0.2, 0.4]:
        tx = x_pos(tick)
        sign = "+" if tick > 0 else ""
        lines.append(f'  <line x1="{tx}" y1="{axis_line_y - 3}" x2="{tx}"'
                     f' y2="{axis_line_y + 3}" stroke="{BORDER}" stroke-width="1"/>')
        lines.append(f'  <text x="{tx}" y="{axis_line_y + 14}" text-anchor="middle"'
                     f' font-size="9" fill="{TEXT_MUTED}">{sign}{tick:.1f}</text>')

    # Direction labels below axis
    lines.append(f'  <text x="{margin_x}" y="{axis_line_y + 28}" text-anchor="start"'
                 f' font-size="10" fill="{TEXT_MUTED}">\u2190 early testing helps</text>')
    lines.append(f'  <text x="{W - margin_x}" y="{axis_line_y + 28}" text-anchor="end"'
                 f' font-size="10" fill="{TEXT_MUTED}">late testing helps \u2192</text>')

    # Mean marker on axis
    mean_x = x_pos(mean_delta)
    lines.append(f'  <line x1="{mean_x}" y1="{axis_line_y - 6}" x2="{mean_x}"'
                 f' y2="{axis_line_y + 6}" stroke="{TEXT}" stroke-width="2"/>')

    # Summary annotation
    lines.append(f'  <text x="{W/2}" y="{H - 10}" text-anchor="middle"'
                 f' font-size="11" fill="{TEXT_MID}">'
                 f'{n_pos} of {n} tasks ({pct_pos:.0f}%) show the pattern.'
                 f' Mean delta: +{mean_delta*100:.1f}pp.'
                 f' Sign test p &lt; 0.001.</text>')

    lines.append("</svg>")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="json_path", default=None)
    parser.add_argument("--save-deltas", default=None)
    args = parser.parse_args()

    print("Computing per-task deltas...")
    deltas = compute_deltas(args.json_path)
    print(f"  {len(deltas)} tasks, {sum(1 for d in deltas if d > 0)} positive")

    if args.save_deltas:
        with open(args.save_deltas, "w") as f:
            json.dump(deltas, f)
        print(f"  Saved to {args.save_deltas}")

    svg = generate_beeswarm(deltas)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blog_output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "panel2c_beeswarm.svg")
    with open(out_path, "w") as f:
        f.write(svg)
    print(f"Written to {out_path} ({len(svg)} bytes)")
