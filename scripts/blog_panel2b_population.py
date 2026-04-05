#!/usr/bin/env python3
"""Generate Panel 2b: population-level histogram of per-task test-timing deltas.

Shows the distribution of (late testers pass rate) - (early testers pass rate)
across all qualifying tasks, with sign test statistics.

Usage:
    python scripts/blog_panel2b_population.py
    python scripts/blog_panel2b_population.py --from deltas.json
"""

import argparse
import json
import math
import os
import sys


# ── Colors & layout ───────────────────────────────────────────────

GREEN = "#2d9a3e"
RED = "#cc3e40"
TEXT_COLOR = "#333"
MUTED = "#888"
FONT = "system-ui, -apple-system, sans-serif"

SVG_W = 700
SVG_H = 300

# Plot area within the SVG
MARGIN_LEFT = 60
MARGIN_RIGHT = 30
MARGIN_TOP = 62
MARGIN_BOTTOM = 56

PLOT_W = SVG_W - MARGIN_LEFT - MARGIN_RIGHT
PLOT_H = SVG_H - MARGIN_TOP - MARGIN_BOTTOM

N_BINS = 20
BIN_LO = -0.55
BIN_HI = 0.55


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def compute_deltas_live(data_path: str) -> list[float]:
    """Compute deltas from moirai's library against real data."""
    sys.path.insert(0, ".")
    from moirai.load import load_runs
    from moirai.analyze.content import select_task_groups
    from moirai.analyze.features import per_task_deltas, _test_position_centroid

    print(f"Loading runs from {data_path} ...")
    runs, warnings = load_runs(data_path)
    print(f"  Loaded {len(runs)} runs ({len(warnings)} warnings)")

    task_groups, skipped = select_task_groups(runs, min_runs=4)
    print(f"  {len(task_groups)} qualifying task groups ({len(skipped)} skipped)")

    deltas = per_task_deltas(task_groups, _test_position_centroid, min_group_size=3)
    print(f"  {len(deltas)} tasks with computable deltas")
    return deltas


def compute_sign_test(deltas: list[float]) -> float | None:
    """Two-sided sign test. Returns p-value or None."""
    n_pos = sum(1 for d in deltas if d > 0)
    n_neg = sum(1 for d in deltas if d < 0)
    n = n_pos + n_neg
    if n < 5:
        return None

    k = max(n_pos, n_neg)
    p_tail = 0.0
    for x in range(k, n + 1):
        p_tail += math.comb(n, x) * (0.5 ** n)
    return min(2.0 * p_tail, 1.0)


def histogram(deltas: list[float], n_bins: int, lo: float, hi: float):
    """Build histogram bins. Returns list of (bin_center, count)."""
    bin_width = (hi - lo) / n_bins
    counts = [0] * n_bins
    for d in deltas:
        idx = int((d - lo) / bin_width)
        idx = max(0, min(n_bins - 1, idx))
        counts[idx] += 1
    bins = []
    for i in range(n_bins):
        center = lo + (i + 0.5) * bin_width
        bins.append((center, counts[i]))
    return bins, bin_width


def build_svg(deltas: list[float]) -> str:
    n_total = len(deltas)
    n_pos = sum(1 for d in deltas if d > 0)
    n_neg = sum(1 for d in deltas if d < 0)
    n_zero = n_total - n_pos - n_neg
    mean_delta = sum(deltas) / n_total if n_total else 0
    p_value = compute_sign_test(deltas)
    pct_pos = 100 * n_pos / n_total if n_total else 0

    bins, bin_width = histogram(deltas, N_BINS, BIN_LO, BIN_HI)
    max_count = max(c for _, c in bins) if bins else 1

    out: list[str] = []

    def w(s):
        out.append(s)

    w(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SVG_W} {SVG_H}"'
      f' style="max-width:700px;width:100%;height:auto;min-height:200px"'
      f' font-family="{FONT}">')
    w(f'<rect width="{SVG_W}" height="{SVG_H}" fill="white"/>')

    cx = SVG_W / 2

    # Title
    w(f'<text x="{cx}" y="22" text-anchor="middle"'
      f' font-size="15" font-weight="700" fill="{TEXT_COLOR}">'
      f'Does later testing predict success? ({n_total:,} tasks)</text>')

    # Subtitle
    w(f'<text x="{cx}" y="40" text-anchor="middle"'
      f' font-size="11" fill="{MUTED}">'
      'Per-task pass-rate delta: (late testers) minus (early testers)</text>')

    # ── Axes ──

    # x-axis scale: map BIN_LO..BIN_HI to MARGIN_LEFT..MARGIN_LEFT+PLOT_W
    def x_of(val):
        return MARGIN_LEFT + (val - BIN_LO) / (BIN_HI - BIN_LO) * PLOT_W

    # y-axis scale: map 0..max_count to MARGIN_TOP+PLOT_H..MARGIN_TOP
    y_ceil = math.ceil(max_count / 10) * 10  # round up to nearest 10
    if y_ceil == 0:
        y_ceil = 10

    def y_of(count):
        return MARGIN_TOP + PLOT_H - (count / y_ceil) * PLOT_H

    # Horizontal gridlines
    for tick in range(0, y_ceil + 1, max(1, y_ceil // 4)):
        yy = y_of(tick)
        w(f'<line x1="{MARGIN_LEFT}" y1="{yy:.1f}" x2="{MARGIN_LEFT + PLOT_W}" y2="{yy:.1f}"'
          f' stroke="#eee" stroke-width="0.5"/>')
        w(f'<text x="{MARGIN_LEFT - 8}" y="{yy + 4:.1f}" text-anchor="end"'
          f' font-size="9" fill="{MUTED}">{tick}</text>')

    # x-axis baseline
    base_y = y_of(0)
    w(f'<line x1="{MARGIN_LEFT}" y1="{base_y:.1f}" x2="{MARGIN_LEFT + PLOT_W}" y2="{base_y:.1f}"'
      f' stroke="#ccc" stroke-width="1"/>')

    # x-axis tick labels
    for tick_val in [-0.4, -0.2, 0.0, 0.2, 0.4]:
        tx = x_of(tick_val)
        w(f'<text x="{tx:.1f}" y="{base_y + 14:.1f}" text-anchor="middle"'
          f' font-size="9" fill="{MUTED}">{tick_val:+.1f}</text>')

    # Y-axis label
    yl_x = 14
    yl_y = MARGIN_TOP + PLOT_H / 2
    w(f'<text x="{yl_x}" y="{yl_y}" text-anchor="middle"'
      f' font-size="10" fill="{MUTED}"'
      f' transform="rotate(-90, {yl_x}, {yl_y})">count of tasks</text>')

    # X-axis label
    w(f'<text x="{cx}" y="{SVG_H - 6}" text-anchor="middle"'
      f' font-size="10" fill="{MUTED}">delta (pass rate difference)</text>')

    # ── Bars ──

    bar_gap = 1  # px gap between bars
    for bin_center, count in bins:
        if count == 0:
            continue
        bx_left = x_of(bin_center - bin_width / 2) + bar_gap / 2
        bx_right = x_of(bin_center + bin_width / 2) - bar_gap / 2
        bw = bx_right - bx_left
        by = y_of(count)
        bh = base_y - by

        color = GREEN if bin_center > 0 else RED
        w(f'<rect x="{bx_left:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}"'
          f' fill="{color}" opacity="0.7"/>')

    # ── Vertical line at x=0 (dashed, gray) ──

    x_zero = x_of(0)
    w(f'<line x1="{x_zero:.1f}" y1="{MARGIN_TOP}" x2="{x_zero:.1f}" y2="{base_y:.1f}"'
      f' stroke="#999" stroke-width="1" stroke-dasharray="4,3"/>')

    # ── Vertical line at mean delta (solid, dark) ──

    x_mean = x_of(mean_delta)
    w(f'<line x1="{x_mean:.1f}" y1="{MARGIN_TOP}" x2="{x_mean:.1f}" y2="{base_y:.1f}"'
      f' stroke="{TEXT_COLOR}" stroke-width="1.5"/>')

    # Label for mean line
    mean_label = f"mean = {mean_delta:+.3f}"
    label_x = x_mean + 5
    label_anchor = "start"
    # If mean is on the right side, put label to the left
    if x_mean > MARGIN_LEFT + PLOT_W * 0.65:
        label_x = x_mean - 5
        label_anchor = "end"
    w(f'<text x="{label_x:.1f}" y="{MARGIN_TOP + 12}" text-anchor="{label_anchor}"'
      f' font-size="10" font-weight="600" fill="{TEXT_COLOR}">{esc(mean_label)}</text>')

    # ── Annotations ──

    # Position annotations in top-right area
    ann_x = MARGIN_LEFT + PLOT_W - 4
    ann_y1 = MARGIN_TOP + 12

    w(f'<text x="{ann_x}" y="{ann_y1}" text-anchor="end"'
      f' font-size="10" fill="{TEXT_COLOR}">'
      f'{n_pos} of {n_total} tasks have positive delta ({pct_pos:.0f}%)</text>')

    # p-value annotation
    if p_value is not None:
        if p_value < 0.001:
            p_str = "sign test p &lt; 0.001"
        else:
            p_str = f"sign test p = {p_value:.3f}"
    else:
        p_str = "sign test: insufficient data"

    w(f'<text x="{ann_x}" y="{ann_y1 + 14}" text-anchor="end"'
      f' font-size="10" font-weight="600" fill="{MUTED}">{p_str}</text>')

    w('</svg>')
    return "\n".join(out)


def print_summary(deltas: list[float]):
    n = len(deltas)
    n_pos = sum(1 for d in deltas if d > 0)
    n_neg = sum(1 for d in deltas if d < 0)
    n_zero = n - n_pos - n_neg
    mean_d = sum(deltas) / n if n else 0
    median_d = sorted(deltas)[n // 2] if n else 0
    p = compute_sign_test(deltas)

    print(f"\n--- Summary stats ---")
    print(f"  Total tasks:    {n}")
    print(f"  Positive delta: {n_pos} ({100 * n_pos / n:.1f}%)")
    print(f"  Negative delta: {n_neg} ({100 * n_neg / n:.1f}%)")
    print(f"  Zero delta:     {n_zero}")
    print(f"  Mean delta:     {mean_d:+.4f}")
    print(f"  Median delta:   {median_d:+.4f}")
    if p is not None:
        print(f"  Sign test p:    {p:.6f}" if p >= 0.001 else f"  Sign test p:    {p:.2e}")
    else:
        print(f"  Sign test p:    N/A (too few non-zero)")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from", dest="json_path", default=None,
        help="Path to pre-computed deltas JSON (list of floats)")
    parser.add_argument(
        "--data", default="/Volumes/mnemosyne/moirai/swe_rebench_v2",
        help="Path to moirai dataset directory (default: SWE-rebench v2)")
    parser.add_argument(
        "--save-deltas", default=None,
        help="Save computed deltas to JSON file for reuse")
    args = parser.parse_args()

    if args.json_path:
        print(f"Loading pre-computed deltas from {args.json_path}")
        with open(args.json_path) as f:
            deltas = json.load(f)
        if not isinstance(deltas, list):
            print("Error: expected a JSON array of floats", file=sys.stderr)
            sys.exit(1)
    else:
        deltas = compute_deltas_live(args.data)

    if not deltas:
        print("Error: no deltas computed", file=sys.stderr)
        sys.exit(1)

    if args.save_deltas:
        with open(args.save_deltas, "w") as f:
            json.dump(deltas, f)
        print(f"Saved {len(deltas)} deltas to {args.save_deltas}")

    print_summary(deltas)

    svg = build_svg(deltas)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blog_output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "panel2b_population.svg")
    with open(out_path, "w") as f:
        f.write(svg)
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
