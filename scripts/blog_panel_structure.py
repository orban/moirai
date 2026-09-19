#!/usr/bin/env python3
"""Generate the structure-vs-lift panel for the trace-divergence post.

One dot per task, 1,096 of them, because the whole claim of that section is that
the aggregate hides two regimes. A bar chart of the two group means would be the
claim restated, not evidence for it; the point only lands if you can see the low
half sitting flat on zero while the high half spreads.

Reads scripts/blog_output/figure_data.json from run_figure_data.py. Prints the
summary statistics it plots, so the prose can quote the figure rather than a
number remembered from a different run.

Usage:
    python scripts/run_figure_data.py <data> --out scripts/blog_output/figure_data.json
    python scripts/blog_panel_structure.py
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blog_design import (  # noqa: E402
    ACCENT, BORDER, FONT_MONO, TEXT_MID, TEXT_MUTED,
    svg_header, title_block,
)

W, H = 700, 430
PAD_L, PAD_R, PAD_T, PAD_B = 62, 24, 64, 58
THRESHOLD = 0.20
N_BINS = 10


def kendall_tau(xs, ys):
    """Tau-b. n=1096 is ~600k pairs, which is fine and avoids a scipy dependency."""
    n = len(xs)
    conc = disc = tx = ty = 0
    for i in range(n):
        xi, yi = xs[i], ys[i]
        for j in range(i + 1, n):
            dx, dy = xi - xs[j], yi - ys[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif (dx > 0) == (dy > 0):
                conc += 1
            else:
                disc += 1
    denom = ((conc + disc + tx) * (conc + disc + ty)) ** 0.5
    return (conc - disc) / denom if denom else 0.0


def build_svg(tasks):
    xs = [t["structure"] for t in tasks]
    ys = [t["acc_divergence"] - t["acc_random"] for t in tasks]
    tau = kendall_tau(xs, ys)

    hi = [(x, y) for x, y in zip(xs, ys) if x >= THRESHOLD]
    lo = [(x, y) for x, y in zip(xs, ys) if x < THRESHOLD]
    hi_mean = sum(y for _, y in hi) / len(hi)
    lo_mean = sum(y for _, y in lo) / len(lo)

    x_max = max(xs) * 1.02
    y_lo, y_hi = min(ys), max(ys)
    pad = (y_hi - y_lo) * 0.06
    y_lo, y_hi = y_lo - pad, y_hi + pad

    plot_w = W - PAD_L - PAD_R
    plot_h = H - PAD_T - PAD_B

    def px(x):
        return PAD_L + (x / x_max) * plot_w

    def py(y):
        return PAD_T + plot_h - ((y - y_lo) / (y_hi - y_lo)) * plot_h

    out = [svg_header(W, H, min_height=300)]
    out.append(title_block(
        W / 2,
        "One quarter of tasks carry the signal",
        f"{len(tasks):,} mixed-outcome tasks · reranking lift over random, K=3 · "
        f"Kendall τ = {tau:+.3f}",
    ))

    # Zero line first, so points sit on top of it.
    out.append(
        f'  <line x1="{PAD_L}" y1="{py(0):.1f}" x2="{PAD_L + plot_w}" y2="{py(0):.1f}"'
        f' stroke="{BORDER}" stroke-width="1"/>'
    )

    # Threshold.
    tx_ = px(THRESHOLD)
    out.append(
        f'  <line x1="{tx_:.1f}" y1="{PAD_T}" x2="{tx_:.1f}" y2="{PAD_T + plot_h}"'
        f' stroke="{TEXT_MUTED}" stroke-width="1" stroke-dasharray="3 3"/>'
    )
    out.append(
        f'  <text x="{tx_ + 5:.1f}" y="{PAD_T + 11}" font-size="10" fill="{TEXT_MUTED}"'
        f' font-family="{FONT_MONO}">structure = {THRESHOLD:.2f}</text>'
    )

    # One dot per task. Low-structure muted, high-structure in the accent: the split
    # is the finding, so it is the only thing colour is spent on.
    for x, y in zip(xs, ys):
        high = x >= THRESHOLD
        out.append(
            f'  <circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="1.9"'
            f' fill="{ACCENT if high else TEXT_MUTED}"'
            f' fill-opacity="{0.55 if high else 0.28}"/>'
        )

    # Group means as rules across each half, labelled.
    for label, pts, mean, x0, x1 in (
        ("low", lo, lo_mean, PAD_L, tx_),
        ("high", hi, hi_mean, tx_, PAD_L + plot_w),
    ):
        colour = ACCENT if label == "high" else TEXT_MID
        out.append(
            f'  <line x1="{x0:.1f}" y1="{py(mean):.1f}" x2="{x1:.1f}" y2="{py(mean):.1f}"'
            f' stroke="{colour}" stroke-width="2"/>'
        )
        anchor = "end" if label == "low" else "start"
        lx = x1 - 6 if label == "low" else x0 + 6
        out.append(
            f'  <text x="{lx:.1f}" y="{py(mean) - 7:.1f}" text-anchor="{anchor}"'
            f' font-size="11" font-weight="600" fill="{colour}">'
            f'{len(pts):,} tasks, mean {mean * 100:+.1f}pp</text>'
        )

    # Axes.
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        v = frac * x_max
        out.append(
            f'  <text x="{px(v):.1f}" y="{PAD_T + plot_h + 18}" text-anchor="middle"'
            f' font-size="10" fill="{TEXT_MUTED}" font-family="{FONT_MONO}">{v:.2f}</text>'
        )
    steps = 5
    for i in range(steps + 1):
        v = y_lo + (y_hi - y_lo) * i / steps
        out.append(
            f'  <text x="{PAD_L - 9}" y="{py(v) + 3.5:.1f}" text-anchor="end"'
            f' font-size="10" fill="{TEXT_MUTED}" font-family="{FONT_MONO}">'
            f'{v * 100:+.0f}</text>'
        )
    out.append(
        f'  <text x="{PAD_L + plot_w / 2:.1f}" y="{H - 24}" text-anchor="middle"'
        f' font-size="11" fill="{TEXT_MID}">Structure score '
        f'(branch gap, earlyness, stability — equal weight)</text>'
    )
    out.append(
        f'  <text x="14" y="{PAD_T + plot_h / 2:.1f}" text-anchor="middle" font-size="11"'
        f' fill="{TEXT_MID}" transform="rotate(-90 14 {PAD_T + plot_h / 2:.1f})">'
        f'Reranking lift (pp)</text>'
    )
    out.append(
        f'  <text x="{W / 2}" y="{H - 8}" text-anchor="middle" font-size="10"'
        f' fill="{TEXT_MUTED}">Each dot is one task. The threshold was set by '
        f'inspecting a held-out slice, not tuned on this plot.</text>'
    )
    out.append("</svg>")
    return "\n".join(out), tau, hi, lo, hi_mean, lo_mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--from", dest="json_path",
        default=os.path.join(here, "blog_output", "figure_data.json"),
        help="figure_data.json from run_figure_data.py",
    )
    args = parser.parse_args()

    if not os.path.exists(args.json_path):
        parser.error(
            f"{args.json_path} not found. Generate it first:\n"
            f"    python scripts/run_figure_data.py <data_dir> --out {args.json_path}\n"
            f"There is no fallback: this panel plots real per-task observations or nothing."
        )

    with open(args.json_path) as f:
        tasks = json.load(f)["tasks"]

    svg, tau, hi, lo, hi_mean, lo_mean = build_svg(tasks)
    out_dir = os.path.join(here, "blog_output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "panel_structure.svg")
    with open(out_path, "w") as f:
        f.write(svg)

    # Printed so the prose can quote the figure instead of a remembered number.
    print(f"Written to {out_path} ({len(svg):,} bytes)")
    print(f"\n  tasks           {len(tasks):,}")
    print(f"  high-structure  {len(hi):,} ({100 * len(hi) / len(tasks):.0f}%)  mean lift {hi_mean * 100:+.1f}pp")
    print(f"  low-structure   {len(lo):,} ({100 * len(lo) / len(tasks):.0f}%)  mean lift {lo_mean * 100:+.1f}pp")
    print(f"  Kendall tau     {tau:+.3f}")


if __name__ == "__main__":
    main()
