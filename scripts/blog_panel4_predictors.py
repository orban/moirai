#!/usr/bin/env python3
"""Generate Panel 4 of the blog explainer: 'What predicts it'.

Produces a diverging bar chart of pass-rate deltas for behavioral features.
Reads from moirai features JSON output (--from), or uses hardcoded fallback.

Usage:
    python scripts/blog_panel4_predictors.py --from /tmp/features_results.json
    python scripts/blog_panel4_predictors.py  # uses fallback values
"""

import argparse
import json
import os
import sys

from blog_design import BG, TEXT, TEXT_MID, TEXT_MUTED, POSITIVE_BAR, NEGATIVE_BAR, BORDER, FONT_BODY, svg_header, title_block

# Display names and descriptions for each feature
DISPLAY = {
    "test_position_centroid": ("Test timing", "Agents that delay testing until they have a fix"),
    "uncertainty_density": ("Uncertainty language", 'Agents that hedge (\u201cmaybe\u201d, \u201cmight\u201d, \u201clet me try\u201d)'),
    "symmetry_index": ("Trajectory shape", "Agents that follow an explore\u2192modify\u2192verify arc"),
    "reasoning_hypothesis_formation": ("Hypothesis formation", 'Agents that form hypotheses (\u201cI think the issue is\u2026\u201d)'),
    "edit_dispersion": ("Edit breadth", "Agents that edit across multiple files instead of repeatedly editing one"),
}

# Layout
SVG_W = 700
BAR_CENTER = 350
BAR_MAX_PX = 180
DELTA_CAP = 8.0
BAR_H = 22
ROW_H = 68
TOP_MARGIN = 58
BOT_MARGIN = 46


def load_features(json_path: str | None) -> list[dict]:
    """Load features from JSON or return fallback values."""
    if json_path:
        with open(json_path) as f:
            data = json.load(f)
        features = []
        for feat in data["features"]:
            display = DISPLAY.get(feat["name"])
            if not display:
                continue
            features.append({
                "name": display[0],
                "delta": feat["delta_pp"],
                "desc": display[1],
                "n_tasks": feat.get("n_tasks"),
            })
        return features

    # Fallback: hardcoded from moirai features output (min_runs=4, 2026-04-05)
    return [
        {"name": "Test timing", "delta": 6.5, "desc": DISPLAY["test_position_centroid"][1]},
        {"name": "Uncertainty language", "delta": -7.0, "desc": DISPLAY["uncertainty_density"][1]},
        {"name": "Trajectory shape", "delta": 6.6, "desc": DISPLAY["symmetry_index"][1]},
        {"name": "Hypothesis formation", "delta": -5.6, "desc": DISPLAY["reasoning_hypothesis_formation"][1]},
        {"name": "Edit breadth", "delta": 4.9, "desc": DISPLAY["edit_dispersion"][1]},
    ]


def bar_width(delta):
    return abs(delta) / DELTA_CAP * BAR_MAX_PX


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_svg(features: list[dict], n_tasks: int | None = None):
    svg_h = TOP_MARGIN + len(features) * ROW_H + BOT_MARGIN
    out = []

    def w(s):
        out.append(s)

    w(svg_header(SVG_W, svg_h, min_height=280))

    cx = SVG_W / 2
    w(title_block(
        cx,
        "What predicts it",
        "Within-task pass-rate deltas (pp = percentage points)",
    ))

    # Zero line
    y_first = TOP_MARGIN + 6
    y_last = TOP_MARGIN + (len(features) - 1) * ROW_H + BAR_H + 6
    w(f'<line x1="{BAR_CENTER}" y1="{y_first}" x2="{BAR_CENTER}" y2="{y_last}"'
      f' stroke="{BORDER}" stroke-width="1"/>')
    w(f'<text x="{BAR_CENTER}" y="{y_first - 4}" text-anchor="middle"'
      f' font-size="9" fill="{TEXT_MUTED}">0</text>')

    for i, feat in enumerate(features):
        row_top = TOP_MARGIN + i * ROW_H
        bar_y = row_top + 6
        bar_cy = bar_y + BAR_H / 2

        delta = feat["delta"]
        bw = bar_width(delta)
        positive = delta > 0
        color = POSITIVE_BAR if positive else NEGATIVE_BAR

        bx = BAR_CENTER if positive else BAR_CENTER - bw
        w(f'<rect x="{bx:.1f}" y="{bar_y}" width="{bw:.1f}" height="{BAR_H}"'
          f' rx="3" fill="{color}" opacity="0.82"/>')

        if positive:
            name_x, name_anchor = BAR_CENTER - 10, "end"
        else:
            name_x, name_anchor = BAR_CENTER + 10, "start"

        w(f'<text x="{name_x}" y="{bar_cy + 5}" text-anchor="{name_anchor}"'
          f' font-size="13" font-weight="600" fill="{TEXT}">'
          f'{esc(feat["name"])}</text>')

        if positive:
            dx, d_anchor, sign = BAR_CENTER + bw + 8, "start", "+"
        else:
            dx, d_anchor, sign = BAR_CENTER - bw - 8, "end", ""

        w(f'<text x="{dx:.1f}" y="{bar_cy + 5}" text-anchor="{d_anchor}"'
          f' font-size="12" font-weight="700" fill="{color}">'
          f'{sign}{delta:.1f}pp</text>')

        desc_y = bar_y + BAR_H + 16
        w(f'<text x="30" y="{desc_y}" text-anchor="start"'
          f' font-size="10.5" fill="{TEXT_MUTED}">{esc(feat["desc"])}</text>')

    # Footer
    task_str = f"{n_tasks:,} mixed-outcome tasks" if n_tasks else "1,096 mixed-outcome tasks"
    w(f'<text x="{cx}" y="{svg_h - 14}" text-anchor="middle"'
      f' font-size="10" fill="{TEXT_MUTED}">'
      f'Each feature measured via natural experiment: within-task median split, '
      f'{task_str}</text>')

    w('</svg>')
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="json_path", default=None, help="Path to moirai features JSON output")
    args = parser.parse_args()

    features = load_features(args.json_path)
    # Sort by absolute delta descending
    features.sort(key=lambda f: -abs(f["delta"]))

    n_tasks = None
    if args.json_path:
        with open(args.json_path) as f:
            data = json.load(f)
        n_tasks = data.get("n_tasks_mixed")

    svg = build_svg(features, n_tasks)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blog_output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "panel4_predictors.svg")
    with open(out_path, "w") as f:
        f.write(svg)
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
