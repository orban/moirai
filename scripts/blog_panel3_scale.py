#!/usr/bin/env python3
"""Generate SVG for blog Panel 3: failure modes at scale.

Reads from moirai divergences JSON output (--from), or uses fallback values.

Usage:
    python scripts/blog_panel3_scale.py --from /tmp/divergences.json
    python scripts/blog_panel3_scale.py  # uses fallback
"""

import argparse
import json
import os
import sys

from blog_design import BG, TEXT, TEXT_MID, TEXT_MUTED, ACCENT, SURFACE, BORDER, FONT_BODY, svg_header, title_block

# Collapse cluster labels into interpretable failure modes
LABEL_MAP = {
    "wasted step": "Wasted orientation",
    "no-op orientation": "Wasted orientation",
    "wrong target": "Wrong file targeted",
    "different file": "Wrong file targeted",
    "skipped reasoning": "Skipped reasoning",
    "acted without thinking": "Skipped reasoning",
}

# Map "Early/Mid divergence: X vs Y" to categories based on step types
STEP_TYPE_CATEGORIES = {
    "test": "Test timing divergence",
    "bash": "Execution approach divergence",
    "search": "Search strategy divergence",
    "read": "Reading strategy divergence",
    "write": "Write strategy divergence",
    "edit": "Edit strategy divergence",
}


def load_categories(json_path: str | None) -> tuple[list[tuple[str, float]], int, int]:
    """Load failure mode categories. Returns (categories, n_tasks, n_pairs)."""
    if json_path:
        with open(json_path) as f:
            data = json.load(f)

        n_tasks = data.get("n_tasks", 0)
        n_pairs = data.get("n_pairs", 0)

        # Aggregate clusters into higher-level failure modes
        modes: dict[str, int] = {}
        for cluster in data.get("clusters", []):
            label = cluster["label"]
            size = cluster["size"]

            # Map to higher-level category
            mapped = None
            label_lower = label.lower()
            for key, mode_name in LABEL_MAP.items():
                if key in label_lower:
                    mapped = mode_name
                    break
            if mapped is None and "divergence:" in label_lower:
                # "Early/Mid divergence: X vs Y" — categorize by step type
                after_colon = label.split(":", 1)[1].strip().lower()
                for step_key, cat_name in STEP_TYPE_CATEGORIES.items():
                    if step_key in after_colon:
                        mapped = cat_name
                        break
            if mapped is None:
                mapped = label.split(":")[0].strip() if ":" in label else label

            modes[mapped] = modes.get(mapped, 0) + size

        total = sum(modes.values())
        categories = [
            (name, 100 * count / total)
            for name, count in sorted(modes.items(), key=lambda x: -x[1])
        ]
        # Top 8
        return categories[:8], n_tasks, n_pairs

    # Fallback: hardcoded from moirai divergences output (2026-04-05)
    return [
        ("Wasted orientation", 17.0),
        ("Early test divergence", 10.0),
        ("Wrong file targeted", 12.0),
        ("Bash execution divergence", 9.0),
        ("Search strategy divergence", 15.0),
        ("Source reading divergence", 12.0),
        ("Skipped reasoning", 3.0),
        ("Write strategy divergence", 2.0),
    ], 1096, 32472


def build_svg(categories: list[tuple[str, float]], n_tasks: int, n_pairs: int) -> str:
    max_pct = max(pct for _, pct in categories)
    bar_area = 460
    label_x = 180
    bar_start = 190

    n_cats = len(categories)
    row_h = 36
    top_margin = 50
    bot_margin = 30
    svg_h = top_margin + n_cats * row_h + bot_margin
    svg_w = 700

    out = []

    def w(s):
        out.append(s)

    w(svg_header(svg_w, svg_h))

    # Title
    cx = svg_w / 2
    w(title_block(
        cx,
        "Failure modes at scale",
        f"{n_tasks:,} mixed-outcome tasks \u00b7 {n_pairs:,} divergence pairs \u00b7 top {n_cats} modes shown",
    ))

    # Grid lines
    for tick_pct in range(5, int(max_pct) + 5, 5):
        tx = bar_start + (tick_pct / max_pct) * bar_area
        if tx < svg_w - 20:
            w(f'<line x1="{tx}" y1="{top_margin}" x2="{tx}"'
              f' y2="{top_margin + n_cats * row_h}" stroke="{BORDER}" stroke-width="1"/>')
            w(f'<text x="{tx}" y="{top_margin + n_cats * row_h + 14}"'
              f' text-anchor="middle" fill="{TEXT_MUTED}" font-size="9">{tick_pct:.0f}%</text>')

    # Bars
    for i, (name, pct) in enumerate(categories):
        y = top_margin + i * row_h
        bar_w = (pct / max_pct) * bar_area
        opacity = 1.0 - (i / n_cats) * 0.4

        # Label
        w(f'<text x="{label_x - 8}" y="{y + row_h/2 + 4}" text-anchor="end"'
          f' fill="{TEXT}" font-size="11.5">{name}</text>')

        # Bar
        w(f'<rect x="{bar_start}" y="{y + 4}" width="{bar_w:.1f}"'
          f' height="{row_h - 8}" rx="3" fill="{ACCENT}" opacity="{opacity:.2f}"/>')

        # Percentage
        w(f'<text x="{bar_start + bar_w + 6}" y="{y + row_h/2 + 4}"'
          f' text-anchor="start" fill="{TEXT}" font-size="11"'
          f' font-weight="600">{pct:.0f}%</text>')

    # Footer
    w(f'<text x="{cx}" y="{svg_h - 6}" text-anchor="middle" fill="{TEXT_MUTED}"'
      f' font-size="9.5" font-style="italic">'
      f'Divergence points clustered by TF-IDF content similarity</text>')

    w('</svg>')
    return '\n'.join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="json_path", default=None,
                        help="Path to moirai divergences JSON output")
    args = parser.parse_args()

    categories, n_tasks, n_pairs = load_categories(args.json_path)
    svg = build_svg(categories, n_tasks, n_pairs)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blog_output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "panel3_scale.svg")
    with open(out_path, "w") as f:
        f.write(svg)
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
