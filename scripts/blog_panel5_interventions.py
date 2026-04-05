#!/usr/bin/env python3
"""Generate SVG for blog Panel 5: scaffold interventions."""

import os
import sys

# --- Data ---

INTERVENTIONS = [
    {
        "num": 1,
        "title": "Delay broad testing",
        "detection": "edited_files >= 2 AND tests_run == 0",
        "nudge": (
            "You\u2019ve edited multiple files. Consider focusing "
            "your testing on the specific change rather than "
            "running the full suite."
        ),
        "icon_lines": [
            # Clock icon (circle + hands)
            '<circle cx="18" cy="18" r="14" fill="none" stroke="{color}" stroke-width="2"/>',
            '<line x1="18" y1="18" x2="18" y2="10" stroke="{color}" stroke-width="2" stroke-linecap="round"/>',
            '<line x1="18" y1="18" x2="24" y2="18" stroke="{color}" stroke-width="2" stroke-linecap="round"/>',
        ],
    },
    {
        "num": 2,
        "title": "Verify before expanding",
        "detection": "new_files_read >= 5 AND tests_run == 0",
        "nudge": (
            "You\u2019ve explored several files. Before reading "
            "more, verify your current understanding by "
            "running relevant tests."
        ),
        "icon_lines": [
            # Magnifying glass
            '<circle cx="16" cy="16" r="10" fill="none" stroke="{color}" stroke-width="2"/>',
            '<line x1="23" y1="23" x2="30" y2="30" stroke="{color}" stroke-width="2.5" stroke-linecap="round"/>',
            # Small checkmark inside lens
            '<polyline points="11,16 15,20 21,12" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>',
        ],
    },
    {
        "num": 3,
        "title": "Stop hammering one file",
        "detection": "same_file_edits >= 3",
        "nudge": (
            "You\u2019ve edited this file 3+ times. Step back "
            "and verify before making more edits to "
            "this file."
        ),
        "icon_lines": [
            # Branching arrows (spread out)
            '<line x1="18" y1="30" x2="18" y2="14" stroke="{color}" stroke-width="2" stroke-linecap="round"/>',
            '<line x1="18" y1="14" x2="8" y2="6" stroke="{color}" stroke-width="2" stroke-linecap="round"/>',
            '<line x1="18" y1="14" x2="28" y2="6" stroke="{color}" stroke-width="2" stroke-linecap="round"/>',
            '<circle cx="8" cy="6" r="3" fill="none" stroke="{color}" stroke-width="1.8"/>',
            '<circle cx="28" cy="6" r="3" fill="none" stroke="{color}" stroke-width="1.8"/>',
            '<circle cx="18" cy="30" r="3" fill="{color}"/>',
        ],
    },
]

# --- Colors ---

BG = "#fff"
CARD_BG = "#f8f7f5"
CARD_BORDER = "#e0ddd8"
TITLE_COLOR = "#3a9e96"
TEXT_COLOR = "#333"
SUBTITLE_COLOR = "#777"
CODE_BG = "#2c2c2c"
CODE_TEXT = "#e0e0e0"
NUM_BG = "#3a9e96"
NUM_TEXT = "#fff"

# --- Layout ---

VIEW_W = 700
VIEW_H = 320
CARD_W = 210
CARD_H = 238
CARD_GAP = 15
CARD_RX = 6
CODE_RX = 4

# Cards are centered horizontally
TOTAL_CARDS_W = 3 * CARD_W + 2 * CARD_GAP
CARDS_X0 = (VIEW_W - TOTAL_CARDS_W) / 2
CARDS_Y0 = 46


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _wrap_text(text: str, max_chars: int) -> list[str]:
    """Simple word-wrap that respects max_chars per line."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def generate_svg() -> str:
    parts: list[str] = []
    w = parts.append

    w(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {VIEW_W} {VIEW_H}"'
      f' style="font-family: system-ui, -apple-system, sans-serif; background: {BG}; max-width:700px; width:100%; height:auto; min-height:210px;">')

    # --- Header ---
    w(f'<text x="{VIEW_W / 2}" y="20" text-anchor="middle"'
      f' fill="{TEXT_COLOR}" font-size="16" font-weight="600">'
      f'What to do about it</text>')
    w(f'<text x="{VIEW_W / 2}" y="36" text-anchor="middle"'
      f' fill="{SUBTITLE_COLOR}" font-size="10.5">'
      f'Scaffold interventions: detect the pattern, nudge the behavior</text>')

    # --- Cards ---
    for i, intv in enumerate(INTERVENTIONS):
        cx = CARDS_X0 + i * (CARD_W + CARD_GAP)
        cy = CARDS_Y0

        # Card background
        w(f'<rect x="{cx}" y="{cy}" width="{CARD_W}" height="{CARD_H}"'
          f' rx="{CARD_RX}" fill="{CARD_BG}" stroke="{CARD_BORDER}" stroke-width="1"/>')

        # Number badge (small circle top-left)
        badge_cx = cx + 18
        badge_cy = cy + 18
        w(f'<circle cx="{badge_cx}" cy="{badge_cy}" r="10"'
          f' fill="{NUM_BG}"/>')
        w(f'<text x="{badge_cx}" y="{badge_cy + 4}" text-anchor="middle"'
          f' fill="{NUM_TEXT}" font-size="11" font-weight="700">'
          f'{intv["num"]}</text>')

        # Icon (top-right area of card)
        icon_x = cx + CARD_W - 48
        icon_y = cy + 4
        w(f'<g transform="translate({icon_x},{icon_y})">')
        for line in intv["icon_lines"]:
            w("  " + line.format(color=TITLE_COLOR))
        w("</g>")

        # Title
        title_y = cy + 48
        w(f'<text x="{cx + 14}" y="{title_y}" fill="{TITLE_COLOR}"'
          f' font-size="13" font-weight="700">'
          f'{_escape(intv["title"])}</text>')

        # Detection code box
        code_x = cx + 10
        code_y = title_y + 8
        code_w = CARD_W - 20
        code_h = 24
        w(f'<rect x="{code_x}" y="{code_y}" width="{code_w}" height="{code_h}"'
          f' rx="{CODE_RX}" fill="{CODE_BG}"/>')
        w(f'<text x="{code_x + 8}" y="{code_y + 16}"'
          f' fill="{CODE_TEXT}" font-size="8.5"'
          f' font-family="\'SF Mono\', \'Menlo\', \'Consolas\', monospace">'
          f'{_escape(intv["detection"])}</text>')

        # Nudge text (wrapped)
        nudge_lines = _wrap_text(intv["nudge"], 34)
        nudge_y0 = code_y + code_h + 18
        line_h = 14
        for j, line in enumerate(nudge_lines):
            w(f'<text x="{cx + 14}" y="{nudge_y0 + j * line_h}"'
              f' fill="{TEXT_COLOR}" font-size="10.5">'
              f'{_escape(line)}</text>')

    # --- Footer ---
    footer_y = VIEW_H - 12
    w(f'<text x="{VIEW_W / 2}" y="{footer_y}" text-anchor="middle"'
      f' fill="{SUBTITLE_COLOR}" font-size="8.5">'
      f'Detection runs in real-time during agent execution.'
      f' Open source: github.com/orban/moirai</text>')

    w("</svg>")
    return "\n".join(parts)


def main():
    svg = generate_svg()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blog_output")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "panel5_interventions.svg")
    with open(out_path, "w") as f:
        f.write(svg)

    print(svg)
    print(f"\nWritten to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
