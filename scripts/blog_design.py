"""Shared design system for blog SVG panels.

Matches the Working Record design at ryanorban.com — the tokens here are the
`--record-*` custom properties in that site's static/css/home.css, copied by value
because these SVGs are inlined into markdown and cannot read the page's CSS.

That coupling is the thing to watch. When the site's palette changed in September
2026 these panels did not, and seven figures shipped to production as off-white
cards with green bars sitting on warm paper. If home.css changes again, change the
eight chrome tokens below and re-run the panel scripts.

All panel scripts import from here for visual consistency.
"""

# ── Typography ─────────────────────────────────────────────────────
# Both faces are self-hosted by the site (static/fonts/) and preloaded in head.html.
FONT_BODY = "'Geist Sans', system-ui, -apple-system, sans-serif"
FONT_MONO = "'JetBrains Mono', monospace"

# ── Colors: chrome ─────────────────────────────────────────────────
# Page furniture. BG is deliberately the page background, not a lighter card: a
# figure should sit on the paper rather than float above it in a panel.
BG = "#f6f4ef"                  # --record-bg
TEXT = "#17161a"                # --record-ink
TEXT_MID = "#4a4a52"            # --record-mid
TEXT_MUTED = "#62626a"          # --record-muted
ACCENT = "#8a3b1e"              # --record-accent (oxide; was a green that clashed)
ACCENT_SOFT = "#e4c9b6"         # --record-accent-soft
BORDER = "#e0dcd2"              # --record-line
SURFACE = "#eae5da"             # a shade off --record-bg, for absence/gap cells

# ── Colors: step palette (muted, cohesive) ─────────────────────────
# Categorical and load-bearing: nine step types cannot be encoded in one accent, and
# the published prose refers to these by name ("the orange strip shows the p-value").
# Left unchanged across the palette migration on purpose — only the chrome moved.
STEP_COLORS = {
    "read_source": "#4a7399",   # steel blue
    "read_other": "#7a9ab8",    # lighter steel
    "search": "#3d8a7a",        # muted teal
    "edit": "#c07028",          # warm amber
    "write": "#b09820",         # muted gold
    "test_pass": "#4a8a56",     # forest green
    "test_fail": "#b84040",     # muted red
    "bash": "#9a5a52",          # dusty red-brown
    "reason": "#7a6890",        # muted purple
    "gap": SURFACE,             # absence of a step in an aligned column
    "other": "#c8c4be",         # warm gray
}

# ── Colors: data visualization ─────────────────────────────────────
PASS_COLOR = "#4a8a56"          # matches test_pass
FAIL_COLOR = "#b84040"          # matches test_fail
POSITIVE_BAR = "#4a8a56"        # for positive deltas
NEGATIVE_BAR = "#b84040"        # for negative deltas
DIVERGENCE_HOT = "#c07028"      # amber for significant divergence
DIVERGENCE_COLD = TEXT_MUTED    # muted for fail-biased
NEUTRAL_DOT = TEXT_MID          # for beeswarm

# ── Spacing ────────────────────────────────────────────────────────
MARGIN = 16
CELL_H = 16
CELL_GAP = 3
GROUP_GAP = 12
BORDER_W = 3


def step_color(label: str) -> str:
    """Map an enriched step name to a theme-consistent color."""
    if label == "-":
        return STEP_COLORS["gap"]
    if label.startswith("read(source"):
        return STEP_COLORS["read_source"]
    if label.startswith("read("):
        return STEP_COLORS["read_other"]
    if label.startswith("search("):
        return STEP_COLORS["search"]
    if label.startswith("edit("):
        return STEP_COLORS["edit"]
    if label.startswith("write("):
        return STEP_COLORS["write"]
    if label == "test(pass)":
        return STEP_COLORS["test_pass"]
    if label == "test(fail)":
        return STEP_COLORS["test_fail"]
    if label.startswith("bash("):
        return STEP_COLORS["bash"]
    if label == "reason":
        return STEP_COLORS["reason"]
    return STEP_COLORS["other"]


def svg_header(w: int, h: int, min_height: int = 250) -> str:
    """Standard SVG opening tag with responsive sizing and theme fonts."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}"'
        f' style="max-width:{w}px;width:100%;height:auto;min-height:{min_height}px;'
        f'font-family:{FONT_BODY}">'
        f'\n  <rect width="{w}" height="{h}" fill="{BG}"/>'
    )


def title_block(cx: float, title: str, subtitle: str, y1: int = 22, y2: int = 42) -> str:
    """Standard title + subtitle for a panel."""
    return (
        f'  <text x="{cx}" y="{y1}" text-anchor="middle" font-size="17"'
        f' font-weight="600" fill="{TEXT}" letter-spacing="-0.02em">{title}</text>\n'
        f'  <text x="{cx}" y="{y2}" text-anchor="middle" font-size="11"'
        f' fill="{TEXT_MUTED}">{subtitle}</text>'
    )
