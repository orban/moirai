"""Shared design system for blog SVG panels.

Matches the Hugo theme (hugo-theme-til) at ryanorban.com.
All panel scripts import from here for visual consistency.
"""

# ── Typography ─────────────────────────────────────────────────────
# Geist Sans loaded by Hugo theme via CDN, JetBrains Mono from local files
FONT_BODY = "'Geist Sans', system-ui, -apple-system, sans-serif"
FONT_MONO = "'JetBrains Mono', monospace"

# ── Colors: theme ──────────────────────────────────────────────────
BG = "#fafaf8"
TEXT = "#232220"
TEXT_MID = "#555350"
TEXT_MUTED = "#8a8784"
ACCENT = "#5a7a64"
ACCENT_SOFT = "#edf2ed"
BORDER = "#e5e2de"
SURFACE = "#f2f0ec"

# ── Colors: step palette (muted, cohesive) ─────────────────────────
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
    "gap": SURFACE,             # matches theme surface
    "other": "#c8c4be",         # warm gray
}

# ── Colors: data visualization ─────────────────────────────────────
PASS_COLOR = "#4a8a56"          # matches test_pass
FAIL_COLOR = "#b84040"          # matches test_fail
POSITIVE_BAR = "#4a8a56"        # for positive deltas
NEGATIVE_BAR = "#b84040"        # for negative deltas
DIVERGENCE_HOT = "#c07028"      # amber for significant divergence
DIVERGENCE_COLD = "#8a8784"     # muted for fail-biased
NEUTRAL_DOT = "#555350"         # for beeswarm

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
