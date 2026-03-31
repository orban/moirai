from __future__ import annotations

import os
from pathlib import Path
from string import Template

from moirai.compress import compress_phases, NOISE_STEPS
from moirai.schema import (
    Alignment,
    ClusterResult,
    CohortDiff,
    DivergencePoint,
    GAP,
    Run,
)

_TEMPLATE_DIR = Path(__file__).parent / "templates"

# Step colors — for dark backgrounds
STEP_COLORS: dict[str, str] = {
    "read(source)": "#5b8fbe", "read(config)": "#7eb0d5", "read(test_file)": "#4a8ab5",
    "read(other)": "#90c0de", "read": "#5b8fbe",
    "search(glob)": "#6dc4be", "search(specific)": "#4aada7", "search": "#6dc4be",
    "edit(source)": "#e8943a", "edit(config)": "#d4a56a", "edit": "#e8943a",
    "write(source)": "#e8d44d", "write(config)": "#d4b84a", "write(other)": "#c9b96a",
    "write(test_file)": "#b8a840", "write": "#e8d44d",
    "test": "#5cb85c", "test(pass)": "#3fb950", "test(fail)": "#b5cf6b",
    "bash(python)": "#e8585a", "bash(explore)": "#ff9da7", "bash(setup)": "#d4a5a7",
    "bash(read)": "#c9827a", "bash(other)": "#d48a8c", "bash": "#e8585a",
    "reason": "#b893c4", "subagent": "#a87bb0", "plan": "#b89a7a",
    "result": "#8b949e", "task": "#6e7681", "taskoutput": "#6e7681",
    GAP: "#21262d",
}

STEP_LEGEND = [
    ("read", "#5b8fbe"), ("search", "#6dc4be"), ("edit", "#e8943a"),
    ("write", "#e8d44d"), ("test", "#5cb85c"), ("bash", "#e8585a"),
    ("subagent", "#a87bb0"), ("plan", "#b89a7a"),
]

DEFAULT_COLOR = "#6e7681"
TYPE_COLORS = {"tool": "#5b8fbe", "llm": "#b893c4", "system": "#b89a7a", "judge": "#5cb85c", "error": "#e8585a"}


# ---------------------------------------------------------------------------
# Branch dashboard (main entry point)
# ---------------------------------------------------------------------------

def write_branch_html(
    alignment: Alignment | None,
    points: list[DivergencePoint],
    runs: list[Run],
    path: Path,
    task_results: list | None = None,
) -> Path:
    path = Path(path)
    if not runs:
        _write_empty(path, "No runs to display.")
        return path

    # Build task sections using split divergences
    task_sections_html = ""
    total_splits = 0
    if task_results:
        from moirai.analyze.splits import find_split_divergences

        for tid, task_runs, task_alignment, _task_points in task_results:
            if not task_alignment.matrix or not task_alignment.matrix[0]:
                continue
            safe_tid = tid.replace(" ", "_").replace("/", "_")
            splits, Z, dendro = find_split_divergences(task_alignment, task_runs)
            significant = [s for s in splits if s.separation > 0.3]
            total_splits += len(significant)
            task_sections_html += _render_task_section(
                tid, safe_tid, task_runs, task_alignment, splits, Z, dendro,
            )

    # Stats
    n_pass = sum(1 for r in runs if r.result.success)
    n_tasks = len(task_results) if task_results else 0
    n_div = total_splits
    pass_rate = f"{n_pass / len(runs):.0%}" if runs else "0%"

    # Legend
    legend_items = "".join(
        f'<span class="legend-item"><span class="legend-swatch" style="background:{color}"></span>{name}</span>'
        for name, color in STEP_LEGEND
    )

    # Patterns
    patterns_html = _build_patterns_table(runs)

    # Load template and substitute
    template_path = _TEMPLATE_DIR / "branch.html"
    tpl = Template(template_path.read_text(encoding="utf-8"))
    html = tpl.safe_substitute(
        total_runs=len(runs),
        pass_rate=pass_rate,
        n_tasks=n_tasks,
        n_div=n_div,
        legend_items=legend_items,
        task_sections=task_sections_html,
        patterns_html=patterns_html,
    )

    path.write_text(html, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Per-task section
# ---------------------------------------------------------------------------

def _render_task_section(
    tid: str,
    safe_tid: str,
    runs: list[Run],
    alignment: Alignment,
    splits: list,
    Z,
    dendro: dict,
) -> str:
    from moirai.schema import SplitDivergence

    n_pass = sum(1 for r in runs if r.result.success)
    n_fail = len(runs) - n_pass
    n_cols = len(alignment.matrix[0])

    # Filter to meaningful splits (separation > 0.3)
    significant = [s for s in splits if s.separation > 0.3]

    s = f'<div class="task-panel" id="task-{safe_tid}">'
    s += f'<div class="task-header"><span class="task-name">{tid}</span>'
    s += f'<span class="task-meta">{len(runs)} runs '
    s += f'<span class="pass-count">{n_pass}P</span> '
    s += f'<span class="fail-count">{n_fail}F</span> '
    s += f'{n_cols} cols</span></div>'

    s += _clustering_interpretation_from_splits(splits, runs)
    s += f'<div class="heatmap-container">{_build_dendrogram_heatmap(alignment, runs, significant, Z, dendro, safe_tid)}</div>'

    # Pairwise run comparisons at split points
    run_map = {r.run_id: r for r in runs}
    comparisons_shown = 0
    for i, split in enumerate(significant, 1):
        if comparisons_shown >= 3:
            break
        comp = _build_run_comparison(split, run_map, number=i, task_id=safe_tid)
        if comp:
            s += comp
            comparisons_shown += 1

    s += '</div>'
    return s


# ---------------------------------------------------------------------------
# Clustering interpretation
# ---------------------------------------------------------------------------

def _clustering_interpretation_from_splits(splits: list, runs: list[Run]) -> str:
    if len(runs) < 3:
        return ""

    n_pass = sum(1 for r in runs if r.result.success)
    n_fail = len(runs) - n_pass
    if n_pass == 0 or n_fail == 0:
        return ""

    significant = [s for s in splits if s.separation > 0.3]
    if not significant:
        return f'<div class="cluster-interpretation" style="color:var(--text-muted)">No significant structural divergence across {len(runs)} runs.</div>'

    predictive = [s for s in significant if s.p_value is not None and s.p_value < 0.2]

    if predictive:
        best = min(predictive, key=lambda s: s.p_value or 1.0)
        left_top = max(best.left_values, key=best.left_values.get) if best.left_values else "?"
        right_top = max(best.right_values, key=best.right_values.get) if best.right_values else "?"
        lr = f"{best.left_success_rate:.0%}" if best.left_success_rate is not None else "?"
        rr = f"{best.right_success_rate:.0%}" if best.right_success_rate is not None else "?"
        msg = f"Key split at column {best.column}: <code>{left_top}</code> ({lr} pass) vs <code>{right_top}</code> ({rr} pass), p={best.p_value:.3f}"
        color = "var(--success)" if best.p_value < 0.05 else "var(--accent)"
    else:
        msg = f"{len(significant)} structural split{'s' if len(significant) != 1 else ''}, none significantly predict outcome."
        color = "var(--text-muted)"

    return f'<div class="cluster-interpretation" style="color:{color}">{msg}</div>'


# ---------------------------------------------------------------------------
# Fork cards
# ---------------------------------------------------------------------------

def _build_run_comparison(split, run_map: dict[str, Run], number: int | None = None, task_id: str = "") -> str | None:
    """Build a side-by-side comparison of a pass run vs a fail run from a split.

    Picks the best pair: one from each subtree with different outcomes.
    Shows their full step sequences with the divergence point marked.
    """
    # Collect all pass/fail runs from both subtrees
    all_pass = [run_map[rid] for rid in split.left_runs + split.right_runs if run_map.get(rid) and run_map[rid].result.success is True]
    all_fail = [run_map[rid] for rid in split.left_runs + split.right_runs if run_map.get(rid) and run_map[rid].result.success is False]

    if not all_pass or not all_fail:
        return None

    # Pick the best pair: prefer same harness condition, then prefer runs with reasoning
    pass_run, fail_run = _pick_best_pair(all_pass, all_fail)

    if not pass_run or not fail_run:
        return None

    col = split.column
    card_id = f'fork-{task_id}-{number}' if number and task_id else ""
    hover = f'@mouseenter="highlightCol(\'{task_id}\', {col})" @mouseleave="clearHighlight()"' if task_id else ""

    s = f'<div class="fork-card" id="{card_id}" {hover} style="padding:16px">'

    # Header
    s += '<div class="fork-card-header">'
    if number is not None and task_id:
        s += f'<span class="fork-number" @click="scrollToHeatmap(\'{task_id}\', {number})">{number}</span>'
    s += '<span class="fork-summary">Pass vs fail comparison at the split point</span>'
    s += '</div>'

    if split.p_value is not None:
        s += f'<div class="fork-pvalue">p={split.p_value:.3f} · separation: {split.separation:.0%}</div>'

    # Align the two runs with NW to find matching/divergent steps
    from moirai.analyze.align import _nw_align
    from moirai.compress import step_enriched_name

    pass_steps = [(step, step_enriched_name(step)) for step in pass_run.steps if step_enriched_name(step) is not None]
    fail_steps = [(step, step_enriched_name(step)) for step in fail_run.steps if step_enriched_name(step) is not None]

    pass_names = [name for _, name in pass_steps]
    fail_names = [name for _, name in fail_steps]
    aligned_pass, aligned_fail = _nw_align(pass_names, fail_names)

    # Find first divergence point
    first_div = None
    for idx, (a, b) in enumerate(zip(aligned_pass, aligned_fail)):
        if a != b:
            first_div = idx
            break

    # Generate narrative
    if first_div is not None:
        pa = aligned_pass[first_div]
        fa = aligned_fail[first_div]

        # Get detail for the divergent steps
        p_detail = _find_step_detail(pass_steps, pa, aligned_pass, first_div)
        f_detail = _find_step_detail(fail_steps, fa, aligned_fail, first_div)

        # Build readable descriptions
        if pa == GAP:
            p_desc = "skipped this step"
        else:
            p_desc = f"<code>{pa}</code>"
            if p_detail:
                p_desc += f" ({p_detail})"

        if fa == GAP:
            f_desc = "skipped this step"
        else:
            f_desc = f"<code>{fa}</code>"
            if f_detail:
                f_desc += f" ({f_detail})"

        same_harness = pass_run.harness == fail_run.harness
        harness_note = ""
        if not same_harness:
            harness_note = f' <span style="color:var(--text-muted)">(different conditions: {pass_run.harness} vs {fail_run.harness})</span>'

        narrative = (
            f'After {first_div} identical step{"s" if first_div != 1 else ""}, these runs diverge: '
            f'the pass run chose {p_desc} '
            f'while the fail run chose {f_desc}.{harness_note}'
        )
        s += f'<div style="font-size:12px;color:var(--text);margin:12px 0;line-height:1.5">{narrative}</div>'

        # Show reasoning at the divergence point if available
        p_reasoning = _find_step_reasoning(pass_steps, aligned_pass, first_div)
        f_reasoning = _find_step_reasoning(fail_steps, aligned_fail, first_div)
        if p_reasoning or f_reasoning:
            s += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:8px 0">'
            if p_reasoning:
                s += f'<div style="font-size:10px;color:var(--text-muted);font-style:italic;padding:6px 10px;background:rgba(63,185,80,0.06);border-left:2px solid var(--success);border-radius:3px"><span style="color:var(--success);font-weight:600">Pass reasoning:</span> {p_reasoning[:250]}</div>'
            else:
                s += '<div></div>'
            if f_reasoning:
                s += f'<div style="font-size:10px;color:var(--text-muted);font-style:italic;padding:6px 10px;background:rgba(248,81,73,0.06);border-left:2px solid var(--failure);border-radius:3px"><span style="color:var(--failure);font-weight:600">Fail reasoning:</span> {f_reasoning[:250]}</div>'
            else:
                s += '<div></div>'
            s += '</div>'

    # Aligned diff view
    s += '<div style="margin-top:12px;font-size:10px;overflow-x:auto">'
    s += '<table style="width:100%;border-collapse:collapse">'
    s += '<tr><th style="width:50%;text-align:left;padding:4px 8px;border-bottom:1px solid var(--border)">'
    s += f'<span style="color:var(--success);font-weight:600">PASS</span> <span style="color:var(--text-muted)">{pass_run.run_id}</span></th>'
    s += f'<th style="width:50%;text-align:left;padding:4px 8px;border-bottom:1px solid var(--border)">'
    s += f'<span style="color:var(--failure);font-weight:600">FAIL</span> <span style="color:var(--text-muted)">{fail_run.run_id}</span></th></tr>'

    pass_orig_idx = 0
    fail_orig_idx = 0
    for idx, (a, b) in enumerate(zip(aligned_pass, aligned_fail)):
        is_match = (a == b)
        is_div_point = (idx == first_div)

        if is_div_point:
            row_bg = "background:rgba(210,153,34,0.08);"
            border = "border-left:3px solid var(--accent);"
        elif is_match:
            row_bg = ""
            border = "border-left:3px solid transparent;"
        else:
            row_bg = "background:rgba(210,153,34,0.04);"
            border = "border-left:3px solid rgba(210,153,34,0.3);"

        # Pass cell
        if a == GAP:
            p_cell = '<span style="color:var(--border)">—</span>'
        else:
            p_step, _ = pass_steps[pass_orig_idx] if pass_orig_idx < len(pass_steps) else (None, "")
            p_detail_text = _step_detail_text(p_step) if p_step else ""
            text_color = "var(--text-muted)" if is_match else "var(--text-bright)"
            p_cell = f'<span style="color:{text_color}">{a}</span>'
            if p_detail_text:
                p_cell += f' <span style="color:#484f58">{p_detail_text}</span>'
            pass_orig_idx += 1

        # Fail cell
        if b == GAP:
            f_cell = '<span style="color:var(--border)">—</span>'
        else:
            f_step, _ = fail_steps[fail_orig_idx] if fail_orig_idx < len(fail_steps) else (None, "")
            f_detail_text = _step_detail_text(f_step) if f_step else ""
            text_color = "var(--text-muted)" if is_match else "var(--text-bright)"
            f_cell = f'<span style="color:{text_color}">{b}</span>'
            if f_detail_text:
                f_cell += f' <span style="color:#484f58">{f_detail_text}</span>'
            fail_orig_idx += 1

        s += f'<tr style="{row_bg}"><td style="padding:2px 8px;{border}">{p_cell}</td><td style="padding:2px 8px;{border}">{f_cell}</td></tr>'

    s += '</table></div>'

    s += '</div>'
    return s


def _find_step_detail(steps: list[tuple], value: str, aligned: list[str], aligned_idx: int) -> str:
    """Find the attrs detail for a step at an aligned position."""
    orig_idx = sum(1 for i in range(aligned_idx) if aligned[i] != GAP)
    if orig_idx < len(steps):
        step, _ = steps[orig_idx]
        return _step_detail_text(step)
    return ""


def _find_step_reasoning(steps: list[tuple], aligned: list[str], aligned_idx: int) -> str:
    """Find the reasoning for a step at an aligned position."""
    orig_idx = sum(1 for i in range(aligned_idx) if aligned[i] != GAP)
    if orig_idx < len(steps):
        step, _ = steps[orig_idx]
        return step.output.get("reasoning", "") if step.output else ""
    return ""


def _step_detail_text(step) -> str:
    """Get a short detail string from step attrs."""
    if not step or not step.attrs:
        return ""
    fp = step.attrs.get("file_path", "")
    if fp:
        return os.path.basename(fp)
    cmd = step.attrs.get("command", "")
    if cmd:
        return cmd[:50]
    pat = step.attrs.get("pattern", "")
    if pat:
        return pat[:40]
    return ""


def _pick_best_pair(pass_runs: list[Run], fail_runs: list[Run]) -> tuple[Run, Run]:
    """Pick the best pass/fail pair for comparison.

    Priority: same harness condition > both have reasoning > any pair.
    """
    def _has_reasoning(r: Run) -> bool:
        return any(s.output.get("reasoning") for s in r.steps if s.output)

    def _harness(r: Run) -> str:
        return r.harness or ""

    # Try same-condition pairs with reasoning first
    for p in pass_runs:
        for f in fail_runs:
            if _harness(p) == _harness(f) and _has_reasoning(p) and _has_reasoning(f):
                return p, f

    # Same condition, at least one has reasoning
    for p in pass_runs:
        for f in fail_runs:
            if _harness(p) == _harness(f) and (_has_reasoning(p) or _has_reasoning(f)):
                return p, f

    # Same condition, no reasoning
    for p in pass_runs:
        for f in fail_runs:
            if _harness(p) == _harness(f):
                return p, f

    # Any pair with reasoning
    for p in pass_runs:
        for f in fail_runs:
            if _has_reasoning(p) or _has_reasoning(f):
                return p, f

    return pass_runs[0], fail_runs[0]


def _render_run_column(run: Run, divergence_col: int, is_pass: bool) -> str:
    """Render one run as a vertical step list with the divergence point marked."""
    from moirai.compress import step_enriched_name

    tag = "PASS" if is_pass else "FAIL"
    color = "var(--success)" if is_pass else "var(--failure)"
    bg = "rgba(63,185,80,0.04)" if is_pass else "rgba(248,81,73,0.04)"
    border_color = "var(--success)" if is_pass else "var(--failure)"

    s = f'<div style="border:1px solid {border_color};border-radius:6px;overflow:hidden">'
    s += f'<div style="padding:8px 12px;background:{bg};border-bottom:1px solid var(--border);font-size:11px">'
    s += f'<span style="color:{color};font-weight:700">{tag}</span> '
    s += f'<span style="color:var(--text-muted)">{run.run_id}</span>'
    s += '</div>'

    s += '<div style="padding:8px 0">'

    filtered_idx = 0
    for step in run.steps:
        enriched = step_enriched_name(step)
        if enriched is None:
            continue

        # Get detail
        detail = ""
        if step.attrs:
            fp = step.attrs.get("file_path", "")
            if fp:
                detail = os.path.basename(fp)
            cmd = step.attrs.get("command", "")
            if cmd:
                detail = cmd[:60]
            pat = step.attrs.get("pattern", "")
            if pat:
                detail = pat[:40]

        reasoning = step.output.get("reasoning", "") if step.output else ""

        # Is this near the divergence point?
        is_divergence = (filtered_idx == divergence_col)
        step_color = STEP_COLORS.get(enriched, DEFAULT_COLOR)

        if is_divergence:
            s += f'<div style="padding:4px 12px;background:rgba(210,153,34,0.1);border-left:3px solid var(--accent);margin:2px 0">'
        else:
            s += f'<div style="padding:2px 12px;margin:1px 0;border-left:3px solid transparent">'

        # Step name with color indicator
        s += f'<div style="font-size:11px;display:flex;align-items:center;gap:6px">'
        s += f'<span style="width:4px;height:4px;border-radius:50%;background:{step_color};flex-shrink:0"></span>'
        s += f'<span style="color:var(--text-bright)">{enriched}</span>'
        if detail:
            s += f' <span style="color:var(--text-muted);font-size:10px">{detail}</span>'
        s += '</div>'

        # Reasoning (if present)
        if reasoning:
            s += f'<div style="font-size:10px;color:var(--text-muted);font-style:italic;margin:2px 0 2px 10px;padding-left:8px;border-left:1px solid var(--border)">'
            s += f'{reasoning[:150]}'
            if len(reasoning) > 150:
                s += '...'
            s += '</div>'

        s += '</div>'
        filtered_idx += 1

    s += '</div></div>'
    return s


def _build_split_card(split, number: int | None = None, task_id: str = "") -> str:
    """Render a card for a dendrogram split divergence."""
    p_str = f"p={split.p_value:.3f}" if split.p_value is not None else ""
    col = split.column

    card_id = f'fork-{task_id}-{number}' if number and task_id else ""
    hover = f'@mouseenter="highlightCol(\'{task_id}\', {col})" @mouseleave="clearHighlight()"' if task_id else ""

    # Summary: what separates the two subtrees
    left_top = max(split.left_values, key=split.left_values.get) if split.left_values else "?"
    right_top = max(split.right_values, key=split.right_values.get) if split.right_values else "?"
    lr = f"{split.left_success_rate:.0%}" if split.left_success_rate is not None else "?"
    rr = f"{split.right_success_rate:.0%}" if split.right_success_rate is not None else "?"

    summary = (
        f"At column {col}, {len(split.left_runs)} runs chose "
        f"<code>{left_top}</code> ({lr} pass) while "
        f"{len(split.right_runs)} runs chose <code>{right_top}</code> ({rr} pass)."
    )

    s = f'<div class="fork-card" id="{card_id}" {hover}>'
    s += '<div class="fork-card-header">'
    if number is not None and task_id:
        s += f'<span class="fork-number" @click="scrollToHeatmap(\'{task_id}\', {number})">{number}</span>'
    elif number is not None:
        s += f'<span class="fork-number">{number}</span>'
    s += f'<span class="fork-summary">{summary}</span></div>'

    meta_parts = []
    if p_str:
        meta_parts.append(p_str)
    meta_parts.append(f"separation: {split.separation:.0%}")
    meta_parts.append(f"merge distance: {split.merge_distance:.3f}")
    s += f'<div class="fork-pvalue">{" · ".join(meta_parts)}</div>'

    # Left subtree branch
    s += _split_branch_html(split.left_values, split.left_success_rate, len(split.left_runs), "left subtree")
    # Right subtree branch
    s += _split_branch_html(split.right_values, split.right_success_rate, len(split.right_runs), "right subtree")

    s += '</div>'
    return s


def _split_branch_html(values: dict[str, int], success_rate: float | None, n_runs: int, label: str) -> str:
    rate = success_rate
    if rate is not None and rate >= 0.6:
        cls, rate_cls = "branch-success", "success"
    elif rate is not None and rate <= 0.2:
        cls, rate_cls = "branch-failure", "failure"
    else:
        cls, rate_cls = "branch-mixed", ""

    rate_str = f"{rate:.0%}" if rate is not None else "?"
    vals_str = ", ".join(f"<code>{v}</code> ({c})" for v, c in sorted(values.items(), key=lambda x: -x[1]))

    s = f'<div class="branch {cls}">'
    s += f'<div class="branch-header"><span class="branch-rate {rate_cls}">{rate_str} pass</span> — {n_runs} runs ({label})</div>'
    s += f'<div class="branch-trajectory">Values: {vals_str}</div>'
    s += '</div>'
    return s


# ---------------------------------------------------------------------------
# Dendrogram + heatmap SVG
# ---------------------------------------------------------------------------

def _scipy_coords_to_svg(
    icoord: list[list[float]],
    dcoord: list[list[float]],
    cell_h: float,
    dendro_w: float,
) -> list[str]:
    if not icoord:
        return []

    max_d = max(max(d) for d in dcoord) if dcoord else 1.0
    if max_d == 0:
        max_d = 1.0

    paths = []
    for ic, dc in zip(icoord, dcoord):
        pts = []
        for y_sc, x_sc in zip(ic, dc):
            svg_y = (y_sc - 5) / 10 * cell_h + cell_h / 2
            svg_x = dendro_w * (1 - x_sc / max_d)
            pts.append((svg_x, svg_y))
        d = (f"M {pts[0][0]:.1f},{pts[0][1]:.1f} "
             f"L {pts[1][0]:.1f},{pts[1][1]:.1f} "
             f"L {pts[2][0]:.1f},{pts[2][1]:.1f} "
             f"L {pts[3][0]:.1f},{pts[3][1]:.1f}")
        paths.append(f'<path d="{d}" fill="none" stroke="#484f58" stroke-width="1"/>')
    return paths


def _build_dendrogram_heatmap(
    alignment: Alignment,
    runs: list[Run],
    splits: list,
    Z,
    dendro: dict,
    task_id: str = "",
) -> str:
    if not alignment.matrix or not alignment.matrix[0]:
        return '<p style="color:var(--text-muted)">No alignment data.</p>'

    n_runs = len(alignment.run_ids)
    n_cols = len(alignment.matrix[0])
    success_map = {r.run_id: r.result.success for r in runs}

    # Step detail lookup
    step_details: dict[str, list[str]] = {}
    for run in runs:
        details = []
        for st in run.steps:
            if st.name in NOISE_STEPS:
                continue
            detail = ""
            if st.attrs:
                fp = st.attrs.get("file_path", "")
                if fp:
                    detail = os.path.basename(fp)
                cmd = st.attrs.get("command", "")
                if cmd:
                    detail = cmd[:50]
                pat = st.attrs.get("pattern", "")
                if pat:
                    detail = pat[:50]
            details.append(detail)
        step_details[run.run_id] = details

    # Dendrogram ordering from precomputed data
    dendro_paths: list[str] = []
    leaf_order: list[int] = list(range(n_runs))
    dendro_w = 0

    if dendro and "leaves" in dendro:
        leaf_order = list(dendro["leaves"])
        cell_h_tmp = max(8, min(16, 500 // max(n_runs, 1)))
        dendro_w = 100
        dendro_paths = _scipy_coords_to_svg(dendro["icoord"], dendro["dcoord"], cell_h_tmp, dendro_w)

    # Layout
    cell_w = max(8, min(16, 900 // max(n_cols, 1)))
    cell_h = max(8, min(16, 500 // max(n_runs, 1)))
    outcome_w = 6
    label_w = 140
    gap = 4
    heatmap_x = dendro_w + gap + outcome_w + gap + label_w
    svg_w = heatmap_x + n_cols * cell_w + 10
    marker_h = 20
    svg_h = n_runs * cell_h + marker_h + 4

    svg_id = f'id="heatmap-{task_id}"' if task_id else ""
    parts = [f'<svg {svg_id} width="{svg_w}" height="{svg_h}" xmlns="http://www.w3.org/2000/svg" '
             f'style="font-family:\'IBM Plex Mono\',monospace;font-size:9px">']

    y_off = marker_h

    # Dendrogram
    if dendro_paths:
        parts.append(f'<g transform="translate(0,{y_off})">')
        parts.extend(dendro_paths)
        parts.append('</g>')

    # Numbered divergence markers from splits + connector lines from dendrogram
    div_columns: dict[int, int] = {}
    split_connectors: list[str] = []

    max_d = max(max(d) for d in dendro["dcoord"]) if dendro.get("dcoord") else 1.0
    if max_d == 0:
        max_d = 1.0

    for i, sp in enumerate(splits):
        if sp.column < n_cols:
            num = i + 1
            div_columns[sp.column] = num

            col_x = heatmap_x + sp.column * cell_w + cell_w // 2

            # Find the dendrogram branch point for this split's merge distance
            # The branch point is at x = dendro_w * (1 - merge_dist/max_d),
            # y = average of the two subtree centers
            if dendro_w > 0:
                branch_x = dendro_w * (1 - sp.merge_distance / max_d)
                # Approximate branch y: average of left and right subtree leaf positions
                left_rows = [leaf_order.index(alignment.run_ids.index(rid)) for rid in sp.left_runs if rid in alignment.run_ids]
                right_rows = [leaf_order.index(alignment.run_ids.index(rid)) for rid in sp.right_runs if rid in alignment.run_ids]
                if left_rows and right_rows:
                    branch_y = y_off + ((sum(left_rows) / len(left_rows) + sum(right_rows) / len(right_rows)) / 2) * cell_h + cell_h / 2
                    # Connector: dashed line from branch point to column marker
                    is_sig = sp.p_value is not None and sp.p_value < 0.2
                    stroke_color = "#d29922" if is_sig else "#484f58"
                    stroke_opacity = "0.5" if is_sig else "0.2"
                    split_connectors.append(
                        f'<path d="M {branch_x:.1f},{branch_y:.1f} '
                        f'C {(branch_x + col_x) / 2:.1f},{branch_y:.1f} '
                        f'{(branch_x + col_x) / 2:.1f},{marker_h // 2 + 1:.1f} '
                        f'{col_x:.1f},{marker_h // 2 + 1:.1f}" '
                        f'fill="none" stroke="{stroke_color}" stroke-width="1" '
                        f'stroke-dasharray="3,3" opacity="{stroke_opacity}"/>'
                    )

    for col, num in div_columns.items():
        x = heatmap_x + col * cell_w + cell_w // 2
        parts.append(f'<line x1="{x}" y1="{marker_h}" x2="{x}" y2="{svg_h}" stroke="#d29922" stroke-width="0.5" opacity="0.15" data-col="{col}"/>')
        click = f' @click="scrollToFork(\'{task_id}\', {num})" style="cursor:pointer"' if task_id else ""
        parts.append(f'<circle cx="{x}" cy="{marker_h // 2 + 1}" r="7" fill="#d29922" data-div-num="{num}"{click}/>')
        parts.append(f'<text x="{x}" y="{marker_h // 2 + 4}" text-anchor="middle" fill="#0d1117" font-size="9" font-weight="700" style="pointer-events:none">{num}</text>')

    # Draw connectors after markers so they render on top
    parts.extend(split_connectors)

    # Rows
    for row_idx, orig_idx in enumerate(leaf_order):
        rid = alignment.run_ids[orig_idx]
        y = y_off + row_idx * cell_h
        succ = success_map.get(rid)

        parts.append(f'<g class="heatmap-row">')

        # Outcome strip
        oc = "#3fb950" if succ is True else ("#f85149" if succ is False else "#30363d")
        parts.append(f'<rect x="{dendro_w + gap}" y="{y}" width="{outcome_w}" height="{cell_h - 1}" fill="{oc}" rx="1"/>')

        # Label
        tag = "P" if succ is True else ("F" if succ is False else "?")
        tc = "#3fb950" if succ else "#f85149"
        short_id = rid[:18] + ".." if len(rid) > 20 else rid
        lx = dendro_w + gap + outcome_w + gap
        parts.append(f'<text x="{lx}" y="{y + cell_h - 3}" fill="#8b949e" font-size="8">{short_id}</text>')
        parts.append(f'<text x="{heatmap_x - 4}" y="{y + cell_h - 3}" text-anchor="end" fill="{tc}" font-size="9" font-weight="600">{tag}</text>')

        # Heatmap cells
        row = alignment.matrix[orig_idx]
        details = step_details.get(rid, [])
        non_gap_idx = 0
        for col in range(n_cols):
            val = row[col] if col < len(row) else GAP
            x = heatmap_x + col * cell_w
            if val == GAP:
                parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="#21262d"/>')
            else:
                color = STEP_COLORS.get(val, TYPE_COLORS.get(val, DEFAULT_COLOR))
                detail = details[non_gap_idx] if non_gap_idx < len(details) else ""
                tip_detail = detail.replace('"', '&quot;').replace("'", "&#39;") if detail else ""
                tip_step = val.replace('"', '&quot;').replace("'", "&#39;")
                is_div = col in div_columns
                stroke = ' stroke="#d29922" stroke-width="1.5"' if is_div else ""
                parts.append(
                    f'<rect x="{x}" y="{y}" width="{cell_w - 1}" height="{cell_h - 1}" '
                    f'fill="{color}" rx="1" opacity="0.85" data-col="{col}"{stroke} '
                    f'@mouseenter="showTip($event, \'{tip_step}\', \'{tip_detail}\')" '
                    f'@mousemove="moveTip($event)" '
                    f'@mouseleave="hideTip()"/>'
                )
                non_gap_idx += 1

        parts.append('</g>')

    parts.append('</svg>')
    return '\n'.join(parts)


# ---------------------------------------------------------------------------
# Patterns table
# ---------------------------------------------------------------------------

def _build_patterns_table(runs: list[Run]) -> str:
    from moirai.analyze.motifs import find_motifs

    motifs, _ = find_motifs(runs, min_n=3, max_n=5, min_count=3)
    known = [r for r in runs if r.result.success is not None]
    if not known:
        return '<p style="color:var(--text-muted)">No outcome data.</p>'
    baseline = sum(1 for r in known if r.result.success) / len(known)

    positive = [m for m in motifs if m.lift > 1.05][:6]
    negative = [m for m in motifs if m.lift < 0.95][:6]

    if not positive and not negative:
        return f'<p style="color:var(--text-muted)">No significant patterns (baseline: {baseline:.0%}).</p>'

    html = f'<p style="font-size:11px;color:var(--text-muted);margin:0 0 8px">Baseline: {baseline:.0%} across {len(known)} runs</p>'
    html += '<table><tr><th style="text-align:left">Pattern</th><th style="text-align:right">Success</th><th style="text-align:right">Runs</th><th style="text-align:right">p</th></tr>'

    for m in positive:
        p_str = f"{m.p_value:.3f}" if m.p_value is not None else ""
        html += f'<tr class="positive"><td style="font-size:11px">{m.display}</td>'
        html += f'<td style="text-align:right" class="success"><strong>{m.success_rate:.0%}</strong></td>'
        html += f'<td style="text-align:right;color:var(--text-muted)">{m.total_runs}</td>'
        html += f'<td style="text-align:right;color:#484f58">{p_str}</td></tr>'

    if positive and negative:
        html += '<tr><td colspan="4" style="padding:4px"></td></tr>'

    for m in negative:
        p_str = f"{m.p_value:.3f}" if m.p_value is not None else ""
        html += f'<tr class="negative"><td style="font-size:11px">{m.display}</td>'
        html += f'<td style="text-align:right" class="failure"><strong>{m.success_rate:.0%}</strong></td>'
        html += f'<td style="text-align:right;color:var(--text-muted)">{m.total_runs}</td>'
        html += f'<td style="text-align:right;color:#484f58">{p_str}</td></tr>'

    html += '</table>'
    return html


# ---------------------------------------------------------------------------
# Clusters and Diff (unchanged)
# ---------------------------------------------------------------------------

def write_clusters_html(result: ClusterResult, path: Path, runs: list[Run] | None = None) -> Path:
    import plotly.graph_objects as go

    path = Path(path)
    if not result.clusters:
        _write_empty(path, "No clusters.")
        return path

    sorted_clusters = sorted(result.clusters, key=lambda c: -c.count)
    run_map = {r.run_id: r for r in runs} if runs else {}
    labels, counts, rates, hovers = [], [], [], []

    for info in sorted_clusters:
        labels.append(f"C{info.cluster_id} ({info.count})")
        counts.append(info.count)
        rates.append(info.success_rate * 100 if info.success_rate is not None else 0)
        if run_map:
            members = [run_map[rid] for rid, cid in result.labels.items() if cid == info.cluster_id and rid in run_map]
            if members:
                rep = sorted(members, key=lambda r: len(r.steps))[len(members) // 2]
                hovers.append(compress_phases(rep)[:100])
            else:
                hovers.append("")
        else:
            hovers.append("")

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Count", x=labels, y=counts, marker_color="steelblue",
                         hovertext=hovers, hovertemplate="%{x}<br>%{y} runs<br>%{hovertext}<extra></extra>"))
    fig.add_trace(go.Bar(name="Success %", x=labels, y=rates, marker_color="seagreen"))
    fig.update_layout(title="Cluster Distribution", barmode="group", height=450)
    fig.write_html(str(path), include_plotlyjs=True)
    return path


def write_diff_html(diff: CohortDiff, a_label: str, b_label: str, path: Path) -> Path:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    path = Path(path)
    a_rate, b_rate = diff.a_summary.success_rate, diff.b_summary.success_rate
    if a_rate is None and b_rate is None:
        _write_empty(path, "No data.")
        return path

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Metrics", "Cluster Shifts"))
    fig.add_trace(go.Bar(name=f"A: {a_label}", x=["Success Rate", "Steps"], y=[(a_rate or 0) * 100, diff.a_summary.avg_steps], marker_color="steelblue"), row=1, col=1)
    fig.add_trace(go.Bar(name=f"B: {b_label}", x=["Success Rate", "Steps"], y=[(b_rate or 0) * 100, diff.b_summary.avg_steps], marker_color="coral"), row=1, col=1)

    if diff.cluster_shifts:
        from moirai.viz.terminal import _compress_prototype
        sl, sv, sc = [], [], []
        for proto, delta in diff.cluster_shifts[:10]:
            c = _compress_prototype(proto)
            sl.append(c[:27] + "..." if len(c) > 30 else c)
            sv.append(delta)
            sc.append("seagreen" if delta > 0 else "indianred")
        fig.add_trace(go.Bar(x=sv, y=sl, orientation="h", marker_color=sc, showlegend=False), row=1, col=2)

    fig.update_layout(title=f"{a_label} vs {b_label}", barmode="group", height=450)
    fig.write_html(str(path), include_plotlyjs=True)
    return path


def _write_empty(path: Path, message: str) -> None:
    path.write_text(
        f'<html><body style="background:#0d1117;color:#c9d1d9;font-family:monospace;padding:40px">'
        f'<h1>moirai</h1><p>{message}</p></body></html>',
        encoding="utf-8",
    )
