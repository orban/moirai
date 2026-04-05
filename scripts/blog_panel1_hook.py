"""Panel 1: The Hook — Same task, same agent, different outcomes.

Loads vyperlang/vyper-4385 (10 runs, 5P/5F) via moirai's library.
Shows trajectory bars with colored step blocks and pass/fail borders.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from moirai.schema import Run, Step, Result
from moirai.compress import step_enriched_name
from blog_design import (
    step_color, svg_header, title_block,
    BG, TEXT, TEXT_MID, TEXT_MUTED,
    PASS_COLOR, FAIL_COLOR,
    MARGIN, CELL_H, CELL_GAP, GROUP_GAP, BORDER_W,
    FONT_BODY, STEP_COLORS,
)

DATA_DIR = "/Volumes/mnemosyne/moirai/swe_rebench_v2/"
TASK_ID = "vyperlang__vyper-4385"


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
            Step(idx=s.get("idx", 0), type=s.get("type", ""),
                 name=s.get("name", ""), status=s.get("status", "ok"),
                 input=s.get("input", {}), output=s.get("output", {}),
                 metrics=s.get("metrics", {}), attrs=s.get("attrs", {}))
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
                summary=result.get("summary")),
        ))
    pass_runs = [r for r in runs if r.result.success][:5]
    fail_runs = [r for r in runs if not r.result.success][:5]
    return pass_runs + fail_runs


def generate_panel1(runs: list[Run]) -> str:
    # Build enriched sequences
    run_data = []
    for r in runs:
        enriched = [step_enriched_name(s) for s in r.steps]
        enriched = [e for e in enriched if e is not None]
        run_data.append({
            "outcome": "pass" if r.result.success else "fail",
            "steps": enriched,
        })

    W, H = 700, 420
    margin_left = 12
    margin_right = 12
    bar_area_w = W - margin_left - margin_right
    bar_h = 22
    bar_gap = 6
    group_gap = 14
    top_offset = 70
    border_w = 4

    max_steps = max(len(r["steps"]) for r in run_data)
    block_w = bar_area_w / max_steps

    lines = []
    lines.append(svg_header(W, H, min_height=280))

    # Header
    n_pass = sum(1 for r in run_data if r["outcome"] == "pass")
    n_fail = sum(1 for r in run_data if r["outcome"] == "fail")
    title = "Same task. Same agent. Ten runs."
    subtitle = (f"{n_pass} pass, {n_fail} fail."
                f" vyperlang/vyper #4385 \u2014 OpenHands + Qwen3-Coder")
    lines.append(title_block(W / 2, title, subtitle, y1=28, y2=50))

    for i, run in enumerate(run_data):
        is_pass = run["outcome"] == "pass"
        extra_gap = group_gap if i >= 5 else 0
        y = top_offset + i * (bar_h + bar_gap) + extra_gap

        border_color = PASS_COLOR if is_pass else FAIL_COLOR
        lines.append(f'  <rect x="{margin_left}" y="{y}" width="{border_w}"'
                     f' height="{bar_h}" fill="{border_color}" rx="2"/>')

        x = margin_left + border_w + 1
        for step in run["steps"]:
            color = step_color(step)
            bw = block_w - 0.5
            lines.append(f'  <rect x="{x:.1f}" y="{y}" width="{bw:.1f}"'
                         f' height="{bar_h}" fill="{color}"/>')
            x += block_w

    # Legend — two rows
    legend_items = [
        ("read", STEP_COLORS["read_source"]), ("search", STEP_COLORS["search"]),
        ("edit", STEP_COLORS["edit"]), ("write", STEP_COLORS["write"]),
        ("test pass", PASS_COLOR), ("test fail", FAIL_COLOR),
        ("bash", STEP_COLORS["bash"]), ("reason", STEP_COLORS["reason"]),
    ]
    row1 = legend_items[:4]
    row2 = legend_items[4:]
    swatch = 12
    item_w = (W - 2 * margin_left) / 4
    for row_idx, items in enumerate([row1, row2]):
        legend_y = H - 48 + row_idx * 22
        for j, (label, color) in enumerate(items):
            lx = margin_left + j * item_w + 8
            lines.append(f'  <rect x="{lx}" y="{legend_y}" width="{swatch}"'
                         f' height="{swatch}" fill="{color}" rx="2"/>')
            lines.append(f'  <text x="{lx + swatch + 6}" y="{legend_y + 10}"'
                         f' font-size="12" fill="{TEXT_MID}">{label}</text>')

    # Pass/fail labels
    lines.append(f'  <text x="{W - margin_right}" y="{top_offset + 14}"'
                 f' text-anchor="end" font-size="11" fill="{PASS_COLOR}"'
                 f' font-weight="500">pass</text>')
    fail_y = top_offset + 5 * (bar_h + bar_gap) + group_gap + 14
    lines.append(f'  <text x="{W - margin_right}" y="{fail_y}"'
                 f' text-anchor="end" font-size="11" fill="{FAIL_COLOR}"'
                 f' font-weight="500">fail</text>')

    lines.append('</svg>')
    return '\n'.join(lines)


if __name__ == "__main__":
    runs = load_runs()
    print(f"Loaded {len(runs)} runs for {TASK_ID}")
    svg = generate_panel1(runs)
    out_dir = os.path.join(os.path.dirname(__file__), "blog_output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "panel1_hook.svg")
    with open(out_path, "w") as f:
        f.write(svg)
    print(f"Written to {out_path} ({len(svg)} bytes)")
