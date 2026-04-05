"""Panel 2b: Decision flow — behavioral strategies that emerge from stochastic variation.

Shows the decision flow for vyperlang/vyper-4385: all runs start the same way,
then diverge into strategies with different success rates.
Computed from real data via moirai's library.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from moirai.schema import Run, Step, Result
from moirai.compress import step_enriched_name, phase_sequence
from moirai.analyze.content import compute_test_centroid
from blog_design import (
    BG, TEXT, TEXT_MID, TEXT_MUTED, PASS_COLOR, FAIL_COLOR, ACCENT,
    ACCENT_SOFT, SURFACE, BORDER, FONT_BODY, svg_header, title_block,
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
    return runs


def classify_strategy(run: Run) -> dict:
    """Classify a run into a behavioral strategy based on real data."""
    enriched = [step_enriched_name(s) for s in run.steps]
    enriched = [e for e in enriched if e is not None]
    n = len(enriched)

    centroid = compute_test_centroid(run)
    test_count = sum(1 for e in enriched if e.startswith("test("))
    edit_count = sum(1 for e in enriched if e.startswith("edit("))
    bash_count = sum(1 for e in enriched if e.startswith("bash("))

    # Find first test position
    first_test = None
    for i, e in enumerate(enriched):
        if e.startswith("test("):
            first_test = i / (n - 1) if n > 1 else 0
            break

    return {
        "centroid": centroid or 0,
        "test_count": test_count,
        "edit_count": edit_count,
        "bash_count": bash_count,
        "first_test": first_test,
        "success": run.result.success,
        "n_steps": n,
    }


def _esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def generate_flow(runs: list[Run]) -> str:
    strategies = [classify_strategy(r) for r in runs]

    # Classify into two groups by centroid
    late_testers = [s for s in strategies if s["centroid"] > 0.5]
    early_testers = [s for s in strategies if s["centroid"] <= 0.5]

    late_pass = sum(1 for s in late_testers if s["success"])
    late_total = len(late_testers)
    early_pass = sum(1 for s in early_testers if s["success"])
    early_total = len(early_testers)

    late_avg_tests = sum(s["test_count"] for s in late_testers) / max(late_total, 1)
    early_avg_tests = sum(s["test_count"] for s in early_testers) / max(early_total, 1)
    late_avg_edits = sum(s["edit_count"] for s in late_testers) / max(late_total, 1)
    early_avg_edits = sum(s["edit_count"] for s in early_testers) / max(early_total, 1)
    late_avg_bash = sum(s["bash_count"] for s in late_testers) / max(late_total, 1)
    early_avg_bash = sum(s["bash_count"] for s in early_testers) / max(early_total, 1)

    W = 700
    H = 420
    cx = W / 2

    lines = []
    lines.append(svg_header(W, H, min_height=280))

    # Title
    lines.append(title_block(
        cx,
        'Two strategies emerge from the same agent',
        'vyperlang/vyper #4385 \u00b7 10 runs \u00b7'
        ' same agent, same task, same prompt',
    ))

    # ── Start node ─────────────────────────────────────────────────────
    start_y = 68
    node_w = 200
    node_h = 32
    lines.append(f'  <rect x="{cx - node_w/2}" y="{start_y}" width="{node_w}"'
                 f' height="{node_h}" rx="6" fill="{SURFACE}" stroke="{BORDER}" stroke-width="1"/>')
    lines.append(f'  <text x="{cx}" y="{start_y + 20}" text-anchor="middle"'
                 f' font-size="12" font-weight="600" fill="{TEXT_MID}">All 10 runs start here</text>')

    # Start detail
    lines.append(f'  <text x="{cx}" y="{start_y + node_h + 14}" text-anchor="middle"'
                 f' font-size="10" fill="{TEXT_MUTED}">read repo \u2192 search code \u2192 run setup</text>')

    # ── Fork point ─────────────────────────────────────────────────────
    fork_y = start_y + node_h + 30
    # Arrow from start to fork
    lines.append(f'  <line x1="{cx}" y1="{start_y + node_h}" x2="{cx}" y2="{fork_y + 8}"'
                 f' stroke="{BORDER}" stroke-width="1.5"/>')

    # Diamond fork
    diamond_y = fork_y + 10
    diamond_s = 16
    lines.append(f'  <polygon points="{cx},{diamond_y-diamond_s} {cx+diamond_s},{diamond_y}'
                 f' {cx},{diamond_y+diamond_s} {cx-diamond_s},{diamond_y}"'
                 f' fill="{ACCENT_SOFT}" stroke="{ACCENT}" stroke-width="2"/>')
    lines.append(f'  <text x="{cx}" y="{diamond_y + 4}" text-anchor="middle"'
                 f' font-size="9" font-weight="700" fill="{ACCENT}">?</text>')

    # Fork label
    lines.append(f'  <text x="{cx}" y="{diamond_y + diamond_s + 14}" text-anchor="middle"'
                 f' font-size="11" font-weight="600" fill="{ACCENT}">'
                 f'Stochastic fork: when does the agent first test?</text>')

    # ── Two branches ───────────────────────────────────────────────────
    branch_y = diamond_y + diamond_s + 30
    left_cx = 175  # fail path center
    right_cx = 525  # pass path center
    branch_w = 260
    branch_h = 200

    # Arrows from diamond to branches
    lines.append(f'  <path d="M{cx-diamond_s},{diamond_y} Q{left_cx},{diamond_y + 20} {left_cx},{branch_y}"'
                 f' fill="none" stroke="{FAIL_COLOR}" stroke-width="1.5"/>')
    lines.append(f'  <polygon points="{left_cx-4},{branch_y-4} {left_cx+4},{branch_y-4} {left_cx},{branch_y+2}"'
                 f' fill="{FAIL_COLOR}"/>')

    lines.append(f'  <path d="M{cx+diamond_s},{diamond_y} Q{right_cx},{diamond_y + 20} {right_cx},{branch_y}"'
                 f' fill="none" stroke="{PASS_COLOR}" stroke-width="1.5"/>')
    lines.append(f'  <polygon points="{right_cx-4},{branch_y-4} {right_cx+4},{branch_y-4} {right_cx},{branch_y+2}"'
                 f' fill="{PASS_COLOR}"/>')

    # ── Left branch: early testing (fail) ──────────────────────────────
    FAIL_SOFT = "#f5eaea"  # soft red derived from FAIL_COLOR
    lx = left_cx - branch_w / 2
    lines.append(f'  <rect x="{lx}" y="{branch_y}" width="{branch_w}"'
                 f' height="{branch_h}" rx="8" fill="{FAIL_SOFT}" stroke="{FAIL_COLOR}"'
                 f' stroke-width="1.5"/>')

    # Header
    lines.append(f'  <text x="{left_cx}" y="{branch_y + 24}" text-anchor="middle"'
                 f' font-size="14" font-weight="700" fill="{FAIL_COLOR}">'
                 f'Test early, get stuck</text>')
    lines.append(f'  <text x="{left_cx}" y="{branch_y + 42}" text-anchor="middle"'
                 f' font-size="11" fill="{FAIL_COLOR}">'
                 f'{early_total} runs \u00b7 {early_pass}/{early_total} pass'
                 f' ({100*early_pass//max(early_total,1)}%)</text>')

    # Flow steps
    flow_y = branch_y + 60
    early_flow = [
        ("test early", FAIL_COLOR, "step 5\u201313"),
        ("bash loops", FAIL_COLOR, f"avg {early_avg_bash:.0f} bash steps"),
        ("few edits", ACCENT, f"avg {early_avg_edits:.0f} edit steps"),
        ("barely re-test", FAIL_COLOR, f"avg {early_avg_tests:.0f} total tests"),
    ]
    for i, (label, color, detail) in enumerate(early_flow):
        fy = flow_y + i * 30
        # Step box
        lines.append(f'  <rect x="{lx + 12}" y="{fy}" width="{branch_w - 24}"'
                     f' height="{22}" rx="4" fill="{BG}" stroke="{color}"'
                     f' stroke-width="1" opacity="0.9"/>')
        lines.append(f'  <text x="{lx + 22}" y="{fy + 15}" font-size="11"'
                     f' font-weight="600" fill="{color}">{_esc(label)}</text>')
        lines.append(f'  <text x="{lx + branch_w - 16}" y="{fy + 15}"'
                     f' text-anchor="end" font-size="9" fill="{TEXT_MUTED}">{_esc(detail)}</text>')
        # Arrow between steps
        if i < len(early_flow) - 1:
            lines.append(f'  <line x1="{left_cx}" y1="{fy + 22}" x2="{left_cx}" y2="{fy + 30}"'
                         f' stroke="{BORDER}" stroke-width="1"/>')

    # ── Right branch: late testing (pass) ──────────────────────────────
    rx = right_cx - branch_w / 2
    lines.append(f'  <rect x="{rx}" y="{branch_y}" width="{branch_w}"'
                 f' height="{branch_h}" rx="8" fill="{ACCENT_SOFT}" stroke="{PASS_COLOR}"'
                 f' stroke-width="1.5"/>')

    # Header
    lines.append(f'  <text x="{right_cx}" y="{branch_y + 24}" text-anchor="middle"'
                 f' font-size="14" font-weight="700" fill="{PASS_COLOR}">'
                 f'Explore first, test after editing</text>')
    lines.append(f'  <text x="{right_cx}" y="{branch_y + 42}" text-anchor="middle"'
                 f' font-size="11" fill="{PASS_COLOR}">'
                 f'{late_total} runs \u00b7 {late_pass}/{late_total} pass'
                 f' ({100*late_pass//max(late_total,1)}%)</text>')

    # Flow steps
    late_flow = [
        ("search deeply", ACCENT, "read + grep"),
        ("edit with context", ACCENT, f"avg {late_avg_edits:.0f} edit steps"),
        ("test after changes", PASS_COLOR, f"avg {late_avg_tests:.0f} total tests"),
        ("iterate: edit \u2192 test", PASS_COLOR, "converge on fix"),
    ]
    for i, (label, color, detail) in enumerate(late_flow):
        fy = flow_y + i * 30
        lines.append(f'  <rect x="{rx + 12}" y="{fy}" width="{branch_w - 24}"'
                     f' height="{22}" rx="4" fill="{BG}" stroke="{color}"'
                     f' stroke-width="1" opacity="0.9"/>')
        lines.append(f'  <text x="{rx + 22}" y="{fy + 15}" font-size="11"'
                     f' font-weight="600" fill="{color}">{_esc(label)}</text>')
        lines.append(f'  <text x="{rx + branch_w - 16}" y="{fy + 15}"'
                     f' text-anchor="end" font-size="9" fill="{TEXT_MUTED}">{_esc(detail)}</text>')
        if i < len(late_flow) - 1:
            lines.append(f'  <line x1="{right_cx}" y1="{fy + 22}" x2="{right_cx}" y2="{fy + 30}"'
                         f' stroke="{BORDER}" stroke-width="1"/>')

    # ── Bottom annotation ──────────────────────────────────────────────
    lines.append(f'  <text x="{cx}" y="{H - 10}" text-anchor="middle"'
                 f' font-size="11" fill="{TEXT_MID}">'
                 f'The agent doesn\u2019t choose a strategy. Stochastic variation determines which path it takes.</text>')

    lines.append("</svg>")
    return "\n".join(lines)


if __name__ == "__main__":
    runs = load_runs()
    print(f"Loaded {len(runs)} runs")

    strategies = [classify_strategy(r) for r in runs]
    late = [s for s in strategies if s["centroid"] > 0.5]
    early = [s for s in strategies if s["centroid"] <= 0.5]
    print(f"Late testers: {len(late)} ({sum(1 for s in late if s['success'])}/{len(late)} pass)")
    print(f"Early testers: {len(early)} ({sum(1 for s in early if s['success'])}/{len(early)} pass)")

    svg = generate_flow(runs)
    out_dir = os.path.join(os.path.dirname(__file__), "blog_output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "panel2b_flow.svg")
    with open(out_path, "w") as f:
        f.write(svg)
    print(f"Written to {out_path} ({len(svg)} bytes)")
