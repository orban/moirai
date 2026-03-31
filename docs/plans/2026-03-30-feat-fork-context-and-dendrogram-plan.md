---
title: Fork context and dendrogram heatmap
type: feat
date: 2026-03-30
spec: docs/superpowers/specs/2026-03-30-fork-context-and-dendrogram-design.md
---

# Fork context and dendrogram heatmap

## Overview

Close the gap between "moirai found a fork" and "I know what to change." Three changes in dependency order: enrich the eval-harness converter with agent reasoning from Claude Code transcripts, build fork cards that show WHY agents diverge (not just WHERE), and add a dendrogram+heatmap visualization that shows behavioral clustering.

## Phase 1: Eval-harness converter enrichment

**Goal:** Add `output.reasoning` and `output.result` to each step by parsing Claude Code JSONL transcripts.

### 1.1 JSONL transcript parser

Create a transcript parsing module that the converter can call.

**File:** `scripts/convert_eval_harness.py` (extend existing)

**Tasks:**
- [ ] Add `--transcripts` CLI flag via argparse (default: `~/.claude/projects/`)
- [ ] Add `find_transcript(trial_path, transcripts_dir)` → `Path | None`
  - Build expected workspace directory name from trial metadata: `-Users-ryo-dev-intent-layer-eval-harness-workspaces-{task}-*-{condition}-r{rep}/`
  - Glob for matching directories in transcripts_dir
  - Find the JSONL file inside (may be in root or `subagents/` subdirectory — pick the one with the most lines)
  - Return None if no match (fall back to log-only)
- [ ] Add `parse_transcript(jsonl_path)` → `list[dict]`
  - Read JSONL line by line
  - For each `type: "assistant"` message, walk the `content` array
  - For each `tool_use` block, walk backward to find the nearest preceding `thinking` or `text` block in the same content array
  - Build step dicts with `output.reasoning` (truncated to 500 chars) from that block
  - For each `type: "user"` message after a tool_use, extract `tool_result` content as `output.result` (first 300 chars)
  - Match each `tool_result` back to its `tool_use` by `tool_use_id` (not by position)
  - Map tool names: `Read` → `read`, `Edit` → `edit`, `Write` → `write`, `Bash` → `bash`, `Grep`/`Glob` → `search`, `Agent` → `subagent`, `TodoWrite` → `plan`
  - Prefer `thinking` over `text` when both precede a `tool_use`; use whichever is closest
  - Skip malformed JSONL lines (warn to stderr, don't crash)
- [ ] Modify `convert_trial()` to accept optional `transcripts_dir`
  - If transcripts_dir provided, call `find_transcript()` then `parse_transcript()`
  - Merge reasoning/result into log-parsed steps by **sequential position**: first log step maps to first transcript tool_use, second to second, etc. Tool name serves as a sanity-check assertion, not the primary matching key.
  - If no transcript found, fall back to current log-only behavior with a warning to stderr
- [ ] Refactor `main()` to use argparse (currently uses `sys.argv` directly) and pass `--transcripts` through

### 1.2 Tests

**File:** `tests/test_convert_eval_harness.py` (new)

- [ ] Test `parse_transcript()` with a synthetic JSONL file containing:
  - Assistant message with thinking + tool_use + thinking + tool_use (multiple tool calls per message)
  - User message with tool_result
  - Verify per-tool_use reasoning extraction (each tool_use gets its own preceding thinking block)
- [ ] Test `find_transcript()` with a mock directory structure
  - Exact match found → returns path
  - No match → returns None
- [ ] Test reasoning truncation at 500 chars
- [ ] Test result truncation at 300 chars
- [ ] Test `tool_result` matching by `tool_use_id`
- [ ] Test fallback when no transcript directory exists
- [ ] Test malformed JSONL lines are skipped without crashing
- [ ] Test `convert_trial()` merge path: mock log + mock JSONL for the same session, verify reasoning lands on correct steps by sequential position

### 1.3 Re-convert existing data

- [ ] Run the enriched converter on existing eval-harness data: `python scripts/convert_eval_harness.py /path/to/eval-harness examples/eval_harness/ --transcripts ~/.claude/projects/`
- [ ] Spot-check a few output files to verify `output.reasoning` is populated
- [ ] Verify existing moirai commands still work on the enriched data (`moirai validate`, `moirai branch`)

**Commit:** `feat: eval-harness converter — extract reasoning from Claude Code transcripts`

---

## Phase 2: Fork cards

**Goal:** Replace narrative findings + decision trees with fork cards showing windowed trajectories and agent reasoning.

### 2.1 Narrate module changes

**File:** `moirai/analyze/narrate.py`

- [ ] Add `reasoning: str | None = None` field to `StepDetail` dataclass
- [ ] Add `reasoning: str | None = None` field to `BranchExample` dataclass
- [ ] In `_build_branches()`, after building the step list for the representative run:
  - Find the step at `fork_step_idx` in the representative run's actual steps (filtering noise as currently done)
  - Extract `step.output.get("reasoning")` if available
  - Set it on the `BranchExample`
- [ ] In `_build_branches()`, also populate `StepDetail.reasoning` for the fork step specifically

### 2.2 HTML fork card rendering

**File:** `moirai/viz/html.py`

- [ ] Remove `_build_divergence_tree()` function entirely
- [ ] Remove `_get_context_str()` helper function
- [ ] Remove unused CSS classes: `.tree-node`, `.tree-branch`, `.tree-label`, `.tree-stats`, `.tree-context`
- [ ] Replace the current per-task narrative rendering (lines ~66-124 of `write_branch_html`) with fork card rendering:
  - For each finding, render a card with:
    - Header: summary line + p-value
    - Per branch: success rate badge, windowed trajectory (slice `branch.steps[max(0, fork_position - 4) : fork_position + 5]`), fork step highlighted, reasoning excerpt if available, run ID
  - Windowing happens here in the renderer, not in narrate.py — clamp to list bounds for forks near start/end of trajectory
  - When `reasoning` is None, skip the reasoning block (don't show an empty quote)
  - Drop `finding.recommendation` from fork card display (keep in `Finding` dataclass for terminal output)
- [ ] Keep the alignment matrix in a `<details>` collapsible below the fork cards (still uses `_build_trajectory_matrix` in Phase 2; Phase 3 replaces it)
- [ ] Remove the separate decision tree `<details>` section

### 2.3 Tests

**File:** `tests/test_narrate.py` (new)

- [ ] Test `narrate_task()` produces findings with reasoning when steps have `output.reasoning`
- [ ] Test `narrate_task()` produces findings without reasoning when steps lack `output.reasoning`
- [ ] Test `BranchExample.reasoning` is populated from the fork step's output
- [ ] Test with empty points list → returns `[]`
- [ ] Test with single branch → returns `[]` (need 2+ branches)
- [ ] Test fork at position 0 (first step) — windowing clamps correctly
- [ ] Test fork at last step in trajectory — windowing clamps correctly

**File:** `tests/test_cli.py` (extend)

- [ ] Test `branch --html` produces HTML with fork card markup (check for reasoning CSS class or structure)
- [ ] Verify decision tree markup is NOT present in output

**Commit:** `feat: fork cards — divergence context with agent reasoning excerpts`

---

## Phase 3: Dendrogram + heatmap

**Goal:** Replace the flat pass/fail-sorted alignment matrix with a dendrogram-ordered heatmap showing behavioral clustering.

### 3.1 Dendrogram heatmap builder

**File:** `moirai/viz/html.py`

- [ ] Add `_scipy_coords_to_svg(icoord, dcoord, leaves, cell_h, dendro_w)` helper:
  - scipy `dendrogram(no_plot=True)` returns `icoord` and `dcoord` as lists of 4-element lists (bracket coordinates)
  - Default leaf spacing is 5, 15, 25... (increments of 10, starting at 5)
  - Map y-coordinates: `svg_y = (scipy_y - 5) / 10 * cell_h + cell_h / 2`
  - Map x-coordinates: scale `dcoord` values to fit within `dendro_w`, with 0 (leaves) on the right and max merge distance on the left. Guard against zero max distance (all identical trajectories) by rendering a flat dendrogram.
  - Return list of SVG `<path>` elements (U-shaped brackets: down, across, up)
- [ ] Add `_build_dendrogram_heatmap(alignment, runs, points)` → `str`:
  - Compute `distance_matrix(runs, level="name")` — must match alignment level
  - Handle edge cases: 1 run → skip dendrogram, render flat row. Need 2+ runs for linkage.
  - `linkage(condensed, method="average")` → Z
  - `dendrogram(Z, no_plot=True)` → coords + `leaves` ordering
  - Reorder alignment matrix rows by `leaves`
  - Compute layout dimensions based on data size (as current `_build_trajectory_matrix` does)
  - Render single SVG containing:
    1. Dendrogram paths (from `_scipy_coords_to_svg`)
    2. Outcome strip: thin colored rectangles (green/red) per row
    3. Run labels (existing pattern)
    4. Heatmap cells (existing pattern from `_build_trajectory_matrix`, reordered)
    5. Divergence point tick marks on column header (red ticks at `point.column` positions)
- [ ] Add `from scipy.cluster.hierarchy import linkage, dendrogram as scipy_dendrogram` to html.py imports
- [ ] Remove `_build_trajectory_matrix()` (kept alive through Phase 2, replaced here)
- [ ] Update `write_branch_html()` to call `_build_dendrogram_heatmap` instead of `_build_trajectory_matrix`, passing `task_points`
- [ ] Use `alignment.level` for distance matrix computation (validate it's `"name"` to match heatmap content)

### 3.2 Tests

**File:** `tests/test_html.py` (new)

- [ ] Test `_scipy_coords_to_svg()` with known dendrogram output:
  - 3 items: verify produces 2 bracket paths (2 merges)
  - Verify leaf positions map correctly to SVG row centers
- [ ] Test `_build_dendrogram_heatmap()` with 1 run → returns SVG without dendrogram paths
- [ ] Test `_build_dendrogram_heatmap()` with 2 runs → trivial dendrogram (single merge), still renders
- [ ] Test `_build_dendrogram_heatmap()` with 3 runs → returns SVG containing dendrogram paths, outcome strip, and heatmap
- [ ] Test with identical trajectories (all distances zero) → no division-by-zero crash
- [ ] Test row ordering matches dendrogram `leaves` output
- [ ] Test divergence tick marks appear at correct column positions

**Commit:** `feat: dendrogram+heatmap — hierarchical clustering beside aligned trajectories`

---

## Acceptance criteria

- [ ] `moirai branch runs/ --html report.html` produces fork cards with agent reasoning excerpts (when transcript data available)
- [ ] Fork cards show windowed trajectory context (3-5 steps before/after fork) with actual file names and commands
- [ ] Fork cards degrade gracefully when no reasoning available (show attrs context only)
- [ ] Alignment matrix in HTML shows dendrogram on left, outcome strip, rows ordered by behavioral similarity
- [ ] 1-run and 2-run task groups render without errors
- [ ] Decision trees are fully removed from HTML output
- [ ] All existing tests pass
- [ ] New tests cover converter parsing, narrate reasoning, and dendrogram rendering

## Context

### Key files to modify

| File | Phase | Change |
|------|-------|--------|
| `scripts/convert_eval_harness.py` | 1 | Add `--transcripts`, JSONL parsing, reasoning extraction |
| `moirai/analyze/narrate.py` | 2 | Add reasoning fields to StepDetail and BranchExample |
| `moirai/viz/html.py` | 2+3 | Fork cards, dendrogram+heatmap, remove decision trees |

### Key files to create

| File | Phase | Purpose |
|------|-------|---------|
| `tests/test_convert_eval_harness.py` | 1 | Converter JSONL parsing tests |
| `tests/test_narrate.py` | 2 | Narrate reasoning extraction tests |
| `tests/test_html.py` | 3 | Dendrogram rendering tests |

### Key files for reference (read-only)

- `moirai/schema.py` — Step.output is `dict[str, Any]`, no changes needed
- `moirai/analyze/align.py` — `distance_matrix(runs, level)` already public
- `moirai/compress.py` — `step_enriched_name()`, `NOISE_STEPS`
- `moirai/cli.py` — no changes needed, fork cards and dendrogram are viz-layer only

### JSONL transcript location

`~/.claude/projects/-Users-ryo-dev-intent-layer-eval-harness-workspaces-{task}-{uuid}-{condition}-r{rep}/`

Each directory contains session JSONL files. Assistant messages have `content` arrays with `thinking`, `text`, and `tool_use` blocks. User messages carry `tool_result` blocks. Reasoning is extracted per-`tool_use` by walking backward in the content array.
