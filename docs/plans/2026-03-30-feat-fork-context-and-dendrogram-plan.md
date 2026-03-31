---
title: Fork context and dendrogram heatmap
type: feat
date: 2026-03-30
spec: docs/superpowers/specs/2026-03-30-fork-context-and-dendrogram-design.md
---

# Fork context and dendrogram heatmap

Close the gap between "moirai found a fork" and "I know what to change." Two phases: enrich the converter with agent reasoning, then overhaul the HTML visualization (fork cards + dendrogram in one pass).

## Phase 1: Eval-harness converter enrichment

Add `output.reasoning` and `output.result` to each step by parsing Claude Code JSONL transcripts.

**File:** `scripts/convert_eval_harness.py`

- [x] Refactor `main()` to use argparse (currently `sys.argv`). Add `--transcripts` flag (default: `~/.claude/projects/`)
- [x] Add `find_transcript(trial_path, transcripts_dir)` — glob for matching workspace directory by exact path pattern, return JSONL path or None. Warn to stderr when multiple matches found.
- [x] Add `parse_transcript(jsonl_path)` — read JSONL, extract per-`tool_use` reasoning from the nearest preceding `thinking`/`text` block in the same content array. Match `tool_result` back to `tool_use` by `tool_use_id`. Skip malformed lines.
- [x] Modify `convert_trial()` — merge transcript reasoning into log-parsed steps by sequential position. Filter `[result]` system steps before merging (they have no transcript counterpart). Tool name is a sanity-check assertion: when it fails, stop merging at that point and warn (don't silently attach wrong reasoning to remaining steps).

### Tests

**File:** `tests/test_convert_eval_harness.py` (new)

- [x] `parse_transcript` on a synthetic JSONL with multi-tool-use message — verify each tool_use gets its own reasoning
- [x] `convert_trial` merge path — mock log + mock JSONL, verify reasoning lands on correct steps
- [x] `convert_trial` without transcript — fallback to log-only, no crash

**Commit:** `feat: eval-harness converter — extract reasoning from Claude Code transcripts`

---

## Phase 2: Visualization overhaul (fork cards + dendrogram)

Replace narrative findings + decision trees + flat heatmap with fork cards and dendrogram+heatmap in one pass over `html.py`.

### Narrate changes

**File:** `moirai/analyze/narrate.py`

- [x] Add `reasoning: str | None = None` to `StepDetail` and `BranchExample`
- [x] In `_build_branches()`, populate reasoning from the representative run's fork step `output.get("reasoning")`

### HTML overhaul

**File:** `moirai/viz/html.py`

**Fork cards (replacing narrative + decision tree):**
- [x] Remove `_build_divergence_tree()`, `_get_context_str()`, tree CSS classes
- [x] Replace per-task narrative rendering with fork cards: header (summary + p-value), per-branch windowed trajectory (slice around `fork_position`, clamped), reasoning excerpt when available, run ID
- [x] Drop `recommendation` from fork card display

**Dendrogram+heatmap (replacing flat matrix):**
- [x] Add `_scipy_coords_to_svg()` — map scipy dendrogram coordinates to SVG paths. Guard zero max distance.
- [x] Add `_build_dendrogram_heatmap(alignment, runs, points)` — compute `distance_matrix(runs, level="name")`, run `linkage` + `dendrogram(no_plot=True)`, render dendrogram + outcome strip + reordered heatmap + divergence tick marks as single SVG. Import scipy lazily inside the function. 1 run: skip dendrogram.
- [x] Remove `_build_trajectory_matrix()`

### Tests

**File:** `tests/test_narrate.py` (new)

- [x] `narrate_task` with reasoning present — verify `BranchExample.reasoning` populated
- [x] `narrate_task` without reasoning — verify graceful degradation (None, no crash)

**File:** `tests/test_html.py` (new)

- [x] `_build_dendrogram_heatmap` with 3+ runs — SVG contains dendrogram paths, outcome strip, heatmap
- [x] `_build_dendrogram_heatmap` with 1 run — no crash, no dendrogram paths
- [x] Identical trajectories (zero distances) — no division-by-zero crash

**Commit:** `feat: fork cards + dendrogram heatmap`

---

## Acceptance criteria

- [x] `moirai branch runs/ --html report.html` produces fork cards with reasoning excerpts (when available)
- [x] Fork cards degrade gracefully without reasoning (attrs context only)
- [x] Alignment matrix shows dendrogram on left, outcome strip, rows ordered by behavioral similarity
- [x] Decision trees fully removed from HTML output
