---
title: "feat: unified branch command with sharp wedge UX"
type: feat
date: 2026-04-07
---

# Unified Branch Command — Wedge UX

## Overview

Upgrade the existing `moirai branch` command into a single, demo-ready entry point that answers one question in under 15 seconds: **"Where did runs diverge, and which path worked?"**

The pipeline pieces exist (`align_runs`, `find_divergence_points`, `find_split_divergences`, `write_branch_html`). This plan composes them into a sharper UX with two new layers: plain-language summaries and branch cards (terminal + HTML).

## Problem Statement

The current `branch` command outputs raw divergence points with q-values and value counts. A user has to mentally reconstruct what happened. The HTML report shows dendrogram + heatmap SVGs and comparison triples — useful for deep analysis but not for the "15-second reaction" the wedge needs.

The spec demands a user opens the report and immediately sees: runs diverged, where, which path worked. If that reaction isn't immediate, it's not done.

## Proposed Solution

No new modules. One new function in `divergence.py`, CLI upgrades, HTML template upgrade.

### Review-driven scope cuts

Per plan review (DHH, Kieran, Simplicity — all three unanimous):

- **Cut `confidence.py` entirely** — YAGNI. Scores individual runs, not divergence points. Doesn't make "these runs split here, and it mattered" clearer.
- **No `summarize.py` module** — put `summarize_point()` in `divergence.py` next to the data it describes.
- **Start with 2 summary patterns, not 5** — GAP detection + generic fallback. Add more when real output shows the fallback fires too often.
- **JSON output early** — it's the composability wedge (~20 lines of code).

## Technical Approach

### Phase 1: Summaries + JSON output

**Add `summarize_point()` to `moirai/analyze/divergence.py`**

Takes a `DivergencePoint` + the alignment matrix row data at that column. Produces a plain-language summary string.

Two patterns to start:

| Pattern | Detection | Example output |
|---------|-----------|----------------|
| GAP (presence vs absence) | One variant value is `"-"` (GAP) | "Failing runs skip the search step at position 12" |
| Generic fallback | Always fires | "Runs split at position 7: read(test_file) (80% pass) vs edit(source) (17% pass)" |

```python
# In moirai/analyze/divergence.py

def summarize_point(point: DivergencePoint) -> str:
    """Plain-language summary of a divergence point.

    Uses value_counts, success_by_value, and phase_context
    already on the DivergencePoint — no extra inputs needed.
    """
    ...
```

Signature is tight: only takes `DivergencePoint`. All data needed (value_counts, success_by_value, phase_context) is already on the object. No `list[Run]` parameter — that was flagged as a testing burden.

- [x] `summarize_point(point)` — GAP pattern + generic fallback
- [x] Tests in `tests/test_divergence.py` (extend existing test file)

**Add `--json PATH` flag to `branch` command**

JSON output format:
```json
{
  "tasks": [
    {
      "task_id": "...",
      "n_runs": 11,
      "n_pass": 6,
      "n_fail": 5,
      "branch_points": [
        {
          "position": 7,
          "q_value": 0.003,
          "summary": "Runs split at position 7: read(test_file) (80% pass) vs edit(source) (17% pass)",
          "variants": [
            {"value": "read(test_file)", "n_runs": 5, "n_pass": 4, "pass_rate": 0.8},
            {"value": "edit(source)", "n_runs": 6, "n_pass": 1, "pass_rate": 0.167}
          ]
        }
      ]
    }
  ]
}
```

- [x] `--json PATH` flag on `branch` command
- [x] Structured JSON output with summaries

### Phase 2: Terminal branch cards

Replace the raw divergence point output (cli.py lines 273-288) with compact cards:

```
━━━ Branch 1 (q=0.003) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"Runs split at position 7: read(test_file) (80% pass) vs edit(source) (17% pass)"

  read(test_file): 5 runs, 4 pass / 1 fail (80%)
  edit(source):    6 runs, 1 pass / 5 fail (17%)

  Context: ... search(glob) → read(source) → [SPLIT] → edit → test ...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Inline in the `branch` command for now. Extract to `terminal.py` later if a second caller appears.

- [x] Branch card rendering in `branch` command
- [x] Context snippet: 2 steps before/after the split column from the alignment matrix

### Phase 3: HTML branch cards + golden test

**HTML branch cards in `moirai/viz/html.py` and `templates/branch.html`**

Add branch card data to the JSON payload alongside existing dendrogram/comparison data. Cards appear above the existing views as the first thing a reader sees.

Branch Card HTML structure:
- Summary sentence (from `summarize_point()`)
- Variant breakdown: value, run count, pass/fail, pass rate bar
- Context snippet: steps before → split → steps after

- [x] `_build_branch_card_data(point, summary)` in `html.py`
- [x] Add `branch_cards` list to per-task data in `_build_task_data()`
- [x] Template component in `branch.html` for branch cards

**Golden test**

Synthetic dataset with known divergence:
- 5 runs, same task
- 3 pass: read → read(test) → edit → test(pass)
- 2 fail: read → edit → test(fail) → edit → test(fail)
- Known split at position 1: read(test) vs edit

Assert: branch detected, variants grouped correctly, summary text present, JSON parseable.

- [x] `tests/test_branch_e2e.py` with synthetic fixture
- [x] Assert detection, grouping, summary, JSON schema

## Acceptance Criteria

### Functional
- [x] `moirai branch path/` runs end-to-end
- [x] Terminal output shows branch cards with summaries
- [x] `--html` produces HTML with branch cards + existing dendrogram
- [x] `--json` produces valid JSON with branch points, variants, summaries

### Analytical
- [x] Detects divergence points on existing SWE-rebench data
- [x] Summaries are readable without statistical knowledge

### UX
- [ ] A reader can answer "where did runs diverge?" from terminal or HTML output
- [ ] A reader can answer "which path worked?" from the branch card alone

## Implementation Order

1. `summarize_point()` in `divergence.py` + tests
2. `--json` flag with structured output
3. Terminal branch cards (inline in `branch` command)
4. HTML branch cards (data builder + template)
5. Golden test (e2e)

## References

- Existing `branch` command: `moirai/cli.py:162-299`
- Divergence: `moirai/analyze/divergence.py` — `find_divergence_points()`, `DivergencePoint`
- Splits: `moirai/analyze/splits.py` — `find_split_divergences()`, `SplitDivergence`
- HTML: `moirai/viz/html.py` — `write_branch_html()`, `_build_comparison_data()`
- Schema: `moirai/schema.py` — `Run`, `Step`, `Alignment`, `DivergencePoint`
- Compress: `moirai/compress.py` — `step_enriched_name()`, `PHASE_MAP`
