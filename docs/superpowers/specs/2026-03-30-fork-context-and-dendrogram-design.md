# Fork context, targeted diff, and dendrogram heatmap

Date: 2026-03-30

## Problem

moirai identifies divergence points — forks where agent behavior predicts success or failure. But identifying the fork isn't enough. The current output says "bash(python) → 0%, read(config) → 100%" and stops. The user is left asking "so what?"

Three gaps:

1. **No WHY at the fork.** We show which tool the agent chose, but not what it was thinking or what it had just seen. Without reasoning context, you can't design an intervention.
2. **No verification loop.** After applying an intervention (prompt change, tool restriction), there's no way to check whether the specific fork changed without re-running full analysis and eyeballing the difference.
3. **No structural overview.** The trajectory heatmap rows are sorted by pass/fail, not by behavioral similarity. You can't see whether similar-behaving runs cluster together or whether outcome correlates with trajectory structure.

## Scope

Four changes, built in dependency order:

1. **Converter enrichment** — capture agent reasoning and richer attrs from raw traces
2. **Fork cards** — rich divergence context with reasoning excerpts in the HTML report
3. **Targeted diff** — fork-level before/after comparison
4. **Dendrogram + heatmap** — scipy hierarchical clustering tree beside the alignment matrix

## 1. Converter enrichment

### Eval-harness converter

The raw Claude Code transcripts live in `~/.claude/projects/`, organized by workspace path. Each eval-harness run has a JSONL transcript file containing the full conversation: thinking tokens, tool calls, tool results, and visible reasoning.

JSONL message structure:
```json
{
  "type": "assistant",
  "message": {
    "role": "assistant",
    "content": [
      {"type": "thinking", "thinking": "I should check the config..."},
      {"type": "tool_use", "name": "Read", "input": {"file_path": "/path/to/file"}}
    ]
  }
}
```

Changes to `scripts/convert_eval_harness.py`:

- Add `--transcripts` flag (defaults to `~/.claude/projects/`)
- For each trial, match the workspace path to find the corresponding JSONL transcript. The workspace directory names follow the pattern `-Users-ryo-dev-intent-layer-eval-harness-workspaces-{task}-{uuid}-{condition}-r{rep}/`
- Parse the JSONL and extract per-step:
  - `output.reasoning`: the last `thinking` or `text` block before each `tool_use` — the agent's intent ("let me test this fix", "I should check the config first")
  - `output.result`: summary of the tool result from the next user message (first 300 chars of tool output)
- Fall back to current log-only behavior when no transcript found
- Existing attrs handling (file_path, pattern, command) stays the same

### SWE-smith converter

Changes to `scripts/convert_swe_smith.py`:

- Parse file paths and commands from action content → populate `attrs` (currently always `{}`)
  - Extract file paths from `str_replace_editor` arguments
  - Extract commands from bash execution content
- Keep full content in `output.reasoning` instead of 150-char `output.summary` truncation
- Better observation classification:
  - Directory listings and file content → `system/observation` with `status: "ok"`, not `error_observation`
  - Reserve `error_observation` for actual errors/tracebacks
  - Keep test results as `judge/test_result` (existing)

### Schema

No changes needed. `Step.input` and `Step.output` dicts already exist. Convention:
- `output.reasoning` — agent's reasoning/intent before this step
- `output.result` — tool output summary after this step

## 2. Fork cards

Replace the current narrative findings + collapsible tree with richer fork cards as the primary visualization for each divergence point.

### What a fork card shows

For each divergence point within a task:

**Header:**
- Fork position, p-value, one-line summary (from existing `narrate_task`)

**Per branch (2+ branches, sorted by success rate):**
- Success rate badge + run count
- 3-5 steps before the fork with enriched names + attrs detail (file names, commands)
- The fork step itself, highlighted
- Agent reasoning excerpt at the fork step (`output.reasoning`) if available — the new part
- 3-5 steps after the fork showing what happened next
- Representative run ID for trace lookup

### Example

```
Fork at step 15 (p=0.001)

┌─ 100% success (3 runs)
│  ...read(source) utils.py → read(source) helpers.py →
│  [read(config) setup.cfg] → read(test_file) test_utils.py → test pytest → PASS
│
│  "Before making changes, I should understand the project's
│   test configuration to know how to verify my fix."
│
│  Run: ansible_ansible-85709-none-r2

┌─ 0% success (4 runs)
│  ...read(source) utils.py → read(source) helpers.py →
│  [bash(python) python test_fix.py] → bash(python) python test_fix.py →
│  bash(python) python -c "import..." → FAIL
│
│  "Let me quickly verify this works by running the file."
│
│  Run: ansible_ansible-85709-none-r0
```

### Code changes

- `analyze/narrate.py`:
  - Add `reasoning` field to `StepDetail` dataclass
  - Extract `output.reasoning` from the representative run's step at the fork position
  - Add `reasoning` field to `BranchExample`
- `viz/html.py`:
  - Replace the current narrative + collapsible `<details>` tree section with fork cards as the primary per-task view
  - The alignment matrix moves below as supporting evidence (still in `<details>`)
  - Remove `_build_divergence_tree()` entirely

## 3. Targeted diff

### The verification loop

After applying an intervention and re-running evals:

```
moirai diff --a runs_before/ --b runs_after/
```

Currently compares aggregate metrics and cluster shifts. Add a new section: **fork comparison**.

### How it works

For each task with mixed outcomes in either cohort:

1. Run `find_divergence_points()` on both cohorts independently (per-task)
2. Match forks between cohorts by: task_id + approximate column position (within 3) + overlapping branch values
3. Classify each fork:
   - **Resolved** — fork existed in A, gone in B (agents stopped taking the bad branch)
   - **Improved** — fork exists in both, but success rate shifted toward the good branch
   - **New** — fork exists in B but not A (regression: a new failure mode)
   - **Unchanged** — fork exists in both, same distribution

### Output

```
Fork changes (before → after):

ansible_ansible-85709:
  ✓ RESOLVED  step 15: bash(python) vs read(config)
              before: 0% vs 100% (7 runs)
              after: fork gone — all runs chose read(config)

ansible_ansible-85516:
  ✗ NEW       step 22: edit(source) vs search(glob)
              0% vs 75% (8 runs, p=0.04)
              not present in before cohort

Overall: 3 resolved, 1 new, 2 unchanged
```

### Code changes

- `analyze/compare.py`: Add `compare_forks(a_runs, b_runs, p_threshold=0.2)` that runs per-task divergence on both cohorts and matches forks
- New dataclass `ForkComparison` with fields: task_id, status (resolved/improved/new/unchanged), before_fork, after_fork
- `cli.py`: Add fork comparison section to `diff` command output
- `viz/terminal.py`: `print_fork_comparison()` for CLI display
- `viz/html.py`: Add fork comparison section to diff HTML output

## 4. Dendrogram + heatmap

Replace the flat pass/fail-sorted alignment matrix with a dendrogram + heatmap per task section.

### Layout (left to right)

1. **Dendrogram** (~120px) — scipy hierarchical clustering tree
2. **Outcome strip** (~12px) — single column: green = pass, red = fail
3. **Run labels** (~180px) — truncated run ID
4. **Heatmap** (remaining width) — aligned trajectory matrix, rows reordered by dendrogram leaf order

### How it works

For each task's runs:

1. Compute pairwise NW distance matrix (`align.distance_matrix()`, already exists)
2. `scipy.cluster.hierarchy.linkage(condensed, method="average")` → linkage matrix Z
3. `scipy.cluster.hierarchy.dendrogram(Z, no_plot=True)` → `icoord`/`dcoord` bracket coordinates, `leaves` row ordering
4. Reorder alignment matrix rows to match `leaves`
5. Render dendrogram as SVG `<path>` elements, vertically aligned with heatmap row centers
6. Render outcome strip as thin colored rectangles between dendrogram and labels
7. Render heatmap with the new row order

### What this gives you

Rows ordered by behavioral similarity instead of pass/fail. If the dendrogram splits cleanly into a green subtree and a red subtree, trajectory structure predicts outcome. If pass/fail are interleaved, the divergence is more subtle than gross structural differences.

Red tick marks on the column header at divergence point positions connect the dendrogram view back to the fork cards above.

### Dendrogram rendering

- Use scipy `dendrogram(Z, no_plot=True)` to get exact bracket coordinates
- Map `icoord` (y-axis, leaf positions) and `dcoord` (x-axis, merge distances) to SVG coordinates
- Black lines, 1px stroke
- Branch heights proportional to merge distance
- Leaf nodes align exactly with heatmap row centers
- ~40 lines of SVG generation code

### Code changes

- `viz/html.py`:
  - New `_build_dendrogram_heatmap(alignment, runs)` replaces `_build_trajectory_matrix()`
  - Computes distance matrix and linkage internally (for rendering only)
  - Renders dendrogram SVG paths + outcome strip + reordered heatmap as a single SVG
  - Remove `_build_divergence_tree()` entirely (replaced by fork cards in section 2)
- `align.py`: No changes — `distance_matrix()` already public
- No new dependencies — scipy already in use

## Build sequence

1. **Converters** — enrich eval-harness and SWE-smith converters, re-convert existing data
2. **Fork cards** — update narrate.py and html.py, replace current narrative display
3. **Targeted diff** — add compare_forks to compare.py, wire into diff command
4. **Dendrogram** — replace trajectory matrix with dendrogram+heatmap

Each step is independently useful. Tests at each step.

## Files changed

| File | Change |
|------|--------|
| `scripts/convert_eval_harness.py` | Add `--transcripts` flag, parse JSONL for reasoning/results |
| `scripts/convert_swe_smith.py` | Extract attrs, keep full reasoning, better observation classification |
| `moirai/analyze/narrate.py` | Add reasoning to StepDetail and BranchExample |
| `moirai/analyze/compare.py` | Add `compare_forks()` and `ForkComparison` dataclass |
| `moirai/viz/html.py` | Fork cards, dendrogram+heatmap, remove decision trees |
| `moirai/viz/terminal.py` | Add `print_fork_comparison()` |
| `moirai/cli.py` | Wire fork comparison into diff command |
| `moirai/schema.py` | Add `ForkComparison` dataclass |

## Edge cases and fallbacks

**Transcript matching (converter):**
- Workspace path pattern may not match every trial. Try exact match first, then fuzzy match on task_id + condition + rep. If no match found, fall back to log-only behavior and emit a warning.
- JSONL files may contain multiple sessions. Use the session with the most tool_use messages.

**Missing reasoning (fork cards):**
- When `output.reasoning` is empty (log-only conversion, or sparse traces), show the fork card without the reasoning block. The attrs context (file names, commands) is still useful on its own.

**Small task groups (dendrogram):**
- 1 run: skip dendrogram entirely, show flat heatmap row.
- 2 runs: dendrogram is trivial (single merge). Still render it — the outcome strip still communicates pass/fail.
- 50+ runs: cap SVG height and reduce cell height. Dendrogram remains readable because scipy handles the layout.

**Fork matching (targeted diff):**
- Match forks between cohorts by: same task_id, column position within ±3, and at least one shared branch value.
- When a task exists in only one cohort, skip it (can't compare).
- When a task has no mixed outcomes in either cohort, skip it (no forks to compare).

**SWE-smith attrs extraction:**
- Content parsing for file paths is heuristic. When parsing fails, leave attrs empty and fall back to current behavior. Don't crash on malformed content.

## What stays the same

- Core analysis pipeline (align, cluster, divergence, motifs)
- Step schema (no new fields)
- All existing CLI commands and flags
- Patterns table in HTML report
- Terminal output for non-diff commands
