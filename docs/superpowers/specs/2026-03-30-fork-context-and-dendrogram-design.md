# Fork context and dendrogram heatmap

Date: 2026-03-30
Updated: 2026-03-30 (post-review: cut targeted diff, cut SWE-smith converter, trim implementation details)

## Problem

moirai identifies divergence points — forks where agent behavior predicts success or failure. But identifying the fork isn't enough. The current output says "bash(python) → 0%, read(config) → 100%" and stops. The user is left asking "so what?"

Two gaps:

1. **No WHY at the fork.** We show which tool the agent chose, but not what it was thinking or what it had just seen. Without reasoning context, you can't design an intervention.
2. **No structural overview.** The trajectory heatmap rows are sorted by pass/fail, not by behavioral similarity. You can't see whether similar-behaving runs cluster together or whether outcome correlates with trajectory structure.

## Scope

Three changes, built in dependency order:

1. **Eval-harness converter enrichment** — capture agent reasoning from Claude Code transcripts
2. **Fork cards** — rich divergence context with reasoning excerpts in the HTML report
3. **Dendrogram + heatmap** — scipy hierarchical clustering tree beside the alignment matrix

Deferred (build when needed):
- SWE-smith converter enrichment (no immediate data to test against)
- Targeted fork-level diff (build after we've used fork cards and know what the verification workflow needs)

## 1. Eval-harness converter enrichment

The raw Claude Code transcripts live in `~/.claude/projects/`, organized by workspace path. Each eval-harness run has a JSONL transcript containing thinking tokens, tool calls, tool results, and visible reasoning.

JSONL message structure (one assistant turn may contain multiple tool calls):
```json
{
  "type": "assistant",
  "message": {
    "content": [
      {"type": "thinking", "thinking": "I should check the config..."},
      {"type": "tool_use", "name": "Read", "input": {"file_path": "/path/to/file"}},
      {"type": "thinking", "thinking": "Now let me also check the test file..."},
      {"type": "tool_use", "name": "Read", "input": {"file_path": "/path/to/test"}}
    ]
  }
}
```

Changes to `scripts/convert_eval_harness.py`:

- Add `--transcripts` flag (defaults to `~/.claude/projects/`)
- For each trial, match the workspace directory by exact path pattern: `-Users-ryo-dev-intent-layer-eval-harness-workspaces-{task}-{uuid}-{condition}-r{rep}/`. Exact match only — no fuzzy matching. Warn and fall back to log-only if no match.
- Parse the JSONL and extract **per-`tool_use`** (not per-message):
  - `output.reasoning`: the `thinking` or `text` block immediately preceding this specific `tool_use` in the content array. Truncate to 500 chars.
  - `output.result`: first 300 chars of the corresponding tool result from the next user message.
- Existing attrs handling (file_path, pattern, command) stays the same.

No schema changes. `Step.output` is already `dict[str, Any]`. Convention: `output["reasoning"]` for intent, `output["result"]` for tool output.

## 2. Fork cards

Replace the current narrative findings + collapsible decision tree with fork cards as the primary per-task visualization.

### What a fork card shows

For each divergence point within a task:

**Header:** fork position, p-value, one-line summary (from existing `narrate_task`)

**Per branch (2+ branches, sorted by success rate):**
- Success rate badge + run count
- Windowed trajectory: 3-5 steps before and after the fork, with enriched names + attrs detail
- The fork step itself, highlighted
- Agent reasoning excerpt at the fork step if available
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
  - Add `reasoning: str | None = None` to `StepDetail`
  - Add `reasoning: str | None = None` to `BranchExample` (extracted from the fork step's `output.reasoning`)
  - `_build_branches` populates reasoning from the representative run's step at the fork position
  - `BranchExample.steps` stays as the full trajectory (no windowing in the data layer)
- `viz/html.py`:
  - Fork cards as the primary per-task view: render windowed trajectory (slice around `fork_position`) with reasoning block
  - Alignment matrix moves below in `<details>` (no longer a separate tree section)
  - Remove `_build_divergence_tree()` and its helper `_get_context_str()`
  - Remove unused CSS classes (`.tree-node`, `.tree-branch`, `.tree-label`, etc.)

When `output.reasoning` is empty (log-only conversion), show the fork card without the reasoning block. The attrs context (file names, commands) is still useful on its own.

## 3. Dendrogram + heatmap

Replace the flat pass/fail-sorted alignment matrix with a dendrogram + heatmap per task section.

### Layout (left to right)

1. **Dendrogram** — scipy hierarchical clustering tree
2. **Outcome strip** — single column: green = pass, red = fail
3. **Run labels** — truncated run ID
4. **Heatmap** — aligned trajectory matrix, rows reordered by dendrogram leaf order

### How it works

For each task's runs:

1. Compute pairwise NW distance matrix using `align.distance_matrix(runs, level="name")` — must use `"name"` level to match the heatmap's enriched-name alignment
2. `scipy.cluster.hierarchy.linkage(condensed, method="average")` → linkage matrix Z
3. `scipy.cluster.hierarchy.dendrogram(Z, no_plot=True)` → `icoord`/`dcoord` bracket coordinates, `leaves` row ordering
4. Reorder alignment matrix rows to match `leaves`
5. Map scipy's default leaf spacing (5, 15, 25...) to SVG row center positions via a `_scipy_coords_to_svg()` helper
6. Render dendrogram + outcome strip + reordered heatmap as a single SVG

Red tick marks on the column header at divergence point positions connect the dendrogram view back to the fork cards above.

### Edge cases

- 1 run: skip dendrogram, show flat heatmap row
- 2 runs: trivial dendrogram (single merge), still render it

### Code changes

- `viz/html.py`:
  - New `_build_dendrogram_heatmap(alignment, runs, points)` replaces `_build_trajectory_matrix()`
  - `_scipy_coords_to_svg()` helper maps dendrogram coordinates to SVG positions
  - Computes distance matrix and linkage internally for rendering

## Build sequence

1. **Converter** — enrich eval-harness converter, re-convert existing data
2. **Fork cards** — update narrate.py and html.py, replace current narrative display
3. **Dendrogram** — replace trajectory matrix with dendrogram+heatmap

Each step is independently useful. Tests at each step.
