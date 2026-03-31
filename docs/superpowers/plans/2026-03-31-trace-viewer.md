# Trace viewer implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an interactive stream-based trajectory viewer to moirai's HTML output, where runs flow through a branching stream that splits at divergence points, with three color modes and drill-down navigation.

**Architecture:** Python pre-computes stream geometry from the existing dendrogram/alignment pipeline and serializes it as JSON. A React app (compiled to a single HTML file via Vite) renders the stream SVG using visx, handles color mode switching, bifurcation drill-down, and step-level detail expansion. The built React template ships with the Python package as a static asset.

**Tech Stack:** Python (existing moirai pipeline), React 18, TypeScript, Vite + vite-plugin-singlefile, visx (SVG primitives), zustand (state), @tanstack/react-virtual (list virtualization)

**Spec:** `docs/superpowers/specs/2026-03-31-trace-viewer-design.md`

---

## File structure

### New files

```
viewer/                              # React app (compiled to single HTML)
  package.json
  tsconfig.json
  vite.config.ts                     # vite-plugin-singlefile config
  index.html                         # Vite entry HTML (has __DATA__ script tag)
  src/
    main.tsx                         # React entry, reads window.__MOIRAI_DATA__
    store.ts                         # zustand store: focus node, color mode, expanded runs
    types.ts                         # TypeScript types matching Python JSON payload
    App.tsx                          # Top-level layout
    components/
      StreamPlot.tsx                 # SVG stream rendering with visx
      BifurcationNode.tsx            # Clickable split nodes
      ColorModeToggle.tsx            # Cluster / composition / outcome toggle
      Breadcrumbs.tsx                # Drill-down navigation trail
      BranchCards.tsx                # Summary cards for child branches
      RunList.tsx                    # Virtualized run list
      StepTimeline.tsx               # Vertical step cards for expanded run
    lib/
      layout.ts                      # Band geometry: y-positions, widths, Bezier paths
      colors.ts                      # Color palettes for all three modes

moirai/viz/stream.py                 # Stream geometry computation from dendrogram
tests/test_stream.py                 # Tests for stream geometry
```

### Modified files

```
moirai/viz/html.py                   # Add write_stream_html() function
moirai/cli.py                        # Add --viewer flag to branch command
pyproject.toml                       # Add viewer template to package data
```

---

### Task 1: Scaffold the viewer project

**Files:**
- Create: `viewer/package.json`
- Create: `viewer/tsconfig.json`
- Create: `viewer/vite.config.ts`
- Create: `viewer/index.html`
- Create: `viewer/src/main.tsx`
- Create: `viewer/src/App.tsx`
- Create: `viewer/src/types.ts`

- [ ] **Step 1: Create `viewer/package.json`**

```json
{
  "name": "moirai-viewer",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build"
  },
  "dependencies": {
    "@tanstack/react-virtual": "^3.13.0",
    "@visx/curve": "^3.12.0",
    "@visx/group": "^3.12.0",
    "@visx/scale": "^3.12.0",
    "@visx/shape": "^3.12.0",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "zustand": "^5.0.0"
  },
  "devDependencies": {
    "@types/react": "^18.3.0",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.0",
    "typescript": "^5.6.0",
    "vite": "^6.0.0",
    "vite-plugin-singlefile": "^2.0.0"
  }
}
```

- [ ] **Step 2: Create `viewer/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "isolatedModules": true,
    "moduleDetection": "force",
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "noUncheckedIndexedAccess": true
  },
  "include": ["src"]
}
```

- [ ] **Step 3: Create `viewer/vite.config.ts`**

```typescript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteSingleFile } from "vite-plugin-singlefile";

export default defineConfig({
  plugins: [react(), viteSingleFile()],
  build: {
    target: "es2020",
    outDir: "dist",
  },
});
```

- [ ] **Step 4: Create `viewer/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>moirai</title>
  <script>window.__MOIRAI_DATA__ = "__DATA_PLACEHOLDER__";</script>
</head>
<body>
  <div id="root"></div>
  <script type="module" src="/src/main.tsx"></script>
</body>
</html>
```

- [ ] **Step 5: Create `viewer/src/types.ts`**

This defines the TypeScript types matching the JSON payload Python will inject. This is the contract between Python and the viewer.

```typescript
/** A single step in a run's trajectory */
export interface StepDetail {
  idx: number;
  enriched: string;       // e.g. "read(source)"
  detail: string;         // e.g. "urls.py" or "python -m pytest ..."
  status: string;         // "ok" or "error"
  phase: string;          // "explore", "modify", "verify", etc.
  color: string;          // hex color from STEP_COLORS
  metrics: {
    tokens_in?: number;
    tokens_out?: number;
    latency_ms?: number;
  };
}

/** Metadata for a single run */
export interface RunMeta {
  run_id: string;
  success: boolean | null;
  step_count: number;
  trajectory: string;     // compressed: "read(source) → edit → test"
  phase_mix: string;      // "69% explore, 15% verify, ..."
  harness: string | null;
  model: string | null;
}

/** Per-run step detail (tier 2 data, loaded on expand) */
export interface RunDetail {
  run_id: string;
  steps: StepDetail[];
}

/** One branch in the stream tree */
export interface StreamBranch {
  id: string;                          // unique branch identifier
  run_ids: string[];                   // runs in this branch
  success_rate: number;                // 0.0 - 1.0
  /** Per-column step type proportions (for step composition mode).
   *  Keyed by column index, values are {step_name: proportion} */
  step_proportions: Record<number, Record<string, number>>;
  /** Representative compressed trajectory */
  trajectory: string;
  /** Phase mix percentages */
  phase_mix: Record<string, number>;
  /** Start and end columns in the alignment */
  col_start: number;
  col_end: number;
}

/** A bifurcation point in the stream tree */
export interface Bifurcation {
  id: string;                          // matches internal dendrogram node id
  column: number;                      // alignment column where split occurs
  x_position: number;                  // normalized 0-1 position on x-axis
  separation: number;                  // 0-1 separation score
  p_value: number | null;              // Fisher's exact test
  significant: boolean;                // p < 0.2
  left_branch_id: string;
  right_branch_id: string;
  left_success_rate: number;
  right_success_rate: number;
}

/** The full stream tree for a task */
export interface StreamTree {
  task_id: string;
  n_runs: number;
  n_pass: number;
  n_fail: number;
  n_cols: number;                      // alignment width
  root_branch_id: string;
  branches: Record<string, StreamBranch>;
  bifurcations: Record<string, Bifurcation>;
  /** Tree structure: bifurcation_id -> [left_child, right_child]
   *  where children are either bifurcation IDs or branch IDs (leaves) */
  tree: Record<string, [string, string]>;
}

/** Top-level data payload injected by Python */
export interface MoiraiData {
  stats: {
    total_runs: number;
    pass_rate: string;
    n_tasks: number;
    n_div: number;
  };
  legend: Array<[string, string]>;     // [name, color_hex]
  tasks: StreamTree[];
  runs: Record<string, RunMeta>;       // keyed by run_id
  run_details: Record<string, RunDetail>;  // tier 2, keyed by run_id
  step_colors: Record<string, string>; // enriched_name -> hex color
}
```

- [ ] **Step 6: Create `viewer/src/main.tsx`**

```tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import type { MoiraiData } from "./types";

declare global {
  interface Window {
    __MOIRAI_DATA__: MoiraiData | string;
  }
}

function loadData(): MoiraiData {
  const raw = window.__MOIRAI_DATA__;
  if (typeof raw === "string") {
    return JSON.parse(raw) as MoiraiData;
  }
  return raw;
}

const data = loadData();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App data={data} />
  </StrictMode>,
);
```

- [ ] **Step 7: Create `viewer/src/App.tsx`**

Minimal shell that confirms data is loaded:

```tsx
import type { MoiraiData } from "./types";

export function App({ data }: { data: MoiraiData }) {
  return (
    <div style={{
      fontFamily: "'IBM Plex Mono', monospace",
      background: "#0d1117",
      color: "#c9d1d9",
      minHeight: "100vh",
      padding: "32px 40px",
    }}>
      <h1 style={{ fontSize: 20, color: "#e6edf3", margin: "0 0 16px" }}>
        moirai
      </h1>
      <div style={{ fontSize: 12, color: "#8b949e" }}>
        {data.stats.total_runs} runs &middot; {data.stats.pass_rate} pass rate &middot; {data.stats.n_tasks} tasks
      </div>
      <pre style={{ fontSize: 10, color: "#484f58", marginTop: 24 }}>
        {JSON.stringify(data.tasks[0]?.task_id ?? "no tasks", null, 2)}
      </pre>
    </div>
  );
}
```

- [ ] **Step 8: Install dependencies and verify build**

```bash
cd viewer && npm install
npm run build
```

Expected: `dist/index.html` exists, is a single self-contained file.

- [ ] **Step 9: Verify dev server works with mock data**

Create `viewer/public/mock.html` that sets `window.__MOIRAI_DATA__` to a small test fixture, then run `npm run dev` and verify the page renders.

- [ ] **Step 10: Commit**

```bash
git add viewer/
git commit -m "scaffold viewer: react + vite + visx single-file app"
```

---

### Task 2: Python stream geometry computation

**Files:**
- Create: `moirai/viz/stream.py`
- Create: `tests/test_stream.py`

This is the core new Python code. It walks the dendrogram linkage matrix and builds the stream tree data structure that the viewer consumes.

- [ ] **Step 1: Write test for basic stream tree construction**

```python
# tests/test_stream.py
import numpy as np
from moirai.schema import Run, Step, Result, Alignment
from moirai.viz.stream import build_stream_tree


def _make_run(run_id: str, task_id: str, success: bool, steps: list[str]) -> Run:
    return Run(
        run_id=run_id,
        task_id=task_id,
        task_family=None,
        agent=None,
        model=None,
        harness=None,
        timestamp=None,
        tags={},
        steps=[Step(idx=i, type="tool", name=s, status="ok") for i, s in enumerate(steps)],
        result=Result(success=success, score=None, label=None, error_type=None, summary=None),
    )


def test_build_stream_tree_basic():
    """4 runs, 2 pass (read→edit→test), 2 fail (read→bash→bash). Should produce
    a tree with one root bifurcation splitting into two branches."""
    runs = [
        _make_run("r1", "t1", True,  ["read", "edit", "test"]),
        _make_run("r2", "t1", True,  ["read", "edit", "test"]),
        _make_run("r3", "t1", False, ["read", "bash", "bash"]),
        _make_run("r4", "t1", False, ["read", "bash", "bash"]),
    ]
    alignment = Alignment(
        run_ids=["r1", "r2", "r3", "r4"],
        matrix=[
            ["read", "edit", "test"],
            ["read", "edit", "test"],
            ["read", "bash", "bash"],
            ["read", "bash", "bash"],
        ],
        level="name",
    )
    # Linkage: r1+r2 merge first (distance 0), r3+r4 merge (distance 0),
    # then the two pairs merge (distance > 0)
    Z = np.array([
        [0, 1, 0.0, 2],  # node 4: r1+r2
        [2, 3, 0.0, 2],  # node 5: r3+r4
        [4, 5, 1.0, 4],  # node 6: all
    ])

    tree = build_stream_tree(alignment, runs, Z)

    assert tree["task_id"] == "t1"
    assert tree["n_runs"] == 4
    assert tree["n_pass"] == 2
    assert tree["n_fail"] == 2
    assert tree["n_cols"] == 3
    # Should have 1 bifurcation and 2 leaf branches
    assert len(tree["bifurcations"]) == 1
    assert len(tree["branches"]) >= 2
    # Root branch should reference all 4 runs
    root = tree["branches"][tree["root_branch_id"]]
    assert len(root["run_ids"]) == 4
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_stream.py::test_build_stream_tree_basic -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'moirai.viz.stream'`

- [ ] **Step 3: Implement `build_stream_tree`**

```python
# moirai/viz/stream.py
"""Compute stream geometry from dendrogram for the interactive viewer."""

from __future__ import annotations

import numpy as np

from moirai.compress import step_enriched_name, step_phase
from moirai.schema import Alignment, Run
from moirai.viz.html import STEP_COLORS, _color_for


def build_stream_tree(
    alignment: Alignment,
    runs: list[Run],
    Z: np.ndarray,
) -> dict:
    """Build a stream tree from an alignment and its linkage matrix.

    The stream tree describes how runs split into branches at each
    internal node of the dendrogram. Each branch tracks its runs,
    success rate, and per-column step type proportions.

    Args:
        alignment: Aligned step matrix (runs x columns).
        runs: The runs for this task (same order as alignment.run_ids).
        Z: scipy linkage matrix (n_runs-1 x 4).

    Returns:
        Dict matching the StreamTree TypeScript type.
    """
    n_runs = len(alignment.run_ids)
    n_cols = len(alignment.matrix[0]) if alignment.matrix else 0
    run_map = {r.run_id: r for r in runs}

    # Map run_id -> index in alignment
    rid_to_idx = {rid: i for i, rid in enumerate(alignment.run_ids)}

    # Build index sets for each internal node by walking Z bottom-up
    # Leaf nodes are 0..n_runs-1, internal nodes are n_runs..2*n_runs-2
    leaf_sets: dict[int, list[int]] = {}
    for i in range(n_runs):
        leaf_sets[i] = [i]

    for i, row in enumerate(Z):
        left_id, right_id = int(row[0]), int(row[1])
        node_id = n_runs + i
        leaf_sets[node_id] = leaf_sets[left_id] + leaf_sets[right_id]

    # Build branches and bifurcations
    branches: dict[str, dict] = {}
    bifurcations: dict[str, dict] = {}
    tree_children: dict[str, tuple[str, str]] = {}

    def _success_rate(indices: list[int]) -> float:
        successes = sum(
            1 for idx in indices
            if run_map[alignment.run_ids[idx]].result.success
        )
        return successes / len(indices) if indices else 0.0

    def _step_proportions(indices: list[int]) -> dict[int, dict[str, float]]:
        """Per-column step type proportions for a set of runs."""
        props: dict[int, dict[str, float]] = {}
        for col in range(n_cols):
            counts: dict[str, int] = {}
            total = 0
            for idx in indices:
                val = alignment.matrix[idx][col]
                if val != "-":
                    counts[val] = counts.get(val, 0) + 1
                    total += 1
            if total > 0:
                props[col] = {k: v / total for k, v in counts.items()}
        return props

    def _representative_trajectory(indices: list[int]) -> str:
        """Pick the run closest to the median length as representative."""
        if not indices:
            return ""
        from moirai.compress import compress_steps
        runs_subset = [run_map[alignment.run_ids[i]] for i in indices]
        runs_subset.sort(key=lambda r: len(r.steps))
        median_run = runs_subset[len(runs_subset) // 2]
        return compress_steps(median_run.steps)

    def _phase_mix(indices: list[int]) -> dict[str, float]:
        """Phase distribution across runs."""
        counts: dict[str, int] = {}
        total = 0
        for idx in indices:
            run = run_map[alignment.run_ids[idx]]
            for s in run.steps:
                p = step_phase(s)
                counts[p] = counts.get(p, 0) + 1
                total += 1
        if total == 0:
            return {}
        return {k: round(v / total, 3) for k, v in sorted(counts.items(), key=lambda x: -x[1])}

    def _make_branch(branch_id: str, indices: list[int], col_start: int, col_end: int) -> dict:
        return {
            "id": branch_id,
            "run_ids": [alignment.run_ids[i] for i in indices],
            "success_rate": _success_rate(indices),
            "step_proportions": _step_proportions(indices),
            "trajectory": _representative_trajectory(indices),
            "phase_mix": _phase_mix(indices),
            "col_start": col_start,
            "col_end": col_end,
        }

    def _discriminating_column(left_indices: list[int], right_indices: list[int]) -> tuple[int, float]:
        """Find the alignment column that best separates two sets of runs.
        Returns (column, separation_score)."""
        best_col = 0
        best_sep = 0.0
        for col in range(n_cols):
            left_vals = set()
            right_vals = set()
            for idx in left_indices:
                v = alignment.matrix[idx][col]
                if v != "-":
                    left_vals.add(v)
            for idx in right_indices:
                v = alignment.matrix[idx][col]
                if v != "-":
                    right_vals.add(v)
            if not left_vals or not right_vals:
                continue
            # Separation: fraction of values that don't overlap
            overlap = left_vals & right_vals
            all_vals = left_vals | right_vals
            sep = 1.0 - len(overlap) / len(all_vals) if all_vals else 0.0
            if sep > best_sep:
                best_sep = sep
                best_col = col
        return best_col, best_sep

    def _compute_p_value(left_indices: list[int], right_indices: list[int]) -> float | None:
        """Fisher's exact test: does this split predict outcome?"""
        from scipy.stats import fisher_exact
        left_pass = sum(1 for i in left_indices if run_map[alignment.run_ids[i]].result.success)
        left_fail = len(left_indices) - left_pass
        right_pass = sum(1 for i in right_indices if run_map[alignment.run_ids[i]].result.success)
        right_fail = len(right_indices) - right_pass
        table = [[left_pass, left_fail], [right_pass, right_fail]]
        if min(left_pass + left_fail, right_pass + right_fail) == 0:
            return None
        try:
            _, p = fisher_exact(table)
            return p
        except ValueError:
            return None

    # Walk the linkage matrix to build the tree
    # Process internal nodes from top (root) down
    root_node = n_runs + len(Z) - 1

    def _walk(node_id: int, col_start: int, col_end: int) -> str:
        """Recursively build stream tree. Returns the branch/bifurcation id for this node."""
        if node_id < n_runs:
            # Leaf node — single run, make a branch
            bid = f"branch_{node_id}"
            branches[bid] = _make_branch(bid, [node_id], col_start, col_end)
            return bid

        row_idx = node_id - n_runs
        left_id = int(Z[row_idx, 0])
        right_id = int(Z[row_idx, 1])
        left_indices = leaf_sets[left_id]
        right_indices = leaf_sets[right_id]
        all_indices = left_indices + right_indices

        # Make the parent branch (covers all runs in this subtree)
        parent_bid = f"branch_{node_id}"
        branches[parent_bid] = _make_branch(parent_bid, all_indices, col_start, col_end)

        # Find where these two subtrees diverge
        div_col, separation = _discriminating_column(left_indices, right_indices)
        p_value = _compute_p_value(left_indices, right_indices)

        # Bifurcation x-position: normalized to 0-1
        x_pos = div_col / n_cols if n_cols > 0 else 0.5

        bif_id = f"bif_{node_id}"
        left_child = _walk(left_id, div_col, col_end)
        right_child = _walk(right_id, div_col, col_end)

        bifurcations[bif_id] = {
            "id": bif_id,
            "column": div_col,
            "x_position": x_pos,
            "separation": round(separation, 3),
            "p_value": round(p_value, 4) if p_value is not None else None,
            "significant": p_value is not None and p_value < 0.2,
            "left_branch_id": left_child,
            "right_branch_id": right_child,
            "left_success_rate": _success_rate(left_indices),
            "right_success_rate": _success_rate(right_indices),
        }
        tree_children[bif_id] = (left_child, right_child)

        return parent_bid

    root_bid = _walk(root_node, 0, n_cols)

    task_id = runs[0].task_id if runs else ""
    return {
        "task_id": task_id,
        "n_runs": n_runs,
        "n_pass": sum(1 for r in runs if r.result.success),
        "n_fail": sum(1 for r in runs if not r.result.success),
        "n_cols": n_cols,
        "root_branch_id": root_bid,
        "branches": branches,
        "bifurcations": bifurcations,
        "tree": tree_children,
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest tests/test_stream.py::test_build_stream_tree_basic -v
```

Expected: PASS

- [ ] **Step 5: Write test for step proportions**

```python
# tests/test_stream.py (append)

def test_step_proportions():
    """Step proportions at each column should sum to 1.0."""
    runs = [
        _make_run("r1", "t1", True,  ["read", "edit", "test"]),
        _make_run("r2", "t1", True,  ["read", "edit", "test"]),
        _make_run("r3", "t1", False, ["read", "bash", "bash"]),
    ]
    alignment = Alignment(
        run_ids=["r1", "r2", "r3"],
        matrix=[
            ["read", "edit", "test"],
            ["read", "edit", "test"],
            ["read", "bash", "bash"],
        ],
        level="name",
    )
    Z = np.array([
        [0, 1, 0.0, 2],
        [2, 3, 1.0, 3],
    ])
    tree = build_stream_tree(alignment, runs, Z)
    root = tree["branches"][tree["root_branch_id"]]

    # Column 0: all "read" → {"read": 1.0}
    assert root["step_proportions"][0] == {"read": 1.0}
    # Column 1: 2/3 "edit", 1/3 "bash"
    col1 = root["step_proportions"][1]
    assert abs(col1["edit"] - 2 / 3) < 0.01
    assert abs(col1["bash"] - 1 / 3) < 0.01
```

- [ ] **Step 6: Run test**

```bash
.venv/bin/pytest tests/test_stream.py::test_step_proportions -v
```

Expected: PASS

- [ ] **Step 7: Write test for p-value and significance**

```python
# tests/test_stream.py (append)

def test_bifurcation_significance():
    """A split with all-pass left and all-fail right should have a low p-value."""
    runs = [
        _make_run("r1", "t1", True,  ["read", "edit"]),
        _make_run("r2", "t1", True,  ["read", "edit"]),
        _make_run("r3", "t1", True,  ["read", "edit"]),
        _make_run("r4", "t1", False, ["read", "bash"]),
        _make_run("r5", "t1", False, ["read", "bash"]),
        _make_run("r6", "t1", False, ["read", "bash"]),
    ]
    alignment = Alignment(
        run_ids=[f"r{i}" for i in range(1, 7)],
        matrix=[
            ["read", "edit"],
            ["read", "edit"],
            ["read", "edit"],
            ["read", "bash"],
            ["read", "bash"],
            ["read", "bash"],
        ],
        level="name",
    )
    Z = np.array([
        [0, 1, 0.0, 2],
        [3, 4, 0.0, 2],
        [2, 6, 0.0, 3],
        [5, 7, 0.0, 3],
        [8, 9, 1.0, 6],
    ])
    tree = build_stream_tree(alignment, runs, Z)

    # Find the root bifurcation
    bifs = list(tree["bifurcations"].values())
    root_bif = [b for b in bifs if b["left_success_rate"] != b["right_success_rate"]]
    assert len(root_bif) >= 1
    bif = root_bif[0]
    assert bif["p_value"] is not None
    assert bif["p_value"] < 0.1  # 3 pass vs 3 fail should be significant
    assert bif["significant"]
```

- [ ] **Step 8: Run all stream tests**

```bash
.venv/bin/pytest tests/test_stream.py -v
```

Expected: all PASS

- [ ] **Step 9: Commit**

```bash
git add moirai/viz/stream.py tests/test_stream.py
git commit -m "feat: stream geometry computation from dendrogram"
```

---

### Task 3: Build run metadata and detail serialization

**Files:**
- Modify: `moirai/viz/stream.py`
- Modify: `tests/test_stream.py`

Add functions to serialize run metadata (tier 1) and step detail (tier 2) for the viewer.

- [ ] **Step 1: Write test for run metadata serialization**

```python
# tests/test_stream.py (append)

from moirai.viz.stream import build_run_meta, build_run_detail


def test_build_run_meta():
    run = _make_run("r1", "t1", True, ["read", "edit", "test"])
    meta = build_run_meta(run)
    assert meta["run_id"] == "r1"
    assert meta["success"] is True
    assert meta["step_count"] == 3
    assert "trajectory" in meta
    assert "phase_mix" in meta


def test_build_run_detail():
    run = _make_run("r1", "t1", True, ["read", "edit", "test"])
    detail = build_run_detail(run)
    assert detail["run_id"] == "r1"
    assert len(detail["steps"]) == 3
    assert detail["steps"][0]["enriched"] == "read"
    assert "phase" in detail["steps"][0]
    assert "color" in detail["steps"][0]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_stream.py::test_build_run_meta -v
```

Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `build_run_meta` and `build_run_detail`**

```python
# moirai/viz/stream.py (append)

def build_run_meta(run: Run) -> dict:
    """Build tier-1 run metadata (small, always loaded)."""
    from moirai.compress import compress_steps, compress_phases
    return {
        "run_id": run.run_id,
        "success": run.result.success,
        "step_count": len(run.steps),
        "trajectory": compress_steps(run.steps),
        "phase_mix": compress_phases(run),
        "harness": run.harness,
        "model": run.model,
    }


def build_run_detail(run: Run) -> dict:
    """Build tier-2 run detail (loaded on expand)."""
    steps = []
    for s in run.steps:
        enriched = step_enriched_name(s)
        if enriched is None:
            continue  # skip noise steps
        steps.append({
            "idx": s.idx,
            "enriched": enriched,
            "detail": _step_detail_str(s),
            "status": s.status,
            "phase": step_phase(s),
            "color": STEP_COLORS.get(enriched, "#6e7681"),
            "metrics": {
                k: v for k, v in s.metrics.items()
                if k in ("tokens_in", "tokens_out", "latency_ms")
            },
        })
    return {"run_id": run.run_id, "steps": steps}


def _step_detail_str(step) -> str:
    """Extract a human-readable detail string from step attrs."""
    if "file_path" in step.attrs:
        path = step.attrs["file_path"]
        # Show just the filename, not the full path
        return path.rsplit("/", 1)[-1] if "/" in path else path
    if "command" in step.attrs:
        cmd = step.attrs["command"]
        return cmd[:80] + "..." if len(cmd) > 80 else cmd
    if "pattern" in step.attrs:
        return step.attrs["pattern"]
    return ""
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_stream.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add moirai/viz/stream.py tests/test_stream.py
git commit -m "feat: run metadata and detail serialization for viewer"
```

---

### Task 4: Wire up Python HTML generation for viewer

**Files:**
- Modify: `moirai/viz/html.py`
- Modify: `moirai/cli.py`
- Create: `moirai/viz/templates/viewer.html` (placeholder until React build exists)

- [ ] **Step 1: Add `write_stream_html` function**

```python
# moirai/viz/html.py (append, after write_branch_html)

def write_stream_html(
    runs: list[Run],
    task_results: list,
    path: Path,
) -> Path:
    """Write the interactive stream viewer HTML.

    Computes stream geometry for each task and injects the data
    into the pre-built React viewer template.
    """
    import json
    from moirai.viz.stream import build_stream_tree, build_run_meta, build_run_detail
    from moirai.analyze.splits import find_split_divergences

    tasks_data = []
    all_runs_meta: dict[str, dict] = {}
    all_run_details: dict[str, dict] = {}

    for tid, task_runs, alignment, _points in task_results:
        # Compute dendrogram
        _splits, Z, _dendro = find_split_divergences(alignment, task_runs)

        # Build stream tree
        tree = build_stream_tree(alignment, task_runs, Z)
        tasks_data.append(tree)

        # Build per-run data
        for r in task_runs:
            if r.run_id not in all_runs_meta:
                all_runs_meta[r.run_id] = build_run_meta(r)
                all_run_details[r.run_id] = build_run_detail(r)

    # Build legend
    legend = [(name, color) for name, color in STEP_COLORS.items()]

    payload = {
        "stats": {
            "total_runs": len(runs),
            "pass_rate": f"{sum(1 for r in runs if r.result.success) / len(runs) * 100:.0f}%",
            "n_tasks": len(tasks_data),
            "n_div": sum(len(t["bifurcations"]) for t in tasks_data),
        },
        "legend": legend,
        "tasks": tasks_data,
        "runs": all_runs_meta,
        "run_details": all_run_details,
        "step_colors": dict(STEP_COLORS),
    }

    # Load the viewer template
    template_path = Path(__file__).parent / "templates" / "viewer.html"
    if not template_path.exists():
        # Fallback: check for built viewer
        template_path = Path(__file__).parent / "templates" / "viewer-built.html"

    template = template_path.read_text()
    data_json = json.dumps(payload, default=str)
    html = template.replace('"__DATA_PLACEHOLDER__"', data_json)

    path.write_text(html)
    return path
```

- [ ] **Step 2: Add `--viewer` flag to branch command**

In `moirai/cli.py`, add a `viewer` option to the `branch` command:

```python
# In the branch() function signature, add:
    viewer: Path | None = typer.Option(None, help="Write interactive stream viewer HTML"),
```

And at the end of the branch function (after the existing `if html:` block), add:

```python
    if viewer:
        from moirai.viz.html import write_stream_html
        out = write_stream_html(runs, task_results, viewer)
        console.print(f"\nViewer written to {out}")
```

- [ ] **Step 3: Create placeholder viewer template**

```html
<!-- moirai/viz/templates/viewer.html -->
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>moirai viewer</title>
  <script>window.__MOIRAI_DATA__ = "__DATA_PLACEHOLDER__";</script>
  <style>
    body { font-family: monospace; background: #0d1117; color: #c9d1d9; padding: 40px; }
    pre { font-size: 12px; white-space: pre-wrap; }
  </style>
</head>
<body>
  <h1>moirai viewer (placeholder)</h1>
  <p>Replace this template with the built React viewer.</p>
  <pre id="out"></pre>
  <script>
    const data = typeof window.__MOIRAI_DATA__ === 'string'
      ? JSON.parse(window.__MOIRAI_DATA__)
      : window.__MOIRAI_DATA__;
    document.getElementById('out').textContent =
      JSON.stringify(data.stats, null, 2) + '\n\n' +
      data.tasks.map(t => t.task_id + ': ' + Object.keys(t.bifurcations).length + ' bifurcations').join('\n');
  </script>
</body>
</html>
```

- [ ] **Step 4: Test end-to-end**

```bash
.venv/bin/moirai branch examples/eval_harness/ --viewer /tmp/moirai_viewer.html
```

Expected: writes HTML file, prints path. Open the file in a browser — should show stats and task IDs.

- [ ] **Step 5: Commit**

```bash
git add moirai/viz/html.py moirai/cli.py moirai/viz/templates/viewer.html
git commit -m "feat: wire up --viewer flag with stream data pipeline"
```

---

### Task 5: Stream band layout (TypeScript)

**Files:**
- Create: `viewer/src/lib/layout.ts`
- Create: `viewer/src/lib/colors.ts`

The layout module computes y-positions, widths, and SVG path data for stream bands from the stream tree data. This is pure math — no React, no DOM.

- [ ] **Step 1: Create `viewer/src/lib/colors.ts`**

```typescript
// viewer/src/lib/colors.ts

/** Cluster identity: distinct hues for branches */
const CLUSTER_PALETTE = [
  "#5b8fbe", "#e8943a", "#6dc4be", "#b893c4", "#e8585a",
  "#3fb950", "#e8d44d", "#a87bb0", "#b89a7a", "#7eb0d5",
  "#c4a265", "#6e9e6e", "#d47f8c", "#8ca8c4", "#c9a052",
];

export function clusterColor(index: number): string {
  return CLUSTER_PALETTE[index % CLUSTER_PALETTE.length];
}

/** Outcome rate: blue (pass) to red (fail) */
export function outcomeColor(rate: number): string {
  // Interpolate hue from 210 (blue) to 0 (red)
  const hue = rate * 210;
  return `hsl(${hue}, 65%, 55%)`;
}

/** Low confidence overlay pattern ID */
export const LOW_CONFIDENCE_PATTERN_ID = "low-confidence-hatch";

/** Step category colors (matches Python STEP_COLORS) */
export type StepColors = Record<string, string>;
```

- [ ] **Step 2: Create `viewer/src/lib/layout.ts`**

```typescript
// viewer/src/lib/layout.ts
import type { StreamTree, StreamBranch, Bifurcation } from "../types";

export interface BandSegment {
  branchId: string;
  x0: number;  // start x (pixels)
  x1: number;  // end x (pixels)
  y0: number;  // top edge y (pixels)
  y1: number;  // bottom edge y (pixels)
}

export interface ForkPath {
  bifurcationId: string;
  /** SVG path "d" attribute for the fork transition */
  parentBand: BandSegment;
  leftBand: BandSegment;
  rightBand: BandSegment;
  /** Bezier control point x offset for fork curve */
  controlOffset: number;
}

export interface StreamLayout {
  bands: BandSegment[];
  forks: ForkPath[];
  width: number;
  height: number;
}

const MIN_BAND_HEIGHT = 4;

/**
 * Compute the stream layout for a given focus node.
 *
 * @param tree - The full stream tree data
 * @param focusBranchId - Which branch to render as root (for drill-down)
 * @param width - Available SVG width in pixels
 * @param height - Available SVG height in pixels
 */
export function computeStreamLayout(
  tree: StreamTree,
  focusBranchId: string,
  width: number,
  height: number,
): StreamLayout {
  const bands: BandSegment[] = [];
  const forks: ForkPath[] = [];

  const focusBranch = tree.branches[focusBranchId];
  if (!focusBranch) return { bands: [], forks: [], width, height };

  const totalRuns = focusBranch.run_ids.length;
  const padding = 20;
  const usableHeight = height - padding * 2;
  const usableWidth = width - padding * 2;

  // Find bifurcations that are children of the focus branch
  // Walk the tree structure to find relevant bifurcations
  const relevantBifs = findDescendantBifurcations(tree, focusBranchId);

  // Recursive layout: each branch gets vertical space proportional to its run count
  function layoutSubtree(
    branchId: string,
    x0: number,
    x1: number,
    yCenter: number,
    availableHeight: number,
  ): void {
    const branch = tree.branches[branchId];
    if (!branch) return;

    const bandHeight = Math.max(
      MIN_BAND_HEIGHT,
      (branch.run_ids.length / totalRuns) * usableHeight,
    );
    const y0 = yCenter - bandHeight / 2;
    const y1 = yCenter + bandHeight / 2;

    // Find if this branch has a bifurcation
    const bif = Object.values(tree.bifurcations).find(
      (b) => tree.tree[b.id] &&
        (findParentBranch(tree, b.id) === branchId),
    );

    if (!bif || !relevantBifs.has(bif.id)) {
      // Leaf segment — just a straight band
      bands.push({ branchId, x0, x1, y0, y1 });
      return;
    }

    // Has a bifurcation — band runs from x0 to fork point, then splits
    const forkX = padding + bif.x_position * usableWidth;
    const clampedForkX = Math.max(x0 + 20, Math.min(forkX, x1 - 20));

    // Parent band before fork
    bands.push({ branchId, x0, x1: clampedForkX, y0, y1 });

    // Child bands after fork
    const leftId = bif.left_branch_id;
    const rightId = bif.right_branch_id;
    const leftBranch = tree.branches[leftId];
    const rightBranch = tree.branches[rightId];

    if (!leftBranch || !rightBranch) return;

    const leftRatio = leftBranch.run_ids.length / branch.run_ids.length;
    const rightRatio = rightBranch.run_ids.length / branch.run_ids.length;

    const gap = Math.min(4, availableHeight * 0.05);
    const leftHeight = Math.max(MIN_BAND_HEIGHT, (availableHeight - gap) * leftRatio);
    const rightHeight = Math.max(MIN_BAND_HEIGHT, (availableHeight - gap) * rightRatio);

    const leftCenter = yCenter - (rightHeight + gap) / 2;
    const rightCenter = yCenter + (leftHeight + gap) / 2;

    const controlOffset = Math.min(40, (x1 - clampedForkX) * 0.3);

    const leftBand: BandSegment = {
      branchId: leftId,
      x0: clampedForkX,
      x1: clampedForkX + controlOffset,
      y0: leftCenter - leftHeight / 2,
      y1: leftCenter + leftHeight / 2,
    };
    const rightBand: BandSegment = {
      branchId: rightId,
      x0: clampedForkX,
      x1: clampedForkX + controlOffset,
      y0: rightCenter - rightHeight / 2,
      y1: rightCenter + rightHeight / 2,
    };

    forks.push({
      bifurcationId: bif.id,
      parentBand: { branchId, x0: clampedForkX - controlOffset, x1: clampedForkX, y0, y1 },
      leftBand,
      rightBand,
      controlOffset,
    });

    // Recurse into children
    layoutSubtree(leftId, clampedForkX + controlOffset, x1, leftCenter, leftHeight);
    layoutSubtree(rightId, clampedForkX + controlOffset, x1, rightCenter, rightHeight);
  }

  layoutSubtree(focusBranchId, padding, width - padding, height / 2, usableHeight);

  return { bands, forks, width, height };
}

/** Find all bifurcation IDs that descend from a given branch */
function findDescendantBifurcations(tree: StreamTree, branchId: string): Set<string> {
  const result = new Set<string>();
  for (const [bifId, children] of Object.entries(tree.tree)) {
    // A bifurcation descends from branchId if its parent branch is branchId
    // or any descendant of branchId
    if (findParentBranch(tree, bifId) === branchId) {
      result.add(bifId);
      // Also include bifurcations in child branches
      const [left, right] = children;
      for (const childBif of findDescendantBifurcations(tree, left)) {
        result.add(childBif);
      }
      for (const childBif of findDescendantBifurcations(tree, right)) {
        result.add(childBif);
      }
    }
  }
  return result;
}

/** Find which branch a bifurcation belongs to (its parent) */
function findParentBranch(tree: StreamTree, bifId: string): string | null {
  const bif = tree.bifurcations[bifId];
  if (!bif) return null;
  // The parent branch contains all runs from both children
  for (const [bid, branch] of Object.entries(tree.branches)) {
    const children = tree.tree[bifId];
    if (!children) continue;
    const leftBranch = tree.branches[children[0]];
    const rightBranch = tree.branches[children[1]];
    if (!leftBranch || !rightBranch) continue;
    const childRunCount = leftBranch.run_ids.length + rightBranch.run_ids.length;
    if (branch.run_ids.length === childRunCount &&
        bid.startsWith("branch_") && bid !== children[0] && bid !== children[1]) {
      return bid;
    }
  }
  return null;
}

/**
 * Generate SVG path "d" for a fork transition.
 * Uses cubic Bezier curves for smooth splitting.
 */
export function forkPath(
  parent: BandSegment,
  child: BandSegment,
  controlOffset: number,
): string {
  const px1 = parent.x1;
  const cx0 = child.x0 + controlOffset;

  // Top edge: parent top → child top (cubic bezier)
  // Bottom edge: child bottom → parent bottom (cubic bezier, reversed)
  return [
    `M ${px1} ${parent.y0}`,
    `C ${px1 + controlOffset} ${parent.y0}, ${cx0 - controlOffset} ${child.y0}, ${cx0} ${child.y0}`,
    `L ${cx0} ${child.y1}`,
    `C ${cx0 - controlOffset} ${child.y1}, ${px1 + controlOffset} ${parent.y1}, ${px1} ${parent.y1}`,
    `Z`,
  ].join(" ");
}
```

- [ ] **Step 3: Verify TypeScript compiles**

```bash
cd viewer && npx tsc --noEmit
```

Expected: no errors

- [ ] **Step 4: Commit**

```bash
git add viewer/src/lib/
git commit -m "feat: stream band layout and color utilities"
```

---

### Task 6: StreamPlot component (cluster identity mode)

**Files:**
- Create: `viewer/src/store.ts`
- Create: `viewer/src/components/StreamPlot.tsx`
- Create: `viewer/src/components/BifurcationNode.tsx`
- Create: `viewer/src/components/ColorModeToggle.tsx`
- Modify: `viewer/src/App.tsx`

- [ ] **Step 1: Create zustand store**

```typescript
// viewer/src/store.ts
import { create } from "zustand";

export type ColorMode = "cluster" | "composition" | "outcome";

interface ViewerState {
  /** Which task is currently selected */
  activeTaskIndex: number;
  setActiveTask: (index: number) => void;

  /** Current color mode */
  colorMode: ColorMode;
  setColorMode: (mode: ColorMode) => void;

  /** Drill-down path: stack of branch IDs from root to current focus */
  focusPath: string[];
  drillDown: (branchId: string) => void;
  drillUp: (toIndex: number) => void;
  resetFocus: () => void;

  /** Set of expanded run IDs (for step detail) */
  expandedRuns: Set<string>;
  toggleRun: (runId: string) => void;
}

export const useViewerStore = create<ViewerState>((set) => ({
  activeTaskIndex: 0,
  setActiveTask: (index) => set({ activeTaskIndex: index, focusPath: [], expandedRuns: new Set() }),

  colorMode: "outcome",
  setColorMode: (mode) => set({ colorMode: mode }),

  focusPath: [],
  drillDown: (branchId) => set((s) => ({ focusPath: [...s.focusPath, branchId] })),
  drillUp: (toIndex) => set((s) => ({ focusPath: s.focusPath.slice(0, toIndex + 1) })),
  resetFocus: () => set({ focusPath: [] }),

  expandedRuns: new Set(),
  toggleRun: (runId) =>
    set((s) => {
      const next = new Set(s.expandedRuns);
      if (next.has(runId)) next.delete(runId);
      else next.add(runId);
      return { expandedRuns: next };
    }),
}));
```

- [ ] **Step 2: Create `StreamPlot.tsx`**

```tsx
// viewer/src/components/StreamPlot.tsx
import { useMemo } from "react";
import { Group } from "@visx/group";
import type { StreamTree } from "../types";
import type { ColorMode } from "../store";
import { computeStreamLayout, forkPath } from "../lib/layout";
import { clusterColor, outcomeColor } from "../lib/colors";
import { BifurcationNode } from "./BifurcationNode";

interface Props {
  tree: StreamTree;
  focusBranchId: string;
  colorMode: ColorMode;
  stepColors: Record<string, string>;
  width: number;
  height: number;
  onBifurcationClick: (branchId: string) => void;
}

export function StreamPlot({
  tree,
  focusBranchId,
  colorMode,
  stepColors,
  width,
  height,
  onBifurcationClick,
}: Props) {
  const layout = useMemo(
    () => computeStreamLayout(tree, focusBranchId, width, height),
    [tree, focusBranchId, width, height],
  );

  // Assign cluster index to each branch for cluster color mode
  const branchColorIndex = useMemo(() => {
    const map = new Map<string, number>();
    let idx = 0;
    // Leaf branches (no bifurcation that uses them as parent) get unique colors
    for (const band of layout.bands) {
      if (!map.has(band.branchId)) {
        map.set(band.branchId, idx++);
      }
    }
    return map;
  }, [layout]);

  function bandFill(branchId: string): string {
    const branch = tree.branches[branchId];
    if (!branch) return "#6e7681";

    switch (colorMode) {
      case "cluster":
        return clusterColor(branchColorIndex.get(branchId) ?? 0);
      case "outcome":
        return outcomeColor(branch.success_rate);
      case "composition":
        // For now, use cluster color; composition mode will be a stacked render
        return clusterColor(branchColorIndex.get(branchId) ?? 0);
    }
  }

  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      <Group>
        {/* Straight band segments */}
        {layout.bands.map((band, i) => (
          <rect
            key={`band-${i}`}
            x={band.x0}
            y={band.y0}
            width={band.x1 - band.x0}
            height={band.y1 - band.y0}
            fill={bandFill(band.branchId)}
            opacity={0.85}
            rx={2}
          />
        ))}

        {/* Fork transitions */}
        {layout.forks.map((fork) => {
          const leftPath = forkPath(fork.parentBand, fork.leftBand, fork.controlOffset);
          const rightPath = forkPath(fork.parentBand, fork.rightBand, fork.controlOffset);
          return (
            <g key={fork.bifurcationId}>
              <path d={leftPath} fill={bandFill(fork.leftBand.branchId)} opacity={0.85} />
              <path d={rightPath} fill={bandFill(fork.rightBand.branchId)} opacity={0.85} />
            </g>
          );
        })}

        {/* Bifurcation nodes */}
        {layout.forks.map((fork) => {
          const bif = tree.bifurcations[fork.bifurcationId];
          if (!bif) return null;
          const x = fork.parentBand.x1;
          const y = (fork.parentBand.y0 + fork.parentBand.y1) / 2;
          return (
            <BifurcationNode
              key={bif.id}
              bif={bif}
              x={x}
              y={y}
              onClick={() => onBifurcationClick(fork.parentBand.branchId)}
            />
          );
        })}
      </Group>
    </svg>
  );
}
```

- [ ] **Step 3: Create `BifurcationNode.tsx`**

```tsx
// viewer/src/components/BifurcationNode.tsx
import type { Bifurcation } from "../types";

interface Props {
  bif: Bifurcation;
  x: number;
  y: number;
  onClick: () => void;
}

export function BifurcationNode({ bif, x, y, onClick }: Props) {
  const r = 8;
  return (
    <g
      style={{ cursor: "pointer" }}
      onClick={(e) => { e.stopPropagation(); onClick(); }}
    >
      <circle
        cx={x}
        cy={y}
        r={r}
        fill={bif.significant ? "#d29922" : "transparent"}
        stroke={bif.significant ? "#d29922" : "#484f58"}
        strokeWidth={bif.significant ? 0 : 1.5}
        strokeDasharray={bif.significant ? undefined : "3,3"}
        opacity={0.9}
      />
      {bif.significant && (
        <text
          x={x}
          y={y + 1}
          textAnchor="middle"
          dominantBaseline="central"
          fontSize={9}
          fontWeight={700}
          fill="#0d1117"
          style={{ pointerEvents: "none" }}
        >
          {bif.p_value !== null ? (bif.p_value < 0.01 ? "**" : "*") : ""}
        </text>
      )}
    </g>
  );
}
```

- [ ] **Step 4: Create `ColorModeToggle.tsx`**

```tsx
// viewer/src/components/ColorModeToggle.tsx
import type { ColorMode } from "../store";

interface Props {
  mode: ColorMode;
  onChange: (mode: ColorMode) => void;
}

const MODES: Array<{ value: ColorMode; label: string }> = [
  { value: "cluster", label: "Clusters" },
  { value: "composition", label: "Step types" },
  { value: "outcome", label: "Outcomes" },
];

export function ColorModeToggle({ mode, onChange }: Props) {
  return (
    <div style={{ display: "flex", gap: 2, background: "#161b22", borderRadius: 6, padding: 2 }}>
      {MODES.map((m) => (
        <button
          key={m.value}
          onClick={() => onChange(m.value)}
          style={{
            padding: "6px 14px",
            fontSize: 11,
            fontFamily: "inherit",
            border: "none",
            borderRadius: 4,
            cursor: "pointer",
            background: mode === m.value ? "#30363d" : "transparent",
            color: mode === m.value ? "#e6edf3" : "#8b949e",
            fontWeight: mode === m.value ? 600 : 400,
            transition: "all 0.15s",
          }}
        >
          {m.label}
        </button>
      ))}
    </div>
  );
}
```

- [ ] **Step 5: Update `App.tsx` to render stream plot**

```tsx
// viewer/src/App.tsx
import { useViewerStore } from "./store";
import { StreamPlot } from "./components/StreamPlot";
import { ColorModeToggle } from "./components/ColorModeToggle";
import type { MoiraiData } from "./types";

export function App({ data }: { data: MoiraiData }) {
  const { activeTaskIndex, setActiveTask, colorMode, setColorMode, focusPath, drillDown, resetFocus, drillUp } =
    useViewerStore();

  const task = data.tasks[activeTaskIndex];
  if (!task) return <div style={{ color: "#c9d1d9", padding: 40 }}>No tasks with mixed outcomes.</div>;

  const currentFocusBranch = focusPath.length > 0
    ? focusPath[focusPath.length - 1]
    : task.root_branch_id;

  return (
    <div style={{
      fontFamily: "'IBM Plex Mono', monospace",
      background: "#0d1117",
      color: "#c9d1d9",
      minHeight: "100vh",
    }}>
      {/* Header */}
      <div style={{ padding: "24px 40px 16px", borderBottom: "1px solid #30363d" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div>
            <h1 style={{ fontSize: 20, color: "#e6edf3", margin: 0, fontFamily: "sans-serif" }}>
              moirai
            </h1>
            <div style={{ fontSize: 10, color: "#8b949e", textTransform: "uppercase", letterSpacing: 1.5, marginTop: 2 }}>
              trajectory viewer
            </div>
          </div>
          <ColorModeToggle mode={colorMode} onChange={setColorMode} />
        </div>
      </div>

      {/* Stats bar */}
      <div style={{
        display: "flex",
        gap: 40,
        padding: "12px 40px",
        borderBottom: "1px solid #30363d",
        background: "#161b22",
        fontSize: 11,
      }}>
        <span><strong style={{ fontSize: 20, color: "#e6edf3" }}>{data.stats.total_runs}</strong>{" "}runs</span>
        <span><strong style={{ fontSize: 20, color: "#e6edf3" }}>{data.stats.pass_rate}</strong>{" "}pass rate</span>
        <span><strong style={{ fontSize: 20, color: "#e6edf3" }}>{data.stats.n_tasks}</strong>{" "}tasks</span>
      </div>

      {/* Task tabs */}
      {data.tasks.length > 1 && (
        <div style={{
          display: "flex",
          gap: 4,
          padding: "8px 40px",
          borderBottom: "1px solid #30363d",
          background: "#161b22",
          overflowX: "auto",
        }}>
          {data.tasks.map((t, i) => (
            <button
              key={t.task_id}
              onClick={() => { setActiveTask(i); resetFocus(); }}
              style={{
                padding: "4px 12px",
                fontSize: 11,
                fontFamily: "inherit",
                border: "1px solid",
                borderColor: i === activeTaskIndex ? "#d29922" : "#30363d",
                borderRadius: 4,
                cursor: "pointer",
                background: i === activeTaskIndex ? "rgba(210,153,34,0.1)" : "transparent",
                color: i === activeTaskIndex ? "#d29922" : "#8b949e",
                whiteSpace: "nowrap",
              }}
            >
              {t.task_id} ({t.n_pass}P/{t.n_fail}F)
            </button>
          ))}
        </div>
      )}

      {/* Breadcrumbs */}
      {focusPath.length > 0 && (
        <div style={{ padding: "8px 40px", fontSize: 11, color: "#8b949e" }}>
          <span
            style={{ cursor: "pointer", color: "#d29922" }}
            onClick={resetFocus}
          >
            All ({task.n_runs})
          </span>
          {focusPath.map((bid, i) => {
            const branch = task.branches[bid];
            return (
              <span key={bid}>
                {" > "}
                <span
                  style={{ cursor: "pointer", color: i === focusPath.length - 1 ? "#e6edf3" : "#d29922" }}
                  onClick={() => drillUp(i)}
                >
                  {branch ? `${branch.run_ids.length} runs (${Math.round(branch.success_rate * 100)}%)` : bid}
                </span>
              </span>
            );
          })}
        </div>
      )}

      {/* Stream plot */}
      <div style={{ padding: "20px 40px" }}>
        <StreamPlot
          tree={task}
          focusBranchId={currentFocusBranch}
          colorMode={colorMode}
          stepColors={data.step_colors}
          width={1200}
          height={400}
          onBifurcationClick={drillDown}
        />
      </div>
    </div>
  );
}
```

- [ ] **Step 6: Verify TypeScript compiles and build works**

```bash
cd viewer && npx tsc --noEmit && npm run build
```

Expected: compiles without errors, `dist/index.html` produced.

- [ ] **Step 7: End-to-end test with real data**

```bash
# Rebuild the viewer template
cp viewer/dist/index.html moirai/viz/templates/viewer.html

# Generate a report
.venv/bin/moirai branch examples/eval_harness/ --viewer /tmp/moirai_viewer.html

# Open in browser
open /tmp/moirai_viewer.html
```

Expected: stream plot renders with colored bands splitting at bifurcation points. Color mode toggle switches between cluster colors and outcome gradient. Clicking a bifurcation node drills down.

- [ ] **Step 8: Commit**

```bash
git add viewer/src/ moirai/viz/templates/viewer.html
git commit -m "feat: stream plot with cluster and outcome color modes"
```

---

### Task 7: Step composition color mode

**Files:**
- Modify: `viewer/src/components/StreamPlot.tsx`

The step composition mode renders stacked proportions within each band cross-section. Instead of a single fill color, each band becomes a stack of thin horizontal stripes showing the step type distribution.

- [ ] **Step 1: Add composition rendering to StreamPlot**

Update `StreamPlot.tsx` to render stacked rectangles when `colorMode === "composition"`:

```tsx
// In StreamPlot.tsx, add a new function and update the band rendering:

function renderCompositionBand(
  band: BandSegment,
  tree: StreamTree,
  stepColors: Record<string, string>,
  key: string,
) {
  const branch = tree.branches[band.branchId];
  if (!branch) return null;

  // Use the midpoint column's proportions for this band
  const midCol = Math.round((branch.col_start + branch.col_end) / 2);
  const props = branch.step_proportions[midCol] ?? {};

  const bandHeight = band.y1 - band.y0;
  const entries = Object.entries(props).sort((a, b) => b[1] - a[1]);

  let yOffset = band.y0;
  return (
    <g key={key}>
      {entries.map(([step, proportion]) => {
        const h = proportion * bandHeight;
        const rect = (
          <rect
            key={step}
            x={band.x0}
            y={yOffset}
            width={band.x1 - band.x0}
            height={h}
            fill={stepColors[step] ?? "#6e7681"}
            opacity={0.85}
          />
        );
        yOffset += h;
        return rect;
      })}
    </g>
  );
}
```

Then in the render section, replace the band `<rect>` with a conditional:

```tsx
{colorMode === "composition"
  ? renderCompositionBand(band, tree, stepColors, `band-${i}`)
  : (
    <rect
      key={`band-${i}`}
      x={band.x0}
      y={band.y0}
      width={band.x1 - band.x0}
      height={band.y1 - band.y0}
      fill={bandFill(band.branchId)}
      opacity={0.85}
      rx={2}
    />
  )}
```

- [ ] **Step 2: Build and test**

```bash
cd viewer && npm run build
cp dist/index.html ../moirai/viz/templates/viewer.html
cd .. && .venv/bin/moirai branch examples/eval_harness/ --viewer /tmp/moirai_viewer.html
open /tmp/moirai_viewer.html
```

Expected: switching to "Step types" mode shows colored stacks within each stream band.

- [ ] **Step 3: Commit**

```bash
git add viewer/src/components/StreamPlot.tsx
git commit -m "feat: step composition color mode with stacked proportions"
```

---

### Task 8: Branch summary cards and run list

**Files:**
- Create: `viewer/src/components/BranchCards.tsx`
- Create: `viewer/src/components/RunList.tsx`
- Modify: `viewer/src/App.tsx`

- [ ] **Step 1: Create `BranchCards.tsx`**

```tsx
// viewer/src/components/BranchCards.tsx
import type { StreamTree, Bifurcation } from "../types";

interface Props {
  tree: StreamTree;
  focusBranchId: string;
}

export function BranchCards({ tree, focusBranchId }: Props) {
  // Find the first bifurcation under the focus branch
  const bif = Object.values(tree.bifurcations).find((b) => {
    const children = tree.tree[b.id];
    if (!children) return false;
    const leftBranch = tree.branches[children[0]];
    const rightBranch = tree.branches[children[1]];
    if (!leftBranch || !rightBranch) return false;
    const parent = tree.branches[focusBranchId];
    if (!parent) return false;
    return leftBranch.run_ids.length + rightBranch.run_ids.length === parent.run_ids.length;
  });

  if (!bif) return null;

  const left = tree.branches[bif.left_branch_id];
  const right = tree.branches[bif.right_branch_id];
  if (!left || !right) return null;

  return (
    <div style={{ display: "flex", gap: 16 }}>
      {[left, right].map((branch) => (
        <div
          key={branch.id}
          style={{
            flex: 1,
            background: "#161b22",
            border: "1px solid #30363d",
            borderRadius: 8,
            padding: "14px 18px",
            borderLeft: `3px solid ${branch.success_rate > 0.5 ? "#3fb950" : "#f85149"}`,
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
            <span style={{ fontSize: 13, fontWeight: 600, color: "#e6edf3" }}>
              {branch.run_ids.length} runs
            </span>
            <span style={{
              fontSize: 12,
              fontWeight: 600,
              color: branch.success_rate > 0.5 ? "#3fb950" : "#f85149",
            }}>
              {Math.round(branch.success_rate * 100)}% pass
            </span>
          </div>
          <div style={{ fontSize: 11, color: "#8b949e", lineHeight: 1.6 }}>
            {branch.trajectory}
          </div>
          <div style={{ fontSize: 10, color: "#484f58", marginTop: 8 }}>
            {Object.entries(branch.phase_mix)
              .slice(0, 4)
              .map(([phase, pct]) => `${Math.round(pct * 100)}% ${phase}`)
              .join(", ")}
          </div>
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 2: Create `RunList.tsx`**

```tsx
// viewer/src/components/RunList.tsx
import { useVirtualizer } from "@tanstack/react-virtual";
import { useRef } from "react";
import type { RunMeta } from "../types";

interface Props {
  runs: RunMeta[];
  expandedRuns: Set<string>;
  onToggleRun: (runId: string) => void;
}

export function RunList({ runs, expandedRuns, onToggleRun }: Props) {
  const parentRef = useRef<HTMLDivElement>(null);

  const virtualizer = useVirtualizer({
    count: runs.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 36,
    overscan: 10,
  });

  return (
    <div
      ref={parentRef}
      style={{
        maxHeight: 400,
        overflow: "auto",
        background: "#161b22",
        border: "1px solid #30363d",
        borderRadius: 8,
      }}
    >
      <div style={{ height: virtualizer.getTotalSize(), position: "relative" }}>
        {virtualizer.getVirtualItems().map((vi) => {
          const run = runs[vi.index];
          const expanded = expandedRuns.has(run.run_id);
          return (
            <div
              key={run.run_id}
              ref={virtualizer.measureElement}
              data-index={vi.index}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                transform: `translateY(${vi.start}px)`,
              }}
            >
              <div
                onClick={() => onToggleRun(run.run_id)}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                  padding: "8px 16px",
                  cursor: "pointer",
                  borderBottom: "1px solid #21262d",
                  fontSize: 12,
                }}
              >
                <span style={{
                  width: 6,
                  height: 6,
                  borderRadius: "50%",
                  background: run.success ? "#3fb950" : "#f85149",
                  flexShrink: 0,
                }} />
                <span style={{ color: "#e6edf3", flex: 1, fontFamily: "monospace" }}>
                  {run.run_id}
                </span>
                <span style={{ color: "#8b949e", fontSize: 10 }}>
                  {run.step_count} steps
                </span>
                <span style={{ color: "#484f58", fontSize: 10 }}>
                  {expanded ? "▾" : "▸"}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Wire up in App.tsx**

Add below the StreamPlot in `App.tsx`:

```tsx
import { BranchCards } from "./components/BranchCards";
import { RunList } from "./components/RunList";

// ... inside the App return, after the StreamPlot div:

      {/* Branch summary cards */}
      <div style={{ padding: "0 40px 16px" }}>
        <BranchCards tree={task} focusBranchId={currentFocusBranch} />
      </div>

      {/* Run list */}
      <div style={{ padding: "0 40px 40px" }}>
        <h3 style={{ fontSize: 12, color: "#8b949e", margin: "0 0 8px", fontWeight: 400 }}>
          Runs ({task.branches[currentFocusBranch]?.run_ids.length ?? 0})
        </h3>
        <RunList
          runs={
            (task.branches[currentFocusBranch]?.run_ids ?? [])
              .map((rid) => data.runs[rid])
              .filter(Boolean)
          }
          expandedRuns={useViewerStore.getState().expandedRuns}
          onToggleRun={useViewerStore.getState().toggleRun}
        />
      </div>
```

- [ ] **Step 4: Build and test**

```bash
cd viewer && npm run build
cp dist/index.html ../moirai/viz/templates/viewer.html
cd .. && .venv/bin/moirai branch examples/eval_harness/ --viewer /tmp/moirai_viewer.html
open /tmp/moirai_viewer.html
```

Expected: branch cards appear below the stream. Run list appears with pass/fail indicators.

- [ ] **Step 5: Commit**

```bash
git add viewer/src/
git commit -m "feat: branch summary cards and virtualized run list"
```

---

### Task 9: Step timeline (run detail expansion)

**Files:**
- Create: `viewer/src/components/StepTimeline.tsx`
- Modify: `viewer/src/components/RunList.tsx`

- [ ] **Step 1: Create `StepTimeline.tsx`**

```tsx
// viewer/src/components/StepTimeline.tsx
import type { StepDetail } from "../types";

interface Props {
  steps: StepDetail[];
}

export function StepTimeline({ steps }: Props) {
  return (
    <div style={{ padding: "8px 16px 16px 34px" }}>
      {steps.map((step, i) => (
        <div
          key={step.idx}
          style={{
            display: "flex",
            alignItems: "flex-start",
            gap: 10,
            position: "relative",
            paddingBottom: i < steps.length - 1 ? 2 : 0,
          }}
        >
          {/* Connecting line */}
          {i < steps.length - 1 && (
            <div style={{
              position: "absolute",
              left: 2,
              top: 20,
              width: 1,
              height: "calc(100% - 4px)",
              background: "#30363d",
            }} />
          )}

          {/* Color indicator */}
          <div style={{
            width: 5,
            minHeight: 28,
            borderRadius: 2,
            background: step.color,
            flexShrink: 0,
            marginTop: 2,
          }} />

          {/* Content */}
          <div style={{ flex: 1, minWidth: 0, paddingBottom: 6 }}>
            <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
              <span style={{ fontSize: 12, fontWeight: 600, color: "#e6edf3" }}>
                {step.enriched}
              </span>
              {step.detail && (
                <span style={{ fontSize: 11, color: "#8b949e", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {step.detail}
                </span>
              )}
              <span style={{ marginLeft: "auto", fontSize: 10, flexShrink: 0 }}>
                {step.status === "ok"
                  ? <span style={{ color: "#3fb950" }}>✓</span>
                  : <span style={{ color: "#f85149" }}>✗</span>}
              </span>
            </div>
            {/* Metrics */}
            {Object.keys(step.metrics).length > 0 && (
              <div style={{ fontSize: 10, color: "#484f58", marginTop: 2 }}>
                {step.metrics.tokens_in != null && `${(step.metrics.tokens_in / 1000).toFixed(0)}k in`}
                {step.metrics.tokens_out != null && ` · ${(step.metrics.tokens_out / 1000).toFixed(1)}k out`}
                {step.metrics.latency_ms != null && ` · ${(step.metrics.latency_ms / 1000).toFixed(1)}s`}
              </div>
            )}
            {/* Phase tag */}
            <span style={{
              display: "inline-block",
              fontSize: 9,
              color: "#8b949e",
              background: "#21262d",
              borderRadius: 3,
              padding: "1px 5px",
              marginTop: 3,
            }}>
              {step.phase}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 2: Integrate into RunList**

Update `RunList.tsx` to render `StepTimeline` when a run is expanded. Import the component and the data types, then conditionally render it below the run row:

```tsx
import { StepTimeline } from "./StepTimeline";
import type { RunMeta, RunDetail } from "../types";

// Add to Props:
interface Props {
  runs: RunMeta[];
  runDetails: Record<string, RunDetail>;
  expandedRuns: Set<string>;
  onToggleRun: (runId: string) => void;
}

// After the run row div, add:
{expanded && runDetails[run.run_id] && (
  <StepTimeline steps={runDetails[run.run_id].steps} />
)}
```

And pass `runDetails={data.run_details}` from App.tsx.

- [ ] **Step 3: Build and test end-to-end**

```bash
cd viewer && npm run build
cp dist/index.html ../moirai/viz/templates/viewer.html
cd .. && .venv/bin/moirai branch examples/eval_harness/ --viewer /tmp/moirai_viewer.html
open /tmp/moirai_viewer.html
```

Expected: clicking a run in the list expands a step timeline showing each step with its enriched name, detail, status, and metrics.

- [ ] **Step 4: Commit**

```bash
git add viewer/src/
git commit -m "feat: step timeline with detail expansion in run list"
```

---

### Task 10: Build pipeline and packaging

**Files:**
- Modify: `pyproject.toml`
- Create: `scripts/build-viewer.sh`

- [ ] **Step 1: Create build script**

```bash
#!/usr/bin/env bash
# scripts/build-viewer.sh
# Build the React viewer and copy it into the Python package
set -euo pipefail

cd "$(dirname "$0")/../viewer"
npm ci
npm run build
cp dist/index.html ../moirai/viz/templates/viewer.html
echo "Viewer built and copied to moirai/viz/templates/viewer.html"
```

```bash
chmod +x scripts/build-viewer.sh
```

- [ ] **Step 2: Update pyproject.toml to include template**

Add the viewer template to the package data. In `pyproject.toml`, ensure the templates directory is included:

```toml
[tool.setuptools.package-data]
moirai = ["viz/templates/*.html"]
```

- [ ] **Step 3: Build and verify full pipeline**

```bash
# Build viewer
./scripts/build-viewer.sh

# Reinstall moirai
pip install -e .

# Generate viewer report
.venv/bin/moirai branch examples/eval_harness/ --viewer /tmp/moirai_viewer.html

# Verify it works
open /tmp/moirai_viewer.html
```

Expected: full interactive viewer opens in browser.

- [ ] **Step 4: Commit**

```bash
git add scripts/build-viewer.sh pyproject.toml moirai/viz/templates/viewer.html
git commit -m "feat: viewer build pipeline and packaging"
```

---

### Task 11: Polish and edge cases

**Files:**
- Modify: `viewer/src/components/StreamPlot.tsx`
- Modify: `moirai/viz/stream.py`

- [ ] **Step 1: Handle single-task case (no tabs needed)**

In `App.tsx`, the task tabs already skip rendering for single tasks (`data.tasks.length > 1`). Verify this works with a single-task dataset.

- [ ] **Step 2: Handle tasks with very few runs (3-4 runs)**

Generate a viewer with minimal data and verify the stream doesn't collapse to invisible:

```bash
# Use synthetic helpdesk which has fewer runs
.venv/bin/moirai branch examples/synthetic_helpdesk/ --viewer /tmp/moirai_small.html
open /tmp/moirai_small.html
```

- [ ] **Step 3: Add low-confidence pattern for outcome mode**

In `StreamPlot.tsx`, add an SVG `<defs>` pattern for branches with fewer than 5 runs:

```tsx
// Add to the SVG, inside <svg>:
<defs>
  <pattern id="low-confidence-hatch" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)">
    <line x1="0" y1="0" x2="0" y2="6" stroke="rgba(255,255,255,0.08)" strokeWidth="1" />
  </pattern>
</defs>

// For outcome mode bands with few runs, overlay the hatch:
{colorMode === "outcome" && branch.run_ids.length < 5 && (
  <rect
    x={band.x0}
    y={band.y0}
    width={band.x1 - band.x0}
    height={band.y1 - band.y0}
    fill="url(#low-confidence-hatch)"
  />
)}
```

- [ ] **Step 4: Sanitize attrs in Python before serialization**

In `moirai/viz/stream.py`, ensure `_step_detail_str` escapes any HTML-like content:

```python
import html

def _step_detail_str(step) -> str:
    """Extract a human-readable detail string from step attrs."""
    raw = ""
    if "file_path" in step.attrs:
        path = step.attrs["file_path"]
        raw = path.rsplit("/", 1)[-1] if "/" in path else path
    elif "command" in step.attrs:
        cmd = step.attrs["command"]
        raw = cmd[:80] + "..." if len(cmd) > 80 else cmd
    elif "pattern" in step.attrs:
        raw = step.attrs["pattern"]
    return html.escape(raw)
```

- [ ] **Step 5: Build, test, commit**

```bash
cd viewer && npm run build
cp dist/index.html ../moirai/viz/templates/viewer.html
cd .. && .venv/bin/moirai branch examples/eval_harness/ --viewer /tmp/moirai_viewer.html
open /tmp/moirai_viewer.html
```

```bash
git add viewer/src/ moirai/viz/stream.py moirai/viz/templates/viewer.html
git commit -m "fix: edge cases, low-confidence patterns, sanitization"
```

---

### Task 12: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add viewer section to README**

After the existing HTML report section, add:

```markdown
### Interactive viewer

`moirai branch --viewer viewer.html` generates an interactive stream-based trajectory explorer. Runs flow left to right as a stream that splits at divergence points. Three color modes:

- **Clusters** — each behavioral branch gets a distinct color
- **Step types** — stacked proportions of step categories (read, search, edit, test, bash) at each position
- **Outcomes** — success rate gradient from blue (pass) to red (fail)

Click any bifurcation point to drill into that subtree. Expand individual runs to see step-by-step detail with file paths, commands, and metrics.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add viewer section to README"
```
