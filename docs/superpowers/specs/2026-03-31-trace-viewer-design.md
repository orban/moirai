# Trace viewer: interactive stream-based trajectory explorer

## Problem

The current HTML branch report is a static view. You can see heatmaps and fork comparisons, but you can't interact with the data — you can't click a split in the dendrogram to focus on a subtree, you can't see how trajectory composition changes along a branch, and you can't drill into individual runs. The report shows you the answer but doesn't let you explore the question.

## Design

Replace the static heatmap with an interactive stream plot inspired by single-cell trajectory inference visualizations (STREAM, Slingshot, ElPiGraph). Agent runs flow left to right through aligned step positions. The stream splits at bifurcation points derived from the existing dendrogram. Users navigate by clicking splits to drill into subtrees, switching color modes to see different facets of the data, and expanding individual runs to see step-level detail.

The frontend moves from Alpine.js + inline JS to React + Vite, compiled into a single self-contained HTML file. Python pre-computes all data and injects it into the built template. Users still run `moirai branch --html report.html` and get one file, no server needed.

## Stream plot

### Geometry

- X-axis: aligned step positions (columns from the Needleman-Wunsch alignment)
- Y-axis: no semantic meaning, used to separate branches vertically
- All runs start as one stream at x=0
- At each bifurcation (internal dendrogram node), the stream forks into two child streams
- Width of each stream segment is proportional to the number of runs in that branch
- Minimum width enforced so small branches don't disappear
- Fork transitions use cubic Bezier curves; straight segments are flat bands
- Branches with extreme outcome rates (relative to baseline) get a badge/marker

### Bifurcation x-positioning

Each split needs an x-position along the alignment axis. Strategy:

1. Primary: use the centroid of divergent columns within that subtree (columns where the two child clusters show different step distributions), smoothed over a window
2. Fallback: dendrogram merge height, scaled to the column range
3. Splits are ordered by dendrogram height (coarsest splits first)

Significant splits (p < 0.2 from existing Fisher's exact test) get solid bifurcation nodes. Non-significant splits get faded/dashed nodes so users don't mistake noise for signal.

### Color modes

Three modes, toggled via a bar at the top. The stream spatial layout is shared across all modes. Each mode requires its own pre-computed band geometry (not just a repaint).

**1. Cluster identity**
Each branch gets a distinct hue. Shows behavioral families. Good for orientation.

**2. Step composition**
At each x-position, the stream cross-section is stacked proportions of step categories: read, search, edit, test, bash, subagent, plan, other. Uses the existing step category color palette. Width encodes run count; color bands encode relative mix (not absolute counts). Shows how agent behavior shifts along each branch — e.g., a branch that starts with mostly explore (blue/green) then shifts to modify (orange) then verify (yellow).

**3. Outcome rate**
Stream segments colored by success rate. Uses a diverging blue (pass) to red (fail) palette (colorblind-safe, avoids red/green). Shows which forks are outcome-determining at a glance. Branches with very few runs get a subtle "low confidence" pattern fill so users don't over-interpret small samples.

## Interaction

### Bifurcation drill-down

Click a bifurcation node in the stream:

- View re-renders showing only runs in that subtree
- Uses pre-computed stream geometry for that internal dendrogram node (no client-side re-clustering)
- X-axis rescales to fill available width (more resolution on the steps within this branch)
- Breadcrumb trail appears: `All (483) > left (287) > left (84)` — labels show run count and success rate; click any ancestor to zoom back out. Branch labels are positional (left/right from the dendrogram split) since clusters don't have inherent names.

### Branch summary cards

Below the stream on drill-down, for each child branch at the next split:

- Run count and success rate
- Representative compressed trajectory: `read(config)x2 > search > edit > test > result`
- Phase mix percentages (explore/modify/verify/execute breakdown)

### Run list

Below branch summaries, a collapsible list of all runs in the current view:

- Each row: run_id, outcome badge (pass/fail), step count
- Click a run to expand its step cards
- Virtualized rendering (tanstack-virtual or similar) for large lists

### Step cards

When a run is expanded, each step renders as a compact card in a vertical timeline:

- Left edge colored by step category (same palette as stream)
- Enriched name in bold: `read(source)`
- Attrs detail in muted text: `urls.py` or `python -m pytest test/...`
- Status indicator (checkmark or x)
- Metrics if present: `377k in / 3.2k out / 2.1s`
- Phase label as subtle tag
- Cards connected by a thin vertical line (timeline, not table)

## Architecture

### Python side (pre-computation)

All heavy computation happens in Python before the HTML is written:

- Alignment matrix (Needleman-Wunsch) — existing
- Dendrogram / hierarchical clustering — existing
- Divergence point detection with p-values — existing
- **New: stream geometry tree.** For each internal dendrogram node, pre-compute:
  - Which runs belong to each child branch
  - Bifurcation x-position (centroid of divergent columns)
  - Per-column step type proportions for each branch (for step composition mode)
  - Per-branch success rates (for outcome mode)
  - Branch summary (representative trajectory, phase mix)
- **Two-tier data payload:**
  - Tier 1 (always loaded): stream geometry, branch summaries, run metadata (id, outcome, step count). Small.
  - Tier 2 (loaded on expand): per-run step detail (attrs, metrics, input/output summaries). Stored in compact format, parsed on demand.
- All serialized as JSON, injected into the built HTML template

Output: `write_branch_html()` produces one self-contained HTML file, same as today.

### Frontend (React + Vite)

```
viewer/
  package.json
  vite.config.ts          # vite-plugin-singlefile for single-HTML output
  src/
    main.tsx              # entry, reads injected JSON data
    App.tsx               # top-level layout, color mode state, drill-down state
    components/
      StreamPlot.tsx      # SVG stream rendering (visx for D3 primitives + React)
      BifurcationNode.tsx # clickable split nodes with significance indicators
      ColorModeToggle.tsx # cluster / composition / outcome toggle
      Breadcrumbs.tsx     # drill-down navigation
      BranchCards.tsx     # summary cards for child branches
      RunList.tsx         # virtualized run list
      StepTimeline.tsx    # vertical step cards for expanded run
    hooks/
      useStreamData.ts    # parses injected JSON, provides filtered data for current drill-down level
    lib/
      streamLayout.ts     # band geometry helpers (Bezier paths, stacked bands)
      colors.ts           # palettes for all three modes
```

**Key libraries:**
- `visx` — D3 primitives with React rendering, for the stream SVG
- `@tanstack/react-virtual` — virtualized run list
- `vite-plugin-singlefile` — compiles into one HTML file

**State model:** single React state tree (useReducer or zustand) as source of truth. Stream plot reads from it, user clicks dispatch actions (drill down, change color mode, expand run). No Alpine.js, no split state ownership.

**Build and integration:**
- `npm run build` in `viewer/` produces `dist/index.html` — one file, all JS/CSS inlined
- Python reads this template at package build time, ships it in the wheel
- At report generation time, Python injects the JSON data payload into a `<script>` tag in the template
- Users never need Node.js. The built template is a static asset in the Python package.

### Migration path

Phase 1: build the stream viewer as a new component alongside the existing branch report. `moirai branch --html` continues to produce the current heatmap report. A new flag (`moirai branch --viewer` or `moirai branch --html-v2`) produces the React-based stream viewer.

Phase 2: once the stream viewer is stable, it becomes the default `--html` output. The old heatmap view can be preserved as a tab/toggle within the React app if it's still useful.

Phase 3: other HTML reports (clusters, diff) migrate into the same React app as additional views.

## Data sanitization

All attrs values (file paths, commands, patterns) are escaped before embedding in the JSON payload. No raw HTML or script content reaches the template. React's JSX escaping handles the rendering side.

## Known limitations

- **Tree model only.** Agent trajectories can diverge and reconverge; the dendrogram forces irreversible splits. This is a limitation of the underlying clustering, not the viewer. A DAG/graph model would be a separate project.
- **Aligned column positions are not true time.** Horizontal distance in the stream reflects alignment positions, not wall-clock time or step count. The UI should label this clearly ("aligned position", not "time").
- **File size.** For very large populations (1000+ runs with rich attrs), the self-contained HTML may get large. The two-tier data approach mitigates this, but at extreme scale we may need an external JSON sidecar or a server mode.

## Not in scope

- Server mode or live data
- DAG/graph trajectory model (reconvergence)
- Content-level analysis (what the agent read/wrote, not just which tool)
- Migration of existing heatmap report to React (phase 2, not phase 1)
