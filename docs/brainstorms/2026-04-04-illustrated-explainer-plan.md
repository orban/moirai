---
title: "Implementation plan: 5-panel illustrated explainer"
date: 2026-04-04
status: ready-to-implement
---

## Overview

Build a Hugo blog post at `~/dev/web/ryanorban.com/content/posts/what-stochastic-variation-reveals.md` with 5 inline SVG figures generated from real SWE-rebench data. Light theme matching the blog. Sequel to "Stop Testing AI Agents Like Deterministic Code."

## 5 panels

### Panel 1: The hook
"Same task. Same agent. Ten runs. Five pass, five fail."
10 horizontal trajectory bars, colored step blocks, pass/fail bordered. Simple, immediate.

### Panel 2: The fork (money panel)
Aligned trajectories, zoom into divergence column. Side-by-side: pass run reasoning (2-3 sentences) vs fail run reasoning. Uses real data from a specific SWE-rebench task.

### Panel 3: At scale
1,096 tasks. Ranked horizontal bars of failure modes: wasted orientation, wrong file, skipped reasoning, edit-read loops. Computed from divergence clustering on reconverted data.

### Panel 4: What predicts it
Plain-language pass-rate deltas (NOT correlation coefficients):
- Test timing: +6.8pp
- Uncertainty: -6.5pp
- Trajectory arc: +6.1pp
- Hypothesis formation: -5.7pp
- Edit focus: +4.8pp

### Panel 5: What to do about it
Three scaffold interventions with detection conditions. Link to GitHub. Open source.

Footer: correction note + scope caveat.

## 9 implementation steps

### Step 1: Task selection script (30 min)
`scripts/blog_select_task.py` — find best SWE-rebench task for panels 1-2. Needs mixed outcomes, early divergence, clear reasoning text on both sides. Good candidate: fastavro__fastavro-773 or stephenhillier__starlette_exporter-81.

### Step 2: Panel 1 SVG (45 min)
`scripts/blog_panel1_hook.py` — 10 horizontal bars, colored step blocks, pass/fail borders. ViewBox-based responsive SVG. Light-theme colors.

### Step 3: Panel 2 SVG (60 min — critical path)
`scripts/blog_panel2_fork.py` — aligned matrix + divergence callout with foreignObject text wrapping for reasoning. Two-part figure: aligned bars on top, reasoning comparison on bottom.

### Step 4: Panel 3 SVG (30 min)
`scripts/blog_panel3_scale.py` — horizontal bar chart of failure modes. Run cluster_divergences() on DPO pairs or use validated numbers from experiment results.

### Step 5: Panel 4 SVG (45 min)
`scripts/blog_panel4_predictors.py` — pass-rate deltas in plain language. Load data, compute within-task median splits, report absolute rates for each group. Or use pre-validated NE deltas.

### Step 6: Panel 5 SVG (30 min)
`scripts/blog_panel5_interventions.py` — layout-only, three intervention cards with detection conditions. No data computation.

### Step 7: Hugo markdown post (45 min)
Assemble prose + SVG placeholders. Match existing post style: short paragraphs, direct address, no hedging. Frontmatter, style block, 5 panels with transitions.

### Step 8: Assembly script (15 min)
`scripts/blog_build.sh` or single Python script. Run all panel scripts, inject SVGs into markdown, write to Hugo content dir.

### Step 9: Preview (15 min)
`hugo server`, verify all 5 SVGs render, check responsive, check mobile, verify links.

## Dependency graph

```
Step 1 (task selection)
  ↓
Steps 2, 3 (need task_id)      Steps 4, 5, 6 (independent)
  ↓                               ↓
Step 7 (markdown assembly)
  ↓
Step 8 (build) → Step 9 (preview)
```

Steps 4, 5, 6 can run in parallel with 2, 3. Critical path: 1 → 3 → 7 → 8 → 9.

## Light-theme color palette

| Step family | Color | Hex |
|---|---|---|
| read(source) | Blue | #4a7fae |
| search | Teal | #3a9e96 |
| edit(source) | Orange | #d4802a |
| write | Amber | #c4a820 |
| test(pass) | Green | #2d9a3e |
| test(fail) | Olive | #8aad42 |
| bash | Red | #cc3e40 |
| reason | Purple | #9070a0 |
| gap | Light gray | #e5e2de |
| pass border | Green | #2d9a3e |
| fail border | Red | #cc3e40 |

## Data sources

- Reconverted SWE-rebench: /Volumes/mnemosyne/moirai/swe_rebench_v2/ (12,854 runs)
- DPO pairs: /Volumes/mnemosyne/moirai/dpo_same_agent.jsonl
- Swarm results: /Volumes/mnemosyne/moirai/swarm/results.jsonl
- Blog: ~/dev/web/ryanorban.com/ (Hugo, hugo-theme-til, light theme)
- Existing post for style reference: content/posts/stop-testing-agents-like-deterministic-code.md

## Key technical notes

- SVGs inline in markdown via raw HTML blocks (Hugo goldmark allows unsafe HTML)
- Use foreignObject for text wrapping in Panel 2
- ViewBox-based sizing for responsiveness (no fixed pixel widths)
- Python scripts import from moirai package (sys.path.insert)
- All step enrichment uses the CORRECTED compress.py (cd prefix stripping, test detection)

## Total estimated time: ~5 hours (3 hours on critical path with parallelism)
