---
title: "feat: feature-aware branch command"
type: feat
date: 2026-04-05
status: deepened
---

# feat: feature-aware branch command

## Overview

Add `--task` and `--feature` flags to `moirai branch`. `--task` filters to a single task. `--feature` annotates each run with a behavioral feature value from the registry, shows the within-task median split, and labels each run as HIGH or LOW. This connects `moirai features` (what predicts success) to `moirai branch` (what one task's runs look like).

## Problem statement

`moirai features` and `moirai branch` are disconnected tools. Features identifies test_position_centroid as the strongest predictor, but branch has no way to show that feature on the alignment view. The user has to mentally connect "test timing matters" with "here's where test steps are." The tool should make the connection.

## The connected workflow

```bash
# Step 1: What predicts success?
moirai features examples/swe_rebench --output results.json
# → test_position_centroid +6.5pp

# Step 2: Show me that feature on one task
moirai branch examples/swe_rebench \
    --task "vyperlang__vyper-4385" \
    --feature test_position_centroid
# → alignment + per-run centroid values + median split + HIGH/LOW labels
```

## Proposed solution

### Files touched

| File | Lines | Change |
|---|---|---|
| `moirai/analyze/features.py` | end | Add `compute_feature_for_runs()` |
| `moirai/cli.py` | 162-171 | Add `--task` and `--feature` to `branch` params |
| `moirai/cli.py` | 186-188 | Filter `mixed_tasks` by `--task` |
| `moirai/cli.py` | 220-241 | Add feature annotation to terminal output |
| `moirai/viz/terminal.py` | end | Add `print_feature_annotation()` |
| `tests/test_features.py` | end | Tests for `compute_feature_for_runs()` |

### `moirai/analyze/features.py` addition

```python
def compute_feature_for_runs(
    runs: list[Run],
    feature_name: str,
) -> dict[str, float | None]:
    """Compute a named feature for each run.

    Returns {run_id: value} dict. Raises ValueError if feature_name
    is not in the FEATURES registry.
    """
    spec = None
    for f in FEATURES:
        if f.name == feature_name:
            spec = f
            break
    if spec is None:
        valid = [f.name for f in FEATURES]
        raise ValueError(f"Unknown feature '{feature_name}'. Valid: {valid}")

    return {r.run_id: spec.compute(r) for r in runs}
```

Returns dict (not list of tuples) for O(1) lookup by run_id in the viz layer.

### CLI changes (`moirai/cli.py:162-251`)

**Add two params** to existing `branch` command at line 162:

```python
def branch(
    path: Path = typer.Argument(..., help="Path to a run file or directory"),
    strict: bool = typer.Option(False, help="Treat warnings as errors"),
    model: str | None = typer.Option(None, help="Filter by model"),
    harness: str | None = typer.Option(None, help="Filter by harness"),
    task_family: str | None = typer.Option(None, "--task-family", help="Filter by task family"),
    task: str | None = typer.Option(None, "--task", help="Filter to a specific task ID"),
    feature: str | None = typer.Option(None, "--feature",
        help="Annotate runs with a behavioral feature (from moirai features)"),
    html: Path | None = typer.Option(None, help="Write HTML output to path"),
    analyze: bool = typer.Option(False, help="Run LLM analysis"),
    viewer: Path | None = typer.Option(None, help="Write interactive viewer HTML"),
) -> None:
```

**Filter mixed_tasks** at line 186-188 (after building mixed_tasks dict):

```python
if task:
    if task in mixed_tasks:
        mixed_tasks = {task: mixed_tasks[task]}
    else:
        err_console.print(f"[red]Task '{task}' not found or has no mixed outcomes.[/red]")
        # Show close matches
        raise typer.Exit(2)
```

**Compute feature values** after alignment, before viz (line ~215):

```python
feature_values = None
feature_spec = None
if feature:
    from moirai.analyze.features import compute_feature_for_runs, FEATURES
    feature_values = compute_feature_for_runs(task_runs, feature)
    feature_spec = next(f for f in FEATURES if f.name == feature)
```

**Pass to terminal output** in the per-task print loop (line ~220):

```python
# Existing output: task header, alignment, divergence points
# NEW: if feature_values, print feature annotation after task header
if feature_values:
    from moirai.viz.terminal import print_feature_annotation
    print_feature_annotation(task_runs, feature_values, feature_spec)
```

### Terminal output (`moirai/viz/terminal.py`)

```python
def print_feature_annotation(
    runs: list[Run],
    feature_values: dict[str, float | None],
    feature_spec,
) -> None:
    """Print per-run feature values with median split and HIGH/LOW labels."""
    # Collect values with outcomes
    valued = [(r, feature_values.get(r.run_id)) for r in runs]
    valued_nums = [(r, v) for r, v in valued if v is not None]

    if not valued_nums:
        console.print(f"  [dim]No runs have data for {feature_spec.name}[/dim]")
        return

    # Median split
    sorted_vals = sorted(v for _, v in valued_nums)
    median = sorted_vals[len(sorted_vals) // 2]

    high = [(r, v) for r, v in valued_nums if v > median]
    low = [(r, v) for r, v in valued_nums if v <= median]
    high_pass = sum(1 for r, _ in high if r.result.success) / len(high) if high else 0
    low_pass = sum(1 for r, _ in low if r.result.success) / len(low) if low else 0

    # Header
    direction = "higher" if feature_spec.direction == "positive" else "lower"
    console.print(f"\n  [bold]Feature: {feature_spec.name}[/bold]")
    console.print(f"  [dim]{feature_spec.description} ({direction} = better)[/dim]")
    console.print(f"  Median split: HIGH (>{median:.2f}) = {high_pass:.0%} pass"
                  f" | LOW (≤{median:.2f}) = {low_pass:.0%} pass\n")

    # Per-run values sorted by feature value
    for r, v in sorted(valued_nums, key=lambda x: x[1] or 0):
        outcome = "[green]pass[/green]" if r.result.success else "[red]fail[/red]"
        group = "[bold]HIGH[/bold]" if v > median else "[dim]LOW[/dim]"
        console.print(f"    {outcome}  {feature_spec.name}: {v:.2f}  {group}")
```

### Expected output

```
$ moirai branch data/ --task "vyperlang__vyper-4385" --feature test_position_centroid

vyperlang__vyper-4385: 10 runs (5P/5F), aligned to 156 columns

  Feature: test_position_centroid
  Where in the trajectory testing is concentrated (higher = better)
  Median split: HIGH (>0.28) = 100% pass | LOW (≤0.28) = 0% pass

    fail  test_position_centroid: 0.05  LOW
    fail  test_position_centroid: 0.05  LOW
    fail  test_position_centroid: 0.07  LOW
    fail  test_position_centroid: 0.14  LOW
    fail  test_position_centroid: 0.28  LOW
    pass  test_position_centroid: 0.68  HIGH
    pass  test_position_centroid: 0.69  HIGH
    pass  test_position_centroid: 0.78  HIGH
    pass  test_position_centroid: 0.81  HIGH
    pass  test_position_centroid: 0.84  HIGH

  Divergence points (top 3):
    ...existing output...
```

## Acceptance criteria

- [ ] `moirai branch path --task X` filters to one task
- [ ] `moirai branch path --task X --feature test_position_centroid` shows per-run values with median split
- [ ] Invalid feature names produce a clear error listing valid options
- [ ] Invalid task names produce a clear error (with close matches if possible)
- [ ] `--feature` without `--task` works (shows feature for all mixed tasks)
- [ ] `compute_feature_for_runs` returns dict and is tested
- [ ] Existing `branch` behavior unchanged when `--task` and `--feature` not provided

## Implementation phases

### Phase 1: Library function (10 min)
- `compute_feature_for_runs()` in features.py returning `dict[str, float | None]`
- Tests: valid feature, invalid feature (ValueError), runs with None values

### Phase 2: CLI + terminal (25 min)
- Add `--task` and `--feature` params to `branch` command
- Task filtering logic (with error on not found)
- Feature computation after alignment
- `print_feature_annotation()` in terminal.py
- Median split + HIGH/LOW labels

### Phase 3: Blog integration (15 min)
- Update Panel 2a to reference `moirai branch --feature`
- Update blog prose to show the connected workflow
- Panel 2a SVG script optionally reads from branch output

## Edge cases

- **Feature returns None for all runs** (e.g., no test steps): print "No runs have data for this feature"
- **All runs have same feature value** (degenerate median): everyone is LOW, print warning
- **`--task` with non-mixed task**: print error explaining it needs mixed outcomes
- **`--feature` without `--task`**: works, shows feature for each mixed task (could be verbose — cap at 5 tasks with a note)

## References

- Brainstorm: `docs/brainstorms/2026-04-05-feature-aware-branch-brainstorm.md`
- Features plan: `docs/plans/2026-04-05-feat-behavioral-features-pipeline-plan.md`
- Existing branch command: `moirai/cli.py:162-251`
- Existing terminal output: `moirai/viz/terminal.py:358-446`
