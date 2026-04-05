---
date: 2026-04-05
topic: feature-aware-branch
---

# Feature-aware branch command

## What we're building

Connect `moirai features` (macro: what predicts success across tasks) to `moirai branch` (micro: what does one task's runs look like) by adding a `--feature` flag that annotates the alignment view with per-run feature values.

## Why this approach

The features pipeline and the alignment pipeline are currently disconnected tools. Features compute per-run metrics without alignment. Alignment finds structural divergence without knowing about features. There's no workflow where the output of one informs the other.

The connection: features tell you WHAT to look for. Alignment shows you WHERE it happens. The `--feature` flag bridges them — the user runs `moirai features`, sees test_position_centroid is the strongest predictor, then runs `moirai branch --feature test_position_centroid` on a specific task and sees the aligned trajectories annotated with that feature's values.

## The workflow

```bash
# 1. Discover: what behavioral features predict success?
moirai features examples/swe_rebench --output results.json
# → test_position_centroid +6.5pp, uncertainty_density -7.0pp, ...

# 2. Drill down: what does that feature look like on one task?
moirai branch examples/swe_rebench --task "vyperlang__vyper-4385" \
    --feature test_position_centroid
# → alignment with per-run feature values, median split shown, pass/fail labeled
```

## Key decisions

- **`--feature` takes a feature name from the registry** — validated against `FEATURES` list
- **Per-run feature values shown alongside alignment** — right margin annotation (like Panel 2a already does with centroid values)
- **Median split annotated** — show which runs fall above/below the within-task median, and the pass rates for each group
- **Works with existing `branch` output** — adds a column, doesn't replace the alignment view
- **Terminal and HTML output** — terminal gets a feature value column, HTML gets markers

## Open questions

- Should `--feature` also work without `--task` (showing feature distributions across all tasks)?
- Should the output include a mini summary like "on this task, late testers pass at 100% vs early testers at 0%"?

## Next steps

Run `/workflows:plan` for implementation details.
