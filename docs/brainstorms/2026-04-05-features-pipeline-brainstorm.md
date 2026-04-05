---
date: 2026-04-05
topic: features-pipeline
---

# End-to-end behavioral features pipeline

## What we're building

A `moirai/analyze/features.py` module and `moirai features` CLI command that goes from raw traces to ranked behavioral features with effect sizes, p-values, and split-half validation. Output is a terminal table + JSON export. Blog panel scripts read from the JSON instead of hardcoding numbers.

## Why this approach

The headline findings (test_position_centroid +6.8pp, 4 robust features) were produced by ad hoc swarm scripts, not by moirai. Without reproducibility, the findings can't be verified. The library needs to be the source of truth.

Approach chosen: library functions with CLI on top, plus a documented example script. Library-first means the functions are testable and composable. CLI means someone can clone and run. Example script means the full workflow is narrated.

## Scope: 6 features

The 4 split-half survivors + 2 from the blog presentation:

| Feature | What it measures | Direction |
|---|---|---|
| test_position_centroid | Where in the trajectory testing is concentrated | later = better |
| symmetry_index | How well the trajectory follows explore→modify→verify | higher = better |
| reasoning_hypothesis_formation | Explicit hypothesizing in reasoning text | more = worse |
| edit_concentration | Whether edits are spread across files or hammering one | spread = better |
| uncertainty_density | Hedging language in reasoning ("maybe", "might") | more = worse |
| trajectory_arc_score | How closely the trajectory matches the ideal arc | higher = better |

Designed as a registry so adding features 7-16 is just adding functions.

## Key decisions

- **6 features for v1**: the credible set we'd stake our reputation on + the blog set
- **Registry pattern**: each feature is a function `(Run) -> float | None` with metadata (name, direction, description)
- **Within-task natural experiment**: median-split within each task, sign test p-value, BH correction across features
- **Split-half validation**: random 50/50 task split, feature must be significant on both halves
- **Output**: terminal table + `--output results.json` for programmatic consumption
- **JSON is the source of truth**: blog panel scripts read from this, no more hardcoded numbers

## Output format (JSON)

```json
{
  "dataset": "examples/swe_rebench",
  "n_runs": 12854,
  "n_tasks": 1096,
  "features": [
    {
      "name": "test_position_centroid",
      "description": "Where in the trajectory testing is concentrated",
      "direction": "positive",
      "delta_pp": 6.8,
      "p_value": 0.00012,
      "q_value": 0.00048,
      "split_half": true,
      "pass_mean": 0.58,
      "fail_mean": 0.51
    }
  ]
}
```

## End-to-end demo flow

```bash
# 1. Import traces
python scripts/convert_swe_rebench.py /path/to/raw examples/swe_rebench

# 2. Compute features and rank
moirai features examples/swe_rebench --min-runs 10 --output results.json

# 3. Generate blog panels from results
python scripts/blog_panel4_predictors.py --from results.json
python scripts/blog_assemble.py
```

## Open questions

- Should `moirai features` also run the activity divergence analysis, or keep that as a separate `moirai divergence` step?
- How to handle the SWE-rebench data dependency (12,854 runs on SSD) — should the example work with a small bundled sample?

## Next steps

Run `/workflows:plan` for implementation details.
