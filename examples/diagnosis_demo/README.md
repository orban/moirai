# Diagnosis demo: The Invisible Regression

Proves that pass-rate-based evals miss behavioral regressions that trajectory structure reveals.

## Quick start

```bash
# Generate synthetic data and run diagnosis
bash examples/diagnosis_demo/run_demo.sh
```

Or step by step:

```bash
# Generate the "invisible regression" scenario
python examples/diagnosis_demo/generate.py prompt_regression /tmp/demo_runs --seed 42

# Compare pass rates (looks fine — both ~70%)
moirai diff /tmp/demo_runs --a variant=baseline --b variant=current

# Extract behavioral evidence (reveals the problem)
moirai evidence /tmp/demo_runs --baseline variant=baseline --current variant=current

# Rank candidate causes
moirai diagnose /tmp/demo_runs \
    --baseline variant=baseline --current variant=current \
    --causes examples/diagnosis_demo/causes.json
```

## The scenario

A team deploys a system prompt change that removes "always run tests before submitting." They also upgraded the model and reduced tool timeouts.

**Pass rate comparison**: baseline 76%, current 72%. Looks like noise. Ship it.

**Trajectory analysis reveals**: agents stopped testing their fixes. `TEST_AFTER_EDIT_RATE` dropped from 91% to 27%. `BLIND_SUBMIT_RATE` jumped from 9% to 73%. Hard tasks regressed badly (-30pp), but easy tasks improved (+20pp), hiding the problem in the aggregate.

**Cause ranking**: The prompt change (C1) scores 0.84. Model upgrade (C2) scores 0.06. Tool timeout (C3) scores 0.04. The system correctly identifies the prompt change as the cause.

## Negative control

The timeout scenario (`timeout_regression`) has the tool timeout as the actual cause. The ranking correctly flips: C3 (timeout) scores highest, C1 (prompt) is secondary. This proves the model responds to evidence, not cause ordering.

## Robustness

Run with `--seed 42`, `--seed 123`, `--seed 456`. C1 ranks first in all three.

## Files

- `generate.py` — synthetic run generator (two scenarios)
- `causes.json` — three candidate causes with expected behavioral shifts
- `run_demo.sh` — full demo script with both scenarios + robustness check
