# Real-data diagnosis demo

Two scenarios using real agent trace data from public SWE-bench datasets. Both prove the same point: pass-rate comparisons miss what trajectory analysis reveals.

## Quick start

```bash
# If you already have data in examples/swe_agent and examples/eval_harness:
bash examples/real_data_demo/run_demo.sh

# Or download and convert fresh data first:
bash examples/real_data_demo/setup_data.sh
bash examples/real_data_demo/run_demo.sh
```

## Scenario 1: The Cognitive Gap

**SWE-agent** (Llama-70B, command-line tool use) vs **Claude Code eval harness** (Claude-sonnet, function calls).

Pass rate: 10% vs 16%. A manager would say "eval harness is slightly better, but both are low." The diagnosis system says something completely different:

| Feature | SWE-agent | Eval harness | Shift |
|---------|-----------|--------------|-------|
| REASONING_DENSITY | 1.00 | 0.07 | -0.93 (large) |
| BLIND_SUBMIT_RATE | 0.33 | 0.00 | -0.33 (large) |
| ITERATIVE_FIX_RATE | 0.24 | 0.00 | -0.24 (large) |
| TEST_AFTER_EDIT_RATE | 0.50 | 0.32 | -0.18 (small) |

The diagnosis ranks **reasoning_approach** at 0.90 confidence. These agents have fundamentally different cognitive strategies despite similar outcomes. SWE-agent embeds reasoning in every step and blind-submits a third of the time. The eval harness uses pure tool execution with no explicit reasoning and never blind-submits.

The pass-rate comparison tells you nothing about this.

## Scenario 2: The Ambiguous Regression

**CoderForge** (Qwen-32B, OpenHands framework) vs **OpenHands** (various models, same framework).

Pass rate: 53% vs 48%. No single cause dominates:

| Cause | Score | CI |
|-------|-------|----|
| exploration_depth | 0.34 | [0.30, 0.37] |
| tool_environment | 0.25 | [0.23, 0.27] |
| verification_strategy | 0.15 | [0.13, 0.18] |
| reasoning_approach | 0.11 | [0.09, 0.12] |
| Unknown | 0.16 | — |

The system is honest about ambiguity. Four factors contribute, none dominant. The 16% unknown bucket says "something else is going on that our causes don't cover." This is what real diagnosis looks like — not every regression has a single root cause.

## Why these two scenarios matter

Together they demonstrate both modes of the diagnosis system:

1. **Clear signal**: when one factor dominates, the system finds it with high confidence and tight CIs
2. **Ambiguous signal**: when multiple factors compete, the system reports uncertainty honestly instead of forcing a false answer

Both scenarios share the same property: **the pass-rate comparison is nearly useless**. The 6-8% difference is within noise for these sample sizes. Only the trajectory-level behavioral features reveal what's actually different.

## Data sources

- [SWE-agent trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) — 80K traces, Llama-70B/8B
- [CoderForge](https://huggingface.co/datasets/togethercomputer/CoderForge-Preview) — 258K traces, Qwen-32B
- [OpenHands trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) — 67K traces, various models
