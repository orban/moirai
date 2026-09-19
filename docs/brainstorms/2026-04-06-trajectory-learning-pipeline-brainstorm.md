---
date: 2026-04-06
topic: trajectory-learning-pipeline
---

# Trajectory Learning Pipeline: From Stochastic Variation to Training Signal

## What We're Building

A three-layer system that consumes moirai's trajectory analysis outputs as training signal for agent improvement. Not a single training method — three stacking layers, each validated before the next.

| Layer | Role | Consumes from moirai |
|-------|------|---------------------|
| SFT | Base capability | Successful trajectories from mixed-outcome tasks |
| ORM (process reward model) | Guide inference / search | Behavioral features, divergence points |
| Localized DPO | Refine decision boundaries | Preference pairs at divergence points |

## Why This Approach

### What the SWE-bench fine-tuning landscape shows

Five teams have fine-tuned Qwen2.5-Coder-32B for SWE-bench:

- **OpenHands-LM-32B**: SFT on 491 successes from GPT-4o/Claude → 37.2%
- **SWE-agent-LM-32B**: SFT on 5K successes from Claude 3.7 Sonnet → 40.2%
- **Skywork-SWE-32B**: SFT on 8K successes from 8 proprietary LLMs → 38.0%
- **SWE-Dev-32B**: SFT+RFT on 20K from DeepSeek-V3, **tested KTO/OREO preference learning, rejected it** → 36.6%
- **DeepSWE**: Pure online RL (GRPO) on Qwen3-32B, no SFT → 42.2%

The pattern: all successful approaches exploit variation across trajectories. SFT selects successes from many attempts. GRPO compares multiple rollouts per task. ORM scores trajectories post-hoc. The signal comes from *contrast between trajectories*, not single outputs.

SWE-Dev's preference learning failed because they used trajectory-level "this whole run is good vs bad" — which discards the localized signal at decision points that moirai captures. Whether step-level preferences at divergence points carry signal that trajectory-level preferences don't is an open empirical question.

### Why ORM before DPO

If DPO gives ambiguous results, we won't know if: the signal is weak, the implementation is wrong, the model can't absorb it, the dataset is too small, or the weighting is off. ORM provides immediate observability — if moirai's features predict outcomes at inference time, the signal is real. Then DPO becomes "can we distill this into the policy?"

ORM also gives immediate product value: plug it into OpenHands' retry/search loop for performance gains without touching the base model.

### The DeepSWE connection

GRPO compares rollouts on the same task, computes relative advantage, updates policy based on differences. That's structurally identical to what moirai does with stochastic variation analysis. The difference: DeepSWE is online (expensive, 64 H100s for 6 days), we're offline (cheap). The real question: can offline structured comparison approximate GRPO-style signal?

## Key Decisions

- **SFT training data**: Only successful trajectories from *mixed-outcome* tasks. If a task always succeeds, the success is trivial and carries no learning signal. Mixed-outcome successes encode something non-trivial.
- **Base model**: Qwen2.5-Coder-32B-Instruct (same base as all 4 SFT fine-tunes). Clean attribution — any improvement is from our training signal.
- **Existing data**: Use SWE-rebench trajectories from Qwen3-Coder-480B (12,854 runs, 1,096 mixed-outcome tasks, already on disk). Cross-model trajectory transfer is established (OpenHands-LM used GPT-4o/Claude trajectories to train a 32B).
- **ORM architecture**: Lightweight classifier on moirai's behavioral features (test centroid, uncertainty density, symmetry index, edit dispersion, hypothesis formation). Predict P(success | partial trajectory) at each step.
- **DPO data**: Existing 11K pairs from moirai export. Step-level, divergence-localized. Convert from plain text to TRL chat format.
- **Eval**: SWE-bench Verified (500 tasks, industry standard). Stratify results by task family per Simpson's Paradox finding.
- **Compute**: Cloud GPU (RunPod/Vast.ai). A100 80GB for training. vLLM for serving during eval.

## Execution Order

### Phase 1: SFT baseline
- Extract successful trajectories from mixed-outcome tasks in SWE-rebench
- Convert to SFT format (agent turn sequences)
- QLoRA fine-tune Qwen2.5-Coder-32B-Instruct
- Eval on SWE-bench Verified → baseline number

### Phase 2: ORM / process reward model
- Define feature vector from moirai's per-step behavioral features
- Train lightweight model: R(partial_trajectory) → P(success)
- Deploy as inference-time scoring: prune bad branches, retry when score drops, rank candidates
- Eval: SFT + ORM reranking → second number

### Phase 3: Localized DPO
- Convert 11K divergence pairs to TRL chat format
- DPO fine-tune (from SFT checkpoint, not base model)
- Eval: SFT + DPO → third number

### Phase 4: Compare
- SFT alone
- SFT + ORM
- SFT + DPO
- SFT + ORM + DPO

### What we learn
- Does SFT on mixed-outcome successes beat SFT on all successes?
- Does ORM from moirai features provide signal at inference time?
- Does localized DPO add anything on top of SFT?
- Do ORM and DPO compose?

## Open Questions

- **ORM architecture**: Linear probe on feature vector? Small MLP? Fine-tune a separate small LM as a critic? Trade-off between simplicity (validates signal faster) and capacity (might need context to score well).
- **Feature computation at inference time**: moirai's features currently require the full trajectory. For a process reward model, we need incremental computation — features from a *partial* trajectory. Some features (test centroid, edit dispersion) are naturally incremental. Others (symmetry index) need the full arc.
- **Cross-model transfer validity**: The DPO pairs are from 480B stochastic variation. Do divergence-point preferences generalize to a 32B model with different failure modes? SFT transfer works (proven by all 4 teams), but DPO transfer is untested.
- **Simpson's Paradox in training**: Should we stratify DPO pairs by task family? Or does mixing families provide useful generalization pressure?
- **Eval cost**: Each SWE-bench Verified pass requires serving the model + running 500 Docker containers. ~$100-200 per pass on cloud GPU. 4 conditions x 1 pass each = $400-800 minimum for the comparison. Multiple runs for variance? Gets expensive.

## Cost Estimate

| Phase | Compute | Estimated cost |
|-------|---------|---------------|
| SFT training (QLoRA, 32B, 1 epoch) | 1x A100, 6-12 hrs | $10-20 |
| ORM training (lightweight) | CPU or single GPU, <1 hr | <$5 |
| DPO training (QLoRA, 32B, 1 epoch) | 1x A100, 6-12 hrs | $10-20 |
| Eval: 4 conditions x SWE-bench Verified | 1x A100 serving, ~4-8 hrs each | $100-200 |
| Hyperparameter sweeps (7B model) | 1x A100, misc | $20-50 |
| **Total** | | **$150-300** |

Data generation cost: $0 (reusing existing SWE-rebench trajectories).

## Next Steps

→ `/workflows:plan` for implementation details, starting with Phase 1 (SFT) and Phase 2 (ORM) in parallel.
