# Complete experiment results (final, corrected)

Date: 2026-04-04
Dataset: Nebius SWE-rebench (12,854 same-agent runs, OpenHands+Qwen3-Coder-480B, 1,096 tasks)
Reconverted with fixed test detection, improved uncertainty regex, expanded enrichment vocabulary (31 labels)

## Dataset

One agent (OpenHands v0.54.0 with Qwen3-Coder-480B) ran each task 10-20 times. 1,096 tasks had mixed outcomes. All analysis uses within-task centering (subtract per-task mean, pool across tasks) to control for task difficulty.

## Feature engineering swarm

9 agents explored 125 features across 2 rounds. Categories: test sequences, reasoning quality, temporal patterns, search quality, edit patterns, error handling, structural interactions, content-based features, composite features.

Results: 38 significant at p<0.001, 28 double-validated (correlation + natural experiment), 16 non-redundant after pairwise |r|>0.5 clustering, 4 survive split-half validation.

## The 4 robust features

| Feature | r | NE delta | NE p | Split-half |
|---|---|---|---|---|
| test_position_centroid | +0.121 | +6.8% | <0.001 | YES |
| symmetry_index | +0.101 | +6.1% | <0.001 | YES |
| reasoning_hypothesis_formation | -0.088 | -5.7% | <0.001 | YES |
| edit_concentration | +0.064 | +4.8% | <0.001 | YES |

## All within-task correlations (reconverted data)

| Feature | Within-task r | p |
|---|---|---|
| test_position_centroid | +0.121 | <0.001 |
| uncertainty_density | -0.109 | <0.001 |
| symmetry_index | +0.101 | <0.001 |
| reasoning_hypothesis_formation | -0.088 | <0.001 |
| trajectory_arc_score | +0.088 | <0.001 |
| reasoning_self_correction | -0.086 | <0.001 |
| test_ratio | +0.085 | <0.001 |
| step_count | -0.081 | <0.001 |
| reasoning_confidence_ratio | +0.067 | <0.001 |
| test_cadence_regularity | -0.067 | <0.001 |
| test_pass_count | +0.066 | <0.001 |
| edit_concentration | +0.064 | <0.001 |
| tool_diversity_entropy | +0.064 | <0.001 |
| reasoning_question_density | -0.063 | <0.001 |
| diagnosis_density | -0.059 | <0.001 |
| test_count | +0.055 | <0.001 |
| has_edit_test | +0.040 | <0.001 |
| edit_density | -0.034 | <0.001 |
| reasoning_density | +0.030 | <0.001 |

## Natural experiments (within-task median split)

| Feature | Delta | Sign test p | t-test p |
|---|---|---|---|
| has_edit_test | +14.9% | 0.001 | — |
| test_position_centroid | +6.8% | <0.001 | — |
| symmetry_index | +6.1% | <0.001 | — |
| step_count | -6.0% | <0.001 | <0.001 |
| test_ratio | +5.9% | <0.001 | <0.001 |
| reasoning_hypothesis_formation | -5.7% | <0.001 | <0.001 |
| uncertainty_density | -6.5% | <0.001 | <0.001 |
| edit_concentration | +4.8% | <0.001 | <0.001 |
| diagnosis_density | -3.4% | 0.001 | <0.001 |
| edit_density | -2.9% | 0.052 | 0.026 |

## Motif discovery (reconverted data)

83 contiguous + 1,960 gapped motifs (BH-corrected, q<0.05).

Success patterns: test(pass)→test(pass)→test(pass) (59%, lift 1.10). Failure patterns: edit(source)→write(test_file)→test(pass) at 45% (lift 0.85 — editing source then writing tests is a failure pattern).

## Strategy clustering (reconverted data)

75% of tasks form 2+ clusters (agglomerative on NW distance). Median outcome spread: 40pp between best and worst cluster.

## Divergence clustering (reconverted data)

13 clusters covering 98% of divergence points. Top: wasted orientation (18%), wrong file targeted, skipped reasoning.

## Enrichment vocabulary (after fixes)

31 labels. Top: read(source) 16%, test(pass) 13%, bash(python) 12%, search(grep_targeted) 10%, edit(source) 6%. No catch-all bucket exceeds 16%.

## Corrections applied

1. Test detection: substring match instead of startswith, test check before cd prefix classification, exit code parsing for pass/fail. Recovered 6,125 test steps from bash(explore).
2. Uncertainty regex: removed attempt (90% FP), perhaps (pass-biased), not sure (zero matches). Added actually-comma, let me step back/reconsider, another approach.
3. Step enrichment: strip cd prefix before classifying. Split bash(explore) catch-all into test, search, python, git, setup, explore, other. Added read(dir), read(docs).

## What doesn't work (confirmed on corrected data)

exploration_ratio, causal_density, code_ref_density, test_fail_count, has_test_fail_loop, concordance within-cluster, NW distance to consensus, structural transition bigrams, HMM latent states (on 31-label vocabulary).

## Eval harness results (Claude Code + Claude Sonnet)

Opshin: 42 trials, 17P/25F, 1 mixed-outcome task (fix-testcase 2P/1F).
Graphiti: 74 trials, 0P/74F, 0 mixed. Claude can't solve these cold.
Key finding: Claude Sonnet is more deterministic than Qwen3-Coder on these tasks. Most tasks are either always-pass or always-fail with no stochastic variation.

## Data locations

- Reconverted SWE-rebench: /Volumes/mnemosyne/moirai/swe_rebench_v2/ (12,854 runs)
- Swarm results: /Volumes/mnemosyne/moirai/swarm/results.jsonl (130 features)
- Feature selection report: /Volumes/mnemosyne/moirai/swarm/FEATURE_SELECTION_REPORT.md
- DPO pairs: /Volumes/mnemosyne/moirai/dpo_same_agent.jsonl (11,006 pairs)
- Opshin eval: /Volumes/mnemosyne/eval-results/results-opshin/
- Graphiti eval: /Volumes/mnemosyne/eval-results/results-graphiti/
