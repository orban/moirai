# Same agent, same task, different outcome: what stochastic variation reveals about coding agents

When an AI coding agent runs the same SWE-bench task twenty times, it succeeds roughly half the time. The failures aren't random. By aligning trajectories and finding where successful and unsuccessful runs make different choices, we can identify specific behavioral patterns that predict outcomes — and measure their effects with the precision of a within-subjects experiment.

## The data

The Nebius SWE-rebench dataset contains 12,854 runs of one agent (OpenHands v0.54.0 with Qwen3-Coder-480B) across 1,096 software engineering tasks, each attempted 10-20 times. Same model, same scaffold, same configuration. 707 tasks have at least 3 passing and 3 failing runs — enough for statistical analysis within each task.

All results below use within-task analysis: we subtract the per-task mean for each feature before pooling across tasks. This controls for task difficulty by design. The stochastic variation also controls for agent architecture and scaffold differences (there's only one agent). What remains is the effect of behavioral choices under conditions where the primary known confounds are eliminated.

A caveat: "within-task control" is strong but not airtight. The agent's early stochastic choices create path dependence — a run that reads the right file first may be on a qualitatively different trajectory from one that doesn't, and downstream behaviors correlate with that initial choice. The effects we measure are predictive associations in a well-controlled setting, not randomized-trial causal estimates. The causal interpretation is reasonable for several findings but should be treated as a hypothesis for future intervention experiments to confirm.

## Five within-task predictive effects

For each feature, we split runs within each task at the median and compare pass rates between the high and low groups. With stochastic variation as the source of between-run differences, within-task comparisons isolate behavioral variation from task-level confounds.

| Feature | Within-task Δ | p (sign test) | p (t-test) | Tasks | Effect |
|---|---|---|---|---|---|
| Test after editing | +39pp | 0.0003 | <0.0001 | 28 | Runs that verify edits pass at 59% vs 19% |
| Uncertainty density | -8.3pp | <0.0001 | <0.0001 | 707 | More hedging language → more failure |
| Step count | -6.9pp | <0.0001 | <0.0001 | 625 | Shorter runs succeed more |
| Test ratio | +5.9pp | <0.0001 | <0.0001 | 701 | Higher fraction of testing → more success |
| Edit density | -2.9pp | 0.052 | 0.026 | 707 | Marginal; more editing → slightly worse |

Test-after-edit is the strongest effect by a large margin: roughly a 40 percentage point gap from a single behavioral choice. The estimate comes from 28 qualifying tasks (those where the agent stochastically varied on this behavior), so the confidence interval is wide — the true effect could be anywhere from 20pp to 55pp. But the direction is consistent: 23 of 27 non-tied tasks show the same sign (p=0.0003 by sign test).

Uncertainty density is the second strongest and the most novel. We count hedging words ("maybe," "I think," "not sure," "could be") per reasoning step using regex. When the same agent hedges more on the same task, it fails 8pp more often. This was previously tested on cross-agent data and dismissed as "confounded with task difficulty." On same-agent data, the confound disappears and the signal is clear.

With Bonferroni correction across the 12 behavioral features tested, the significance threshold is p<0.004. The top four features (test-after-edit, uncertainty density, step count, test ratio) survive this correction. Edit density does not.

## The agent's fate is decided early

Pairwise Needleman-Wunsch alignment of pass/fail run pairs finds that trajectories diverge at median position 0.15 — roughly the agent's first real strategic choice, about 10-15 steps into a 70-step trajectory.

Clustering these divergence points (TF-IDF on the divergence content, KMeans) produces 17 groups covering 98% of cases. The four largest:

- **Wasted orientation (25%)**: the first substantive step is `cd && pwd` or a version check — producing zero useful information
- **Wrong file targeted (18%)**: pass and fail runs both explore, but pick different starting files
- **Skipped reasoning (13%)**: the passing run thinks before acting; the failing run acts immediately
- **Edit-read loops (8%)**: the agent cycles between reading and editing without progress

These cluster labels are descriptive interpretations of unsupervised clusters, not validated causal categories. The 98% coverage means the clustering captured the data, not that 98% of failures are explained. Without a permutation baseline, some of the 25pp outcome spread between clusters could reflect sampling noise rather than genuine strategy differences.

## Strategy clustering: the agent has distinct modes

Agglomerative clustering on NW trajectory distance (average linkage, k=2 and k=3 evaluated, best outcome-separation retained) shows that 78% of tasks produce 2+ distinct behavioral clusters from the same agent, with a median 25 percentage point outcome spread between the best and worst cluster. Without a permutation null for this spread, some of it may reflect small-sample noise in per-cluster pass rates. But the magnitude — a quarter of tasks show >50pp spread between clusters — suggests genuine strategy variation.

## What the semantic layer adds

We classified 14,844 steps across 210 runs (35 tasks) using Claude Haiku, assigning each step to one of 12 outcome-blind categories. Two semantic features reach within-task significance:

- **FIX_ATTEMPT fraction** (r=-0.186, p=0.006): runs with more fix-attempt steps fail more. Partially overlaps with structural `edit_density`.
- **TEST_VERIFY fraction** (r=+0.143, p=0.038): runs with more verification steps succeed more. This is new signal — the structural layer can't distinguish test-verification from general bash activity.

Four semantic transitions reach within-task significance across the 35 tasks, though none survive Bonferroni correction for the ~70 bigrams tested. The most suggestive: `EXPLORE_BROAD → FIX_ATTEMPT` (jumping from broad search to fixing, fail-correlated, p=0.005) and `FIX_ATTEMPT → REPRODUCE` (verifying after fixing, pass-correlated, p=0.033). These are directional findings that need replication at larger scale.

A cautionary note on small samples: at n=30 runs, `EXPLORE_TARGETED` showed r=+0.346, looking like the strongest predictor in the dataset. At n=210, it shrank to r=+0.034. The features that survived scaling (FIX_ATTEMPT, TEST_VERIFY) could shrink further. The semantic layer's most robust contribution is in transition patterns, not single-feature prediction.

## HMM latent states

A K=8 Hidden Markov Model on enriched step name sequences (50-task subsample, 686 runs) discovers two significant within-task states:

- **"Thinking" state** (86% reasoning, 14% read): r=+0.121, p=0.001. Runs that spend more time in this state pass more.
- **"Stuck editing" state** (42% edit, 36% read source): r=-0.119, p=0.002. Runs that spend more time here fail more.

Both survive Bonferroni correction across the 8 HMM states (threshold p<0.006). These were discovered without supervision — the HMM finds the behavioral modes from the step sequence alone. Earlier testing on cross-agent data concluded HMMs "learn family confounds, not behavioral strategies." On same-agent data with no family confounds to learn, they discover genuine strategy variation. The earlier dismissal was correct for that data. It doesn't generalize to same-agent settings where family confounds are absent.

## Motif discovery

564 significant gapped motifs and 121 contiguous motifs (BH-corrected, q<0.05). Effect sizes are small — the top motifs show 5-8pp lift over baseline. Examples:

- Success: `write(source) → ... → edit(test_file) → ... → finish` (58% vs 53% baseline)
- Failure: `edit(source) → ... → write(source) → ... → write(source)` (50% vs 53%)

The sheer number of significant motifs (685 total) should be interpreted cautiously. With thousands of candidate motifs tested, BH correction controls the false discovery rate at 5%, meaning ~34 of these could be false positives. The specific motif examples are illustrative, not definitive.

## What doesn't work

Exploration ratio (r=-0.012, p=0.17): how much the agent explores doesn't predict success. Causal language density (p=0.18): saying "because" and "therefore" doesn't help. Code reference density (p=0.55): mentioning specific code doesn't help. Test-fail loops (p=0.29): too rare in this agent (3% of runs). Concordance within clusters (1/67 significant): which cluster a run is in predicts outcome; how typical it is within the cluster doesn't. Structural transition bigrams: dead on same-agent data (50% directional consistency).

## Limitations

**Single agent.** All findings are from OpenHands v0.54.0 + Qwen3-Coder-480B. The earlier cross-agent analysis found features that flip sign across frameworks (test-after-edit was r=-0.188 on eval_harness). These findings might be specific to this agent-model combination. Replication on other agents is needed.

**Single task domain.** SWE-rebench tasks are software engineering bug fixes in Python repositories. Generalization to other coding tasks (greenfield development, multi-language) or non-coding agent tasks is unknown.

**No held-out validation.** All features, motifs, and HMM states were discovered and evaluated on the same dataset. No held-out test set was used. The behavioral feature correlations and natural experiments use within-task controls, which mitigates overfitting to specific tasks, but a fully independent replication would be stronger.

**No intervention experiment.** We measured within-task predictive associations, not the effect of actually changing the agent's behavior. The test-after-edit finding is the closest to causal (stochastic variation as natural treatment assignment), but a proper intervention — modifying the scaffold to enforce test-after-edit and measuring the improvement — has not been run. The intervention experiment is designed but blocked on agent infrastructure setup.

**Exploratory analysis.** The features tested were not pre-registered. They were discovered through iterative analysis across multiple datasets over several weeks. The reported p-values are individually valid but should be interpreted in the context of an exploratory study, not a confirmatory one.

## The correction

Earlier work on cross-agent datasets led us to conclude that "no universal behavioral predictor of agent outcomes exists" and that "every abstraction layer is conditional on framework, task, and difficulty." That conclusion was correct for cross-agent data and the question it was answering: "do behavioral features predict outcomes across different agents?" They don't.

The same-agent stochastic reruns answer a different question: "within a single agent's behavioral variation, do features predict outcomes?" They do — strongly and consistently, with no Simpson's paradox, no sign flips, and four effects validated at p<0.001 in within-task natural experiments.

The earlier methodology was sound. The data couldn't answer the question we're now asking. What changed is the data, not the competence of the analysis. Cross-agent comparison answers "which agent is better." Same-agent stochastic analysis answers "what makes an agent succeed" — a harder and more useful question.

## Implications

**For agent training.** The 11,006 preference pairs extracted from divergence points are step-level DPO training signal — the agent's full reasoning and action at the exact moment where its choice determined the outcome. Labs doing RL on coding agents should mine stochastic variation from their own eval runs. The marginal cost is low if you're already running multi-trial evals for variance estimation, and the signal is cleaner than cross-task comparisons or synthetic judges.

**For agent evaluation.** Cross-agent behavioral comparisons are confounded by architecture. Published claims about "what makes agents succeed" based on multi-framework analyses should be treated skeptically unless within-agent controls are applied. Benchmarks should require multi-run submissions and report confidence intervals.

**For agent design.** Test-after-edit could be enforced as a scaffold constraint (not a prompt suggestion). The within-task association is +39pp; the actual improvement from enforcing it would likely be smaller but is worth testing. Auto-prepopulating the working directory could address the 25% of divergence points in the "wasted orientation" cluster, though the actual impact needs testing. Uncertainty detection (simple regex on reasoning text) could trigger structured hypothesis prompts.

**For interpretability.** Uncertainty density predicts failure from text alone (r=-0.100). HMM discovers "thinking" and "stuck" states from step sequences. These are external behavioral signals that point interpretability researchers to specific moments and internal states worth investigating — the model's activations when it generates hedging language in runs that fail vs succeed on the same task.

---

## Visualizations for README and blog post

**Figure 1: The test-after-edit effect.** Paired bar chart. Each thin grey bar pair shows one task's pass rate with vs without test-after-edit. Thick colored bars show the aggregate (59% vs 19%). Include a note that N=28 tasks and the confidence interval is wide. Takeaway: the gap is large and consistent across tasks.

**Figure 2: Within-task predictive effects.** Forest plot. Each row is one feature. X-axis: within-task delta in percentage points. Point estimate with bootstrapped 95% CI. Bonferroni threshold marked. Solid dots for p<0.004 (survives correction), hollow for p<0.05 only. Takeaway: four features survive strict correction; test-after-edit dominates.

**Figure 3: Divergence position distribution.** Histogram. X-axis: normalized position in trajectory (0=start, 1=end). Y-axis: count. Vertical line at median 0.15. Takeaway: the agent's strategy is set in the first 15% of the trajectory.

**Figure 4: Strategy clusters within one task.** Dendrogram + outcome heatmap. Pick one task with 3 clusters and large outcome spread. Left: NW-distance dendrogram. Right: horizontal bars colored pass (green) / fail (red). Label clusters with dominant strategy. Takeaway: same agent, same task, different strategies, different outcomes.

**Figure 5: Semantic transitions.** Directed graph. Nodes: 12 semantic categories. Edges: 4 significant within-task transitions, green (pass) or red (fail). Width proportional to effect size. Caveat label: "suggestive, p<0.05 uncorrected." Takeaway: the semantic layer captures intent-level patterns structural labels miss.

**Figure 6: The methodological correction.** Two panels. Left ("Cross-agent, 68K runs"): feature arrows pointing in contradictory directions, Simpson's paradox. Right ("Same-agent, 12K runs"): same features all pointing consistently. Takeaway: the techniques work when the data is right.

**Figure 7: Failure mode taxonomy.** Treemap or waffle chart. 17 divergence clusters sized by prevalence, top 4 labeled. Caveat: "unsupervised clusters, labels are descriptive interpretations." Takeaway: failure modes are concentrated — top 4 account for 64% of divergence points.
