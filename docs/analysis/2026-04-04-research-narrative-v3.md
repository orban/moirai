# Same agent, same task, different outcome: what stochastic variation reveals about coding agents

When an AI coding agent runs the same SWE-bench task twenty times, it succeeds roughly half the time. The failures aren't random. By aligning trajectories and finding where successful and unsuccessful runs make different choices, we can identify specific behavioral patterns that predict outcomes — and measure their effects with the precision of a within-subjects experiment.

The strongest single predictor: where in its trajectory the agent runs tests. Agents that test late — after exploring and editing — succeed more often. Not "test more" but "test at the right time."

## The data

The Nebius SWE-rebench dataset contains 12,854 runs of one agent (OpenHands v0.54.0 with Qwen3-Coder-480B) across 1,096 software engineering tasks, each attempted 10-20 times. Same model, same scaffold, same configuration. 707 tasks have at least 3 passing and 3 failing runs — enough for statistical analysis within each task.

All results below use within-task analysis: we subtract the per-task mean for each feature before pooling across tasks. This controls for task difficulty by design. The stochastic variation also controls for agent architecture and scaffold differences (there's only one agent). What remains is the effect of behavioral choices under conditions where the primary known confounds are eliminated.

A caveat: "within-task control" is strong but not airtight. The agent's early stochastic choices create path dependence — a run that reads the right file first may be on a qualitatively different trajectory from one that doesn't, and downstream behaviors correlate with that initial choice. The effects we measure are predictive associations in a well-controlled setting, not randomized-trial causal estimates. The causal interpretation is reasonable for several findings but should be treated as a hypothesis for future intervention experiments to confirm.

## Nine within-task predictive effects

125 candidate features were evaluated across two rounds of automated feature engineering (details in "The feature engineering swarm" below). Nine survive rigorous validation. For each, we report the within-task partial correlation and either a natural experiment delta or split-half consistency.

| Feature | Within-task r | Natural experiment | Split-half | Description |
|---|---|---|---|---|
| test_position_centroid | +0.121 | +6.8pp | YES | Tests concentrated later in trajectory → success |
| uncertainty_density | -0.109 | -8.3pp | — | More hedging language → failure |
| symmetry_index | +0.101 | +6.1pp | YES | Explore→modify→verify arc → success |
| reasoning_hypothesis_formation | -0.088 | -5.7pp | YES | Explicit hypothesizing → failure |
| trajectory_arc_score | +0.088 | +6.3pp | — | How well the run follows the ideal arc |
| test_ratio | +0.085 | +5.9pp | — | Higher fraction of testing → success |
| step_count | -0.081 | -6.9pp | — | Shorter runs succeed more |
| edit_concentration | +0.064 | +4.8pp | YES | Editing different files (not hammering one) → success |
| has_edit_test | +0.040 | +14.9pp (N=345) | — | Testing after an edit → success |

Test position centroid is the strongest predictor — agents that place their test invocations later in the trajectory, after exploration and editing, succeed at a higher rate. This isn't about testing frequency (test_ratio captures that separately at r=+0.085). It's about sequencing: the agents that explore first, make changes, then verify, outperform those that test early or scatter tests throughout.

Uncertainty density is the second strongest individual predictor and the most legible. We count hedging markers ("maybe," "could be," "I'm not sure") per reasoning step using regex. When the same agent hedges more on the same task, it fails more. This feature was refined during the swarm process (see corrections below); the updated regex has 49% more matches and better fail/pass discrimination (1.43x ratio, up from 1.34x).

Symmetry index captures an overall trajectory shape: does the run follow an explore-then-modify-then-verify arc? Agents whose runs have this symmetrical structure succeed more. The trajectory_arc_score feature measures something related — how cleanly the run conforms to the ideal three-phase arc — and both survive with similar effect sizes.

Reasoning hypothesis formation is a surprise negative predictor. Agents that explicitly write "I think the bug is..." or "my hypothesis is..." in their reasoning fail more. The interpretation: hypothesizing is a marker of not knowing, not of productive reasoning. Agents that see the problem clearly don't need to hypothesize — they just fix it.

Edit concentration measures whether the agent spreads its edits across different files versus repeatedly editing the same file. Broader editing predicts success, consistent with agents that understand the full scope of a change versus those stuck in a single-file loop.

Has_edit_test — whether the agent tests after editing — was the headline finding in v1 and v2 of this analysis. It remains a genuine predictor (r=+0.040, p<0.001) but the effect size is much smaller than originally reported. See "The corrections" below for what happened.

## The corrections

Three bugs in our feature engineering changed specific numbers in earlier versions of this analysis. The overall story — behavioral features predict outcomes within-task — is unchanged. The headline effect shifted from has_edit_test to test_position_centroid.

**has_edit_test was inflated.** v1 and v2 reported +39pp from 28 tasks. The test detection was broken: 3,637 pytest invocations were classified as bash(explore) because commands start with `cd /workspace && pytest`, and the classifier matched the `cd` prefix before reaching the test command. After fixing test detection, the effect is r=+0.040 (was +0.116), and the natural experiment shows +14.9pp from 345 tasks (was +39pp from 28). The old number was a selection bias artifact — only 28 edge-case tasks had variation when tests went undetected. With tests properly detected, 345 tasks show meaningful variation, and the effect is real but modest.

**uncertainty_density improved.** A regex audit removed 3 false-positive patterns ("attempt," "perhaps," "not sure" when preceded by negation) and added 3 high-signal patterns ("actually," plus "let me step back/reconsider," plus "another approach"). Result: 49% more matches, fail/pass discrimination ratio improved from 1.34x to 1.43x. Correlation went from r=-0.100 to r=-0.109.

**Step enrichment vocabulary fixed.** 48.3% of steps were landing in the bash(explore) catch-all category. Root cause: `cd /workspace && X` commands matched the `cd` prefix before reaching the actual command. After stripping the cd prefix: 31 labels (was 22), Shannon entropy from 2.65 to 4.27 bits. Test steps became visible (10% of steps), search/grep split into targeted/recursive, and bash(git) separated out. This is what caused the has_edit_test inflation — with most test steps misclassified, the binary "did this run test after editing" feature only had variance on 28 tasks instead of 345.

These bugs were discovered during the feature engineering swarm (below). The swarm agents noticed that features expected to be strong (like test-after-edit) were showing weak correlations, investigated, and found the classifier bugs. The correction process is part of the methodology story.

## The feature engineering swarm

The nine features above are the survivors of a systematic search. 9 automated agents evaluated 125 candidate features across 2 rounds.

**Round 1:** Each agent proposed and implemented features in a specific domain (test behavior, reasoning quality, trajectory shape, edit patterns, etc.), computed within-task correlations and natural experiment deltas on the full 12,854-run dataset, and reported results. 68 features evaluated, 19 passed the initial bar (correlation p<0.001 AND natural experiment p<0.05).

**Round 2:** Agents reviewed Round 1 results, proposed refinements and new features informed by what worked, and re-evaluated. 57 additional features, 9 more passed. Total: 28 double-validated features from 125 candidates.

**Split-half validation:** The 28 surviving features were tested on random halves of the data. 4 survived: test_position_centroid, symmetry_index, reasoning_hypothesis_formation, and edit_concentration. The other 5 features in the table above are retained because they have strong theoretical priors and consistent natural experiment results, but they didn't pass the split-half gate.

**Bug discovery:** During Round 1, agents noticed unexpectedly weak signals from test-related features. One agent traced this to the step enrichment vocabulary bug; another found the test detection bug. The regex audit on uncertainty_density came from an agent that was trying to improve the feature and noticed the false-positive patterns. The swarm's most valuable output may have been these corrections rather than any single new feature.

Notable features that didn't make the final table but showed interesting signals:
- reasoning_self_correction (r=-0.086): "wait," "actually," "I was wrong" in reasoning predicts failure. Related to uncertainty_density but captures a different behavior — active backtracking rather than hedging.
- test_triggered_edit_success (r=+0.058, NE=+9.1%): when edits follow test failures and actually fix the problem. Strong natural experiment but narrow sample (4,531 runs had the pattern).
- test_cadence_regularity (r=-0.067): irregular testing (clustered at end) beats evenly-spaced testing. Consistent with the test_position_centroid finding.
- tool_diversity_entropy (r=+0.064): agents using more diverse tools succeed more.

## The agent's fate is decided early

Pairwise Needleman-Wunsch alignment of pass/fail run pairs finds that trajectories diverge at median position 0.15 — roughly the agent's first real strategic choice, about 10-15 steps into a 70-step trajectory.

Clustering these divergence points (TF-IDF on the divergence content, KMeans) produces 17 groups covering 98% of cases. The four largest:

- **Wasted orientation (25%)**: the first substantive step is `cd && pwd` or a version check — producing zero useful information
- **Wrong file targeted (18%)**: pass and fail runs both explore, but pick different starting files
- **Skipped reasoning (13%)**: the passing run thinks before acting; the failing run acts immediately
- **Edit-read loops (8%)**: the agent cycles between reading and editing without progress

These cluster labels are descriptive interpretations of unsupervised clusters, not validated categories. The 98% coverage means the clustering captured the data, not that 98% of failures are explained. Without a permutation baseline, some of the 25pp outcome spread between clusters could reflect sampling noise rather than genuine strategy differences.

## Strategy clustering: the agent has distinct modes

Agglomerative clustering on NW trajectory distance (average linkage, k=2 and k=3 evaluated, best outcome-separation retained) shows that 78% of tasks produce 2+ distinct behavioral clusters from the same agent, with a median 25 percentage point outcome spread between the best and worst cluster. Without a permutation null for this spread, some of it may reflect small-sample noise in per-cluster pass rates. But the magnitude — a quarter of tasks show >50pp spread between clusters — suggests genuine strategy variation.

## What the semantic layer adds

We classified 14,844 steps across 210 runs (35 tasks) using Claude Haiku, assigning each step to one of 12 outcome-blind categories. Two semantic features reach within-task significance:

- **FIX_ATTEMPT fraction** (r=-0.186, p=0.006): runs with more fix-attempt steps fail more. Partially overlaps with structural `edit_density`.
- **TEST_VERIFY fraction** (r=+0.143, p=0.038): runs with more verification steps succeed more. This is new signal — the structural layer can't distinguish test-verification from general bash activity.

Four semantic transitions reach within-task significance across the 35 tasks, though none survive Bonferroni correction for the ~70 bigrams tested. The most suggestive: `EXPLORE_BROAD → FIX_ATTEMPT` (jumping from broad search to fixing, fail-correlated, p=0.005) and `FIX_ATTEMPT → REPRODUCE` (verifying after fixing, pass-correlated, p=0.033). These are directional findings that need replication at larger scale.

A cautionary note on small samples: at n=30 runs, `EXPLORE_TARGETED` showed r=+0.346, looking like the strongest predictor in the dataset. At n=210, it shrank to r=+0.034. The features that survived scaling (FIX_ATTEMPT, TEST_VERIFY) could shrink further. The semantic layer's most robust contribution is in transition patterns, not single-feature prediction.

## HMM latent states

A K=8 Hidden Markov Model on enriched step name sequences (50-task subsample, 686 runs) was re-run on the corrected step vocabulary (31 labels, up from 22). The richer vocabulary fragments emission probabilities: on this subsample, 0 states reach within-task significance (previously 2 states were significant with the coarser vocabulary).

The earlier "thinking state" (86% reasoning, 14% read, r=+0.121) and "stuck editing state" (42% edit, 36% read source, r=-0.119) were genuine patterns in the data they were trained on, but they don't survive the vocabulary correction. The HMM either needs more data (the 50-task subsample may be too small for 31-label emissions) or a reduced vocabulary that preserves the original signal while fixing the bash(explore) catch-all. This is an open question.

## Motif discovery

564 significant gapped motifs and 121 contiguous motifs (BH-corrected, q<0.05). Effect sizes are small — the top motifs show 5-8pp lift over baseline. Examples:

- Success: `write(source) → ... → edit(test_file) → ... → finish` (58% vs 53% baseline)
- Failure: `edit(source) → ... → write(source) → ... → write(source)` (50% vs 53%)

The sheer number of significant motifs (685 total) should be interpreted cautiously. With thousands of candidate motifs tested, BH correction controls the false discovery rate at 5%, meaning ~34 of these could be false positives. The specific motif examples are illustrative, not definitive.

## What doesn't work

Exploration ratio (r=-0.012, p=0.17): how much the agent explores doesn't predict success. Causal language density (p=0.18): saying "because" and "therefore" doesn't help. Code reference density (p=0.55): mentioning specific code doesn't help. Test-fail loops (p=0.29): too rare in this agent (3% of runs). Concordance within clusters (1/67 significant): which cluster a run is in predicts outcome; how typical it is within the cluster doesn't. Structural transition bigrams: dead on same-agent data (50% directional consistency).

From the swarm, features that looked promising but failed validation: search_convergence (p=0.035, too weak), edit_test_interleave (p=0.21), monotonicity_score (p=0.46), repeated same-error counts (p=0.005, no natural experiment). Many features that capture "amount of X" don't predict outcomes — what matters is sequencing and timing, not volume.

## Limitations

**Single agent.** All findings are from OpenHands v0.54.0 + Qwen3-Coder-480B. The earlier cross-agent analysis found features that flip sign across frameworks (test-after-edit was r=-0.188 on eval_harness). These findings might be specific to this agent-model combination. Replication on other agents is needed.

**Single task domain.** SWE-rebench tasks are software engineering bug fixes in Python repositories. Generalization to other coding tasks (greenfield development, multi-language) or non-coding agent tasks is unknown.

**No held-out validation.** All features, motifs, and HMM states were discovered and evaluated on the same dataset. The split-half test on the swarm features is a partial mitigation — 4 of 28 features survived — but a fully independent replication would be stronger.

**No intervention experiment.** We measured within-task predictive associations, not the effect of actually changing the agent's behavior. The test_position_centroid finding suggests a scaffold change (enforce explore→edit→test ordering), but the actual improvement from enforcing it is unknown. The intervention experiment is designed but blocked on agent infrastructure setup.

**Exploratory analysis.** The features tested were not pre-registered. They were discovered through iterative analysis across multiple datasets over several weeks, culminating in the automated swarm search. The reported p-values are individually valid but should be interpreted in the context of an exploratory study, not a confirmatory one. The swarm evaluated 125 features; even with Bonferroni correction across all 125, the top 4 (test_position_centroid, uncertainty_density, symmetry_index, reasoning_hypothesis_formation) remain significant.

## The correction

Earlier work on cross-agent datasets led us to conclude that "no universal behavioral predictor of agent outcomes exists" and that "every abstraction layer is conditional on framework, task, and difficulty." That conclusion was correct for cross-agent data and the question it was answering: "do behavioral features predict outcomes across different agents?" They don't.

The same-agent stochastic reruns answer a different question: "within a single agent's behavioral variation, do features predict outcomes?" They do — consistently, with no Simpson's paradox and no sign flips. The earlier methodology was sound. The data couldn't answer the question we're now asking.

Within the same-agent analysis, we also corrected our own feature engineering. The has_edit_test effect shrank from +39pp to +14.9pp when we fixed test detection, and has_edit_test dropped from the strongest predictor to a modest one. Test_position_centroid — a more granular measure of the same underlying behavior (test timing) — took its place. The story didn't change (testing matters, timing matters) but the specific numbers did, and the emphasis shifted from a binary choice to a continuous property of the trajectory.

## Implications

**For agent training.** The 11,006 preference pairs extracted from divergence points are step-level DPO training signal — the agent's full reasoning and action at the exact moment where its choice determined the outcome. Labs doing RL on coding agents should mine stochastic variation from their own eval runs. The marginal cost is low if you're already running multi-trial evals for variance estimation, and the signal is cleaner than cross-task comparisons or synthetic judges.

**For agent evaluation.** Cross-agent behavioral comparisons are confounded by architecture. Published claims about "what makes agents succeed" based on multi-framework analyses should be treated skeptically unless within-agent controls are applied. Benchmarks should require multi-run submissions and report confidence intervals.

**For agent design.** The test_position_centroid finding suggests that scaffolds should enforce an explore→edit→verify sequence rather than allowing arbitrary interleaving. This is a structural constraint, not a prompt suggestion. The natural experiment shows +6.8pp from higher test_position_centroid; the actual improvement from enforcing it would likely be smaller but is worth testing. Auto-prepopulating the working directory could address the 25% of divergence points in the "wasted orientation" cluster. Uncertainty detection (simple regex on reasoning text) could trigger structured hypothesis prompts — though reasoning_hypothesis_formation being a negative predictor suggests the prompt should push toward direct action, not more hypothesizing.

**For interpretability.** Uncertainty density predicts failure from text alone (r=-0.109). Reasoning hypothesis formation and self-correction are both negative predictors, pointing to specific moments where the model's internal state might be studied — the activations when it generates hedging language in runs that fail vs succeed on the same task.

---

## Visualizations for README and blog post

**Figure 1: Within-task predictive effects.** Forest plot. Each row is one of the 9 features from the table. X-axis: within-task correlation coefficient. Point estimate with bootstrapped 95% CI. Bonferroni threshold marked (p<0.004 after correcting for 125 features tested in the swarm). Solid dots for split-half validated features, hollow for others. Highlight test_position_centroid as the strongest. Takeaway: test timing is the strongest predictor; trajectory shape and reasoning style matter.

**Figure 2: Test position centroid effect.** Scatter or binned bar chart. X-axis: test_position_centroid (0=tests at start, 1=tests at end). Y-axis: within-task pass rate delta. Show the +6.8pp natural experiment result. Include density of runs along the x-axis. Takeaway: agents that test later (after exploring and editing) succeed more.

**Figure 3: The has_edit_test correction.** Two panels. Left ("Before fix, N=28 tasks"): large 39pp gap, wide CI, dramatic. Right ("After fix, N=345 tasks"): 14.9pp gap, tight CI, real but modest. Annotation: "3,637 pytest calls were misclassified as bash(explore)." Takeaway: the effect is real but the original number was inflated by a bug.

**Figure 4: Divergence position distribution.** Histogram. X-axis: normalized position in trajectory (0=start, 1=end). Y-axis: count. Vertical line at median 0.15. Takeaway: the agent's strategy is set in the first 15% of the trajectory.

**Figure 5: Strategy clusters within one task.** Dendrogram + outcome heatmap. Pick one task with 3 clusters and large outcome spread. Left: NW-distance dendrogram. Right: horizontal bars colored pass (green) / fail (red). Label clusters with dominant strategy. Takeaway: same agent, same task, different strategies, different outcomes.

**Figure 6: The feature engineering swarm.** Funnel diagram. 125 features → 28 double-validated → 4 split-half validated (+ 5 retained with priors). Show the bug discovery as a side branch. Takeaway: systematic search with automated correction produces reliable features.

**Figure 7: Semantic transitions.** Directed graph. Nodes: 12 semantic categories. Edges: 4 significant within-task transitions, green (pass) or red (fail). Width proportional to effect size. Caveat label: "suggestive, p<0.05 uncorrected." Takeaway: the semantic layer captures intent-level patterns structural labels miss.

**Figure 8: The methodological correction.** Two panels. Left ("Cross-agent, 68K runs"): feature arrows pointing in contradictory directions, Simpson's paradox. Right ("Same-agent, 12K runs"): same features all pointing consistently. Takeaway: the techniques work when the data is right.

**Figure 9: Failure mode taxonomy.** Treemap or waffle chart. 17 divergence clusters sized by prevalence, top 4 labeled. Caveat: "unsupervised clusters, labels are descriptive interpretations." Takeaway: failure modes are concentrated — top 4 account for 64% of divergence points.
