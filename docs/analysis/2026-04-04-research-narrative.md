# Same agent, same task, different outcome: what stochastic variation reveals about coding agents

The strongest predictor of whether an AI coding agent solves a task is a single behavioral choice: does it run the test suite after editing code? Runs that verify their edits pass at 59%. Runs that don't pass at 19%. That's a 39 percentage point gap, and it's causal.

We know it's causal because the data comes from a natural experiment. The Nebius SWE-rebench dataset contains 12,854 runs of one agent (OpenHands v0.54.0 with Qwen3-Coder-480B) on 1,096 software engineering tasks, each task attempted 10-20 times. Same model, same scaffold, same config. The only variable is the LLM's stochastic choices. When we compare runs within the same task, there are no possible confounds: no task difficulty differences, no cross-agent architecture effects, no framework artifacts. The variation is the experiment.

## We were wrong about universal predictors

Earlier work on cross-agent datasets led us to conclude that no universal behavioral predictors exist. The evidence seemed clear: `test_after_edit` was negatively correlated with success in one framework and positively correlated in another. Exploration ratio flipped sign across datasets. We wrote "most features are context-dependent" and recommended per-dataset feature selection.

That conclusion was wrong. It was an artifact of testing across 133 different agent architectures in the SWE-bench verified corpus, where architectural differences between agents overwhelmed behavioral signals. When one agent's scaffold forces test-after-edit and that agent happens to be weaker, you get a negative correlation between testing and success that has nothing to do with whether testing helps.

On same-agent stochastic reruns, the picture is simple: 11 features predict outcomes within-task at p<0.05. Four are causally validated at p<0.001. Zero show Simpson's paradox. Zero sign flips between pooled and within-task analysis. The signals we thought were contradictory were consistent all along. We just couldn't see them through the cross-agent noise.

## Four causal effects from natural experiments

For each behavioral feature, we split runs within each task by whether the feature value falls above or below the task median. Because variation is stochastic, the within-task comparison is a natural experiment.

**Test-after-edit (+38.9pp, p<0.0001):** The largest effect by far. Runs that test their edits pass at 59% vs 19% for runs that don't. This is a single binary choice the agent makes, and it nearly triples the success rate.

**Uncertainty density (-8.3pp, p<0.0001):** Runs where the agent uses more hedging language in its reasoning ("maybe," "I think," "could be") fail more. The agent's own confidence, expressed in its chain-of-thought, predicts whether the fix is right. This feature was previously dismissed as "confounded with task difficulty." It isn't.

**Step count (-6.9pp, p<0.0001):** Shorter runs succeed more. This isn't just "hard tasks take longer" -- within the same task, the run that finishes in fewer steps is more likely to pass. Each extra step is evidence the agent is stuck, not making progress.

**Test ratio (+5.9pp, p<0.0001):** Runs that spend more of their trajectory testing (as a fraction of total steps) succeed more. The denominator matters here: it's not just "run more tests" but "spend a higher fraction of your effort on verification rather than editing."

Two more features reach the causal threshold from the semantic layer. Runs with more FIX_ATTEMPT steps (repeated fix cycles) fail at 22pp higher rates. Runs with more TEST_VERIFY steps succeed more. The structural and semantic layers converge on the same story: verification works, retry loops don't.

## The agent's fate is decided in the first 15% of the trajectory

Pairwise alignment of pass/fail run pairs within the same task shows that runs diverge at median position 0.15 -- roughly the agent's first real strategic choice. Clustering these divergence points with TF-IDF produces 17 failure modes covering 98% of cases.

The top four: wasted orientation (25% -- the first step is a `pwd` or version check), wrong file targeted (18% -- the agent reads the wrong file first), skipped reasoning (13% -- acting before thinking), and edit-read loops (8% -- cycling without progress). Three of the four are decisions made in the first few steps.

This early-divergence finding pairs with the strategy clustering result: 78% of tasks produce two or more distinct behavioral clusters, with a median 25 percentage point gap in outcomes between the best and worst cluster. The agent has genuine strategy variation, and the strategy matters far more than the execution details.

## What the semantic layer adds

We classified 14,844 steps across 210 runs using Claude Haiku, assigning each step to one of 12 outcome-blind categories (ORIENT, EXPLORE_BROAD, FIX_ATTEMPT, TEST_VERIFY, etc.). Two findings:

First, semantic transitions carry signal that structural transitions can't. On same-agent data, structural step-type transitions (edit->test, read->edit) produce no significant within-task signal. But semantic transitions do: `EXPLORE_BROAD -> FIX_ATTEMPT` (jumping from broad search to fixing without narrowing) correlates with failure at p=0.005. `FIX_ATTEMPT -> REPRODUCE` (verifying after fixing) correlates with success at p=0.033. The structural layer can't distinguish productive from unproductive instances of the same step type. The semantic layer can.

Second, small samples lie. At n=30, `EXPLORE_TARGETED` showed r=+0.346 and looked like the strongest predictor in the dataset. At n=210, it was r=+0.034 -- noise. `FLAIL` went from r=-0.203 to r=-0.053. The features that survived scaling (FIX_ATTEMPT fraction, TEST_VERIFY fraction) are the ones that overlap most with the structural layer. The semantic layer's unique contribution is in transitions and motifs, not single-feature prediction.

## What doesn't work

Honesty requires listing the dead ends. Exploration ratio (r=-0.012, p=0.17): exploring more doesn't help. Causal language density (r=-0.012, p=0.18): saying "because" and "therefore" doesn't predict success. Code reference density (r=-0.005, p=0.55): mentioning specific code doesn't predict success. Test-fail loops (r=+0.009, p=0.29): too rare at 3% of runs. Concordance within clusters (1/67 significant): knowing which strategy cluster a run belongs to predicts outcome, but how typical the run is within that cluster doesn't.

Structural transition bigrams, which showed strong signal on cross-agent data, produce 50% directional consistency on same-agent data -- coin-flip territory. The cross-agent signal was entirely a framework artifact.

## What this means

The practical implication is that agent behavior is more predictable and more fixable than the field's current trajectory suggests. The test-after-edit finding alone -- a 39pp causal effect from a single behavioral choice -- means that relatively simple interventions (force verification after edits, detect and break retry loops, penalize hedging language in chain-of-thought) could produce large improvements. These aren't speculative: the causal estimates come from thousands of within-task natural experiments.

The methodological implication is that the field needs same-agent stochastic rerun datasets. Cross-agent comparisons, which make up the bulk of current benchmark analysis, are confounded by architecture differences that mask behavioral signals. The Nebius SWE-rebench dataset (and datasets like it) are where real behavioral science on agents becomes possible. Testing across architectures answers "which agent is better." Testing within an agent's stochastic variation answers "what makes agents succeed" -- a harder and more useful question.

The meta-finding is about our own earlier work. We ran seven experiments across four datasets and concluded that most behavioral features were context-dependent, that HMMs learned confounds, and that no universal predictors existed. All of that was correct for cross-agent data and wrong for same-agent data. The techniques were sound. The data was confounded.

---

## Visualizations for README and blog post

**Figure 1: The test-after-edit effect (paired bar chart)**
X-axis: two groups per task (runs with test-after-edit, runs without). Y-axis: pass rate. Show 10-15 individual tasks as thin grey paired bars, with the aggregate (59% vs 19%) as a thick colored bar. Takeaway: the gap is consistent across tasks, not driven by outliers.

**Figure 2: Within-task causal effects (forest plot)**
Each row is one feature. X-axis: within-task delta in percentage points. Point estimate with 95% CI. Ordered by effect size. Color: green for positive (helps), red for negative (hurts). Show the four features at p<0.001 with solid dots, the rest with hollow dots. Takeaway: test-after-edit dominates; four features are causally validated.

**Figure 3: Early divergence (histogram)**
X-axis: divergence position (0.0 = start, 1.0 = end of trajectory). Y-axis: count of task pairs. Vertical line at median 0.15. Takeaway: the agent's fate is decided in the first 15% of the trajectory.

**Figure 4: Strategy clusters within a single task (dendrogram + outcome heatmap)**
Pick one task with 3+ clusters and a large outcome spread. Left: dendrogram of NW-distance-based clustering. Right: horizontal bars colored by pass/fail. Annotate the clusters with their dominant strategy (e.g., "tests early" vs "edits repeatedly"). Takeaway: same task, same agent, genuinely different strategies, different outcomes.

**Figure 5: Semantic transition patterns (directed graph)**
Nodes are the 12 semantic categories. Edges are the 4 significant within-task transitions, colored green (pass-correlated) or red (fail-correlated). Edge width proportional to effect size. Highlight the key story: `FIX_ATTEMPT -> REPRODUCE` (good) vs `EXPLORE_BROAD -> FIX_ATTEMPT` (bad). Takeaway: the semantic layer captures intent-level patterns that structural labels miss.

**Figure 6: The methodological correction (before/after comparison)**
Two panels. Left panel ("Cross-agent"): feature correlation plot where test_after_edit shows mixed/negative effects, arrows pointing in different directions across datasets. Right panel ("Same-agent"): same features all pointing consistently, with confidence intervals that don't cross zero. Takeaway: the techniques work when the data is right.

**Figure 7: Failure mode taxonomy (treemap or waffle chart)**
17 divergence clusters, sized by prevalence. Top 4 labeled with names and percentages (wasted orientation 25%, wrong file 18%, skipped reasoning 13%, edit-read loop 8%). Takeaway: failure modes are classifiable and concentrated -- the top 4 account for 64% of divergence points.
