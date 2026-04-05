# Technique audit: what works, what failed, what to keep

Date: 2026-04-04 (final, post-swarm, post-correction)
Datasets: eval_harness (483), swe_smith (1000), swe_agent (500), swebench_verified (68,902 cross-agent), swe_rebench (12,854 same-agent, reconverted with fixed enrichment)

## Critical methodological lessons

1. **Cross-agent evaluation is fundamentally confounded.** Every behavioral feature flips sign when pooled across agent architectures. Same-agent stochastic reruns are required for valid behavioral analysis.

2. **Verify your feature extraction before trusting your features.** Our initial test detection had zero true positives (3,637 pytest invocations classified as bash(explore) due to `cd &&` prefix). 48% of steps were in a single catch-all bucket. The flagship +39pp result was a selection bias artifact from broken detection. After fixing: +14.9pp from 345 tasks.

3. **Small samples inflate effects.** Haiku semantic features went from r=+0.346 (n=30) to r=+0.034 (n=210). Always validate at scale.

## The 4 robust features (survive split-half validation)

These were discovered by a 9-agent feature engineering swarm (125 features evaluated, 28 double-validated, 4 survive split-half on independent task halves):

| Feature | Within-task r | NE delta | NE p | Split-half | What it measures |
|---|---|---|---|---|---|
| test_position_centroid | +0.121 | +6.8% | <0.001 | YES | Agents that test later in trajectory succeed more |
| symmetry_index | +0.101 | +6.1% | <0.001 | YES | Explore→modify→verify arc predicts success |
| reasoning_hypothesis_formation | -0.088 | -5.7% | <0.001 | YES | Explicit hypothesizing predicts failure |
| edit_concentration | +0.064 | +4.8% | <0.001 | YES | Editing different files (not hammering one) predicts success |

## 16 non-redundant features (survive redundancy analysis)

After removing features with pairwise |r| > 0.5 (5 redundancy clusters identified), 16 independent features remain. All robust on split-half. Top tier shown above; second tier includes:

- uncertainty_density (r=-0.109): hedging language in reasoning predicts failure (improved regex)
- trajectory_arc_score (r=+0.088): how well agent follows explore→modify→verify
- reasoning_self_correction (r=-0.086): "wait", "actually", "I was wrong" predict failure
- reasoning_confidence_ratio (r=+0.067): confident language predicts success (fails split-half — use with caution)
- test_cadence_regularity (r=-0.067): irregular testing (clustered at end) beats regular testing
- edit_concentration (r=+0.064): editing different files, not hammering one
- tool_diversity_entropy (r=+0.064): agents using more diverse tools succeed more

## Structural/analytical techniques

| Technique | Status | Evidence |
|---|---|---|
| NW alignment | **Core** | Works across all datasets |
| Pairwise divergence | **Core** | Median divergence at position 0.15 |
| Divergence clustering (TF-IDF) | **Core** | 13 clusters, 98% coverage on reconverted data |
| Strategy clustering | **Validated** | 75% of tasks form 2+ clusters, 40pp median spread (reconverted) |
| Gapped motifs | **Validated** | 1,960 significant on reconverted same-agent data |
| Contiguous motifs | **Validated** | 83 significant on reconverted data |
| Stratified motif mining | **Validated** | Essential for cross-agent data |
| Cohort comparison | **Works** | Clean A/B version comparison |
| Step enrichment | **Fixed** | 31 labels (was 22), Shannon entropy 2.65→4.27 bits |
| HTML branch visualization | **Works** | Dendrogram + heatmap + divergence badges |

## Haiku semantic classification (210 runs, 35 tasks)

Two features reach significance:
- sem_fix_attempt_frac (r=-0.186, p=0.006): more fix attempts = failure
- sem_test_verify_frac (r=+0.143, p=0.038): more verification = success

Four semantic transitions significant (uncorrected): EXPLORE_BROAD→FIX_ATTEMPT (fail, p=0.005), ORIENT→TEST_EXPLORE (pass, p=0.012), FIX_ATTEMPT→TEST_EXPLORE (fail, p=0.016), FIX_ATTEMPT→REPRODUCE (pass, p=0.033). None survive Bonferroni.

Two features that looked strong at n=30 collapsed at n=210: sem_explore_targeted (r=+0.346→+0.034), sem_flail (r=-0.203→-0.053).

## What's dead

| Technique | Evidence | Why |
|---|---|---|
| exploration_ratio | r=-0.012, p=0.17 | Not predictive within-task |
| causal_density | r=-0.012, p=0.18 | Using causal language doesn't predict |
| code_ref_density | r=-0.005, p=0.55 | Referencing code doesn't predict |
| test_fail_count | +0.002, p=0.86 | Not significant |
| has_test_fail_loop | +0.016, p=0.064 | Borderline, 13% of runs |
| Concordance (within-cluster) | 1/67 significant | Cluster identity predicts, typicality doesn't |
| NW distance to consensus | r=+0.087, p=0.09 | Not significant |
| Structural transition bigrams | 50% consistency | Dead on same-agent data |
| HMM latent states | 0 significant (K=8, reconverted) | Vocabulary too rich for sample size |

Note: HMM was significant (2 states) on the pre-fix vocabulary (22 labels). Lost significance on the corrected 31-label vocabulary because emission probabilities fragment. Needs either more data or reduced vocabulary.

## Corrections from this session

| What | Before | After | Cause |
|---|---|---|---|
| has_edit_test | r=+0.116, +39pp | r=+0.040, +14.9pp | Test detection broken (cd prefix) |
| uncertainty_density | r=-0.100 | r=-0.109 | Improved regex (+49% matches) |
| Step vocabulary | 22 labels, 48% catch-all | 31 labels, no catch-all | cd prefix stripping |
| Top feature | has_edit_test | test_position_centroid (r=+0.121) | Swarm discovery |
| HMM | 2 significant states | 0 significant | Richer vocabulary fragments emissions |
| Motifs | 564 gapped | 1,960 gapped | Richer vocabulary enables test/search patterns |
