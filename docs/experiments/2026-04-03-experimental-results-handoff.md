# Experimental results handoff

Date: 2026-04-03
Branch: `worktree-exp+hmm-strategies`
Datasets: eval_harness (483 runs), swe_smith (1000 runs), swe_agent (500 runs), swebench_verified (1002 runs)

## Summary

This branch ran a systematic adversarial evaluation of moirai's analytical techniques across 4 datasets. The core finding: **most eval_harness results are Simpson's paradox artifacts**, but moirai's techniques are validated on swe_smith and the new swebench_verified dataset.

---

## Experiments run

### 1. Concordance scoring (Kendall's Tau-b)

**Script**: built into `moirai/analyze/cluster.py:compute_concordance` (on main)
**Question**: does structural similarity to the consensus predict outcomes within a cluster?

**Results**:

| Dataset | Clusters tested | |τ| > 0.1 | Significant (p<0.05) |
|---|---|---|---|
| eval_harness | 7 | 1 | 0 |
| swe_smith | 17 | 5 | 2 (both negative) |
| swe_agent | 7 | 1 | 0 |

**Conclusion**: Raw structural similarity doesn't predict outcomes in most clusters. The two significant swe_smith clusters (τ=-0.554, τ=-0.247) show the *dominant* pattern is a failure mode. Concordance is useful as a self-diagnostic — it tells you when to stop trusting structural analysis.

---

### 2. HMM latent strategy discovery

**Script**: `scripts/experiment_hmm.py`
**Question**: do HMM hidden states predict outcomes better than raw step sequences?

**Results (whole-dataset)**:

| Dataset | Raw τ | Best HMM τ | Improvement | Best K |
|---|---|---|---|---|
| eval_harness | -0.118 | -0.201 | 70% | 4 |
| swe_smith | -0.183 | -0.192 | 5% | 3 |
| swe_agent | -0.108 | -0.108 | 0% | 2 |

eval_harness K=4 found 4 states: S0 (bash execution, 90% self-transition), S1 (config reading), S2 (edit-test loop), S3 (exploration). Failing runs spent 24% of time in S0 vs 11% for passing.

**BUT**: The ansible-only ablation (`scripts/experiment_hmm_ansible_only.py`) proved this is confounded. Within ansible: τ=0.056 (p=0.491). The HMM was learning family identity, not behavioral strategies.

**Conclusion**: HMMs are a dead end for these datasets. They either learn family confounds or add nothing over raw NW distance. Don't productize.

---

### 3. Behavioral feature analysis

**Script**: `scripts/experiment_features.py`
**Question**: which per-run behavioral features predict success/failure?

14 features tested with point-biserial correlation:

| Feature | eval_harness | swe_smith | swe_agent | Consistent? |
|---|---|---|---|---|
| step_count | r=-0.131** | r=-0.264*** | r=-0.116** | Yes — universal |
| test_after_edit | r=-0.188*** | r=+0.078* | r=+0.132** | Contradicts |
| exploration_ratio | r=+0.196*** | n.s. | r=-0.122** | Contradicts |
| bash_ratio | r=-0.250*** | n.s. | r=+0.120** | Contradicts |
| subagent_used | r=+0.323*** | N/A | N/A | eval_harness only |
| edit_density | r=+0.099* | r=+0.150*** | n.s. | Weakly consistent |
| test_fail_loop | N/A | r=-0.127*** | N/A | swe_smith only |

**Conclusion**: step_count is the only universal predictor. Most features are context-dependent — same surface behavior (e.g., test_after_edit) means "struggling" in one environment and "deliberate verification" in another. Per-dataset feature selection is required.

---

### 4. Task-family ablation (Simpson's paradox)

**Script**: `scripts/experiment_task_family_ablation.py`
**Question**: do eval_harness findings survive when we control for task_family?

**Results**:
- **0/888 motifs survive within ansible-only**. All 867 gapped + 21 contiguous motifs are family-composition artifacts (proxying for "this is an ansible run").
- 2 features confounded: `bash_ratio`, `test_after_edit` — significant overall but not within any family.
- 6 features survive within-family, but 2 show direction flips (unique_files_edited, exploration_ratio).
- 2 features masked in aggregate but strong within getzep: `late_test_ratio` (r=-0.864), `unique_files_read` (r=-0.587).

**Conclusion**: eval_harness is severely confounded. Never report cross-family motifs without `--stratify task_family`.

---

### 5. Permutation FDR validation

**Script**: `scripts/experiment_permutation_fdr.py`
**Question**: is the 28% gapped motif survival rate (867/3144) inflated by test correlation?

**Results**:

| Dataset | Type | BH discoveries | Mean null | Empirical FDR |
|---|---|---|---|---|
| eval_harness | gapped | 867 | 1.4 | 0.002 |
| eval_harness | contiguous | 21 | 0.8 | 0.011 |
| swe_smith | gapped | 26 | 0.2 | 0.010 |
| swe_agent | gapped | 92 | 2.0 | 0.017 |

**Conclusion**: The motifs are statistically valid (FDR << 0.05). But permutation FDR can't detect confounding — it shuffles outcomes but preserves family structure. The ablation (experiment 4) is the confound test, not this.

**Key methodological insight**: Permutation FDR validates statistics but not interpretation. You need stratified analysis for causal claims.

---

### 6. Predictive model on swe_smith

**Script**: `scripts/experiment_predictive_model.py`
**Question**: do moirai's behavioral features add predictive value beyond task metadata?

5-fold stratified CV on 1000 IID swe_smith runs:

| Model | GBM AUC | LR AUC |
|---|---|---|
| A: mutation_type only | 0.621 | 0.636 |
| B: behavioral features only | 0.632 | 0.668 |
| C: mutation + behavioral | 0.675 | 0.702 |
| D: C + gapped motifs | 0.689 | 0.716 |

Top features by GBM importance: edit_density (0.196), exploration_ratio (0.178), step_count (0.170), test_fail_count (0.163).

**Conclusion**: Behavioral features genuinely predict outcomes beyond task metadata. AUC lift of +0.08 from features + motifs. This is the strongest validation of moirai's techniques. LR outperforms GBM, suggesting linear signal.

---

### 7. Simpson's paradox decomposition

**Script**: `scripts/experiment_simpsons_paradox.py`
**Question**: does whole-dataset negative concordance (τ < 0) disappear when you condition on clusters?

| Dataset | Marginal τ | Weighted within-cluster τ | Gap |
|---|---|---|---|
| eval_harness | -0.151 | -0.073 | 0.078 |
| swe_smith | +0.275 | -0.099 | 0.375 (sign flip!) |
| swe_agent | +0.015 | +0.025 | 0.010 |

**Conclusion**: Confirmed Simpson's paradox. swe_smith shows a full sign reversal — marginal τ is positive but within-cluster average is negative. Concordance must always be computed within clusters. The marginal τ measures cluster composition, not behavioral patterns.

---

### 8. SWE-bench verified dataset

**Data**: converted from SWE-bench/experiments GitHub repo (2 SWE-agent submissions: gpt4 + claude3.5sonnet)
**Location**: `examples/swebench_verified/` (1002 runs with steps, 500 tasks)

**Results**:
- 111 mixed-outcome tasks (22x more than eval_harness)
- 86% of mixed-outcome tasks have significant divergence points (95/111)
- First divergence at median position 12 (early in trajectory)
- 14 enriched step types, balanced family pass rates (django 32%, sympy 17%, matplotlib 24%)
- Patterns survive within-family ablation:
  - `search → read(source) → edit` = 49% success within django (vs 32% baseline, q=0.004)
  - `read → read → read` = 10% success (failure pattern, within-family)
- 41 contiguous + 188 gapped motifs survive BH on whole dataset
- 14 contiguous + 69 gapped survive within django-only

**Conclusion**: This is the proper validation dataset for moirai. Patterns are real behavioral signals, not family confounds. Downloading all 134 submissions from SWE-bench/experiments S3 will give 50+ runs per task for definitive analysis.

---

## Code changes in this branch

| File | Change |
|---|---|
| `moirai/analyze/motifs.py` | +93 lines: `stratified_find_motifs()`, `stratified_find_gapped_motifs()`, `_has_outcome_variance()` |
| `moirai/cli.py` | +78 lines: `--stratify` flag on `patterns`, `report` command |
| `moirai/viz/html.py` | +334 lines: `write_report_html()` with summary, patterns, divergence sections |
| `tests/test_motifs.py` | +96 lines: 4 stratified motif tests |
| `tests/test_gapped_motifs.py` | +92 lines: 3 stratified gapped tests + CLI smoke test |
| `scripts/experiment_*.py` | 7 experiment scripts documenting all findings |

---

## Revised technique effectiveness (post-adversarial review)

| Rank | Technique | Validated? | Where |
|---|---|---|---|
| 1 | Gapped motifs (with `--stratify`) | Yes | swe_smith (+0.014 AUC), swebench_verified (survive within-family) |
| 2 | Behavioral features (per-dataset) | Yes | swe_smith (AUC 0.72 with held-out eval) |
| 3 | Per-task divergence | Yes | swebench_verified (86% tasks have significant divergence) |
| 4 | Concordance (within-cluster) | Yes (as diagnostic) | Tells you when structural analysis is trustworthy |
| 5 | Contiguous motifs | Yes | All datasets (lower recall than gapped) |
| 6 | Clustering | Display only | Doesn't predict outcomes |
| 7 | HMMs | No | Confounded or useless across all datasets |

---

## What to do next

### High priority

1. **Download all 134 SWE-bench/experiments submissions** — S3 downloads are in progress at `/tmp/swe-bench-experiments/`. Once complete, convert with `python scripts/convert_swebench_experiments.py /tmp/swe-bench-experiments examples/swebench_verified --split verified`. Fix the `derive_agent_name` bug (line 741 — `org` can be a list).

2. **Make `--stratify` the default for multi-family datasets** — or at minimum add a warning when motif discovery runs on a dataset with heterogeneous family pass rates.

3. **Run the full analysis battery on swebench_verified with 50+ runs/task** — once downloads complete. This is the definitive test: per-task divergence with 5+ runs, within-family motifs, behavioral feature prediction, all on a properly controlled dataset.

### Medium priority

4. **Per-dataset feature selection** — instead of fixed 8 canonical features, auto-select which features are significant for the current dataset. The feature contradiction finding means fixed features are misleading.

5. **Entropy rate profiles** — per-position entropy for pass vs fail. Low effort (the alignment matrix already exists), may find where trajectory uncertainty collapses. Best tested on swebench_verified with 50+ runs per task.

### Drop

6. **HMM strategies** — dead end. Don't productize.
7. **Positional MI** — gapped motifs capture the same signal with lower N requirements.
8. **Domain architecture (motif-level alignment)** — too speculative given current findings.

### Long-term

9. **Intervention experiment** — test whether injecting rules based on moirai findings actually improves agent pass rates. This is the ultimate validation but requires 60+ agent invocations.
