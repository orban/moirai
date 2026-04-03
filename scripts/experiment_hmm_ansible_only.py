#!/usr/bin/env python3
"""Experiment: does the HMM "bash trap" survive within a single task family?

The whole-dataset HMM found K=4 best with tau=-0.201 and an absorbing "bash
trap" state S0 (90% self-transition, 24% fail time vs 11% pass time). But the
ablation experiment showed ALL motif findings in eval_harness were Simpson's
paradox artifacts driven by family composition (ansible passes at 72%, four
other families at 0%).

This script re-runs the HMM on ansible-only and getzep_graphiti-only to see
whether the latent-state structure reflects real behavioral differences or
just encodes task-family membership.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from hmmlearn.hmm import CategoricalHMM
from scipy.stats import kendalltau

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moirai.analyze.align import _consensus, _nw_align, align_runs
from moirai.compress import step_enriched_name
from moirai.load import load_runs
from moirai.schema import Run


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def tau_b(x: list[float], y: list[float]) -> tuple[float, float | None]:
    """Kendall's Tau-b, returning (tau, p) or (0.0, None) on degenerate input."""
    if len(x) != len(y) or len(x) < 5:
        return 0.0, None
    if len(set(x)) < 2 or len(set(y)) < 2:
        return 0.0, None
    result = kendalltau(x, y, variant="b")
    t = float(result.statistic)
    p = float(result.pvalue)
    if math.isnan(t):
        return 0.0, None
    return t, (p if not math.isnan(p) else None)


def get_enriched_sequence(run: Run) -> list[str]:
    return [n for s in run.steps if (n := step_enriched_name(s)) is not None]


def encode_sequences(
    runs: list[Run],
) -> tuple[list[np.ndarray], dict[str, int], list[str]]:
    vocab: dict[str, int] = {}
    id_to_name: list[str] = []
    sequences: list[np.ndarray] = []
    for run in runs:
        names = get_enriched_sequence(run)
        ids = []
        for name in names:
            if name not in vocab:
                vocab[name] = len(id_to_name)
                id_to_name.append(name)
            ids.append(vocab[name])
        sequences.append(np.array(ids, dtype=int))
    return sequences, vocab, id_to_name


# ---------------------------------------------------------------------------
# HMM fitting and analysis
# ---------------------------------------------------------------------------

def fit_hmm(
    runs: list[Run],
    sequences: list[np.ndarray],
    n_states: int,
    n_vocab: int,
    seed: int = 42,
) -> tuple[float, float | None, CategoricalHMM | None]:
    """Fit HMM, Viterbi decode, return concordance (tau, p, model)."""
    known = [(r, s) for r, s in zip(runs, sequences)
             if r.result.success is not None and len(s) > 0]
    if len(known) < 5:
        return 0.0, None, None

    has_mixed = (any(r.result.success for r, _ in known)
                 and any(not r.result.success for r, _ in known))
    if not has_mixed:
        return 0.0, None, None

    known_runs = [r for r, _ in known]
    known_seqs = [s for _, s in known]

    lengths = [len(s) for s in known_seqs]
    X = np.concatenate(known_seqs).reshape(-1, 1)

    model = CategoricalHMM(
        n_components=n_states,
        n_iter=100,
        random_state=seed,
        n_features=n_vocab,
    )

    try:
        model.fit(X, lengths)
    except Exception as e:
        print(f"    HMM fit failed (K={n_states}): {e}", file=sys.stderr)
        return 0.0, None, None

    # Viterbi decode -> state sequences
    state_seqs: list[list[int]] = []
    for seq in known_seqs:
        try:
            _, states = model.decode(seq.reshape(-1, 1))
            state_seqs.append(states.tolist())
        except Exception:
            state_seqs.append([0] * len(seq))

    # Align state sequences, compute distance from consensus
    state_str_seqs = [[str(s) for s in ss] for ss in state_seqs]
    if len(state_str_seqs) < 2:
        return 0.0, None, model

    # Align all against first, build padded matrix
    ref = state_str_seqs[0]
    aligned = [ref]
    for seq in state_str_seqs[1:]:
        al_a, al_b = _nw_align(ref, seq)
        aligned.append(al_b)
        if len(al_a) > len(ref):
            ref = al_a

    max_len = max(len(a) for a in aligned)
    for i in range(len(aligned)):
        while len(aligned[i]) < max_len:
            aligned[i].append("-")

    consensus = _consensus(aligned)
    n_cols = len(consensus)
    if n_cols == 0:
        return 0.0, None, model

    distances = []
    for row in aligned:
        mismatches = sum(1 for a, b in zip(row, consensus) if a != b)
        distances.append(mismatches / n_cols)

    if len(set(distances)) < 2:
        return 0.0, None, model

    outcomes = [1.0 if r.result.success else 0.0 for r in known_runs]
    t, p = tau_b([-d for d in distances], outcomes)
    return t, p, model


def describe_states(model: CategoricalHMM, id_to_name: list[str], n_states: int) -> list[str]:
    emission = model.emissionprob_
    descriptions = []
    for k in range(n_states):
        top_idx = np.argsort(emission[k])[::-1][:5]
        top = [(id_to_name[i] if i < len(id_to_name) else f"?{i}", emission[k][i])
               for i in top_idx if emission[k][i] > 0.03]
        desc = ", ".join(f"{name} ({prob:.0%})" for name, prob in top)
        self_trans = model.transmat_[k, k]
        descriptions.append(f"S{k}: self={self_trans:.0%} | {desc}")
    return descriptions


def state_time_analysis(
    model: CategoricalHMM,
    runs: list[Run],
    sequences: list[np.ndarray],
    n_states: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Compute fraction of time pass/fail runs spend in each state."""
    time_pass = np.zeros(n_states)
    time_fail = np.zeros(n_states)
    n_pass = 0
    n_fail = 0

    for run, seq in zip(runs, sequences):
        if run.result.success is None or len(seq) == 0:
            continue
        try:
            _, states = model.decode(seq.reshape(-1, 1))
        except Exception:
            continue

        if run.result.success:
            for s in states:
                time_pass[s] += 1
            n_pass += 1
        else:
            for s in states:
                time_fail[s] += 1
            n_fail += 1

    if time_pass.sum() > 0:
        time_pass /= time_pass.sum()
    if time_fail.sum() > 0:
        time_fail /= time_fail.sum()

    return time_pass, time_fail, n_pass, n_fail


def raw_concordance(runs: list[Run]) -> tuple[float, float | None]:
    """Baseline concordance from raw NW distance to consensus."""
    known = [r for r in runs if r.result.success is not None]
    if len(known) < 5:
        return 0.0, None
    if not (any(r.result.success for r in known) and any(not r.result.success for r in known)):
        return 0.0, None

    alignment = align_runs(known, level="name")
    if not alignment.matrix or not alignment.matrix[0]:
        return 0.0, None

    consensus = _consensus(alignment.matrix)
    n_cols = len(consensus)
    if n_cols == 0:
        return 0.0, None

    distances = []
    for row in alignment.matrix:
        mismatches = sum(1 for a, b in zip(row, consensus) if a != b)
        distances.append(mismatches / n_cols)

    if len(set(distances)) < 2:
        return 0.0, None

    outcomes = [1.0 if r.result.success else 0.0 for r in known]
    return tau_b([-d for d in distances], outcomes)


# ---------------------------------------------------------------------------
# per-family analysis
# ---------------------------------------------------------------------------

def analyze_family(label: str, runs: list[Run]):
    """Full HMM analysis on a single subset of runs."""
    known = [r for r in runs if r.result.success is not None]
    n_pass = sum(1 for r in known if r.result.success)
    n_fail = len(known) - n_pass
    pass_rate = n_pass / len(known) if known else 0

    print(f"\n{'='*70}")
    print(f"{label}: {len(runs)} runs, {n_pass}P/{n_fail}F ({pass_rate:.0%} pass)")
    print(f"{'='*70}")

    if not (n_pass > 0 and n_fail > 0):
        print("  SKIP: no outcome variation (all pass or all fail)")
        return

    # encode
    sequences, vocab, id_to_name = encode_sequences(runs)
    n_vocab = len(vocab)
    seq_lens = [len(s) for s in sequences if len(s) > 0]
    print(f"Vocabulary: {n_vocab} unique enriched step types")
    print(f"Sequence lengths: mean={np.mean(seq_lens):.0f}, "
          f"range={np.min(seq_lens)}-{np.max(seq_lens)}")

    # baseline
    raw_tau, raw_p = raw_concordance(runs)
    p_str = f", p={raw_p:.3f}" if raw_p is not None else ""
    print(f"\nRaw structural concordance: tau={raw_tau:.3f}{p_str}")

    # HMM sweep
    print(f"\nHMM concordance by K:")
    best_k = 0
    best_tau = 0.0
    best_p = None
    best_model = None
    results = []

    for k in [2, 3, 4, 5]:
        t, p, model = fit_hmm(runs, sequences, k, n_vocab)
        p_str = f", p={p:.3f}" if p is not None else ""
        delta = t - raw_tau
        marker = " <<<" if abs(t) > abs(raw_tau) and abs(t) > 0.05 else ""
        print(f"  K={k}: tau={t:.3f}{p_str}  (delta={delta:+.3f}){marker}")
        results.append((k, t, p, model))
        if abs(t) > abs(best_tau):
            best_tau = t
            best_p = p
            best_k = k
            best_model = model

    # describe best model
    if best_model is not None and best_k > 0:
        print(f"\nBest HMM: K={best_k} (tau={best_tau:.3f} vs raw tau={raw_tau:.3f})")
        print("Hidden states:")
        for desc in describe_states(best_model, id_to_name, best_k):
            print(f"  {desc}")

        # transition matrix
        trans = best_model.transmat_
        print(f"\nTransition matrix:")
        header = "     " + "  ".join(f"  S{j}" for j in range(best_k))
        print(f"  {header}")
        for i in range(best_k):
            row = "  ".join(f"{trans[i,j]:.2f}" for j in range(best_k))
            print(f"  S{i}   {row}")

        # state-time pass vs fail
        time_p, time_f, np_, nf_ = state_time_analysis(
            best_model, runs, sequences, best_k)
        if np_ > 0 and nf_ > 0:
            print(f"\nTime in each state (pass n={np_}, fail n={nf_}):")
            print(f"  {'State':<8} {'Pass':>8} {'Fail':>8} {'Delta':>8}")
            for s in range(best_k):
                d = time_p[s] - time_f[s]
                flag = " *" if abs(d) > 0.05 else ""
                print(f"  S{s:<7} {time_p[s]:>7.1%} {time_f[s]:>7.1%} {d:>+7.1%}{flag}")

            # identify if there's a "bash trap" analog: high self-transition,
            # disproportionate fail time
            print(f"\n  Bash-trap candidates (self-trans > 0.7 AND fail_time - pass_time > 0.05):")
            found_trap = False
            for s in range(best_k):
                self_t = trans[s, s]
                delta_time = time_f[s] - time_p[s]
                if self_t > 0.7 and delta_time > 0.05:
                    # find top emission for this state
                    top_em = np.argsort(best_model.emissionprob_[s])[::-1][:3]
                    top_names = [id_to_name[i] for i in top_em if i < len(id_to_name)]
                    print(f"    S{s}: self={self_t:.0%}, fail_excess={delta_time:+.1%}, "
                          f"top_emissions={top_names}")
                    found_trap = True
            if not found_trap:
                print("    (none found)")
    else:
        print(f"\nNo HMM improved over raw concordance.")

    return {
        "label": label,
        "n_runs": len(runs),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "pass_rate": pass_rate,
        "raw_tau": raw_tau,
        "raw_p": raw_p,
        "best_k": best_k,
        "best_tau": best_tau,
        "best_p": best_p,
        "n_vocab": n_vocab,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    examples_dir = Path("/Users/ryo/dev/moirai/examples/eval_harness")
    if not examples_dir.exists():
        print(f"ERROR: {examples_dir} not found")
        sys.exit(1)

    runs, warnings = load_runs(examples_dir)
    if warnings:
        print(f"({len(warnings)} load warnings suppressed)")

    known = [r for r in runs if r.result.success is not None]
    print(f"Loaded {len(runs)} runs total, {len(known)} with known outcomes")

    # show family breakdown
    from collections import Counter
    families = Counter(r.task_family for r in runs)
    print(f"\nTask family breakdown:")
    for fam, count in families.most_common():
        fam_runs = [r for r in runs if r.task_family == fam]
        n_pass = sum(1 for r in fam_runs if r.result.success)
        n_known = sum(1 for r in fam_runs if r.result.success is not None)
        rate = n_pass / n_known if n_known else 0
        print(f"  {fam}: {count} runs, {n_pass}P/{n_known - n_pass}F ({rate:.0%})")

    # --------------- whole dataset (reference) ---------------
    whole_result = analyze_family("WHOLE DATASET (reference)", runs)

    # --------------- ansible only ---------------
    ansible_runs = [r for r in runs if r.task_family and "ansible" in r.task_family.lower()]
    ansible_result = analyze_family("ANSIBLE ONLY", ansible_runs)

    # --------------- getzep_graphiti only ---------------
    getzep_runs = [r for r in runs if r.task_family and "graphiti" in r.task_family.lower()]
    getzep_result = analyze_family("GETZEP_GRAPHITI ONLY", getzep_runs)

    # --------------- comparison summary ---------------
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print()
    print("Whole-dataset HMM (prior finding):")
    print("  K=4 best, tau=-0.201, bash trap S0 (self=90%, fail 24% vs pass 11%)")
    print()

    for r in [whole_result, ansible_result, getzep_result]:
        if r is None:
            continue
        print(f"{r['label']}:")
        print(f"  runs={r['n_runs']} ({r['n_pass']}P/{r['n_fail']}F, {r['pass_rate']:.0%})")
        raw_p_str = f" (p={r['raw_p']:.3f})" if r['raw_p'] is not None else ""
        print(f"  vocab={r['n_vocab']}, raw_tau={r['raw_tau']:.3f}{raw_p_str}")
        best_p_str = f" (p={r['best_p']:.3f})" if r['best_p'] is not None else ""
        print(f"  best K={r['best_k']}, hmm_tau={r['best_tau']:.3f}{best_p_str}")
        delta = r['best_tau'] - r['raw_tau']
        print(f"  HMM improvement over raw: {delta:+.3f}")
        sig = r['best_p'] is not None and r['best_p'] < 0.05
        print(f"  statistically significant: {'YES' if sig else 'NO'}")
        print()

    # verdict
    def _is_significant(result):
        if result is None:
            return False
        return (abs(result['best_tau']) >= 0.05
                and result['best_p'] is not None
                and result['best_p'] < 0.05)

    print("VERDICT:")
    print()
    a_sig = _is_significant(ansible_result)
    g_sig = _is_significant(getzep_result)

    if ansible_result and not a_sig:
        a_tau = ansible_result['best_tau']
        a_p = ansible_result.get('best_p')
        a_p_str = f", p={a_p:.3f}" if a_p is not None else ""
        print(f"  ANSIBLE: tau={a_tau:.3f}{a_p_str} -- NOT significant.")
        print("  The whole-dataset 'bash trap' (tau=-0.201, p<0.001) collapses to noise")
        print("  when you control for task family. This is a Simpson's paradox artifact,")
        print("  consistent with the ablation experiment's findings on motifs.")
    elif ansible_result and a_sig:
        a_tau = ansible_result['best_tau']
        a_p = ansible_result['best_p']
        print(f"  ANSIBLE: tau={a_tau:.3f}, p={a_p:.3f} -- signal survives within-family!")
        print("  The HMM captures real behavioral differences, not just family membership.")
    else:
        print("  Insufficient data for ansible analysis.")

    print()
    if getzep_result and g_sig:
        g_tau = getzep_result['best_tau']
        g_p = getzep_result['best_p']
        print(f"  GETZEP: tau={g_tau:.3f}, p={g_p:.3f} -- signal present in second family.")
        print("  HMM latent states carry within-family predictive power here.")
    elif getzep_result:
        g_tau = getzep_result.get('best_tau', 0)
        g_p = getzep_result.get('best_p')
        g_p_str = f", p={g_p:.3f}" if g_p is not None else ""
        print(f"  GETZEP: tau={g_tau:.3f}{g_p_str} -- "
              f"{'significant' if g_sig else 'not significant'}.")

    print()
    if not a_sig and g_sig:
        print("  Mixed result: HMM has no power within ansible (the largest family)")
        print("  but does within getzep. The whole-dataset finding was confounded.")
    elif not a_sig and not g_sig:
        print("  The HMM finding is entirely a family-composition artifact.")
    elif a_sig and g_sig:
        print("  HMM latent states carry real within-family signal in both families.")


if __name__ == "__main__":
    main()
