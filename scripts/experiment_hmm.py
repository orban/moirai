#!/usr/bin/env python3
"""Experiment: do HMM latent states predict outcomes better than raw structure?

Fits CategoricalHMMs with K=2..6 hidden states on enriched step sequences,
Viterbi-decodes each run, then compares concordance (Kendall's Tau-b) between:
  - Raw NW distance from consensus vs outcomes (what moirai already does)
  - HMM state-sequence distance vs outcomes (what HMMs add)

If HMM concordance > raw concordance, the latent representation captures
something the surface tool-call sequence doesn't.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from hmmlearn import hmm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moirai.analyze.align import _consensus, _nw_align, align_runs
from moirai.compress import step_enriched_name
from moirai.load import load_runs
from moirai.schema import Run


def kendall_tau_b(x: list[float], y: list[float]) -> tuple[float, float | None]:
    """Kendall's Tau-b rank correlation."""
    import math
    if len(x) != len(y) or len(x) < 5:
        return 0.0, None
    if len(set(x)) < 2 or len(set(y)) < 2:
        return 0.0, None
    from scipy.stats import kendalltau
    result = kendalltau(x, y, variant="b")
    tau_val = float(result.statistic)
    p_raw = float(result.pvalue)
    if math.isnan(tau_val):
        return 0.0, None
    p_val = p_raw if not math.isnan(p_raw) else None
    return tau_val, p_val


def get_enriched_sequence(run: Run) -> list[str]:
    """Get enriched step names, filtering noise."""
    return [n for s in run.steps if (n := step_enriched_name(s)) is not None]


def encode_sequences(runs: list[Run]) -> tuple[list[np.ndarray], dict[str, int], list[str]]:
    """Encode enriched step names as integer sequences.

    Returns (encoded_sequences, name_to_id, id_to_name).
    """
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


def raw_concordance(runs: list[Run]) -> tuple[float, float | None]:
    """Baseline: concordance from raw NW distance to consensus."""
    known = [r for r in runs if r.result.success is not None]
    if len(known) < 5:
        return 0.0, None

    has_mixed = (any(r.result.success for r in known) and
                 any(not r.result.success for r in known))
    if not has_mixed:
        return 0.0, None

    alignment = align_runs(known, level="name")
    if not alignment.matrix or not alignment.matrix[0]:
        return 0.0, None

    consensus = _consensus(alignment.matrix)
    n_cols = len(consensus)

    distances = []
    for row in alignment.matrix:
        mismatches = sum(1 for a, b in zip(row, consensus) if a != b)
        distances.append(mismatches / n_cols if n_cols > 0 else 0.0)

    if len(set(distances)) < 2:
        return 0.0, None

    outcomes = [1.0 if r.result.success else 0.0 for r in known]
    return kendall_tau_b([-d for d in distances], outcomes)


def hmm_concordance(
    runs: list[Run],
    sequences: list[np.ndarray],
    n_states: int,
    n_vocab: int,
    seed: int = 42,
) -> tuple[float, float | None, object]:
    """Concordance from HMM state-sequence distances."""
    known_indices = [i for i, r in enumerate(runs) if r.result.success is not None]
    known_runs = [runs[i] for i in known_indices]
    known_seqs = [sequences[i] for i in known_indices]

    if len(known_runs) < 5:
        return 0.0, None, None

    has_mixed = (any(r.result.success for r in known_runs) and
                 any(not r.result.success for r in known_runs))
    if not has_mixed:
        return 0.0, None, None

    # Filter empty sequences
    valid = [(r, s) for r, s in zip(known_runs, known_seqs) if len(s) > 0]
    if len(valid) < 5:
        return 0.0, None, None
    known_runs, known_seqs = zip(*valid)
    known_runs = list(known_runs)
    known_seqs = list(known_seqs)

    # Fit HMM
    lengths = [len(s) for s in known_seqs]
    X = np.concatenate(known_seqs).reshape(-1, 1)

    model = hmm.CategoricalHMM(
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

    # Viterbi decode each run
    state_seqs: list[list[int]] = []
    for seq in known_seqs:
        try:
            _, states = model.decode(seq.reshape(-1, 1))
            state_seqs.append(states.tolist())
        except Exception:
            state_seqs.append([0] * len(seq))

    # Compute pairwise distances between state sequences using NW
    # Then distance from "consensus state sequence"
    state_str_seqs = [[str(s) for s in ss] for ss in state_seqs]

    # Build consensus from pairwise alignment
    if len(state_str_seqs) < 2:
        return 0.0, None, model

    # Simple approach: align all against the first, build consensus
    aligned = []
    ref = state_str_seqs[0]
    aligned.append(ref)
    for seq in state_str_seqs[1:]:
        al_a, al_b = _nw_align(ref, seq)
        aligned.append(al_b)
        if len(al_a) > len(ref):
            ref = al_a  # expand reference

    # Pad to same length
    max_len = max(len(a) for a in aligned)
    for i in range(len(aligned)):
        while len(aligned[i]) < max_len:
            aligned[i].append("-")

    consensus = _consensus(aligned)
    n_cols = len(consensus)

    distances = []
    for row in aligned:
        mismatches = sum(1 for a, b in zip(row, consensus) if a != b)
        distances.append(mismatches / n_cols if n_cols > 0 else 0.0)

    if len(set(distances)) < 2:
        return 0.0, None, model

    outcomes = [1.0 if r.result.success else 0.0 for r in known_runs]
    tau, p = kendall_tau_b([-d for d in distances], outcomes)
    return tau, p, model


def describe_states(model, id_to_name: list[str], n_states: int) -> list[str]:
    """Describe each hidden state by its top emissions."""
    descriptions = []
    emission = model.emissionprob_
    for k in range(n_states):
        top_indices = np.argsort(emission[k])[::-1][:3]
        top = [(id_to_name[i] if i < len(id_to_name) else f"?{i}", emission[k][i])
               for i in top_indices if emission[k][i] > 0.05]
        desc = ", ".join(f"{name} ({prob:.0%})" for name, prob in top)
        descriptions.append(f"S{k}: {desc}")
    return descriptions


def main():
    # Examples are gitignored, so use absolute path to main repo
    examples_dir = Path("/Users/ryo/dev/moirai/examples")

    datasets = [
        ("eval_harness", str(examples_dir / "eval_harness")),
        ("swe_smith", str(examples_dir / "swe_smith")),
        ("swe_agent", str(examples_dir / "swe_agent")),
    ]

    for name, path in datasets:
        data_path = Path(path)
        if not data_path.exists():
            print(f"Skipping {name}: {path} not found")
            continue

        runs, _ = load_runs(data_path)
        known = [r for r in runs if r.result.success is not None]
        n_pass = sum(1 for r in known if r.result.success)
        pass_rate = n_pass / len(known) if known else 0

        print(f"\n{'='*70}")
        print(f"{name}: {len(runs)} runs, {n_pass}P/{len(known)-n_pass}F ({pass_rate:.0%} pass)")
        print(f"{'='*70}")

        # Encode
        sequences, vocab, id_to_name = encode_sequences(runs)
        n_vocab = len(vocab)
        print(f"Vocabulary: {n_vocab} unique step types")
        print(f"Sequence lengths: {np.mean([len(s) for s in sequences]):.0f} avg, "
              f"{np.min([len(s) for s in sequences if len(s) > 0])}-{np.max([len(s) for s in sequences])} range")

        # Baseline concordance
        raw_tau, raw_p = raw_concordance(runs)
        p_str = f", p={raw_p:.3f}" if raw_p is not None else ""
        print(f"\nRaw structural concordance: τ={raw_tau:.3f}{p_str}")

        # HMM concordance for various K
        print(f"\nHMM concordance by number of hidden states:")
        best_k = 0
        best_tau = raw_tau
        best_model = None

        for k in [2, 3, 4, 5, 6]:
            tau, p, model = hmm_concordance(runs, sequences, k, n_vocab)
            p_str = f", p={p:.3f}" if p is not None else ""
            delta = tau - raw_tau
            marker = " <<<" if abs(tau) > abs(raw_tau) and abs(tau) > 0.05 else ""
            print(f"  K={k}: τ={tau:.3f}{p_str}  (delta from raw: {delta:+.3f}){marker}")

            if abs(tau) > abs(best_tau):
                best_tau = tau
                best_k = k
                best_model = model

        # Describe best model's states
        if best_model is not None and best_k > 0:
            print(f"\nBest HMM: K={best_k} (τ={best_tau:.3f} vs raw τ={raw_tau:.3f})")
            print(f"Hidden state descriptions:")
            for desc in describe_states(best_model, id_to_name, best_k):
                print(f"  {desc}")

            # Show transition matrix
            trans = best_model.transmat_
            print(f"\nTransition matrix:")
            header = "     " + "  ".join(f"S{j}" for j in range(best_k))
            print(f"  {header}")
            for i in range(best_k):
                row = "  ".join(f"{trans[i,j]:.2f}" for j in range(best_k))
                print(f"  S{i}  {row}")

            # State-outcome correlation: which states do passing vs failing runs spend time in?
            known_indices = [i for i, r in enumerate(runs) if r.result.success is not None]
            state_time_pass = np.zeros(best_k)
            state_time_fail = np.zeros(best_k)
            n_pass_decoded = 0
            n_fail_decoded = 0

            for idx in known_indices:
                seq = sequences[idx]
                if len(seq) == 0:
                    continue
                try:
                    _, states = best_model.decode(seq.reshape(-1, 1))
                    if runs[idx].result.success:
                        for s in states:
                            state_time_pass[s] += 1
                        n_pass_decoded += 1
                    else:
                        for s in states:
                            state_time_fail[s] += 1
                        n_fail_decoded += 1
                except Exception:
                    pass

            if n_pass_decoded > 0 and n_fail_decoded > 0:
                # Normalize
                state_time_pass /= state_time_pass.sum()
                state_time_fail /= state_time_fail.sum()

                print(f"\nTime spent in each state (normalized):")
                print(f"  {'State':<8} {'Pass':>8} {'Fail':>8} {'Delta':>8}")
                for s in range(best_k):
                    delta = state_time_pass[s] - state_time_fail[s]
                    marker = " *" if abs(delta) > 0.05 else ""
                    print(f"  S{s:<7} {state_time_pass[s]:>7.1%} {state_time_fail[s]:>7.1%} {delta:>+7.1%}{marker}")

        print()


if __name__ == "__main__":
    main()
