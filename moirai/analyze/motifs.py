"""Motif discovery — find step patterns that predict success or failure.

Extracts n-gram subsequences from runs and tests which patterns
discriminate between successful and failing runs.
"""
from __future__ import annotations

from dataclasses import dataclass

from moirai.analyze.stats import benjamini_hochberg, fishers_exact_2x2
from moirai.compress import step_enriched_name
from moirai.schema import Run


@dataclass
class Motif:
    """A recurring step pattern with outcome correlation."""
    pattern: tuple[str, ...]
    total_runs: int          # runs containing this pattern
    success_runs: int        # successful runs containing it
    fail_runs: int           # failing runs containing it
    success_rate: float      # success rate of runs with this pattern
    baseline_rate: float     # overall success rate for comparison
    lift: float              # success_rate / baseline_rate (>1 = positive, <1 = negative)
    p_value: float | None    # Fisher's exact test (raw)
    avg_position: float      # average position (0-1 normalized) where the pattern appears
    q_value: float | None = None  # BH-adjusted p-value

    @property
    def display(self) -> str:
        return " → ".join(self.pattern)


def _filtered_names(run: Run) -> list[str]:
    """Get enriched step name sequence (no noise, with attr context)."""
    return [n for s in run.steps if (n := step_enriched_name(s)) is not None]


def _extract_ngrams(names: list[str], min_n: int = 3, max_n: int = 5) -> list[tuple[tuple[str, ...], float]]:
    """Extract all n-grams with their normalized positions.

    Returns list of (ngram, position) where position is in [0, 1].
    """
    results: list[tuple[tuple[str, ...], float]] = []
    length = len(names)
    if length == 0:
        return results

    for n in range(min_n, max_n + 1):
        for i in range(length - n + 1):
            gram = tuple(names[i:i + n])
            pos = i / max(length - 1, 1)
            results.append((gram, pos))

    return results


def find_motifs(
    runs: list[Run],
    min_n: int = 3,
    max_n: int = 5,
    min_count: int = 3,
    q_threshold: float = 0.05,
) -> tuple[list[Motif], int]:
    """Find step patterns that correlate with success or failure.

    Returns (motifs, n_candidates_tested). Motifs are sorted by q-value
    ascending with effect size as tiebreaker. BH correction is applied
    internally.
    """
    if not runs:
        return [], 0

    # Baseline success rate
    known = [r for r in runs if r.result.success is not None]
    if not known:
        return [], 0
    baseline = sum(1 for r in known if r.result.success) / len(known)
    if baseline == 0.0 or baseline == 1.0:
        return [], 0  # no variation to explain

    total_success = sum(1 for r in known if r.result.success)
    total_fail = len(known) - total_success

    # For each run, extract n-grams
    run_grams: dict[str, set[tuple[str, ...]]] = {}  # run_id -> set of unique grams
    gram_positions: dict[tuple[str, ...], list[float]] = {}  # gram -> list of positions

    for run in runs:
        names = _filtered_names(run)
        grams_with_pos = _extract_ngrams(names, min_n, max_n)

        seen: set[tuple[str, ...]] = set()
        for gram, pos in grams_with_pos:
            if gram not in seen:
                seen.add(gram)
                if gram not in gram_positions:
                    gram_positions[gram] = []
                gram_positions[gram].append(pos)

        run_grams[run.run_id] = seen

    # For each gram, count success/fail
    gram_counts: dict[tuple[str, ...], tuple[int, int]] = {}  # gram -> (success, fail)
    for run in known:
        grams = run_grams.get(run.run_id, set())
        for gram in grams:
            if gram not in gram_counts:
                gram_counts[gram] = (0, 0)
            s, f = gram_counts[gram]
            if run.result.success:
                gram_counts[gram] = (s + 1, f)
            else:
                gram_counts[gram] = (s, f + 1)

    # Build candidate motifs with raw p-values (no filtering yet)
    candidates: list[Motif] = []
    for gram, (succ, fail) in gram_counts.items():
        total = succ + fail
        if total < min_count:
            continue

        rate = succ / total
        lift = rate / baseline if baseline > 0 else 1.0

        without_s = total_success - succ
        without_f = total_fail - fail
        p_val = fishers_exact_2x2(succ, fail, without_s, without_f)

        avg_pos = sum(gram_positions.get(gram, [0.0])) / max(len(gram_positions.get(gram, [0.0])), 1)

        candidates.append(Motif(
            pattern=gram,
            total_runs=total,
            success_runs=succ,
            fail_runs=fail,
            success_rate=rate,
            baseline_rate=baseline,
            lift=lift,
            p_value=p_val,
            avg_position=avg_pos,
        ))

    # Apply BH correction
    raw_p = [m.p_value for m in candidates]
    adjusted = benjamini_hochberg(raw_p, q=q_threshold)
    for motif, qv in zip(candidates, adjusted):
        motif.q_value = qv

    # Filter by q-value
    motifs = [m for m in candidates if m.q_value is not None and m.q_value <= q_threshold]

    # Sort by q-value ascending, then by effect size as tiebreaker
    motifs.sort(key=lambda m: (m.q_value if m.q_value is not None else 1.0, -abs(m.success_rate - m.baseline_rate)))

    return motifs, len(candidates)
