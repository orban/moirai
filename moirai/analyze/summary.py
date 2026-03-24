from __future__ import annotations

from collections import Counter
from statistics import median

from moirai.schema import Run, RunSummary, signature


def summarize_runs(runs: list[Run]) -> RunSummary:
    """Compute aggregate statistics across a set of runs."""
    n = len(runs)
    if n == 0:
        return RunSummary(
            run_count=0,
            success_rate=None,
            avg_steps=0.0,
            median_steps=0.0,
            avg_tokens_in=None,
            avg_tokens_out=None,
            avg_latency_ms=None,
            top_signatures=[],
            error_counts={},
        )

    # Success rate (excluding null)
    success_values = [r.result.success for r in runs if r.result.success is not None]
    if success_values:
        success_rate = sum(1 for s in success_values if s) / len(success_values)
    else:
        success_rate = None

    # Step counts
    step_counts = [len(r.steps) for r in runs]
    avg_steps = sum(step_counts) / n
    median_steps = float(median(step_counts))

    # Metrics aggregation
    all_tokens_in: list[float] = []
    all_tokens_out: list[float] = []
    all_latency: list[float] = []

    for run in runs:
        for step in run.steps:
            if "tokens_in" in step.metrics:
                all_tokens_in.append(step.metrics["tokens_in"])
            if "tokens_out" in step.metrics:
                all_tokens_out.append(step.metrics["tokens_out"])
            if "latency_ms" in step.metrics:
                all_latency.append(step.metrics["latency_ms"])

    avg_tokens_in = sum(all_tokens_in) / len(all_tokens_in) if all_tokens_in else None
    avg_tokens_out = sum(all_tokens_out) / len(all_tokens_out) if all_tokens_out else None
    avg_latency_ms = sum(all_latency) / len(all_latency) if all_latency else None

    # Signatures
    sig_counter = Counter(signature(r) for r in runs)
    top_signatures = sig_counter.most_common(10)

    # Error types
    error_counter: Counter[str] = Counter()
    for run in runs:
        if run.result.error_type:
            error_counter[run.result.error_type] += 1

    return RunSummary(
        run_count=n,
        success_rate=success_rate,
        avg_steps=avg_steps,
        median_steps=median_steps,
        avg_tokens_in=avg_tokens_in,
        avg_tokens_out=avg_tokens_out,
        avg_latency_ms=avg_latency_ms,
        top_signatures=top_signatures,
        error_counts=dict(error_counter),
    )
