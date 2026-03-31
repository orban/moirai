from __future__ import annotations

from rich.console import Console
from rich.table import Table

from moirai.compress import (
    compress_phases,
    compress_run,
    cluster_subpatterns,
    phase_summary_str,
)
from moirai.schema import (
    ClusterResult,
    CohortDiff,
    DivergencePoint,
    Run,
    RunSummary,
    ValidationResult,
)

console = Console()


def print_validation(results: list[ValidationResult]) -> None:
    """Print validation results as a Rich table."""
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    table = Table(title="Validation Results")
    table.add_column("File", style="cyan")
    table.add_column("Status")
    table.add_column("Details")

    for r in results:
        status = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
        details_parts: list[str] = []
        for e in r.errors:
            details_parts.append(f"[red]{e}[/red]")
        for w in r.warnings:
            details_parts.append(f"[yellow]{w}[/yellow]")
        details = "\n".join(details_parts) if details_parts else "-"
        table.add_row(r.file_path, status, details)

    console.print(table)
    console.print(f"\n{passed} passed, {failed} failed, {len(results)} total")


def print_summary(summary: RunSummary, runs: list[Run] | None = None) -> None:
    """Print aggregate run summary."""
    console.print(f"Runs: {summary.run_count}")

    if summary.success_rate is not None:
        console.print(f"Success rate: {summary.success_rate:.1%}")
    else:
        console.print("Success rate: N/A")

    console.print(f"Avg steps: {summary.avg_steps:.1f}")
    console.print(f"Median steps: {summary.median_steps:.0f}")

    tin = f"{summary.avg_tokens_in:.0f}" if summary.avg_tokens_in is not None else "N/A"
    tout = f"{summary.avg_tokens_out:.0f}" if summary.avg_tokens_out is not None else "N/A"
    console.print(f"Avg tokens in/out: {tin} / {tout}")

    if summary.avg_latency_ms is not None:
        latency_s = summary.avg_latency_ms / 1000
        console.print(f"Avg latency: {latency_s:.1f}s")
    else:
        console.print("Avg latency: N/A")

    if summary.top_signatures and runs:
        # Check if step-level signatures are mostly unique
        top_count = summary.top_signatures[0][1] if summary.top_signatures else 0
        all_unique = top_count <= 1 and len(runs) > 5

        if all_unique:
            # Step-level is too granular — show phase-level patterns instead
            from collections import Counter
            phase_counter: Counter[str] = Counter()
            for run in runs:
                phase_counter[compress_phases(run)] += 1
            top_phases = phase_counter.most_common(10)

            console.print("\nTop phase patterns:")
            for i, (pattern, count) in enumerate(top_phases, 1):
                display = pattern if len(pattern) <= 80 else pattern[:77] + "..."
                console.print(f"  {i}. {display}  [dim]({count})[/dim]")
        else:
            from moirai.schema import signature as sig_fn
            sig_to_compressed: dict[str, str] = {}
            for run in runs:
                s = sig_fn(run)
                if s not in sig_to_compressed:
                    sig_to_compressed[s] = compress_run(run)

            console.print("\nTop patterns:")
            for i, (sig, count) in enumerate(summary.top_signatures[:10], 1):
                display = sig_to_compressed.get(sig, sig[:80])
                console.print(f"  {i}. {display}  [dim]({count})[/dim]")
    elif summary.top_signatures:
        console.print("\nTop signatures:")
        for i, (sig, count) in enumerate(summary.top_signatures[:10], 1):
            console.print(f"  {i}. {sig[:80]}  [dim]({count})[/dim]")

    if summary.error_counts:
        console.print("\nErrors:")
        for error_type, count in sorted(summary.error_counts.items(), key=lambda x: -x[1]):
            console.print(f"  {error_type}: {count}")


def print_trace(run: Run, expand: bool = False) -> None:
    """Print a single run trace."""
    # Metadata
    console.print(f"[bold]Run:[/bold] {run.run_id}")
    console.print(f"[bold]Task:[/bold] {run.task_id}")
    if run.task_family:
        console.print(f"[bold]Family:[/bold] {run.task_family}")
    if run.model:
        console.print(f"[bold]Model:[/bold] {run.model}")
    if run.harness:
        console.print(f"[bold]Harness:[/bold] {run.harness}")
    if run.agent:
        console.print(f"[bold]Agent:[/bold] {run.agent}")

    # Result
    success = run.result.success
    if success is True:
        result_str = "[green]PASS[/green]"
    elif success is False:
        result_str = "[red]FAIL[/red]"
    else:
        result_str = "[yellow]UNKNOWN[/yellow]"

    if run.result.score is not None:
        result_str += f" (score: {run.result.score:.2f})"
    if run.result.error_type:
        result_str += f" [red]({run.result.error_type})[/red]"
    console.print(f"[bold]Result:[/bold] {result_str}")

    if run.result.summary:
        console.print(f"[bold]Summary:[/bold] {run.result.summary}")

    # Compressed trajectory
    console.print(f"\n[bold]Trajectory:[/bold] {compress_run(run)}")
    console.print(f"[bold]Phases:[/bold] {compress_phases(run)}")
    console.print(f"[bold]Mix:[/bold] {phase_summary_str(run)}")
    console.print()

    # Step table
    table = Table(title="Steps")
    table.add_column("idx", justify="right", style="dim")
    table.add_column("type", style="cyan")
    table.add_column("name", style="green")
    table.add_column("status")
    table.add_column("tokens_in", justify="right")
    table.add_column("tokens_out", justify="right")
    table.add_column("latency_ms", justify="right")

    for step in run.steps:
        status_style = "" if step.status == "ok" else "[red]"
        status_end = "" if step.status == "ok" else "[/red]"
        status_str = f"{status_style}{step.status}{status_end}"

        tin = str(int(step.metrics["tokens_in"])) if "tokens_in" in step.metrics else "-"
        tout = str(int(step.metrics["tokens_out"])) if "tokens_out" in step.metrics else "-"
        lat = str(int(step.metrics["latency_ms"])) if "latency_ms" in step.metrics else "-"

        table.add_row(str(step.idx), step.type, step.name, status_str, tin, tout, lat)

    console.print(table)

    if expand:
        console.print("\n[bold]Step details:[/bold]")
        for step in run.steps:
            console.print(f"\n[dim]--- step {step.idx}: {step.type}:{step.name} ---[/dim]")
            if step.input:
                console.print(f"  [bold]Input:[/bold] {step.input}")
            if step.output:
                console.print(f"  [bold]Output:[/bold] {step.output}")


def _truncate_middle(text: str, max_len: int) -> str:
    """Truncate long text preserving beginning and ending."""
    if len(text) <= max_len:
        return text
    # Show beginning and ending with ... in the middle
    half = (max_len - 5) // 2
    return text[:half] + " ... " + text[-half:]


def _compress_prototype(proto: str) -> str:
    """Compress a raw signature prototype into a phase-level display.

    Prototypes are in "type:name > type:name" format. We extract step names,
    filter noise, classify into phases, and RLE the result.
    """
    from moirai.compress import NOISE_STEPS, PHASE_MAP, TYPE_PHASE_MAP, _rle, _format_rle

    parts = proto.split(" > ")
    phases: list[str] = []
    for p in parts:
        if ":" in p:
            step_type, name = p.split(":", 1)
        else:
            step_type, name = p, p
        if name in NOISE_STEPS:
            continue
        phase = PHASE_MAP.get(name, TYPE_PHASE_MAP.get(step_type, "other"))
        phases.append(phase)

    if not phases:
        return "(empty)"
    return _format_rle(_rle(phases))


def print_clusters(result: ClusterResult, runs: list[Run] | None = None) -> None:
    """Print cluster summary with compressed signatures and sub-patterns."""
    if not result.clusters:
        console.print("No clusters found.")
        return

    # Build run lookup if we have runs
    run_map: dict[str, Run] = {}
    if runs:
        run_map = {r.run_id: r for r in runs}

    for info in sorted(result.clusters, key=lambda c: -c.count):
        rate_str = f"{info.success_rate:.0%}" if info.success_rate is not None else "N/A"
        console.print(f"[bold]Cluster {info.cluster_id}[/bold]: {info.count} runs, {rate_str} success")

        if run_map:
            cluster_runs_list = [run_map[rid] for rid, cid in result.labels.items()
                                 if cid == info.cluster_id and rid in run_map]
            if cluster_runs_list:
                # Show phase pattern from a representative run (median length)
                sorted_by_len = sorted(cluster_runs_list, key=lambda r: len(r.steps))
                representative = sorted_by_len[len(sorted_by_len) // 2]
                phases = compress_phases(representative)
                console.print(f"  [cyan]{_truncate_middle(phases, 80)}[/cyan]")

                # Phase mix for the cluster
                from collections import Counter
                from moirai.compress import phase_summary as ps
                merged: Counter[str] = Counter()
                for r in cluster_runs_list:
                    merged.update(ps(r))
                total = sum(merged.values())
                if total:
                    mix = ", ".join(f"{c/total:.0%} {p}" for p, c in merged.most_common(4))
                    console.print(f"  [dim]{mix}[/dim]")

                # Sub-patterns
                groups = cluster_subpatterns(cluster_runs_list)
                for pattern_name, pattern_runs in sorted(groups.items(), key=lambda x: -len(x[1])):
                    if len(pattern_runs) == 0:
                        continue
                    n = len(pattern_runs)
                    successes = sum(1 for r in pattern_runs if r.result.success)
                    rate = f"{successes/n:.0%}" if n > 0 else "N/A"
                    avg_steps = sum(len(r.steps) for r in pattern_runs) / n
                    label = pattern_name.replace("_", " ")
                    example = compress_run(pattern_runs[0])
                    if len(example) > 60:
                        example = example[:57] + "..."
                    console.print(f"    {label} ({n}, {rate}, ~{avg_steps:.0f} steps): {example}")
        else:
            # Fallback: compress the raw prototype
            console.print(f"  [cyan]{_compress_prototype(info.prototype)}[/cyan]")

        if info.error_types:
            errors = ", ".join(f"{k}: {v}" for k, v in info.error_types.items())
            console.print(f"  [red]Errors: {errors}[/red]")

        console.print()


def print_cluster_divergence(
    cluster_divergences: list[tuple],
) -> None:
    """Print per-cluster divergence analysis.

    Each tuple is (ClusterInfo, Alignment, list[DivergencePoint], list[Run]).
    """
    if not cluster_divergences:
        console.print("No clusters with enough runs for divergence analysis.")
        return

    for info, alignment, points, cluster_runs_list in cluster_divergences:
        rate_str = f"{info.success_rate:.0%}" if info.success_rate is not None else "N/A"
        console.print(f"[bold]Cluster {info.cluster_id}[/bold]: {info.count} runs, {rate_str} success")

        # Show phase pattern
        sorted_by_len = sorted(cluster_runs_list, key=lambda r: len(r.steps))
        representative = sorted_by_len[len(sorted_by_len) // 2]
        phases = compress_phases(representative)
        console.print(f"  [cyan]{_truncate_middle(phases, 80)}[/cyan]")

        if not points:
            console.print("  [dim]No divergence points (all runs follow the same path)[/dim]")
            console.print()
            continue

        # Alignment quality stats
        n_cols = len(alignment.matrix[0]) if alignment.matrix and alignment.matrix[0] else 0
        avg_gaps = 0.0
        if alignment.matrix and n_cols > 0:
            avg_gaps = sum(
                sum(1 for v in row if v == "-") / n_cols
                for row in alignment.matrix
            ) / len(alignment.matrix)
        console.print(f"  [dim]Aligned to {n_cols} columns, {avg_gaps:.0%} avg gaps, {len(points)} divergence points[/dim]")

        # Show top 3 divergence points for this cluster
        run_lookup = {r.run_id: r for r in cluster_runs_list}
        for point in points[:3]:
            p_str = f", p={point.p_value:.3f}" if point.p_value is not None else ""
            console.print(f"\n  [bold]Position {point.column}[/bold] (entropy {point.entropy:.2f}{p_str})")

            if point.phase_context:
                console.print(f"  [dim]{point.phase_context}[/dim]")

            for value, count in sorted(point.value_counts.items(), key=lambda x: -x[1]):
                rate = point.success_by_value.get(value)
                rate_str = f"{rate:.0%}" if rate is not None else "N/A"
                if rate is not None and rate >= 0.7:
                    color = "green"
                elif rate is not None and rate <= 0.3:
                    color = "red"
                else:
                    color = "yellow"
                console.print(f"    [{color}]{value}[/{color}]: {count} runs, {rate_str} success")

                if alignment and hasattr(alignment, 'matrix'):
                    _print_branch_context(point, value, alignment, run_lookup)

        console.print()


def print_divergence(
    points: list[DivergencePoint],
    runs: list[Run] | None = None,
    alignment: object | None = None,
) -> None:
    """Print divergence points with context narratives."""
    if not points:
        console.print("No divergence points found (all runs have identical trajectories).")
        return

    # Build run lookup
    run_map: dict[str, Run] = {}
    if runs:
        run_map = {r.run_id: r for r in runs}

    console.print(f"[bold]Top divergence points ({len(points)} found):[/bold]\n")

    for point in points[:8]:  # top 8
        console.print(f"[bold]Position {point.column}[/bold] (entropy {point.entropy:.2f})")

        for value, count in sorted(point.value_counts.items(), key=lambda x: -x[1]):
            rate = point.success_by_value.get(value)
            rate_str = f"{rate:.0%}" if rate is not None else "N/A"

            # Color by success rate
            if rate is not None and rate >= 0.7:
                color = "green"
            elif rate is not None and rate <= 0.3:
                color = "red"
            else:
                color = "yellow"

            console.print(f"  [{color}]{value}[/{color}]: {count} runs, {rate_str} success")

            # Show representative run context if we have runs and alignment
            if run_map and alignment and hasattr(alignment, 'matrix') and hasattr(alignment, 'run_ids'):
                _print_branch_context(point, value, alignment, run_map)

        console.print()


def _print_branch_context(
    point: DivergencePoint,
    value: str,
    alignment: object,
    run_map: dict[str, Run],
) -> None:
    """Show context window (5 steps before/after) around a divergence for one branch."""
    from moirai.schema import GAP

    run_ids = alignment.run_ids  # type: ignore
    matrix = alignment.matrix  # type: ignore
    branch_rids: list[str] = []

    for run_idx, rid in enumerate(run_ids):
        if run_idx < len(matrix) and point.column < len(matrix[run_idx]):
            if matrix[run_idx][point.column] == value:
                branch_rids.append(rid)

    if not branch_rids or branch_rids[0] not in run_map:
        return

    # Get the aligned row for the representative run
    rep_rid = branch_rids[0]
    rep_idx = run_ids.index(rep_rid)
    row = matrix[rep_idx]

    # Extract context window: 5 before, the split, 5 after
    col = point.column
    start = max(0, col - 5)
    end = min(len(row), col + 6)
    window = row[start:end]

    # Filter gaps and format with the split point marked
    parts: list[str] = []
    for i, val in enumerate(window):
        if val == GAP:
            continue
        actual_col = start + i
        if actual_col == col:
            parts.append(f"\\[{val}]")  # escaped bracket so Rich doesn't parse as markup
        else:
            parts.append(val)

    if parts:
        context = " → ".join(parts)
        if len(context) > 70:
            context = context[:67] + "..."
        console.print(f"    [dim]context: ...{context}...[/dim]")


def _fmt_delta(a: float | None, b: float | None, fmt: str = ".1f", suffix: str = "") -> str:
    if a is None or b is None:
        return "N/A"
    delta = b - a
    sign = "+" if delta >= 0 else ""
    return f"{a:{fmt}}{suffix} -> {b:{fmt}}{suffix} ({sign}{delta:{fmt}})"


def _fmt_rate_delta(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return "N/A"
    delta = b - a
    sign = "+" if delta >= 0 else ""
    return f"{a:.1%} -> {b:.1%} ({sign}{delta * 100:.1f})"


def print_diff(diff: CohortDiff, a_label: str, b_label: str) -> None:
    """Print cohort diff with compressed signatures."""
    a = diff.a_summary
    b = diff.b_summary

    console.print(f"[bold]Cohort A:[/bold] {a_label} ({a.run_count} runs)")
    console.print(f"[bold]Cohort B:[/bold] {b_label} ({b.run_count} runs)")
    console.print()

    console.print(f"Success rate: {_fmt_rate_delta(a.success_rate, b.success_rate)}")
    console.print(f"Avg steps: {_fmt_delta(a.avg_steps, b.avg_steps)}")
    console.print(f"Avg tokens in: {_fmt_delta(a.avg_tokens_in, b.avg_tokens_in, '.0f')}")
    console.print(f"Avg tokens out: {_fmt_delta(a.avg_tokens_out, b.avg_tokens_out, '.0f')}")

    if diff.cluster_shifts:
        console.print("\n[bold]Cluster shifts:[/bold]")
        for proto, delta in diff.cluster_shifts:
            compressed = _truncate_middle(_compress_prototype(proto), 50)
            sign = "+" if delta > 0 else ""
            color = "green" if delta > 0 else "red"
            console.print(f"  {compressed}: [{color}]{sign}{delta} runs[/{color}]")

    if diff.a_only_signatures:
        console.print(f"\n[bold]Patterns only in A:[/bold]")
        for sig, count in diff.a_only_signatures[:5]:
            compressed = _truncate_middle(_compress_prototype(sig), 60)
            console.print(f"  {compressed}  [dim]({count})[/dim]")

    if diff.b_only_signatures:
        console.print(f"\n[bold]Patterns only in B:[/bold]")
        for sig, count in diff.b_only_signatures[:5]:
            compressed = _truncate_middle(_compress_prototype(sig), 60)
            console.print(f"  {compressed}  [dim]({count})[/dim]")


def print_motifs(motifs: list, baseline: float, n_runs: int, n_tested: int = 0) -> None:
    """Print discriminative motif patterns.

    Args:
        motifs: list of Motif objects from analyze/motifs.py
        baseline: overall success rate
        n_runs: total number of runs analyzed
        n_tested: total candidates tested before BH correction (for transition UX)
    """
    if not motifs:
        if n_tested > 0:
            console.print(f"[dim]{n_tested} patterns tested, 0 survived BH correction at q=0.05[/dim]")
        else:
            console.print("No significant patterns found.")
        return

    console.print(f"[bold]Discriminative patterns[/bold] ({n_runs} runs, {baseline:.0%} baseline success)\n")

    # Split into positive (lift > 1) and negative (lift < 1)
    positive = [m for m in motifs if m.lift > 1.05]
    negative = [m for m in motifs if m.lift < 0.95]

    if positive:
        console.print("[green][bold]Patterns correlated with success:[/bold][/green]")
        for m in positive[:10]:
            q_str = f"q={m.q_value:.3f}" if getattr(m, "q_value", None) is not None else (f"p={m.p_value:.3f}" if m.p_value is not None else "")
            pos_str = "early" if m.avg_position < 0.3 else ("late" if m.avg_position > 0.7 else "mid")
            console.print(
                f"  [green]{m.display}[/green]  "
                f"{m.success_rate:.0%} success ({m.total_runs} runs) vs {baseline:.0%} baseline  "
                f"[dim]{pos_str}, {q_str}[/dim]"
            )
        console.print()

    if negative:
        console.print("[red][bold]Patterns correlated with failure:[/bold][/red]")
        for m in negative[:10]:
            q_str = f"q={m.q_value:.3f}" if getattr(m, "q_value", None) is not None else (f"p={m.p_value:.3f}" if m.p_value is not None else "")
            pos_str = "early" if m.avg_position < 0.3 else ("late" if m.avg_position > 0.7 else "mid")
            console.print(
                f"  [red]{m.display}[/red]  "
                f"{m.success_rate:.0%} success ({m.total_runs} runs) vs {baseline:.0%} baseline  "
                f"[dim]{pos_str}, {q_str}[/dim]"
            )
        console.print()
