from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from moirai.filters import filter_runs
from moirai.load import load_runs, validate_file, _find_json_files
from moirai.schema import Run

app = typer.Typer(
    name="moirai",
    help="Trajectory-level debugging for stochastic agent systems.",
    add_completion=False,
)

console = Console()
err_console = Console(stderr=True)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Trajectory-level debugging for stochastic agent systems."""
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit(0)


def _print_warnings(warnings: list[str]) -> None:
    for w in warnings:
        err_console.print(f"[yellow]warning:[/yellow] {w}")


def _load_and_filter(
    path: Path,
    strict: bool,
    model: str | None = None,
    harness: str | None = None,
    task_family: str | None = None,
) -> list[Run]:
    if not path.exists():
        err_console.print(f"[red]error:[/red] path not found: {path}")
        raise typer.Exit(1)

    try:
        runs, warnings = load_runs(path, strict=strict)
    except (FileNotFoundError, ValueError) as e:
        err_console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(1)
    _print_warnings(warnings)

    if not runs:
        err_console.print("[red]error:[/red] no valid runs found")
        raise typer.Exit(1)

    filtered = filter_runs(runs, model=model, harness=harness, task_family=task_family)
    if not filtered:
        err_console.print("[red]error:[/red] no runs matched the given filters")
        raise typer.Exit(1)

    return filtered


@app.command()
def validate(
    path: Path = typer.Argument(..., help="Path to a run file or directory"),
    strict: bool = typer.Option(False, help="Treat warnings as errors"),
) -> None:
    """Check JSON validity and schema compliance."""
    if not path.exists():
        err_console.print(f"[red]error:[/red] path not found: {path}")
        raise typer.Exit(1)

    json_files = _find_json_files(path)
    if not json_files:
        err_console.print(f"[red]error:[/red] no JSON files found in {path}")
        raise typer.Exit(1)

    from moirai.viz.terminal import print_validation

    results = [validate_file(f, strict=strict) for f in json_files]
    print_validation(results)

    if any(not r.passed for r in results):
        raise typer.Exit(1)


@app.command()
def summary(
    path: Path = typer.Argument(..., help="Path to a run file or directory"),
    strict: bool = typer.Option(False, help="Treat warnings as errors"),
    model: str | None = typer.Option(None, help="Filter by model"),
    harness: str | None = typer.Option(None, help="Filter by harness"),
    task_family: str | None = typer.Option(None, "--task-family", help="Filter by task family"),
) -> None:
    """Aggregate stats across runs."""
    runs = _load_and_filter(path, strict, model=model, harness=harness, task_family=task_family)

    from moirai.analyze.summary import summarize_runs
    from moirai.viz.terminal import print_summary

    result = summarize_runs(runs)
    print_summary(result, runs)


@app.command()
def trace(
    path: Path = typer.Argument(..., help="Path to a single run JSON file"),
    expand: bool = typer.Option(False, help="Show input/output details for each step"),
) -> None:
    """Inspect a single run."""
    if path.is_dir():
        err_console.print("[red]error:[/red] trace requires a single run file, not a directory")
        raise typer.Exit(1)

    runs, warnings = load_runs(path)
    _print_warnings(warnings)

    if not runs:
        err_console.print("[red]error:[/red] failed to load run")
        raise typer.Exit(1)

    from moirai.viz.terminal import print_trace

    print_trace(runs[0], expand=expand)


@app.command()
def clusters(
    path: Path = typer.Argument(..., help="Path to a run file or directory"),
    level: str = typer.Option("type", help="Sequence level: type or name"),
    threshold: float = typer.Option(0.3, help="Clustering distance threshold"),
    strict: bool = typer.Option(False, help="Treat warnings as errors"),
    model: str | None = typer.Option(None, help="Filter by model"),
    harness: str | None = typer.Option(None, help="Filter by harness"),
    task_family: str | None = typer.Option(None, "--task-family", help="Filter by task family"),
    html: Path | None = typer.Option(None, help="Write HTML output to path"),
) -> None:
    """Cluster runs by trajectory structure."""
    runs = _load_and_filter(path, strict, model=model, harness=harness, task_family=task_family)

    from moirai.analyze.cluster import cluster_runs
    from moirai.viz.terminal import print_clusters

    result = cluster_runs(runs, level=level, threshold=threshold)
    print_clusters(result, runs)

    if html:
        from moirai.viz.html import write_clusters_html
        out = write_clusters_html(result, html, runs)
        console.print(f"\nHTML written to {out}")


@app.command()
def branch(
    path: Path = typer.Argument(..., help="Path to a run file or directory"),
    strict: bool = typer.Option(False, help="Treat warnings as errors"),
    model: str | None = typer.Option(None, help="Filter by model"),
    harness: str | None = typer.Option(None, help="Filter by harness"),
    task_family: str | None = typer.Option(None, "--task-family", help="Filter by task family"),
    html: Path | None = typer.Option(None, help="Write HTML output to path"),
) -> None:
    """Per-task divergence analysis. Groups by task_id, aligns repeated runs, finds where they split."""
    runs = _load_and_filter(path, strict, model=model, harness=harness, task_family=task_family)

    from collections import defaultdict
    from moirai.analyze.align import align_runs
    from moirai.analyze.divergence import find_divergence_points

    # Group by task_id — alignment only makes sense for repeated runs of the same task
    tasks: dict[str, list] = defaultdict(list)
    for r in runs:
        tasks[r.task_id].append(r)

    # Only tasks with mixed outcomes (both pass and fail)
    mixed_tasks = {tid: trs for tid, trs in tasks.items()
                   if any(r.result.success for r in trs) and any(not r.result.success for r in trs)
                   and len(trs) >= 3}

    if not mixed_tasks:
        console.print("No tasks with mixed pass/fail outcomes (need 3+ runs per task)")
        raise typer.Exit(0)

    console.print(f"[bold]{len(mixed_tasks)} tasks with mixed outcomes[/bold] (out of {len(tasks)} total)\n")

    task_results = []
    for tid in sorted(mixed_tasks, key=lambda t: -len(mixed_tasks[t])):
        task_runs = mixed_tasks[tid]
        alignment = align_runs(task_runs, level="name")
        points = find_divergence_points(alignment, task_runs, min_branch_size=1, p_threshold=0.5)

        n_pass = sum(1 for r in task_runs if r.result.success)
        n_fail = len(task_runs) - n_pass
        n_cols = len(alignment.matrix[0]) if alignment.matrix and alignment.matrix[0] else 0

        task_results.append((tid, task_runs, alignment, points))

        console.print(f"[bold]{tid}[/bold]: {len(task_runs)} runs ({n_pass}P/{n_fail}F), aligned to {n_cols} columns")

        if not points:
            console.print("  [dim]No significant divergence points[/dim]\n")
            continue

        for point in points[:3]:
            p_str = f"p={point.p_value:.3f}" if point.p_value is not None else ""
            console.print(f"  [bold]Position {point.column}[/bold] ({p_str})")
            if point.phase_context:
                console.print(f"  [dim]{point.phase_context}[/dim]")
            for value, count in sorted(point.value_counts.items(), key=lambda x: -x[1]):
                rate = point.success_by_value.get(value)
                rate_str = f"{rate:.0%}" if rate is not None else "?"
                if rate is not None and rate >= 0.6:
                    color = "green"
                elif rate is not None and rate <= 0.2:
                    color = "red"
                else:
                    color = "yellow"
                console.print(f"    [{color}]{value}[/{color}]: {count} runs, {rate_str} success")
        console.print()

    if html:
        from moirai.viz.html import write_branch_html
        out = write_branch_html(None, [], runs, html, task_results=task_results)
        console.print(f"\nHTML written to {out}")


@app.command()
def patterns(
    path: Path = typer.Argument(..., help="Path to a run file or directory"),
    min_n: int = typer.Option(3, help="Minimum pattern length"),
    max_n: int = typer.Option(5, help="Maximum pattern length"),
    min_count: int = typer.Option(3, help="Minimum runs containing pattern"),
    strict: bool = typer.Option(False, help="Treat warnings as errors"),
    model: str | None = typer.Option(None, help="Filter by model"),
    harness: str | None = typer.Option(None, help="Filter by harness"),
    task_family: str | None = typer.Option(None, "--task-family", help="Filter by task family"),
) -> None:
    """Find step patterns that predict success or failure."""
    runs = _load_and_filter(path, strict, model=model, harness=harness, task_family=task_family)

    from moirai.analyze.motifs import find_motifs
    from moirai.viz.terminal import print_motifs

    motifs = find_motifs(runs, min_n=min_n, max_n=max_n, min_count=min_count)

    known = [r for r in runs if r.result.success is not None]
    baseline = sum(1 for r in known if r.result.success) / len(known) if known else 0.0
    print_motifs(motifs, baseline, len(runs))


@app.command()
def diff(
    path: Path = typer.Argument(..., help="Path to a run file or directory"),
    a: list[str] | None = typer.Option(None, "--a", help="Cohort A filter (K=V, repeatable)"),
    b: list[str] | None = typer.Option(None, "--b", help="Cohort B filter (K=V, repeatable)"),
    level: str = typer.Option("type", help="Sequence level: type or name"),
    threshold: float = typer.Option(0.3, help="Clustering distance threshold"),
    strict: bool = typer.Option(False, help="Treat warnings as errors"),
    html: Path | None = typer.Option(None, help="Write HTML output to path"),
) -> None:
    """Compare two cohorts."""
    if not a or not b:
        err_console.print("[red]error:[/red] both --a and --b filters are required")
        err_console.print("example: moirai diff runs/ --a harness=baseline --b harness=router")
        raise typer.Exit(2)

    # Load all runs
    runs = _load_and_filter(path, strict)

    from moirai.filters import apply_kv_filters
    from moirai.analyze.compare import compare_cohorts
    from moirai.viz.terminal import print_diff

    try:
        a_runs = apply_kv_filters(runs, a)
        b_runs = apply_kv_filters(runs, b)
    except ValueError as e:
        err_console.print(f"[red]error:[/red] invalid filter: {e}")
        raise typer.Exit(2)

    if not a_runs:
        # Show available values for the filter keys
        err_console.print(f"[red]error:[/red] cohort A matched 0 runs")
        _show_available_values(runs, a)
        raise typer.Exit(1)

    if not b_runs:
        err_console.print(f"[red]error:[/red] cohort B matched 0 runs")
        _show_available_values(runs, b)
        raise typer.Exit(1)

    result = compare_cohorts(a_runs, b_runs, level=level, threshold=threshold)
    a_label = " ".join(a)
    b_label = " ".join(b)
    print_diff(result, a_label, b_label)

    if html:
        from moirai.viz.html import write_diff_html
        out = write_diff_html(result, a_label, b_label, html)
        console.print(f"\nHTML written to {out}")


@app.command()
def explain(
    path: Path = typer.Argument(..., help="Path to a run file or directory"),
    run_id: str = typer.Option(..., "--run", help="Run ID to explain"),
    level: str = typer.Option("type", help="Sequence level: type or name"),
    threshold: float = typer.Option(0.3, help="Clustering distance threshold"),
    strict: bool = typer.Option(False, help="Treat warnings as errors"),
) -> None:
    """Explain why a specific run succeeded or failed compared to similar runs."""
    all_runs = _load_and_filter(path, strict)

    # Find the target run
    target = None
    for r in all_runs:
        if r.run_id == run_id or r.run_id.startswith(run_id):
            target = r
            break

    if target is None:
        err_console.print(f"[red]error:[/red] run '{run_id}' not found")
        # Show available run IDs
        err_console.print("available runs:")
        for r in all_runs[:20]:
            s = "PASS" if r.result.success else "FAIL"
            err_console.print(f"  {r.run_id} [{s}]")
        if len(all_runs) > 20:
            err_console.print(f"  ... and {len(all_runs) - 20} more")
        raise typer.Exit(1)

    from moirai.analyze.cluster import cluster_runs as do_cluster
    from moirai.analyze.align import align_runs
    from moirai.analyze.divergence import find_divergence_points
    from moirai.compress import compress_run, compress_phases, phase_summary_str

    # Cluster
    cluster_result = do_cluster(all_runs, level=level, threshold=threshold)
    target_cluster_id = cluster_result.labels.get(target.run_id)

    # Get cluster siblings
    siblings = [r for r in all_runs if cluster_result.labels.get(r.run_id) == target_cluster_id]
    cluster_info = None
    for c in cluster_result.clusters:
        if c.cluster_id == target_cluster_id:
            cluster_info = c
            break

    # Header
    success = target.result.success
    if success is True:
        result_tag = "[green]PASS[/green]"
    elif success is False:
        result_tag = "[red]FAIL[/red]"
    else:
        result_tag = "[yellow]UNKNOWN[/yellow]"

    console.print(f"[bold]Run:[/bold] {target.run_id} {result_tag}")
    console.print(f"[bold]Task:[/bold] {target.task_id}")
    if target.model:
        console.print(f"[bold]Model:[/bold] {target.model}")
    if target.harness:
        console.print(f"[bold]Harness:[/bold] {target.harness}")

    if cluster_info:
        rate_str = f"{cluster_info.success_rate:.0%}" if cluster_info.success_rate is not None else "N/A"
        console.print(f"[bold]Cluster:[/bold] {cluster_info.cluster_id} ({cluster_info.count} runs, {rate_str} success)")

    # Trajectory
    console.print(f"\n[bold]Trajectory:[/bold] {compress_run(target)}")
    console.print(f"[bold]Phases:[/bold] {compress_phases(target)}")
    console.print(f"[bold]Mix:[/bold] {phase_summary_str(target)}")
    console.print(f"[bold]Steps:[/bold] {len(target.steps)}")

    if target.result.error_type:
        console.print(f"[bold]Error:[/bold] [red]{target.result.error_type}[/red]")
    if target.result.summary:
        console.print(f"[bold]Summary:[/bold] {target.result.summary}")

    # Comparison to cluster
    if siblings and len(siblings) > 1:
        avg_steps = sum(len(r.steps) for r in siblings) / len(siblings)
        pct_diff = ((len(target.steps) - avg_steps) / avg_steps * 100) if avg_steps > 0 else 0

        console.print(f"\n[bold]Compared to cluster:[/bold]")
        if abs(pct_diff) > 20:
            direction = "shorter" if pct_diff < 0 else "longer"
            console.print(f"  This run is {abs(pct_diff):.0f}% {direction} than cluster average ({avg_steps:.0f} steps)")

        pass_siblings = [r for r in siblings if r.result.success]
        fail_siblings = [r for r in siblings if r.result.success is False]

        if target.result.success and fail_siblings:
            console.print(f"\n  [bold]In failing runs from this cluster ({len(fail_siblings)}):[/bold]")
            avg_fail_steps = sum(len(r.steps) for r in fail_siblings) / len(fail_siblings)
            console.print(f"    avg {avg_fail_steps:.0f} steps (vs {len(target.steps)} in this run)")
            # Show compressed example of a failing run
            example = compress_run(fail_siblings[0])
            if len(example) > 70:
                example = example[:67] + "..."
            console.print(f"    e.g. {example}")

        elif not target.result.success and pass_siblings:
            console.print(f"\n  [bold]In passing runs from this cluster ({len(pass_siblings)}):[/bold]")
            avg_pass_steps = sum(len(r.steps) for r in pass_siblings) / len(pass_siblings)
            console.print(f"    avg {avg_pass_steps:.0f} steps (vs {len(target.steps)} in this run)")
            example = compress_run(pass_siblings[0])
            if len(example) > 70:
                example = example[:67] + "..."
            console.print(f"    e.g. {example}")

    # Key divergence — align siblings at name level for fine-grained detail
    if siblings and len(siblings) >= 3:
        alignment = align_runs(siblings, level="name")
        points = find_divergence_points(alignment, siblings)

        if points:
            # Find the divergence point most relevant to this run's outcome
            best_point = None
            best_spread = 0.0
            target_idx = alignment.run_ids.index(target.run_id) if target.run_id in alignment.run_ids else None

            for point in points[:10]:
                if target_idx is not None and point.column < len(alignment.matrix[target_idx]):
                    target_val = alignment.matrix[target_idx][point.column]
                    target_rate = point.success_by_value.get(target_val)
                    if target_rate is not None:
                        # How much does this branch differ from the overall?
                        other_rates = [r for v, r in point.success_by_value.items()
                                       if v != target_val and r is not None]
                        if other_rates:
                            spread = abs(target_rate - sum(other_rates) / len(other_rates))
                            if spread > best_spread:
                                best_spread = spread
                                best_point = point

            if best_point and target_idx is not None:
                target_val = alignment.matrix[target_idx][best_point.column]
                target_rate = best_point.success_by_value.get(target_val)

                console.print(f"\n[bold]Key divergence (position {best_point.column}):[/bold]")
                for value, count in sorted(best_point.value_counts.items(), key=lambda x: -x[1]):
                    rate = best_point.success_by_value.get(value)
                    rate_str = f"{rate:.0%}" if rate is not None else "N/A"
                    marker = " [bold]<-- this run[/bold]" if value == target_val else ""
                    color = "green" if rate and rate >= 0.7 else ("red" if rate is not None and rate <= 0.3 else "yellow")
                    console.print(f"  [{color}]{value}[/{color}]: {count} runs, {rate_str} success{marker}")


def _show_available_values(runs: list[Run], kv_pairs: list[str]) -> None:
    """Show available values for filter keys to help the user."""
    from moirai.filters import parse_kv_filter

    for kv in kv_pairs:
        try:
            key, _ = parse_kv_filter(kv)
        except ValueError:
            continue

        values: set[str] = set()
        for run in runs:
            val = getattr(run, key, None)
            if val is None:
                val = run.tags.get(key)
            if val is not None:
                values.add(str(val))

        if values:
            err_console.print(f"  available {key} values: {', '.join(sorted(values))}")
