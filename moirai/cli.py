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
        out = write_clusters_html(result, html)
        console.print(f"\nHTML written to {out}")


@app.command()
def branch(
    path: Path = typer.Argument(..., help="Path to a run file or directory"),
    level: str = typer.Option("type", help="Sequence level: type or name"),
    strict: bool = typer.Option(False, help="Treat warnings as errors"),
    model: str | None = typer.Option(None, help="Filter by model"),
    harness: str | None = typer.Option(None, help="Filter by harness"),
    task_family: str | None = typer.Option(None, "--task-family", help="Filter by task family"),
    html: Path | None = typer.Option(None, help="Write HTML output to path"),
) -> None:
    """Multi-run divergence analysis."""
    runs = _load_and_filter(path, strict, model=model, harness=harness, task_family=task_family)

    from moirai.analyze.align import align_runs
    from moirai.analyze.divergence import find_divergence_points
    from moirai.viz.terminal import print_divergence

    alignment = align_runs(runs, level=level)
    points = find_divergence_points(alignment, runs)
    print_divergence(points, runs, alignment)

    if html:
        from moirai.viz.html import write_branch_html
        out = write_branch_html(alignment, points, runs, html)
        console.print(f"\nHTML written to {out}")


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
