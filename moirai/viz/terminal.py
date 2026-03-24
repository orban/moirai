from __future__ import annotations

from rich.console import Console
from rich.table import Table

from moirai.schema import Run, RunSummary, ValidationResult

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


def print_summary(summary: RunSummary) -> None:
    """Print aggregate run summary matching spec output format."""
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

    if summary.top_signatures:
        console.print("\nTop signatures:")
        for i, (sig, count) in enumerate(summary.top_signatures, 1):
            # Show type-only signature for readability
            types_only = " > ".join(part.split(":")[0] for part in sig.split(" > ")) if sig else "(empty)"
            console.print(f"{i}. {types_only} ({count})")

    if summary.error_counts:
        console.print("\nErrors:")
        for error_type, count in sorted(summary.error_counts.items(), key=lambda x: -x[1]):
            console.print(f"{error_type}: {count}")


def print_trace(run: Run, expand: bool = False) -> None:
    """Print a single run trace as a Rich table."""
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
