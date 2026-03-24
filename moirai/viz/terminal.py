from __future__ import annotations

from rich.console import Console
from rich.table import Table

from moirai.schema import ValidationResult

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
