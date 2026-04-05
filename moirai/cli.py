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
    concordance: bool = typer.Option(False, help="Compute structural concordance per cluster"),
) -> None:
    """Cluster runs by trajectory structure."""
    runs = _load_and_filter(path, strict, model=model, harness=harness, task_family=task_family)

    from moirai.analyze.cluster import cluster_runs, compute_concordance
    from moirai.viz.terminal import print_clusters

    result = cluster_runs(runs, level=level, threshold=threshold)

    concordance_scores = None
    if concordance:
        concordance_scores = compute_concordance(runs, result.labels, level="name")

    print_clusters(result, runs, concordance=concordance_scores)

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
    task: str | None = typer.Option(None, "--task", help="Filter to a specific task ID"),
    feature: str | None = typer.Option(None, "--feature",
        help="Annotate runs with a behavioral feature (from moirai features)"),
    html: Path | None = typer.Option(None, help="Write HTML output to path"),
    analyze: bool = typer.Option(False, help="Run LLM analysis on divergence points (requires anthropic SDK)"),
    viewer: Path | None = typer.Option(None, help="Write interactive heatmap viewer HTML"),
) -> None:
    """Per-task divergence analysis. Groups by task_id, aligns repeated runs, finds where they split."""
    runs = _load_and_filter(path, strict, model=model, harness=harness, task_family=task_family)

    from collections import defaultdict
    from moirai.analyze.align import consensus, align_runs
    from moirai.analyze.divergence import find_divergence_points
    from moirai.analyze.stats import kendall_tau_b

    # Group by task_id — alignment only makes sense for repeated runs of the same task
    tasks: dict[str, list] = defaultdict(list)
    for r in runs:
        tasks[r.task_id].append(r)

    # Only tasks with mixed outcomes (both pass and fail)
    mixed_tasks = {tid: trs for tid, trs in tasks.items()
                   if any(r.result.success for r in trs) and any(not r.result.success for r in trs)
                   and len(trs) >= 3}

    # Filter to specific task if requested
    if task:
        if task in mixed_tasks:
            mixed_tasks = {task: mixed_tasks[task]}
        else:
            err_console.print(f"[red]Task '{task}' not found or has no mixed outcomes.[/red]")
            raise typer.Exit(2)

    if not mixed_tasks:
        console.print("No tasks with mixed pass/fail outcomes (need 3+ runs per task)")
        raise typer.Exit(0)

    # Resolve feature spec if requested
    feature_spec = None
    if feature:
        from moirai.analyze.features import FEATURES as _FEATURES
        feature_spec = next((f for f in _FEATURES if f.name == feature), None)
        if feature_spec is None:
            valid = [f.name for f in _FEATURES]
            err_console.print(f"[red]Unknown feature '{feature}'. Valid: {', '.join(valid)}[/red]")
            raise typer.Exit(2)

    console.print(f"[bold]{len(mixed_tasks)} tasks with mixed outcomes[/bold] (out of {len(tasks)} total)\n")

    task_results = []
    for tid in sorted(mixed_tasks, key=lambda t: -len(mixed_tasks[t])):
        task_runs = mixed_tasks[tid]
        alignment = align_runs(task_runs, level="name")
        points, _ = find_divergence_points(alignment, task_runs, min_branch_size=1, q_threshold=0.5)

        n_pass = sum(1 for r in task_runs if r.result.success)
        n_fail = len(task_runs) - n_pass
        n_cols = len(alignment.matrix[0]) if alignment.matrix and alignment.matrix[0] else 0

        task_results.append((tid, task_runs, alignment, points))

        # Compute concordance from existing alignment (no extra cost)
        tau_str = ""
        known = [r for r in task_runs if r.result.success is not None]
        has_mixed = any(r.result.success for r in known) and any(not r.result.success for r in known)
        if len(known) >= 5 and has_mixed and alignment.matrix and n_cols > 0:
            cons = consensus(alignment.matrix)
            distances = [sum(1 for a, b in zip(row, cons) if a != b) / n_cols for row in alignment.matrix]
            if len(set(distances)) >= 2:
                outcomes = [1.0 if r.result.success else 0.0 for r in task_runs]
                tau, _ = kendall_tau_b([-d for d in distances], outcomes)
                tau_str = f", concordance: τ={tau:.2f}"

        console.print(f"[bold]{tid}[/bold]: {len(task_runs)} runs ({n_pass}P/{n_fail}F), aligned to {n_cols} columns{tau_str}")

        # Feature annotation (if --feature was given)
        if feature_spec is not None:
            valued = [(r, feature_spec.compute(r)) for r in task_runs]
            valued_nums = [(r, v) for r, v in valued if v is not None]
            if valued_nums:
                sorted_vals = sorted(v for _, v in valued_nums)
                median = sorted_vals[len(sorted_vals) // 2]
                high = [(r, v) for r, v in valued_nums if v > median]
                low = [(r, v) for r, v in valued_nums if v <= median]
                high_pass = sum(1 for r, _ in high if r.result.success) / len(high) if high else 0
                low_pass = sum(1 for r, _ in low if r.result.success) / len(low) if low else 0
                direction = "higher" if feature_spec.direction == "positive" else "lower"

                console.print(f"\n  [bold]Feature: {feature_spec.name}[/bold]"
                              f" [dim]({feature_spec.description}, {direction} = better)[/dim]")
                console.print(f"  Median split: HIGH (>{median:.2f}) = {high_pass:.0%} pass"
                              f" | LOW (\u2264{median:.2f}) = {low_pass:.0%} pass")

                for r, v in sorted(valued_nums, key=lambda x: x[1]):
                    outcome = "[green]pass[/green]" if r.result.success else "[red]fail[/red]"
                    group = "[bold]HIGH[/bold]" if v > median else "[dim]LOW[/dim]"
                    console.print(f"    {outcome}  {feature_spec.name}: {v:.2f}  {group}")
                console.print()
            else:
                console.print(f"  [dim]No runs have data for {feature_spec.name}[/dim]\n")

        if not points:
            console.print("  [dim]No significant divergence points[/dim]\n")
            continue

        for point in points[:3]:
            p_str = f"q={point.q_value:.3f}" if getattr(point, "q_value", None) is not None else (f"p={point.p_value:.3f}" if point.p_value is not None else "")
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
        out = write_branch_html(None, [], runs, html, task_results=task_results, analyze=analyze)
        console.print(f"\nHTML written to {out}")

    if viewer:
        from moirai.viz.html import write_stream_html
        out = write_stream_html(runs, task_results, viewer)
        console.print(f"\nViewer written to {out}")


@app.command()
def patterns(
    path: Path = typer.Argument(..., help="Path to a run file or directory"),
    min_n: int = typer.Option(3, help="Minimum pattern length"),
    max_n: int = typer.Option(5, help="Maximum pattern length"),
    min_count: int = typer.Option(3, help="Minimum runs containing pattern"),
    gapped: bool = typer.Option(False, help="Also discover gapped (ordered subsequence) patterns"),
    max_length: int = typer.Option(3, "--max-length", help="Max gapped pattern length (default 3, use 4 for deeper search)"),
    permutation_test: int | None = typer.Option(None, "--permutation-test", help="Run N permutations to estimate empirical FDR"),
    stratify: str | None = typer.Option(None, "--stratify", help="Stratify motif discovery by field (e.g. task_family) to prevent cross-group confounds"),
    strict: bool = typer.Option(False, help="Treat warnings as errors"),
    model: str | None = typer.Option(None, help="Filter by model"),
    harness: str | None = typer.Option(None, help="Filter by harness"),
    task_family: str | None = typer.Option(None, "--task-family", help="Filter by task family"),
) -> None:
    """Find step patterns that predict success or failure."""
    runs = _load_and_filter(path, strict, model=model, harness=harness, task_family=task_family)

    from moirai.analyze.motifs import find_gapped_motifs, find_motifs, stratified_find_motifs, stratified_find_gapped_motifs
    from moirai.viz.terminal import print_motifs

    if stratify:
        stratify_fn = lambda r: getattr(r, stratify, None) or r.tags.get(stratify)
        motifs, n_tested = stratified_find_motifs(runs, stratify_by=stratify_fn, min_n=min_n, max_n=max_n, min_count=min_count)
    else:
        motifs, n_tested = find_motifs(runs, min_n=min_n, max_n=max_n, min_count=min_count)

    all_results: list = list(motifs)
    total_tested = n_tested

    if gapped:
        if stratify:
            stratify_fn = lambda r: getattr(r, stratify, None) or r.tags.get(stratify)
            gapped_motifs, gapped_tested = stratified_find_gapped_motifs(runs, stratify_by=stratify_fn, max_length=max_length, min_count=min_count)
        else:
            gapped_motifs, gapped_tested = find_gapped_motifs(runs, max_length=max_length, min_count=min_count)
        all_results.extend(gapped_motifs)
        total_tested += gapped_tested
        # Re-sort merged results by q-value
        all_results.sort(key=lambda m: (m.q_value if m.q_value is not None else 1.0, -abs(m.success_rate - m.baseline_rate)))

    known = [r for r in runs if r.result.success is not None]
    baseline = sum(1 for r in known if r.result.success) / len(known) if known else 0.0
    print_motifs(all_results, baseline, len(runs), n_tested=total_tested)

    if permutation_test is not None and all_results:
        import numpy as np
        from moirai.analyze.motifs import _extract_ngrams, _filtered_names
        from moirai.analyze.stats import permutation_fdr

        # Build boolean membership matrix (patterns × runs)
        # For contiguous motifs, check n-gram membership
        # For gapped motifs, check ordered subsequence membership
        from moirai.analyze.motifs import _is_subsequence
        from moirai.schema import GappedMotif

        run_names: list[list[str]] = [_filtered_names(run) for run in known]
        run_grams: list[set[tuple[str, ...]]] = [
            {g for g, _ in _extract_ngrams(names, min_n, max_n)}
            for names in run_names
        ]

        rows: list[list[bool]] = []
        for m in all_results:
            if isinstance(m, GappedMotif):
                row = [_is_subsequence(m.anchors, tuple(names)) for names in run_names]
            else:
                row = [m.pattern in grams for grams in run_grams]
            rows.append(row)

        membership = np.array(rows, dtype=bool)
        outcomes = np.array([r.result.success for r in known], dtype=bool)

        console.print(f"\n[bold]Permutation test[/bold] ({permutation_test} permutations)")
        fdr, discoveries = permutation_fdr(
            membership, outcomes,
            n_permutations=permutation_test,
        )
        mean_null = sum(discoveries) / len(discoveries) if discoveries else 0
        console.print(
            f"  Empirical FDR: [bold]{fdr:.1%}[/bold] "
            f"({mean_null:.1f} mean null discoveries vs {len(all_results)} actual)"
        )


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
    run_id: str | None = typer.Option(None, "--run", help="Explain a single run by ID"),
    task: str | None = typer.Option(None, "--task", help="Restrict to a specific task_id"),
    top_n: int = typer.Option(5, "--top-n", help="Divergent columns to analyze"),
    mode: str = typer.Option("auto", "--mode", help="auto, claude, codex, or structural"),
    do_cluster: bool = typer.Option(False, "--cluster", help="Enable clustering + concordance"),
    fmt: str = typer.Option("terminal", "--format", help="Output format: terminal or json"),
    max_runs: int = typer.Option(50, "--max-runs", help="Cap runs per task group"),
    timeout: int = typer.Option(120, "--timeout", help="LLM subprocess timeout in seconds"),
    seed: int = typer.Option(42, "--seed", help="Random seed for sampling"),
    level: str = typer.Option("type", help="Alignment level for --run mode"),
    threshold: float = typer.Option(0.3, help="Cluster threshold for --run mode"),
    strict: bool = typer.Option(False, help="Treat warnings as errors"),
) -> None:
    """Explain why agent runs succeed or fail.

    Without --run: cross-run differential analysis using LLM content comparison.
    With --run: explain a single run within its cluster context.
    """
    all_runs = _load_and_filter(path, strict)

    if run_id:
        _explain_single_run(all_runs, run_id, level, threshold)
        return

    # --- Cross-run content analysis path ---
    if mode not in ("auto", "claude", "codex", "structural"):
        err_console.print(f"[red]error:[/red] unknown mode: {mode}")
        raise typer.Exit(1)

    from moirai.analyze.content import run_explain

    reports = run_explain(
        all_runs, mode=mode, task_filter=task, top_n=top_n,
        max_runs=max_runs, timeout=timeout, seed=seed, cluster=do_cluster,
    )

    # Skip summary
    if reports and reports[0].n_skipped > 0:
        err_console.print(f"[dim]{reports[0].n_skipped} task groups skipped[/dim]")

    # Auto-mode degradation indicator
    if mode == "auto" and reports and all(r.summary == "" for r in reports):
        err_console.print("[dim]structural only — no LLM CLI found[/dim]")

    if not reports:
        err_console.print("[dim]no qualifying task groups[/dim]")

    if fmt == "json":
        import json as json_mod
        from dataclasses import asdict
        console.print(json_mod.dumps([asdict(r) for r in reports], indent=2, default=str))
    else:
        from moirai.viz.terminal import print_explanation
        for report in reports:
            print_explanation(report)

    # Exit codes
    if mode in ("claude", "codex") and any(r.summary == "" for r in reports):
        raise typer.Exit(2)


def _explain_single_run(
    all_runs: list,
    run_id: str,
    level: str,
    threshold: float,
) -> None:
    """Explain a single run within its cluster context (existing behavior)."""
    target = None
    for r in all_runs:
        if r.run_id == run_id or r.run_id.startswith(run_id):
            target = r
            break

    if target is None:
        err_console.print(f"[red]error:[/red] run '{run_id}' not found")
        err_console.print("available runs:")
        for r in all_runs[:20]:
            s = "PASS" if r.result.success else "FAIL"
            err_console.print(f"  {r.run_id} [{s}]")
        if len(all_runs) > 20:
            err_console.print(f"  ... and {len(all_runs) - 20} more")
        raise typer.Exit(1)

    from moirai.analyze.cluster import cluster_runs as do_cluster_runs
    from moirai.analyze.align import align_runs
    from moirai.analyze.divergence import find_divergence_points
    from moirai.compress import compress_run, compress_phases, phase_summary_str

    cluster_result = do_cluster_runs(all_runs, level=level, threshold=threshold)
    target_cluster_id = cluster_result.labels.get(target.run_id)

    siblings = [r for r in all_runs if cluster_result.labels.get(r.run_id) == target_cluster_id]
    cluster_info = None
    for c in cluster_result.clusters:
        if c.cluster_id == target_cluster_id:
            cluster_info = c
            break

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

    console.print(f"\n[bold]Trajectory:[/bold] {compress_run(target)}")
    console.print(f"[bold]Phases:[/bold] {compress_phases(target)}")
    console.print(f"[bold]Mix:[/bold] {phase_summary_str(target)}")
    console.print(f"[bold]Steps:[/bold] {len(target.steps)}")

    if target.result.error_type:
        console.print(f"[bold]Error:[/bold] [red]{target.result.error_type}[/red]")
    if target.result.summary:
        console.print(f"[bold]Summary:[/bold] {target.result.summary}")

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

    if siblings and len(siblings) >= 3:
        alignment = align_runs(siblings, level="name")
        points, _ = find_divergence_points(alignment, siblings)

        if points:
            best_point = None
            best_spread = 0.0
            target_idx = alignment.run_ids.index(target.run_id) if target.run_id in alignment.run_ids else None

            for point in points[:10]:
                if target_idx is not None and point.column < len(alignment.matrix[target_idx]):
                    target_val = alignment.matrix[target_idx][point.column]
                    target_rate = point.success_by_value.get(target_val)
                    if target_rate is not None:
                        other_rates = [r for v, r in point.success_by_value.items()
                                       if v != target_val and r is not None]
                        if other_rates:
                            spread = abs(target_rate - sum(other_rates) / len(other_rates))
                            if spread > best_spread:
                                best_spread = spread
                                best_point = point

            if best_point and target_idx is not None:
                target_val = alignment.matrix[target_idx][best_point.column]

                console.print(f"\n[bold]Key divergence (position {best_point.column}):[/bold]")
                for value, count in sorted(best_point.value_counts.items(), key=lambda x: -x[1]):
                    rate = best_point.success_by_value.get(value)
                    rate_str = f"{rate:.0%}" if rate is not None else "N/A"
                    marker = " [bold]<-- this run[/bold]" if value == target_val else ""
                    color = "green" if rate and rate >= 0.7 else ("red" if rate is not None and rate <= 0.3 else "yellow")
                    console.print(f"  [{color}]{value}[/{color}]: {count} runs, {rate_str} success{marker}")


@app.command()
def divergence(
    path: Path = typer.Argument(..., help="Path to a run file or directory"),
    task: str = typer.Option(None, "--task", help="Task ID (prefix match). If omitted, shows all mixed-outcome tasks."),
    strict: bool = typer.Option(False, help="Treat warnings as errors"),
    model: str | None = typer.Option(None, help="Filter by model"),
    harness: str | None = typer.Option(None, help="Filter by harness"),
    task_family: str | None = typer.Option(None, "--task-family", help="Filter by task family"),
) -> None:
    """Output structured pass/fail comparison for LLM analysis.

    Generates a rich comparison document showing where a passing and failing run
    diverge, including reasoning, edit diffs, and tool outputs. Designed to be
    consumed by an LLM (Claude Code, Codex, etc.) for analysis.

    Usage:
        moirai divergence runs/ --task ansible_ansible-85709
        moirai divergence runs/ --task ansible_ansible-85709 | pbcopy
    """
    runs = _load_and_filter(path, strict, model=model, harness=harness, task_family=task_family)

    from collections import defaultdict
    from moirai.analyze.explain import explain_task

    # Group by task
    by_task: dict[str, list] = defaultdict(list)
    for r in runs:
        by_task[r.task_id].append(r)

    # Filter to mixed-outcome tasks
    mixed = {tid: task_runs for tid, task_runs in by_task.items()
             if any(r.result.success for r in task_runs) and any(not r.result.success for r in task_runs)}

    if not mixed:
        err_console.print("[yellow]No tasks with mixed outcomes found.[/yellow]")
        raise typer.Exit(0)

    if task:
        # Match by prefix
        matches = {tid: r for tid, r in mixed.items() if tid.startswith(task)}
        if not matches:
            err_console.print(f"[red]error:[/red] no mixed-outcome task matching '{task}'")
            err_console.print("available tasks with mixed outcomes:")
            for tid, task_runs in sorted(mixed.items()):
                n_p = sum(1 for r in task_runs if r.result.success)
                n_f = len(task_runs) - n_p
                err_console.print(f"  {tid} ({n_p}P/{n_f}F)")
            raise typer.Exit(1)
        targets = matches
    else:
        targets = mixed

    for tid in sorted(targets):
        output = explain_task(tid, targets[tid])
        console.print(output)
        if len(targets) > 1:
            console.print("\n" + "=" * 80 + "\n")


def _split_cohorts(
    runs: list[Run],
    baseline_filters: list[str],
    current_filters: list[str],
) -> tuple[list[Run], list[Run], str, str]:
    """Split runs into baseline and current cohorts using K=V filters.

    Returns (baseline_runs, current_runs, baseline_label, current_label).
    Exits with error if either cohort is empty.
    """
    from moirai.filters import apply_kv_filters

    try:
        b_runs = apply_kv_filters(runs, baseline_filters)
        c_runs = apply_kv_filters(runs, current_filters)
    except ValueError as e:
        err_console.print(f"[red]error:[/red] invalid filter: {e}")
        raise typer.Exit(2)

    if not b_runs:
        err_console.print("[red]error:[/red] baseline matched 0 runs")
        _show_available_values(runs, baseline_filters)
        raise typer.Exit(1)
    if not c_runs:
        err_console.print("[red]error:[/red] current matched 0 runs")
        _show_available_values(runs, current_filters)
        raise typer.Exit(1)

    return b_runs, c_runs, " ".join(baseline_filters), " ".join(current_filters)


@app.command()
def evidence(
    path: Path = typer.Argument(..., help="Path to a run file or directory"),
    baseline: list[str] = typer.Option(..., "--baseline", help="Baseline filter (K=V, repeatable)"),
    current: list[str] = typer.Option(..., "--current", help="Current filter (K=V, repeatable)"),
    strict: bool = typer.Option(False, help="Treat warnings as errors"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Extract behavioral feature shifts between two variants."""
    runs = _load_and_filter(path, strict)
    b_runs, c_runs, b_label, c_label = _split_cohorts(runs, baseline, current)

    from moirai.analyze.evidence import compare_variants

    result = compare_variants(b_runs, c_runs, baseline_label=b_label, current_label=c_label)

    if json_output:
        import json as json_mod
        from dataclasses import asdict
        console.print(json_mod.dumps(asdict(result), indent=2))
    else:
        from moirai.viz.terminal import print_evidence
        print_evidence(result)


@app.command()
def diagnose(
    path: Path = typer.Argument(..., help="Path to a run file or directory"),
    baseline: list[str] = typer.Option(..., "--baseline", help="Baseline filter (K=V, repeatable)"),
    current: list[str] = typer.Option(..., "--current", help="Current filter (K=V, repeatable)"),
    causes: Path = typer.Option(..., "--causes", help="Path to causes JSON file"),
    bootstrap: int = typer.Option(0, "--bootstrap", help="Number of bootstrap iterations for CIs (0=skip)"),
    strict: bool = typer.Option(False, help="Treat warnings as errors"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Diagnose regression causes from trajectory evidence.

    Compares two variants, extracts behavioral feature shifts, and ranks
    candidate causes by evidence strength. Uses existing tag filters
    (e.g., --baseline variant=baseline --current variant=current).
    """
    runs = _load_and_filter(path, strict)
    b_runs, c_runs, _, _ = _split_cohorts(runs, baseline, current)

    from moirai.analyze.evidence import compare_variants
    from moirai.diagnose.causes import load_causes
    from moirai.diagnose.ranking import bootstrap_confidence, score_causes

    if not causes.exists():
        err_console.print(f"[red]error:[/red] causes file not found: {causes}")
        raise typer.Exit(1)

    candidate_causes = load_causes(causes)
    if not candidate_causes:
        err_console.print("[red]error:[/red] no causes defined in file")
        raise typer.Exit(1)

    if bootstrap > 0:
        result = bootstrap_confidence(b_runs, c_runs, candidate_causes, n_bootstrap=bootstrap)
    else:
        comparison = compare_variants(b_runs, c_runs)
        result = score_causes(comparison, candidate_causes)

    if json_output:
        import json as json_mod
        from dataclasses import asdict
        console.print(json_mod.dumps(asdict(result), indent=2))
    else:
        from moirai.viz.terminal import print_diagnosis
        print_diagnosis(result)


@app.command()
def report(
    path: Path = typer.Argument(..., help="Path to a run file or directory"),
    html: Path = typer.Option(..., help="Output HTML file path"),
    strict: bool = typer.Option(False, help="Treat warnings as errors"),
    model: str | None = typer.Option(None, help="Filter by model"),
    harness: str | None = typer.Option(None, help="Filter by harness"),
    task_family: str | None = typer.Option(None, "--task-family", help="Filter by task family"),
) -> None:
    """Generate a combined HTML report (summary + patterns + branch analysis)."""
    runs = _load_and_filter(path, strict, model=model, harness=harness, task_family=task_family)

    from collections import defaultdict
    from moirai.analyze.summary import summarize_runs
    from moirai.analyze.motifs import find_motifs, find_gapped_motifs
    from moirai.analyze.align import align_runs
    from moirai.analyze.divergence import find_divergence_points

    console.print("[bold]Running summary...[/bold]")
    summary_result = summarize_runs(runs)

    console.print("[bold]Finding patterns...[/bold]")
    motifs, n_tested = find_motifs(runs, min_n=3, max_n=5, min_count=3)
    gapped, gapped_tested = find_gapped_motifs(runs, max_length=3, min_count=3)
    total_tested = n_tested + gapped_tested

    console.print("[bold]Analyzing per-task divergence...[/bold]")
    tasks: dict[str, list] = defaultdict(list)
    for r in runs:
        tasks[r.task_id].append(r)

    mixed_tasks = {tid: trs for tid, trs in tasks.items()
                   if any(r.result.success for r in trs) and any(not r.result.success for r in trs)
                   and len(trs) >= 3}

    task_divergences = []
    for tid in sorted(mixed_tasks, key=lambda t: -len(mixed_tasks[t])):
        task_runs = mixed_tasks[tid]
        alignment = align_runs(task_runs, level="name")
        points, _ = find_divergence_points(alignment, task_runs, min_branch_size=1, q_threshold=0.5)
        task_divergences.append((tid, task_runs, points))

    console.print("[bold]Writing HTML...[/bold]")
    from moirai.viz.html import write_report_html
    out = write_report_html(
        summary=summary_result,
        motifs=motifs,
        gapped_motifs=gapped,
        n_tested=total_tested,
        task_divergences=task_divergences,
        runs=runs,
        path=html,
    )

    n_motifs = len(motifs) + len(gapped)
    console.print(
        f"\n[green]Report written to {out}[/green]\n"
        f"  {summary_result.run_count} runs, "
        f"{n_motifs} significant patterns, "
        f"{len(task_divergences)} mixed-outcome tasks"
    )


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


@app.command()
def export(
    path: Path = typer.Argument(..., help="Path to a run file or directory"),
    format: str = typer.Option("dpo", "--format", "-f", help="Export format (dpo)"),
    output: Path = typer.Option("moirai_export.jsonl", "--output", "-o", help="Output file"),
    strict: bool = typer.Option(False, help="Treat warnings as errors"),
    model: str | None = typer.Option(None, help="Filter by model"),
    harness: str | None = typer.Option(None, help="Filter by harness"),
    task_family: str | None = typer.Option(None, "--task-family", help="Filter by task family"),
    max_pairs_per_task: int = typer.Option(25, "--max-per-task", help="Max pass*fail pairs per task"),
    min_steps: int = typer.Option(3, "--min-steps", help="Minimum steps for a run to be included"),
    min_content: int = typer.Option(50, "--min-content", help="Min chars in both chosen and rejected"),
    max_position: float = typer.Option(0.8, "--max-position", help="Max normalized divergence position (0-1)"),
) -> None:
    """Export analysis results in machine-consumable formats.

    Currently supports DPO preference pair extraction: for each task with
    mixed outcomes, aligns pass/fail trajectory pairs and extracts
    (prompt, chosen, rejected) triples at divergence points.
    """
    import json

    if format != "dpo":
        err_console.print(f"[red]Unknown format: {format}. Supported: dpo[/red]")
        raise typer.Exit(1)

    runs = _load_and_filter(path, strict, model=model, harness=harness, task_family=task_family)

    # Filter to runs with enough steps and known outcomes
    runs = [r for r in runs if len(r.steps) >= min_steps and r.result.success is not None]

    from collections import defaultdict
    tasks: dict[str, list[Run]] = defaultdict(list)
    for r in runs:
        tasks[r.task_id].append(r)

    # Find mixed-outcome tasks
    mixed = {
        tid: trs for tid, trs in tasks.items()
        if any(r.result.success for r in trs) and any(not r.result.success for r in trs)
    }

    console.print(f"[bold]{len(runs)} runs, {len(mixed)} mixed-outcome tasks[/bold]")

    from moirai.analyze.dpo import extract_pairwise_preferences, format_pair_as_dpo

    total_pairs = 0
    total_with_reasoning = 0
    skipped_position = 0
    skipped_content = 0
    skipped_dedup = 0

    with open(output, "w") as fh:
        for tid in sorted(mixed):
            task_runs = mixed[tid]
            pass_runs = [r for r in task_runs if r.result.success]
            fail_runs = [r for r in task_runs if not r.result.success]

            # Cap to avoid combinatorial explosion
            import math
            max_per_side = max(2, int(math.sqrt(max_pairs_per_task)))
            if len(pass_runs) > max_per_side:
                pass_runs = pass_runs[:max_per_side]
            if len(fail_runs) > max_per_side:
                fail_runs = fail_runs[:max_per_side]

            pairs = extract_pairwise_preferences(
                pass_runs + fail_runs, tid,
            )

            # Deduplicate by (task_id, divergence_column) — keep first
            seen_cols: set[int] = set()
            task_exported = 0

            for pair in pairs:
                if task_exported >= max_pairs_per_task:
                    break

                # Skip late-trajectory divergences
                if pair.divergence_position_norm > max_position:
                    skipped_position += 1
                    continue

                # Deduplicate within task
                if pair.divergence_column in seen_cols:
                    skipped_dedup += 1
                    continue

                dpo = format_pair_as_dpo(pair)

                # Filter: both sides need substantive content
                if len(dpo["chosen"]) < min_content or len(dpo["rejected"]) < min_content:
                    skipped_content += 1
                    continue

                seen_cols.add(pair.divergence_column)
                fh.write(json.dumps(dpo) + "\n")
                total_pairs += 1
                task_exported += 1
                if "Thinking:" in dpo["chosen"] or "Thinking:" in dpo["rejected"]:
                    total_with_reasoning += 1

    console.print(
        f"\n[green]Exported {total_pairs} DPO pairs to {output}[/green]\n"
        f"  {total_with_reasoning} pairs with reasoning content "
        f"({total_with_reasoning * 100 // max(total_pairs, 1)}%)\n"
        f"  from {len(mixed)} tasks\n"
        f"  skipped: {skipped_position} late-trajectory, "
        f"{skipped_content} low-content, {skipped_dedup} duplicates"
    )


@app.command()
def divergences(
    path: Path = typer.Argument(..., help="Path to a run file or directory"),
    min_runs: int = typer.Option(4, "--min-runs", help="Min runs per task (need pass+fail)"),
    n_clusters: int | None = typer.Option(None, "--n-clusters", help="Number of clusters (auto if omitted)"),
    max_pairs: int = typer.Option(5000, "--max-pairs", help="Cap on pairs for clustering performance"),
    method: str = typer.Option("tfidf", "--method", help="Clustering method: tfidf or embedding"),
    output: Path | None = typer.Option(None, "--output", "-o", help="JSON output path"),
    strict: bool = typer.Option(False, help="Treat warnings as errors"),
    model: str | None = typer.Option(None, help="Filter by model"),
    harness: str | None = typer.Option(None, help="Filter by harness"),
    task_family: str | None = typer.Option(None, "--task-family", help="Filter by task family"),
) -> None:
    """Cluster divergence points across tasks to find recurring failure modes."""
    runs = _load_and_filter(path, strict, model=model, harness=harness, task_family=task_family)

    from collections import defaultdict
    from moirai.analyze.dpo import extract_pairwise_preferences
    from moirai.analyze.divergence_clusters import cluster_divergences

    # Group by task, keep only tasks with mixed outcomes and enough runs
    tasks: dict[str, list] = defaultdict(list)
    for r in runs:
        if r.result.success is not None:
            tasks[r.task_id].append(r)

    mixed_tasks = {
        tid: trs for tid, trs in tasks.items()
        if (any(r.result.success for r in trs)
            and any(not r.result.success for r in trs)
            and len(trs) >= min_runs)
    }

    if not mixed_tasks:
        err_console.print("[red]error:[/red] no tasks with mixed outcomes and enough runs")
        raise typer.Exit(1)

    # Extract pairwise preference pairs across all tasks
    console.print(f"[bold]Extracting divergence pairs from {len(mixed_tasks)} tasks...[/bold]")
    all_pairs = []
    for tid in sorted(mixed_tasks):
        task_runs = mixed_tasks[tid]
        pairs = extract_pairwise_preferences(task_runs, tid)
        all_pairs.extend(pairs)

    if not all_pairs:
        err_console.print("[red]error:[/red] no divergence pairs extracted")
        raise typer.Exit(1)

    console.print(f"  {len(all_pairs):,} pairs from {len(mixed_tasks):,} tasks")

    # Cluster
    console.print("[bold]Clustering...[/bold]")
    clusters = cluster_divergences(
        all_pairs,
        n_clusters=n_clusters,
        max_pairs=max_pairs,
        method=method,
    )

    if not clusters:
        err_console.print("[yellow]No clusters found (all below min size).[/yellow]")
        raise typer.Exit(0)

    # Terminal output
    from rich.table import Table

    total_clustered = sum(c.size for c in clusters)
    console.print(
        f"\n[bold]Divergence clusters[/bold] — "
        f"{len(mixed_tasks):,} tasks, {len(all_pairs):,} pairs\n"
    )

    table = Table(show_header=True, header_style="bold", pad_edge=False)
    table.add_column("Cluster", min_width=30)
    table.add_column("Size", justify="right")
    table.add_column("%", justify="right")
    table.add_column("Position", justify="right")

    for c in clusters:
        pct = c.size / total_clustered * 100 if total_clustered else 0
        table.add_row(
            c.label,
            str(c.size),
            f"{pct:.0f}%",
            f"{c.mean_position:.2f}",
        )

    console.print(table)

    # JSON output
    if output:
        import json

        envelope = {
            "n_pairs": len(all_pairs),
            "n_tasks": len(mixed_tasks),
            "clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "label": c.label,
                    "size": c.size,
                    "percentage": round(c.size / total_clustered * 100, 1) if total_clustered else 0,
                    "mean_position": round(c.mean_position, 3),
                    "action_summary": c.action_summary,
                    "n_tasks": len(c.task_ids),
                    "preferred_labels": c.preferred_labels,
                    "dispreferred_labels": c.dispreferred_labels,
                }
                for c in clusters
            ],
        }
        with open(output, "w") as f:
            json.dump(envelope, f, indent=2)
        console.print(f"\n[green]JSON written to {output}[/green]")


@app.command()
def features(
    path: Path = typer.Argument(..., help="Path to a run file or directory"),
    min_runs: int = typer.Option(10, "--min-runs", help="Min runs per task to include"),
    output: Path | None = typer.Option(None, "--output", "-o", help="JSON output path"),
    seed: int = typer.Option(42, "--seed", help="Random seed for split-half validation"),
    strict: bool = typer.Option(False, help="Treat warnings as errors"),
    model: str | None = typer.Option(None, help="Filter by model"),
    harness: str | None = typer.Option(None, help="Filter by harness"),
    task_family: str | None = typer.Option(None, "--task-family", help="Filter by task family"),
) -> None:
    """Compute behavioral features and rank by predictive power.

    Runs within-task natural experiments (median-split + sign test) for each
    behavioral feature, validates with split-half, and reports ranked results.
    """
    runs = _load_and_filter(path, strict, model=model, harness=harness, task_family=task_family)

    from moirai.analyze.features import rank_features
    from moirai.viz.terminal import print_features

    results = rank_features(runs, min_runs=min_runs, seed=seed)

    if not results:
        err_console.print("[yellow]No mixed-outcome tasks found with enough runs.[/yellow]")
        raise typer.Exit(2)

    print_features(results, runs)

    if output:
        import json

        n_total_tasks = len({r.task_id for r in runs})
        n_mixed = results[0].n_tasks if results else 0
        envelope = {
            "dataset": str(path),
            "n_runs": len(runs),
            "n_tasks_mixed": n_mixed,
            "n_tasks_total": n_total_tasks,
            "features": [
                {
                    "name": r.name,
                    "description": r.description,
                    "direction": r.direction,
                    "delta_pp": round(r.delta_pp, 2),
                    "p_value": r.p_value,
                    "q_value": r.q_value,
                    "split_half": r.split_half,
                    "pass_mean": round(r.pass_mean, 4),
                    "fail_mean": round(r.fail_mean, 4),
                    "n_tasks": r.n_tasks,
                    "n_runs": r.n_runs,
                }
                for r in results
            ],
        }
        with open(output, "w") as f:
            json.dump(envelope, f, indent=2)
        console.print(f"\n[green]JSON written to {output}[/green]")
