#!/usr/bin/env python3
"""Analyze instrumented trial JSONs from eval-harness reporter.write_trial.

Usage: python3 scripts/analyze_harness_throughput.py <results_dir> \\
           --nominal-parallel 8 --timeout 3600 [--run-log PATH]

Accepts the results dir (with trials/ subdir) or trials/ itself. Fails fast
if started_at/finished_at/cost_usd are absent (pre-patch data).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REQUIRED = ("started_at", "finished_at", "cost_usd")

def find_trials(path: Path) -> list[Path]:
    d = path / "trials" if (path / "trials").is_dir() else path
    if not d.is_dir():
        sys.exit(f"error: {d} is not a directory")
    files = sorted(d.glob("*.json"))
    if not files:
        sys.exit(f"error: no trial JSONs found under {d}")
    return files

def load_trials(files: list[Path]) -> list[dict]:
    out = []
    for f in files:
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            sys.exit(f"error: {f} is not valid JSON: {e}")
        missing = [k for k in REQUIRED if k not in data]
        if missing:
            sys.exit(
                f"error: {f.name} missing fields {missing}. This data predates "
                f"the throughput-instrumentation patch; rerun against newer trials."
            )
        out.append(data)
    return out

def percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)

def fmt_p(label: str, values: list[float], fmt: str = "{:.2f}") -> str:
    if not values:
        return f"  {label}: (no samples)"
    return (
        f"  {label}: p50={fmt.format(percentile(values, 50))} "
        f"p95={fmt.format(percentile(values, 95))} "
        f"p99={fmt.format(percentile(values, 99))} (n={len(values)})"
    )

def report_wall_clock(trials: list[dict], timeout: float) -> None:
    print("## Wall-clock tail (seconds)")
    print(fmt_p("success", [t["wall_clock_seconds"] for t in trials if t["success"]], "{:.1f}"))
    print(fmt_p("failure", [t["wall_clock_seconds"] for t in trials if not t["success"]], "{:.1f}"))
    pinned = sum(1 for t in trials if t["wall_clock_seconds"] >= timeout - 1)
    if pinned:
        print(f"  ! {pinned}/{len(trials)} pinned at timeout ({timeout}s) — right-censored")
    print()

def concurrency_timeline(trials: list[dict]) -> tuple[list[int], float]:
    events = []
    for t in trials:
        events.append((datetime.fromisoformat(t["started_at"]), +1))
        events.append((datetime.fromisoformat(t["finished_at"]), -1))
    events.sort()
    origin, end = events[0][0], events[-1][0]
    duration = max(1.0, (end - origin).total_seconds())
    bins = int(duration) + 1
    active = [0] * bins
    count, idx = 0, 0
    for sec in range(bins):
        while idx < len(events) and (events[idx][0] - origin).total_seconds() <= sec:
            count += events[idx][1]
            idx += 1
        active[sec] = count
    return active, duration

def report_concurrency(trials: list[dict], nominal: int) -> None:
    print("## Concurrency-by-second")
    active, duration = concurrency_timeline(trials)
    samples = [float(x) for x in active]
    p50, p95, peak = percentile(samples, 50), percentile(samples, 95), max(active)
    at_full = sum(1 for x in active if x >= nominal) / len(active)
    print(f"  nominal --parallel: {nominal}")
    print(f"  effective p50: {p50:.1f}    p95: {p95:.1f}    peak: {peak}")
    print(f"  fraction at full parallelism (>= {nominal}): {at_full:.1%}")
    print(f"  wall duration: {duration:.0f}s across {len(trials)} trials")
    rows = min(peak + 1, 12)
    hist = [0] * rows
    for v in active:
        hist[min(v, rows - 1)] += 1
    scale = max(hist) or 1
    print("  active-count distribution (seconds at each level):")
    for level in range(rows - 1, -1, -1):
        bar = "#" * int(hist[level] / scale * 40)
        prefix = f">={level}" if level == rows - 1 else f" {level} "
        print(f"    {prefix:>4} | {bar} {hist[level]}")
    print()

def report_downshifts(results_dir: Path, run_log: Path | None) -> None:
    """Scan run log for cli.py:904's 'Reducing parallelism' events so the
    concurrency chart isn't silently dragged down by unseen downshifts."""
    print("## Parallelism downshift events")
    logs = [run_log] if (run_log and run_log.is_file()) else (
        sorted(results_dir.glob("*.log")) if results_dir.is_dir() else []
    )
    if not logs:
        print("  (no run log; pass --run-log to scan for downshifts)\n")
        return
    total = 0
    for log in logs:
        try:
            hits = [l for l in log.read_text(errors="replace").splitlines()
                    if "Reducing parallelism" in l]
        except OSError as e:
            print(f"  ! could not read {log}: {e}")
            continue
        if hits:
            print(f"  {log.name}: {len(hits)} downshift event(s)")
            for h in hits:
                print(f"    {h.strip()}")
            total += len(hits)
    if total == 0:
        print("  no downshift events found — concurrency reflects nominal target")
    print()

def report_cost(trials: list[dict]) -> None:
    print("## Cost-vs-token ratio (cost_usd / input_tokens)")
    ratios = [t["cost_usd"] / t["input_tokens"] for t in trials
              if t["input_tokens"] > 0 and t["cost_usd"] > 0]
    if not ratios:
        print("  (no trials with positive cost and input tokens)\n")
        return
    print(fmt_p("ratio", ratios, "{:.3e}"))
    p50, p99 = percentile(ratios, 50), percentile(ratios, 99)
    if p50 > 0:
        verdict = ("cache reads likely dominate at tail (cache-bloat consistent)"
                   if (p50 - p99) / p50 > 0.3
                   else "tail ratio comparable to median (cache bloat not evident)")
        print(f"  p99/p50 ratio: {p99 / p50:.2f}  →  {verdict}")
    print()

def report_rates(trials: list[dict]) -> None:
    print("## Per-trial rate distributions")
    live = [t for t in trials if t["wall_clock_seconds"] > 0]
    tok = [(t["input_tokens"] + t["output_tokens"]) / t["wall_clock_seconds"] for t in live]
    tc = [t["tool_calls"] / t["wall_clock_seconds"] for t in live]
    print(fmt_p("tokens/sec", tok, "{:.1f}"))
    print(fmt_p("tool_calls/sec", tc, "{:.3f}"))
    print()

def _iso_delta(a: str | None, b: str | None) -> float | None:
    if not a or not b:
        return None
    return (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds()

def report_docker(trials: list[dict]) -> None:
    """Per-Docker-invocation queue/exec split.

    queue_wait = invoked_at → first_byte_at = Docker daemon queue +
                 image pull + container create.
    exec_time  = first_byte_at → finished_at = actual command runtime.

    A growing queue_wait under load corroborates the Docker-as-serializer
    hypothesis. exec_time variance reflects the work itself.
    """
    print("## Docker invocation timings")
    invocations = [
        (t.get("task_id", "?"), inv)
        for t in trials
        for inv in t.get("docker_invocations") or []
    ]
    if not invocations:
        print("  no docker_invocations field — data predates the docker-instrumentation patch\n")
        return

    by_phase: dict[str, list[tuple[float | None, float | None]]] = {}
    for _, inv in invocations:
        q = _iso_delta(inv.get("invoked_at"), inv.get("first_byte_at"))
        e = _iso_delta(inv.get("first_byte_at") or inv.get("invoked_at"),
                       inv.get("finished_at"))
        by_phase.setdefault(inv.get("phase", "?"), []).append((q, e))

    counts = [len(t.get("docker_invocations") or []) for t in trials]
    print(f"  invocations across {len(trials)} trials: total={len(invocations)}, "
          f"per-trial p50={percentile([float(c) for c in counts], 50):.0f}, "
          f"max={max(counts) if counts else 0}")
    print(f"  phases observed: {', '.join(sorted(by_phase))}")
    print()
    print(f"  {'phase':<22} {'n':>4} {'queue_wait p50':>15} {'p95':>7} "
          f"{'exec p50':>10} {'p95':>8}")
    for phase in sorted(by_phase):
        pairs = by_phase[phase]
        qs = [q for q, _ in pairs if q is not None]
        es = [e for _, e in pairs if e is not None]
        qp50 = f"{percentile(qs, 50):.2f}s" if qs else "—"
        qp95 = f"{percentile(qs, 95):.2f}s" if qs else "—"
        ep50 = f"{percentile(es, 50):.2f}s" if es else "—"
        ep95 = f"{percentile(es, 95):.2f}s" if es else "—"
        print(f"  {phase:<22} {len(pairs):>4} {qp50:>15} {qp95:>7} {ep50:>10} {ep95:>8}")

    # Headline: how much of trial wrapper time was Docker
    docker_per_trial = []
    overhead_share = []
    for t in trials:
        invs = t.get("docker_invocations") or []
        if not invs:
            continue
        total_docker = 0.0
        for inv in invs:
            d = _iso_delta(inv.get("invoked_at"), inv.get("finished_at"))
            if d:
                total_docker += d
        wrap = _iso_delta(t.get("started_at"), t.get("finished_at"))
        if wrap and wrap > 0:
            docker_per_trial.append(total_docker)
            overhead_share.append(total_docker / wrap)
    if docker_per_trial:
        print()
        print(fmt_p("docker total per trial (s)", docker_per_trial, "{:.1f}"))
        print(fmt_p("docker share of wrapper", [s * 100 for s in overhead_share], "{:.1f}%"))
    print()

def report_stream_events(trials: list[dict]) -> None:
    """Per-stream-event timing decomposition.

    Splits each trial's Claude wall-clock into:
      user→assistant gaps  — Claude is generating the next response
      assistant→user gaps  — a tool call is running in-container

    If "tool execution" dominates, the bottleneck is local (Bash, Docker exec,
    Read/Edit — all things we can profile). If "LLM generation" dominates, the
    bottleneck is the API call itself, and there's not much the harness can
    do beyond switching models or increasing parallelism.
    """
    print("## Stream-event timing decomposition")
    trials_with = [t for t in trials if t.get("stream_events")]
    if not trials_with:
        print("  no stream_events field — predates stream-event instrumentation\n")
        return

    ttfe: list[float] = []          # arrival of first event
    ttfu: list[float] = []          # arrival of first tool_use
    n_events: list[float] = []
    user_to_asst: list[float] = []  # per-trial sum, Claude generating
    asst_to_user: list[float] = []  # per-trial sum, tools running
    turns_per_trial: list[float] = []

    for t in trials_with:
        evs = t["stream_events"]
        if not evs:
            continue
        ttfe.append(evs[0]["t"])
        n_events.append(float(len(evs)))
        for ev in evs:
            if ev.get("tools"):
                ttfu.append(ev["t"])
                break
        u2a = a2u = 0.0
        n_turns = 0
        for prev, curr in zip(evs, evs[1:]):
            gap = curr["t"] - prev["t"]
            ptype, ctype = prev.get("type"), curr.get("type")
            if ptype == "assistant" and ctype == "user":
                a2u += gap
            elif ptype == "user" and ctype == "assistant":
                u2a += gap
                n_turns += 1
        if u2a > 0:
            user_to_asst.append(u2a)
        if a2u > 0:
            asst_to_user.append(a2u)
        turns_per_trial.append(float(n_turns))

    print(f"  trials with stream_events: {len(trials_with)}/{len(trials)}")
    print(fmt_p("time-to-first-event (s)", ttfe, "{:.2f}"))
    if ttfu:
        print(fmt_p("time-to-first-tool-use (s)", ttfu, "{:.2f}"))
    print(fmt_p("events per trial", n_events, "{:.0f}"))
    print(fmt_p("turns per trial", turns_per_trial, "{:.0f}"))
    print(fmt_p("user→assistant gap sum [Claude generating] (s)", user_to_asst, "{:.1f}"))
    print(fmt_p("assistant→user gap sum [tools running] (s)", asst_to_user, "{:.1f}"))

    if user_to_asst and asst_to_user:
        gens = sum(user_to_asst)
        tools = sum(asst_to_user)
        total = gens + tools
        if total > 0:
            print()
            print(f"  total user→assistant (LLM generation): {gens:.0f}s "
                  f"({gens / total * 100:.0f}%)")
            print(f"  total assistant→user (tool execution): {tools:.0f}s "
                  f"({tools / total * 100:.0f}%)")
    print()


CAVEATS = """## Known limits
  Per-Docker queue_wait is a proxy: invoked_at is the moment we called
  subprocess.Popen, first_byte_at is the first stdout/stderr line from inside
  the container. The gap captures Docker daemon dispatch + image pull +
  container create + bash startup before the command emits anything. Under
  contention this gap should grow with active concurrent invocations; cross-
  reference with the concurrency timeline above to test.
"""

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--nominal-parallel", type=int, required=True)
    ap.add_argument("--timeout", type=float, required=True, help="per-trial timeout seconds")
    ap.add_argument("--run-log", type=Path, default=None,
                    help="console log to scan for 'Reducing parallelism' events")
    args = ap.parse_args()

    trials = load_trials(find_trials(args.results_dir))
    print(f"# Harness throughput analysis — {len(trials)} trials from {args.results_dir}\n")
    n_pass = sum(1 for t in trials if t["success"])
    n_infra = sum(1 for t in trials if t.get("error_class") == "infra")
    print(f"pass: {n_pass}/{len(trials)}   infra errors: {n_infra}\n")

    report_wall_clock(trials, args.timeout)
    report_concurrency(trials, args.nominal_parallel)
    report_downshifts(args.results_dir, args.run_log)
    report_cost(trials)
    report_rates(trials)
    report_docker(trials)
    report_stream_events(trials)
    print(CAVEATS)

if __name__ == "__main__":
    main()
