"""``moirai intervention`` command group."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(name="intervention", help="Causal intervention study harness (engineering pilot).", add_completion=False)
console = Console()
err = Console(stderr=True)


@app.callback(invoke_without_command=True)
def _main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit(0)


@app.command("audit-raw")
def audit_raw(raw_dir: Path = typer.Argument(..., help="Directory of raw SWE-rebench rows (*.json)"),
              output: Path | None = typer.Option(None, "--output", "-o")) -> None:
    """Ingestion audit: multi-call messages, orphan/missing observations, duplicate ids."""
    from moirai.intervention.raw import index_events, load_raw_dir

    raw = load_raw_dir(raw_dir)
    audits = [index_events(t)[1] for t in raw.values()]
    tot = Counter()
    for a in audits:
        tot.update({k: v for k, v in asdict(a).items() if isinstance(v, int)})
    console.print(f"{len(audits)} trajectories; totals: {dict(tot)}")
    if output:
        output.write_text(json.dumps([asdict(a) for a in audits], indent=1) + "\n")


@app.command()
def provenance(runs_path: Path, raw_dir: Path, output: Path = typer.Option(Path("provenance-manifest.json"), "--output", "-o")) -> None:
    """Map converted runs to raw events; fail closed per run on any mismatch."""
    from moirai.intervention.provenance import resolve_all, write_manifest
    from moirai.intervention.raw import load_raw_dir
    from moirai.load import load_runs

    runs, _ = load_runs(runs_path)
    raw = load_raw_dir(raw_dir)
    maps = resolve_all(runs, raw)
    write_manifest(output, maps)
    ok = sum(1 for m in maps.values() if m.status == "resolved")
    console.print(f"{ok}/{len(maps)} runs resolved losslessly -> {output}")
    for rid, m in maps.items():
        if m.status != "resolved":
            err.print(f"  [yellow]{rid}[/yellow]: {m.reasons[0] if m.reasons else 'failed'}")


@app.command()
def eligibility(raw_dir: Path,
                ledger: Path = typer.Option(Path("eligibility-ledger.jsonl"), "--ledger"),
                report: Path = typer.Option(Path("replay-gap-report.md"), "--report"),
                policy: str = typer.Option("non_mutating_v1", "--policy", help="non_mutating_v1 | verified_replay_v2"),
                replay_dir: list[Path] = typer.Option([], "--replay-dir", help="Replay report directories (policy v2 certificates)"),
                late_from: int = typer.Option(10, "--late-from", help="Message index from which an anchor counts as past the opening exploration")) -> None:
    """Screen every raw tool call under a frozen policy; write ledger and replay-gap report."""
    from moirai.intervention.eligibility import POLICY_V2, POLICY_VERSION, load_certificates, screen_events, screen_events_v2
    from moirai.intervention.raw import index_events, load_raw_dir

    if policy not in (POLICY_VERSION, POLICY_V2):
        err.print(f"[red]unknown policy {policy}[/red]")
        raise typer.Exit(2)
    raw = load_raw_dir(raw_dir)
    certs = load_certificates(replay_dir) if policy == POLICY_V2 else {}
    if policy == POLICY_V2 and not certs:
        err.print("[yellow]policy v2 with no replay certificates: nothing can enrol[/yellow]")
    rows = []
    for t in raw.values():
        ev, _ = index_events(t)
        if policy == POLICY_V2:
            rows.extend(screen_events_v2(t.trajectory_id, t.instance_id, ev, certs.get(t.trajectory_id)))
        else:
            rows.extend(screen_events(t.trajectory_id, t.instance_id, ev))
    with open(ledger, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), sort_keys=True) + "\n")
    reasons = Counter(r.exclusion_reason or "ENROLLED" for r in rows)
    per_traj = Counter(r.run_id for r in rows if r.enrolled)
    late = sum(1 for r in rows if r.enrolled and r.message_idx >= late_from)
    lines = [f"# Replay gap report ({policy})", "",
             f"Trajectories: {len(raw)}. Tool calls screened: {len(rows)}. Enrolled: {reasons['ENROLLED']} "
             f"(late, message index >= {late_from}: {late}).", "",
             "| Reason | Count |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in reasons.most_common()]
    if policy == POLICY_V2:
        lines += ["", "Replay certificates:", "", "| Trajectory | Baseline | Replayed through | First rejected | Certified through | Complete |",
                  "|---|---|---:|---:|---:|---|"]
        for t in raw.values():
            c = certs.get(t.trajectory_id)
            if c is None:
                lines.append(f"| {t.trajectory_id} | none | | | | |")
            else:
                lines.append(f"| {t.trajectory_id} | {'ok' if c.baseline_ok else 'FAIL'} | {c.replayed_through} | "
                             f"{'' if c.first_rejected_message_idx is None else c.first_rejected_message_idx} | "
                             f"{'' if c.certified_through is None else c.certified_through} | {'yes' if c.complete else 'no'} |")
    lines += ["", "Enrolled anchors per trajectory:", ""]
    lines += [f"- {t.trajectory_id} ({t.instance_id}): {per_traj.get(t.trajectory_id, 0)}" for t in raw.values()]
    if policy == POLICY_V2:
        lines += ["", "`prefix:*` rows fail the static screen (stdin to a running process, network commands, tools the",
                  "replay does not emulate). `replay:*` rows have an admissible prefix that no stored replay certified."]
    else:
        lines += ["", "`prefix:*` rows are anchors whose prior history includes actions the replay policy cannot",
                  "reconstruct without live process state. They are excluded, not coerced."]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"{len(rows)} calls screened, {reasons['ENROLLED']} enrolled ({late} late) under {policy} -> {ledger}, {report}")


@app.command()
def select(runs_path: Path, raw_dir: Path,
           selector: str = typer.Option("moirai", "--selector", help="moirai | heuristic | random"),
           seed: int = typer.Option(20260905, "--seed"),
           min_runs: int = typer.Option(4, "--min-runs"),
           policy: str = typer.Option("non_mutating_v1", "--policy", help="non_mutating_v1 | verified_replay_v2"),
           replay_dir: list[Path] = typer.Option([], "--replay-dir", help="Replay report directories (policy v2 certificates)"),
           output: Path = typer.Option(Path("candidates.json"), "--output", "-o")) -> None:
    """Run one frozen selector over discovery data; emit candidates and exclusions."""
    from moirai.analyze.structure import compute_structure_score
    from moirai.intervention.eligibility import POLICY_V2, load_certificates
    from moirai.intervention.raw import load_raw_dir
    from moirai.intervention.selectors import DiscoveryDataset, TaskSpec, build_selector, select_all
    from moirai.load import load_runs

    runs, _ = load_runs(runs_path)
    raw = load_raw_dir(raw_dir)
    runs = [r for r in runs if r.run_id in raw]
    certs = load_certificates(replay_dir) if policy == POLICY_V2 else {}
    ds = DiscoveryDataset.build(runs, raw, policy=policy, certificates=certs)
    tasks = []
    for tid, trs in sorted(ds.runs.items()):
        known = [r for r in trs if r.result.success is not None]
        score = compute_structure_score(known, tid).composite if len(known) >= min_runs else None
        tasks.append(TaskSpec(tid, score))
    sel = build_selector(selector, **({"seed": seed} if selector == "random" else {}))
    cands, excls = select_all(sel, ds, tasks)
    output.write_text(json.dumps({"selector": sel.name, "version": sel.version, "config": sel.config(), "policy": policy,
                                  "certificates": {k: asdict(v) for k, v in sorted(certs.items())},
                                  "candidates": [asdict(c) for c in cands], "exclusions": [asdict(e) for e in excls]},
                                 indent=1, sort_keys=True) + "\n", encoding="utf-8")
    console.print(f"{sel.name}/{sel.version}: {len(cands)} candidates, {len(excls)} exclusions -> {output}")
    for e in excls:
        err.print(f"  [dim]{e.task_id}: {e.reason} {e.detail}[/dim]")


@app.command("fork-coverage")
def fork_coverage_cmd(runs_path: Path, raw_dir: Path,
                      policy: str = typer.Option("non_mutating_v1", "--policy"),
                      replay_dir: list[Path] = typer.Option([], "--replay-dir"),
                      min_branch_size: int = typer.Option(1, "--min-branch-size"),
                      q_threshold: float = typer.Option(1.0, "--q-threshold", help="1.0 disables the significance filter"),
                      min_support: int = typer.Option(1, "--min-support"),
                      late_from: int = typer.Option(10, "--late-from"),
                      output: Path = typer.Option(Path("fork-coverage.json"), "--output", "-o")) -> None:
    """Diagnostic: outcome-rated forks per task and how many anchors at them the policy admits."""
    from moirai.intervention.eligibility import POLICY_V2, load_certificates
    from moirai.intervention.raw import load_raw_dir
    from moirai.intervention.selectors import DiscoveryDataset, TaskSpec, fork_coverage
    from moirai.load import load_runs

    runs, _ = load_runs(runs_path)
    raw = load_raw_dir(raw_dir)
    runs = [r for r in runs if r.run_id in raw]
    certs = load_certificates(replay_dir) if policy == POLICY_V2 else {}
    ds = DiscoveryDataset.build(runs, raw, policy=policy, certificates=certs)
    results = [fork_coverage(ds, TaskSpec(tid), min_branch_size, q_threshold, min_support, late_from=late_from)
               for tid in sorted(ds.runs)]
    output.write_text(json.dumps({"policy": policy, "min_branch_size": min_branch_size, "q_threshold": q_threshold,
                                  "min_support": min_support, "tasks": results}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    console.print(f"policy={policy}  forks / with admitted anchor / candidate possible / late forks with admitted anchor")
    tot = [0, 0, 0, 0]
    for r in results:
        f = r["forks"]
        row = [len(f), sum(x["n_admitted"] > 0 for x in f), sum(x["candidate_possible"] for x in f),
               sum(x["n_admitted"] > 0 and x["late"] for x in f)]
        tot = [a + b for a, b in zip(tot, row)]
        console.print(f"  {r['task_id']}: {row[0]} / {row[1]} / {row[2]} / {row[3]}" + (f"  ({r['reason']})" if r["reason"] else ""))
    console.print(f"  total: {tot[0]} / {tot[1]} / {tot[2]} / {tot[3]} -> {output}")


@app.command("dry-run")
def dry_run(output_dir: Path = typer.Option(..., "--output-dir"),
            tasks: int = typer.Option(3, "--tasks"), reps: int = typer.Option(3, "--reps"),
            seed: int = typer.Option(20260905, "--seed"),
            null: bool = typer.Option(False, "--null", help="Null fixture: all arms share one success probability")) -> None:
    """Synthetic three-arm pilot proving isolation, protocol and accounting. No model calls."""
    from moirai.intervention.dryrun import run_dry_run

    s = run_dry_run(output_dir, n_tasks=tasks, reps=reps, seed=seed, null=null)
    console.print(json.dumps({k: s[k] for k in ("instrument_checks", "run", "native_untouched", "forced_arms_injected",
                                                 "distinct_workspaces", "at_most_once", "budget")}, indent=1, default=str))
    for r in s["selector_reports"]:
        for key in ("branch", "uplift"):
            c = r[key]
            if c:
                console.print(f"{key}: {c['estimate']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] "
                              f"bounds [{c['bound_low']:+.3f}, {c['bound_high']:+.3f}] missing={c['n_missing']}")
    console.print(f"summary -> {Path(output_dir) / 'dryrun-summary.json'}")


@app.command("convert-raw")
def convert_raw(raw_dir: Path, output_dir: Path) -> None:
    """Convert raw rows (SWE-rebench or CoderForge) to moirai runs with the shared converter."""
    from moirai.converters.swe_rebench import convert_messages
    from moirai.intervention.raw import load_raw_dir

    raw = load_raw_dir(raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for t in raw.values():
        model = "Qwen/Qwen2.5-Coder-32B-Instruct" if t.source == "coderforge" else "Qwen3-Coder-480B"
        run = convert_messages(t.trajectory_id, t.instance_id, t.messages, t.resolved, repo=t.repo,
                               harness=t.source, model=model, tags={"image": t.image, **t.extra})
        (output_dir / f"{t.trajectory_id}.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    console.print(f"{len(raw)} runs -> {output_dir}")


@app.command()
def replay(raw_file: Path,
           message_idx: int | None = typer.Option(None, "--message-idx", help="Anchor boundary; overrides --anchor"),
           anchor: str = typer.Option("auto", "--anchor", help="auto (first eligible action after the first test run) | deep (last eligible, at most 40 prior calls) | both | full (every call; certifies all earlier anchors)"),
           stop_on_reject: bool = typer.Option(False, "--stop-on-reject", help="Stop replaying at the first rejected step"),
           meta: Path | None = typer.Option(None, "--meta", help="instances.json from scripts/fetch_swe_rebench_meta.py (base_commit, test_patch)"),
           apply_test_patch: bool = typer.Option(False, "--apply-test-patch"),
           no_pin: bool = typer.Option(False, "--no-pin", help="Skip baseline pinning (anchor can never be accepted)"),
           image: str | None = typer.Option(None, "--image"),
           snapshot_tag: str | None = typer.Option(None, "--snapshot-tag"),
           network: str = typer.Option("none", "--network", help="Docker network for the replay container; 'none' (default) makes network-dependent steps fail visibly"),
           timeout: float = typer.Option(180.0, "--command-timeout"),
           output_dir: Path | None = typer.Option(None, "--output-dir", help="Write <trajectory>-<anchor>.json here"),
           output: Path | None = typer.Option(None, "--output", "-o")) -> None:
    """Replay a prefix in the task container under baseline pinning; report per-step acceptance."""
    from moirai.intervention.raw import index_events, load_raw_file
    from moirai.intervention.replay import BaselineSpec, choose_deep_anchor, choose_full_anchor, choose_post_test_anchor, replay_prefix

    traj = load_raw_file(raw_file)
    events, _ = index_events(traj)
    anchors: list[tuple[str, int]] = []
    if message_idx is not None:
        anchors = [("manual", message_idx)]
    elif anchor == "full":
        full = choose_full_anchor(events)
        if full is not None:
            anchors = [("full", full)]
    else:
        auto = choose_post_test_anchor(events)
        deep = choose_deep_anchor(events)
        if anchor in ("auto", "both") and auto is not None:
            anchors.append(("auto", auto))
        if anchor in ("deep", "both") and deep is not None and deep != auto:
            anchors.append(("deep", deep))
    if not anchors:
        err.print("[red]no eligible anchor in this trajectory[/red]")
        raise typer.Exit(2)

    spec = None
    if not no_pin:
        m = json.loads(meta.read_text()).get(traj.instance_id, {}) if meta else {}
        if not m.get("base_commit"):
            err.print(f"[yellow]no base_commit for {traj.instance_id}; anchors cannot be accepted[/yellow]")
        spec = BaselineSpec(base_commit=m.get("base_commit"), test_patch=m.get("test_patch"), apply_test_patch=apply_test_patch)

    for kind, idx in anchors:
        rep = replay_prefix(traj, idx, image=image, baseline=spec, snapshot_tag=snapshot_tag, command_timeout=timeout,
                            stop_on_reject=stop_on_reject, network=network)
        target = output if (output and len(anchors) == 1) else ((output_dir or raw_file.parent) / f"{traj.trajectory_id}-{kind}.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rep.to_json() + "\n", encoding="utf-8")
        b = rep.baseline
        console.print(f"{traj.trajectory_id} [{kind}] anchor={idx} baseline_ok={b['ok']} head={str(b['head_after'])[:10]} "
                      f"accepted={rep.anchor_accepted} steps={rep.n_steps} ok={rep.n_accepted} match={rep.n_match} near={rep.n_near} "
                      f"mismatch={rep.n_mismatch} edit_failed={rep.n_edit_failed} skipped={rep.n_skipped} timeout={rep.n_timeout} "
                      f"first_rejected={rep.first_rejected_message_idx} restarts={rep.shell_restarts} {rep.wall_seconds:.0f}s -> {target}")
        for e in b["errors"]:
            err.print(f"  baseline: {e}")
        for s in rep.steps:
            if not s.accepted:
                console.print(f"  [reject] msg {s.message_idx} {s.tool_name}/{s.action_class} {s.status} sim={None if s.similarity is None else round(s.similarity, 3)} {s.reason} {s.detail[:100]}")


@app.command("replay-rejudge")
def replay_rejudge(report_dir: Path, raw_dir: Path = typer.Option(..., "--raw-dir")) -> None:
    """Re-apply the current acceptance rule to stored replay reports (no containers)."""
    from moirai.intervention.raw import index_events, load_raw_dir
    from moirai.intervention.replay import ORDER_INSENSITIVE_CLASSES, instrument_hash, judge_step, similarity_status

    raw = load_raw_dir(raw_dir)
    events_by: dict[str, dict] = {}
    n = 0
    for p in sorted(Path(report_dir).glob("*.json")):
        r = json.loads(p.read_text())
        if "anchor_accepted" not in r:
            continue
        tid = r["trajectory_id"]
        if tid not in events_by:
            ev, _ = index_events(raw[tid])
            events_by[tid] = {e.tool_call_id: e for e in ev}
        for st in r["steps"]:
            ev = events_by[tid].get(st["tool_call_id"])
            args = (ev.call.arguments or {}) if ev else {}
            if st["status"] == "merged":
                st["accepted"], st["reason"] = True, "merged_into_timed_out_command"
                continue
            if st["status"] in ("match", "near", "mismatch"):
                # similarity labels depend on normalisation, which is part of the rule: recompute
                order_free = st["action_class"] in ORDER_INSENSITIVE_CLASSES or (
                    st["tool_name"] == "str_replace_editor" and args.get("command") == "view"
                    and (st["recorded"] or "").startswith("Here's the files"))
                st["status"] = similarity_status(st["recorded"], st["replayed"], order_free)
            st["accepted"], st["reason"] = judge_step(st["action_class"], st["tool_name"], args, st["recorded"], st["replayed"],
                                                       st["status"], st["replayed_exit"], 0.95)
        r["n_accepted"] = sum(st["accepted"] for st in r["steps"])
        r["first_rejected_message_idx"] = next((st["message_idx"] for st in r["steps"] if not st["accepted"]), None)
        r["anchor_accepted"] = bool(r["baseline"]["ok"] and all(st["accepted"] for st in r["steps"]))
        r["judged_with"] = instrument_hash()
        p.write_text(json.dumps(r, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        n += 1
    console.print(f"re-judged {n} reports with rule {instrument_hash()}")


@app.command("replay-summary")
def replay_summary(dirs: list[Path] = typer.Argument(..., help="Directories of replay report JSON files"),
                   output: Path | None = typer.Option(None, "--output", "-o")) -> None:
    """Tabulate replay reports: per-anchor acceptance, rejection reasons, wall time."""
    from collections import Counter

    rows = []
    for d in dirs:
        for p in sorted(Path(d).glob("*.json")):
            r = json.loads(p.read_text())
            if "anchor_accepted" not in r:
                continue
            kind = p.stem.rsplit("-", 1)[-1]
            rejected = [s for s in r["steps"] if not s["accepted"]]
            rows.append({
                "variant": Path(d).name, "task": r["instance_id"], "trajectory": r["trajectory_id"], "anchor": kind,
                "anchor_idx": r["anchor_message_idx"], "baseline_ok": r["baseline"]["ok"], "patched": r["baseline"]["test_patch_applied"],
                "accepted": r["anchor_accepted"], "steps": r["n_steps"], "n_accepted": r["n_accepted"],
                "executed": sum(1 for s in r["steps"] if s["status"] not in ("skipped", "merged")),
                "first_rejected": r["first_rejected_message_idx"],
                "reasons": Counter(s["reason"].split(":")[0] for s in rejected),
                "wall_s": round(r["wall_seconds"]), "restarts": r.get("shell_restarts", 0), "instrument": r.get("instrument_hash", ""),
            })
    for variant in sorted({x["variant"] for x in rows}):
        vr = [x for x in rows if x["variant"] == variant]
        acc = sum(x["accepted"] for x in vr)
        steps_acc = sum(x["n_accepted"] for x in vr); steps_all = sum(x["steps"] for x in vr)
        console.print(f"\n[bold]{variant}[/bold]: {acc}/{len(vr)} anchors accepted; {steps_acc}/{steps_all} steps accepted; "
                      f"baseline ok {sum(x['baseline_ok'] for x in vr)}/{len(vr)}; wall {sum(x['wall_s'] for x in vr)}s")
        console.print(f"  {'task':<28s} {'traj':>5s} {'anch':<5s} {'idx':>4s} {'base':>4s} {'acc':>4s} {'ok/steps':>9s} {'first_rej':>9s} {'wall':>5s}  reasons")
        for x in vr:
            console.print(f"  {x['task'][:28]:<28s} {x['trajectory'].rsplit('_', 1)[-1]:>5s} {x['anchor']:<5s} {x['anchor_idx']:>4d} "
                          f"{'ok' if x['baseline_ok'] else 'NO':>4s} {'yes' if x['accepted'] else 'no':>4s} {x['n_accepted']:>4d}/{x['steps']:<4d} "
                          f"{str(x['first_rejected']):>9s} {x['wall_s']:>5d}  {dict(x['reasons'])}")
        tax = Counter()
        for x in vr:
            tax.update(x["reasons"])
        console.print(f"  rejection taxonomy: {dict(tax.most_common())}")
    if output:
        for x in rows:
            x["reasons"] = dict(x["reasons"])
        output.write_text(json.dumps(rows, indent=1) + "\n")


@app.command()
def pilot(candidates: Path,
          runtime: str = typer.Option("none", "--runtime", help="Runtime adapter. Only 'synthetic' exists; 'none' fails closed."),
          cap_usd: float = typer.Option(200.0, "--cap-usd")) -> None:
    """Capped engineering pilot. Refuses to run without a verified live runtime adapter."""
    if runtime != "synthetic":
        err.print("[red]refusing to run:[/red] no live runtime adapter is implemented for OpenHands/SWE-rebench replay. "
                  "Use `moirai intervention dry-run` for the synthetic instrument test.")
        raise typer.Exit(2)
    err.print("synthetic pilots run through `dry-run`; no paid inference is wired here.")
    raise typer.Exit(2)


@app.command()
def analyze(ledger_path: Path, candidates: Path | None = typer.Option(None, "--candidates"),
            n_boot: int = typer.Option(2000, "--n-boot"), seed: int = typer.Option(0, "--seed"),
            output: Path | None = typer.Option(None, "--output", "-o")) -> None:
    """Fixed-weight contrasts per selector from an immutable ledger."""
    from moirai.intervention.analyze import compare_selectors, selector_report
    from moirai.intervention.ledger import Ledger
    from moirai.intervention.schema import ActionPayload, AnchorRef, CandidateIntervention, CostRecord, Exclusion, from_dict

    led = Ledger(ledger_path)
    recs = led.records()
    cands: list[CandidateIntervention] = []
    excls: list[Exclusion] = []
    if candidates:
        data = json.loads(candidates.read_text())
        for c in data.get("candidates", []):
            c = dict(c)
            c["anchor"] = from_dict(AnchorRef, c["anchor"])
            c["favored"] = from_dict(ActionPayload, c["favored"])
            c["disfavored"] = from_dict(ActionPayload, c["disfavored"]) if c.get("disfavored") else None
            c["selection_cost"] = from_dict(CostRecord, c.get("selection_cost", {}))
            cands.append(from_dict(CandidateIntervention, c))
        for e in data.get("exclusions", []):
            e = dict(e)
            e["selection_cost"] = from_dict(CostRecord, e.get("selection_cost", {}))
            excls.append(from_dict(Exclusion, e))
    selectors = sorted({r.selector for r in recs})
    reports = [selector_report(s, recs, cands, excls, n_boot=n_boot, seed=seed) for s in selectors]
    for r in reports:
        console.print(f"[bold]{r.selector}[/bold] trials={r.n_trials} outcomes={r.n_outcomes} infra={r.n_infra_failures} "
                      f"coverage={r.coverage:.2f} exec=${r.execution_cost_usd:.2f}")
        for key, c in (("branch fav-dis", r.branch), ("uplift fav-native", r.uplift)):
            if c:
                console.print(f"  {key}: {c.estimate:+.3f} [{c.ci_low:+.3f}, {c.ci_high:+.3f}] bounds [{c.bound_low:+.3f}, {c.bound_high:+.3f}] tasks={c.n_tasks}")
    out = {"reports": [asdict(r) for r in reports], "comparisons": [asdict(c) for c in compare_selectors(reports)]}
    if output:
        output.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")


@app.command()
def power(tasks: int = typer.Option(10), checkpoints: int = typer.Option(2), reps: int = typer.Option(5),
          gap: float = typer.Option(0.10), native_offset: float = typer.Option(-0.05),
          missing: float = typer.Option(0.0), sims: int = typer.Option(100), seed: int = typer.Option(20260905)) -> None:
    """Simulate the exact estimator under a hypothetical design (planning aid, not preregistration)."""
    from moirai.intervention.power import Design, simulate

    r = simulate(Design(tasks, checkpoints, reps, gap, native_offset, missing_rate=missing), n_sims=sims, seed=seed)
    console.print(json.dumps(asdict(r), indent=1, default=str))
