"""Synthetic end-to-end pilot: proves isolation, protocol and accounting without a model."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from moirai.intervention.analyze import compare_selectors, neutral_summary, selector_report
from moirai.intervention.budget import BudgetGuard
from moirai.intervention.checkpoint import LocalDirWorkspaceProvider, build_manifest, contamination_probe, verify_independent
from moirai.intervention.grade import FixtureGrader
from moirai.intervention.ledger import Ledger
from moirai.intervention.randomize import assign, balance_report, load_manifest, write_manifest
from moirai.intervention.runner import CheckpointBundle, run_trials
from moirai.intervention.schema import ActionPayload, AnchorRef, CandidateIntervention, content_hash
from moirai.intervention.synthetic import SyntheticRuntime, synthetic_budget

SYNTHETIC_TOOLS = [{"function": {"name": "str_replace_editor"}, "type": "function"},
                   {"function": {"name": "execute_bash"}, "type": "function"}]
SYNTHETIC_MODEL = {"model": "synthetic", "revision": "fixture", "temperature": 0.0}
SYNTHETIC_ENV = {"image": "sha256:fixture", "cwd": "/workspace", "harness": "synthetic"}


def synthetic_prefix(task_id: str) -> list[dict]:
    return [
        {"role": "system", "content": "fixture system prompt", "name": None, "tool_call_id": None, "tool_calls": None, "task_id": task_id},
        {"role": "user", "content": f"task {task_id}", "name": None, "tool_call_id": None, "tool_calls": None},
        {"role": "assistant", "content": "", "name": None, "tool_call_id": None,
         "tool_calls": [{"id": "chatcmpl-tool-prefix0", "type": "function",
                         "function": {"name": "str_replace_editor", "arguments": json.dumps({"command": "view", "path": "/workspace"})}}]},
        {"role": "tool", "content": "/workspace\n/workspace/app.py", "name": "str_replace_editor", "tool_call_id": "chatcmpl-tool-prefix0", "tool_calls": None},
    ]


def synthetic_candidate(task_id: str, selector: str = "moirai", with_disfavored: bool = True) -> CandidateIntervention:
    prefix = synthetic_prefix(task_id)
    anchor = AnchorRef(anchor_run_id=f"{task_id}-anchor", message_idx=len(prefix), step_idx=1, prefix_hash=content_hash(prefix))
    fav = ActionPayload("execute_bash", json.dumps({"command": f"grep -rn bug /workspace/{task_id}"}), "search", f"{task_id}-p1", 4, "chatcmpl-tool-fav")
    dis = ActionPayload("str_replace_editor", json.dumps({"command": "view", "path": "/workspace/README.md"}), "view", f"{task_id}-f1", 4, "chatcmpl-tool-dis") if with_disfavored else None
    cid = content_hash({"task": task_id, "selector": selector, "fav": fav.payload_hash, "dis": dis.payload_hash if dis else None})[:16]
    return CandidateIntervention(cid, task_id, selector, "fixture", "fixture", anchor, fav, dis, {"fixture": True},
                                 structure_score=0.1 * (hash(task_id) % 10))


def run_dry_run(
    out_dir: Path,
    n_tasks: int = 3,
    reps: int = 3,
    seed: int = 20260905,
    null: bool = False,
    cap_usd: float = 200.0,
    p_favored: float = 0.8,
    p_disfavored: float = 0.2,
    p_native: float = 0.5,
    n_boot: int = 500,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    provider = LocalDirWorkspaceProvider(out_dir / "workspaces")

    candidates: dict[str, CandidateIntervention] = {}
    bundles: dict[str, CheckpointBundle] = {}
    checkpoint_ids: dict[str, str] = {}
    effects: dict[str, float] = {}
    for i in range(n_tasks):
        task_id = f"fixture-task-{i}"
        cand = synthetic_candidate(task_id)
        candidates[cand.candidate_id] = cand
        base_dir = out_dir / "workspaces" / "base" / task_id
        base_dir.mkdir(parents=True, exist_ok=True)
        (base_dir / "app.py").write_text(f"# {task_id}\nprint('hello')\n", encoding="utf-8")
        base = provider.register_base(base_dir, f"base-{task_id}")
        prefix = synthetic_prefix(task_id)
        manifest = build_manifest(task_id, cand.anchor, prefix, SYNTHETIC_TOOLS, base, SYNTHETIC_ENV, SYNTHETIC_MODEL,
                                  synthetic_budget(50), "replayable")
        bundles[manifest.checkpoint_id] = CheckpointBundle(manifest, prefix, SYNTHETIC_TOOLS, base)
        checkpoint_ids[cand.candidate_id] = manifest.checkpoint_id
        effects[cand.favored.payload_hash] = p_native if null else p_favored
        if cand.disfavored is not None:
            effects[cand.disfavored.payload_hash] = p_native if null else p_disfavored

    # Instrument checks before randomisation: clone identity, independence, contamination.
    checks: dict[str, object] = {}
    some = next(iter(bundles.values()))
    c1, c2 = provider.clone(some.base_workspace), provider.clone(some.base_workspace)
    checks["clone_identity"] = c1.hash() == c2.hash() == some.manifest.workspace_hash
    checks["clone_independence_problems"] = verify_independent(c1, c2)
    checks["contaminated_by_probe"] = contamination_probe(c1, [c2], some.base_workspace)

    manifest = assign(list(candidates.values()), checkpoint_ids, reps=reps, seed=seed)
    write_manifest(out_dir / "assignments.json", manifest)
    manifest = load_manifest(out_dir / "assignments.json")   # round-trip + hash check
    checks["balance"] = balance_report(manifest)

    trial_ids = [a.trial_id for a in manifest.assignments]
    runtime = SyntheticRuntime(
        payload_effects=effects, native_p=p_native,
        infra_fail_trials={trial_ids[1]} if len(trial_ids) > 1 else set(),
        agent_error_trials={trial_ids[2]} if len(trial_ids) > 2 else set(),
        grader_flaky_trials={trial_ids[3]} if len(trial_ids) > 3 else set(),
    )
    ledger = Ledger(out_dir / "ledger.jsonl")
    budget = BudgetGuard(cap_usd=cap_usd, spent_usd=ledger.spent_usd())
    report = run_trials(manifest, candidates, bundles, runtime, FixtureGrader(), ledger, budget, provider)
    ledger.verify()

    # Hand-audit facts.
    recs = ledger.records()
    native_untouched = all(r.injected_tool_call_id is None for r in recs if r.arm == "native")
    forced_injected = all(r.injected_tool_call_id is not None for r in recs if r.arm != "native" and r.status != "infra_failure" or (r.arm != "native" and r.failure_class == "infra:continue"))
    distinct_ws = len({str(w.path) for w in provider.clones}) == len(provider.clones)
    executed_once = len(runtime.executed) == len(set(runtime.executed))
    regenerated = Ledger(out_dir / "ledger.jsonl").summary()

    sel_reports = [selector_report("moirai", recs, list(candidates.values()), [], n_boot=n_boot, seed=seed)]
    summary = {
        "n_tasks": n_tasks, "reps": reps, "seed": seed, "null_fixture": null,
        "instrument_checks": checks,
        "run": asdict(report),
        "native_untouched": native_untouched,
        "forced_arms_injected": forced_injected,
        "distinct_workspaces": distinct_ws,
        "at_most_once": executed_once,
        "ledger_regenerated": regenerated,
        "neutral_summary": neutral_summary(recs),
        "budget": {"cap_usd": cap_usd, "spent_usd": budget.spent_usd, "reserve_per_trial_usd": budget.reserve_per_trial_usd},
        "selector_reports": [asdict(r) for r in sel_reports],
        "comparisons": [asdict(c) for c in compare_selectors(sel_reports)],
    }
    (out_dir / "dryrun-summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return summary
