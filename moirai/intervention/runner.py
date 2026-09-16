"""Three-arm randomised runner with fail-closed budget and drift checks."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from moirai.intervention.budget import BudgetExceeded, BudgetGuard
from moirai.intervention.checkpoint import (
    InfrastructureError,
    ManifestDrift,
    RuntimeAdapter,
    Workspace,
    WorkspaceProvider,
    verify_clone,
)
from moirai.intervention.grade import Grader, grade_with_retry
from moirai.intervention.inject import AtMostOnce, build_forced_events, new_tool_call_id, validate_history, validate_prefix_boundary
from moirai.intervention.ledger import Ledger
from moirai.intervention.randomize import AssignmentManifest
from moirai.intervention.schema import (
    Assignment,
    CandidateIntervention,
    CheckpointManifest,
    CostRecord,
    TrialRecord,
    content_hash,
    to_dict,
)


@dataclass
class CheckpointBundle:
    manifest: CheckpointManifest
    prefix_messages: list[dict]
    tools: list[dict]
    base_workspace: Workspace


@dataclass
class RunReport:
    executed: int = 0
    skipped_existing: int = 0
    success: int = 0
    failure: int = 0
    infra_failure: int = 0
    stopped_reason: str | None = None
    spent_usd: float = 0.0
    trial_ids: list[str] = field(default_factory=list)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run_trials(
    manifest: AssignmentManifest,
    candidates: dict[str, CandidateIntervention],
    checkpoints: dict[str, CheckpointBundle],
    runtime: RuntimeAdapter,
    grader: Grader,
    ledger: Ledger,
    budget: BudgetGuard,
    provider: WorkspaceProvider,
    max_trials: int | None = None,
    grader_max_attempts: int = 3,
) -> RunReport:
    report = RunReport()
    once = AtMostOnce()
    for a in sorted(manifest.assignments, key=lambda x: x.order):
        if max_trials is not None and report.executed >= max_trials:
            report.stopped_reason = "max_trials"
            break
        if ledger.has(a.trial_id):
            report.skipped_existing += 1
            continue
        try:
            budget.authorize(1)
        except BudgetExceeded as e:
            report.stopped_reason = f"budget: {e}"
            break
        try:
            rec = _run_one(a, candidates[a.candidate_id], checkpoints[a.checkpoint_id], runtime, grader, provider, once, grader_max_attempts)
        except ManifestDrift as e:
            budget.release()
            report.stopped_reason = f"manifest drift: {e}"
            break
        cost = budget.commit(_usage(rec))
        rec.usage["cost_usd"] = cost
        ledger.append(rec)
        report.executed += 1
        report.trial_ids.append(rec.trial_id)
        report.spent_usd += cost
        if rec.status == "success":
            report.success += 1
        elif rec.status == "failure":
            report.failure += 1
        else:
            report.infra_failure += 1
    return report


def _usage(rec: TrialRecord) -> CostRecord:
    u = rec.usage
    return CostRecord(
        input_tokens=int(u.get("input_tokens", 0)), cached_input_tokens=int(u.get("cached_input_tokens", 0)),
        output_tokens=int(u.get("output_tokens", 0)), requests=int(u.get("requests", 0)),
        wall_seconds=float(u.get("wall_seconds", 0.0)), cost_usd=float(u.get("cost_usd", 0.0)),
    )


def _run_one(
    a: Assignment,
    cand: CandidateIntervention,
    bundle: CheckpointBundle,
    runtime: RuntimeAdapter,
    grader: Grader,
    provider: WorkspaceProvider,
    once: AtMostOnce,
    grader_max_attempts: int,
) -> TrialRecord:
    m = bundle.manifest
    started = _now()
    base = dict(
        trial_id=a.trial_id, task_id=a.task_id, checkpoint_id=a.checkpoint_id, candidate_id=a.candidate_id,
        selector=a.selector, arm=a.arm, block=a.block, seed=a.seed,
        model_config_hash=content_hash(m.model_config), prompt_hash=m.prefix_hash, tools_hash=m.tools_hash,
        environment_hash=content_hash(m.environment), workspace_hash_before=m.workspace_hash,
        workspace_hash_after=None, injected_tool_call_id=None, injected_payload_hash=None,
        trajectory_ref=None, retry_of=None, started_at=started,
    )
    usage = CostRecord()

    boundary_errors = validate_prefix_boundary(bundle.prefix_messages)
    if boundary_errors:
        raise ManifestDrift("; ".join(boundary_errors))

    # Provision: an independent clone per trial, verified against the manifest.
    try:
        clone = provider.clone(bundle.base_workspace)
    except OSError as e:
        return TrialRecord(**base, status="infra_failure", outcome=None, failure_class="infra:provision", grade=None,
                           usage=to_dict(usage), finished_at=_now(), infra={"error": str(e)})
    drift = verify_clone(m, clone, bundle.prefix_messages, bundle.tools, m.model_config)
    if drift:
        raise ManifestDrift(f"trial {a.trial_id}: " + "; ".join(drift))

    try:
        ctx = runtime.restore(m, bundle.prefix_messages, clone, a.trial_id, a.seed)
    except InfrastructureError as e:
        return TrialRecord(**base, status="infra_failure", outcome=None, failure_class="infra:restore", grade=None,
                           usage=to_dict(usage), finished_at=_now(), infra={"error": str(e), "workspace": str(clone.path)})

    injected_id = None
    payload_hash = None
    if a.arm != "native":
        action = cand.favored if a.arm == "favored" else cand.disfavored
        if action is None:
            return TrialRecord(**base, status="failure", outcome=False, failure_class="invalid_action:no_comparator",
                               grade=None, usage=to_dict(usage), finished_at=_now())
        injected_id = new_tool_call_id(a.trial_id, action.payload_hash)
        payload_hash = action.payload_hash
        once.claim(a.trial_id, injected_id)
        try:
            observation, act_usage = runtime.execute_action(ctx, action, injected_id)
        except InfrastructureError as e:
            return TrialRecord(**{**base, "injected_tool_call_id": injected_id, "injected_payload_hash": payload_hash},
                               status="infra_failure", outcome=None, failure_class="infra:execute_action", grade=None,
                               usage=to_dict(usage), finished_at=_now(), infra={"error": str(e), "workspace": str(clone.path)})
        usage = usage + act_usage
        assistant_ev, tool_ev = build_forced_events(action, injected_id, observation)
        ctx.messages.extend([assistant_ev, tool_ev])
        ctx.injected_tool_call_ids.append(injected_id)
        errs = validate_history(ctx.messages)
        if errs:
            raise ManifestDrift(f"trial {a.trial_id}: history protocol errors: {errs}")
        # Charge the forced step against the shared budget; no reset.
        ctx.budget = type(ctx.budget)(
            remaining_iterations=ctx.budget.remaining_iterations - 1,
            remaining_cost_usd=ctx.budget.remaining_cost_usd,
            iteration_at_checkpoint=ctx.budget.iteration_at_checkpoint + 1,
        )

    base.update(injected_tool_call_id=injected_id, injected_payload_hash=payload_hash)

    try:
        cont = runtime.continue_(ctx)
    except InfrastructureError as e:
        return TrialRecord(**base, status="infra_failure", outcome=None, failure_class="infra:continue", grade=None,
                           usage=to_dict(usage), finished_at=_now(), infra={"error": str(e), "workspace": str(clone.path)})
    usage = usage + cont.usage
    base["trajectory_ref"] = cont.trajectory_ref
    base["workspace_hash_after"] = clone.hash()

    if cont.status != "completed":
        # Agent-side failures are outcomes under the frozen policy.
        return TrialRecord(**base, status="failure", outcome=False, failure_class=cont.status, grade=None,
                           usage=to_dict(usage), finished_at=_now(), infra={"detail": cont.detail})

    g = grade_with_retry(grader, clone, a.task_id, max_attempts=grader_max_attempts)
    if g.status != "ok":
        return TrialRecord(**base, status="infra_failure", outcome=None, failure_class="infra:grader", grade=to_dict(g),
                           usage=to_dict(usage), finished_at=_now(), infra={"workspace": str(clone.path)})
    return TrialRecord(**base, status="success" if g.passed else "failure", outcome=bool(g.passed),
                       failure_class=None if g.passed else "graded_fail", grade=to_dict(g),
                       usage=to_dict(usage), finished_at=_now())
