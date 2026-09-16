"""Map normalized moirai steps back to raw trajectory events, failing closed.

A converted run is a lossy view (system/user messages dropped, arguments and
observations truncated). Resolution re-runs the pinned converter on the raw
messages with origin tracking and requires the regenerated steps to match the
stored run field-for-field on every retained field. Any discrepancy fails the
whole run: partial provenance is not usable for replay.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

from moirai.converters import swe_rebench as converter
from moirai.converters.swe_rebench import (
    ACTION_TRUNCATION,
    REASONING_TRUNCATION,
    RESULT_TRUNCATION,
    StepOrigin,
    parse_openhands_trajectory,
)
from moirai.intervention.schema import (
    ProvenanceMap,
    RawEvent,
    RawTrajectory,
    StepProvenance,
    content_hash,
)
from moirai.schema import Run


def converter_hash() -> str:
    """Hash of the converter source so provenance records pin the mapping code."""
    src = inspect.getsource(converter)
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def truncated_fields(step_dict: dict) -> tuple[str, ...]:
    """Fields whose stored length equals the converter cap (possibly truncated)."""
    out = []
    o = step_dict.get("output", {}) or {}
    if len(o.get("action", "") or "") >= ACTION_TRUNCATION:
        out.append("action")
    if len(o.get("result", "") or "") >= RESULT_TRUNCATION:
        out.append("result")
    if len(o.get("reasoning", "") or "") >= REASONING_TRUNCATION:
        out.append("reasoning")
    a = step_dict.get("attrs", {}) or {}
    if len(a.get("command", "") or "") >= converter.COMMAND_TRUNCATION:
        out.append("command")
    return tuple(out)


def _step_view(step) -> dict:
    """Comparable projection of a normalized Step."""
    return {
        "idx": step.idx,
        "type": step.type,
        "name": step.name,
        "status": step.status,
        "output": dict(step.output),
        "attrs": dict(step.attrs),
    }


def _regen_view(d: dict) -> dict:
    return {
        "idx": d["idx"],
        "type": d["type"],
        "name": d["name"],
        "status": d["status"],
        "output": dict(d["output"]),
        "attrs": dict(d["attrs"]),
    }


def resolve_provenance(run: Run, traj: RawTrajectory, events: list[RawEvent] | None = None) -> ProvenanceMap:
    """Resolve every step of ``run`` to a raw event; fail closed on any mismatch."""
    reasons: list[str] = []
    if run.run_id != traj.trajectory_id:
        reasons.append(f"run_id {run.run_id} != trajectory_id {traj.trajectory_id}")
    if run.task_id != traj.instance_id:
        reasons.append(f"task_id {run.task_id} != instance_id {traj.instance_id}")

    origins: list[StepOrigin] = []
    regenerated = parse_openhands_trajectory(traj.messages, origins)

    if len(regenerated) != len(run.steps):
        reasons.append(f"step count mismatch: stored {len(run.steps)}, regenerated {len(regenerated)}")
    if len(regenerated) != len(origins):
        reasons.append("internal: origins/steps length mismatch")

    if reasons:
        return ProvenanceMap(run.run_id, traj.trajectory_id, "failed", [], reasons, converter_hash())

    by_id: dict[str, RawEvent] = {}
    if events is not None:
        by_id = {e.tool_call_id: e for e in events}

    entries: list[StepProvenance] = []
    for step, regen, origin in zip(run.steps, regenerated, origins):
        sv, rv = _step_view(step), _regen_view(regen)
        if sv != rv:
            diff = sorted(k for k in set(sv) | set(rv) if sv.get(k) != rv.get(k))
            reasons.append(f"step {step.idx}: stored != regenerated on {diff}")
            continue
        if origin.observation_misassociated:
            reasons.append(
                f"step {step.idx}: observation {origin.observation_tool_call_id} "
                f"attached to call {origin.tool_call_id}"
            )
            continue
        if origin.tool_call_id is not None and by_id and origin.tool_call_id not in by_id:
            reasons.append(f"step {step.idx}: tool_call_id {origin.tool_call_id} not in raw event index")
            continue
        args_hash = None
        if origin.tool_call_id is not None and origin.tool_call_id in by_id:
            args_hash = content_hash(by_id[origin.tool_call_id].call.arguments_json)
        entries.append(StepProvenance(
            step_idx=step.idx,
            message_idx=origin.message_idx,
            call_idx=origin.call_idx,
            tool_call_id=origin.tool_call_id,
            tool_name=origin.tool_name,
            arguments_hash=args_hash,
            observation_message_idx=origin.observation_message_idx,
            truncated_fields=truncated_fields(regen),
        ))

    if reasons:
        return ProvenanceMap(run.run_id, traj.trajectory_id, "failed", [], reasons, converter_hash())
    return ProvenanceMap(run.run_id, traj.trajectory_id, "resolved", entries, [], converter_hash())


def resolve_all(runs: list[Run], raw: dict[str, RawTrajectory]) -> dict[str, ProvenanceMap]:
    from moirai.intervention.raw import index_events

    out: dict[str, ProvenanceMap] = {}
    for run in runs:
        traj = raw.get(run.run_id)
        if traj is None:
            out[run.run_id] = ProvenanceMap(run.run_id, "", "failed", [], ["raw trajectory not available"], converter_hash())
            continue
        events, _ = index_events(traj)
        out[run.run_id] = resolve_provenance(run, traj, events)
    return out


def write_manifest(path: Path, maps: dict[str, ProvenanceMap]) -> None:
    import json
    from dataclasses import asdict

    payload = {
        "converter_hash": converter_hash(),
        "runs": {rid: asdict(m) for rid, m in sorted(maps.items())},
    }
    Path(path).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
