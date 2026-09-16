"""Pluggable intervention selectors over discovery data only.

Every selector maps (discovery dataset, task) to exactly one
``CandidateIntervention`` or an explicit ``Exclusion``. Selectors never see
confirmatory outcomes: the dataset they receive contains discovery runs only,
and the anchor (the prefix that is cloned into every arm) is chosen by an
outcome-blind rule, the lowest SHA-256 of the run id.

Families implemented here:

- ``moirai``   same-task repeated-run contrast (alignment + divergence points)
- ``heuristic`` frozen behavioural rule, ``search_before_edit`` v0.1
- ``random``   random eligible checkpoint / action contrast under a frozen seed
- ``critic``   adapter for an external critic; only a scripted critic ships
"""
from __future__ import annotations

import hashlib
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

from moirai.analyze.align import align_runs
from moirai.analyze.divergence import _best_worst, find_divergence_points
from moirai.compress import step_enriched_name
from moirai.intervention.eligibility import POLICY_VERSION, anchor_admitted, classify_action
from moirai.intervention.provenance import resolve_provenance
from moirai.intervention.raw import index_events, model_visible_prefix
from moirai.intervention.schema import (
    ActionVerdict,
    ActionPayload,
    AnchorRef,
    CandidateIntervention,
    CostRecord,
    Exclusion,
    ProvenanceMap,
    RawEvent,
    ReplayCertificate,
    RawTrajectory,
    content_hash,
)
from moirai.schema import GAP, Alignment, Run


# ── Inputs ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    structure_score: float | None = None


@dataclass
class DiscoveryDataset:
    runs: dict[str, list[Run]]                         # task_id -> discovery runs
    raw: dict[str, RawTrajectory]                      # run_id -> raw trajectory
    events: dict[str, list[RawEvent]] = field(default_factory=dict)
    provenance: dict[str, ProvenanceMap] = field(default_factory=dict)
    policy: str = POLICY_VERSION
    certificates: dict[str, ReplayCertificate] = field(default_factory=dict)

    @classmethod
    def build(cls, runs: list[Run], raw: dict[str, RawTrajectory], policy: str = POLICY_VERSION,
              certificates: dict[str, ReplayCertificate] | None = None) -> "DiscoveryDataset":
        by_task: dict[str, list[Run]] = {}
        for r in runs:
            by_task.setdefault(r.task_id, []).append(r)
        ds = cls(runs=by_task, raw=raw, policy=policy, certificates=dict(certificates or {}))
        for r in runs:
            traj = raw.get(r.run_id)
            if traj is None:
                ds.provenance[r.run_id] = ProvenanceMap(r.run_id, "", "failed", [], ["raw trajectory not available"])
                continue
            ev, _ = index_events(traj)
            ds.events[r.run_id] = ev
            ds.provenance[r.run_id] = resolve_provenance(r, traj, ev)
        return ds

    def resolved_runs(self, task_id: str) -> list[Run]:
        return [r for r in self.runs.get(task_id, []) if self.provenance.get(r.run_id) and self.provenance[r.run_id].status == "resolved"]

    def event_for_step(self, run_id: str, step_idx: int) -> RawEvent | None:
        pm = self.provenance.get(run_id)
        if pm is None or pm.status != "resolved":
            return None
        entry = pm.for_step(step_idx)
        if entry is None or entry.tool_call_id is None:
            return None
        for ev in self.events.get(run_id, []):
            if ev.tool_call_id == entry.tool_call_id:
                return ev
        return None

    def classify(self, ev: RawEvent) -> ActionVerdict:
        """Action rule under the dataset's frozen policy."""
        return classify_action(ev.call.tool_name, ev.call.arguments_json, self.policy)

    def anchor_admitted(self, run_id: str, message_idx: int) -> bool:
        """Prefix rule under the dataset's frozen policy (v1 static, v2 replay-certified)."""
        return anchor_admitted(self.policy, self.events.get(run_id, []), message_idx, self.certificates.get(run_id))

    def enrolled_events(self, task_id: str) -> list[tuple[str, RawEvent]]:
        """Eligible action + replayable prefix + observation present, all runs."""
        out: list[tuple[str, RawEvent]] = []
        for r in self.resolved_runs(task_id):
            evs = self.events.get(r.run_id, [])
            for ev in evs:
                if ev.observation is None:
                    continue
                v = self.classify(ev)
                if not v.eligible:
                    continue
                if not self.anchor_admitted(r.run_id, ev.call.message_idx):
                    continue
                out.append((r.run_id, ev))
        return out


# ── Selector protocol ─────────────────────────────────────────────


@runtime_checkable
class InterventionSelector(Protocol):
    name: str
    version: str

    def config(self) -> dict: ...

    def select(self, discovery_data: DiscoveryDataset, task: TaskSpec) -> CandidateIntervention | Exclusion: ...


def anchor_rank(run_id: str) -> str:
    """Outcome-blind deterministic ordering of runs."""
    return hashlib.sha256(run_id.encode("utf-8")).hexdigest()


def payload_from_event(run_id: str, ev: RawEvent, policy: str = POLICY_VERSION) -> ActionPayload:
    v = classify_action(ev.call.tool_name, ev.call.arguments_json, policy)
    return ActionPayload(
        tool_name=ev.call.tool_name,
        arguments_json=ev.call.arguments_json,
        action_class=v.action_class,
        source_run_id=run_id,
        source_message_idx=ev.call.message_idx,
        source_tool_call_id=ev.tool_call_id,
    )


def make_anchor(ds: DiscoveryDataset, run_id: str, message_idx: int, step_idx: int | None) -> AnchorRef:
    prefix = model_visible_prefix(ds.raw[run_id], message_idx)
    return AnchorRef(anchor_run_id=run_id, message_idx=message_idx, step_idx=step_idx, prefix_hash=content_hash(prefix))


def _candidate_id(selector: str, version: str, task_id: str, anchor: AnchorRef, fav: ActionPayload, dis: ActionPayload | None) -> str:
    return content_hash({
        "selector": selector, "version": version, "task": task_id,
        "anchor": [anchor.anchor_run_id, anchor.message_idx, anchor.prefix_hash],
        "favored": fav.payload_hash, "disfavored": dis.payload_hash if dis else None,
    })[:16]


def _config_hash(name: str, version: str, cfg: dict) -> str:
    return content_hash({"name": name, "version": version, "config": cfg})[:16]


def _anchor_is_replayable(ds: DiscoveryDataset, run_id: str, message_idx: int) -> bool:
    return ds.anchor_admitted(run_id, message_idx)


# ── Alignment column -> step mapping ─────────────────────────────


def column_to_step_idx(alignment: Alignment, run: Run, column: int) -> int | None:
    """Map an alignment column to the run's step idx, or None on a gap."""
    try:
        row = alignment.matrix[alignment.run_ids.index(run.run_id)]
    except ValueError:
        return None
    if column >= len(row) or row[column] == GAP:
        return None
    k = sum(1 for c in row[:column + 1] if c != GAP)  # 1-based position among named steps
    named = [s for s in run.steps if step_enriched_name(s) is not None]
    if k - 1 >= len(named):
        return None
    return named[k - 1].idx


# ── Moirai selector ───────────────────────────────────────────────


@dataclass
class MoiraiSelector:
    name: str = "moirai"
    version: str = "0.1"
    min_runs: int = 4
    min_branch_size: int = 2
    q_threshold: float = 0.20
    min_support: int = 2

    def config(self) -> dict:
        return {"min_runs": self.min_runs, "min_branch_size": self.min_branch_size,
                "q_threshold": self.q_threshold, "min_support": self.min_support}

    def select(self, ds: DiscoveryDataset, task: TaskSpec) -> CandidateIntervention | Exclusion:
        t0 = time.perf_counter()
        cfg_hash = _config_hash(self.name, self.version, self.config())

        def excl(reason: str, detail: str = "") -> Exclusion:
            return Exclusion(task.task_id, self.name, self.version, reason, detail,
                             CostRecord(wall_seconds=time.perf_counter() - t0))

        runs = [r for r in ds.resolved_runs(task.task_id) if r.result.success is not None]
        if len(runs) < self.min_runs:
            return excl("insufficient_discovery_runs", f"{len(runs)} resolved runs < {self.min_runs}")
        if not any(r.result.success for r in runs) or not any(not r.result.success for r in runs):
            return excl("single_outcome_task")

        alignment = align_runs(runs, level="name")
        if not alignment.matrix or not alignment.matrix[0]:
            return excl("empty_alignment")
        points, n_tested = find_divergence_points(alignment, runs, self.min_branch_size, self.q_threshold)
        if not points:
            return excl("no_divergence_points", f"{n_tested} columns tested")

        points = sorted(points, key=lambda p: (p.column, p.q_value if p.q_value is not None else 1.0))
        why: Counter[str] = Counter()
        for point in points:
            bw = _best_worst(point, self.min_support)
            if bw is None:
                why["no_rated_pair"] += 1
                continue
            best_val, best_rate, best_n, worst_val, worst_rate, worst_n = bw
            if best_val == GAP or worst_val == GAP:
                why["gap_branch"] += 1
                continue

            fav = self._first_eligible(ds, alignment, runs, point.column, best_val)
            dis = self._first_eligible(ds, alignment, runs, point.column, worst_val)
            if fav is None or dis is None:
                why["branch_action_ineligible"] += 1
                continue
            if fav.payload_hash == dis.payload_hash:
                why["identical_payloads"] += 1
                continue

            anchor = self._anchor(ds, alignment, runs, point.column)
            if anchor is None:
                why["no_replayable_anchor"] += 1
                continue

            cand_id = _candidate_id(self.name, self.version, task.task_id, anchor, fav, dis)
            return CandidateIntervention(
                candidate_id=cand_id, task_id=task.task_id,
                selector=self.name, selector_version=self.version, config_hash=cfg_hash,
                anchor=anchor, favored=fav, disfavored=dis,
                evidence={
                    "column": point.column, "value_counts": point.value_counts,
                    "success_by_value": point.success_by_value,
                    "p_value": point.p_value, "q_value": point.q_value,
                    "favored_value": best_val, "favored_rate": best_rate, "favored_n": best_n,
                    "disfavored_value": worst_val, "disfavored_rate": worst_rate, "disfavored_n": worst_n,
                    "n_points": len(points), "n_columns_tested": n_tested,
                    "anchor_rule": "min sha256(run_id) among runs active at column with replayable prefix",
                    "selection_rule": "earliest column; ties by q-value",
                },
                selection_cost=CostRecord(wall_seconds=time.perf_counter() - t0),
                structure_score=task.structure_score,
            )
        return excl("no_eligible_candidate_pair", ", ".join(f"{k}={v}" for k, v in sorted(why.items())))

    def _runs_with_value(self, alignment: Alignment, runs: list[Run], column: int, value: str) -> list[Run]:
        out = []
        for run in runs:
            row = alignment.matrix[alignment.run_ids.index(run.run_id)]
            if column < len(row) and row[column] == value:
                out.append(run)
        return sorted(out, key=lambda r: anchor_rank(r.run_id))

    def _first_eligible(self, ds: DiscoveryDataset, alignment: Alignment, runs: list[Run], column: int, value: str) -> ActionPayload | None:
        for run in self._runs_with_value(alignment, runs, column, value):
            step_idx = column_to_step_idx(alignment, run, column)
            if step_idx is None:
                continue
            ev = ds.event_for_step(run.run_id, step_idx)
            if ev is None or ev.observation is None:
                continue
            if not ds.classify(ev).eligible:
                continue
            return payload_from_event(run.run_id, ev, ds.policy)
        return None

    def _anchor(self, ds: DiscoveryDataset, alignment: Alignment, runs: list[Run], column: int) -> AnchorRef | None:
        active = []
        for run in runs:
            row = alignment.matrix[alignment.run_ids.index(run.run_id)]
            if column < len(row) and row[column] != GAP:
                active.append(run)
        for run in sorted(active, key=lambda r: anchor_rank(r.run_id)):
            step_idx = column_to_step_idx(alignment, run, column)
            if step_idx is None:
                continue
            entry = ds.provenance[run.run_id].for_step(step_idx)
            if entry is None:
                continue
            if not _anchor_is_replayable(ds, run.run_id, entry.message_idx):
                continue
            return make_anchor(ds, run.run_id, entry.message_idx, step_idx)
        return None


# ── Heuristic selector ────────────────────────────────────────────


@dataclass
class HeuristicSelector:
    """``search_before_edit``: intervene before the first edit made without any prior search.

    Location is task-local and alignment-free. The favored action is the most
    common eligible search action across the task's discovery runs (ties by
    argument string). There is no disfavored comparator because the native
    next action at that point is a mutating edit, which policy v1 excludes.
    """
    name: str = "heuristic"
    version: str = "search_before_edit/0.1"

    def config(self) -> dict:
        return {"rule": "first edit with no prior search; favored = modal eligible search action"}

    def select(self, ds: DiscoveryDataset, task: TaskSpec) -> CandidateIntervention | Exclusion:
        t0 = time.perf_counter()
        cfg_hash = _config_hash(self.name, self.version, self.config())

        def excl(reason: str, detail: str = "") -> Exclusion:
            return Exclusion(task.task_id, self.name, self.version, reason, detail,
                             CostRecord(wall_seconds=time.perf_counter() - t0))

        runs = sorted(ds.resolved_runs(task.task_id), key=lambda r: anchor_rank(r.run_id))
        if not runs:
            return excl("no_resolved_runs")

        pool: Counter[str] = Counter()
        payloads: dict[str, ActionPayload] = {}
        for run_id, ev in ds.enrolled_events(task.task_id):
            p = payload_from_event(run_id, ev)
            if p.action_class != "search":
                continue
            pool[p.arguments_json] += 1
            payloads.setdefault(p.arguments_json, p)
        if not pool:
            return excl("no_eligible_search_action")
        fav_args = sorted(pool.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        fav = payloads[fav_args]

        for run in runs:
            seen_search = False
            for step in run.steps:
                if step.name == "search":
                    seen_search = True
                if step.name in ("edit", "write") and not seen_search:
                    entry = ds.provenance[run.run_id].for_step(step.idx)
                    if entry is None or not _anchor_is_replayable(ds, run.run_id, entry.message_idx):
                        break
                    anchor = make_anchor(ds, run.run_id, entry.message_idx, step.idx)
                    cand_id = _candidate_id(self.name, self.version, task.task_id, anchor, fav, None)
                    return CandidateIntervention(
                        candidate_id=cand_id, task_id=task.task_id,
                        selector=self.name, selector_version=self.version, config_hash=cfg_hash,
                        anchor=anchor, favored=fav, disfavored=None,
                        evidence={"rule": self.version, "trigger_step_idx": step.idx,
                                  "favored_support": pool[fav_args], "n_search_actions": sum(pool.values()),
                                  "disfavored": "none: native next action is a mutating edit"},
                        selection_cost=CostRecord(wall_seconds=time.perf_counter() - t0),
                        structure_score=task.structure_score,
                    )
                if step.name in ("edit", "write"):
                    break
        return excl("rule_not_triggered_on_replayable_prefix")


# ── Random selector ───────────────────────────────────────────────


@dataclass
class RandomSelector:
    seed: int = 20260905
    name: str = "random"
    version: str = "0.1"

    def config(self) -> dict:
        return {"seed": self.seed}

    def select(self, ds: DiscoveryDataset, task: TaskSpec) -> CandidateIntervention | Exclusion:
        import random

        t0 = time.perf_counter()
        cfg_hash = _config_hash(self.name, self.version, self.config())
        enrolled = sorted(ds.enrolled_events(task.task_id), key=lambda x: (anchor_rank(x[0]), x[1].call.message_idx))
        if len(enrolled) < 2:
            return Exclusion(task.task_id, self.name, self.version, "fewer_than_two_enrolled_events",
                             f"{len(enrolled)} enrolled", CostRecord(wall_seconds=time.perf_counter() - t0))
        rng = random.Random(f"{self.seed}:{task.task_id}")
        anchor_run, anchor_ev = enrolled[rng.randrange(len(enrolled))]
        dis = payload_from_event(anchor_run, anchor_ev)      # native next action, eligible by construction
        others = [payload_from_event(r, e) for r, e in enrolled if payload_from_event(r, e).payload_hash != dis.payload_hash]
        if not others:
            return Exclusion(task.task_id, self.name, self.version, "no_distinct_alternative_action", "",
                             CostRecord(wall_seconds=time.perf_counter() - t0))
        fav = others[rng.randrange(len(others))]
        step_idx = None
        for e in ds.provenance[anchor_run].entries:
            if e.tool_call_id == anchor_ev.tool_call_id:
                step_idx = e.step_idx
        anchor = make_anchor(ds, anchor_run, anchor_ev.call.message_idx, step_idx)
        cand_id = _candidate_id(self.name, self.version, task.task_id, anchor, fav, dis)
        return CandidateIntervention(
            candidate_id=cand_id, task_id=task.task_id,
            selector=self.name, selector_version=self.version, config_hash=cfg_hash,
            anchor=anchor, favored=fav, disfavored=dis,
            evidence={"seed": self.seed, "n_enrolled": len(enrolled),
                      "favored": "uniform over enrolled actions != native", "disfavored": "native next action"},
            selection_cost=CostRecord(wall_seconds=time.perf_counter() - t0),
            structure_score=task.structure_score,
        )


# ── Critic selector ───────────────────────────────────────────────

CriticFn = Callable[[TaskSpec, str, list[dict], list[tuple[str, RawEvent]]], tuple[int, ActionPayload, ActionPayload | None] | None]
"""(task, anchor_run_id, anchor messages, enrolled (run_id, event) pool) -> (message_idx, favored, disfavored) or None."""


@dataclass
class CriticSelector:
    """Adapter for an external critic that proposes (location, action).

    Only enforces the shared contract: the proposed location must sit on a
    replayable prefix of the anchor run and the proposed actions must pass the
    frozen eligibility policy. No model-backed critic ships in this package;
    see ``docs/experiments/2026-09-05-intervention-harness-status.md``.
    """
    critic: CriticFn
    critic_id: str = "scripted"
    name: str = "critic"
    version: str = "0.1"

    def config(self) -> dict:
        return {"critic_id": self.critic_id}

    def select(self, ds: DiscoveryDataset, task: TaskSpec) -> CandidateIntervention | Exclusion:
        t0 = time.perf_counter()
        cfg_hash = _config_hash(self.name, self.version, self.config())

        def excl(reason: str, detail: str = "") -> Exclusion:
            return Exclusion(task.task_id, self.name, self.version, reason, detail,
                             CostRecord(wall_seconds=time.perf_counter() - t0))

        runs = sorted(ds.resolved_runs(task.task_id), key=lambda r: anchor_rank(r.run_id))
        if not runs:
            return excl("no_resolved_runs")
        anchor_run = runs[0]
        proposal = self.critic(task, anchor_run.run_id, ds.raw[anchor_run.run_id].messages, ds.enrolled_events(task.task_id))
        if proposal is None:
            return excl("critic_declined")
        message_idx, fav, dis = proposal
        for p in (fav, dis):
            if p is not None and not classify_action(p.tool_name, p.arguments_json, ds.policy).eligible:
                return excl("critic_action_ineligible", p.tool_name)
        if dis is not None and dis.payload_hash == fav.payload_hash:
            return excl("identical_payloads")
        if not _anchor_is_replayable(ds, anchor_run.run_id, message_idx):
            return excl("critic_location_not_replayable", f"message_idx={message_idx}")
        step_idx = None
        for e in ds.provenance[anchor_run.run_id].entries:
            if e.message_idx == message_idx:
                step_idx = e.step_idx
        anchor = make_anchor(ds, anchor_run.run_id, message_idx, step_idx)
        cand_id = _candidate_id(self.name, self.version, task.task_id, anchor, fav, dis)
        return CandidateIntervention(
            candidate_id=cand_id, task_id=task.task_id,
            selector=self.name, selector_version=self.version, config_hash=cfg_hash,
            anchor=anchor, favored=fav, disfavored=dis,
            evidence={"critic_id": self.critic_id},
            selection_cost=CostRecord(wall_seconds=time.perf_counter() - t0),
            structure_score=task.structure_score,
        )


# ── Registry ──────────────────────────────────────────────────────

SELECTOR_FAMILIES = ("moirai", "heuristic", "random", "critic")


def build_selector(name: str, **config) -> InterventionSelector:
    if name == "moirai":
        return MoiraiSelector(**config)
    if name == "heuristic":
        return HeuristicSelector(**config)
    if name == "random":
        return RandomSelector(**config)
    if name == "critic":
        if "critic" not in config:
            raise ValueError("critic selector requires a critic callable; no model-backed critic is bundled")
        return CriticSelector(**config)
    raise ValueError(f"unknown selector family: {name}")


def fork_coverage(ds: DiscoveryDataset, task: TaskSpec, min_branch_size: int = 1, q_threshold: float = 1.0,
                  min_support: int = 1, min_runs: int = 4, late_from: int = 10) -> dict:
    """Diagnostic: every outcome-rated fork under relaxed thresholds, with anchor admission per run.

    Answers "how many anchors at Moirai's fork positions does the policy admit?" without
    changing the frozen selector. ``q_threshold=1.0`` disables the significance filter.
    """
    runs = [r for r in ds.resolved_runs(task.task_id) if r.result.success is not None]
    out = {"task_id": task.task_id, "policy": ds.policy, "n_runs": len(runs), "forks": [], "reason": None}
    if len(runs) < min_runs:
        out["reason"] = f"insufficient_discovery_runs:{len(runs)}"
        return out
    if not any(r.result.success for r in runs) or not any(not r.result.success for r in runs):
        out["reason"] = "single_outcome_task"
        return out
    alignment = align_runs(runs, level="name")
    points, n_tested = find_divergence_points(alignment, runs, min_branch_size, q_threshold)
    out["n_columns_tested"] = n_tested
    sel = MoiraiSelector(min_branch_size=min_branch_size, q_threshold=q_threshold, min_support=min_support)
    for point in sorted(points, key=lambda p: p.column):
        anchors = []
        for run in runs:
            step_idx = column_to_step_idx(alignment, run, point.column)
            if step_idx is None:
                continue
            entry = ds.provenance[run.run_id].for_step(step_idx)
            if entry is None:
                continue
            anchors.append({"run_id": run.run_id, "message_idx": entry.message_idx,
                            "admitted": ds.anchor_admitted(run.run_id, entry.message_idx)})
        bw = _best_worst(point, min_support)
        fav = dis = None
        if bw is not None and bw[0] != GAP and bw[3] != GAP:
            fav = sel._first_eligible(ds, alignment, runs, point.column, bw[0])
            dis = sel._first_eligible(ds, alignment, runs, point.column, bw[3])
        n_adm = sum(a["admitted"] for a in anchors)
        out["forks"].append({
            "column": point.column, "p_value": point.p_value, "q_value": point.q_value,
            "value_counts": point.value_counts, "success_by_value": point.success_by_value,
            "anchors": anchors, "n_active": len(anchors), "n_admitted": n_adm,
            "late": any(a["message_idx"] >= late_from for a in anchors),
            "pair_eligible": fav is not None and dis is not None and fav.payload_hash != dis.payload_hash,
            "candidate_possible": n_adm > 0 and fav is not None and dis is not None and fav.payload_hash != dis.payload_hash,
        })
    return out


def select_all(selector: InterventionSelector, ds: DiscoveryDataset, tasks: list[TaskSpec]) -> tuple[list[CandidateIntervention], list[Exclusion]]:
    cands: list[CandidateIntervention] = []
    excls: list[Exclusion] = []
    for t in tasks:
        r = selector.select(ds, t)
        (cands if isinstance(r, CandidateIntervention) else excls).append(r)  # type: ignore[arg-type]
    return cands, excls
