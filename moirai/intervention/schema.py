"""Immutable contracts for the intervention study.

Every record that crosses a stage boundary (provenance -> eligibility ->
selection -> checkpoint -> assignment -> trial -> analysis) is a frozen
dataclass with a stable dict form, so the ledger can be regenerated from logs
and hashed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Literal

Arm = Literal["favored", "disfavored", "native"]
ARMS: tuple[Arm, ...] = ("favored", "disfavored", "native")

TrialStatus = Literal["pending", "success", "failure", "infra_failure"]


def canonical_json(obj: Any) -> str:
    """Deterministic JSON used for every hash in this package."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_default)


def _default(o: Any) -> Any:
    if is_dataclass(o) and not isinstance(o, type):
        return asdict(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def content_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def to_dict(obj: Any) -> dict:
    return asdict(obj)


def from_dict(cls, data: dict):
    """Rebuild a flat dataclass from a dict, ignoring unknown keys."""
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


# ── Raw trajectory model ──────────────────────────────────────────


@dataclass(frozen=True)
class RawToolCall:
    """One tool call as the model emitted it, untruncated."""
    message_idx: int
    call_idx: int
    tool_call_id: str
    tool_name: str
    arguments_json: str           # exact string the model produced
    arguments: dict | None        # decoded, None when not valid JSON object
    assistant_content: str        # text in the same assistant message


@dataclass(frozen=True)
class RawEvent:
    """A tool call paired with its observation, matched by tool_call_id."""
    call: RawToolCall
    observation_message_idx: int | None
    observation: str | None

    @property
    def tool_call_id(self) -> str:
        return self.call.tool_call_id


@dataclass
class RawTrajectory:
    trajectory_id: str
    instance_id: str
    repo: str
    messages: list[dict]
    tools: list[dict] = field(default_factory=list)
    resolved: bool | None = None
    exit_status: str | None = None
    model_patch_hash: str | None = None   # hash only; the patch itself is solution leakage
    image: str | None = None              # per-task container image when the corpus publishes one
    source: str = "swe_rebench"
    extra: dict = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        return self.trajectory_id


@dataclass(frozen=True)
class RawAudit:
    """Ingestion audit for one raw trajectory."""
    trajectory_id: str
    n_messages: int
    n_assistant: int
    n_tool_calls: int
    n_multi_call_messages: int
    n_orphan_observations: int      # tool messages whose id matches no call
    n_missing_observations: int     # calls with no observation
    n_duplicate_ids: int
    n_undecodable_arguments: int


# ── Provenance ────────────────────────────────────────────────────


@dataclass(frozen=True)
class StepProvenance:
    step_idx: int
    message_idx: int
    call_idx: int | None
    tool_call_id: str | None
    tool_name: str | None
    arguments_hash: str | None
    observation_message_idx: int | None
    truncated_fields: tuple[str, ...] = ()


@dataclass
class ProvenanceMap:
    run_id: str
    trajectory_id: str
    status: Literal["resolved", "failed"]
    entries: list[StepProvenance] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    converter_hash: str | None = None

    def for_step(self, step_idx: int) -> StepProvenance | None:
        for e in self.entries:
            if e.step_idx == step_idx:
                return e
        return None


# ── Eligibility ───────────────────────────────────────────────────

ActionClass = Literal[
    "view", "search", "list", "read_shell", "reason",
    "edit", "create", "shell_cwd", "shell_env", "exec", "setup", "test",
    "background", "input", "browse", "unknown",
]

NON_MUTATING_CLASSES: frozenset[str] = frozenset({"view", "search", "list", "read_shell"})
REPLAYABLE_PREFIX_CLASSES: frozenset[str] = NON_MUTATING_CLASSES | {"reason", "edit", "create", "shell_cwd"}


@dataclass(frozen=True)
class ActionVerdict:
    action_class: str
    eligible: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReplayCertificate:
    """What a stored replay report certifies about one trajectory (policy v2).

    Anchors at ``message_idx <= certified_through`` have a prefix whose every replayed
    step was accepted under the rule hash ``judged_with`` after a passing baseline probe.
    """
    run_id: str
    report_path: str
    instrument_hash: str
    judged_with: str
    baseline_ok: bool
    replayed_through: int            # anchor boundary the replay was asked for
    first_rejected_message_idx: int | None
    certified_through: int | None    # None when the baseline probe failed
    complete: bool                   # every call in the trajectory replayed and accepted
    n_steps: int


@dataclass(frozen=True)
class PrefixRestorability:
    status: Literal["replayable", "requires_full_replay", "admissible", "inadmissible"]
    blocking_classes: tuple[str, ...]
    n_prefix_events: int
    first_blocking_message_idx: int | None


@dataclass(frozen=True)
class EligibilityRecord:
    """One row of the eligibility ledger: a candidate raw event on one run."""
    run_id: str
    task_id: str
    message_idx: int
    tool_call_id: str
    tool_name: str
    action_class: str
    action_eligible: bool
    action_reasons: tuple[str, ...]
    prefix_status: str
    prefix_blocking: tuple[str, ...]
    enrolled: bool
    exclusion_reason: str | None
    policy_version: str


# ── Selection ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ActionPayload:
    """Exact executable action transported from a discovery run."""
    tool_name: str
    arguments_json: str
    action_class: str
    source_run_id: str
    source_message_idx: int
    source_tool_call_id: str

    @property
    def payload_hash(self) -> str:
        return content_hash({"tool": self.tool_name, "args": self.arguments_json})


@dataclass(frozen=True)
class AnchorRef:
    """Predecision boundary: clone messages[:message_idx] of anchor_run."""
    anchor_run_id: str
    message_idx: int
    step_idx: int | None
    prefix_hash: str


@dataclass(frozen=True)
class CostRecord:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    wall_seconds: float = 0.0
    cost_usd: float = 0.0

    def __add__(self, other: "CostRecord") -> "CostRecord":
        return CostRecord(
            input_tokens=self.input_tokens + other.input_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            requests=self.requests + other.requests,
            wall_seconds=self.wall_seconds + other.wall_seconds,
            cost_usd=self.cost_usd + other.cost_usd,
        )


@dataclass(frozen=True)
class CandidateIntervention:
    candidate_id: str
    task_id: str
    selector: str
    selector_version: str
    config_hash: str
    anchor: AnchorRef
    favored: ActionPayload
    disfavored: ActionPayload | None
    evidence: dict
    selection_cost: CostRecord = field(default_factory=CostRecord)
    structure_score: float | None = None
    validity: str = "deferred_to_runtime"


@dataclass(frozen=True)
class Exclusion:
    task_id: str
    selector: str
    selector_version: str
    reason: str
    detail: str = ""
    selection_cost: CostRecord = field(default_factory=CostRecord)


# ── Checkpoint ────────────────────────────────────────────────────


@dataclass(frozen=True)
class BudgetState:
    """Remaining resource budget at the checkpoint; inherited by every arm."""
    remaining_iterations: int
    remaining_cost_usd: float | None = None
    iteration_at_checkpoint: int = 0


@dataclass(frozen=True)
class CheckpointManifest:
    checkpoint_id: str
    task_id: str
    anchor: AnchorRef
    prefix_hash: str               # canonical hash of model-visible messages[:idx]
    tools_hash: str                # tool registry visible to the model
    workspace_hash: str            # content hash of writable workspace tree
    environment: dict              # image digest, cwd, env var hashes, versions
    model_config: dict             # model id, revision, sampling params, endpoint
    budget: BudgetState
    canonical_request_hash: str    # hash(prefix, tools, model_config)
    restorability: str
    exceptions: tuple[str, ...] = ()   # logged canonicalisation exceptions


# ── Assignment and trials ─────────────────────────────────────────


@dataclass(frozen=True)
class Assignment:
    trial_id: str                  # opaque
    task_id: str
    checkpoint_id: str
    candidate_id: str
    selector: str
    arm: str
    block: int
    order: int
    seed: int


@dataclass(frozen=True)
class GradeResult:
    passed: bool | None
    status: Literal["ok", "grader_error"]
    detail: str = ""
    evaluator_version: str = ""
    attempts: int = 1


@dataclass
class TrialRecord:
    trial_id: str
    task_id: str
    checkpoint_id: str
    candidate_id: str
    selector: str
    arm: str
    block: int
    seed: int
    status: str                                 # TrialStatus
    outcome: bool | None                        # None when missing
    failure_class: str | None                   # agent_error, timeout, budget_exhausted, invalid_action, infra:<kind>
    grade: dict | None
    usage: dict                                 # CostRecord dict
    model_config_hash: str
    prompt_hash: str
    tools_hash: str
    environment_hash: str
    workspace_hash_before: str
    workspace_hash_after: str | None
    injected_tool_call_id: str | None
    injected_payload_hash: str | None
    trajectory_ref: str | None
    retry_of: str | None
    started_at: str
    finished_at: str | None
    infra: dict = field(default_factory=dict)
    record_hash: str = ""

    def compute_hash(self) -> str:
        d = to_dict(self)
        d.pop("record_hash", None)
        return content_hash(d)
