from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


KNOWN_STEP_TYPES = {"llm", "tool", "system", "memory", "compaction", "judge", "error", "handoff"}

STEP_TYPE_ALIASES: dict[str, str] = {
    "assistant": "llm",
    "completion": "llm",
    "function_call": "tool",
    "retrieval": "tool",
    "compress": "compaction",
    "critic": "judge",
}


@dataclass
class Step:
    idx: int
    type: str
    name: str
    status: str = "ok"
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    success: bool | None
    score: float | None = None
    label: str | None = None
    error_type: str | None = None
    summary: str | None = None


@dataclass
class Run:
    run_id: str
    task_id: str
    task_family: str | None = None
    agent: str | None = None
    model: str | None = None
    harness: str | None = None
    timestamp: str | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)
    result: Result = field(default_factory=lambda: Result(success=None))


@dataclass
class RunSummary:
    run_count: int
    success_rate: float | None
    avg_steps: float
    median_steps: float
    avg_tokens_in: float | None
    avg_tokens_out: float | None
    avg_latency_ms: float | None
    top_signatures: list[tuple[str, int]]
    error_counts: dict[str, int]


@dataclass
class ValidationResult:
    file_path: str
    passed: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


GAP = "-"


@dataclass
class Alignment:
    """Matrix of aligned step values across runs."""
    run_ids: list[str]
    matrix: list[list[str]]  # [run_index][column_index] = value or GAP
    level: str  # "type" or "name"


@dataclass
class DivergencePoint:
    column: int
    value_counts: dict[str, int]
    entropy: float
    success_by_value: dict[str, float | None]
    # Added in v1: quality metrics
    p_value: float | None = None          # Fisher's exact test: does branch predict outcome?
    min_branch_size: int = 0              # smallest branch count — low = unstable
    phase_context: str | None = None      # e.g. "explore→[modify] vs explore→[explore]"


@dataclass
class CohortDiff:
    a_summary: RunSummary
    b_summary: RunSummary
    a_only_signatures: list[tuple[str, int]]  # pre-clustering
    b_only_signatures: list[tuple[str, int]]  # pre-clustering
    cluster_shifts: list[tuple[str, int]]     # post-clustering: (prototype, count_delta)


@dataclass
class ClusterInfo:
    cluster_id: int
    count: int
    success_rate: float | None
    prototype: str
    avg_length: float
    error_types: dict[str, int]


@dataclass
class ClusterResult:
    clusters: list[ClusterInfo]
    labels: dict[str, int]  # run_id -> cluster_id


# --- Sequence extraction (analysis primitives, not normalization) ---

def step_type_sequence(run: Run) -> list[str]:
    return [s.type for s in run.steps]


def step_name_sequence(run: Run) -> list[str]:
    return [s.name for s in run.steps]


def signature(run: Run) -> str:
    parts = [f"{s.type}:{s.name}" for s in run.steps]
    return " > ".join(parts)
