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
    q_value: float | None = None          # BH-adjusted p-value
    min_branch_size: int = 0              # smallest branch count — low = unstable
    phase_context: str | None = None      # e.g. "explore→[modify] vs explore→[explore]"


@dataclass
class SplitDivergence:
    """A divergence derived from a dendrogram split."""
    node_id: int                          # internal node in linkage matrix
    left_runs: list[str]                  # run_ids in left subtree
    right_runs: list[str]                 # run_ids in right subtree
    merge_distance: float                 # linkage distance at this split
    column: int                           # most discriminating alignment column
    separation: float                     # 0-1: how well the column separates the subtrees
    left_values: dict[str, int]           # value_counts in left subtree at column
    right_values: dict[str, int]          # value_counts in right subtree at column
    left_success_rate: float | None       # pass rate of left subtree
    right_success_rate: float | None      # pass rate of right subtree
    p_value: float | None = None          # Fisher's: does this split predict outcome?
    # SVG coordinates (filled by renderer)
    svg_x: float = 0.0
    svg_y: float = 0.0


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


@dataclass
class Motif:
    """A recurring contiguous step pattern with outcome correlation."""
    pattern: tuple[str, ...]
    total_runs: int          # runs containing this pattern
    success_runs: int        # successful runs containing it
    fail_runs: int           # failing runs containing it
    success_rate: float      # success rate of runs with this pattern
    baseline_rate: float     # overall success rate for comparison
    lift: float              # success_rate / baseline_rate (>1 = positive, <1 = negative)
    p_value: float | None    # Fisher's exact test (raw)
    avg_position: float      # average position (0-1 normalized) where the pattern appears
    q_value: float | None = None  # BH-adjusted p-value

    @property
    def display(self) -> str:
        return " → ".join(self.pattern)


@dataclass
class GappedMotif:
    """An ordered subsequence pattern with flexible gaps between anchors."""
    anchors: tuple[str, ...]
    total_runs: int
    success_runs: int
    fail_runs: int
    success_rate: float
    baseline_rate: float
    lift: float
    p_value: float | None
    q_value: float | None = None
    avg_position: float = 0.0

    @property
    def display(self) -> str:
        return " → ... → ".join(self.anchors)


# --- Sequence extraction (analysis primitives, not normalization) ---

def step_type_sequence(run: Run) -> list[str]:
    return [s.type for s in run.steps]


def step_name_sequence(run: Run) -> list[str]:
    return [s.name for s in run.steps]


def signature(run: Run) -> str:
    parts = [f"{s.type}:{s.name}" for s in run.steps]
    return " > ".join(parts)
