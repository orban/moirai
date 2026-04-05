from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple


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
class ActivityDivergence:
    """A column where having a step (vs gap) predicts outcome.

    Tests the 2×2 table: (active/gap) × (pass/fail) via Fisher's exact.
    Complements DivergencePoint which tests different step *types*.
    """
    column: int
    pass_active: int                      # pass runs with a step here
    pass_gap: int                         # pass runs with a gap here
    fail_active: int                      # fail runs with a step here
    fail_gap: int                         # fail runs with a gap here
    p_value: float                        # Fisher's exact (two-sided)
    q_value: float | None = None          # BH-adjusted
    direction: float = 0.0                # pass_rate - fail_rate of being active (>0 = pass-biased)
    active_labels: dict[str, int] = field(default_factory=dict)  # step names at this column
    phase_context: str | None = None


@dataclass(frozen=True)
class FeatureSpec:
    """Definition of a behavioral feature in the registry."""
    name: str
    compute: Callable  # (Run) -> float | None
    direction: Literal["positive", "negative"]
    description: str


class ExperimentResult(NamedTuple):
    """Result of a within-task median-split natural experiment."""
    delta_pp: float
    p_value: float | None
    n_tasks: int


@dataclass
class FeatureResult:
    """Ranked result for one behavioral feature."""
    name: str
    description: str
    direction: Literal["positive", "negative"]
    delta_pp: float
    p_value: float | None
    q_value: float | None
    split_half: bool
    pass_mean: float
    fail_mean: float
    n_tasks: int
    n_runs: int


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
class ConcordanceScore:
    """Structural concordance for a cluster — does typicality predict outcome?"""
    tau: float                    # Kendall's Tau-b, in [-1, 1]
    p_value: float | None         # significance; None = degenerate input
    n_runs: int                   # runs used (after excluding unknown outcomes)
    used_continuous: bool         # True if result.score was used, False if binary


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


# --- Evidence and diagnosis dataclasses ---


@dataclass
class FeatureShift:
    """A single behavioral feature comparison between variants."""
    feature: str            # canonical name (e.g., "TEST_AFTER_EDIT_RATE")
    baseline_value: float
    current_value: float
    shift: float            # current - baseline
    effect_size: float      # Cohen's h for proportions, Cohen's d for means
    ci_lower: float         # 95% CI on shift
    ci_upper: float
    magnitude: str          # "negligible", "small", "medium", "large"


@dataclass
class TaskBreakdown:
    """Per-task success rate change between variants."""
    task_id: str
    baseline_pass_rate: float
    current_pass_rate: float
    delta: float
    baseline_runs: int
    current_runs: int


@dataclass
class VariantComparison:
    """Complete structured comparison of two variants."""
    baseline_label: str
    current_label: str
    baseline_pass_rate: float
    current_pass_rate: float
    pass_rate_delta: float
    feature_shifts: list[FeatureShift]
    task_breakdown: list[TaskBreakdown]


@dataclass
class CauseScore:
    """Ranking result for a single candidate cause."""
    cause_id: str
    cause_description: str
    score: float                    # normalized score (0-1)
    ci_lower: float                 # bootstrap CI
    ci_upper: float
    matched_features: list[str]     # which features matched
    evidence_strength: float        # raw pre-normalization score


@dataclass
class DiagnosisResult:
    """Complete output of the cause ranking pipeline."""
    cause_scores: list[CauseScore]  # sorted by score descending
    unknown_score: float            # residual mass for unmodeled causes
    feature_shifts: list[FeatureShift]
    task_breakdown: list[TaskBreakdown]
    baseline_pass_rate: float
    current_pass_rate: float


# --- Content-aware analysis dataclasses ---

@dataclass
class ReasoningMetrics:
    """Per-run reasoning quality metrics extracted from agent thinking text."""
    uncertainty_density: float     # hedging words per reasoning step
    causal_density: float          # causal connectives per reasoning step
    diagnosis_density: float       # explicit diagnosis phrases per reasoning step
    code_ref_density: float        # code references (files, functions, lines) per reasoning step
    reasoning_per_step: float      # avg reasoning chars per step
    n_reasoning_steps: int         # steps that have reasoning text


@dataclass
class TransitionSignal:
    """A transition bigram that differs between passing and failing runs."""
    from_step: str
    to_step: str
    pass_rate: float        # avg per-run normalized rate in passing runs
    fail_rate: float        # avg per-run normalized rate in failing runs
    delta: float            # pass_rate - fail_rate (positive = pass-correlated)
    total_count: int        # raw count across all runs


@dataclass
class ContentFinding:
    """A single finding from LLM-assisted content comparison."""
    category: str              # free-text from LLM; normalized when possible
    column: int                # alignment column where observed
    description: str           # one-sentence explanation
    evidence: str              # specific content from runs
    pass_runs: list[str] = field(default_factory=list)
    fail_runs: list[str] = field(default_factory=list)


@dataclass
class ExplanationReport:
    """Output of content-aware explain for one task group."""
    task_id: str
    n_runs: int
    pass_rate: float
    findings: list[ContentFinding]
    summary: str                                    # free-form narrative; empty if structural
    confidence: str                                 # "high"/"medium"/"low"; empty if structural
    consensus: str                                  # compressed phase notation
    divergent_columns: list[DivergencePoint]         # reuse existing dataclass
    n_qualifying: int                               # task groups that qualified
    n_skipped: int                                  # task groups skipped
    reasoning: ReasoningMetrics | None = None         # per-group aggregate
    reasoning_pass: ReasoningMetrics | None = None   # passing runs aggregate
    reasoning_fail: ReasoningMetrics | None = None   # failing runs aggregate
    transitions: list[TransitionSignal] = field(default_factory=list)
    concordance_tau: float | None = None             # only with --cluster
    concordance_p: float | None = None               # only with --cluster


# --- Sequence extraction (analysis primitives, not normalization) ---

def step_type_sequence(run: Run) -> list[str]:
    return [s.type for s in run.steps]


def step_name_sequence(run: Run) -> list[str]:
    return [s.name for s in run.steps]


def signature(run: Run) -> str:
    parts = [f"{s.type}:{s.name}" for s in run.steps]
    return " > ".join(parts)
