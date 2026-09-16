"""Synthetic runtime for instrument validation. Never contacts a model."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from moirai.intervention.checkpoint import (
    ContinuationResult,
    ExecutionContext,
    InfrastructureError,
    Workspace,
)
from moirai.intervention.schema import ActionPayload, BudgetState, CheckpointManifest, CostRecord


@dataclass
class SyntheticRuntime:
    """Action-dependent Bernoulli outcomes with visible side effects.

    ``payload_effects`` maps payload hash -> success probability when that
    payload is injected; ``native_p`` applies when nothing is injected. A null
    fixture sets every probability equal. The runtime knows which payload it
    executed, never which arm label the analysis will assign.
    """
    payload_effects: dict[str, float]
    native_p: float
    infra_fail_trials: set[str] = field(default_factory=set)
    agent_error_trials: set[str] = field(default_factory=set)
    grader_flaky_trials: set[str] = field(default_factory=set)
    tokens_per_step: int = 1_000
    steps_per_continuation: int = 5
    executed: list[tuple[str, str]] = field(default_factory=list)

    def restore(self, manifest: CheckpointManifest, prefix_messages: list[dict], workspace: Workspace, trial_id: str, seed: int) -> ExecutionContext:
        if trial_id in self.infra_fail_trials:
            raise InfrastructureError("synthetic provisioning failure")
        return ExecutionContext(trial_id=trial_id, seed=seed, workspace=workspace,
                                messages=list(prefix_messages), budget=manifest.budget)

    def execute_action(self, ctx: ExecutionContext, action: ActionPayload, tool_call_id: str) -> tuple[str, CostRecord]:
        self.executed.append((ctx.trial_id, tool_call_id))
        marker = Path(ctx.workspace.path) / ".moirai_injected"
        marker.write_text(json.dumps({"tool_call_id": tool_call_id, "payload": action.payload_hash}), encoding="utf-8")
        obs = f"[synthetic observation] {action.tool_name} {action.arguments_json[:80]}"
        return obs, CostRecord(input_tokens=self.tokens_per_step, output_tokens=50, requests=1, wall_seconds=0.5)

    def continue_(self, ctx: ExecutionContext) -> ContinuationResult:
        rng = random.Random(ctx.seed)
        p = self.native_p
        for tcid in ctx.injected_tool_call_ids:
            marker = json.loads((Path(ctx.workspace.path) / ".moirai_injected").read_text(encoding="utf-8"))
            p = self.payload_effects.get(marker["payload"], self.native_p)
        usage = CostRecord(
            input_tokens=self.tokens_per_step * self.steps_per_continuation,
            output_tokens=100 * self.steps_per_continuation,
            requests=self.steps_per_continuation, wall_seconds=2.0,
        )
        if ctx.budget.remaining_iterations < self.steps_per_continuation:
            return ContinuationResult("budget_exhausted", usage, None, "synthetic budget exhausted")
        if ctx.trial_id in self.agent_error_trials:
            return ContinuationResult("agent_error", usage, None, "synthetic agent error")
        passed = rng.random() < p
        task_id = ctx.messages[0].get("task_id", "") if ctx.messages else ""
        (Path(ctx.workspace.path) / "outcome.json").write_text(json.dumps({
            "task_id": task_id, "passed": passed,
            "grader_flaky_once": ctx.trial_id in self.grader_flaky_trials,
        }), encoding="utf-8")
        (Path(ctx.workspace.path) / "continuation.log").write_text(f"trial {ctx.trial_id}\n", encoding="utf-8")
        return ContinuationResult("completed", usage, f"synthetic://{ctx.trial_id}")


def synthetic_budget(remaining: int = 50) -> BudgetState:
    return BudgetState(remaining_iterations=remaining, remaining_cost_usd=None, iteration_at_checkpoint=0)
