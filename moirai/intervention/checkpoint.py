"""Checkpoint manifests, workspace cloning, and the runtime adapter contract.

A checkpoint is the model-visible prefix plus a writable workspace plus the
hashes of everything else the model or tools could observe. Cloning produces
one independent writable workspace per arm; equality is verified by tree hash
before randomised execution begins, and contamination is tested by mutation.

The runtime adapter is the only place the historical harness is touched. This
package ships a synthetic adapter for instrument tests. A real OpenHands
adapter must implement the same protocol and must not share workspaces or
reset budgets between arms.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from moirai.intervention.schema import (
    ActionPayload,
    AnchorRef,
    BudgetState,
    CheckpointManifest,
    CostRecord,
    content_hash,
)


class InfrastructureError(RuntimeError):
    """A failure independent of agent behaviour (provisioning, endpoint outage)."""


class ManifestDrift(RuntimeError):
    """A restored clone does not match its frozen manifest."""


# ── Workspaces ────────────────────────────────────────────────────


def tree_hash(root: Path) -> str:
    """Content hash of a directory tree: relative paths, modes and bytes."""
    h = hashlib.sha256()
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            rel = p.relative_to(root).as_posix()
            h.update(rel.encode("utf-8"))
            h.update(b"\0")
            if p.is_symlink():
                h.update(b"L" + os.readlink(p).encode("utf-8"))
            else:
                h.update(b"F")
                with open(p, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 16), b""):
                        h.update(chunk)
            h.update(b"\0")
    return h.hexdigest()


@dataclass
class Workspace:
    workspace_id: str
    path: Path
    parent_id: str | None = None

    def hash(self) -> str:
        return tree_hash(self.path)


class WorkspaceProvider(Protocol):
    def clone(self, source: Workspace) -> Workspace: ...
    def release(self, ws: Workspace) -> None: ...


@dataclass
class LocalDirWorkspaceProvider:
    """Independent clones as separate directories under ``root``."""
    root: Path
    clones: list[Workspace] = field(default_factory=list)

    def register_base(self, path: Path, workspace_id: str | None = None) -> Workspace:
        return Workspace(workspace_id or f"base-{uuid.uuid4().hex[:8]}", Path(path))

    def clone(self, source: Workspace) -> Workspace:
        wid = f"clone-{uuid.uuid4().hex[:12]}"
        dest = Path(self.root) / "clones" / wid
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source.path, dest, symlinks=True)
        ws = Workspace(wid, dest, parent_id=source.workspace_id)
        self.clones.append(ws)
        return ws

    def release(self, ws: Workspace) -> None:
        # Clones are retained for audit; deletion is an explicit operator step.
        return None


def verify_independent(a: Workspace, b: Workspace) -> list[str]:
    """Structural checks that two clones share no writable state."""
    problems: list[str] = []
    ra, rb = Path(a.path).resolve(), Path(b.path).resolve()
    if ra == rb:
        problems.append("same path")
    if ra in rb.parents or rb in ra.parents:
        problems.append("nested paths")
    for dirpath, _, filenames in os.walk(ra):
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.is_symlink():
                target = Path(os.path.realpath(p))
                if rb == target or rb in target.parents:
                    problems.append(f"symlink into other clone: {p.relative_to(ra)}")
            elif p.stat().st_nlink > 1:
                problems.append(f"hard-linked file: {p.relative_to(ra)}")
    return problems


def contamination_probe(mutated: Workspace, others: list[Workspace], source: Workspace) -> list[str]:
    """Mutate one clone and confirm the others and the source did not change."""
    before = {w.workspace_id: w.hash() for w in others + [source]}
    marker = Path(mutated.path) / ".moirai_contamination_probe"
    marker.write_text(uuid.uuid4().hex, encoding="utf-8")
    cache_dir = Path(mutated.path) / "__pycache__"
    cache_dir.mkdir(exist_ok=True)
    (cache_dir / "probe.pyc").write_bytes(b"probe")
    after = {w.workspace_id: w.hash() for w in others + [source]}
    return [wid for wid in before if before[wid] != after[wid]]


# ── Manifests ─────────────────────────────────────────────────────


def canonical_request_hash(prefix_messages: list[dict], tools: list[dict], model_config: dict) -> str:
    return content_hash({"messages": prefix_messages, "tools": tools, "model": model_config})


def build_manifest(
    task_id: str,
    anchor: AnchorRef,
    prefix_messages: list[dict],
    tools: list[dict],
    workspace: Workspace,
    environment: dict,
    model_config: dict,
    budget: BudgetState,
    restorability: str,
    exceptions: tuple[str, ...] = (),
) -> CheckpointManifest:
    prefix_hash = content_hash(prefix_messages)
    if prefix_hash != anchor.prefix_hash:
        raise ManifestDrift(f"prefix hash {prefix_hash[:12]} != anchor prefix hash {anchor.prefix_hash[:12]}")
    if prefix_messages and prefix_messages[-1].get("role") == "assistant":
        raise ManifestDrift("prefix ends with an assistant message; boundary must precede the model's next event")
    tools_hash = content_hash(tools)
    req_hash = canonical_request_hash(prefix_messages, tools, model_config)
    cid = content_hash({"task": task_id, "anchor": [anchor.anchor_run_id, anchor.message_idx], "req": req_hash,
                        "ws": workspace.hash(), "env": environment})[:16]
    return CheckpointManifest(
        checkpoint_id=cid, task_id=task_id, anchor=anchor,
        prefix_hash=prefix_hash, tools_hash=tools_hash, workspace_hash=workspace.hash(),
        environment=dict(environment), model_config=dict(model_config), budget=budget,
        canonical_request_hash=req_hash, restorability=restorability, exceptions=tuple(exceptions),
    )


def verify_clone(manifest: CheckpointManifest, clone: Workspace, prefix_messages: list[dict], tools: list[dict], model_config: dict) -> list[str]:
    problems: list[str] = []
    ws_hash = clone.hash()
    if ws_hash != manifest.workspace_hash:
        problems.append(f"workspace hash {ws_hash[:12]} != manifest {manifest.workspace_hash[:12]}")
    if content_hash(prefix_messages) != manifest.prefix_hash:
        problems.append("prefix messages drifted")
    if content_hash(tools) != manifest.tools_hash:
        problems.append("tool registry drifted")
    if canonical_request_hash(prefix_messages, tools, model_config) != manifest.canonical_request_hash:
        problems.append("canonical request drifted")
    return problems


# ── Runtime adapter contract ──────────────────────────────────────


@dataclass
class ExecutionContext:
    trial_id: str
    seed: int
    workspace: Workspace
    messages: list[dict]
    budget: BudgetState
    injected_tool_call_ids: list[str] = field(default_factory=list)
    usage: CostRecord = field(default_factory=CostRecord)


@dataclass(frozen=True)
class ContinuationResult:
    status: str                    # completed | agent_error | timeout | budget_exhausted | invalid_action
    usage: CostRecord
    trajectory_ref: str | None
    detail: str = ""


class RuntimeAdapter(Protocol):
    """Everything that touches the agent harness goes through this."""

    def restore(self, manifest: CheckpointManifest, prefix_messages: list[dict], workspace: Workspace, trial_id: str, seed: int) -> ExecutionContext: ...

    def execute_action(self, ctx: ExecutionContext, action: ActionPayload, tool_call_id: str) -> tuple[str, CostRecord]:
        """Execute one exact tool call once; return its real observation and cost."""
        ...

    def continue_(self, ctx: ExecutionContext) -> ContinuationResult:
        """Resume the unchanged policy from ctx until it finishes or exhausts budget."""
        ...
