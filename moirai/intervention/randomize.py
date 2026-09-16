"""Blocked, balanced randomisation with an immutable assignment manifest."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from moirai.intervention.schema import ARMS, Assignment, CandidateIntervention, content_hash, from_dict


class ManifestDrift(RuntimeError):
    pass


@dataclass
class AssignmentManifest:
    seed: int
    reps: int
    assignments: list[Assignment]
    manifest_hash: str = ""

    def compute_hash(self) -> str:
        return content_hash({"seed": self.seed, "reps": self.reps, "assignments": [asdict(a) for a in self.assignments]})

    def arm_of(self, trial_id: str) -> str:
        for a in self.assignments:
            if a.trial_id == trial_id:
                return a.arm
        raise KeyError(trial_id)


def _trial_id(seed: int, candidate_id: str, block: int, arm: str) -> str:
    return hashlib.sha256(f"{seed}:{candidate_id}:{block}:{arm}".encode("utf-8")).hexdigest()[:16]


def _trial_seed(seed: int, trial_id: str) -> int:
    return int(hashlib.sha256(f"{seed}:trial:{trial_id}".encode("utf-8")).hexdigest()[:12], 16)


def assign(
    candidates: list[CandidateIntervention],
    checkpoint_ids: dict[str, str],
    reps: int,
    seed: int,
) -> AssignmentManifest:
    """One balanced block per repetition per candidate; blocks interleaved across candidates.

    Within a block every available arm appears exactly once in a seeded random
    order, so endpoint drift over time cannot align with an arm.
    """
    cands = sorted(candidates, key=lambda c: c.candidate_id)
    out: list[Assignment] = []
    order = 0
    for block in range(reps):
        for c in cands:
            arms = [a for a in ARMS if not (a == "disfavored" and c.disfavored is None)]
            rng = random.Random(f"{seed}:{c.candidate_id}:{block}")
            rng.shuffle(arms)
            for arm in arms:
                tid = _trial_id(seed, c.candidate_id, block, arm)
                out.append(Assignment(
                    trial_id=tid, task_id=c.task_id, checkpoint_id=checkpoint_ids[c.candidate_id],
                    candidate_id=c.candidate_id, selector=c.selector, arm=arm,
                    block=block, order=order, seed=_trial_seed(seed, tid),
                ))
                order += 1
    m = AssignmentManifest(seed=seed, reps=reps, assignments=out)
    m.manifest_hash = m.compute_hash()
    return m


def write_manifest(path: Path, manifest: AssignmentManifest) -> None:
    Path(path).write_text(json.dumps(asdict(manifest), indent=1, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> AssignmentManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    m = AssignmentManifest(
        seed=data["seed"], reps=data["reps"],
        assignments=[from_dict(Assignment, a) for a in data["assignments"]],
        manifest_hash=data.get("manifest_hash", ""),
    )
    if m.compute_hash() != m.manifest_hash:
        raise ManifestDrift(f"assignment manifest {path} does not match its recorded hash")
    return m


def balance_report(manifest: AssignmentManifest) -> dict[str, dict[str, int]]:
    """Arm counts per candidate; every block contributes one of each available arm."""
    counts: dict[str, dict[str, int]] = {}
    for a in manifest.assignments:
        counts.setdefault(a.candidate_id, {}).setdefault(a.arm, 0)
        counts[a.candidate_id][a.arm] += 1
    return counts
