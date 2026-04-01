"""Candidate cause schema and JSON loading."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CandidateCause:
    """A candidate system change that might explain a behavioral regression."""
    id: str
    type: str                               # prompt, tool, model, runtime, config
    description: str
    expected_shifts: dict[str, str]         # feature → "increase" | "decrease"
    prior: float = 0.0                      # 0 = uniform (auto-assigned before scoring)
    metadata: dict[str, Any] = field(default_factory=dict)


def load_causes(path: Path) -> list[CandidateCause]:
    """Load candidate causes from a JSON file.

    Expected format: a JSON array of objects with id, type, description,
    expected_shifts, and optional prior/metadata fields.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    causes = []
    for item in data:
        causes.append(CandidateCause(
            id=item["id"],
            type=item.get("type", "unknown"),
            description=item.get("description", ""),
            expected_shifts=item.get("expected_shifts", {}),
            prior=item.get("prior", 0.0),
            metadata=item.get("metadata", {}),
        ))
    return causes
