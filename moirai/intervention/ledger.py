"""Append-only, hash-chained JSONL trial ledger.

Every assigned trial gets exactly one record. Re-appending an identical record
is a no-op; appending a different record under an existing trial id is a
conflict. Outcomes and costs regenerate from the file alone.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from moirai.intervention.schema import CostRecord, TrialRecord, content_hash, from_dict


class LedgerConflict(RuntimeError):
    pass


class LedgerCorrupt(RuntimeError):
    pass


class Ledger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._by_id: dict[str, TrialRecord] = {}
        self._last_hash = "0" * 64
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        prev = "0" * 64
        for n, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            entry = json.loads(line)
            rec = from_dict(TrialRecord, entry["record"])
            if entry.get("prev_hash") != prev:
                raise LedgerCorrupt(f"line {n}: chain broken")
            if rec.compute_hash() != rec.record_hash:
                raise LedgerCorrupt(f"line {n}: record hash mismatch for {rec.trial_id}")
            expected = content_hash({"prev_hash": prev, "record_hash": rec.record_hash})
            if entry.get("entry_hash") != expected:
                raise LedgerCorrupt(f"line {n}: entry hash mismatch")
            if rec.trial_id in self._by_id:
                raise LedgerCorrupt(f"line {n}: duplicate trial_id {rec.trial_id}")
            self._by_id[rec.trial_id] = rec
            prev = entry["entry_hash"]
        self._last_hash = prev

    def has(self, trial_id: str) -> bool:
        return trial_id in self._by_id

    def get(self, trial_id: str) -> TrialRecord | None:
        return self._by_id.get(trial_id)

    def records(self) -> list[TrialRecord]:
        return list(self._by_id.values())

    def append(self, record: TrialRecord) -> bool:
        """Append; return False when an identical record already exists."""
        record.record_hash = record.compute_hash()
        existing = self._by_id.get(record.trial_id)
        if existing is not None:
            if existing.record_hash == record.record_hash:
                return False
            raise LedgerConflict(f"trial {record.trial_id} already recorded with different content")
        entry_hash = content_hash({"prev_hash": self._last_hash, "record_hash": record.record_hash})
        entry = {"prev_hash": self._last_hash, "record": asdict(record), "entry_hash": entry_hash}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
            f.flush()
        self._by_id[record.trial_id] = record
        self._last_hash = entry_hash
        return True

    def verify(self) -> None:
        """Re-read the file from disk and re-validate the whole chain."""
        Ledger(self.path)

    def spent_usd(self) -> float:
        return sum(float(r.usage.get("cost_usd", 0.0)) for r in self._by_id.values())

    def total_usage(self) -> CostRecord:
        total = CostRecord()
        for r in self._by_id.values():
            total = total + from_dict(CostRecord, r.usage)
        return total

    def summary(self) -> dict:
        """Regenerate outcome and cost totals from immutable records."""
        by_arm: dict[str, dict[str, float]] = {}
        for r in self._by_id.values():
            d = by_arm.setdefault(r.arm, {"assigned": 0, "success": 0, "failure": 0, "infra_failure": 0, "cost_usd": 0.0})
            d["assigned"] += 1
            if r.status in d:
                d[r.status] += 1
            d["cost_usd"] += float(r.usage.get("cost_usd", 0.0))
        return {"n_records": len(self._by_id), "by_arm": by_arm, "spent_usd": self.spent_usd()}
