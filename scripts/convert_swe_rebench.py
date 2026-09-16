#!/usr/bin/env python3
"""Convert Nebius SWE-rebench OpenHands trajectories to moirai format.

Downloads from HuggingFace and converts to moirai JSON runs.
Only downloads mixed-outcome instances (both pass and fail for same task).

Usage:
    python scripts/convert_swe_rebench.py OUTPUT_DIR [--min-pass N] [--min-fail N]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moirai.converters.swe_rebench import (  # noqa: E402
    _classify_tool,
    convert_row,
    parse_openhands_trajectory,
)

__all__ = ["parse_openhands_trajectory", "_classify_tool", "convert_row", "main"]



def main():
    parser = argparse.ArgumentParser(description="Convert SWE-rebench trajectories")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--min-pass", type=int, default=1, help="Min pass runs per instance")
    parser.add_argument("--min-fail", type=int, default=1, help="Min fail runs per instance")
    parser.add_argument("--limit", type=int, default=0, help="Max instances to process (0=all)")
    args = parser.parse_args()

    from datasets import load_dataset

    print("Loading dataset (streaming)...", file=sys.stderr)
    ds = load_dataset("nebius/SWE-rebench-openhands-trajectories", split="train", streaming=True)

    # First pass: identify mixed-outcome instances
    print("Pass 1: identifying mixed-outcome instances...", file=sys.stderr)
    instance_outcomes: dict[str, dict] = defaultdict(lambda: {"pass": 0, "fail": 0})
    total = 0
    for row in ds:
        total += 1
        iid = row["instance_id"]
        resolved = row.get("resolved")
        if resolved == 1 or resolved is True:
            instance_outcomes[iid]["pass"] += 1
        elif resolved == 0 or resolved is False:
            instance_outcomes[iid]["fail"] += 1
        if total % 10000 == 0:
            print(f"  scanned {total}...", file=sys.stderr)

    mixed = {
        iid for iid, v in instance_outcomes.items()
        if v["pass"] >= args.min_pass and v["fail"] >= args.min_fail
    }
    print(f"  {len(mixed)} instances with >= {args.min_pass} pass + {args.min_fail} fail", file=sys.stderr)

    if args.limit > 0:
        mixed = set(sorted(mixed)[:args.limit])
        print(f"  limited to {len(mixed)} instances", file=sys.stderr)

    # Second pass: convert mixed-outcome runs
    print("Pass 2: converting trajectories...", file=sys.stderr)
    ds = load_dataset("nebius/SWE-rebench-openhands-trajectories", split="train", streaming=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    converted = 0
    skipped = 0

    for row in ds:
        iid = row["instance_id"]
        if iid not in mixed:
            skipped += 1
            continue

        traj_id = row.get("trajectory_id", f"{iid}_{converted}")
        run = convert_row(row, traj_id)

        if not run["steps"]:
            skipped += 1
            continue

        # Write one file per run
        safe_id = traj_id.replace("/", "_").replace(" ", "_")[:200]
        out_path = args.output_dir / f"{safe_id}.json"
        out_path.write_text(json.dumps(run, indent=2) + "\n")
        converted += 1

        if converted % 1000 == 0:
            print(f"  converted {converted}...", file=sys.stderr)

    print(f"\nDone: {converted} runs converted, {skipped} skipped", file=sys.stderr)
    print(f"Output: {args.output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
