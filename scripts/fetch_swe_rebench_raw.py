#!/usr/bin/env python3
"""Save raw SWE-rebench OpenHands rows (untruncated messages) for provenance and replay.

Streams nebius/SWE-rebench-openhands-trajectories and writes one JSON file per
trajectory for the requested instance or trajectory ids. The whole dataset
streams in a couple of minutes; nothing is cached locally beyond metadata.

Keep raw rows and the HuggingFace cache off the primary volume:

    HF_HOME=/Volumes/mnemosyne/moirai/hf_cache \
    python scripts/fetch_swe_rebench_raw.py /Volumes/mnemosyne/moirai/swe_rebench_raw/tasks \
        --instance numpy__numpydoc-101 --instance PyPSA__linopy-79

Then: moirai intervention audit-raw / provenance / eligibility / select.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--instance", action="append", default=[], help="instance_id to keep (repeatable)")
    ap.add_argument("--trajectory", action="append", default=[], help="trajectory_id to keep (repeatable)")
    ap.add_argument("--limit", type=int, default=0, help="stop after saving this many rows (0 = no limit)")
    ap.add_argument("--max-scan", type=int, default=0, help="stop after scanning this many rows (0 = whole dataset)")
    args = ap.parse_args()

    if not args.instance and not args.trajectory and not args.limit:
        ap.error("give --instance/--trajectory filters or --limit")

    from datasets import load_dataset

    args.output_dir.mkdir(parents=True, exist_ok=True)
    instances, trajectories = set(args.instance), set(args.trajectory)
    ds = load_dataset("nebius/SWE-rebench-openhands-trajectories", split="train", streaming=True)
    t0 = time.time()
    scanned = saved = 0
    for row in ds:
        scanned += 1
        keep = (not instances and not trajectories) or row.get("instance_id") in instances or row.get("trajectory_id") in trajectories
        if keep:
            tid = str(row.get("trajectory_id") or f"row{scanned}")
            (args.output_dir / f"{tid}.json").write_text(json.dumps(row, indent=1) + "\n", encoding="utf-8")
            saved += 1
            print(f"saved {tid} {row.get('instance_id')} resolved={row.get('resolved')}", file=sys.stderr)
        if scanned % 5000 == 0:
            print(f"scanned {scanned}, saved {saved}, {time.time() - t0:.0f}s", file=sys.stderr)
        if (args.limit and saved >= args.limit) or (args.max_scan and scanned >= args.max_scan):
            break
    print(f"done: scanned {scanned}, saved {saved} -> {args.output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
