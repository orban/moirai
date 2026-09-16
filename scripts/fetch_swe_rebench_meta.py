#!/usr/bin/env python3
"""Fetch SWE-rebench instance metadata (base_commit, test_patch, tests, image) for given instance ids.

Uses the HuggingFace datasets-server filter endpoint, so no dataset streaming or cache.

    python scripts/fetch_swe_rebench_meta.py OUTPUT.json --instance BQSKit__bqskit-267 [--instance ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

FIELDS = ("instance_id", "repo", "base_commit", "environment_setup_commit", "version", "created_at",
          "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "docker_image", "image_name", "license_name")


def fetch_one(instance_id: str, split: str = "test") -> dict | None:
    q = urllib.parse.urlencode({"dataset": "nebius/SWE-rebench", "config": "default", "split": split,
                                "where": f"\"instance_id\"='{instance_id}'", "length": 1})
    d: dict = {}
    for attempt in range(4):
        try:
            with urllib.request.urlopen(f"https://datasets-server.huggingface.co/filter?{q}", timeout=60) as r:
                d = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    rows = d.get("rows") or []
    if not rows:
        return None
    row = rows[0]["row"]
    if rows[0].get("truncated_cells"):
        print(f"warning: truncated cells for {instance_id}: {rows[0]['truncated_cells']}", file=sys.stderr)
    return {k: row.get(k) for k in FIELDS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output", type=Path)
    ap.add_argument("--instance", action="append", required=True)
    args = ap.parse_args()
    out: dict[str, dict] = {}
    if args.output.exists():
        out = json.loads(args.output.read_text())
    for iid in args.instance:
        meta = fetch_one(iid) or fetch_one(iid, "filtered")
        if meta is None:
            print(f"not found: {iid}", file=sys.stderr)
            continue
        out[iid] = meta
        print(f"{iid}: base_commit={meta['base_commit'][:10]} image={meta.get('docker_image')} test_patch={len(meta.get('test_patch') or '')}B", file=sys.stderr)
    args.output.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
