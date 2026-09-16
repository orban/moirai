#!/usr/bin/env python3
"""Fetch raw CoderForge-Preview rows for chosen tasks without streaming the whole split.

Reads parquet row-group statistics over HTTP range requests and downloads only the
row groups that can contain the requested trajectory ids (``<task>_run<N>``).
Rows are written verbatim as JSON, one file per trajectory, into OUTPUT_DIR.

    HF_HOME=/Volumes/mnemosyne/moirai/hf_cache \
    python scripts/fetch_coderforge_raw.py /Volumes/mnemosyne/moirai/coderforge_raw/tasks \
        --split SWE_Rebench --task BQSKit__bqskit-267 --task 12rambau__sepal_ui-646
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = "datasets/togethercomputer/CoderForge-Preview"
REV = "refs%2Fconvert%2Fparquet"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--split", default="SWE_Rebench")
    ap.add_argument("--config", default="trajectories")
    ap.add_argument("--task", action="append", required=True, help="task id, repeatable")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    base = f"{REPO}@{REV}/{args.config}/{args.split}"
    files = sorted(p["name"] if isinstance(p, dict) else p for p in fs.ls(base, detail=False))
    tasks = sorted(set(args.task))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    saved = 0
    groups_read = 0
    for fpath in files:
        with fs.open(fpath, "rb") as fh:
            pf = pq.ParquetFile(fh)
            schema_names = pf.schema_arrow.names
            col = schema_names.index("trajectory_id")
            for rg in range(pf.num_row_groups):
                st = pf.metadata.row_group(rg).column(col).statistics
                if st is not None and st.has_min_max:
                    lo, hi = str(st.min), str(st.max)
                    if not any(lo <= f"{t}_run9" and t <= hi for t in tasks):
                        continue
                ids = pf.read_row_group(rg, columns=["trajectory_id"]).column(0).to_pylist()
                hits = [i for i, tid in enumerate(ids) if any(tid.startswith(t + "_run") for t in tasks)]
                if not hits:
                    continue
                groups_read += 1
                table = pf.read_row_group(rg)
                for i in hits:
                    row = {name: table.column(name)[i].as_py() for name in schema_names}
                    tid = row["trajectory_id"]
                    (args.output_dir / f"{tid}.json").write_text(json.dumps(row) + "\n", encoding="utf-8")
                    saved += 1
                    print(f"saved {tid} reward={row.get('reward')} image={row.get('image')}", file=sys.stderr)
        print(f"{fpath.rsplit('/', 1)[-1]}: {saved} saved, {groups_read} groups read, {time.time() - t0:.0f}s", file=sys.stderr)
    print(f"done: {saved} rows -> {args.output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
