#!/usr/bin/env python
"""Download EchoNet-Dynamic and EchoNet-LVH from Redivis (Stanford AIMI) using the project owner's API token.

Requires REDIVIS_API_TOKEN in the environment (never committed; keep it in the server-side secrets/ folder).
Usage: REDIVIS_API_TOKEN=... python scripts/download_echonet_redivis.py --which dynamic,lvh --dest data/echonet
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

TABLES = {
    "dynamic": ("aimi.echonet_dynamic:66s1:v1_0.echonet:fjdn", "EchoNet-Dynamic.zip"),
    "lvh": ("aimi.echonet_lvh:cchq:v1_0.echonet_lvh:m0ea", "EchoNet-LVH.zip"),
}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--which", default="dynamic,lvh"); ap.add_argument("--dest", default="data/echonet")
    a = ap.parse_args()
    if not os.environ.get("REDIVIS_API_TOKEN"):
        sys.exit("REDIVIS_API_TOKEN is not set (create one at redivis.com → workspace settings → API tokens, scope data.data read)")
    import redivis  # noqa: E402  (pip install redivis)
    for name in a.which.split(","):
        ref, fname = TABLES[name]
        out = Path(a.dest) / name; out.mkdir(parents=True, exist_ok=True)
        table = redivis.table(ref)
        print(f"[{name}] listing table files …", flush=True)
        files = table.list_files()
        print(f"[{name}] {len(files)} file(s): {[f.name for f in files][:5]}", flush=True)
        target = out / fname
        if target.exists():
            print(f"[{name}] exists, skipping: {target}", flush=True); continue
        print(f"[{name}] downloading {fname} -> {target}", flush=True)
        table.file(fname).download(str(target), overwrite=True)
        print(f"[{name}] DONE {target} {target.stat().st_size/2**30:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
