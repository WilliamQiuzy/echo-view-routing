#!/usr/bin/env python
"""List and download the CETUS release (echo_routing/geometry/cetus.py) from the organiser's Girder collection into
data/cetus/raw on the server. The data stay there; they may not be redistributed.

  listing   walks the collection and writes data/cetus/listing.json (paths, ids, sizes)
  fetch P   downloads the listed files whose path equals or starts with P (repeatable), checks each size against the
            listing and that every .gz decompresses, and records SHA-256 in data/cetus/sha256.json. The server gives no
            checksum (its sha512 field is empty), so the size and gzip checks are what can be verified.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import urllib.request
from pathlib import Path

API = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1"
COLLECTION = "62eb991b73e9f0048c3a6c45"
ROOT = Path("data/cetus")


def get(path: str):
    with urllib.request.urlopen(API + path, timeout=120) as r:
        return json.load(r)


def listing() -> list[dict]:
    rows = []

    def walk(fid: str, prefix: str) -> None:
        for it in get(f"/item?folderId={fid}&limit=1000"):
            for f in get(f"/item/{it['_id']}/files?limit=100"):
                rows.append({"path": f"{prefix}/{f['name']}", "file": f["_id"], "size": f["size"], "sha512": f.get("sha512")})
        for sub in get(f"/folder?parentType=folder&parentId={fid}&limit=1000"):
            walk(sub["_id"], f"{prefix}/{sub['name']}")

    for top in get(f"/folder?parentType=collection&parentId={COLLECTION}&limit=100"):
        walk(top["_id"], top["name"])
    return rows


def fetch(rows: list[dict], prefixes: list[str]) -> dict:
    todo = [r for r in rows if any(r["path"] == p or r["path"].startswith(p.rstrip("/") + "/") for p in prefixes)]
    ledger_path = ROOT / "sha256.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    failed = []
    for r in todo:
        out = ROOT / "raw" / r["path"]; out.parent.mkdir(parents=True, exist_ok=True)
        if not (out.exists() and out.stat().st_size == r["size"]):
            tmp = out.with_name(out.name + ".part")
            with urllib.request.urlopen(f"{API}/file/{r['file']}/download", timeout=600) as resp, tmp.open("wb") as fh:
                while chunk := resp.read(1 << 20):
                    fh.write(chunk)
            tmp.replace(out)
        ok = out.stat().st_size == r["size"]
        if ok and out.suffix == ".gz":
            try:
                with gzip.open(out, "rb") as fh:
                    while fh.read(1 << 24):
                        pass
            except (OSError, EOFError):
                ok = False
        if not ok:
            failed.append(r["path"])
        ledger[r["path"]] = {"bytes": out.stat().st_size, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "verified": ok}
    ledger_path.write_text(json.dumps(ledger, indent=1, sort_keys=True))
    return {"files": len(todo), "verified": len(todo) - len(failed), "failed": failed}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["listing", "fetch"]); ap.add_argument("prefix", nargs="*")
    a = ap.parse_args(); ROOT.mkdir(parents=True, exist_ok=True)
    if a.mode == "listing":
        rows = listing(); (ROOT / "listing.json").write_text(json.dumps(rows, indent=1))
        print(json.dumps({"files": len(rows), "bytes": sum(r["size"] for r in rows)}))
    else:
        print(json.dumps(fetch(json.loads((ROOT / "listing.json").read_text()), a.prefix), indent=1))


if __name__ == "__main__":
    main()
