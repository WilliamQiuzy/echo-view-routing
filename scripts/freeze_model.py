#!/usr/bin/env python
"""Freeze a verified model: copy its files into models_frozen/<name>/<version>/, write MANIFEST.json (sha256,
provenance, metrics), and make everything read-only. Frozen directories are never modified again; a new version
gets a new directory.

  python scripts/freeze_model.py <name> <version> --file <path> [--file ...] [--meta key=value ...] [--notes "..."]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echo_routing.config import load_paths  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("version")
    ap.add_argument("--file", action="append", required=True, help="file to freeze (repeatable)")
    ap.add_argument("--meta", action="append", default=[], help="key=value provenance (repeatable)")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    paths = load_paths()
    dest = paths.repo_root / "models_frozen" / a.name / a.version
    if dest.exists():
        raise SystemExit(f"refusing to overwrite frozen model: {dest} (use a new version)")
    dest.mkdir(parents=True)
    files = []
    for f in a.file:
        src = Path(f).expanduser().resolve()
        if not src.is_file():
            raise SystemExit(f"missing file: {src}")
        out = dest / src.name
        shutil.copy2(src, out)
        files.append({"file": src.name, "bytes": out.stat().st_size, "sha256": sha256(out), "source": str(src)})
    meta = dict(kv.split("=", 1) for kv in a.meta)
    manifest = {"name": a.name, "version": a.version, "frozen_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "files": files, "provenance": meta, "notes": a.notes,
                "rule": "immutable: never edit, retrain into, or delete this directory; create a new version instead"}
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    for p in dest.iterdir():
        os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    os.chmod(dest, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    print(json.dumps({"frozen": str(dest), "files": [(f["file"], f["sha256"][:12]) for f in files]}, indent=1))


if __name__ == "__main__":
    main()
