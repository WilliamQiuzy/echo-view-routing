#!/usr/bin/env python
"""A41 (configs/experiments/video_echojepa_recognition_v1.yaml): acquire the pinned EchoJEPA code and the two public ViT-L
checkpoints on the server, recording bytes, full sha256, URL / HTTP status / final URL and time.

  E  EchoJEPA vitl-vmix22m-pt220-c55.pt, public Google Drive file 1T_ubAMpDMEByH7V6TJu9iT5Yf9IxlDBB (gdown)
  G  generic V-JEPA 2 ViT-L, https://dl.fbaipublicfiles.com/vjepa2/vitl.pt (curl)

No account, data agreement or training media. Network time is booked as kind "network", hashing as CPU.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO, COMMIT = "https://github.com/bowang-lab/EchoJEPA", "77208e1a8521c0e82f79579eebe4b22f46c5ccbd"
E_ID, E_NAME = "1T_ubAMpDMEByH7V6TJu9iT5Yf9IxlDBB", "vitl-vmix22m-pt220-c55.pt"
G_URL, G_NAME, G_BYTES = "https://dl.fbaipublicfiles.com/vjepa2/vitl.pt", "vitl.pt", 5127726842
OUT, SRC, CLONE = ROOT / "runs/video_echojepa_recognition_v1", ROOT / "checkpoints/a41_jepa", ROOT / "third_party/EchoJEPA"
LEDGER = OUT / "ledger.jsonl"


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def ledger(job: str, kind: str, started: str, seconds: float, ok: bool, note: str = "") -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as fh:
        fh.write(json.dumps({"job": job, "kind": kind, "started": started, "finished": now(), "hours": seconds / 3600, "ok": ok, "note": note}) + "\n")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def git(*a: str) -> str:
    return subprocess.run(["git", "-C", str(CLONE), *a], check=True, capture_output=True, text=True).stdout.strip()


def code() -> dict:
    if not CLONE.exists():
        subprocess.run(["git", "clone", "-q", REPO, str(CLONE)], check=True)
    git("checkout", "-q", COMMIT)
    head, dirty = git("rev-parse", "HEAD"), git("status", "--porcelain")
    if head != COMMIT or dirty:
        raise SystemExit(f"clone at {head}, dirty {bool(dirty)}: engineering stop")
    return {"repo": REPO, "commit": head, "date": git("log", "-1", "--format=%cI"), "clean": True, "path": str(CLONE.relative_to(ROOT))}


def fetch(name: str, cmd: list[str], url: str) -> dict:
    dst = SRC / name; started, t0, ok, note = now(), time.time(), False, ""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True); note = (r.stdout + r.stderr)[-600:]
        if r.returncode != 0 or not dst.is_file():
            raise SystemExit(f"{name}: download failed (exit {r.returncode}): {note[-300:]}")
        ok = True; return {"file": name, "url": url, "tool_output_tail": note, "download_seconds": time.time() - t0, "downloaded_at": started}
    finally:
        ledger(f"download {name} (network wait, not GPU)", "network", started, time.time() - t0, ok, note[-200:] if not ok else "")


def main() -> None:
    SRC.mkdir(parents=True, exist_ok=True); rec = {"started": now(), "code": code(), "files": {}}
    head = subprocess.run(["curl", "-sSIL", "-o", "/dev/null", "-w", "%{http_code} %{url_effective} %{size_download}", G_URL], capture_output=True, text=True).stdout.split()
    hdr = subprocess.run(["curl", "-sSIL", G_URL], capture_output=True, text=True).stdout
    g = fetch(G_NAME, ["curl", "-sSL", "--fail", "-o", str(SRC / G_NAME), "-w", "%{http_code} %{url_effective} %{size_download}", G_URL], G_URL)
    g.update({"head_http_status": head[0] if head else None, "head_content_length": next((ln.split(":", 1)[1].strip() for ln in hdr.splitlines() if ln.lower().startswith("content-length")), None),
              "get_status_final_url_bytes": g["tool_output_tail"].split()[-3:]})
    e = fetch(E_NAME, [str(Path(sys.executable).parent / "gdown"), E_ID, "-O", str(SRC / E_NAME), "--no-cookies"], f"https://drive.google.com/uc?id={E_ID}")
    e.update({"drive_file_id": E_ID, "drive_folder": "https://drive.google.com/drive/folders/1RFEXMe8TTcABMBz4H_qtiLB43K9jD_lf"})
    for k, d in (("G", g), ("E", e)):
        started, t0 = now(), time.time(); p = SRC / d["file"]; d["bytes"] = p.stat().st_size; d["sha256"] = sha256(p); ledger(f"sha256 {d['file']} (CPU)", "cpu", started, time.time() - t0, True); rec["files"][k] = d
    rec["G_bytes_equal_codex_head"] = rec["files"]["G"]["bytes"] == G_BYTES; rec["finished"] = now()
    (OUT / "downloads.json").write_text(json.dumps(rec, indent=1) + "\n"); print(json.dumps({k: {x: v[x] for x in ("file", "bytes", "sha256")} for k, v in rec["files"].items()}, indent=1))


if __name__ == "__main__":
    main()
