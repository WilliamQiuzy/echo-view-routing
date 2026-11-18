"""Writes the EV9V development benchmark of the research code as one plain file: src/echoview/data/ev9v_dev.json.gz.

Runs inside the research repository (branch robust-upgrade), which is not part of this release:

    cd <research repository> && ECHO_INPUT=centre_crop_legacy python <this file> <research root> <output folder>

For each half: every sequence (native recording, pair stream, multi stream, real-order stream) as its fragments. A
fragment is a run of 10 Hz samples from one recording: the recording's key, its label, the gamma of the input variant
and the JPEG frame number of every sample. Recording names are not written: a recording is named by the first 16
hex digits of the SHA-256 of its EV9V folder name, and a native recording's sequence by its position. Whoever holds
EV9V gets the names back by hashing the folder names of their copy (echoview.evaluation.ev9v.recording_key).
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

R, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(R))
from robust import devset  # noqa: E402

CLASSES = ("PLAX", "PSAX", "A4C", "A5C", "SC4C")
FORMAT = "echo-view-routing benchmark v2"


def recording_key(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]


def native_id(half: str, position: int) -> str:
    return f"native_{half}_{position:04d}"


def fragments_from_refs(refs, labels, cuts):
    """cuts: first sample of every fragment. One recording, one label and one gamma per fragment, or the export stops."""
    out = []
    edges = [*cuts, len(refs)]
    for a, b in zip(edges[:-1], edges[1:]):
        videos = {r[0] for r in refs[a:b]}
        gammas = {round(float(r[2]), 6) for r in refs[a:b]}
        labs = {int(x) for x in labels[a:b]}
        if len(videos) != 1 or len(gammas) != 1 or len(labs) != 1:
            raise SystemExit("a fragment mixes recordings, gammas or labels")
        out.append({"video": refs[a][0], "label": labs.pop(), "gamma": gammas.pop(), "frames": [int(r[1]) for r in refs[a:b]]})
    return out


def export_half(half: str) -> tuple[list[dict], dict]:
    h = devset.Half(R, half)
    by_id = {q["id"]: q for q in h.seqs}
    seqs = []
    for k, s in enumerate(h.ev.natives):
        q = by_id[s.seq_id]
        seqs.append({"id": native_id(half, k), "kind": "native", "fragments": fragments_from_refs(q["refs"], q["labels"], [0])})
    for kind, streams in (("pairs", h.ev.pairs), ("multi", h.ev.multi)):
        for s in streams:
            q = by_id[s.recipe.recipe_id]
            if q["kind"] != kind or not np.array_equal(q["labels"], s.labels):
                raise SystemExit("a stream's labels differ from its window plan")
            frags = fragments_from_refs(q["refs"], q["labels"], [0, *[int(j) for j in s.joins]])
            for fr, src in zip(frags, s.recipe.fragments, strict=True):
                if fr["video"] != src.video_id or fr["label"] != int(src.label) or len(fr["frames"]) != src.end - src.start:
                    raise SystemExit("a fragment differs from its recipe")
            seqs.append({"id": q["id"], "kind": kind, "cell": str(s.recipe.cell), "fragments": frags})
    if len(seqs) != len(h.seqs) or any(a["kind"] != b["kind"] for a, b in zip(seqs, h.seqs)):
        raise SystemExit("the exported order differs from the harness order")
    for q in h.exam:
        video = np.array([r[0] for r in q["refs"]])
        cuts = [0, *[int(i) for i in np.flatnonzero(video[1:] != video[:-1]) + 1]]
        seqs.append({"id": q["id"], "kind": "exam", "fragments": fragments_from_refs(q["refs"], q["labels"], cuts)})
    names = sorted({f["video"] for s in seqs for f in s["fragments"]})
    if len({recording_key(n) for n in names}) != len(names):
        raise SystemExit("two recording names share a key")
    for s in seqs:
        for f in s["fragments"]:
            f["video"] = recording_key(f["video"])
    counts = {kind: sum(s["kind"] == kind for s in seqs) for kind in ("native", "pairs", "multi", "exam")}
    return seqs, {"sequences": counts, "recordings": len(names), "windows": len(h.windows["seq"])}


result = {"format": FORMAT, "source": "EV9V (bgx666/EV9V, CC BY 4.0), validation split", "classes": list(CLASSES), "hz": 10.0,
          "recording_key": "first 16 hex digits of the SHA-256 of the EV9V recording folder name (UTF-8)", "halves": {}}
report = {}
for half in devset.HALVES:
    seqs, report[half] = export_half(half)
    result["halves"][half] = {"sequences": seqs}
payload = json.dumps(result, separators=(",", ":")).encode()
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "ev9v_dev.json.gz").write_bytes(gzip.compress(payload, 9, mtime=0))
report["sha256_json"] = hashlib.sha256(payload).hexdigest()
print(json.dumps(report))
