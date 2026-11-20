"""Writes the project's frozen EV9V test benchmark as one plain file: src/echoview/data/ev9v_test.json.gz.

Runs on the machine that holds the research project's verified manifest of the benchmark, which is not part of this
release:

    python export_ev9v_test_benchmark.py <manifest.json> <benchmark_summary.json> <output folder> [<access record>]

The manifest lists the 1,160 items of the benchmark (800 native recordings, 240 pair streams, 120 multi streams) with
the recording, frame number and gamma of every 10 Hz sample. It must have the digest the project recorded for it
(benchmark_summary.json). Nothing but the manifest is read: no frame and no prediction.

Recording names are not written. A recording is named by the first 16 hex digits of the SHA-256 of its EV9V folder
name, and a native recording's sequence by its position. A native recording also gets `group`: the number of its
examination among the examinations' names in sorted order. The project's bootstrap draws examinations in that order,
so the intervals can be recomputed without the names. Streams keep their names, which hold no recording name.

With a fourth argument, one line saying what was read is appended to that file: the project keeps a record of every
read of its test lists.
"""
import gzip
import hashlib
import json
import re
import sys
import time
from pathlib import Path

MANIFEST, SUMMARY, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
ACCESS = Path(sys.argv[4]) if len(sys.argv) > 4 else None
CLASSES = ("PLAX", "PSAX", "A4C", "A5C", "SC4C")
FORMAT = "echo-view-routing benchmark v2"
GAMMA = {"orig": 1.0, "gamma090": 0.9}
EXPECTED = {"native": 800, "pairs": 240, "multi": 120}


def recording_key(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]


def digest(items: list) -> str:
    """The research project's digest of a JSON object."""
    return hashlib.sha256(json.dumps(items, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def fragments_of(item: dict) -> list[dict]:
    """The item's fragments as runs of frame numbers. Every run must be one recording, one view and one gamma, and
    must be the fragment the manifest describes."""
    refs, labels = item["refs"], item["labels"]
    edges = [0, *item["joins"], len(refs)]
    described = item["fragments"]
    if len(refs) != item["n"] or len(labels) != item["n"] or len(edges) - 1 != len(described):
        raise SystemExit("an item's samples, labels and fragments do not fit together")
    out = []
    for (a, b), source in zip(zip(edges[:-1], edges[1:]), described):
        videos = {r[0] for r in refs[a:b]}
        gammas = {round(float(r[2]), 6) for r in refs[a:b]}
        views = {int(x) for x in labels[a:b]}
        if (videos, gammas, views) != ({source["video_id"]}, {GAMMA[source["variant"]]}, {int(source["label"])}) or b - a != source["end"] - source["start"]:
            raise SystemExit("a fragment differs from its description in the manifest")
        out.append({"video": source["video_id"], "label": int(source["label"]), "gamma": GAMMA[source["variant"]], "frames": [int(r[1]) for r in refs[a:b]]})
    return out


items = json.loads(MANIFEST.read_text())["items"]
if digest(items) != json.loads(SUMMARY.read_text())["manifest_digest"]:
    raise SystemExit("the manifest does not have the digest the project recorded for this benchmark")
if ACCESS is not None:
    with ACCESS.open("a") as handle:
        handle.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": "export for the release package",
                                 "read": "the verified manifest: test lists, frame references and reference views (no frame, no prediction)",
                                 "script": "tools/research/export_ev9v_test_benchmark.py of the echo-view-routing release"}) + "\n")

natives = [item for item in items if item["kind"] == "native"]
if [item["kind"] for item in items] != ["native"] * EXPECTED["native"] + ["pairs"] * EXPECTED["pairs"] + ["multi"] * EXPECTED["multi"]:
    raise SystemExit("the items are not 800 native recordings, 240 pair streams and 120 multi streams, in this order")
examinations = sorted({item["id"].rsplit("_", 1)[0] for item in natives})
number = {name: k for k, name in enumerate(examinations)}
sequences = []
for position, item in enumerate(items):
    fragments = fragments_of(item)
    if item["kind"] == "native":
        if len(fragments) != 1 or fragments[0]["video"] != item["id"] or item["joins"]:
            raise SystemExit("a native item is not one whole recording named by its folder")
        sequences.append({"id": f"native_test_{position:04d}", "kind": "native", "group": number[item["id"].rsplit("_", 1)[0]], "fragments": fragments})
    else:
        if not re.fullmatch(rf"{item['kind']}_test_[A-Za-z0-9_]*\d{{4}}", item["id"]):
            raise SystemExit("a stream's name does not have the expected form")
        sequences.append({"id": item["id"], "kind": item["kind"], "fragments": fragments})
names = sorted({f["video"] for s in sequences for f in s["fragments"]})
if len({recording_key(n) for n in names}) != len(names) or any(n in s["id"] for s in sequences for n in names):
    raise SystemExit("two recording names share a key, or a sequence name holds a recording name")
for s in sequences:
    for f in s["fragments"]:
        f["video"] = recording_key(f["video"])
if len({s["id"] for s in sequences}) != len(sequences):
    raise SystemExit("two sequences share a name")

result = {"format": FORMAT, "source": "EV9V (bgx666/EV9V, CC BY 4.0), test split", "classes": list(CLASSES), "hz": 10.0,
          "recording_key": "first 16 hex digits of the SHA-256 of the EV9V recording folder name (UTF-8)",
          "group": "native recordings: the number of the recording's examination, examinations numbered in the sorted order of their names",
          "halves": {"test": {"sequences": sequences}}}
payload = json.dumps(result, separators=(",", ":")).encode()
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "ev9v_test.json.gz").write_bytes(gzip.compress(payload, 9, mtime=0))
print(json.dumps({"sequences": {kind: sum(s["kind"] == kind for s in sequences) for kind in EXPECTED}, "examinations": len(examinations),
                  "recordings": len(names), "samples": sum(len(f["frames"]) for s in sequences for f in s["fragments"]),
                  "frames": len({(f["video"], n, f["gamma"]) for s in sequences for f in s["fragments"] for n in f["frames"]}),
                  "sha256_json": hashlib.sha256(payload).hexdigest()}))
