"""Metadata-only EchoXFlow intake; run on echo-l40s, no model or EV9V access."""
import collections
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

REV = "7a36922880061791519a8f01553c15cad5df564f"
SIZE = 705276151
SHA = "8e90e77079cce8804073ff281558d6c4b5f343117ab042e8791427cd0e369454"
ROOT = Path("/home/william/echo-view-routing/runs/a47_public_evidence_intake_v1")
BASE = f"https://huggingface.co/datasets/Ahus-AIM/EchoXFlow/resolve/{REV}/"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def dump(name, value):
    p = ROOT / name
    p.write_text(json.dumps(value, indent=2, sort_keys=True))


def fetch(name, limit, expected_sha=None, expected_size=None):
    dst = ROOT / name
    if not dst.exists():
        partial = dst.with_suffix(dst.suffix + ".partial")
        # Retain a partial attempt rather than silently starting another transfer.
        with urllib.request.urlopen(BASE + name, timeout=40) as r, open(partial, "xb") as f:
            count = 0
            while True:
                b = r.read(4 * 1024 * 1024)
                if not b:
                    break
                count += len(b)
                if count > limit:
                    raise ValueError("metadata transfer exceeds byte cap")
                f.write(b)
        if expected_size is not None and partial.stat().st_size != expected_size:
            raise ValueError("wrong metadata size")
        if expected_sha is not None and digest(partial) != expected_sha:
            raise ValueError("wrong metadata SHA")
        os.replace(partial, dst)
    if dst.stat().st_size > limit:
        raise ValueError("existing metadata exceeds byte cap")
    if expected_size is not None and dst.stat().st_size != expected_size:
        raise ValueError("existing metadata wrong size")
    if expected_sha is not None and digest(dst) != expected_sha:
        raise ValueError("existing metadata wrong SHA")
    return dst


def main():
    manifest = json.loads(fetch("package_manifest.json", 100000).read_text())
    assert manifest["croissant_sha256"] == SHA
    fetch("LICENSE", 50000)
    fetch("splits.csv", 100000)
    path = fetch("croissant.json", SIZE, SHA, SIZE)
    obj = json.loads(path.read_text())
    sets = {r["@id"]: r for r in obj["recordSet"]}
    inv = {k: {"rows": len(v.get("data", [])), "fields": [f["name"] for f in v.get("field", [])]} for k, v in sets.items()}
    dump("record_sets.json", inv)
    # Aggregate array and recording contracts only. Private candidate links stay here.
    recs = sets["recordings"]["data"]
    arrays = sets["arrays"]["data"]
    candidates = [r for r in recs if any("lvot" in str(p).lower() for p in r.get("recordings/array_paths", []))]
    mode_counts = collections.Counter("+".join(sorted(r["recordings/modes"])) for r in candidates)
    ann_paths = collections.Counter()
    for r in recs:
        for p in r.get("recordings/array_paths", []):
            if any(w in p.lower() for w in ("lvot", "aortic", "sample_volume", "sampling_gate")):
                ann_paths[p] += 1
    all_keys = sorted({k for r in arrays for k in r})
    summary = {
        "revision": REV, "metadata_bytes": path.stat().st_size, "metadata_sha256": digest(path),
        "license": obj.get("license"), "record_set_contracts": inv,
        "recordings": len(recs), "arrays": len(arrays),
        "lvot_named_recordings": len(candidates),
        "lvot_named_exams": len({r["recordings/exam_id"] for r in candidates}),
        "lvot_candidate_modes": dict(mode_counts),
        "target_array_paths": dict(ann_paths), "array_field_names": all_keys,
        "contains_expert_view_labels": False,
    }
    dump("inventory.json", summary)
    dump("candidates.private.json", candidates)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    started, cpu = time.time(), time.process_time()
    ok = False
    try:
        main()
        ok = True
    finally:
        receipt = {"ok": ok, "started_unix": started, "finished_unix": time.time(),
                   "wall_seconds": time.time() - started, "process_cpu_seconds": time.process_time() - cpu,
                   "gpu_hours": 0, "script_sha256": digest(__file__)}
        dump("metadata_receipt.json", receipt)
